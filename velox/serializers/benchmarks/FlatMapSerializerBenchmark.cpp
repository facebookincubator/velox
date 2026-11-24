/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <folly/Benchmark.h>
#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>
#include <folly/init/Init.h>

#include <bit>
#include <numeric>

#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/FlatVector.h"

// Measures PrestoVectorSerde on a map column of 1,024 rows held as a regular
// map and as a flat map. Each case reports the regular map as the baseline and
// the flat map relative to it.
//
// The columns have INTEGER keys and each key is in a row with a fixed
// probability. Their names give the value type, the number of keys and the
// percentage of rows each key is in:
//
//  - real13800Keys34Pct, real13800Keys57Pct, real13800Keys70Pct: 13,800 keys
//    with REAL values.
//  - real3200Keys57Pct: 3,200 keys with REAL values.
//  - array3000Keys57Pct: 3,000 keys with ARRAY(BIGINT) values of 10 to 26
//    elements.
//  - map1000Keys57Pct: 1,000 keys with MAP(BIGINT, REAL) values of 10 to 26
//    entries.
//
// Every column is measured as one page of all its rows, and as 64 pages that
// each take every 64th row 16 times over, the way a shuffle splits a batch by
// partition and buffers each partition over several batches:
//
//  - serialize: serializes the column into its pages.
//  - deserialize: deserializes the pages.
//  - deserializeConsume: deserializes the pages and reads every value of every
//    key through DecodedVector, the way a downstream operator would.

namespace facebook::velox {
namespace {

using serializer::presto::PrestoVectorSerde;

constexpr vector_size_t kNumRows{1'024};
constexpr int32_t kNumAppends{16};

enum class ValueKind {
  kReal,
  kArray,
  kMap,
};

TypePtr valueType(ValueKind valueKind) {
  switch (valueKind) {
    case ValueKind::kReal:
      return REAL();
    case ValueKind::kArray:
      return ARRAY(BIGINT());
    case ValueKind::kMap:
      return MAP(BIGINT(), REAL());
  }
  VELOX_UNREACHABLE();
}

bool isPresent(vector_size_t row, int32_t key, int32_t presencePct) {
  return folly::hash::twang_mix64(row * 1'000'003ULL + key) % 100 <
      static_cast<uint64_t>(presencePct);
}

vector_size_t numElementsAt(vector_size_t row, int32_t key) {
  return 10 + folly::hash::twang_mix64(row * 7'919ULL + key) % 17;
}

// Returns a value for each of 'rows' and 'keys'. The value at i belongs to row
// rows[i] and key keys[i] if bit i of 'present' is set or 'present' is null,
// and is 0 or empty otherwise.
VectorPtr makeValues(
    ValueKind valueKind,
    const std::vector<vector_size_t>& rows,
    const std::vector<int32_t>& keys,
    const uint64_t* present,
    memory::MemoryPool* pool) {
  const vector_size_t size = rows.size();
  if (valueKind == ValueKind::kReal) {
    auto values = BaseVector::create<FlatVector<float>>(REAL(), size, pool);
    for (auto i = 0; i < size; ++i) {
      values->set(
          i,
          present == nullptr || bits::isBitSet(present, i)
              ? rows[i] * 0.5f + keys[i]
              : 0);
    }
    return values;
  }
  auto offsets = allocateOffsets(size, pool);
  auto sizes = allocateSizes(size, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  vector_size_t numElements{0};
  for (auto i = 0; i < size; ++i) {
    rawOffsets[i] = numElements;
    rawSizes[i] = present == nullptr || bits::isBitSet(present, i)
        ? numElementsAt(rows[i], keys[i])
        : 0;
    numElements += rawSizes[i];
  }
  auto elements =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), numElements, pool);
  auto elementValues =
      BaseVector::create<FlatVector<float>>(REAL(), numElements, pool);
  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < rawSizes[i]; ++j) {
      elements->set(rawOffsets[i] + j, rows[i] * 1'000'003LL + keys[i] + j);
      elementValues->set(rawOffsets[i] + j, rows[i] * 0.5f + keys[i] + j);
    }
  }
  if (valueKind == ValueKind::kMap) {
    return std::make_shared<MapVector>(
        pool,
        valueType(valueKind),
        nullptr,
        size,
        std::move(offsets),
        std::move(sizes),
        std::move(elements),
        std::move(elementValues));
  }
  return std::make_shared<ArrayVector>(
      pool,
      valueType(valueKind),
      nullptr,
      size,
      std::move(offsets),
      std::move(sizes),
      std::move(elements));
}

RowVectorPtr makeRow(VectorPtr column, memory::MemoryPool* pool) {
  auto rowType = ROW("c0", column->type());
  return std::make_shared<RowVector>(
      pool,
      std::move(rowType),
      nullptr,
      kNumRows,
      std::vector<VectorPtr>{std::move(column)});
}

RowVectorPtr makeMap(
    ValueKind valueKind,
    int32_t numKeys,
    int32_t presencePct,
    memory::MemoryPool* pool) {
  auto offsets = allocateOffsets(kNumRows, pool);
  auto sizes = allocateSizes(kNumRows, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  std::vector<vector_size_t> entryRows;
  std::vector<int32_t> entryKeys;
  for (auto row = 0; row < kNumRows; ++row) {
    rawOffsets[row] = entryKeys.size();
    for (auto key = 0; key < numKeys; ++key) {
      if (isPresent(row, key, presencePct)) {
        entryRows.push_back(row);
        entryKeys.push_back(key);
      }
    }
    rawSizes[row] = entryKeys.size() - rawOffsets[row];
  }
  auto keys = BaseVector::create<FlatVector<int32_t>>(
      INTEGER(), entryKeys.size(), pool);
  for (auto i = 0; i < entryKeys.size(); ++i) {
    keys->set(i, entryKeys[i]);
  }
  auto values = makeValues(valueKind, entryRows, entryKeys, nullptr, pool);
  return makeRow(
      std::make_shared<MapVector>(
          pool,
          MAP(INTEGER(), valueType(valueKind)),
          nullptr,
          kNumRows,
          std::move(offsets),
          std::move(sizes),
          std::move(keys),
          std::move(values)),
      pool);
}

RowVectorPtr makeFlatMap(
    ValueKind valueKind,
    int32_t numKeys,
    int32_t presencePct,
    memory::MemoryPool* pool) {
  auto distinctKeys =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), numKeys, pool);
  std::vector<vector_size_t> rows(kNumRows);
  std::iota(rows.begin(), rows.end(), 0);
  std::vector<int32_t> keys(kNumRows);
  std::vector<VectorPtr> mapValues(numKeys);
  std::vector<BufferPtr> inMaps(numKeys);
  for (auto key = 0; key < numKeys; ++key) {
    distinctKeys->set(key, key);
    std::fill(keys.begin(), keys.end(), key);
    inMaps[key] = AlignedBuffer::allocate<bool>(kNumRows, pool, false);
    auto* rawInMap = inMaps[key]->asMutable<uint64_t>();
    for (auto row = 0; row < kNumRows; ++row) {
      bits::setBit(rawInMap, row, isPresent(row, key, presencePct));
    }
    mapValues[key] = makeValues(valueKind, rows, keys, rawInMap, pool);
  }
  return makeRow(
      std::make_shared<FlatMapVector>(
          pool,
          MAP(INTEGER(), valueType(valueKind)),
          nullptr,
          kNumRows,
          std::move(distinctKeys),
          std::move(mapValues),
          std::move(inMaps)),
      pool);
}

int32_t numAppends(int32_t numPages) {
  return numPages == 1 ? 1 : kNumAppends;
}

// Serializes 'data' into 'numPages' pages. Page i takes the rows whose number
// modulo 'numPages' is i, appended 'numAppends' times.
std::vector<std::string> serialize(
    const RowVectorPtr& data,
    int32_t numPages,
    memory::MemoryPool* pool) {
  PrestoVectorSerde serde;
  const PrestoVectorSerde::PrestoOptions options;
  Scratch scratch;
  std::vector<std::string> pages;
  std::vector<vector_size_t> rows;
  for (auto page = 0; page < numPages; ++page) {
    rows.clear();
    for (auto row = page; row < data->size(); row += numPages) {
      rows.push_back(row);
    }
    StreamArena arena(pool);
    auto pageSerializer = serde.createIterativeSerializer(
        asRowType(data->type()), rows.size(), &arena, &options);
    for (auto i = 0; i < numAppends(numPages); ++i) {
      pageSerializer->append(
          data, folly::Range(rows.data(), rows.size()), scratch);
    }
    std::ostringstream out;
    OStreamOutputStream output(&out);
    pageSerializer->flush(&output);
    pages.push_back(out.str());
  }
  return pages;
}

std::vector<RowVectorPtr> deserialize(
    const std::vector<std::string>& pages,
    const RowTypePtr& rowType,
    memory::MemoryPool* pool) {
  PrestoVectorSerde serde;
  const PrestoVectorSerde::PrestoOptions options;
  std::vector<RowVectorPtr> results;
  for (const auto& page : pages) {
    BufferInputStream input({ByteRange{
        reinterpret_cast<uint8_t*>(const_cast<char*>(page.data())),
        static_cast<int32_t>(page.size()),
        0}});
    RowVectorPtr result;
    serde.deserialize(&input, pool, rowType, &result, 0, &options);
    results.push_back(std::move(result));
  }
  return results;
}

uint64_t foldAt(const DecodedVector& decoded, vector_size_t index) {
  if (decoded.isNullAt(index)) {
    return 0;
  }
  switch (decoded.base()->typeKind()) {
    case TypeKind::INTEGER:
      return decoded.valueAt<int32_t>(index);
    case TypeKind::BIGINT:
      return decoded.valueAt<int64_t>(index);
    case TypeKind::REAL:
      return std::bit_cast<uint32_t>(decoded.valueAt<float>(index));
    default:
      VELOX_UNSUPPORTED("Unsupported type: {}", decoded.base()->type());
  }
}

// Folds the value at 'index'. An array or map value folds its elements in
// order through 'elements', which decodes the array elements or map keys, and
// for a map through 'elementValues', which decodes the map values.
uint64_t foldValueAt(
    const DecodedVector& decoded,
    const DecodedVector& elements,
    const DecodedVector& elementValues,
    vector_size_t index) {
  const auto kind = decoded.base()->typeKind();
  if ((kind != TypeKind::ARRAY && kind != TypeKind::MAP) ||
      decoded.isNullAt(index)) {
    return foldAt(decoded, index);
  }
  const auto* array = decoded.base()->asUnchecked<ArrayVectorBase>();
  const auto baseIndex = decoded.index(index);
  const auto offset = array->offsetAt(baseIndex);
  const auto size = array->sizeAt(baseIndex);
  uint64_t hash = size;
  for (auto i = offset; i < offset + size; ++i) {
    hash = folly::hash::hash_128_to_64(hash, foldAt(elements, i));
    if (kind == TypeKind::MAP) {
      hash = folly::hash::hash_128_to_64(hash, foldAt(elementValues, i));
    }
  }
  return hash;
}

void decodeElements(
    const DecodedVector& values,
    DecodedVector& elements,
    DecodedVector& elementValues) {
  if (values.base()->typeKind() == TypeKind::ARRAY) {
    elements.decode(*values.base()->asUnchecked<ArrayVector>()->elements());
  } else if (values.base()->typeKind() == TypeKind::MAP) {
    const auto* map = values.base()->asUnchecked<MapVector>();
    elements.decode(*map->mapKeys());
    elementValues.decode(*map->mapValues());
  }
}

uint64_t consumeFlatMap(const FlatMapVector& flatMap) {
  uint64_t checksum{0};
  DecodedVector keys(*flatMap.distinctKeys());
  DecodedVector values;
  DecodedVector elements;
  DecodedVector elementValues;
  for (column_index_t channel = 0; channel < flatMap.numDistinctKeys();
       ++channel) {
    const auto key = foldAt(keys, channel);
    values.decode(*flatMap.mapValuesAt(channel));
    decodeElements(values, elements, elementValues);
    const auto* rowsWithKey = flatMap.rawInMapsAt(channel);
    if (rowsWithKey == nullptr) {
      rowsWithKey = flatMap.rawNulls();
    }
    if (rowsWithKey == nullptr) {
      for (auto row = 0; row < flatMap.size(); ++row) {
        checksum += folly::hash::hash_128_to_64(
            key, foldValueAt(values, elements, elementValues, row));
      }
    } else {
      bits::forEachSetBit(rowsWithKey, 0, flatMap.size(), [&](auto row) {
        checksum += folly::hash::hash_128_to_64(
            key, foldValueAt(values, elements, elementValues, row));
      });
    }
  }
  return checksum;
}

uint64_t consumeMap(const MapVector& map) {
  uint64_t checksum{0};
  const auto* rawNulls = map.rawNulls();
  const auto* rawOffsets = map.rawOffsets();
  const auto* rawSizes = map.rawSizes();
  DecodedVector keys(*map.mapKeys());
  DecodedVector values(*map.mapValues());
  DecodedVector elements;
  DecodedVector elementValues;
  decodeElements(values, elements, elementValues);
  for (auto row = 0; row < map.size(); ++row) {
    if (rawNulls == nullptr || bits::isBitSet(rawNulls, row)) {
      const auto end = rawOffsets[row] + rawSizes[row];
      for (auto entry = rawOffsets[row]; entry < end; ++entry) {
        checksum += folly::hash::hash_128_to_64(
            foldAt(keys, entry),
            foldValueAt(values, elements, elementValues, entry));
      }
    }
  }
  return checksum;
}

uint64_t consume(const std::vector<RowVectorPtr>& results) {
  uint64_t checksum{0};
  for (const auto& result : results) {
    const auto& column = result->childAt(0);
    if (column->encoding() == VectorEncoding::Simple::FLAT_MAP) {
      checksum += consumeFlatMap(*column->asUnchecked<FlatMapVector>());
    } else {
      checksum += consumeMap(*column->asChecked<MapVector>());
    }
  }
  return checksum;
}

// Builds the column of one case as a regular map and as a flat map, with
// their pages, and owns the memory pool they use. Building a column frees the
// one before, so only one column and its pages are in memory at a time.
class BenchmarkData {
 public:
  BenchmarkData() : pool_(memory::memoryManager()->addLeafPool()) {}

  // Builds the column 'name' unless it is the current one. Checks that the
  // regular map and flat map pages read back the same entries.
  void prepare(
      std::string_view name,
      ValueKind valueKind,
      int32_t numKeys,
      int32_t presencePct) {
    if (name == name_) {
      return;
    }
    data_.clear();
    pages_.clear();
    name_ = name;
    data_[false] = makeMap(valueKind, numKeys, presencePct, pool_.get());
    data_[true] = makeFlatMap(valueKind, numKeys, presencePct, pool_.get());
    const auto rowType = asRowType(data_[false]->type());
    for (const auto numPages : {1, 64}) {
      for (const auto flat : {false, true}) {
        pages_[{flat, numPages}] =
            serialize(data_[flat], numPages, pool_.get());
      }
      VELOX_CHECK_EQ(
          consume(deserialize(pages(true, numPages), rowType, pool())),
          consume(deserialize(pages(false, numPages), rowType, pool())),
          "Flat map and map pages read back different entries: {}",
          name);
    }
  }

  const RowVectorPtr& data(bool flat) const {
    return data_.at(flat);
  }

  const std::vector<std::string>& pages(bool flat, int32_t numPages) const {
    return pages_.at({flat, numPages});
  }

  memory::MemoryPool* pool() const {
    return pool_.get();
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_;
  std::string name_;
  folly::F14FastMap<bool, RowVectorPtr> data_;
  folly::F14FastMap<std::pair<bool, int32_t>, std::vector<std::string>> pages_;
};

std::unique_ptr<BenchmarkData> benchmarkData;

void prepare(
    std::string_view name,
    ValueKind valueKind,
    int32_t numKeys,
    int32_t presencePct) {
  folly::BenchmarkSuspender suspender;
  benchmarkData->prepare(name, valueKind, numKeys, presencePct);
}

void runSerialize(bool flat, int32_t numPages) {
  folly::doNotOptimizeAway(
      serialize(benchmarkData->data(flat), numPages, benchmarkData->pool()));
}

void runDeserialize(bool flat, int32_t numPages) {
  folly::doNotOptimizeAway(deserialize(
      benchmarkData->pages(flat, numPages),
      asRowType(benchmarkData->data(flat)->type()),
      benchmarkData->pool()));
}

void runDeserializeConsume(bool flat, int32_t numPages) {
  folly::doNotOptimizeAway(consume(deserialize(
      benchmarkData->pages(flat, numPages),
      asRowType(benchmarkData->data(flat)->type()),
      benchmarkData->pool())));
}

#define FLAT_MAP_SERDE_BENCHMARKS(                                    \
    name, valueKind, numKeys, presencePct, numPages)                  \
  BENCHMARK(serialize_##name##_##numPages) {                          \
    prepare(#name, valueKind, numKeys, presencePct);                  \
    runSerialize(false, numPages);                                    \
  }                                                                   \
  BENCHMARK_RELATIVE(serialize_##name##_##numPages##_flat) {          \
    prepare(#name, valueKind, numKeys, presencePct);                  \
    runSerialize(true, numPages);                                     \
  }                                                                   \
  BENCHMARK(deserialize_##name##_##numPages) {                        \
    prepare(#name, valueKind, numKeys, presencePct);                  \
    runDeserialize(false, numPages);                                  \
  }                                                                   \
  BENCHMARK_RELATIVE(deserialize_##name##_##numPages##_flat) {        \
    prepare(#name, valueKind, numKeys, presencePct);                  \
    runDeserialize(true, numPages);                                   \
  }                                                                   \
  BENCHMARK(deserializeConsume_##name##_##numPages) {                 \
    prepare(#name, valueKind, numKeys, presencePct);                  \
    runDeserializeConsume(false, numPages);                           \
  }                                                                   \
  BENCHMARK_RELATIVE(deserializeConsume_##name##_##numPages##_flat) { \
    prepare(#name, valueKind, numKeys, presencePct);                  \
    runDeserializeConsume(true, numPages);                            \
  }                                                                   \
  BENCHMARK_DRAW_LINE();

FLAT_MAP_SERDE_BENCHMARKS(real13800Keys34Pct, ValueKind::kReal, 13'800, 34, 1)
FLAT_MAP_SERDE_BENCHMARKS(real13800Keys34Pct, ValueKind::kReal, 13'800, 34, 64)
FLAT_MAP_SERDE_BENCHMARKS(real13800Keys57Pct, ValueKind::kReal, 13'800, 57, 1)
FLAT_MAP_SERDE_BENCHMARKS(real13800Keys57Pct, ValueKind::kReal, 13'800, 57, 64)
FLAT_MAP_SERDE_BENCHMARKS(real13800Keys70Pct, ValueKind::kReal, 13'800, 70, 1)
FLAT_MAP_SERDE_BENCHMARKS(real13800Keys70Pct, ValueKind::kReal, 13'800, 70, 64)
FLAT_MAP_SERDE_BENCHMARKS(real3200Keys57Pct, ValueKind::kReal, 3'200, 57, 1)
FLAT_MAP_SERDE_BENCHMARKS(real3200Keys57Pct, ValueKind::kReal, 3'200, 57, 64)
FLAT_MAP_SERDE_BENCHMARKS(array3000Keys57Pct, ValueKind::kArray, 3'000, 57, 1)
FLAT_MAP_SERDE_BENCHMARKS(array3000Keys57Pct, ValueKind::kArray, 3'000, 57, 64)
FLAT_MAP_SERDE_BENCHMARKS(map1000Keys57Pct, ValueKind::kMap, 1'000, 57, 1)
FLAT_MAP_SERDE_BENCHMARKS(map1000Keys57Pct, ValueKind::kMap, 1'000, 57, 64)

} // namespace
} // namespace facebook::velox

int main(int argc, char** argv) {
  using namespace facebook::velox;
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  benchmarkData = std::make_unique<BenchmarkData>();
  folly::runBenchmarks();
  benchmarkData.reset();
  return 0;
}
