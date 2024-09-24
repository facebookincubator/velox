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
#include <folly/init/Init.h>
#include <random>

#include "velox/row/UnsafeRowDeserializers.h"
#include "velox/row/UnsafeRowFast.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::spark::benchmarks {
namespace {
using namespace facebook::velox;
using namespace facebook::velox::row;
using facebook::velox::test::VectorMaker;

class BenchmarkHelper {
 public:
  std::pair<std::vector<BufferPtr>, const RowTypePtr> randomUnsaferows(
      int nFields,
      int nRows,
      const std::vector<TypePtr>& columnTypes) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    names.reserve(nFields);
    types.reserve(nFields);
    for (int32_t i = 0; i < nFields; ++i) {
      names.push_back("");
      auto idx = folly::Random::rand32() % columnTypes.size();
      types.push_back(columnTypes[idx]);
    }
    RowTypePtr rowType =
        TypeFactory<TypeKind::ROW>::create(std::move(names), std::move(types));

    VectorFuzzer::Options opts;
    opts.vectorSize = 1;
    opts.nullRatio = 0.1;
    opts.stringVariableLength = true;
    opts.stringLength = 20;
    // Spark uses microseconds to store timestamp.
    opts.timestampPrecision =
        VectorFuzzer::Options::TimestampPrecision::kMicroSeconds;

    auto seed = folly::Random::rand32();
    VectorFuzzer fuzzer(opts, pool_.get(), seed);
    const auto inputVector = fuzzer.fuzzInputRow(rowType);
    std::vector<BufferPtr> results;
    results.reserve(nRows);
    // Serialize rowVector into bytes.
    UnsafeRowFast unsafeRow(inputVector);
    for (int32_t i = 0; i < nRows; ++i) {
      BufferPtr bufferPtr =
          AlignedBuffer::allocate<char>(kBufferSize, pool_.get(), true);
      char* buffer = bufferPtr->asMutable<char>();
      auto rowSize = unsafeRow.serialize(0, buffer);
      VELOX_CHECK_LE(rowSize, kBufferSize);
      bufferPtr->setSize(rowSize);
      results.push_back(bufferPtr);
    }
    return {results, rowType};
  }

  std::vector<std::optional<std::string_view>> getData(
      std::vector<BufferPtr> buffers) {
    std::vector<std::optional<std::string_view>> data;
    data.reserve(buffers.size());
    for (const auto& buffer : buffers) {
      data.emplace_back(
          std::string_view{buffer->asMutable<char>(), buffer->size()});
    }
    return data;
  }

  void deserialize(
      const std::vector<std::optional<std::string_view>>& data,
      const TypePtr& type) {
    UnsafeRowDeserializer::deserialize(data, type, pool_.get());
  }

  VectorPtr deserialize(
      const uint8_t* memoryAddress,
      const RowTypePtr& type,
      const std::vector<size_t>& offsets) {
    return UnsafeRowDeserializer::deserialize(
        memoryAddress, type, offsets, pool_.get());
  }

  BufferPtr copyBuffers(
      const std::vector<BufferPtr>& buffers,
      int32_t bufferTotalSize) {
    BufferPtr result =
        AlignedBuffer::allocate<char>(bufferTotalSize, pool_.get(), true);
    int32_t offset = 0;
    char* resultPtr = result->asMutable<char>();
    for (const auto& buffer : buffers) {
      memcpy(resultPtr + offset, buffer->asMutable<char>(), buffer->size());
      offset += buffer->size();
    }
    return result;
  }

 private:
  static constexpr uint64_t kBufferSize = 70 << 10; // 70kb
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
};

static const std::vector<TypePtr> kAllTypes{
    BOOLEAN(),
    TINYINT(),
    SMALLINT(),
    INTEGER(),
    BIGINT(),
    REAL(),
    DOUBLE(),
    DECIMAL(20, 2),
    DECIMAL(12, 4),
    VARCHAR(),
    TIMESTAMP(),
    DATE(),
    UNKNOWN(),
    ARRAY(INTEGER()),
    MAP(VARCHAR(), ARRAY(INTEGER())),
    ROW({INTEGER()})};

static const std::vector<TypePtr> kScalarTypes{
    BOOLEAN(),
    TINYINT(),
    SMALLINT(),
    INTEGER(),
    BIGINT(),
    REAL(),
    DOUBLE(),
    DECIMAL(20, 2),
    DECIMAL(12, 4),
    VARCHAR(),
    TIMESTAMP(),
    DATE(),
    UNKNOWN()};

int deserialize(
    int nIters,
    int nFields,
    int nRows,
    std::vector<TypePtr> types) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  auto [buffers, rowType] = helper.randomUnsaferows(nFields, nRows, types);
  suspender.dismiss();
  auto data = helper.getData(buffers);
  for (int i = 0; i < nIters; i++) {
    helper.deserialize(data, rowType);
  }

  return nIters * nFields * nRows;
}

int deserializeFast(
    int nIters,
    int nFields,
    int nRows,
    std::vector<TypePtr> types) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  auto [buffers, rowType] = helper.randomUnsaferows(nFields, nRows, types);
  std::vector<size_t> offsets;
  offsets.reserve(nRows);
  size_t offset = 0;
  for (const auto& buffer : buffers) {
    offsets.push_back(offset);
    offset += buffer->size();
  }
  auto buffer = helper.copyBuffers(buffers, offset);
  suspender.dismiss();
  for (int i = 0; i < nIters; i++) {
    helper.deserialize(buffer->asMutable<uint8_t>(), rowType, offsets);
  }

  return nIters * nFields * nRows;
}

BENCHMARK_NAMED_PARAM_MULTI(
    deserialize,
    batch_10_100k_string_only,
    10,
    100000,
    {VARCHAR()});

BENCHMARK_RELATIVE_NAMED_PARAM_MULTI(
    deserializeFast,
    batch_10_100k_string_only,
    10,
    100000,
    {VARCHAR()});

BENCHMARK_NAMED_PARAM_MULTI(
    deserialize,
    batch_100_100k_string_only,
    100,
    100000,
    {VARCHAR()});

BENCHMARK_RELATIVE_NAMED_PARAM_MULTI(
    deserializeFast,
    batch_100_100k_string_only,
    100,
    100000,
    {VARCHAR()});

BENCHMARK_NAMED_PARAM_MULTI(
    deserialize,
    batch_10_100k_all_types,
    10,
    100000,
    kAllTypes);

BENCHMARK_NAMED_PARAM_MULTI(
    deserialize,
    batch_100_100k_all_types,
    100,
    100000,
    kAllTypes);

BENCHMARK_NAMED_PARAM_MULTI(
    deserialize,
    batch_10_100k_scalar_types,
    10,
    100000,
    kScalarTypes);

BENCHMARK_RELATIVE_NAMED_PARAM_MULTI(
    deserializeFast,
    batch_10_100k_scalar_types,
    10,
    100000,
    kScalarTypes);

BENCHMARK_NAMED_PARAM_MULTI(
    deserialize,
    batch_100_100k_scalar_types,
    100,
    100000,
    kScalarTypes);

BENCHMARK_RELATIVE_NAMED_PARAM_MULTI(
    deserializeFast,
    batch_100_100k_scalar_types,
    100,
    100000,
    kScalarTypes);

} // namespace
} // namespace facebook::spark::benchmarks

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
  return 0;
}
