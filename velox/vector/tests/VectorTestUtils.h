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

#pragma once

#include <gtest/gtest.h>
#include "velox/vector/tests/VectorValueGenerator.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorMakerStats.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

inline std::vector<VectorEncoding::Simple> kFullValueTypes = {
    VectorEncoding::Simple::DICTIONARY,
    VectorEncoding::Simple::FLAT,
    VectorEncoding::Simple::SEQUENCE,
    VectorEncoding::Simple::CONSTANT};

inline std::vector<VectorEncoding::Simple> kNonConstantTypes = {
    VectorEncoding::Simple::DICTIONARY,
    VectorEncoding::Simple::FLAT,
    VectorEncoding::Simple::SEQUENCE,
    VectorEncoding::Simple::BIASED};

inline std::vector<VectorEncoding::Simple> kAllTypes = {
    VectorEncoding::Simple::DICTIONARY,
    VectorEncoding::Simple::FLAT,
    VectorEncoding::Simple::SEQUENCE,
    VectorEncoding::Simple::BIASED,
    VectorEncoding::Simple::CONSTANT};

template <typename T>
using ExpectedData = std::vector<std::optional<T>>;

template <typename T>
class VectorGeneratedData {
 public:
  void sort() {
    std::sort(data_.begin(), data_.end());
  }

  void push_back(std::optional<T> value) {
    if (value.has_value()) {
      auto ownedValue = stringViewBufferHolder_.getOwnedValue(value.value());
      data_.push_back(std::move(ownedValue));
    } else {
      data_.push_back(std::nullopt);
    }
  }

  void pop_back() {
    data_.pop_back();
  }

  void reserve(size_t amount) {
    data_.reserve(amount);
  }

  const ExpectedData<T>& data() const {
    return data_;
  }

  StringViewBufferHolder& stringViewBufferHolder() {
    return stringViewBufferHolder_;
  }

 protected:
  ExpectedData<T> data_;

  // In case T is StringView, the buffer below holds their actual data.
  std::shared_ptr<memory::MemoryPool> scopedPool =
      memory::getDefaultMemoryPool();
  StringViewBufferHolder stringViewBufferHolder_ =
      StringViewBufferHolder(scopedPool.get());
};

template <typename T>
std::optional<T> getVal(
    const ExpectedData<T>& data,
    int32_t count,
    int32_t idx,
    int32_t sequenceCount,
    int32_t sequenceLength) {
  bool useSequences = sequenceCount > 0 && sequenceLength > 1;
  int64_t idxToUse = idx;
  if (useSequences) {
    auto chunkWidth = (count / data.size());
    idxToUse = idx / chunkWidth;
  }

  return data[idxToUse % data.size()];
}

/// Pass a fixed number when we want to generate determined pseudo-random
/// data. In order to generate different sets of determined pseudo-random
/// data, we can pass different integers at callsites. e.g. magic numbers
/// 1000 for callsite A, 1070 for callsite B and etc.
/// Otherwise generate unpredictable pseudo-random for benchmark and perf
/// when seed is std::nullopt.
///
/// `fixedWidthStringSize`: std::nullopt means generating variable-width string
/// elements. A number means generating fixed-width string elements.
template <typename T>
VectorGeneratedData<T> genTestData(
    int32_t cardinality,
    const TypePtr type = CppToType<T>::create(),
    bool includeNulls = false,
    bool sorted = false,
    bool useFullTypeRange = false,
    const std::optional<uint32_t>& seed = std::nullopt,
    const std::optional<uint32_t>& fixedWidthStringSize = std::nullopt) {
  // Generate unpredictable random data by default.
  std::optional<folly::Random::DefaultGenerator> rng = std::nullopt;

  if (seed.has_value()) {
    rng = folly::Random::create();
    // Use the given seed so that it can generate predictable random data.
    rng.value().seed(seed.value());
  }

  VectorGeneratedData<T> testData;
  testData.reserve(cardinality);

  for (int64_t m = 0; m < cardinality; m++) {
    testData.push_back(VectorValueGenerator::cardValueOf<T>(
        useFullTypeRange,
        rng,
        testData.stringViewBufferHolder(),
        fixedWidthStringSize,
        type));
  }

  if (includeNulls) {
    testData.pop_back();
    testData.push_back(std::nullopt);
  }
  if (sorted) {
    testData.sort();
  }
  return testData;
}

template <typename T>
VectorGeneratedData<T> genTestDataWithSequences(
    int32_t count,
    int32_t cardinality,
    bool isSorted,
    bool includeNulls,
    int32_t sequenceCount,
    int32_t sequenceLength,
    bool useFullTypeRange = true,
    const std::optional<uint32_t>& seed = std::nullopt,
    const std::optional<uint32_t>& fixedWidthStringSize = std::nullopt) {
  const bool useSequences = sequenceCount > 0 && sequenceLength > 1;
  auto testData = genTestData<T>(
      cardinality,
      CppToType<T>::create(),
      includeNulls,
      isSorted,
      useFullTypeRange,
      seed,
      fixedWidthStringSize);

  VectorGeneratedData<T> outData;
  int32_t idx = 0;

  do {
    auto val =
        getVal(testData.data(), count, idx, sequenceCount, sequenceLength);

    // Appends a sequence.
    if (useSequences) {
      const int32_t length =
          idx + sequenceLength > count ? count - idx : sequenceLength;
      for (auto i = 0; i < length; ++i) {
        outData.push_back(val);
      }
      idx += sequenceLength;
    } else {
      outData.push_back(val);
      ++idx;
    }
  } while (idx < count);
  return outData;
}

template <typename T>
void assertVector(
    const ExpectedData<T>& expected,
    const SimpleVectorPtr<T>& outVector,
    bool dbgPrintVec = false) {
  EXPECT_EQ(expected.size(), outVector->size());

  for (vector_size_t i = 0; i < expected.size(); i++) {
    auto optionalValue = expected[i];
    auto actualIsNull = outVector->isNullAt(i);
    const bool isNull = (optionalValue == std::nullopt);

    if (isNull) {
      if (dbgPrintVec) {
        LOG(INFO) << "[" << i << "]:"
                  << "NULL";
      }
      EXPECT_EQ(isNull, actualIsNull);
    } else {
      // for all-null DictionaryVector case,
      // outVector->rawValues_ could be NULL
      if (dbgPrintVec) {
        LOG(INFO) << "[" << i << "]:" << *optionalValue;
      }
      if constexpr (std::is_floating_point<T>::value) {
        if (auto isNan = std::isnan(*optionalValue)) {
          EXPECT_EQ(isNan, std::isnan(outVector->valueAt(i)));
        } else {
          EXPECT_EQ(*optionalValue, outVector->valueAt(i));
        }
      } else {
        EXPECT_EQ(*optionalValue, outVector->valueAt(i));
      }
    }
  }
}

template <typename T>
void assertExtraMetadata(
    const VectorMakerStats<T>& expectedStats,
    const SimpleVectorPtr<T>& outVector) {
  EXPECT_EQ(expectedStats.isSorted, outVector->isSorted().value());
  EXPECT_EQ(expectedStats.min.has_value(), outVector->getMin().has_value());
  EXPECT_EQ(expectedStats.max.has_value(), outVector->getMax().has_value());
  if (expectedStats.min.has_value() && outVector->getMin().has_value()) {
    EXPECT_EQ(expectedStats.min.value(), outVector->getMin().value());
  }
  if (expectedStats.max.has_value() && outVector->getMax().has_value()) {
    EXPECT_EQ(expectedStats.max.value(), outVector->getMax().value());
  }
}

template <typename T>
void assertVectorAndProperties(
    const ExpectedData<T>& expected,
    const SimpleVectorPtr<T>& outVector) {
  auto expectedStats = genVectorMakerStats(expected);

  EXPECT_EQ(expected.size(), outVector->size());
  EXPECT_EQ(
      expectedStats.distinctCount(),
      outVector->getDistinctValueCount().value());
  EXPECT_EQ(expectedStats.nullCount, outVector->getNullCount().value());
  // TODO T70862959 check getStorageBytes(), getRepresentedBytes()

  assertExtraMetadata(expectedStats, outVector);
  assertVector<T>(expected, outVector);
}

template <typename T>
SimpleVectorPtr<T> createAndAssert(
    const ExpectedData<T>& expected,
    VectorEncoding::Simple encoding) {
  auto pool = memory::getDefaultMemoryPool();
  VectorMaker maker(pool.get());

  auto vector = maker.encodedVector(encoding, expected);
  assertVectorAndProperties(expected, vector);
  return vector;
}

BufferPtr makeNulls(
    vector_size_t size,
    memory::MemoryPool* pool,
    std::function<bool(vector_size_t /*row*/)> isNullAt);

// Create a flat vector of default values of `kind` with data-dependent flags
// being set. The last row is set to NULL.
template <TypeKind kind>
VectorPtr makeFlatVectorWithFlags(
    vector_size_t size,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  T value;

  auto nulls = makeNulls(size, pool, [&](auto row) { return row == size - 1; });
  auto values = AlignedBuffer::allocate<T>(size, pool, value);
  auto vector = std::make_shared<FlatVector<T>>(
      pool,
      CppToType<T>::create(),
      nulls,
      size,
      std::move(values),
      std::vector<BufferPtr>(),
      /*stats*/ SimpleVectorStats<T>{value, value},
      /*distinctValueCount*/ 1,
      /*nullCount*/ 1,
      /*isSorted*/ true,
      /*representedBytes*/ 0,
      /*storageByteCount*/ 0);
  vector->computeAndSetIsAscii(SelectivityVector(size - 1));
  return vector;
}

// Create a constant vector of default values of `kind` with data-dependent
// flags being set.
template <TypeKind kind>
VectorPtr makeConstantVectorWithFlags(
    vector_size_t size,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  T value;

  auto vector = std::make_shared<ConstantVector<T>>(
      pool,
      size,
      false,
      CppToType<T>::create(),
      std::move(value),
      /*stats*/ SimpleVectorStats<T>{value, value},
      /*representedBytes*/ 0,
      /*storageByteCount*/ 0);
  vector->computeAndSetIsAscii(SelectivityVector(size));
  return vector;
}

// Create a dictionary vector of default values of `kind` with data-dependent
// flags being set. The indices are identity mapping and the last row is set to
// null.
template <TypeKind kind>
VectorPtr makeDictionaryVectorWithFlags(
    vector_size_t size,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<kind>::NativeType;
  using TVariant = typename detail::VariantTypeTraits<kind>::native_type;
  T value;
  TVariant variant{value};

  auto nulls = makeNulls(size, pool, [&](auto row) { return row == size - 1; });
  auto base =
      BaseVector::createConstant(CppToType<T>::create(), variant, size, pool);
  auto vector = std::make_shared<DictionaryVector<T>>(
      pool,
      nulls,
      size,
      base,
      test::makeIndices(
          size, [](auto row) { return row; }, pool),
      /*stats*/ SimpleVectorStats<T>{value, value},
      /*distinctValueCount*/ 1,
      /*nullCount*/ 1,
      /*isSorted*/ true,
      /*representedBytes*/ 0,
      /*storageByteCount*/ 0);
  vector->computeAndSetIsAscii(SelectivityVector(size - 1));
  return vector;
}

// Create a flat map vector of default values of `keyKind` and `valueKind` with
// data-dependent flags being set. The last row of the map vector and of its
// keys and values vectors are set to null.
template <TypeKind keyKind, TypeKind valueKind>
VectorPtr makeMapVectorWithFlags(vector_size_t size, memory::MemoryPool* pool) {
  using K = typename TypeTraits<keyKind>::NativeType;
  using V = typename TypeTraits<valueKind>::NativeType;
  K key;
  V value;

  auto keys = makeFlatVectorWithFlags<keyKind>(size, pool);
  auto values = makeDictionaryVectorWithFlags<valueKind>(size, pool);

  auto offsets = allocateOffsets(size, pool);
  auto* rawOffsets = offsets->asMutable<vector_size_t>();
  auto sizes = allocateSizes(size, pool);
  auto* rawSizes = sizes->asMutable<vector_size_t>();
  for (auto i = 0; i < size; ++i) {
    rawOffsets[i] = i;
    rawSizes[i] = 1;
  }

  auto nulls = makeNulls(size, pool, [&](auto row) { return row == size - 1; });
  auto vector = std::make_shared<MapVector>(
      pool,
      MAP(CppToType<K>::create(), CppToType<V>::create()),
      nulls,
      size,
      offsets,
      sizes,
      keys,
      values,
      /*nullCount*/ 1,
      /*sortedKeys*/ true);
  return vector;
}

// Create a vector through createVector and verify that data-dependent flags
// are set. Then call ensureWritable() or prepareForReuse() through makeMutable
// and verify that data-dependent flags are cleared after the call. mutatedRows
// specifies the rows that are mutated by makeMutable where the asciiness flag
// should be cleared (if exists). If createVector creates an array or map
// vector, mutatedRows should select into its element vector.
void checkVectorFlagsReset(
    const std::function<VectorPtr()>& createVector,
    const std::function<void(VectorPtr&)>& makeMutable,
    const SelectivityVector& mutatedRows);

} // namespace facebook::velox::test
