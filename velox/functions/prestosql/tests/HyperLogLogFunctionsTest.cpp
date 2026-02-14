/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#define XXH_INLINE_ALL
#include <xxhash.h>

using namespace facebook::velox::common::hll;

namespace facebook::velox {
namespace {

template <typename T>
uint64_t hashOne(T value) {
  return XXH64(&value, sizeof(value), 0);
}

class HyperLogLogFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  template <typename TAllocator>
  static std::string serialize(
      int8_t indexBitLength,
      const SparseHll<TAllocator>& sparseHll) {
    std::string serialized;
    serialized.resize(sparseHll.serializedSize());
    sparseHll.serialize(indexBitLength, serialized.data());
    return serialized;
  }

  template <typename TAllocator>
  static std::string serialize(DenseHll<TAllocator>& denseHll) {
    std::string serialized;
    serialized.resize(denseHll.serializedSize());
    denseHll.serialize(serialized.data());
    return serialized;
  }

  // Creates sparse HLL with values from start (inclusive) to end (exclusive).
  // modValue applies modulo operation if non-zero.
  std::string
  createSparse(int8_t indexBitLength, int start, int end, int modValue = 0) {
    SparseHll<> hll{&allocator_};
    for (int i = start; i < end; i++) {
      hll.insertHash(hashOne(modValue ? (i % modValue) : i));
    }
    return serialize(indexBitLength, hll);
  }

  // Creates dense HLL with values from start (inclusive) to end (exclusive).
  // modValue applies modulo operation if non-zero.
  std::string
  createDense(int8_t indexBitLength, int start, int end, int modValue = 0) {
    DenseHll<> hll{indexBitLength, &allocator_};
    for (int i = start; i < end; i++) {
      hll.insertHash(hashOne(modValue ? (i % modValue) : i));
    }
    return serialize(hll);
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(HyperLogLogFunctionsTest, cardinalitySignatures) {
  auto signatures = getSignatureStrings("cardinality");
  ASSERT_LE(3, signatures.size());

  ASSERT_EQ(1, signatures.count("(map(__user_T1,__user_T2)) -> bigint"));
  ASSERT_EQ(1, signatures.count("(array(__user_T1)) -> bigint"));
  ASSERT_EQ(1, signatures.count("(hyperloglog) -> bigint"));
}

TEST_F(HyperLogLogFunctionsTest, emptyApproxSetSignatures) {
  auto signatures = getSignatureStrings("empty_approx_set");
  ASSERT_EQ(2, signatures.size());

  ASSERT_EQ(1, signatures.count("(constant double) -> hyperloglog"));
  ASSERT_EQ(1, signatures.count("() -> hyperloglog"));
}

TEST_F(HyperLogLogFunctionsTest, cardinalitySparse) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", HYPERLOGLOG(), input);
  };

  auto serialized = createSparse(11, 0, 1000, 17);
  EXPECT_EQ(17, cardinality(serialized));
}

TEST_F(HyperLogLogFunctionsTest, cardinalityDense) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", HYPERLOGLOG(), input);
  };

  DenseHll<> expectedHll{12, &allocator_};
  for (int i = 0; i < 10'000'000; i++) {
    expectedHll.insertHash(hashOne(i));
  }
  auto expectedCardinality = expectedHll.cardinality();

  auto serialized = createDense(12, 0, 10'000'000);
  EXPECT_EQ(expectedCardinality, cardinality(serialized));
}

TEST_F(HyperLogLogFunctionsTest, emptyApproxSet) {
  EXPECT_EQ(
      0,
      evaluateOnce<int64_t>(
          "cardinality(empty_approx_set())", makeRowVector(ROW({}), 1)));

  EXPECT_EQ(
      0,
      evaluateOnce<int64_t>(
          "cardinality(empty_approx_set(0.1))", makeRowVector(ROW({}), 1)));
}

TEST_F(HyperLogLogFunctionsTest, mergeHll) {
  // Test merging two HLL instances with different value ranges.
  const int8_t indexBitLength = 12;

  std::vector<std::string> serializedHlls = {
      createSparse(indexBitLength, 0, 10),
      createSparse(indexBitLength, 10, 20)};

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinality = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  EXPECT_EQ(20, cardinality);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllMixedSparseAndDense) {
  // Test merging sparse and dense HLL instances.
  const int8_t indexBitLength = 12;

  std::vector<std::string> serializedHlls = {
      createSparse(indexBitLength, 0, 50),
      createDense(indexBitLength, 50, 2000)};

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinality = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  EXPECT_EQ(1978, cardinality);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllWithNull) {
  auto serialized = createSparse(12, 0, 10);
  auto elements = makeNullableFlatVector<std::string>(
      {serialized, std::nullopt, serialized}, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 3}, elements);

  auto cardinality = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));
  EXPECT_EQ(10, cardinality);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllEmpty) {
  auto input = makeRowVector(
      {makeArrayVectorFromJson<std::string>({"null"}, ARRAY(HYPERLOGLOG()))});
  auto result = evaluate("merge_hll(c0)", input);
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(HyperLogLogFunctionsTest, mergeHllNullArray) {
  auto input = makeRowVector({makeNullableArrayVector<std::string>(
      {std::nullopt}, ARRAY(HYPERLOGLOG()))});
  auto result = evaluate("merge_hll(c0)", input);
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(HyperLogLogFunctionsTest, mergeHllAllNulls) {
  auto input = makeRowVector({makeArrayVectorFromJson<std::string>(
      {"[null, null, null]"}, ARRAY(HYPERLOGLOG()))});
  auto result = evaluate("merge_hll(c0)", input);
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(HyperLogLogFunctionsTest, mergeHllInvalidData) {
  // Test with invalid HLL data format.
  std::string invalidData = "invalid_hll_data";
  auto elements = makeFlatVector<std::string>({invalidData}, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 1}, elements);

  VELOX_ASSERT_THROW(
      evaluate("merge_hll(c0)", makeRowVector({arrayVector})),
      "Invalid HLL data format");
}

TEST_F(HyperLogLogFunctionsTest, mergeHllMismatchedIndexBitLength) {
  const int8_t indexBitLength1 = 11;
  const int8_t indexBitLength2 = 12;

  std::vector<std::string> serializedHlls = {
      createSparse(indexBitLength1, 0, 5),
      createSparse(indexBitLength2, 5, 10)};

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  VELOX_ASSERT_THROW(
      evaluate("merge_hll(c0)", makeRowVector({arrayVector})),
      "Cannot merge HLLs with different indexBitLength");
}

TEST_F(
    HyperLogLogFunctionsTest,
    mergeHllMismatchedIndexBitLengthDenseVsSparse) {
  const int8_t indexBitLength1 = 11;
  const int8_t indexBitLength2 = 12;

  std::vector<std::string> serializedHlls = {
      createSparse(indexBitLength1, 0, 50),
      createDense(indexBitLength2, 50, 2000)};

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  VELOX_ASSERT_THROW(
      evaluate("merge_hll(c0)", makeRowVector({arrayVector})),
      "Cannot merge HLLs with different indexBitLength");
}

TEST_F(HyperLogLogFunctionsTest, mergeHllDenseFirstThenSparse) {
  const int8_t indexBitLength = 12;

  std::vector<std::string> serializedHlls = {
      createDense(indexBitLength, 0, 2000),
      createSparse(indexBitLength, 2000, 2050)};

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinality = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  EXPECT_EQ(2035, cardinality);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllDictionaryEncoded) {
  // Test merge_hll can handle encoded array elements.
  const int8_t indexBitLength = 12;

  auto baseElements = makeFlatVector(
      {createSparse(indexBitLength, 0, 10),
       createSparse(indexBitLength, 10, 20)},
      HYPERLOGLOG());

  struct TestCase {
    std::vector<vector_size_t> indices;
    std::optional<std::vector<bool>> nulls;
    int64_t expectedCardinality;
  };

  std::vector<TestCase> testCases = {
      // Dictionary indices without nulls.
      {{0, 1}, std::nullopt, 20},
      {{1, 0}, std::nullopt, 20},
      {{1, 1, 1}, std::nullopt, 10},
      {{0, 0, 0, 0}, std::nullopt, 10},
      {{1, 1, 0, 0, 1, 0}, std::nullopt, 20},
      // Dictionary adds nulls: [1, null]
      {{1, 0}, {{false, true}}, 10},
      // Dictionary adds nulls: [null, 0, null]
      {{0, 0, 0}, {{true, false, true}}, 10},
      // Dictionary adds nulls: [null, null, 1, 1, null, 0]
      {{0, 0, 1, 1, 0, 0}, {{true, true, false, false, true, false}}, 20},
  };

  for (const auto& testCase : testCases) {
    auto indices = makeIndices(testCase.indices);
    VectorPtr encodedElements;
    if (testCase.nulls.has_value()) {
      auto nulls = makeNulls(testCase.nulls.value());
      encodedElements = BaseVector::wrapInDictionary(
          nulls, indices, testCase.indices.size(), baseElements);
    } else {
      encodedElements = wrapInDictionary(indices, baseElements);
    }
    auto arrayVector = makeArrayVector(
        {0, static_cast<int>(testCase.indices.size())}, encodedElements);
    auto cardinality = evaluateOnce<int64_t>(
        "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));
    EXPECT_EQ(testCase.expectedCardinality, cardinality);
  }
}

} // namespace
} // namespace facebook::velox
