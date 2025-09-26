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

  // Helpers to create HLL for testing
  template <bool IsSparse = true>
  std::string
  createHll(int8_t indexBitLength, int start, int end, int modValue = 0) {
    if constexpr (IsSparse) {
      SparseHll<> hll{&allocator_};
      for (int i = start; i < end; i++) {
        hll.insertHash(hashOne(modValue ? (i % modValue) : i));
      }
      return serialize(indexBitLength, hll);
    } else {
      DenseHll<> hll{indexBitLength, &allocator_};
      for (int i = start; i < end; i++) {
        hll.insertHash(hashOne(modValue ? (i % modValue) : i));
      }
      return serialize(hll);
    }
  }

  template <typename HllType>
  void populateHll(HllType& hll, int start, int end, int modValue = 0) {
    for (int i = start; i < end; i++) {
      hll.insertHash(hashOne(modValue ? (i % modValue) : i));
    }
  }

  void addSerializedHll(
      std::vector<std::string>& serializedHlls,
      DenseHll<>& denseHll) {
    serializedHlls.emplace_back(serialize(denseHll));
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

  auto serialized = createHll<true>(11, 0, 1000, 17);
  EXPECT_EQ(17, cardinality(serialized));
}

TEST_F(HyperLogLogFunctionsTest, cardinalityDense) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", HYPERLOGLOG(), input);
  };

  DenseHll<> expectedHll{12, &allocator_};
  populateHll(expectedHll, 0, 10000000);
  auto expectedCardinality = expectedHll.cardinality();

  auto serialized = createHll<false>(12, 0, 10000000);
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

  std::vector<std::string> serializedHlls;
  serializedHlls.emplace_back(createHll<true>(indexBitLength, 0, 10));
  serializedHlls.emplace_back(createHll<true>(indexBitLength, 10, 20));

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinalityValue = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  EXPECT_EQ(20, cardinalityValue);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllMixedSparseAndDense) {
  // Test merging sparse and dense HLL instances.
  const int8_t indexBitLength = 12;

  std::vector<std::string> serializedHlls;

  // Create sparse HLL with fewer elements (stays sparse)
  serializedHlls.emplace_back(createHll<true>(indexBitLength, 0, 50));

  // Create dense HLL with many elements (becomes dense)
  serializedHlls.emplace_back(createHll<false>(indexBitLength, 50, 2000));

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinalityValue = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  EXPECT_EQ(1978, cardinalityValue);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllWithNull) {
  auto serialized = createHll<true>(12, 0, 10);
  auto elements = makeFlatVector<StringView>(
      {StringView(serialized), StringView(), StringView(serialized)},
      HYPERLOGLOG());
  elements->setNull(1, true);
  auto arrayVector = makeArrayVector({0, 3}, elements);

  auto cardinalityValue = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));
  EXPECT_EQ(10, cardinalityValue);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllEmpty) {
  auto input =
      makeRowVector({makeArrayVector<StringView>({{}}, HYPERLOGLOG())});
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
  // Test with invalid HLL data format
  auto invalidData = std::string("invalid_hll_data");
  auto elements =
      makeFlatVector<StringView>({StringView(invalidData)}, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 1}, elements);

  VELOX_ASSERT_THROW(
      evaluate("merge_hll(c0)", makeRowVector({arrayVector})),
      "Invalid HLL data format");
}

TEST_F(HyperLogLogFunctionsTest, mergeHllMismatchedIndexBitLength) {
  const int8_t indexBitLength1 = 11;
  const int8_t indexBitLength2 = 12;

  std::vector<std::string> serializedHlls;

  // Create first HLL with indexBitLength = 11
  serializedHlls.emplace_back(createHll<true>(indexBitLength1, 0, 5));

  // Create second HLL with indexBitLength = 12
  serializedHlls.emplace_back(createHll<true>(indexBitLength2, 5, 10));

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

  std::vector<std::string> serializedHlls;

  // Create first HLL with indexBitLength = 11 (sparse)
  serializedHlls.emplace_back(createHll<true>(indexBitLength1, 0, 50));

  // Create second HLL with indexBitLength = 12 (dense)
  serializedHlls.emplace_back(createHll<false>(indexBitLength2, 50, 2000));

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  VELOX_ASSERT_THROW(
      evaluate("merge_hll(c0)", makeRowVector({arrayVector})),
      "Cannot merge HLLs with different indexBitLength");
}

TEST_F(HyperLogLogFunctionsTest, mergeHllDenseFirstThenSparse) {
  const int8_t indexBitLength = 12;

  std::vector<std::string> serializedHlls;

  // Create dense HLL first (with many elements)
  serializedHlls.emplace_back(createHll<false>(indexBitLength, 0, 2000));

  // Create sparse HLL second (with fewer elements)
  serializedHlls.emplace_back(createHll<true>(indexBitLength, 2000, 2050));

  auto elements = makeFlatVector(serializedHlls, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinalityValue = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  EXPECT_EQ(2035, cardinalityValue);
}

} // namespace
} // namespace facebook::velox
