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
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#define XXH_INLINE_ALL
#include <xxhash.h>

using facebook::velox::common::hll::DenseHll;
using facebook::velox::common::hll::SparseHll;

namespace facebook::velox {
namespace {

template <typename T>
uint64_t hashOne(T value) {
  return XXH64(&value, sizeof(value), 0);
}

class HyperLogLogFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  static std::string serialize(
      int8_t indexBitLength,
      const SparseHll& sparseHll) {
    std::string serialized;
    serialized.resize(sparseHll.serializedSize());
    sparseHll.serialize(indexBitLength, serialized.data());
    return serialized;
  }

  static std::string serialize(DenseHll& denseHll) {
    std::string serialized;
    serialized.resize(denseHll.serializedSize());
    denseHll.serialize(serialized.data());
    return serialized;
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};

  const TypePtr ARRAY_HYPERLOGLOG = ARRAY(HYPERLOGLOG());
};

TEST_F(HyperLogLogFunctionsTest, cardinalitySignatures) {
  auto signatures = getSignatureStrings("cardinality");
  ASSERT_EQ(3, signatures.size());

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

  SparseHll sparseHll{&allocator_};
  for (int i = 0; i < 1'000; i++) {
    sparseHll.insertHash(hashOne(i % 17));
  }

  auto serialized = serialize(11, sparseHll);
  EXPECT_EQ(17, cardinality(serialized));
}

TEST_F(HyperLogLogFunctionsTest, cardinalityDense) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", HYPERLOGLOG(), input);
  };

  DenseHll denseHll{12, &allocator_};
  for (int i = 0; i < 10'000'000; i++) {
    denseHll.insertHash(hashOne(i));
  }

  auto serialized = serialize(denseHll);
  EXPECT_EQ(denseHll.cardinality(), cardinality(serialized));
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
  // Create HLLs with different values but same index bit length
  const int8_t indexBitLength = 12;
  const int blockSize = 3;
  const int uniqueElementsPerBlock = 10000;
  const int totalUniqueElements = uniqueElementsPerBlock * blockSize;
  const double errorRate = 0.05;

  SparseHll sparseHll1{&allocator_};
  for (int i = 0; i < uniqueElementsPerBlock; i++) {
    sparseHll1.insertHash(hashOne(i));
  }
  auto serialized1 = serialize(indexBitLength, sparseHll1);
  SparseHll sparseHll2{&allocator_};
  for (int i = uniqueElementsPerBlock; i < 2 * uniqueElementsPerBlock; i++) {
    sparseHll2.insertHash(hashOne(i));
  }
  auto serialized2 = serialize(indexBitLength, sparseHll2);
  DenseHll denseHll{indexBitLength, &allocator_};
  for (int i = 2 * uniqueElementsPerBlock; i < 3 * uniqueElementsPerBlock;
       i++) {
    denseHll.insertHash(hashOne(i));
  }
  auto serialized3 = serialize(denseHll);
  auto arrayVector = makeNullableArrayVector<std::string>(
      {{serialized1, serialized2, serialized3}}, ARRAY_HYPERLOGLOG);
  auto result = evaluate("merge_hll(c0)", makeRowVector({arrayVector}));
  auto cardinalityResult = evaluate("cardinality(c0)", makeRowVector({result}));

  auto cardinalityValue =
      cardinalityResult->as<FlatVector<int64_t>>()->valueAt(0);
  int64_t allowedError = static_cast<int64_t>(totalUniqueElements * errorRate);

  EXPECT_GE(cardinalityValue, totalUniqueElements - allowedError);
  EXPECT_LE(cardinalityValue, totalUniqueElements + allowedError);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllWithNull) {
  const int8_t indexBitLength = 12;
  SparseHll sparseHll{&allocator_};
  for (int i = 0; i < 100; i++) {
    sparseHll.insertHash(hashOne(i));
  }
  auto serialized = serialize(indexBitLength, sparseHll);
  auto arrayVector = makeNullableArrayVector<std::string>(
      {{serialized, std::nullopt, serialized}}, ARRAY_HYPERLOGLOG);

  auto result = evaluate("merge_hll(c0)", makeRowVector({arrayVector}));
  auto cardinalityResult = evaluate("cardinality(c0)", makeRowVector({result}));
  auto cardinalityValue =
      cardinalityResult->as<FlatVector<int64_t>>()->valueAt(0);
  EXPECT_GE(cardinalityValue, 95);
  EXPECT_LE(cardinalityValue, 105);
}
TEST_F(HyperLogLogFunctionsTest, mergeHllEmpty) {
  auto arrayVector = makeArrayVector<std::string>({{}}, HYPERLOGLOG());
  auto result = evaluate("merge_hll(c0)", makeRowVector({arrayVector}));
  ASSERT_TRUE(result->isNullAt(0));
}

TEST_F(HyperLogLogFunctionsTest, mergeHllNullArray) {
  auto nullArrayVector =
      makeNullableArrayVector<std::string>({std::nullopt}, ARRAY_HYPERLOGLOG);
  auto expected =
      makeNullableFlatVector<std::string>({std::nullopt}, HYPERLOGLOG());
  auto result = evaluate("merge_hll(c0)", makeRowVector({nullArrayVector}));
  test::assertEqualVectors(expected, result);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllAllNulls) {
  std::vector<std::vector<std::optional<std::string>>> data = {
      {std::nullopt, std::nullopt, std::nullopt}};
  auto allNullsVector = makeNullableArrayVector(data, ARRAY(HYPERLOGLOG()));
  auto result = evaluate("merge_hll(c0)", makeRowVector({allNullsVector}));
  ASSERT_TRUE(result->isNullAt(0));
}

} // namespace
} // namespace facebook::velox
