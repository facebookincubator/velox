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

using facebook::velox::common::hll::SparseHll;
using namespace facebook::velox::common::hll;

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
      common::hll::SparseHll<>& sparseHll) {
    std::string serialized;
    serialized.resize(sparseHll.serializedSize());
    sparseHll.serialize(indexBitLength, serialized.data());
    return serialized;
  }

  static std::string serialize(common::hll::DenseHll<>& denseHll) {
    std::string serialized;
    serialized.resize(denseHll.serializedSize());
    denseHll.serialize(serialized.data());
    return serialized;
  }

  static std::string serialize(
      common::hll::DenseHll<memory::StlAllocator<uint8_t>>& denseHll) {
    std::string serialized;
    serialized.resize(denseHll.serializedSize());
    denseHll.serialize(serialized.data());
    return serialized;
  }

  static std::string serialize(
      int8_t indexBitLength,
      common::hll::SparseHll<memory::StlAllocator<uint8_t>>& sparseHll) {
    std::string serialized;
    serialized.resize(sparseHll.serializedSize());
    sparseHll.serialize(indexBitLength, serialized.data());
    return serialized;
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  // For tests using HashStringAllocator
  HashStringAllocator hsAllocator_{pool_.get()};
  // For tests using MemoryPoolStlAllocator
  memory::StlAllocator<uint8_t> allocator_{*pool_};
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

  common::hll::SparseHll<> sparseHll{&hsAllocator_};
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

  common::hll::DenseHll<> denseHll{12, &hsAllocator_};
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
  // Create simple test using memory::StlAllocator to match merge_hll
  // implementation
  const int8_t indexBitLength = 12;

  // Create sparse HLL with elements 0-9 using memory::StlAllocator
  common::hll::SparseHll<memory::StlAllocator<uint8_t>> sparseHll1{allocator_};
  for (int i = 0; i < 10; i++) {
    sparseHll1.insertHash(hashOne(i));
  }
  auto serialized1 = serialize(indexBitLength, sparseHll1);

  // Create sparse HLL with elements 10-19 using memory::StlAllocator
  common::hll::SparseHll<memory::StlAllocator<uint8_t>> sparseHll2{allocator_};
  for (int i = 10; i < 20; i++) {
    sparseHll2.insertHash(hashOne(i));
  }
  auto serialized2 = serialize(indexBitLength, sparseHll2);

  // Use makeFlatVector to create binary data properly
  auto elements = makeFlatVector<StringView>(
      {StringView(serialized1), StringView(serialized2)}, HYPERLOGLOG());
  auto arrayVector = makeArrayVector({0, 2}, elements);

  auto cardinalityValue = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));

  // Should be close to 20 unique elements
  EXPECT_GE(cardinalityValue, 18);
  EXPECT_LE(cardinalityValue, 22);
}

TEST_F(HyperLogLogFunctionsTest, mergeHllWithNull) {
  const int8_t indexBitLength = 12;

  common::hll::SparseHll<memory::StlAllocator<uint8_t>> sparseHll{allocator_};
  for (int i = 0; i < 10; i++) {
    sparseHll.insertHash(hashOne(i));
  }
  auto serialized = serialize(indexBitLength, sparseHll);
  auto elements = makeFlatVector<StringView>(
      {StringView(serialized), StringView(), StringView(serialized)},
      HYPERLOGLOG());
  elements->setNull(1, true);
  auto arrayVector = makeArrayVector({0, 3}, elements);

  auto cardinalityValue = evaluateOnce<int64_t>(
      "cardinality(merge_hll(c0))", makeRowVector({arrayVector}));
  EXPECT_GE(cardinalityValue, 8);
  EXPECT_LE(cardinalityValue, 12);
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

} // namespace
} // namespace facebook::velox
