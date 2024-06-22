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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class RandnTest : public SparkFunctionBaseTest {
 public:
  RandnTest() {
    // Allow for parsing literal integers as INTEGER, not BIGINT.
    options_.parseIntegerAsBigint = false;
  }

 protected:
  std::optional<double> randn(int32_t seed, int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    return evaluateOnce<double>(
        fmt::format("randn({})", seed), makeRowVector(ROW({}), 1));
  }

  std::optional<double> randnWithNullSeed(int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    std::optional<int32_t> seed = std::nullopt;
    return evaluateOnce<double>("randn(c0)", seed);
  }

  std::optional<double> randnWithNoSeed() {
    setSparkPartitionId(0);
    return evaluateOnce<double>("randn()", makeRowVector(ROW({}), 1));
  }

  VectorPtr randnWithBatchInput(int32_t seed, int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    auto exprSet = compileExpression(fmt::format("randn({})", seed), ROW({}));
    return evaluate(*exprSet, makeRowVector(ROW({}), 20));
  }

  void checkResult(const std::optional<double>& result) {
    EXPECT_NE(result, std::nullopt);
    EXPECT_GE(result.value(), 0.0);
    EXPECT_LT(result.value(), 1.0);
  }

  // Check whether two vectors that have same size & type, but not all same
  // values.
  void assertNotEqualVectors(const VectorPtr& left, const VectorPtr& right) {
    ASSERT_EQ(left->size(), right->size());
    ASSERT_TRUE(left->type()->equivalent(*right->type()));
    for (auto i = 0; i < left->size(); i++) {
      if (!left->equalValueAt(right.get(), i, i)) {
        return;
      }
    }
    FAIL() << "Expect two different vectors are produced.";
  }
};

TEST_F(RandnTest, withSeed) {
  checkResult(randn(0));
  // With same default partitionIndex used, same seed always produces same
  // result.
  EXPECT_EQ(randn(0), randn(0));

  checkResult(randn(1));
  EXPECT_EQ(randn(1), randn(1));

  checkResult(randn(20000));
  EXPECT_EQ(randn(20000), randn(20000));

  // Test with same seed, but different partitionIndex.
  EXPECT_NE(randn(0, 0), randn(0, 1));
  EXPECT_NE(randn(1000, 0), randn(1000, 1));

  checkResult(randnWithNullSeed());
  // Null as seed is identical to 0 as seed.
  EXPECT_EQ(randnWithNullSeed(), randn(0));
  // Same null as seed but different partition index.
  EXPECT_NE(randnWithNullSeed(0), randnWithNullSeed(1));

  // Test with batch input.
  auto batchResult1 = randnWithBatchInput(100);
  ASSERT_FALSE(batchResult1->isConstantEncoding());
  auto batchResult2 = randnWithBatchInput(100);
  // Same seed & partition index produce same results.
  velox::test::assertEqualVectors(batchResult1, batchResult2);
  batchResult1 = randnWithBatchInput(100, 0 /*partitionIndex*/);
  batchResult2 = randnWithBatchInput(100, 1 /*partitionIndex*/);
  // Same seed but different partition index cannot produce absolutely same
  // result.
  assertNotEqualVectors(batchResult1, batchResult2);
}

TEST_F(RandnTest, withoutSeed) {
  auto result1 = randnWithNoSeed();
  auto result2 = randnWithNoSeed();
  auto result3 = randnWithNoSeed();
  checkResult(result1);
  checkResult(result2);
  checkResult(result3);
  // It is impossible to get three same results by three separate callings.
  EXPECT_FALSE(
      (result1.value() == result2.value()) &&
      (result1.value() == result3.value()));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
