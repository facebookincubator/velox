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
#include <cmath>
#include <limits>

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

  VectorPtr randnWithBatchInput(int32_t seed, int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    auto exprSet = compileExpression(fmt::format("randn({})", seed), ROW({}));
    return evaluate(*exprSet, makeRowVector(ROW({}), 20));
  }
};

TEST_F(RandnTest, returnsDouble) {
  auto result = randn(42);
  EXPECT_NE(result, std::nullopt);
  // Gaussian values can be any real number, so just verify it is finite.
  EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(RandnTest, withSeed) {
  // Same seed and partition index produce the same result.
  EXPECT_EQ(randn(0), randn(0));
  EXPECT_EQ(randn(1), randn(1));
  EXPECT_EQ(randn(20000), randn(20000));

  // Same seed, different partition index produce different results.
  EXPECT_NE(randn(0, 0), randn(0, 1));
  EXPECT_NE(randn(1000, 0), randn(1000, 1));

  // Null as seed is identical to 0 as seed.
  EXPECT_EQ(randnWithNullSeed(), randn(0));
  // Same null seed but different partition index.
  EXPECT_NE(randnWithNullSeed(0), randnWithNullSeed(1));
}

TEST_F(RandnTest, batchDeterminism) {
  auto batchResult1 = randnWithBatchInput(100);
  ASSERT_FALSE(batchResult1->isConstantEncoding());
  auto batchResult2 = randnWithBatchInput(100);
  // Same seed & partition index produce same results.
  velox::test::assertEqualVectors(batchResult1, batchResult2);

  // Same seed but different partition index cannot produce same result.
  batchResult1 = randnWithBatchInput(100, 0);
  batchResult2 = randnWithBatchInput(100, 1);
  ASSERT_EQ(batchResult1->size(), batchResult2->size());
  ASSERT_TRUE(batchResult1->type()->equivalent(*batchResult2->type()));
  bool anyDifferent = false;
  for (auto i = 0; i < batchResult1->size(); i++) {
    if (!batchResult1->equalValueAt(batchResult2.get(), i, i)) {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

TEST_F(RandnTest, sparkGoldenValues) {
  // Pin output to known Spark reference values from XORShiftRandom +
  // nextGaussian(). These values can be verified via:
  // spark.sql("SELECT randn(0) FROM range(5)") with partition 0.
  setSparkPartitionId(0);
  auto exprSet = compileExpression("randn(0)", ROW({}));
  auto results = evaluate(*exprSet, makeRowVector(ROW({}), 5));
  auto flat = results->asFlatVector<double>();
  EXPECT_DOUBLE_EQ(flat->valueAt(0), 1.6034991609278433);
  EXPECT_DOUBLE_EQ(flat->valueAt(1), 0.14416006165776865);
  EXPECT_DOUBLE_EQ(flat->valueAt(2), -0.62535644986277439);
  EXPECT_DOUBLE_EQ(flat->valueAt(3), -0.28385414448030416);
  EXPECT_DOUBLE_EQ(flat->valueAt(4), 0.93334950048846421);
}

TEST_F(RandnTest, zeroArgInitializes) {
  // Zero-arg randn() must initialize its generator and return a finite value.
  setSparkPartitionId(0);
  auto result = evaluateOnce<double>("randn()", makeRowVector(ROW({}), 1));
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(std::isfinite(result.value()));
}

TEST_F(RandnTest, bigintSeed) {
  // Exercise the int64_t seed registration path, including boundary values.
  auto randnBigint = [&](int64_t seed, int32_t partitionIndex = 0) {
    setSparkPartitionId(partitionIndex);
    return evaluateOnce<double>(
        fmt::format("randn(cast({} as bigint))", seed),
        makeRowVector(ROW({}), 1));
  };

  // BIGINT max/min boundaries.
  auto maxResult = randnBigint(std::numeric_limits<int64_t>::max());
  ASSERT_TRUE(maxResult.has_value());
  EXPECT_TRUE(std::isfinite(maxResult.value()));

  auto minResult = randnBigint(std::numeric_limits<int64_t>::min());
  ASSERT_TRUE(minResult.has_value());
  EXPECT_TRUE(std::isfinite(minResult.value()));

  // Negative BIGINT seed.
  auto negResult = randnBigint(-1L);
  ASSERT_TRUE(negResult.has_value());
  EXPECT_TRUE(std::isfinite(negResult.value()));

  // Determinism: same BIGINT seed produces same output.
  EXPECT_EQ(
      randnBigint(std::numeric_limits<int64_t>::max()),
      randnBigint(std::numeric_limits<int64_t>::max()));
  EXPECT_EQ(randnBigint(-1L), randnBigint(-1L));

  // Different seeds produce different output.
  EXPECT_NE(randnBigint(1L), randnBigint(2L));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
