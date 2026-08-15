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
#include <random>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::functions::aggregate::test;
using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;

namespace facebook::velox::aggregate::test {

class ApproxWinsorizedMeanAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }
};

TEST_F(ApproxWinsorizedMeanAggregateTest, globalFullRange) {
  auto vectors = makeRowVector(
      {makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 5.0}),
       makeConstant<double>(0.0, 5),
       makeConstant<double>(1.0, 5)});
  auto expected = makeRowVector({makeFlatVector<double>({3.0})});
  testAggregations(
      {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2)"}, {expected});
}

TEST_F(ApproxWinsorizedMeanAggregateTest, groupBy) {
  auto vectors = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 1, 1, 1, 2, 2, 2, 2, 2}),
       makeFlatVector<double>(
           {1.0, 2.0, 3.0, 4.0, 100.0, 10.0, 20.0, 30.0, 40.0, 50.0}),
       makeConstant<double>(0.0, 10),
       makeConstant<double>(1.0, 10)});

  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}), makeFlatVector<double>({22.0, 30.0})});
  testAggregations(
      {vectors}, {"c0"}, {"approx_winsorized_mean(c1, c2, c3)"}, {expected});
}

TEST_F(ApproxWinsorizedMeanAggregateTest, nullHandling) {
  auto vectors = makeRowVector(
      {makeNullableFlatVector<double>(
           {1.0, std::nullopt, 3.0, std::nullopt, 5.0}),
       makeConstant<double>(0.0, 5),
       makeConstant<double>(1.0, 5)});
  auto expected = makeRowVector({makeFlatVector<double>({3.0})});
  testAggregations(
      {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2)"}, {expected});
}

TEST_F(ApproxWinsorizedMeanAggregateTest, allNulls) {
  auto vectors = makeRowVector(
      {makeNullableFlatVector<double>({std::nullopt, std::nullopt}),
       makeConstant<double>(0.0, 2),
       makeConstant<double>(1.0, 2)});
  auto expected =
      makeRowVector({makeNullableFlatVector<double>({std::nullopt})});
  testAggregations(
      {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2)"}, {expected});
}

TEST_F(ApproxWinsorizedMeanAggregateTest, invalidBounds) {
  auto vectors = makeRowVector(
      {makeFlatVector<double>({1.0, 2.0, 3.0}),
       makeConstant<double>(-0.1, 3),
       makeConstant<double>(0.5, 3)});
  VELOX_ASSERT_THROW(
      testAggregations(
          {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2)"}, {""}),
      ">= 0");
}

TEST_F(ApproxWinsorizedMeanAggregateTest, boundsReversed) {
  auto vectors = makeRowVector(
      {makeFlatVector<double>({1.0, 2.0, 3.0}),
       makeConstant<double>(0.9, 3),
       makeConstant<double>(0.1, 3)});
  VELOX_ASSERT_THROW(
      testAggregations(
          {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2)"}, {""}),
      "less than or equal");
}

TEST_F(ApproxWinsorizedMeanAggregateTest, nanInput) {
  auto vectors = makeRowVector(
      {makeFlatVector<double>({1.0, std::numeric_limits<double>::quiet_NaN()}),
       makeConstant<double>(0.0, 2),
       makeConstant<double>(1.0, 2)});
  VELOX_ASSERT_THROW(
      testAggregations(
          {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2)"}, {""}),
      "Cannot add NaN");
}

TEST_F(ApproxWinsorizedMeanAggregateTest, withCompression) {
  std::vector<double> values(1'000);
  for (int i = 0; i < 1'000; ++i) {
    values[i] = i;
  }
  auto vectors = makeRowVector(
      {makeFlatVector<double>(values),
       makeConstant<double>(0.0, 1'000),
       makeConstant<double>(1.0, 1'000),
       makeConstant<double>(200.0, 1'000)});
  auto expected = makeRowVector({makeFlatVector<double>({499.5})});
  testAggregations(
      {vectors}, {}, {"approx_winsorized_mean(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ApproxWinsorizedMeanAggregateTest, nonDefaultBoundsDistributed) {
  // Exercises non-default quantile bounds (0.01, 0.99) through an explicit
  // partial→final plan to verify that quantile bounds survive the
  // intermediate varbinary serialization.
  constexpr int N{1'000};
  std::vector<double> values(N);
  for (int i = 0; i < N; ++i) {
    values[i] = i;
  }
  auto vectors = makeRowVector(
      {makeFlatVector<double>(values),
       makeConstant<double>(0.01, N),
       makeConstant<double>(0.99, N)});

  // Build partial→final plan to test distributed aggregation path.
  auto plan =
      PlanBuilder()
          .values({vectors})
          .partialAggregation({}, {"approx_winsorized_mean(c0, c1, c2)"})
          .finalAggregation()
          .planNode();
  auto resultVector = AssertQueryBuilder(plan).copyResults(pool());
  auto result = resultVector->childAt(0)->asFlatVector<double>()->valueAt(0);
  // Uniform distribution winsorized at 1%/99% — mean should be close to
  // regular mean (499.5) since uniform is symmetric.
  ASSERT_NEAR(result, 499.5, 499.5 * 0.02);
}

TEST_F(ApproxWinsorizedMeanAggregateTest, accuracy) {
  constexpr int N{100'000};
  std::vector<double> values(N);
  std::default_random_engine generator(42);
  std::lognormal_distribution<> distribution(0, 1.0);
  for (int i = 0; i < N; ++i) {
    values[i] = distribution(generator);
  }

  auto vectors = makeRowVector(
      {makeFlatVector<double>(values),
       makeConstant<double>(0.01, N),
       makeConstant<double>(0.99, N)});

  std::sort(values.begin(), values.end());
  int lowerIndex{static_cast<int>(N * 0.01)};
  int upperIndex{static_cast<int>(N * 0.99)};
  double lowBound{values[lowerIndex]};
  double highBound{values[upperIndex]};
  double exactSum{0};
  for (auto value : values) {
    exactSum += std::max(lowBound, std::min(highBound, value));
  }
  double exactMean{exactSum / N};

  auto plan = PlanBuilder()
                  .values({vectors})
                  .singleAggregation({}, {"approx_winsorized_mean(c0, c1, c2)"})
                  .planNode();
  auto resultVector = AssertQueryBuilder(plan).copyResults(pool());
  auto result = resultVector->childAt(0)->asFlatVector<double>()->valueAt(0);
  ASSERT_NEAR(result, exactMean, std::abs(exactMean) * 0.05);
}

} // namespace facebook::velox::aggregate::test
