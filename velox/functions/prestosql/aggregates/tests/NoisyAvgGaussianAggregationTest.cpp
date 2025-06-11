// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {
class NoisyAvgGaussianAggregationTest
    : public functions::aggregate::test::AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  RowTypePtr doubleRowType_{
      ROW({"c0", "c1", "c2"}, {DOUBLE(), DOUBLE(), DOUBLE()})};
  RowTypePtr bigintRowType_{
      ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()})};
  RowTypePtr decimalRowType_{
      ROW({"c0", "c1", "c2"},
          {DECIMAL(20, 5), DECIMAL(20, 5), DECIMAL(20, 5)})};
  RowTypePtr realRowType_{ROW({"c0", "c1", "c2"}, {REAL(), REAL(), REAL()})};
  RowTypePtr integerRowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), INTEGER()})};
  RowTypePtr smallintRowType_{
      ROW({"c0", "c1", "c2"}, {SMALLINT(), SMALLINT(), SMALLINT()})};
  RowTypePtr tinyintRowType_{
      ROW({"c0", "c1", "c2"}, {TINYINT(), TINYINT(), TINYINT()})};
};

TEST_F(NoisyAvgGaussianAggregationTest, basicNoNoise) {
  auto vectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({1, 1, 1, 1, 1}),
      makeFlatVector<double>({1, 2, 3, 4, 5}),
  })};

  createDuckDbTable(vectors);

  testAggregations(
      {vectors},
      {},
      {"noisy_avg_gaussian(c2, 0.0)"},
      "SELECT AVG(c2) FROM tmp");
}

TEST_F(NoisyAvgGaussianAggregationTest, basicWithNoise) {
  auto vectors = {makeRowVector({
      makeFlatVector<double>({1, 2, 3, 4, 5}),
  })};

  // Set the noise scale to 0.1, true average is 3.0.
  // use +/- 50*SD and test the result is within range [-2.0, 8.0].

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values(vectors)
              .singleAggregation({}, {"noisy_avg_gaussian(c0, 0.1)"}, {})
              .planNode(),
          duckDbQueryRunner_)
          .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  ASSERT_TRUE(result->childAt(0)->asFlatVector<double>()->valueAt(0) >= -2.0);
  ASSERT_TRUE(result->childAt(0)->asFlatVector<double>()->valueAt(0) <= 8.0);
}

TEST_F(NoisyAvgGaussianAggregationTest, invalidNoiseScale) {
  auto vectors = makeVectors(doubleRowType_, 3, 3);
  createDuckDbTable(vectors);

  // Test invalid noise scale.
  testFailingAggregations(
      vectors,
      {},
      {"noisy_avg_gaussian(c2, -1.0)"},
      "Noise scale must be a non-negative value.");
}

TEST_F(NoisyAvgGaussianAggregationTest, bigintNoiseScaleType) {
  auto vectors = makeVectors(doubleRowType_, 3, 3);
  createDuckDbTable(vectors);

  testAggregations(
      vectors, {}, {"noisy_avg_gaussian(c2, 0)"}, "SELECT AVG(c2) FROM tmp");
}

TEST_F(NoisyAvgGaussianAggregationTest, groupbyNullsNoNoise) {
  auto vectors = {makeRowVector({
      makeNullableFlatVector<double>({std::nullopt, 1, 1, 1, std::nullopt}),
      makeNullableFlatVector<double>({1, 2, 3, 4, 5}),
  })};

  // Group by c0, aggregate c1. Expected result:
  // c0   | noisy_avg_gaussian(c1, 0.0)
  // NULL | 3.0
  // 1    | 3.0
  auto expectedResult = makeRowVector(
      {makeNullableFlatVector<double>({std::nullopt, 1.0}),
       makeFlatVector<double>({3.0, 3.0})});

  testAggregations(
      vectors, {"c0"}, {"noisy_avg_gaussian(c1, 0.0)"}, {expectedResult});
}

TEST_F(NoisyAvgGaussianAggregationTest, aggregateNullsNoNoise) {
  auto vectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 1, 1, 1}),
  })};

  // group by c0, aggregate c1. Expected result:
  // c0   | noisy_avg_gaussian(c1, 0.1)
  // 1    | NULL
  // 2    | 1.0
  auto expectedResult = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeNullableFlatVector<double>({std::nullopt, 1.0})});

  testAggregations(
      vectors, {"c0"}, {"noisy_avg_gaussian(c1, 0.0)"}, {expectedResult});
}

TEST_F(NoisyAvgGaussianAggregationTest, numericInputTypeTestNoNoise) {
  auto rowTypes = {
      doubleRowType_,
      bigintRowType_,
      decimalRowType_,
      realRowType_,
      integerRowType_,
      smallintRowType_,
      tinyintRowType_};

  for (auto rowType : rowTypes) {
    auto vectors = makeVectors(rowType, 3, 3);
    createDuckDbTable(vectors);

    testAggregations(
        vectors,
        {},
        {"noisy_avg_gaussian(c2, 0.0)"},
        "SELECT AVG(c2) FROM tmp");
  }
}

TEST_F(NoisyAvgGaussianAggregationTest, boundsClipTestNoNoise) {
  auto vectors = {makeRowVector({makeFlatVector<double>({1, 2, 3, 4, 5})})};

  // set the bounds to [4, 4].
  // The clipped sum should be 20 and the average should be 4.
  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values(vectors)
              // lower bigint, upper bigint
              .singleAggregation({}, {"noisy_avg_gaussian(c0, 0.0, 4, 4)"}, {})
              .planNode(),
          duckDbQueryRunner_)
          .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->childAt(0)->asFlatVector<double>()->valueAt(0), 4);

  result =
      AssertQueryBuilder(
          PlanBuilder()
              .values(vectors)
              // lower double, upper bigint
              .singleAggregation({}, {"noisy_avg_gaussian(c0, 0, 4.0, 4)"}, {})
              .planNode(),
          duckDbQueryRunner_)
          .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->childAt(0)->asFlatVector<double>()->valueAt(0), 4);

  result =
      AssertQueryBuilder(
          PlanBuilder()
              .values(vectors)
              // lower bigint, upper double
              .singleAggregation({}, {"noisy_avg_gaussian(c0, 0, 4, 4.0)"}, {})
              .planNode(),
          duckDbQueryRunner_)
          .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->childAt(0)->asFlatVector<double>()->valueAt(0), 4);

  result = AssertQueryBuilder(
               PlanBuilder()
                   .values(vectors)
                   // lower double, upper double
                   .singleAggregation(
                       {}, {"noisy_avg_gaussian(c0, 0, 4.0, 4.0)"}, {})
                   .planNode(),
               duckDbQueryRunner_)
               .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  ASSERT_EQ(result->childAt(0)->asFlatVector<double>()->valueAt(0), 4);
}

TEST_F(NoisyAvgGaussianAggregationTest, randomSeedNoNoise) {
  auto vectors = makeVectors(doubleRowType_, 3, 2);
  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {},
      {"noisy_avg_gaussian(c2, 0.0, 12345)"},
      "SELECT AVG(c2) FROM tmp");
}

TEST_F(NoisyAvgGaussianAggregationTest, randomSeedDeterminismTestWithNoise) {
  auto vectors = makeVectors(doubleRowType_, 10, 5);

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values(vectors)
              .singleAggregation({}, {"noisy_avg_gaussian(c2, 0.5, 12345)"}, {})
              .planNode(),
          duckDbQueryRunner_)
          .copyResults(pool());

  // Test that the noise is deterministic given the same noise_scale,
  // random_seed.
  for (int i = 0; i < 10; i++) {
    testAggregations(
        vectors, {}, {"noisy_avg_gaussian(c2, 0.5, 12345)"}, {result});
  }
}

} // namespace facebook::velox::aggregate::test
