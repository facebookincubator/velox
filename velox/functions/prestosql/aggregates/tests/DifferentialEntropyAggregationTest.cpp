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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class DifferentialEntropyAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  // Builds `count` rows of (value, weight) split across `numBatches` input
  // vectors, so the default testAggregations() plan variations (single-stage
  // and partial+final) both exercise combine()/mergeWith() across multiple
  // groups, not just addInput() within one.
  std::vector<RowVectorPtr> makeBatches(
      vector_size_t count,
      int32_t numBatches,
      const std::function<double(vector_size_t)>& valueAt,
      const std::function<double(vector_size_t)>& weightAt) {
    std::vector<RowVectorPtr> batches;
    vector_size_t batchSize = (count + numBatches - 1) / numBatches;
    for (vector_size_t offset = 0; offset < count; offset += batchSize) {
      const auto size = std::min(batchSize, count - offset);
      batches.push_back(makeRowVector(
          {makeFlatVector<double>(
               size, [&, offset](auto row) { return valueAt(offset + row); }),
           makeFlatVector<double>(size, [&, offset](auto row) {
             return weightAt(offset + row);
           })}));
    }
    return batches;
  }
};

// ---------------------------------------------------------------------------
// Fixed-histogram MLE / Jacknife: deterministic, exact-value tests.
// ---------------------------------------------------------------------------

TEST_F(DifferentialEntropyAggregationTest, fixedHistogramMle) {
  // bucket [0, 5): {1, 2, 3}, weight 1 each -> bucket weight 3.
  // bucket [5, 10]: {6, 7}, weight 1 each -> bucket weight 2.
  auto data = makeRowVector({
      makeFlatVector<double>({1, 2, 3, 6, 7}),
      makeConstant<double>(1.0, 5),
  });
  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {},
              {"differential_entropy(2, c0, c1, 'fixed_histogram_mle', 0.0, 10.0)"})
          .planNode();
  auto expected = makeRowVector({makeFlatVector<double>({3.2928786893420314})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(DifferentialEntropyAggregationTest, fixedHistogramMleAcrossBatches) {
  // Same data as fixedHistogramMle, but split across multiple input batches
  // to exercise mergeWith()/serialize()/deserialize() via partial+final
  // aggregation, not just a single addInput() sequence.
  auto batches = makeBatches(
      5,
      3,
      [](vector_size_t i) -> double {
        static const double values[] = {1, 2, 3, 6, 7};
        return values[i];
      },
      [](vector_size_t) { return 1.0; });

  auto expected = makeRowVector({makeFlatVector<double>({3.2928786893420314})});
  testAggregations(
      batches,
      {},
      {"differential_entropy(2, c0, c1, 'fixed_histogram_mle', 0.0, 10.0)"},
      {expected});
}

TEST_F(DifferentialEntropyAggregationTest, fixedHistogramJacknife) {
  // Same bucketing as fixedHistogramMle; jacknife applies a bias correction
  // so the result differs from the MLE estimate above.
  auto data = makeRowVector({
      makeFlatVector<double>({1, 2, 3, 6, 7}),
      makeConstant<double>(1.0, 5),
  });
  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {},
              {"differential_entropy(2, c0, c1, 'fixed_histogram_jacknife', 0.0, 10.0)"})
          .planNode();
  auto expected = makeRowVector({makeFlatVector<double>({3.478636068026093})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(DifferentialEntropyAggregationTest, fixedHistogramJacknifeMixedWeights) {
  // bucket [0, 5): value 1 & 2 with weight 2.0, value 3 with weight 1.0
  //   -> breakdown {2.0: 2, 1.0: 1}.
  // bucket [5, 10]: value 6 with weight 3.0, value 7 with weight 1.0
  //   -> breakdown {3.0: 1, 1.0: 1}.
  // Exercises grouping distinct (bucket, weight-value) pairs in the jacknife
  // breakdown histogram, and the corresponding MLE aggregate-weight sums.
  auto data = makeRowVector({
      makeFlatVector<double>({1, 2, 3, 6, 7}),
      makeFlatVector<double>({2.0, 2.0, 1.0, 3.0, 1.0}),
  });

  auto mlePlan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {},
              {"differential_entropy(2, c0, c1, 'fixed_histogram_mle', 0.0, 10.0)"})
          .planNode();
  AssertQueryBuilder(mlePlan).assertResults(
      makeRowVector({makeFlatVector<double>({3.313004154725584})}));

  auto jacknifePlan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {},
              {"differential_entropy(2, c0, c1, 'fixed_histogram_jacknife', 0.0, 10.0)"})
          .planNode();
  AssertQueryBuilder(jacknifePlan)
      .assertResults(
          makeRowVector({makeFlatVector<double>({3.617378236765016})}));
}

TEST_F(
    DifferentialEntropyAggregationTest,
    fixedHistogramMixedWeightsAcrossBatches) {
  // Same data as fixedHistogramJacknifeMixedWeights, split across batches so
  // the per-bucket (weight, count) breakdown goes through serialize(),
  // deserialize() and mergeWith() in partial+final plans.
  static const double kValues[] = {1, 2, 3, 6, 7};
  static const double kWeights[] = {2.0, 2.0, 1.0, 3.0, 1.0};
  auto batches = makeBatches(
      5,
      3,
      [](vector_size_t i) { return kValues[i]; },
      [](vector_size_t i) { return kWeights[i]; });

  testAggregations(
      batches,
      {},
      {"differential_entropy(2, c0, c1, 'fixed_histogram_mle', 0.0, 10.0)"},
      {makeRowVector({makeFlatVector<double>({3.313004154725584})})});
  testAggregations(
      batches,
      {},
      {"differential_entropy(2, c0, c1, 'fixed_histogram_jacknife', 0.0, 10.0)"},
      {makeRowVector({makeFlatVector<double>({3.617378236765016})})});
}

TEST_F(DifferentialEntropyAggregationTest, emptyInputReturnsNaN) {
  // Presto returns NaN, not NULL, when a group receives no non-NULL input.
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector({
      makeFlatVector<double>({1, 2, 3}),
      makeConstant<double>(1.0, 3),
  });
  for (
      const auto& aggregate : {
          std::string("differential_entropy(10, c0)"),
          std::string("differential_entropy(10, c0, c1)"),
          std::string(
              "differential_entropy(2, c0, c1, 'fixed_histogram_mle', 0.0, 10.0)"),
          std::string(
              "differential_entropy(2, c0, c1, 'fixed_histogram_jacknife', 0.0, 10.0)"),
      }) {
    SCOPED_TRACE(aggregate);
    auto plan = PlanBuilder()
                    .values({data})
                    .filter("c0 > 100.0")
                    .partialAggregation({}, {aggregate})
                    .finalAggregation()
                    .planNode();
    AssertQueryBuilder(plan).assertResults(
        makeRowVector({makeFlatVector<double>({nan})}));
  }
}

TEST_F(DifferentialEntropyAggregationTest, nullInputReturnsNaN) {
  // Rows with a NULL argument are skipped. Group 0 has only NULL samples and
  // must produce NaN; group 1 must match its fully non-NULL rows alone (the
  // row with value 8 and a NULL weight would otherwise shift the histogram).
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector({
      makeFlatVector<int32_t>({0, 0, 1, 1, 1, 1, 1, 1, 1}),
      makeNullableFlatVector<double>(
          {std::nullopt, std::nullopt, 1, 2, 3, 6, 7, std::nullopt, 8}),
      makeNullableFlatVector<double>(
          {1.0, std::nullopt, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, std::nullopt}),
  });

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1}),
      makeFlatVector<double>({nan, 3.2928786893420314}),
  });
  testAggregations(
      {data},
      {"c0"},
      {"differential_entropy(2, c1, c2, 'fixed_histogram_mle', 0.0, 10.0)"},
      {expected});

  expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1}),
      makeFlatVector<double>({nan, 3.478636068026093}),
  });
  testAggregations(
      {data},
      {"c0"},
      {"differential_entropy(2, c1, c2, 'fixed_histogram_jacknife', 0.0, 10.0)"},
      {expected});

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation(
                      {"c0"},
                      {"differential_entropy(10, c1)",
                       "differential_entropy(10, c1, c2)"})
                  .finalAggregation()
                  .orderBy({"c0"}, false)
                  .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(result->size(), 2);
  for (auto column = 1; column <= 2; ++column) {
    auto values = result->childAt(column)->as<SimpleVector<double>>();
    ASSERT_FALSE(values->isNullAt(0));
    EXPECT_TRUE(std::isnan(values->valueAt(0)));
    ASSERT_FALSE(values->isNullAt(1));
    EXPECT_TRUE(std::isfinite(values->valueAt(1)));
  }
}

TEST_F(DifferentialEntropyAggregationTest, fixedHistogramInvalidMethod) {
  auto data = makeRowVector({
      makeFlatVector<double>({1, 2, 3}),
      makeConstant<double>(1.0, 3),
  });
  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {},
              {"differential_entropy(2, c0, c1, 'not_a_method', 0.0, 10.0)"})
          .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()), "invalid method");
}

TEST_F(DifferentialEntropyAggregationTest, fixedHistogramOutOfRange) {
  auto data = makeRowVector({
      makeFlatVector<double>({1, 2, 11}), // 11 is out of [0, 10] range.
      makeConstant<double>(1.0, 3),
  });
  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {},
              {"differential_entropy(2, c0, c1, 'fixed_histogram_mle', 0.0, 10.0)"})
          .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "sample must be at most max");
}

// ---------------------------------------------------------------------------
// Reservoir sampling variants: randomized, so we check the estimate lands
// close to the known differential entropy of a uniform distribution
// (log2(range)) rather than asserting an exact value.
// ---------------------------------------------------------------------------

TEST_F(DifferentialEntropyAggregationTest, unweightedReservoirUniform) {
  constexpr vector_size_t kCount = 5000;
  constexpr int32_t kReservoirSize = 1000;
  std::default_random_engine gen(42);
  std::uniform_real_distribution<double> dist(0.0, 10.0);
  std::vector<double> values(kCount);
  for (auto& v : values) {
    v = dist(gen);
  }

  auto data = makeRowVector({makeFlatVector<double>(values)});
  auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {}, {fmt::format("differential_entropy({}, c0)", kReservoirSize)})
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  const auto entropy =
      result->as<RowVector>()->childAt(0)->as<FlatVector<double>>()->valueAt(0);
  // Uniform(0, 10) has differential entropy log2(10) ~= 3.32.
  EXPECT_NEAR(entropy, std::log2(10.0), 0.4);
}

TEST_F(DifferentialEntropyAggregationTest, unweightedReservoirAcrossBatches) {
  // Verifies mergeWith() across multiple partial groups doesn't crash and
  // produces a result in the same ballpark as the single-batch case, since
  // the merge logic (not just addInput()) determines correctness here.
  constexpr vector_size_t kCount = 5000;
  constexpr int32_t kReservoirSize = 1000;
  std::default_random_engine gen(43);
  std::uniform_real_distribution<double> dist(0.0, 10.0);

  auto batches = makeBatches(
      kCount,
      10,
      [&](vector_size_t) { return dist(gen); },
      [](vector_size_t) { return 1.0; });
  // Drop the weight column; this variant only takes (size, x).
  std::vector<RowVectorPtr> valueOnlyBatches;
  for (const auto& batch : batches) {
    valueOnlyBatches.push_back(
        makeRowVector({batch->as<RowVector>()->childAt(0)}));
  }

  auto plan =
      PlanBuilder()
          .values(valueOnlyBatches)
          .partialAggregation(
              {}, {fmt::format("differential_entropy({}, c0)", kReservoirSize)})
          .finalAggregation()
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  const auto entropy =
      result->as<RowVector>()->childAt(0)->as<FlatVector<double>>()->valueAt(0);
  EXPECT_NEAR(entropy, std::log2(10.0), 0.4);
}

TEST_F(DifferentialEntropyAggregationTest, weightedReservoirUniform) {
  constexpr vector_size_t kCount = 5000;
  constexpr int32_t kReservoirSize = 1000;
  std::default_random_engine gen(44);
  std::uniform_real_distribution<double> dist(0.0, 10.0);

  auto batches = makeBatches(
      kCount,
      1,
      [&](vector_size_t) { return dist(gen); },
      [](vector_size_t) { return 1.0; });

  auto plan =
      PlanBuilder()
          .values(batches)
          .singleAggregation(
              {},
              {fmt::format("differential_entropy({}, c0, c1)", kReservoirSize)})
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  const auto entropy =
      result->as<RowVector>()->childAt(0)->as<FlatVector<double>>()->valueAt(0);
  // Uniform weight=1 should behave like the unweighted variant.
  EXPECT_NEAR(entropy, std::log2(10.0), 0.4);
}

TEST_F(DifferentialEntropyAggregationTest, weightedReservoirNonUniformWeights) {
  // Sanity check only (no known closed-form entropy for this weighting):
  // verifies non-uniform weights are accepted, addInput()/combine() don't
  // crash across multiple batches, and the result is a finite, sane double.
  constexpr vector_size_t kCount = 3000;
  std::default_random_engine gen(45);
  std::uniform_real_distribution<double> valueDist(0.0, 10.0);
  std::uniform_real_distribution<double> weightDist(0.1, 5.0);

  auto batches = makeBatches(
      kCount,
      5,
      [&](vector_size_t) { return valueDist(gen); },
      [&](vector_size_t) { return weightDist(gen); });

  auto plan = PlanBuilder()
                  .values(batches)
                  .partialAggregation({}, {"differential_entropy(500, c0, c1)"})
                  .finalAggregation()
                  .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  const auto entropy =
      result->as<RowVector>()->childAt(0)->as<FlatVector<double>>()->valueAt(0);
  EXPECT_TRUE(std::isfinite(entropy));
}

} // namespace
} // namespace facebook::velox::aggregate::test
