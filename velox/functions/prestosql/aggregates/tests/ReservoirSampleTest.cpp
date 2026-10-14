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

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#include "velox/common/base/RandomUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::prestosql {

namespace {

class ReservoirSampleTest : public AggregationTestBase {
 protected:
  // Fixes the random seed, so tests that sample distinct values get the same
  // result on every run.
  void SetUp() override {
    AggregationTestBase::SetUp();
    random::setSeed(0);
  }

  RowVectorPtr makeResultDouble(
      int64_t seenCount,
      const std::vector<std::optional<double>>& sample) {
    std::vector<std::optional<std::vector<std::optional<double>>>> data = {
        sample};
    return makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({seenCount}),
        makeNullableArrayVector<double>(data),
    })});
  }

  VectorPtr makeNullArray(int count, const TypePtr& elementType) {
    auto arrayType = ARRAY(elementType);
    auto builder = BaseVector::create(arrayType, count, pool());
    for (int i = 0; i < count; i++) {
      builder->setNull(i, true);
    }
    return builder;
  }

  template <typename T>
  VectorPtr makeFlatConstantVector(T value, int count) {
    return makeFlatVector<T>(std::vector<T>(count, value));
  }

  // Verifies that each trial sampled 2 of the values 0 to 'numValues' - 1, and
  // that every pair of values came back from about the same number of trials.
  // 'result' holds one row per trial: the trial number, then the output of
  // reservoir_sample.
  void assertUniformPairs(const RowVectorPtr& result, int32_t numValues) {
    const auto* output = result->childAt(1)->asChecked<RowVector>();
    const auto* processedCounts =
        output->childAt(0)->asChecked<FlatVector<int64_t>>();
    const auto* samples = output->childAt(1)->asChecked<ArrayVector>();
    const auto* values = samples->elements()->asChecked<FlatVector<int64_t>>();

    std::map<std::pair<int64_t, int64_t>, int32_t> pairCounts;
    for (vector_size_t row = 0; row < result->size(); ++row) {
      ASSERT_EQ(processedCounts->valueAt(row), numValues);
      ASSERT_EQ(samples->sizeAt(row), 2);
      const auto first = values->valueAt(samples->offsetAt(row));
      const auto second = values->valueAt(samples->offsetAt(row) + 1);
      ++pairCounts[{std::min(first, second), std::max(first, second)}];
    }

    // For a uniform sampler, each pair's count follows a binomial
    // distribution. The fixed seed repeats the same counts on every run. If a
    // code change alters the random sequence, a uniform sampler still stays
    // within 6 standard deviations in all but 1 in 10 million sequences.
    const int32_t numPairs = numValues * (numValues - 1) / 2;
    const double expected = static_cast<double>(result->size()) / numPairs;
    const double tolerance = 6 * std::sqrt(expected * (1 - 1.0 / numPairs));
    const auto minCount = static_cast<int32_t>(std::ceil(expected - tolerance));
    const auto maxCount =
        static_cast<int32_t>(std::floor(expected + tolerance));
    ASSERT_EQ(pairCounts.size(), static_cast<size_t>(numPairs));

    bool uniform = true;
    std::string counts;
    for (const auto& [pair, count] : pairCounts) {
      const bool inRange = count >= minCount && count <= maxCount;
      uniform &= inRange;
      counts += fmt::format(
          "\n  {{{}, {}}}: {}{}",
          pair.first,
          pair.second,
          count,
          inRange ? "" : "  <-- out of range");
    }
    EXPECT_TRUE(uniform) << fmt::format(
        "Each pair should come back {} to {} times out of {}:{}",
        minCount,
        maxCount,
        result->size(),
        counts);
  }
};

TEST_F(ReservoirSampleTest, testNoInitialSample) {
  auto data = makeRowVector({
      makeNullArray(5, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  auto expected = makeResultDouble(5L, {1.0, 1.0});

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, testLarge) {
  const int sampleSize = 5000;
  const int inputSize = 15000;

  std::vector<double> inputValues(inputSize, 1.0);
  std::vector<std::optional<double>> expectedSample(sampleSize, 1.0);

  auto data = makeRowVector({
      makeNullArray(inputSize, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, inputSize),
      makeFlatVector<double>(inputValues),
      makeFlatConstantVector<int32_t>(sampleSize, inputSize),
  });

  auto expected =
      makeResultDouble(static_cast<int64_t>(inputSize), expectedSample);

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, testInvalidSampleSizeZero) {
  auto data = makeRowVector({
      makeNullArray(5, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(0, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testInvalidSampleSizeNegative) {
  auto data = makeRowVector({
      makeNullArray(5, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(-1, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testInitialSampleSameSize) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0}}),
      makeFlatConstantVector<int64_t>(10, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  auto expected = makeResultDouble(15L, {1.0, 1.0});

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, testInitialSampleWrongSize) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0}}),
      makeFlatConstantVector<int64_t>(10, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testInitialSampleSmallerThanMaxSize) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0},
           std::vector<double>{1.0},
           std::vector<double>{1.0},
           std::vector<double>{1.0},
           std::vector<double>{1.0}}),
      makeFlatConstantVector<int64_t>(1, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  auto expected = makeResultDouble(6L, {1.0, 1.0});

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(
    ReservoirSampleTest,
    testInitialSampleSeenCountSmallerThanInitialSample) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0}}),
      makeFlatConstantVector<int64_t>(1, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testValidResults) {
  auto data = makeRowVector({
      makeNullArray(10, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 10),
      makeFlatVector<double>(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
      makeFlatConstantVector<int32_t>(4, 10),
  });

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation({}, {"reservoir_sample(c0, c1, c2, c3)"})
              .planNode())
          .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto rowResult = result->childAt(0)->as<RowVector>();
  ASSERT_EQ(rowResult->size(), 1);

  auto processedCount =
      rowResult->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  EXPECT_EQ(processedCount, 10L);

  auto sampleArray = rowResult->childAt(1)->as<ArrayVector>();
  ASSERT_EQ(sampleArray->size(), 1);

  auto sampleSize = sampleArray->sizeAt(0);
  EXPECT_EQ(sampleSize, 4);

  std::set<double> allValues = {
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  auto sampleElements = sampleArray->elements()->as<FlatVector<double>>();
  auto offset = sampleArray->offsetAt(0);

  for (int i = 0; i < sampleSize; i++) {
    double value = sampleElements->valueAt(offset + i);
    EXPECT_TRUE(allValues.count(value) > 0)
        << "Sample contains value " << value << " which is not in input set";
  }
}

// Each trial merges two partial results, as when combining results from
// different workers. The first partial result saw the values 0 and 1 and kept
// both. The second saw the values 2 to 5 and kept 2 of them. Trials take turns
// among the 6 pairs the second could have kept, so that side is exactly
// uniform. Merging should return each of the 15 pairs of 0 to 5 from about the
// same number of trials. The rows are:
//
//   trial | seenCount | sample
//   ------+-----------+-------
//       0 |         2 | [0, 1]
//       0 |         4 | [2, 3]
//       1 |         2 | [0, 1]
//       1 |         4 | [2, 4]
//     ...
//       5 |         2 | [0, 1]
//       5 |         4 | [4, 5]
//       6 |         2 | [0, 1]
//       6 |         4 | [2, 3]
//     ...
//
// Every row has no initial sample and a maximum sample size of 2.
TEST_F(ReservoirSampleTest, mergeIsUniform) {
  constexpr int32_t kNumTrials = 12'000;
  constexpr vector_size_t kSize = 2 * kNumTrials;
  const std::vector<std::vector<int64_t>> secondSamples = {
      {2, 3},
      {2, 4},
      {2, 5},
      {3, 4},
      {3, 5},
      {4, 5},
  };

  std::vector<int32_t> trials;
  std::vector<int64_t> seenCounts;
  std::vector<std::vector<int64_t>> samples;
  for (int32_t trial = 0; trial < kNumTrials; ++trial) {
    trials.push_back(trial);
    seenCounts.push_back(2);
    samples.push_back({0, 1});

    trials.push_back(trial);
    seenCounts.push_back(4);
    samples.push_back(secondSamples[trial % secondSamples.size()]);
  }

  const auto data = makeRowVector(
      {"trial", "partial"},
      {
          makeFlatVector<int32_t>(trials),
          makeRowVector(
              {"initialSample",
               "initialSeenCount",
               "seenCount",
               "maxSampleSize",
               "sample"},
              {
                  makeNullArray(kSize, BIGINT()),
                  makeFlatConstantVector<int64_t>(0, kSize),
                  makeFlatVector<int64_t>(seenCounts),
                  makeFlatConstantVector<int32_t>(2, kSize),
                  makeArrayVector<int64_t>(samples),
              }),
      });

  const auto plan = PlanBuilder()
                        .values({data})
                        .finalAggregation(
                            {"trial"},
                            {"reservoir_sample(partial)"},
                            {{ARRAY(BIGINT()), BIGINT(), BIGINT(), INTEGER()}})
                        .planNode();
  assertUniformPairs(AssertQueryBuilder(plan).copyResults(pool()), 6);
}

// Each trial runs this query, which merges the initial sample [0, 1], drawn
// from 2 values, with the values 2 and 3:
//
//   SELECT reservoir_sample(ARRAY[0, 1], 2, x, 2) FROM (VALUES 2, 3) AS t(x)
//
// It should return each of the 6 pairs of 0 to 3 from about the same number of
// trials. The rows are:
//
//   trial | x
//   ------+---
//       0 | 2
//       0 | 3
//       1 | 2
//       1 | 3
//     ...
//
// The other arguments are the same in every row, as in the query.
TEST_F(ReservoirSampleTest, mergeWithInitialSampleIsUniform) {
  constexpr int32_t kNumTrials = 12'000;
  constexpr vector_size_t kSize = 2 * kNumTrials;
  const auto data = makeRowVector(
      {"trial", "initial_sample", "initial_count", "x", "sample_size"},
      {
          makeFlatVector<int32_t>(kSize, [](auto row) { return row / 2; }),
          makeConstantArray<int64_t>(kSize, {0, 1}),
          makeFlatConstantVector<int64_t>(2, kSize),
          makeFlatVector<int64_t>(kSize, [](auto row) { return 2 + row % 2; }),
          makeFlatConstantVector<int32_t>(2, kSize),
      });

  const auto plan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {"trial"},
              {"reservoir_sample(initial_sample, initial_count, x, sample_size)"})
          .planNode();
  assertUniformPairs(AssertQueryBuilder(plan).copyResults(pool()), 4);
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql
