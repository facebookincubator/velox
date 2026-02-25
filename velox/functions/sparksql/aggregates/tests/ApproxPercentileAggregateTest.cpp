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
#include <iomanip>

#include "velox/common/base/RandomUtil.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

constexpr int32_t kSparkDefaultAccuracy = 10000;

std::vector<TypePtr>
getArgTypes(const TypePtr& dataType, int32_t accuracy, int percentileCount) {
  std::vector<TypePtr> argTypes;
  argTypes.push_back(dataType);
  if (percentileCount == -1) {
    argTypes.push_back(DOUBLE());
  } else {
    argTypes.push_back(ARRAY(DOUBLE()));
  }
  // Always pass accuracy parameter to ensure consistency across aggregation
  // stages.
  if (accuracy >= 0) {
    argTypes.push_back(BIGINT());
  }
  return argTypes;
}

// Build a function call string for spark_approx_percentile.
std::string functionCall(
    bool keyed,
    double percentile,
    int32_t accuracy,
    int percentileCount) {
  std::ostringstream buf;
  // Use fixed notation with enough precision to ensure double values
  // are correctly parsed (e.g., 0.5 instead of 5e-1)
  buf << std::fixed << std::setprecision(6);
  int columnIndex = keyed ? 1 : 0;
  buf << "spark_approx_percentile(c" << columnIndex++;
  buf << ", ";
  if (percentileCount == -1) {
    buf << percentile;
  } else {
    buf << "ARRAY[";
    for (int i = 0; i < percentileCount; ++i) {
      buf << (i == 0 ? "" : ",") << percentile;
    }
    buf << ']';
  }
  // Always pass accuracy parameter to ensure consistency across aggregation
  // stages. Use default value (10000) when accuracy is 0.
  if (accuracy >= 0) {
    buf << ", " << (accuracy == 0 ? kSparkDefaultAccuracy : accuracy);
  }
  buf << ')';
  return buf.str();
}

class ApproxPercentileAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    // Register Spark aggregate functions with "spark_" prefix.
    registerAggregateFunctions("spark_");
    random::setSeed(0);
    // Set fixed random seed for reproducible test results.
    queryConfig_
        [core::QueryConfig::kDebugAggregationApproxPercentileFixedRandomSeed] =
            "0";
  }

  // Test global aggregation (no grouping keys).
  template <typename T>
  void testGlobalAgg(
      const VectorPtr& values,
      double percentile,
      int32_t accuracy,
      std::optional<T> expectedResult) {
    SCOPED_TRACE(
        fmt::format("percentile={} accuracy={}", percentile, accuracy));
    auto rows = makeRowVector({values});

    std::string expected;
    std::string expectedArray;
    if (!expectedResult.has_value()) {
      expected = "SELECT NULL";
      expectedArray = "SELECT NULL";
    } else if (
        (std::is_same<T, float>::value && std::isnan(expectedResult.value())) ||
        (std::is_same<T, double>::value &&
         std::isnan(expectedResult.value()))) {
      expected = "SELECT 'NaN'";
      expectedArray = fmt::format("SELECT ARRAY[{0},{0},{0}]", "'NaN'");
    } else if (
        (std::is_same<T, float>::value &&
         expectedResult.value() == std::numeric_limits<float>::infinity()) ||
        (std::is_same<T, double>::value &&
         expectedResult.value() == std::numeric_limits<double>::infinity())) {
      expected = "SELECT 'INFINITY'";
      expectedArray = fmt::format("SELECT ARRAY[{0},{0},{0}]", "'INFINITY'");
    } else if (
        (std::is_same<T, float>::value &&
         expectedResult.value() == -std::numeric_limits<float>::infinity()) ||
        (std::is_same<T, double>::value &&
         expectedResult.value() == -std::numeric_limits<double>::infinity())) {
      expected = "SELECT '-INFINITY'";
      expectedArray = fmt::format("SELECT ARRAY[{0},{0},{0}]", "'-INFINITY'");
    } else {
      expected = fmt::format("SELECT {}", expectedResult.value());
      expectedArray =
          fmt::format("SELECT ARRAY[{0},{0},{0}]", expectedResult.value());
    }

    // Test single percentile
    enableTestStreaming();
    testAggregations(
        {rows},
        {},
        {functionCall(false, percentile, accuracy, -1)},
        expected,
        queryConfig_);

    // Test percentile array (3 identical percentiles)
    testAggregations(
        {rows},
        {},
        {functionCall(false, percentile, accuracy, 3)},
        expectedArray,
        queryConfig_);

    // Companion functions of approx_percentile do not support test streaming
    // because intermediate results are KLL that has non-deterministic shape.
    disableTestStreaming();
    testAggregationsWithCompanion(
        {rows},
        [](PlanBuilder& /*builder*/) {},
        {},
        {functionCall(false, percentile, accuracy, -1)},
        {getArgTypes(values->type(), accuracy, -1)},
        {},
        expected,
        queryConfig_);
  }

  // Test global aggregation with NaN and Infinity values for floating point.
  template <typename T>
  void testNaN() {
    const T nan = std::is_same_v<T, float> ? std::nanf("") : std::nan("");
    vector_size_t size = 10;
    auto values = makeFlatVector<T>(size, [nan](auto row) {
      if (row > 8)
        return -std::numeric_limits<T>::infinity();
      else if (row > 7)
        return std::numeric_limits<T>::infinity();
      else if (row > 6)
        return nan;
      else
        return (T)row;
    });

    // Test various percentiles with special floating point values
    testGlobalAgg<T>(values, 0.05, 10000, -std::numeric_limits<T>::infinity());
    testGlobalAgg<T>(values, 0.11, 10000, 0.0);
    testGlobalAgg<T>(values, 0.55, 10000, 4.0);
    testGlobalAgg<T>(values, 0.85, 10000, std::numeric_limits<T>::infinity());
    testGlobalAgg<T>(values, 0.95, 10000, nan);
  }

  // Test group-by aggregation.
  void testGroupByAgg(
      const VectorPtr& keys,
      const VectorPtr& values,
      double percentile,
      int32_t accuracy,
      const RowVectorPtr& expectedResult) {
    auto rows = makeRowVector({keys, values});

    enableTestStreaming();
    testAggregations(
        {rows},
        {"c0"},
        {functionCall(true, percentile, accuracy, -1)},
        {expectedResult},
        queryConfig_);

    // Companion functions of approx_percentile do not support test streaming
    // because intermediate results are KLL that has non-deterministic shape.
    disableTestStreaming();
    testAggregationsWithCompanion(
        {rows},
        [](PlanBuilder& /*builder*/) {},
        {"c0"},
        {functionCall(true, percentile, accuracy, -1)},
        {getArgTypes(values->type(), accuracy, -1)},
        {},
        {expectedResult},
        queryConfig_);

    // Test percentile array
    {
      SCOPED_TRACE("Percentile array");
      auto resultValues = expectedResult->childAt(1);
      RowVectorPtr expected = nullptr;
      auto size = resultValues->size();
      if (resultValues->nulls() &&
          bits::countNonNulls(resultValues->rawNulls(), 0, size) == 0) {
        expected = makeRowVector(
            {expectedResult->childAt(0),
             BaseVector::createNullConstant(
                 ARRAY(resultValues->type()), size, pool())});
      } else {
        auto elements = BaseVector::create(
            resultValues->type(), 3 * resultValues->size(), pool());
        auto offsets = allocateOffsets(resultValues->size(), pool());
        auto rawOffsets = offsets->asMutable<vector_size_t>();
        auto sizes = allocateSizes(resultValues->size(), pool());
        auto rawSizes = sizes->asMutable<vector_size_t>();
        for (int i = 0; i < resultValues->size(); ++i) {
          rawOffsets[i] = 3 * i;
          rawSizes[i] = 3;
          elements->copy(resultValues.get(), 3 * i + 0, i, 1);
          elements->copy(resultValues.get(), 3 * i + 1, i, 1);
          elements->copy(resultValues.get(), 3 * i + 2, i, 1);
        }
        expected = makeRowVector(
            {expectedResult->childAt(0),
             std::make_shared<ArrayVector>(
                 pool(),
                 ARRAY(elements->type()),
                 nullptr,
                 resultValues->size(),
                 offsets,
                 sizes,
                 elements)});
      }

      enableTestStreaming();
      testAggregations(
          {rows},
          {"c0"},
          {functionCall(true, percentile, accuracy, 3)},
          {expected},
          queryConfig_);
    }
  }

 private:
  std::unordered_map<std::string, std::string> queryConfig_;
};

// Test global aggregation with basic integer values.
TEST_F(ApproxPercentileAggregateTest, globalAgg) {
  vector_size_t size = 1'000;
  auto values =
      makeFlatVector<int32_t>(size, [](auto row) { return row % 23; });

  // Test with default accuracy
  testGlobalAgg<int32_t>(values, 0.5, 0, 11);
  // Test with explicit accuracy (10000 is Spark default)
  testGlobalAgg<int32_t>(values, 0.5, 10000, 11);
  // Test with higher accuracy
  testGlobalAgg<int32_t>(values, 0.5, 20000, 11);

  // Test with null values
  auto valuesWithNulls = makeFlatVector<int32_t>(
      size, [](auto row) { return row % 23; }, nullEvery(7));

  testGlobalAgg<int32_t>(valuesWithNulls, 0.5, 0, 11);
  testGlobalAgg<int32_t>(valuesWithNulls, 0.5, 10000, 11);
}

// Test group-by aggregation with integer values.
TEST_F(ApproxPercentileAggregateTest, groupByAgg) {
  vector_size_t size = 1'000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 7; });
  auto values = makeFlatVector<int32_t>(
      size, [](auto row) { return (row / 7) % 23 + row % 7; });

  // Test with default accuracy
  auto expectedResult = makeRowVector(
      {makeFlatVector(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}),
       makeFlatVector(std::vector<int32_t>{11, 12, 13, 14, 15, 16, 17})});
  testGroupByAgg(keys, values, 0.5, 0, expectedResult);
  testGroupByAgg(keys, values, 0.5, 10000, expectedResult);

  // Test with null values
  auto valuesWithNulls = makeFlatVector<int32_t>(
      size, [](auto row) { return (row / 7) % 23 + row % 7; }, nullEvery(11));

  expectedResult = makeRowVector(
      {makeFlatVector(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}),
       makeFlatVector(std::vector<int32_t>{11, 12, 13, 14, 15, 16, 17})});
  testGroupByAgg(keys, valuesWithNulls, 0.5, 0, expectedResult);
  testGroupByAgg(keys, valuesWithNulls, 0.5, 10000, expectedResult);
}

// Test Partial + Final aggregation mode (simulating distributed execution).
TEST_F(ApproxPercentileAggregateTest, partialFinal) {
  // Make sure partial aggregation runs out of memory after first batch.
  CursorParameters params;
  params.queryCtx = velox::core::QueryCtx::create(executor_.get());
  params.queryCtx->testingOverrideConfigUnsafe({
      {core::QueryConfig::kMaxPartialAggregationMemory, "300000"},
  });

  auto data = {
      makeRowVector({
          makeFlatVector<int32_t>(1'024, [](auto row) { return row % 117; }),
          makeFlatVector<int32_t>(1'024, [](auto /*row*/) { return 10; }),
      }),
      makeRowVector({
          makeFlatVector<int32_t>(1'024, [](auto row) { return row % 5; }),
          makeFlatVector<int32_t>(1'024, [](auto /*row*/) { return 15; }),
      }),
      makeRowVector({
          makeFlatVector<int32_t>(1'024, [](auto row) { return row % 7; }),
          makeFlatVector<int32_t>(1'024, [](auto /*row*/) { return 20; }),
      }),
  };

  // Build partial + final aggregation plan
  params.planNode =
      PlanBuilder()
          .values(data)
          .project({"c0", "c1", "0.9995"})
          .partialAggregation({"c0"}, {"spark_approx_percentile(c1, p2)"})
          .finalAggregation()
          .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int32_t>(117, [](auto row) { return row; }),
      makeFlatVector<int32_t>(117, [](auto row) { return row < 7 ? 20 : 10; }),
  });
  exec::test::assertQuery(params, {expected});
  waitForAllTasksToBeDeleted();
}

// Test Partial + Final group-by aggregation.
TEST_F(ApproxPercentileAggregateTest, partialFinalGroupBy) {
  // Build test data with two groups
  auto groupKeys = makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2});
  auto values = makeFlatVector<int32_t>({3, 6, 9, 7, 10, 8});
  auto inputRows = makeRowVector({groupKeys, values});

  // Build Partial + Final plan with group-by
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values({inputRows})
          .partialAggregation({"c0"}, {"spark_approx_percentile(c1, 0.5)"})
          .finalAggregation()
          .planNode();

  // Expected: group 1 -> 6, group 2 -> 8
  auto expectedResult = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}), makeFlatVector<int32_t>({6, 8})});

  AssertQueryBuilder(plan).assertResults(expectedResult);
}

// Test final aggregation accuracy by merging multiple partial results.
TEST_F(ApproxPercentileAggregateTest, finalAggregateAccuracy) {
  auto batch = makeRowVector(
      {makeFlatVector<int32_t>(1000, [](auto row) { return row; })});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<std::shared_ptr<const core::PlanNode>> sources;
  // Create 10 partial aggregation sources
  for (int i = 0; i < 10; ++i) {
    sources.push_back(PlanBuilder(planNodeIdGenerator)
                          .values({batch})
                          .partialAggregation(
                              {}, {"spark_approx_percentile(c0, 0.005, 10000)"})
                          .planNode());
  }
  // Merge all partial results in final aggregation
  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartitionRoundRobin(sources)
                .finalAggregation()
                .planNode();
  assertQuery(op, "SELECT 5");
}

// Test with non-flat (dictionary wrapped) percentile array.
TEST_F(ApproxPercentileAggregateTest, nonFlatPercentileArray) {
  auto indices = AlignedBuffer::allocate<vector_size_t>(3, pool());
  auto rawIndices = indices->asMutable<vector_size_t>();
  std::iota(rawIndices, rawIndices + indices->size(), 0);
  auto percentiles = std::make_shared<ArrayVector>(
      pool(),
      ARRAY(DOUBLE()),
      nullptr,
      1,
      AlignedBuffer::allocate<vector_size_t>(1, pool(), 0),
      AlignedBuffer::allocate<vector_size_t>(1, pool(), 3),
      BaseVector::wrapInDictionary(
          nullptr, indices, 3, makeFlatVector<double>({0, 0.5, 1})));
  auto rows = makeRowVector({
      makeFlatVector<int32_t>(10, folly::identity),
      BaseVector::wrapInConstant(1, 0, percentiles),
  });
  auto plan = PlanBuilder()
                  .values({rows})
                  .singleAggregation({}, {"spark_approx_percentile(c0, c1)"})
                  .planNode();
  auto expected = makeRowVector({makeArrayVector<int32_t>({{0, 5, 9}})});
  AssertQueryBuilder(plan).assertResults(expected);
}

// Test when all input values are NULL.
TEST_F(ApproxPercentileAggregateTest, noInput) {
  const int size = 1000;
  auto keys = makeFlatVector<int32_t>(size, [](auto row) { return row % 7; });
  auto values = makeFlatVector<int32_t>(size, [](auto row) { return row % 6; });
  auto nullValues = makeNullConstant(TypeKind::INTEGER, size);

  // Test global aggregation with all nulls
  testGlobalAgg<int32_t>(nullValues, 0.5, 0, std::nullopt);

  // Test group-by aggregation with all nulls
  {
    auto expected = makeRowVector(
        {makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6}),
         makeNullConstant(TypeKind::INTEGER, 7)});

    testGroupByAgg(keys, nullValues, 0.5, 0, expected);
  }

  // Test when all inputs are masked out
  {
    auto testWithMask = [&](bool groupBy, const RowVectorPtr& expected) {
      std::vector<std::string> groupingKeys;
      if (groupBy) {
        groupingKeys.push_back("c0");
      }
      auto plan =
          PlanBuilder()
              .values({makeRowVector({keys, values})})
              .project(
                  {"c0", "c1", "array_constructor(0.5) as pct", "c1 > 6 as m1"})
              .singleAggregation(
                  groupingKeys,
                  {"spark_approx_percentile(c1, 0.5)",
                   "spark_approx_percentile(c1, 0.5, 10000)",
                   "spark_approx_percentile(c1, pct)",
                   "spark_approx_percentile(c1, pct, 10000)"},
                  {"m1", "m1", "m1", "m1"})
              .planNode();

      AssertQueryBuilder(plan).assertResults(expected);
    };

    // Global aggregation with mask
    std::vector<VectorPtr> children{4};
    std::fill_n(children.begin(), 2, makeNullConstant(TypeKind::INTEGER, 1));
    std::fill_n(
        children.begin() + 2,
        2,
        BaseVector::createNullConstant(ARRAY(INTEGER()), 1, pool()));
    auto expected1 = makeRowVector(children);
    testWithMask(false, expected1);

    // Group-by aggregation with mask
    children.resize(5);
    children[0] = makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6});
    std::fill_n(
        children.begin() + 1, 2, makeNullConstant(TypeKind::INTEGER, 7));
    std::fill_n(
        children.begin() + 3,
        2,
        BaseVector::createNullConstant(ARRAY(INTEGER()), 7, pool()));
    auto expected2 = makeRowVector(children);
    testWithMask(true, expected2);
  }
}

// Test that null percentile parameter throws an error.
TEST_F(ApproxPercentileAggregateTest, nullPercentile) {
  auto values = makeFlatVector<int32_t>({1, 2, 3, 4});
  auto percentileOfDouble = makeConstant<double>(std::nullopt, 4);
  auto rows = makeRowVector({values, percentileOfDouble});

  // Test null percentile for spark_approx_percentile(value, percentile)
  VELOX_ASSERT_THROW(
      testAggregations(
          {rows}, {}, {"spark_approx_percentile(c0, c1)"}, "SELECT NULL"),
      "Percentile cannot be null");

  // Test null array percentile (entire array is null, not array with null
  // elements) Note: Spark does not allow null elements within the percentile
  // array. The percentile array itself can be null, which should throw an
  // error.
  auto percentileOfArrayOfDouble =
      BaseVector::createNullConstant(ARRAY(DOUBLE()), 4, pool());
  rows = makeRowVector({values, percentileOfArrayOfDouble});

  // Test null array percentile for spark_approx_percentile(value, percentiles)
  // When the entire percentile array is null, it triggers the null check
  // in ArrayVector and throws "Percentile array cannot contain nulls".
  VELOX_ASSERT_THROW(
      testAggregations(
          {rows}, {}, {"spark_approx_percentile(c0, c1)"}, "SELECT NULL"),
      "Percentile array cannot contain nulls");
}

// Test with NaN values in floating point inputs.
TEST_F(ApproxPercentileAggregateTest, nanPercentile) {
  testNaN<float>();
  testNaN<double>();
}

// Test with various numeric input types supported by Spark.
// KLL sketch uses std::upper_bound to estimate percentile, so for 10 elements
// {1,2,3,4,5,6,7,8,9,10}, the 50th percentile (weight=5) returns 6.
TEST_F(ApproxPercentileAggregateTest, numericTypes) {
  // Test TINYINT
  {
    auto values = makeFlatVector<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    testGlobalAgg<int8_t>(values, 0.5, 0, 6);
  }

  // Test SMALLINT
  {
    auto values = makeFlatVector<int16_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    testGlobalAgg<int16_t>(values, 0.5, 0, 6);
  }

  // Test BIGINT
  {
    auto values = makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    testGlobalAgg<int64_t>(values, 0.5, 0, 6);
  }

  // Test REAL (float)
  {
    auto values = makeFlatVector<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    testGlobalAgg<float>(values, 0.5, 0, 6.0f);
  }

  // Test DOUBLE
  {
    auto values = makeFlatVector<double>(
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    testGlobalAgg<double>(values, 0.5, 0, 6.0);
  }
}

// Test edge cases for percentile parameter values.
TEST_F(ApproxPercentileAggregateTest, percentileEdgeCases) {
  auto values = makeFlatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // Test percentile = 0.0 (minimum)
  testGlobalAgg<int32_t>(values, 0.0, 0, 1);

  // Test percentile = 1.0 (maximum)
  testGlobalAgg<int32_t>(values, 1.0, 0, 10);

  // Test percentile = 0.25 (Q1)
  testGlobalAgg<int32_t>(values, 0.25, 0, 3);

  // Test percentile = 0.75 (Q3)
  testGlobalAgg<int32_t>(values, 0.75, 0, 8);
}

// Test accuracy parameter boundary values.
TEST_F(ApproxPercentileAggregateTest, accuracyBoundary) {
  auto values = makeFlatVector<int32_t>(1000, [](auto row) { return row; });
  auto rows = makeRowVector({values});

  // Test with minimum valid accuracy - allow 480-520 range due to low accuracy
  {
    auto plan =
        PlanBuilder()
            .values({rows})
            .singleAggregation({}, {"spark_approx_percentile(c0, 0.5, 1)"})
            .planNode();
    auto result = AssertQueryBuilder(plan).copyResults(pool());
    auto resultValue =
        result->childAt(0)->as<SimpleVector<int32_t>>()->valueAt(0);
    EXPECT_GE(resultValue, 480)
        << "Result " << resultValue << " is below minimum expected 480";
    EXPECT_LE(resultValue, 520)
        << "Result " << resultValue << " is above maximum expected 520";
  }

  // Test with high accuracy - should be more precise
  testGlobalAgg<int32_t>(values, 0.5, 100000, 500);
}

// Test percentile array with multiple distinct percentiles.
TEST_F(ApproxPercentileAggregateTest, multiplePercentiles) {
  auto values = makeFlatVector<int32_t>(100, [](auto row) { return row; });
  auto percentiles = makeArrayVector<double>({{0.1, 0.5, 0.9}});
  auto rows = makeRowVector({
      values,
      BaseVector::wrapInConstant(100, 0, percentiles),
  });

  // Test with array of different percentiles [0.1, 0.5, 0.9]
  auto plan = PlanBuilder()
                  .values({rows})
                  .singleAggregation({}, {"spark_approx_percentile(c0, c1)"})
                  .planNode();

  // Expected: approximately [10, 50, 90]
  auto expected = makeRowVector({makeArrayVector<int32_t>({{10, 50, 90}})});
  AssertQueryBuilder(plan).assertResults(expected);
}

// Test that invalid accuracy throws an error.
TEST_F(ApproxPercentileAggregateTest, invalidAccuracy) {
  auto values = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto rows = makeRowVector({values});

  // Accuracy = 0 is invalid
  VELOX_ASSERT_THROW(
      testAggregations(
          {rows}, {}, {"spark_approx_percentile(c0, 0.5, 0)"}, "SELECT 3"),
      "Accuracy must be greater than 0");
}

// Test single row input.
TEST_F(ApproxPercentileAggregateTest, singleRow) {
  auto values = makeFlatVector<int32_t>(1, [](auto /*row*/) { return 42; });
  testGlobalAgg<int32_t>(values, 0.5, 0, 42);
  testGlobalAgg<int32_t>(values, 0.0, 0, 42);
  testGlobalAgg<int32_t>(values, 1.0, 0, 42);
}

// Test large dataset for stress testing.
TEST_F(ApproxPercentileAggregateTest, largeDataset) {
  vector_size_t size = 100'000;
  auto values = makeFlatVector<int32_t>(size, [](auto row) { return row; });
  auto rows = makeRowVector({values});

  // Median of 0..99999 should be approximately 50000, allow ±5 error
  auto plan =
      PlanBuilder()
          .values({rows})
          .singleAggregation({}, {"spark_approx_percentile(c0, 0.5, 10000)"})
          .planNode();
  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto resultValue =
      result->childAt(0)->as<SimpleVector<int32_t>>()->valueAt(0);
  EXPECT_GE(resultValue, 49995)
      << "Result " << resultValue << " is below minimum expected 49995";
  EXPECT_LE(resultValue, 50005)
      << "Result " << resultValue << " is above maximum expected 50005";
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
