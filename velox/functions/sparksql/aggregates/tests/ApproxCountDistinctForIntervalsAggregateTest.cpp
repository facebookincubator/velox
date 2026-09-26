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
#include <vector>

#include <fmt/format.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

class ApproxCountDistinctForIntervalsAggregateTest
    : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
  }

  template <typename T>
  VectorPtr makeEndpointsVector(
      vector_size_t size,
      const std::vector<T>& endpoints,
      const TypePtr& elementType = CppToType<T>::create()) {
    auto arrayVector = makeArrayVector<T>({endpoints}, elementType);
    return BaseVector::wrapInConstant(size, 0, arrayVector);
  }

  std::vector<int64_t> runGlobalAggregation(
      const RowVectorPtr& data,
      const std::string& expression,
      bool usePartial) {
    auto builder = PlanBuilder().values({data});
    if (usePartial) {
      builder.partialAggregation({}, {expression}).finalAggregation();
    } else {
      builder.singleAggregation({}, {expression});
    }
    auto result = AssertQueryBuilder(builder.planNode()).copyResults(pool());
    auto rows = materialize(result);
    VELOX_CHECK_EQ(rows.size(), 1);
    VELOX_CHECK_EQ(rows[0].size(), 1);
    return rows[0][0].array<int64_t>();
  }

  void checkNdvs(
      const std::vector<int64_t>& ndvs,
      const std::vector<int64_t>& expected,
      double rsd) {
    ASSERT_EQ(ndvs.size(), expected.size());
    for (size_t i = 0; i < ndvs.size(); ++i) {
      const auto expectedNdv = expected[i];
      const auto ndv = ndvs[i];
      if (expectedNdv == 0) {
        EXPECT_EQ(ndv, 0);
        continue;
      }
      EXPECT_GT(ndv, 0);
      const double error =
          std::abs(ndv / static_cast<double>(expectedNdv) - 1.0);
      EXPECT_LE(error, rsd * 3.0) << "Index " << i;
    }
  }

  template <typename T>
  void testIntegerInputType(
      const std::vector<int32_t>& valuesData,
      const std::vector<int32_t>& endpointsData,
      const std::vector<int64_t>& expected) {
    std::vector<T> values(valuesData.begin(), valuesData.end());
    std::vector<T> endpoints(endpointsData.begin(), endpointsData.end());
    auto valuesVector = makeFlatVector<T>(values);
    auto endpointsVector =
        makeEndpointsVector<T>(valuesVector->size(), endpoints);
    auto data = makeRowVector({valuesVector, endpointsVector});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }
};

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, tooFewEndpoints) {
  auto values = makeFlatVector<int32_t>({1});
  auto endpoints = makeEndpointsVector<int32_t>(values->size(), {0});
  auto data = makeRowVector({values, endpoints});
  auto expected = makeRowVector({makeArrayVector<int64_t>({{0}})});

  VELOX_ASSERT_USER_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1, 0.05)"},
          {expected}),
      "approx_count_distinct_for_intervals requires at least 2 endpoints");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, endpointsNotSorted) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeEndpointsVector<double>(values->size(), {0.0, 2.0, 1.0});
  auto data = makeRowVector({values, endpoints});

  VELOX_ASSERT_USER_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1, 0.05)"},
          {makeRowVector({makeArrayVector<int64_t>({{0, 0}})})}),
      "Endpoints must be sorted in ascending order");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, endpointsNonFoldable) {
  // Endpoints that are not constant-encoded are accepted as long as every row
  // carries the same array, e.g. when a constant was materialized by an
  // exchange or a table scan.
  {
    auto values = makeFlatVector<double>({0.5, 1.5, 1.5});
    auto endpoints = makeArrayVector<double>(
        {{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}});
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    EXPECT_EQ(ndvs, std::vector<int64_t>({1, 1}));
  }

  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeArrayVector<double>({{0.0, 1.0}, {0.0, 2.0}});
  auto data = makeRowVector({values, endpoints});

  VELOX_ASSERT_USER_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1, 0.05)"},
          {makeRowVector({makeArrayVector<int64_t>({{0}})})}),
      "Endpoints must be constant for all input rows of "
      "approx_count_distinct_for_intervals");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, relativeSdNonFoldable) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeEndpointsVector<double>(values->size(), {0.0, 3.0});
  auto relativeSd = makeFlatVector<double>({0.01, 0.05});
  auto data = makeRowVector({values, endpoints, relativeSd});

  VELOX_ASSERT_USER_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1, c2)"},
          {makeRowVector({makeArrayVector<int64_t>({{0}})})}),
      "relativeSD must be constant for all input rows of "
      "approx_count_distinct_for_intervals");
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    globalAggregationWithoutInputReturnsDefaultCounts) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto data = makeRowVector({values});

  core::AggregationNode::Aggregate aggregate;
  aggregate.call = std::make_shared<core::CallTypedExpr>(
      ARRAY(BIGINT()),
      "approx_count_distinct_for_intervals",
      std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c0"),
      std::make_shared<core::ConstantTypedExpr>(
          makeArrayVector<double>({{0.0, 1.0, 1.0, 2.0}})),
      std::make_shared<core::ConstantTypedExpr>(DOUBLE(), 0.05));
  aggregate.rawInputTypes = {DOUBLE(), ARRAY(DOUBLE()), DOUBLE()};

  auto source = PlanBuilder().values({data}).filter("false").planNode();
  auto plan = std::make_shared<core::AggregationNode>(
      "single",
      core::AggregationNode::Step::kSingle,
      std::vector<core::FieldAccessTypedExprPtr>{},
      std::vector<core::FieldAccessTypedExprPtr>{},
      std::vector<std::string>{"a0"},
      std::vector<core::AggregationNode::Aggregate>{aggregate},
      false,
      false,
      source);

  auto result = AssertQueryBuilder(plan).copyResults(pool());
  auto rows = materialize(result);
  ASSERT_EQ(rows.size(), 1);
  EXPECT_EQ(rows[0][0].array<int64_t>(), std::vector<int64_t>({0, 1, 0}));
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, nanInputsRejected) {
  auto values = makeFlatVector<double>(
      {0.25, std::numeric_limits<double>::quiet_NaN(), 1.25});
  auto endpoints = makeEndpointsVector<double>(values->size(), {0.0, 1.0, 2.0});
  auto data = makeRowVector({values, endpoints});

  VELOX_ASSERT_USER_THROW(
      runGlobalAggregation(
          data, "approx_count_distinct_for_intervals(c0, c1, 0.01)", true),
      "NaN input is rejected for approx_count_distinct_for_intervals");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, mergeEquivalence) {
  auto values =
      makeFlatVector<int64_t>(1'000, [](vector_size_t row) { return row; });
  auto endpoints = makeEndpointsVector<int64_t>(values->size(), {0, 500, 1000});
  auto data = makeRowVector({values, endpoints});

  auto singlePlan =
      PlanBuilder()
          .values({data})
          .singleAggregation(
              {}, {"approx_count_distinct_for_intervals(c0, c1, 0.05)"})
          .planNode();
  auto partialPlan =
      PlanBuilder()
          .values({data})
          .partialAggregation(
              {}, {"approx_count_distinct_for_intervals(c0, c1, 0.05)"})
          .finalAggregation()
          .planNode();

  assertEqualResults(singlePlan, partialPlan);
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, intervalIndexing) {
  auto values = makeFlatVector<double>({0, 3, 6, 10, 2, 4, 8});
  auto endpoints = makeEndpointsVector<double>(values->size(), {0, 3, 6, 10});
  auto data = makeRowVector({values, endpoints});

  auto ndvs = runGlobalAggregation(
      data, "approx_count_distinct_for_intervals(c0, c1, 0.01)", true);
  checkNdvs(ndvs, {3, 2, 2}, 0.01);
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, basicOperations) {
  const std::vector<double> endpoints = {0, 0.33, 0.6, 0.6, 0.6, 1.0};
  const std::vector<double> valuesData = {
      0, 0.6, 0.3, 1, 0.6, 0.5, 0.6, 0.33, 2.0};

  for (const double rsd : {0.01, 0.05, 0.1}) {
    auto values = makeFlatVector<double>(valuesData);
    auto endpointsVector =
        makeEndpointsVector<double>(values->size(), endpoints);
    auto data = makeRowVector({values, endpointsVector});

    auto ndvs = runGlobalAggregation(
        data,
        fmt::format("approx_count_distinct_for_intervals(c0, c1, {})", rsd),
        true);
    checkNdvs(ndvs, {3, 2, 1, 1, 1}, rsd);
  }
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, inputTypes) {
  const std::vector<int32_t> endpointsData = {0, 33, 60, 60, 60, 100};
  const std::vector<int32_t> valuesData = {0, 60, 30, 100, 60, 50, 60, 33};
  const std::vector<int64_t> expected = {3, 2, 1, 1, 1};

  // Integer types.
  testIntegerInputType<int8_t>(valuesData, endpointsData, expected);
  testIntegerInputType<int16_t>(valuesData, endpointsData, expected);
  testIntegerInputType<int32_t>(valuesData, endpointsData, expected);
  testIntegerInputType<int64_t>(valuesData, endpointsData, expected);

  // Date.
  {
    auto values = makeFlatVector<int32_t>(valuesData, DATE());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData, DATE());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Timestamp.
  {
    std::vector<Timestamp> valuesDataTs;
    valuesDataTs.reserve(valuesData.size());
    for (auto value : valuesData) {
      valuesDataTs.push_back(Timestamp::fromMicros(value));
    }

    std::vector<Timestamp> endpointsDataTs;
    endpointsDataTs.reserve(endpointsData.size());
    for (auto value : endpointsData) {
      endpointsDataTs.push_back(Timestamp::fromMicros(value));
    }

    auto values = makeFlatVector<Timestamp>(valuesDataTs);
    auto endpoints =
        makeEndpointsVector<Timestamp>(values->size(), endpointsDataTs);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval day to second (millis).
  {
    std::vector<int64_t> valuesDataInterval;
    std::vector<int64_t> endpointsDataInterval;
    valuesDataInterval.reserve(valuesData.size());
    endpointsDataInterval.reserve(endpointsData.size());
    for (auto value : valuesData) {
      valuesDataInterval.push_back(static_cast<int64_t>(value));
    }
    for (auto value : endpointsData) {
      endpointsDataInterval.push_back(static_cast<int64_t>(value));
    }

    auto values =
        makeFlatVector<int64_t>(valuesDataInterval, INTERVAL_DAY_TIME());
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), endpointsDataInterval, INTERVAL_DAY_TIME());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval year to month (months).
  {
    auto values = makeFlatVector<int32_t>(valuesData, INTERVAL_YEAR_MONTH());
    auto endpoints = makeEndpointsVector<int32_t>(
        values->size(), endpointsData, INTERVAL_YEAR_MONTH());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Decimal (scale 2).
  {
    auto decimalType = DECIMAL(20, 2);
    std::vector<int128_t> valuesDataDecimal;
    std::vector<int128_t> endpointsDataDecimal;
    valuesDataDecimal.reserve(valuesData.size());
    endpointsDataDecimal.reserve(endpointsData.size());
    for (auto value : valuesData) {
      valuesDataDecimal.push_back(static_cast<int128_t>(value) * 100);
    }
    for (auto value : endpointsData) {
      endpointsDataDecimal.push_back(static_cast<int128_t>(value) * 100);
    }
    auto values = makeFlatVector<int128_t>(valuesDataDecimal, decimalType);
    auto endpoints = makeEndpointsVector<int128_t>(
        values->size(), endpointsDataDecimal, decimalType);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, endpointsDifferentType) {
  // Double input with integer endpoints.
  {
    auto values = makeFlatVector<double>({0.2, 0.4, 1.2, 1.6, 1.6, 2.5});
    auto endpoints = makeEndpointsVector<int32_t>(values->size(), {0, 1, 2});
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.01)", true);
    checkNdvs(ndvs, {2, 2}, 0.01);
  }

  // Integer input with double endpoints.
  {
    auto values = makeFlatVector<int32_t>({0, 1, 2, 3, 3});
    auto endpoints =
        makeEndpointsVector<double>(values->size(), {0.0, 1.5, 3.0});
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.01)", true);
    checkNdvs(ndvs, {2, 2}, 0.01);
  }
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    timestampBoundaryAtExactDoubleLimit) {
  const int64_t baseMicros = (1LL << 53) - 2;
  auto values = makeFlatVector<Timestamp>(
      {Timestamp::fromMicros(baseMicros),
       Timestamp::fromMicros(baseMicros + 1),
       Timestamp::fromMicros(baseMicros + 2)});
  auto endpoints = makeEndpointsVector<Timestamp>(
      values->size(),
      {Timestamp::fromMicros(baseMicros),
       Timestamp::fromMicros(baseMicros + 1),
       Timestamp::fromMicros(baseMicros + 2)});
  auto data = makeRowVector({values, endpoints});

  auto ndvs = runGlobalAggregation(
      data, "approx_count_distinct_for_intervals(c0, c1, 0.01)", true);
  checkNdvs(ndvs, {2, 1}, 0.01);
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    highScaleDecimalBoundaries) {
  auto decimalType = DECIMAL(38, 18);
  auto values = makeFlatVector<int128_t>({1, 2, 3}, decimalType);
  auto endpoints =
      makeEndpointsVector<int128_t>(values->size(), {1, 2, 3}, decimalType);
  auto data = makeRowVector({values, endpoints});

  auto ndvs = runGlobalAggregation(
      data, "approx_count_distinct_for_intervals(c0, c1, 0.01)", true);
  checkNdvs(ndvs, {2, 1}, 0.01);
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    endpointsDifferentTypeExtended) {
  const std::vector<int32_t> endpointsData = {0, 33, 60, 60, 60, 100};
  const std::vector<int32_t> valuesData = {0, 60, 30, 100, 60, 50, 60, 33};
  const std::vector<int64_t> expected = {3, 2, 1, 1, 1};

  // Integer input with date endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData, DATE());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Date input with integer endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData, DATE());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Timestamp input with integer endpoints.
  {
    std::vector<Timestamp> valuesDataTs;
    valuesDataTs.reserve(valuesData.size());
    for (auto value : valuesData) {
      valuesDataTs.push_back(Timestamp::fromMicros(value));
    }
    auto values = makeFlatVector<Timestamp>(valuesDataTs);
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Integer input with timestamp endpoints.
  {
    std::vector<Timestamp> endpointsDataTs;
    endpointsDataTs.reserve(endpointsData.size());
    for (auto value : endpointsData) {
      endpointsDataTs.push_back(Timestamp::fromMicros(value));
    }
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints =
        makeEndpointsVector<Timestamp>(values->size(), endpointsDataTs);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Integer input with interval day to second endpoints. Interval millis are
  // evaluated as Spark microseconds, so the integer values are scaled to the
  // endpoints' microsecond magnitudes.
  {
    std::vector<int64_t> endpointsDataInterval;
    endpointsDataInterval.reserve(endpointsData.size());
    for (auto value : endpointsData) {
      endpointsDataInterval.push_back(static_cast<int64_t>(value));
    }
    std::vector<int32_t> valuesDataScaled;
    valuesDataScaled.reserve(valuesData.size());
    for (auto value : valuesData) {
      valuesDataScaled.push_back(value * 1000);
    }
    auto values = makeFlatVector<int32_t>(valuesDataScaled);
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), endpointsDataInterval, INTERVAL_DAY_TIME());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval day to second input with integer endpoints. Interval millis are
  // evaluated as Spark microseconds, so the integer endpoints are scaled to
  // the values' microsecond magnitudes.
  {
    std::vector<int64_t> valuesDataInterval;
    valuesDataInterval.reserve(valuesData.size());
    for (auto value : valuesData) {
      valuesDataInterval.push_back(static_cast<int64_t>(value));
    }
    std::vector<int32_t> endpointsDataScaled;
    endpointsDataScaled.reserve(endpointsData.size());
    for (auto value : endpointsData) {
      endpointsDataScaled.push_back(value * 1000);
    }
    auto values =
        makeFlatVector<int64_t>(valuesDataInterval, INTERVAL_DAY_TIME());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsDataScaled);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Integer input with interval year to month endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints = makeEndpointsVector<int32_t>(
        values->size(), endpointsData, INTERVAL_YEAR_MONTH());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval year to month input with integer endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData, INTERVAL_YEAR_MONTH());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Integer input with decimal endpoints.
  {
    auto decimalType = DECIMAL(10, 0);
    std::vector<int64_t> endpointsDataDecimal;
    endpointsDataDecimal.reserve(endpointsData.size());
    for (auto value : endpointsData) {
      endpointsDataDecimal.push_back(static_cast<int64_t>(value));
    }
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), endpointsDataDecimal, decimalType);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Decimal input with integer endpoints.
  {
    auto decimalType = DECIMAL(10, 0);
    std::vector<int64_t> valuesDataDecimal;
    valuesDataDecimal.reserve(valuesData.size());
    for (auto value : valuesData) {
      valuesDataDecimal.push_back(static_cast<int64_t>(value));
    }
    auto values = makeFlatVector<int64_t>(valuesDataDecimal, decimalType);
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    checkNdvs(ndvs, expected, 0.05);
  }
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, relativeSdOutOfRange) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeEndpointsVector<double>(values->size(), {0.0, 3.0});
  auto data = makeRowVector({values, endpoints});

  for (const auto& relativeSd : {"0.3", "0.001", "0.0"}) {
    VELOX_ASSERT_USER_THROW(
        runGlobalAggregation(
            data,
            fmt::format(
                "approx_count_distinct_for_intervals(c0, c1, {})", relativeSd),
            false),
        "Max standard error must be in [0.0040625, 0.26] range");
  }
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    constantArgumentsAcrossBatches) {
  auto makeBatch = [&](const std::vector<double>& values,
                       const std::vector<double>& endpoints,
                       double relativeSd) {
    auto valuesVector = makeFlatVector<double>(values);
    return makeRowVector(
        {valuesVector,
         makeEndpointsVector<double>(valuesVector->size(), endpoints),
         makeConstant<double>(relativeSd, valuesVector->size())});
  };
  const std::vector<double> valuesA = {0.5, 1.5, 2.5};
  const std::vector<double> valuesB = {0.6, 1.6, 2.6};
  const std::string aggregate =
      "approx_count_distinct_for_intervals(c0, c1, c2)";

  // Batches with the same constant arguments are aggregated together: the
  // second batch contributes distinct values, so both batches are counted.
  {
    auto plan = PlanBuilder()
                    .values(
                        {makeBatch(valuesA, {0.0, 1.0, 3.0}, 0.05),
                         makeBatch(valuesB, {0.0, 1.0, 3.0}, 0.05)})
                    .singleAggregation({}, {aggregate})
                    .planNode();
    auto rows = materialize(AssertQueryBuilder(plan).copyResults(pool()));
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(rows[0][0].array<int64_t>(), std::vector<int64_t>({2, 4}));
  }

  // Each batch is constant on its own, but relativeSD differs across batches.
  {
    auto plan = PlanBuilder()
                    .values(
                        {makeBatch(valuesA, {0.0, 1.0, 3.0}, 0.05),
                         makeBatch(valuesB, {0.0, 1.0, 3.0}, 0.1)})
                    .singleAggregation({}, {aggregate})
                    .planNode();
    VELOX_ASSERT_USER_THROW(
        AssertQueryBuilder(plan).copyResults(pool()),
        "relativeSD must be constant for all input rows of "
        "approx_count_distinct_for_intervals");
  }

  // Each batch is constant on its own, but the endpoints differ across
  // batches, both in value and in size.
  for (const auto& endpoints :
       std::vector<std::vector<double>>{{0.0, 2.0, 3.0}, {0.0, 1.0}}) {
    auto plan = PlanBuilder()
                    .values(
                        {makeBatch(valuesA, {0.0, 1.0, 3.0}, 0.05),
                         makeBatch(valuesB, endpoints, 0.05)})
                    .singleAggregation({}, {aggregate})
                    .planNode();
    VELOX_ASSERT_USER_THROW(
        AssertQueryBuilder(plan).copyResults(pool()),
        "Endpoints must be constant for all input rows of "
        "approx_count_distinct_for_intervals");
  }
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    intermediateResultsValidation) {
  // Merges intermediate rows, i.e. row(array(double), array(varbinary)), with
  // the companion merge_extract function. The companion is registered with a
  // result type suffix because all signatures share the intermediate type.
  const std::string mergeExtract =
      "approx_count_distinct_for_intervals_merge_extract_array_bigint(c0)";
  auto mergeIntermediate = [&](const RowVectorPtr& intermediate) {
    auto plan = PlanBuilder()
                    .values({makeRowVector({intermediate})})
                    .singleAggregation({}, {mergeExtract})
                    .planNode();
    auto rows = materialize(AssertQueryBuilder(plan).copyResults(pool()));
    VELOX_CHECK_EQ(rows.size(), 1);
    return rows[0][0].array<int64_t>();
  };
  auto makeIntermediate =
      [&](const std::vector<std::vector<double>>& endpointsPerRow,
          const std::vector<std::vector<std::string>>& hllsPerRow) {
        std::vector<std::vector<StringView>> hllViews;
        for (const auto& hlls : hllsPerRow) {
          std::vector<StringView> views;
          for (const auto& hll : hlls) {
            views.emplace_back(hll);
          }
          hllViews.push_back(std::move(views));
        }
        return makeRowVector(
            {makeArrayVector<double>(endpointsPerRow),
             makeArrayVector<StringView>(hllViews, VARBINARY())});
      };
  auto partialResult = [&](const std::vector<double>& values) {
    auto valuesVector = makeFlatVector<double>(values);
    auto data = makeRowVector(
        {valuesVector,
         makeEndpointsVector<double>(valuesVector->size(), {0.0, 1.0, 2.0})});
    auto plan =
        PlanBuilder()
            .values({data})
            .partialAggregation(
                {}, {"approx_count_distinct_for_intervals(c0, c1, 0.05)"})
            .planNode();
    auto result = AssertQueryBuilder(plan).copyResults(pool());
    return std::dynamic_pointer_cast<RowVector>(result->childAt(0));
  };

  const auto empty9 = common::hll::SparseHlls::serializeEmpty(9);
  const auto empty11 = common::hll::SparseHlls::serializeEmpty(11);

  // Well-formed intermediate rows produced by partial aggregations merge into
  // the union of their inputs.
  {
    auto first = partialResult({0.5, 1.5});
    auto second = partialResult({0.5, 1.6, 1.7});
    auto merged = BaseVector::create(first->type(), 0, pool());
    merged->append(first.get());
    merged->append(second.get());
    EXPECT_EQ(
        mergeIntermediate(std::dynamic_pointer_cast<RowVector>(merged)),
        std::vector<int64_t>({1, 3}));
  }
  EXPECT_EQ(
      mergeIntermediate(makeIntermediate(
          {{0.0, 1.0, 1.0, 2.0}, {0.0, 1.0, 1.0, 2.0}},
          {{empty9, empty9, empty9}, {empty9, empty9, empty9}})),
      std::vector<int64_t>({0, 1, 0}));

  // Every intermediate row must carry the same endpoints.
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeIntermediate(
          {{0.0, 1.0, 2.0}, {0.0, 1.5, 2.0}},
          {{empty9, empty9}, {empty9, empty9}})),
      "Endpoints must be constant for all input rows of "
      "approx_count_distinct_for_intervals");
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeIntermediate(
          {{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0, 3.0}},
          {{empty9, empty9}, {empty9, empty9, empty9}})),
      "Endpoints must be constant for all input rows of "
      "approx_count_distinct_for_intervals");

  // The number of HLLs must match the number of intervals.
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeIntermediate({{0.0, 1.0, 2.0}}, {{empty9}})),
      "HLL array size 1 does not match the number of intervals 2");

  // Every HLL must use the same precision, whether sparse or dense.
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeIntermediate(
          {{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}},
          {{empty9, empty9}, {empty11, empty11}})),
      "Cannot merge HLLs with different number of buckets");
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(
          makeIntermediate({{0.0, 1.0, 2.0}}, {{empty9, empty11}})),
      "Cannot merge HLLs with different number of buckets");
  {
    HashStringAllocator allocator(pool());
    common::hll::DenseHll<> dense(11, &allocator);
    for (uint64_t hash = 1; hash <= 100; ++hash) {
      dense.insertHash(hash * 0x9E3779B97F4A7C15ULL);
    }
    std::string serializedDense(dense.serializedSize(), '\0');
    dense.serialize(serializedDense.data());
    VELOX_ASSERT_USER_THROW(
        mergeIntermediate(
            makeIntermediate({{0.0, 1.0, 2.0}}, {{empty9, serializedDense}})),
        "Cannot merge HLLs with different number of buckets");
    // A dense HLL with matching precision merges fine.
    auto ndvs = mergeIntermediate(
        makeIntermediate({{0.0, 1.0, 2.0}}, {{empty11, serializedDense}}));
    ASSERT_EQ(ndvs.size(), 2);
    EXPECT_EQ(ndvs[0], 0);
    EXPECT_NEAR(ndvs[1], 100, 100 * 3 * 0.05);
  }

  // Empty, truncated and garbage HLLs are rejected without being read.
  for (const auto& malformed : std::vector<std::string>{
           "", empty9.substr(0, 2), "not an hll", std::string(4, '\0')}) {
    VELOX_ASSERT_USER_THROW(
        mergeIntermediate(
            makeIntermediate({{0.0, 1.0, 2.0}}, {{empty9, malformed}})),
        "Invalid serialized HyperLogLog");
  }
  // A sparse HLL that claims more entries than it carries.
  auto truncated = empty9;
  truncated[2] = 5;
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(
          makeIntermediate({{0.0, 1.0, 2.0}}, {{truncated, empty9}})),
      "Invalid serialized HyperLogLog");

  // Null endpoints, HLL arrays, HLL entries and endpoint values are rejected.
  using NullableDoubles = std::vector<std::optional<double>>;
  using NullableHlls = std::vector<std::optional<StringView>>;
  auto endpointsArray = makeArrayVector<double>({{0.0, 1.0, 2.0}});
  auto hllsArray = makeArrayVector<StringView>(
      {{StringView(empty9), StringView(empty9)}}, VARBINARY());
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeRowVector(
          {makeNullableArrayVector<double>(
               std::vector<std::optional<NullableDoubles>>{std::nullopt}),
           hllsArray})),
      "Malformed intermediate result for approx_count_distinct_for_intervals");
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeRowVector(
          {endpointsArray,
           makeNullableArrayVector<StringView>(
               std::vector<std::optional<NullableHlls>>{std::nullopt},
               ARRAY(VARBINARY()))})),
      "Malformed intermediate result for approx_count_distinct_for_intervals");
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeRowVector(
          {endpointsArray,
           makeNullableArrayVector<StringView>(
               std::vector<NullableHlls>{{StringView(empty9), std::nullopt}},
               ARRAY(VARBINARY()))})),
      "Serialized HLL entries must not be null");
  VELOX_ASSERT_USER_THROW(
      mergeIntermediate(makeRowVector(
          {makeNullableArrayVector<double>(
               std::vector<NullableDoubles>{{0.0, std::nullopt, 2.0}}),
           hllsArray})),
      "Endpoints must not contain null values");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, intervalOutOfSparkRange) {
  // Velox intervals hold milliseconds, so values beyond INT64_MAX / 1000 have
  // no Spark microsecond representation.
  const auto kMax = std::numeric_limits<int64_t>::max();
  const auto kMin = std::numeric_limits<int64_t>::min();

  for (const auto value : {kMax, kMin, kMax / 1000 + 1}) {
    auto values = makeFlatVector<int64_t>({0, value}, INTERVAL_DAY_TIME());
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), {0, 1000}, INTERVAL_DAY_TIME());
    auto data = makeRowVector({values, endpoints});
    VELOX_ASSERT_USER_THROW(
        runGlobalAggregation(
            data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", false),
        "is out of range for Spark's DayTimeIntervalType");
  }

  // Endpoints are converted the same way.
  {
    auto values = makeFlatVector<int64_t>({0, 1}, INTERVAL_DAY_TIME());
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), {0, kMax}, INTERVAL_DAY_TIME());
    auto data = makeRowVector({values, endpoints});
    VELOX_ASSERT_USER_THROW(
        runGlobalAggregation(
            data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", false),
        "is out of range for Spark's DayTimeIntervalType");
  }

  // The largest representable value is accepted.
  {
    auto values =
        makeFlatVector<int64_t>({0, kMax / 1000}, INTERVAL_DAY_TIME());
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), {0, kMax / 1000}, INTERVAL_DAY_TIME());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", false);
    checkNdvs(ndvs, {2}, 0.05);
  }
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, windowEmptyFrames) {
  // Window functions compute the result for empty frames right after the
  // aggregate is created, so the constant arguments must reach it before any
  // input row does.
  auto data = makeRowVector(
      {"p", "s", "x"},
      {makeFlatVector<int32_t>({1, 1, 1, 1}),
       makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<double>({0.5, 1.5, 2.5, 3.5})});

  auto call = std::make_shared<core::CallTypedExpr>(
      ARRAY(BIGINT()),
      "approx_count_distinct_for_intervals",
      std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "x"),
      std::make_shared<core::ConstantTypedExpr>(
          makeArrayVector<double>({{0.0, 1.0, 2.0, 2.0, 4.0}})),
      std::make_shared<core::ConstantTypedExpr>(DOUBLE(), 0.05));

  // ROWS BETWEEN 1 FOLLOWING AND 1 FOLLOWING: each row aggregates the next row
  // only, so the last row of the partition has an empty frame.
  core::WindowNode::Frame frame{
      core::WindowNode::WindowType::kRows,
      core::WindowNode::BoundType::kFollowing,
      std::make_shared<core::ConstantTypedExpr>(
          BIGINT(), Variant(static_cast<int64_t>(1))),
      core::WindowNode::BoundType::kFollowing,
      std::make_shared<core::ConstantTypedExpr>(
          BIGINT(), Variant(static_cast<int64_t>(1)))};

  auto plan = std::make_shared<core::WindowNode>(
      "window",
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "p")},
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "s")},
      std::vector<core::SortOrder>{core::kAscNullsLast},
      std::vector<std::string>{"w0"},
      std::vector<core::WindowNode::Function>{{call, frame, false}},
      false,
      PlanBuilder().values({data}).planNode());

  // Each frame holds a single value, so the counts are exact. The empty frame
  // of the last row yields zero counts, with the duplicate-endpoint interval
  // (2, 2] set to 1.
  auto expected = makeRowVector(
      {"p", "s", "x", "w0"},
      {data->childAt(0),
       data->childAt(1),
       data->childAt(2),
       makeArrayVector<int64_t>(
           {{0, 1, 1, 0}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 0}})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, relativeSdBoundaries) {
  // The smallest and largest accepted values map to 16 and 4 index bits.
  auto values = makeFlatVector<double>({0.5, 1.5, 1.5, 2.5});
  auto endpoints =
      makeEndpointsVector<double>(values->size(), {0.0, 1.0, 2.0, 2.0, 3.0});
  auto data = makeRowVector({values, endpoints});

  for (const auto& relativeSd : {"0.0040625", "0.26"}) {
    for (const bool usePartial : {false, true}) {
      auto ndvs = runGlobalAggregation(
          data,
          fmt::format(
              "approx_count_distinct_for_intervals(c0, c1, {})", relativeSd),
          usePartial);
      EXPECT_EQ(ndvs, std::vector<int64_t>({1, 1, 1, 1}));
    }
  }
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, endpointsWithNan) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeEndpointsVector<double>(
      values->size(), {0.0, std::numeric_limits<double>::quiet_NaN()});
  auto data = makeRowVector({values, endpoints});

  VELOX_ASSERT_USER_THROW(
      runGlobalAggregation(
          data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", false),
      "Endpoints must not contain NaN");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, realEndpoints) {
  // Spark converts endpoints through their string representation, so a REAL
  // endpoint 0.1f is the double 0.1, while a REAL input 0.1f is widened to
  // 0.10000000149011612.
  {
    auto values = makeFlatVector<float>({0.1f, 0.05f});
    auto endpoints = makeEndpointsVector<float>(values->size(), {0.0f, 0.1f});
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    // 0.1f widened exceeds the last endpoint 0.1 and is ignored.
    EXPECT_EQ(ndvs, std::vector<int64_t>({1}));
  }
  {
    auto values = makeFlatVector<double>({0.1, 0.5});
    auto endpoints = makeEndpointsVector<float>(values->size(), {0.1f, 1.0f});
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
    // The double 0.1 equals the first endpoint and belongs to the interval.
    EXPECT_EQ(ndvs, std::vector<int64_t>({2}));
  }
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, signedZeroEndpoints) {
  // Spark orders -0.0 before 0.0 when locating the interval, so 0.0 falls in
  // (-0.0, 1.0] while -0.0 falls in [-1.0, -0.0].
  auto values = makeFlatVector<double>({0.0, -0.0});
  auto endpoints =
      makeEndpointsVector<double>(values->size(), {-1.0, -0.0, 1.0});
  auto data = makeRowVector({values, endpoints});
  auto ndvs = runGlobalAggregation(
      data, "approx_count_distinct_for_intervals(c0, c1, 0.05)", true);
  EXPECT_EQ(ndvs, std::vector<int64_t>({1, 1}));

  // Endpoints (0.0, -0.0) are not in ascending order.
  auto unsorted =
      makeEndpointsVector<double>(values->size(), {-1.0, 0.0, -0.0, 1.0});
  VELOX_ASSERT_USER_THROW(
      runGlobalAggregation(
          makeRowVector({values, unsorted}),
          "approx_count_distinct_for_intervals(c0, c1, 0.05)",
          false),
      "Endpoints must be sorted in ascending order");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, groupBy) {
  // Small per-interval cardinalities are estimated exactly, so the standard
  // aggregation test harness can compare against exact results across all
  // its plan variants.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3}),
       makeNullableFlatVector<int64_t>(
           {0, 5, 5, 10, 15, 20, 25, 10, 10, 11, std::nullopt, std::nullopt}),
       makeEndpointsVector<int64_t>(12, {0, 10, 20})});

  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeArrayVector<int64_t>({{3, 2}, {1, 1}, {0, 0}})});

  testAggregations(
      {data},
      {"c0"},
      {"approx_count_distinct_for_intervals(c1, c2, 0.05)"},
      {expected});
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
