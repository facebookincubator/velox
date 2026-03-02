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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Variant.h"
#include "velox/vector/BaseVector.h"

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
};

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, invalidInputType) {
  static const std::string kErrorMessage =
      "Aggregate function signature is not supported";
  auto values = makeFlatVector<bool>({true, false});
  auto endpoints = makeArrayVector<bool>({{false, true}, {false, true}});
  auto data = makeRowVector({values, endpoints});
  auto builder = PlanBuilder().values({data});
  VELOX_ASSERT_THROW(
      builder.singleAggregation(
          {}, {"approx_count_distinct_for_intervals(c0, c1)"}),
      kErrorMessage);
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, endpointsNotArray) {
  static const std::string kErrorMessage =
      "Aggregate function signature is not supported";
  auto values = makeFlatVector<int32_t>({1, 2});
  auto endpoints = makeFlatVector<int32_t>({0, 1});
  auto data = makeRowVector({values, endpoints});
  auto builder = PlanBuilder().values({data});
  VELOX_ASSERT_THROW(
      builder.singleAggregation(
          {}, {"approx_count_distinct_for_intervals(c0, c1)"}),
      kErrorMessage);
}

TEST_F(
    ApproxCountDistinctForIntervalsAggregateTest,
    endpointsWrongElementType) {
  static const std::string kErrorMessage =
      "Aggregate function signature is not supported";
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints =
      makeArrayVector<StringView>({{"a"_sv, "b"_sv}, {"a"_sv, "b"_sv}});
  auto data = makeRowVector({values, endpoints});
  auto builder = PlanBuilder().values({data});
  VELOX_ASSERT_THROW(
      builder.singleAggregation(
          {}, {"approx_count_distinct_for_intervals(c0, c1)"}),
      kErrorMessage);
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, tooFewEndpoints) {
  auto values = makeFlatVector<int32_t>({1});
  auto endpoints = makeEndpointsVector<int32_t>(values->size(), {0});
  auto data = makeRowVector({values, endpoints});
  auto expected = makeRowVector({makeArrayVector<int64_t>({{0}})});

  VELOX_ASSERT_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1)"},
          {expected}),
      "approx_count_distinct_for_intervals requires at least 2 endpoints");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, endpointsNonFoldable) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeArrayVector<double>({{0.0, 1.0}, {0.0, 2.0}});
  auto data = makeRowVector({values, endpoints});

  VELOX_ASSERT_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1)"},
          {makeRowVector({makeArrayVector<int64_t>({{0}})})}),
      "Endpoints must be constant for approx_count_distinct_for_intervals");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, relativeSdNonFoldable) {
  auto values = makeFlatVector<double>({1.0, 2.0});
  auto endpoints = makeEndpointsVector<double>(values->size(), {0.0, 3.0});
  auto relativeSd = makeFlatVector<double>({0.01, 0.05});
  auto data = makeRowVector({values, endpoints, relativeSd});

  VELOX_ASSERT_THROW(
      testAggregations(
          {data},
          {},
          {"approx_count_distinct_for_intervals(c0, c1, c2)"},
          {makeRowVector({makeArrayVector<int64_t>({{0}})})}),
      "relativeSD must be constant for approx_count_distinct_for_intervals");
}

TEST_F(ApproxCountDistinctForIntervalsAggregateTest, mergeEquivalence) {
  auto values =
      makeFlatVector<int64_t>(1'000, [](vector_size_t row) { return row; });
  auto endpoints = makeEndpointsVector<int64_t>(values->size(), {0, 500, 1000});
  auto data = makeRowVector({values, endpoints});

  auto singlePlan = PlanBuilder()
                        .values({data})
                        .singleAggregation(
                            {}, {"approx_count_distinct_for_intervals(c0, c1)"})
                        .planNode();
  auto partialPlan =
      PlanBuilder()
          .values({data})
          .partialAggregation(
              {}, {"approx_count_distinct_for_intervals(c0, c1)"})
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

  // Integer.
  {
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Date.
  {
    auto values = makeFlatVector<int32_t>(valuesData, DATE());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData, DATE());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval year to month (months).
  {
    auto values = makeFlatVector<int32_t>(valuesData, INTERVAL_YEAR_MONTH());
    auto endpoints = makeEndpointsVector<int32_t>(
        values->size(), endpointsData, INTERVAL_YEAR_MONTH());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Date input with integer endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData, DATE());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Integer input with interval day to second endpoints.
  {
    std::vector<int64_t> endpointsDataInterval;
    endpointsDataInterval.reserve(endpointsData.size());
    for (auto value : endpointsData) {
      endpointsDataInterval.push_back(static_cast<int64_t>(value));
    }
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints = makeEndpointsVector<int64_t>(
        values->size(), endpointsDataInterval, INTERVAL_DAY_TIME());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval day to second input with integer endpoints.
  {
    std::vector<int64_t> valuesDataInterval;
    valuesDataInterval.reserve(valuesData.size());
    for (auto value : valuesData) {
      valuesDataInterval.push_back(static_cast<int64_t>(value));
    }
    auto values =
        makeFlatVector<int64_t>(valuesDataInterval, INTERVAL_DAY_TIME());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Integer input with interval year to month endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData);
    auto endpoints = makeEndpointsVector<int32_t>(
        values->size(), endpointsData, INTERVAL_YEAR_MONTH());
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }

  // Interval year to month input with integer endpoints.
  {
    auto values = makeFlatVector<int32_t>(valuesData, INTERVAL_YEAR_MONTH());
    auto endpoints =
        makeEndpointsVector<int32_t>(values->size(), endpointsData);
    auto data = makeRowVector({values, endpoints});
    auto ndvs = runGlobalAggregation(
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
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
        data, "approx_count_distinct_for_intervals(c0, c1)", true);
    checkNdvs(ndvs, expected, 0.05);
  }
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
