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
#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/WindowFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/HistogramNumericAggregate.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

class HistogramNumericAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("");
  }

  void assertSingle(
      const RowVectorPtr& input,
      const std::string& expression,
      const RowVectorPtr& expected) {
    parse::ParseOptions options;
    options.parseIntegerAsBigint = false;
    auto plan = PlanBuilder()
                    .setParseOptions(options)
                    .values({input})
                    .singleAggregation({}, {expression})
                    .planNode();
    AssertQueryBuilder(plan).assertResults(expected);
  }

  RowVectorPtr expectedDouble(
      const std::vector<std::tuple<double, double>>& bins) {
    std::vector<std::optional<std::tuple<double, double>>> values;
    values.reserve(bins.size());
    for (const auto& bin : bins) {
      values.push_back(bin);
    }
    return makeRowVector({makeArrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<double, double>>>>{
            std::move(values)},
        ROW({"x", "y"}, {DOUBLE(), DOUBLE()}))});
  }
};

TEST_F(HistogramNumericAggregateTest, registrationAndSignatures) {
  const auto names = exec::getAggregateFunctionNames();
  for (const auto* name : {
           "histogram_numeric",
           "histogram_numeric_legacy",
       }) {
    EXPECT_NE(std::find(names.begin(), names.end(), name), names.end()) << name;
  }
  EXPECT_EQ(
      exec::getAggregateFunctionSignatures("histogram_numeric")->size(), 11);
  EXPECT_FALSE(
      exec::getAggregateFunctionSignatures("histogram_numeric_partial")
          .has_value());
  EXPECT_FALSE(
      exec::getAggregateFunctionSignatures("histogram_numeric_legacy_partial")
          .has_value());
}

TEST_F(HistogramNumericAggregateTest, doubleAndLegacyResults) {
  auto input = makeRowVector({makeFlatVector<double>({1, 2, 3})});
  auto expected = expectedDouble({{1, 1}, {2, 1}, {3, 1}});
  assertSingle(input, "histogram_numeric(c0, 3)", expected);
  assertSingle(input, "histogram_numeric_legacy(c0, 3)", expected);
}

TEST_F(HistogramNumericAggregateTest, typedIntegralResults) {
  auto input = makeRowVector({makeFlatVector<int8_t>({-2, 1, 4})});
  auto expected = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<int8_t, double>>>>{
          {std::tuple<int8_t, double>{-2, 1},
           std::tuple<int8_t, double>{1, 1},
           std::tuple<int8_t, double>{4, 1}}},
      ROW({"x", "y"}, {TINYINT(), DOUBLE()}))});
  assertSingle(input, "histogram_numeric(c0, 3)", expected);
}

TEST_F(HistogramNumericAggregateTest, typedTemporalResults) {
  auto dates = makeRowVector({makeFlatVector<int32_t>({-1, 0, 1}, DATE())});
  auto expectedDates = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<int32_t, double>>>>{
          {std::tuple<int32_t, double>{-1, 1},
           std::tuple<int32_t, double>{0, 1},
           std::tuple<int32_t, double>{1, 1}}},
      ROW({"x", "y"}, {DATE(), DOUBLE()}))});
  assertSingle(dates, "histogram_numeric(c0, 3)", expectedDates);

  auto timestamps = makeRowVector({makeFlatVector<Timestamp>(
      {Timestamp(-1, 999'999'000), Timestamp(0, 0), Timestamp(0, 1'000)})});
  auto expectedTimestamps = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<Timestamp, double>>>>{{
          std::tuple<Timestamp, double>{Timestamp(-1, 999'999'000), 1},
          std::tuple<Timestamp, double>{Timestamp(0, 0), 1},
          std::tuple<Timestamp, double>{Timestamp(0, 1'000), 1},
      }},
      ROW({"x", "y"}, {TIMESTAMP(), DOUBLE()}))});
  assertSingle(timestamps, "histogram_numeric(c0, 3)", expectedTimestamps);
}

TEST_F(HistogramNumericAggregateTest, nullEmptyAndAllNull) {
  auto input = makeRowVector(
      {makeNullableFlatVector<double>({std::nullopt, std::nullopt})});
  auto expected = makeRowVector({BaseVector::createNullConstant(
      ARRAY(ROW({"x", "y"}, {DOUBLE(), DOUBLE()})), 1, pool())});
  assertSingle(input, "histogram_numeric(c0, 3)", expected);

  auto empty = makeRowVector({makeFlatVector<double>({})});
  assertSingle(empty, "histogram_numeric(c0, 3)", expected);
}

TEST_F(HistogramNumericAggregateTest, validatesNumBinsWithoutNonNullValues) {
  auto input = makeRowVector({makeNullableFlatVector<double>({std::nullopt})});
  parse::ParseOptions options;
  options.parseIntegerAsBigint = false;

  auto singlePlan = PlanBuilder()
                        .setParseOptions(options)
                        .values({input})
                        .singleAggregation({}, {"histogram_numeric(c0, 1)"})
                        .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(singlePlan).copyResults(pool()), "at least 2");

  auto partialFinalPlan =
      PlanBuilder()
          .setParseOptions(options)
          .values({input})
          .partialAggregation({}, {"histogram_numeric(c0, 1)"})
          .finalAggregation()
          .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(partialFinalPlan).copyResults(pool()), "at least 2");
}

TEST_F(HistogramNumericAggregateTest, decimalTyped) {
  auto input =
      makeRowVector({makeFlatVector<int64_t>({125, 250}, DECIMAL(3, 2))});
  auto expected = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<int64_t, double>>>>{
          {std::tuple<int64_t, double>{125, 1},
           std::tuple<int64_t, double>{250, 1}}},
      ROW({"x", "y"}, {DECIMAL(3, 2), DOUBLE()}))});
  assertSingle(input, "histogram_numeric(c0, 2)", expected);
}

TEST_F(HistogramNumericAggregateTest, decimalBoundaryMaterializesNullCenter) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({999'999'999'999'999'999}, DECIMAL(18, 0))});
  auto bins = makeRowVector(
      {"x", "y"},
      {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(18, 0)),
       makeFlatVector<double>({1})});
  auto expected = makeRowVector({makeArrayVector({0}, bins)});
  parse::ParseOptions options;
  options.parseIntegerAsBigint = false;

  auto singlePlan = PlanBuilder()
                        .setParseOptions(options)
                        .values({input})
                        .singleAggregation({}, {"histogram_numeric(c0, 2)"})
                        .planNode();
  AssertQueryBuilder(singlePlan).assertResults(expected);

  auto partialFinalPlan =
      PlanBuilder()
          .setParseOptions(options)
          .values({input})
          .partialAggregation({}, {"histogram_numeric(c0, 2)"})
          .finalAggregation()
          .planNode();
  AssertQueryBuilder(partialFinalPlan).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
