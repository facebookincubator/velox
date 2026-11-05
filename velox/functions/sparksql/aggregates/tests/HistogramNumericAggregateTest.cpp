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
#include <string_view>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Spill.h"
#include "velox/exec/WindowFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/HistogramNumericAggregate.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/type/HugeInt.h"

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
        ROW({"x", "y"}, DOUBLE()))});
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
      exec::getAggregateFunctionSignatures("histogram_numeric")->size(), 10);
  EXPECT_EQ(
      exec::getAggregateFunctionSignatures("histogram_numeric_legacy")->size(),
      11);
  EXPECT_TRUE(
      exec::getAggregateFunctionSignatures("histogram_numeric_partial")
          .has_value());
  EXPECT_TRUE(
      exec::getAggregateFunctionSignatures("histogram_numeric_legacy_partial")
          .has_value());
  registerHistogramNumericAggregates(
      "no_companions_", /*withCompanionFunctions=*/false, /*overwrite=*/true);
  for (const auto* name : {
           "no_companions_histogram_numeric",
           "no_companions_histogram_numeric_legacy",
       }) {
    EXPECT_TRUE(exec::getAggregateFunctionSignatures(name).has_value());
    EXPECT_FALSE(
        exec::getAggregateFunctionSignatures(std::string(name) + "_partial")
            .has_value());
  }
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
      ROW({{"x", TINYINT()}, {"y", DOUBLE()}}))});
  assertSingle(input, "histogram_numeric(c0, 3)", expected);
}

TEST_F(HistogramNumericAggregateTest, groupedTypedResults) {
  const std::vector<VectorPtr> values{
      makeFlatVector<int8_t>({-2, 1, 4, 10, 20, 30}),
      makeFlatVector<int16_t>({-2, 1, 4, 10, 20, 30}),
      makeFlatVector<int32_t>({-2, 1, 4, 10, 20, 30}),
      makeFlatVector<int64_t>({-2, 1, 4, 10, 20, 30}),
      makeFlatVector<float>({-2, 1, 4, 10, 20, 30}),
      makeFlatVector<double>({-2, 1, 4, 10, 20, 30}),
      makeFlatVector<int32_t>({-2, 1, 4, 10, 20, 30}, DATE()),
      makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(-2),
           Timestamp::fromMicros(1),
           Timestamp::fromMicros(4),
           Timestamp::fromMicros(10),
           Timestamp::fromMicros(20),
           Timestamp::fromMicros(30)}),
      makeFlatVector<Timestamp>(
          {Timestamp::fromMicros(-2),
           Timestamp::fromMicros(1),
           Timestamp::fromMicros(4),
           Timestamp::fromMicros(10),
           Timestamp::fromMicros(20),
           Timestamp::fromMicros(30)},
          TIMESTAMP_UTC()),
      makeFlatVector<int32_t>({-2, 1, 4, 10, 20, 30}, INTERVAL_YEAR_MONTH()),
  };
  const auto keys = makeFlatVector<int32_t>({0, 0, 0, 1, 1, 1});
  const auto expectedKeys = makeFlatVector<int32_t>({0, 1});
  const auto heights = makeConstant<double>(1, 6);
  const auto legacyBins = makeArrayVector(
      {0, 3},
      makeRowVector(
          {"x", "y"},
          {makeFlatVector<double>({-2, 1, 4, 10, 20, 30}), heights}));
  const auto legacyExpected = makeRowVector({expectedKeys, legacyBins});
  for (const auto& centers : values) {
    SCOPED_TRACE(centers->type()->toString());
    const auto input = makeRowVector({keys, centers});
    const auto batches = split(input, 3);
    const auto bins =
        makeArrayVector({0, 3}, makeRowVector({"x", "y"}, {centers, heights}));
    const auto expected = makeRowVector({expectedKeys, bins});
    for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
      SCOPED_TRACE(name);
      const auto expression = fmt::format("{}(c1, INTEGER '3')", name);
      const auto& result = std::string_view(name) == "histogram_numeric"
          ? expected
          : legacyExpected;
      AggregationTester(*this)
          .inputs(batches)
          .groupBy({"c0"})
          .aggregates({expression})
          .assertResults(result);

      // These distinct inputs never trim, so results are order-independent even
      // though the aggregate's metadata disables automatic spill/streaming
      // tests.
      {
        SCOPED_TRACE("Single with spilling");
        core::PlanNodeId aggregationNodeId;
        const auto plan = PlanBuilder()
                              .values(batches)
                              .singleAggregation({"c0"}, {expression})
                              .capturePlanNodeId(aggregationNodeId)
                              .planNode();
        const auto spillDirectory =
            common::testutil::TempDirectoryPath::create();
        exec::TestScopedSpillInjection scopedSpillInjection(100);
        const auto task =
            AssertQueryBuilder(plan)
                .config(core::QueryConfig::kSpillEnabled, true)
                .config(core::QueryConfig::kAggregationSpillEnabled, true)
                .spillDirectory(spillDirectory->getPath())
                .assertResults(result);
        EXPECT_GT(
            exec::toPlanStats(task->taskStats())
                .at(aggregationNodeId)
                .spilledBytes,
            0);
      }

      {
        SCOPED_TRACE("Streaming single");
        const auto plan = PlanBuilder()
                              .values(batches)
                              .streamingAggregation(
                                  {"c0"},
                                  {expression},
                                  {},
                                  core::AggregationNode::Step::kSingle,
                                  false)
                              .planNode();
        AssertQueryBuilder(plan).assertResults(result);
      }

      for (const auto intermediate : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "Streaming partial + final, intermediate={}", intermediate));
        PlanBuilder builder;
        builder.values(batches).partialStreamingAggregation(
            {"c0"}, {expression});
        if (intermediate) {
          builder.intermediateAggregation();
        }
        builder.finalAggregation();
        AssertQueryBuilder(builder.planNode()).assertResults(result);
      }
    }
  }
}

TEST_F(HistogramNumericAggregateTest, companions) {
  const auto centers = makeFlatVector<int32_t>({-2, 1, 4, 10, 20, 30});
  const auto input =
      makeRowVector({makeFlatVector<int32_t>({0, 0, 0, 1, 1, 1}), centers});
  const auto heights = makeConstant<double>(1, 6);
  for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
    SCOPED_TRACE(name);
    VectorPtr outputCenters = std::string_view(name) == "histogram_numeric"
        ? VectorPtr(centers)
        : makeFlatVector<double>({-2, 1, 4, 10, 20, 30});
    const auto expected = makeRowVector({
        makeFlatVector<int32_t>({0, 1}),
        makeArrayVector(
            {0, 3}, makeRowVector({"x", "y"}, {outputCenters, heights})),
    });
    AggregationTester(*this)
        .inputs(split(input, 3))
        .groupBy({"c0"})
        .aggregates({fmt::format("{}(c1, INTEGER '3')", name)})
        .withCompanion()
        .assertResults(expected);
  }
}

TEST_F(HistogramNumericAggregateTest, trimmingNarrowsIntegralCenters) {
  auto input = makeRowVector({makeFlatVector<int32_t>({0, 1, 2})});
  auto expected = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<int32_t, double>>>>{
          {std::tuple<int32_t, double>{0, 1},
           std::tuple<int32_t, double>{1, 2}}},
      ROW({{"x", INTEGER()}, {"y", DOUBLE()}}))});
  assertSingle(input, "histogram_numeric(c0, 2)", expected);
}

TEST_F(HistogramNumericAggregateTest, typedTemporalResults) {
  auto dates = makeRowVector({makeFlatVector<int32_t>({-1, 0, 1}, DATE())});
  auto expectedDates = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<int32_t, double>>>>{
          {std::tuple<int32_t, double>{-1, 1},
           std::tuple<int32_t, double>{0, 1},
           std::tuple<int32_t, double>{1, 1}}},
      ROW({{"x", DATE()}, {"y", DOUBLE()}}))});
  assertSingle(dates, "histogram_numeric(c0, 3)", expectedDates);

  auto timestamps = makeRowVector({makeFlatVector<Timestamp>(
      {Timestamp(-1, 999'999'000), Timestamp(0, 0), Timestamp(0, 1'000)})});
  auto expectedTimestamps = makeRowVector({makeArrayOfRowVector(
      std::vector<std::vector<std::optional<std::tuple<Timestamp, double>>>>{{
          std::tuple<Timestamp, double>{Timestamp(-1, 999'999'000), 1},
          std::tuple<Timestamp, double>{Timestamp(0, 0), 1},
          std::tuple<Timestamp, double>{Timestamp(0, 1'000), 1},
      }},
      ROW({{"x", TIMESTAMP()}, {"y", DOUBLE()}}))});
  assertSingle(timestamps, "histogram_numeric(c0, 3)", expectedTimestamps);
}

TEST_F(HistogramNumericAggregateTest, nullEmptyAndAllNull) {
  auto input = makeRowVector(
      {makeNullableFlatVector<double>({std::nullopt, std::nullopt})});
  auto expected = makeRowVector({BaseVector::createNullConstant(
      ARRAY(ROW({"x", "y"}, DOUBLE())), 1, pool())});
  assertSingle(input, "histogram_numeric(c0, 3)", expected);

  auto empty = makeRowVector({makeFlatVector<double>({})});
  assertSingle(empty, "histogram_numeric(c0, 3)", expected);
}

TEST_F(HistogramNumericAggregateTest, companionsPreserveEmptyCapacity) {
  const auto expected = makeRowVector({makeFlatVector<std::string>(
      {std::string("\0\0\0\3\0\0\0\0", 8)}, VARBINARY())});
  const auto expectedNull = makeRowVector({BaseVector::createNullConstant(
      ARRAY(ROW({"x", "y"}, DOUBLE())), 1, pool())});
  for (const auto& input : {
           makeRowVector({makeFlatVector<double>({})}),
           makeRowVector({makeNullableFlatVector<double>({std::nullopt})}),
           makeRowVector({makeFlatVector<double>({1, 2, 3})}),
       }) {
    for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
      for (const auto masked : {false, true}) {
        SCOPED_TRACE(
            fmt::format("{}: {} rows, masked={}", name, input->size(), masked));
        if (!masked && input->size() == 3) {
          continue;
        }
        PlanBuilder builder;
        builder.values({input}).project({"c0", "false AS keep"});
        builder.singleAggregation(
            {},
            {fmt::format("{}_partial(c0, INTEGER '3')", name)},
            masked ? std::vector<std::string>{"keep"}
                   : std::vector<std::string>{});
        AssertQueryBuilder(builder.planNode()).assertResults(expected);
        builder.singleAggregation({}, {fmt::format("{}_merge(a0)", name)});
        AssertQueryBuilder(builder.planNode()).assertResults(expected);
        builder.singleAggregation(
            {},
            {fmt::format(
                "{}_merge_extract_array_row_double_double_endrow(a0)", name)});
        AssertQueryBuilder(builder.planNode()).assertResults(expectedNull);
      }
    }
  }
}

TEST_F(HistogramNumericAggregateTest, companionsValidateEmptyNumBins) {
  for (const auto& input : {
           makeRowVector({makeFlatVector<double>({})}),
           makeRowVector({makeNullableFlatVector<double>({std::nullopt})}),
       }) {
    for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
      SCOPED_TRACE(fmt::format("{}: {} rows", name, input->size()));
      for (const auto* numBins : {"INTEGER '1'", "CAST(NULL AS INTEGER)"}) {
        auto plan =
            PlanBuilder()
                .values({input})
                .singleAggregation(
                    {}, {fmt::format("{}_partial(c0, {})", name, numBins)})
                .planNode();
        VELOX_ASSERT_USER_THROW(
            AssertQueryBuilder(plan).copyResults(pool()),
            std::string_view(numBins) == "INTEGER '1'"
                ? "histogram_numeric requires numBins at least 2"
                : "histogram_numeric numBins argument must not be null");
      }
    }
  }
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
      AssertQueryBuilder(singlePlan).copyResults(pool()),
      "histogram_numeric requires numBins at least 2");

  auto partialFinalPlan =
      PlanBuilder()
          .setParseOptions(options)
          .values({input})
          .partialAggregation({}, {"histogram_numeric(c0, 1)"})
          .finalAggregation()
          .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(partialFinalPlan).copyResults(pool()),
      "histogram_numeric requires numBins at least 2");
}

TEST_F(HistogramNumericAggregateTest, windowEmptyFramesValidateNumBins) {
  auto input = makeRowVector({makeFlatVector<double>({1})});
  auto expected = makeRowVector({BaseVector::createNullConstant(
      ARRAY(ROW({"x", "y"}, DOUBLE())), 1, pool())});
  parse::ParseOptions options;
  options.parseIntegerAsBigint = false;
  for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
    SCOPED_TRACE(name);
    for (const auto numBins : {1, 2}) {
      auto plan =
          PlanBuilder()
              .setParseOptions(options)
              .values({input})
              .window({fmt::format(
                  "{}(c0, {}) over (order by c0 rows between 1 preceding and 1 preceding)",
                  name,
                  numBins)})
              .project({"w0"})
              .planNode();
      if (numBins == 1) {
        VELOX_ASSERT_THROW(
            AssertQueryBuilder(plan).copyResults(pool()),
            "histogram_numeric requires numBins at least 2");
      } else {
        AssertQueryBuilder(plan).assertResults(expected);
      }
    }
  }
}

TEST_F(
    HistogramNumericAggregateTest,
    windowRejectsNullNumBinsAtInitialization) {
  HashStringAllocator allocator(pool());
  const core::QueryConfig config({});
  const auto resultType = ARRAY(ROW({"x", "y"}, DOUBLE()));
  const std::vector<exec::WindowFunctionArg> args{
      {DOUBLE(), nullptr, 0},
      {INTEGER(),
       BaseVector::createNullConstant(INTEGER(), 1, pool()),
       std::nullopt}};
  for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
    SCOPED_TRACE(name);
    VELOX_ASSERT_USER_THROW(
        exec::WindowFunction::create(
            name, args, resultType, false, pool(), &allocator, config),
        "histogram_numeric numBins argument must not be null");
  }
}

TEST_F(HistogramNumericAggregateTest, decimalCentersUnsupported) {
  parse::ParseOptions options;
  options.parseIntegerAsBigint = false;
  const core::QueryConfig config({});
  using Step = core::AggregationNode::Step;
  const auto* functionEntry =
      exec::getAggregateFunctionEntry("histogram_numeric");
  ASSERT_NE(functionEntry, nullptr);
  for (const auto& type : {
           DECIMAL(3, 2),
           DECIMAL(18, 0),
           DECIMAL(19, 0),
           DECIMAL(38, 0),
           DECIMAL(38, 38),
       }) {
    SCOPED_TRACE(type->toString());
    auto input =
        makeRowVector({BaseVector::createNullConstant(type, 1, pool())});
    VELOX_ASSERT_USER_THROW(
        PlanBuilder()
            .setParseOptions(options)
            .values({input})
            .singleAggregation({}, {"histogram_numeric(c0, 2)"}),
        "Aggregate function signature is not supported");

    const TypePtr resultType = ARRAY(ROW({{"x", type}, {"y", DOUBLE()}}));
    for (const auto step : {Step::kSingle, Step::kPartial, Step::kFinal}) {
      const auto argTypes = step == Step::kFinal
          ? std::vector<TypePtr>{VARBINARY()}
          : std::vector<TypePtr>{type, INTEGER()};
      VELOX_ASSERT_USER_THROW(
          functionEntry->factory(
              step,
              argTypes,
              step == Step::kPartial ? VARBINARY() : resultType,
              config),
          "histogram_numeric does not support DECIMAL centers; use "
          "histogram_numeric_legacy for DOUBLE centers");
    }
  }
}

TEST_F(HistogramNumericAggregateTest, decimalLegacy) {
  const std::vector<std::pair<RowVectorPtr, RowVectorPtr>> cases{
      {makeRowVector({makeFlatVector<int64_t>({125, 250}, DECIMAL(3, 2))}),
       expectedDouble({{1.25, 1}, {2.5, 1}})},
      {makeRowVector({makeFlatVector<int64_t>(
           {999'999'999'999'999'999}, DECIMAL(18, 0))}),
       expectedDouble({{1e18, 1}})},
      {makeRowVector({makeFlatVector<int128_t>(
           {HugeInt::parse("100000000000000000000000")}, DECIMAL(38, 0))}),
       expectedDouble({{1e23, 1}})},
  };
  for (const auto& [input, expected] : cases) {
    SCOPED_TRACE(input->childAt(0)->type()->toString());
    AggregationTester(*this)
        .inputs(input)
        .aggregates({"histogram_numeric_legacy(c0, INTEGER '2')"})
        .withCompanion()
        .assertResults(expected);
  }
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
