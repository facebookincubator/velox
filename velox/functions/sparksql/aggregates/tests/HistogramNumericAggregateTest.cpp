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
#include <limits>
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
#include "velox/vector/DecodedVector.h"

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

TEST_F(HistogramNumericAggregateTest, rejectsMismatchedResultCenterType) {
  const core::QueryConfig config({});
  const auto* canonical = exec::getAggregateFunctionEntry("histogram_numeric");
  const auto* legacy =
      exec::getAggregateFunctionEntry("histogram_numeric_legacy");
  ASSERT_NE(canonical, nullptr);
  ASSERT_NE(legacy, nullptr);
  const std::vector<std::pair<TypePtr, TypePtr>> cases{
      {INTEGER(), DOUBLE()},
      {DATE(), INTEGER()},
      {TIMESTAMP_UTC(), TIMESTAMP()},
  };
  for (const auto& [inputType, centerType] : cases) {
    for (const auto step : {
             core::AggregationNode::Step::kSingle,
             core::AggregationNode::Step::kPartial,
         }) {
      SCOPED_TRACE(inputType->toString());
      const auto resultType = ARRAY(ROW({{"x", centerType}, {"y", DOUBLE()}}));
      VELOX_ASSERT_USER_THROW(
          canonical->factory(step, {inputType, INTEGER()}, resultType, config),
          fmt::format(
              "histogram_numeric result center type mismatch: expected {}, got {}",
              inputType->toString(),
              centerType->toString()));
    }
  }
  VELOX_ASSERT_USER_THROW(
      legacy->factory(
          core::AggregationNode::Step::kSingle,
          {INTEGER(), INTEGER()},
          ARRAY(ROW({{"x", INTEGER()}, {"y", DOUBLE()}})),
          config),
      "histogram_numeric result center type mismatch: expected DOUBLE, got INTEGER");
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

TEST_F(HistogramNumericAggregateTest, bigintCenterLimitsAndRounding) {
  const std::vector<std::pair<int64_t, int64_t>> cases{
      {std::numeric_limits<int64_t>::max(),
       std::numeric_limits<int64_t>::max()},
      {std::numeric_limits<int64_t>::min(),
       std::numeric_limits<int64_t>::min()},
      {9'007'199'254'740'993, 9'007'199'254'740'992},
      {9'007'199'254'740'995, 9'007'199'254'740'996},
      {-9'007'199'254'740'993, -9'007'199'254'740'992},
  };
  for (const auto& [value, center] : cases) {
    SCOPED_TRACE(value);
    auto input = makeRowVector({makeFlatVector<int64_t>({value})});
    auto expected = makeRowVector({makeArrayOfRowVector(
        std::vector<std::vector<std::optional<std::tuple<int64_t, double>>>>{
            {std::tuple<int64_t, double>{center, 1}}},
        ROW({{"x", BIGINT()}, {"y", DOUBLE()}}))});
    AggregationTester(*this)
        .inputs(input)
        .aggregates({"histogram_numeric(c0, INTEGER '2')"})
        .withCompanion()
        .assertResults(expected);
    assertSingle(
        input,
        "histogram_numeric_legacy(c0, 2)",
        expectedDouble({{static_cast<double>(value), 1}}));
  }
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

TEST_F(HistogramNumericAggregateTest, timestampCentersAtMicrosecondLimits) {
  for (const auto& type : {TIMESTAMP(), TIMESTAMP_UTC()}) {
    for (const auto& value : {
             Timestamp(-9'223'372'036'855, 224'192'000),
             Timestamp(9'223'372'036'854, 775'807'000),
             Timestamp(-1, 998'999'000),
             Timestamp(0, 1'001'000),
         }) {
      SCOPED_TRACE(type->toString());
      SCOPED_TRACE(
          fmt::format(
              "{} seconds, {} nanos", value.getSeconds(), value.getNanos()));
      auto centers = makeFlatVector<Timestamp>({value}, type);
      auto input = makeRowVector({centers});
      auto expected = makeRowVector({makeArrayVector(
          {0},
          makeRowVector({"x", "y"}, {centers, makeFlatVector<double>({1})}))});
      AggregationTester(*this)
          .inputs(input)
          .aggregates({"histogram_numeric(c0, INTEGER '2')"})
          .withCompanion()
          .assertResults(expected);
      assertSingle(
          input,
          "histogram_numeric_legacy(c0, 2)",
          expectedDouble({{static_cast<double>(value.toMicros()), 1}}));
    }
  }
}

TEST_F(HistogramNumericAggregateTest, rejectsInvalidTimestampInputs) {
  const std::vector<std::pair<Timestamp, std::string>> cases{
      {Timestamp(0, 1),
       "histogram_numeric timestamp requires microsecond precision"},
      {Timestamp(-1, 999'999'999),
       "histogram_numeric timestamp requires microsecond precision"},
      {Timestamp(-9'223'372'036'855, 224'191'000),
       "Could not convert Timestamp(-9223372036855, 224191000) to microseconds"},
      {Timestamp(9'223'372'036'854, 775'808'000),
       "Could not convert Timestamp(9223372036854, 775808000) to microseconds"},
  };
  for (const auto& type : {TIMESTAMP(), TIMESTAMP_UTC()}) {
    for (const auto& [value, message] : cases) {
      const auto input =
          makeRowVector({makeFlatVector<Timestamp>({value}, type)});
      for (const auto* name :
           {"histogram_numeric", "histogram_numeric_legacy"}) {
        SCOPED_TRACE(name);
        const auto plan =
            PlanBuilder()
                .values({input})
                .singleAggregation(
                    {}, {fmt::format("{}(c0, INTEGER '2')", name)})
                .planNode();
        VELOX_ASSERT_USER_THROW(
            AssertQueryBuilder(plan).copyResults(pool()), message);
      }
    }
  }
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
                ? "numBins must be at least 2"
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
      "numBins must be at least 2");

  auto partialFinalPlan =
      PlanBuilder()
          .setParseOptions(options)
          .values({input})
          .partialAggregation({}, {"histogram_numeric(c0, 1)"})
          .finalAggregation()
          .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(partialFinalPlan).copyResults(pool()),
      "numBins must be at least 2");
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
            "numBins must be at least 2");
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

TEST_F(HistogramNumericAggregateTest, decimalLegacyNormalizationBoundaries) {
  struct TestCase {
    int32_t precision;
    int32_t scale;
    std::string unscaled;
    double expected;
  };
  const std::vector<TestCase> cases{
      {18, 18, "999999999999999999", 1.0},
      {18, 18, "-1", -1e-18},
      {3, 2, "10", 0.1},
      {1, 1, "0", 0.0},
      {19, 0, "9999999999999999999", 1e19},
      {19, 19, "9999999999999999999", 1.0},
      {38, 0, "99999999999999999999999999999999999999", 1e38},
      {38, 38, "99999999999999999999999999999999999999", 1.0},
      {38, 38, "-99999999999999999999999999999999999999", -1.0},
      {38, 38, "-1", -1e-38},
      {21,
       20,
       "100000000000000011103",
       std::bit_cast<double>(uint64_t{0x3ff0000000000001})},
      {21,
       20,
       "-100000000000000011103",
       std::bit_cast<double>(uint64_t{0xbff0000000000001})},
      {21, 20, "100000000000000011102", 1.0},
      {16, 0, "9007199254740992", 0x1p53},
      {16, 0, "-9007199254740992", -0x1p53},
      {23, 22, "9007199254740992", 9.007199254740992e-7},
      {24, 23, "9007199254740992", 9.007199254740992e-8},
      {23, 22, "9007199254740993", 9.007199254740993e-7},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(
        fmt::format(
            "DECIMAL({}, {}): {}", test.precision, test.scale, test.unscaled));
    const auto type = DECIMAL(test.precision, test.scale);
    const auto unscaled = HugeInt::parse(test.unscaled);
    const VectorPtr values = type->isShortDecimal()
        ? VectorPtr(
              makeFlatVector<int64_t>({static_cast<int64_t>(unscaled)}, type))
        : makeFlatVector<int128_t>({unscaled}, type);
    const auto input = makeRowVector({values});
    AggregationTester(*this)
        .inputs(input)
        .aggregates({"histogram_numeric_legacy(c0, INTEGER '2')"})
        .withCompanion()
        .assertResults(expectedDouble({{test.expected, 1}}));
    const auto plan = PlanBuilder()
                          .values({input})
                          .singleAggregation(
                              {}, {"histogram_numeric_legacy(c0, INTEGER '2')"})
                          .planNode();
    const auto result = AssertQueryBuilder(plan).copyResults(pool());
    const auto* bins = result->childAt(0)->as<ArrayVector>();
    ASSERT_FALSE(bins->isNullAt(0));
    ASSERT_EQ(bins->sizeAt(0), 1);
    const auto* rows = bins->elements()->as<RowVector>();
    DecodedVector centers(*rows->childAt(0));
    EXPECT_EQ(
        std::bit_cast<uint64_t>(centers.valueAt<double>(bins->offsetAt(0))),
        std::bit_cast<uint64_t>(test.expected));
  }
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
