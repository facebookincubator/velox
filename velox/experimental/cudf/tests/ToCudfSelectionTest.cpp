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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Time.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class ToCudfSelectionTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  bool wasCudfAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType.starts_with("CudfAggregation") ||
            operatorStats.operatorType.starts_with("CudfGroupby") ||
            operatorStats.operatorType.starts_with("CudfReduce") ||
            operatorStats.operatorType.starts_with("CudfDistinct")) {
          return true;
        }
      }
    }
    return false;
  }

  bool wasDefaultHashAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == OperatorType::kAggregation) {
          return true;
        }
      }
    }
    return false;
  }

  bool wasCudfFilterProjectUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfFilterProject") {
          return true;
        }
      }
    }
    return false;
  }

  bool wasDefaultFilterProjectUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == OperatorType::kFilterProject) {
          return true;
        }
      }
    }
    return false;
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(),
           DOUBLE(),
           VARCHAR()})};
};

// A datetime format the GPU cannot reproduce exactly must leave the projection
// on the CPU operator rather than return a different answer. Selection is the
// only thing that can see this: the value suites force GPU evaluation, so they
// are blind to a fallback.
//
// Joda treats a token's run length as a minimum width while cuDF's specifiers
// are fixed, so 'y-M-d' prints 2026-1-2 on CPU and pads on the GPU.
TEST_F(ToCudfSelectionTest, formatDatetimeVariableFieldWidthFallsBack) {
  auto input = makeRowVector(
      {"event_ts"},
      {makeFlatVector<int64_t>(
          {pack(1'609'466'400'000, tz::getTimeZoneID("Asia/Kolkata"))},
          TIMESTAMP_WITH_TIME_ZONE())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"format_datetime(event_ts, 'y-M-d') AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_FALSE(wasCudfFilterProjectUsed(task));
  ASSERT_TRUE(wasDefaultFilterProjectUsed(task));
}

// A width the GPU does render identically stays on it, so the gate above is not
// simply refusing every format.
TEST_F(ToCudfSelectionTest, formatDatetimeFixedFieldWidthUsesCudf) {
  auto input = makeRowVector(
      {"event_ts"},
      {makeFlatVector<int64_t>(
          {pack(1'609'466'400'000, tz::getTimeZoneID("Asia/Kolkata"))},
          TIMESTAMP_WITH_TIME_ZONE())});

  auto plan =
      PlanBuilder()
          .values({input})
          .project({"format_datetime(event_ts, 'yyyy-MM-dd') AS result"})
          .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

// parse_datetime has three further formats it cannot handle: a two-digit year,
// because cuDF pivots at 69 and Joda at 70, so '69' would parse to 1969 here
// and 2069 on CPU; a month or weekday name, which cuDF's parser rejects
// outright; and a zone name, from which no numeric offset can be recovered.
TEST_F(ToCudfSelectionTest, parseDatetimeUnsupportedFormatsFallBack) {
  // Each format needs text it actually matches: falling back means CPU parses
  // for real, and a mismatched input would fail the query rather than the
  // selection assertion.
  for (const auto& [format, text] :
       std::vector<std::pair<std::string, std::string>>{
           {"yy-MM-dd", "21-01-02"},
           {"EEE, dd MMM yyyy", "Sat, 02 Jan 2021"},
           {"yyyy-MM-dd z", "2021-01-02 UTC"},
       }) {
    SCOPED_TRACE(format);
    auto input = makeRowVector({"text"}, {makeFlatVector<std::string>({text})});
    auto plan =
        PlanBuilder()
            .values({input})
            .project({"parse_datetime(text, '" + format + "') AS result"})
            .planNode();

    std::shared_ptr<Task> task;
    AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

    ASSERT_FALSE(wasCudfFilterProjectUsed(task));
    ASSERT_TRUE(wasDefaultFilterProjectUsed(task));
  }
}

TEST_F(ToCudfSelectionTest, supportedPrestoDateAddDateUsesCudf) {
  auto input = makeRowVector(
      {"amount", "event_date"},
      {makeFlatVector<int64_t>({1, 2, -1, 13}),
       makeFlatVector<int32_t>(
           {DATE()->toDays("2020-01-31"),
            DATE()->toDays("2020-02-29"),
            DATE()->toDays("2020-03-01"),
            DATE()->toDays("2020-12-31")},
           DATE())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_add('month', amount, event_date) AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

TEST_F(ToCudfSelectionTest, prestoDateAddVariableUnitFallsBack) {
  auto input = makeRowVector(
      {"unit", "amount", "event_date"},
      {makeFlatVector<std::string>({"day", "week", "month", "year"}),
       makeFlatVector<int64_t>({1, 2, -1, 13}),
       makeFlatVector<int32_t>(
           {DATE()->toDays("2020-01-31"),
            DATE()->toDays("2020-02-29"),
            DATE()->toDays("2020-03-01"),
            DATE()->toDays("2020-12-31")},
           DATE())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_add(unit, amount, event_date) AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_FALSE(wasCudfFilterProjectUsed(task));
  ASSERT_TRUE(wasDefaultFilterProjectUsed(task));
}

TEST_F(ToCudfSelectionTest, prestoDateAddTimestampFallsBack) {
  auto input = makeRowVector(
      {"amount", "event_ts"},
      {makeFlatVector<int64_t>({1, 2, -1, 13}),
       makeFlatVector<Timestamp>(
           {Timestamp(1675209600, 0),
            Timestamp(1677628800, 0),
            Timestamp(1680307200, 0),
            Timestamp(1703980800, 0)},
           TIMESTAMP())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_add('day', amount, event_ts) AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_FALSE(wasCudfFilterProjectUsed(task));
  ASSERT_TRUE(wasDefaultFilterProjectUsed(task));
}

TEST_F(ToCudfSelectionTest, prestoDateTruncTimestampAdjustTimezoneUsesCudf) {
  // date_trunc(timestamp) is timezone-aware on GPU, so it runs on cuDF even
  // when adjust_timestamp_to_session_timezone is enabled.
  auto input = makeRowVector(
      {"event_ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1767314700, 0), Timestamp(1767318300, 0)}, TIMESTAMP())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_trunc('hour', event_ts) AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan)
      .config("cudf.enabled", true)
      .config(QueryConfig::kSessionTimezone, "Asia/Kolkata")
      .config(QueryConfig::kAdjustTimestampToTimezone, "true")
      .countResults(task);

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

TEST_F(ToCudfSelectionTest, prestoDateTruncTimestampWithTimeZoneUsesCudf) {
  // date_trunc(timestamp with time zone) is evaluated on GPU (per-row embedded
  // zone), so the plan runs on cuDF rather than falling back.
  auto input = makeRowVector(
      {"c0"},
      {makeFlatVector<int64_t>(
          {pack(1'736'971'261'123, tz::getTimeZoneID("America/Los_Angeles")),
           pack(1'736'971'261'123, tz::getTimeZoneID("Asia/Kolkata"))},
          TIMESTAMP_WITH_TIME_ZONE())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_trunc('day', c0) AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

TEST_F(ToCudfSelectionTest, prestoDateTruncSubHourAdjustTimezoneUsesCudf) {
  auto input = makeRowVector(
      {"event_ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1767314700, 0), Timestamp(1767318300, 0)}, TIMESTAMP())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project(
                      {"date_trunc('second', event_ts) AS second",
                       "date_trunc('minute', event_ts) AS minute"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan)
      .config("cudf.enabled", true)
      .config(QueryConfig::kSessionTimezone, "Asia/Kolkata")
      .config(QueryConfig::kAdjustTimestampToTimezone, "true")
      .countResults(task);

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

TEST_F(
    ToCudfSelectionTest,
    nestedPrestoDateTruncTimestampAdjustTimezoneUsesCudf) {
  auto input = makeRowVector(
      {"event_ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1767314700, 0), Timestamp(1767318300, 0)}, TIMESTAMP())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_trunc('hour', event_ts) = "
                            "TIMESTAMP '2026-01-02 00:00:00' AS result"})
                  .planNode();

  std::shared_ptr<Task> cudfTask;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(cudfTask);
  ASSERT_TRUE(wasCudfFilterProjectUsed(cudfTask));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(cudfTask));

  // A nested timezone-sensitive date_trunc(timestamp) also stays on cuDF under
  // adjust_timestamp_to_session_timezone now that it is timezone-aware.
  std::shared_ptr<Task> adjustTask;
  AssertQueryBuilder(plan)
      .config("cudf.enabled", true)
      .config(QueryConfig::kSessionTimezone, "Asia/Kolkata")
      .config(QueryConfig::kAdjustTimestampToTimezone, "true")
      .countResults(adjustTask);

  ASSERT_TRUE(wasCudfFilterProjectUsed(adjustTask));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(adjustTask));
}

TEST_F(ToCudfSelectionTest, prestoDateTruncDateAdjustTimezoneUsesCudf) {
  auto input = makeRowVector(
      {"event_date"},
      {makeFlatVector<int32_t>(
          {DATE()->toDays("2026-01-02"), DATE()->toDays("2026-01-03")},
          DATE())});

  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_trunc('day', event_date) AS result"})
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan)
      .config("cudf.enabled", true)
      .config(QueryConfig::kSessionTimezone, "Asia/Kolkata")
      .config(QueryConfig::kAdjustTimestampToTimezone, "true")
      .countResults(task);

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

// now() reaches the GPU as a constant, never as a call. It takes no arguments,
// so the optimizer's "all inputs are constant" test is vacuously true and
// expression::optimize always constant folds it (ExprOptimizer.cpp:121-130);
// every operator optimizes before compiling. The projection CudfFilterProject
// compiles is therefore a TIMESTAMP WITH TIME ZONE constant.
//
// Pin both halves of that, because they can fail independently: the folded
// value must still be the session start time packed with the session zone, and
// a constant-only projection of a custom type must still be claimed by the GPU
// operator. Value parity alone would keep passing if the projection silently
// fell back to CPU.
TEST_F(ToCudfSelectionTest, nowFoldsToConstantAndStaysOnGpu) {
  constexpr int64_t kStartMs = 1'609'466'400'000; // 2021-01-01T02:00:00 UTC.
  auto input =
      makeRowVector({"amount"}, {makeFlatVector<int64_t>({1, 2, 3, 4})});

  auto plan =
      PlanBuilder().values({input}).project({"now() AS result"}).planNode();

  std::shared_ptr<Task> task;
  auto results =
      AssertQueryBuilder(plan)
          .config("cudf.enabled", true)
          .config(QueryConfig::kSessionTimezone, "America/Los_Angeles")
          .config(QueryConfig::kAdjustTimestampToTimezone, "true")
          .config(QueryConfig::kSessionStartTime, std::to_string(kStartMs))
          .copyResults(pool(), task);

  ASSERT_EQ(results->size(), input->size());
  const auto& resultColumn = results->childAt(0);
  ASSERT_TRUE(isTimestampWithTimeZoneType(resultColumn->type()))
      << "now() must produce TIMESTAMP WITH TIME ZONE, got "
      << resultColumn->type()->toString();
  const auto packed = resultColumn->as<SimpleVector<int64_t>>()->valueAt(0);
  EXPECT_EQ(unpackMillisUtc(packed), kStartMs);
  EXPECT_EQ(unpackZoneKeyId(packed), tz::getTimeZoneID("America/Los_Angeles"));

  ASSERT_TRUE(wasCudfFilterProjectUsed(task));
  ASSERT_FALSE(wasDefaultFilterProjectUsed(task));
}

// now() must fail exactly where CPU fails, and the constant folding above does
// not weaken that. CPU's CurrentTimestampFunction::initialize fails when
// getTimeZoneFromConfig returns null, which happens when
// adjust_timestamp_to_session_timezone is off or the session timezone is empty.
// Folding runs that same initialize, so the query still fails for both
// configurations. Pin it here: a GPU path that produced a value where CPU fails
// would be a silently wrong result, and this is the only check that would catch
// it.
TEST_F(ToCudfSelectionTest, nowWithoutAdjustedSessionTimezoneRejected) {
  auto input =
      makeRowVector({"amount"}, {makeFlatVector<int64_t>({1, 2, 3, 4})});

  auto plan =
      PlanBuilder().values({input}).project({"now() AS result"}).planNode();

  // Adjustment enabled, but no session timezone to adjust to.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .config("cudf.enabled", true)
          .config(QueryConfig::kAdjustTimestampToTimezone, "true")
          .copyResults(pool()),
      "Timezone cannot be null");

  // Session timezone present, but adjustment disabled.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan)
          .config("cudf.enabled", true)
          .config(QueryConfig::kSessionTimezone, "America/Los_Angeles")
          .config(QueryConfig::kAdjustTimestampToTimezone, "false")
          .copyResults(pool()),
      "Timezone cannot be null");
}

// Test supported aggregation should use CUDF
TEST_F(ToCudfSelectionTest, supportedAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "count(c2)", "min(c3)", "max(c4)", "avg(c5)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, sum(c1), count(c2), min(c3), max(c4), avg(c5) FROM tmp GROUP BY c0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation should fall back to CPU
TEST_F(ToCudfSelectionTest, unsupportedAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"stddev(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, stddev(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test mixed supported/unsupported should fall back
TEST_F(ToCudfSelectionTest, mixedSupportFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT c0, sum(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported global aggregation should use CUDF
TEST_F(ToCudfSelectionTest, supportedGlobalAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"sum(c1)", "count(c2)", "max(c3)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT sum(c1), count(c2), max(c3) FROM tmp");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported global aggregation should fall back
TEST_F(ToCudfSelectionTest, unsupportedGlobalAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"stddev(c1)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT stddev(c1) FROM tmp");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported grouping key expressions should use CUDF
TEST_F(ToCudfSelectionTest, supportedGroupingKeyExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project(
                      {"c0",
                       "c1",
                       "c2",
                       "c3",
                       "c4",
                       "c5",
                       "c6",
                       "c0 + c1 as key1",
                       "length(c6) as key2"})
                  .aggregation(
                      {"key1", "key2"},
                      {"sum(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0 + c1, length(c6), sum(c2) FROM tmp GROUP BY c0 + c1, length(c6)");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation functions should fall back
TEST_F(ToCudfSelectionTest, unsupportedAggregationFunctionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT c0, sum(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test complex grouping key expressions should fall back
TEST_F(ToCudfSelectionTest, complexGroupingKeyExpressionsFallsBack) {
  // Use a deterministic input vector.
  // Input: c0=[1,2,1,2], c2=[100,200,300,400]
  // After grouping by to_big_endian_64(c0):
  // - Group 1 (c0=1): sum(c2) = 100 + 300 = 400
  // - Group 2 (c0=2): sum(c2) = 200 + 400 = 600
  auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 1, 2}), // c0
      makeFlatVector<int16_t>({10, 20, 30, 40}), // c1
      makeFlatVector<int32_t>({100, 200, 300, 400}), // c2
  });

  auto plan =
      PlanBuilder()
          .values({input})
          .project({"c0", "c1", "c2", "to_big_endian_64(c0) AS complex_key"})
          .aggregation(
              {"complex_key"},
              {"sum(c2)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported aggregation input expressions should use CUDF
TEST_F(ToCudfSelectionTest, supportedAggregationInputExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "max(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), max(c2) FROM tmp GROUP BY c0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation input expressions should fall back
TEST_F(ToCudfSelectionTest, unsupportedAggregationInputExpressionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT c0, sum(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Non-count constant aggregates are not supported by cuDF aggregation.
TEST_F(ToCudfSelectionTest, nonCountConstantAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan =
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {}, {"sum(1)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT sum(1) FROM tmp");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test zero-column count(*) should stay on GPU.
TEST_F(ToCudfSelectionTest, zeroColumnCountStarUsesCudf) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .filter("c0 > 0")
          .project({})
          .aggregation(
              {}, {"count(*)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test zero-column count(constant) should use cudf GPU.
TEST_F(ToCudfSelectionTest, zeroColumnCountConstantUsesGpu) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .filter("c0 > 0")
          .project({})
          .aggregation(
              {}, {"count(1)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT count(1) FROM tmp WHERE c0 > 0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Count-only aggregation plans (single and partial/final) should use
// CudfAggregation.
TEST_F(ToCudfSelectionTest, countAggregatesOnlyUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto assertCountUsesCudf = [this](
                                 const core::PlanNodePtr& plan,
                                 const std::string& duckSql,
                                 const char* caseLabel) {
    SCOPED_TRACE(caseLabel);
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config("cudf.enabled", true)
                    .plan(plan)
                    .assertResults(duckSql);
    ASSERT_TRUE(wasCudfAggregationUsed(task));
    ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
  };

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {}, {"count(*)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode(),
      "SELECT count(*) FROM tmp",
      "global count(*) kSingle");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {},
              {"count(c0)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode(),
      "SELECT count(c0) FROM tmp",
      "global count(c0) kSingle");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {"c0"},
              {"count(c2)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode(),
      "SELECT c0, count(c2) FROM tmp GROUP BY c0",
      "group by c0, count(c2) kSingle");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .partialAggregation({}, {"count(*)"})
          .finalAggregation()
          .planNode(),
      "SELECT count(*) FROM tmp",
      "global count(*) partial+final");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .partialAggregation({"c0"}, {"count(c2)"})
          .finalAggregation()
          .planNode(),
      "SELECT c0, count(c2) FROM tmp GROUP BY c0",
      "group by c0, count(c2) partial+final");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .partialAggregation({"c0"}, {"count(*)"})
          .finalAggregation()
          .planNode(),
      "SELECT c0, count(*) FROM tmp GROUP BY c0",
      "group by c0, count(*) partial+final");
}

// Test count(NULL) runs on cudf GPU (returns 0).
TEST_F(ToCudfSelectionTest, countNullAggregationUsesGpu) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"count(null)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT count(NULL) FROM tmp");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test CUDF disabled should always use regular aggregation
TEST_F(ToCudfSelectionTest, cudfDisabledUsesRegularAggregation) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "count(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", false)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), count(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

} // namespace facebook::velox::exec::test
