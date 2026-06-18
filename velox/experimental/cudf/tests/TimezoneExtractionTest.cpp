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

// Class A reproducers: silent wrong results from GPU (cuDF) date/time
// extraction functions when a non-UTC session timezone is configured.
//
// On CPU, year/month/day/hour/... read the session timezone (via
// getTimeZoneFromConfig + getDateTime in velox/functions/lib/TimeUtils.h) and
// return the component of the *local* wall-clock time whenever
// adjust_timestamp_to_session_timezone is set. On GPU,
// ExtractComponentFunction::eval (velox/experimental/cudf/expression/
// ExpressionEvaluator.cpp) calls cudf::datetime::extract_datetime_component
// directly on the raw epoch column, so it always returns the UTC component and
// ignores the session timezone.
//
// Each divergence test runs the same projection twice with a non-UTC session
// timezone -- once with cuDF registered (GPU) and once without (CPU) -- and
// asserts the two results differ. The timestamps are chosen so the local value
// lands in a different calendar field than the UTC value. A matching control
// under UTC proves the divergence is specifically timezone-driven and not a
// pre-existing extraction bug.
//
// The plan/operator path is used here rather than
// CudfFunctionBaseTest::assertExpressionMatchesCpu because that lightweight
// harness evaluates the expression with finalize=false and cannot relabel a
// narrow cuDF result (e.g. extract_datetime_component returns SMALLINT) to the
// Velox BIGINT result type. The operator path applies the finalizing cast, the
// same way a real query does, and is already exercised by
// FilterProjectTest.extractTimestampComponents.
//
// These tests require a GPU and are labeled cuda_driver; they will not run in a
// CPU-only environment.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

class TimezoneExtractionTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);
  }

  void TearDown() override {
    cudf_velox::unregisterFunctions();
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  // Builds a single-row TIMESTAMP input column named ts.
  RowVectorPtr timestampInput(int64_t seconds, uint64_t nanos = 0) {
    return makeRowVector(
        {"ts"},
        {makeFlatVector<Timestamp>({Timestamp(seconds, nanos)}, TIMESTAMP())});
  }

  // Evaluates the projection with the given session timezone. cuDF is expected
  // to be registered or unregistered by the caller, which selects GPU or CPU
  // execution respectively (allowCpuFallback is false, so a registered cuDF
  // never falls back to CPU).
  RowVectorPtr project(
      const RowVectorPtr& input,
      const std::string& projection,
      const std::string& timezone) {
    auto plan =
        PlanBuilder().values({input}).project({projection}).planNode();
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kSessionTimezone, timezone)
        .config(core::QueryConfig::kAdjustTimestampToTimezone, "true")
        .copyResults(pool());
  }

  // Returns true if the single output column matches row-for-row.
  static bool resultsEqual(const RowVectorPtr& a, const RowVectorPtr& b) {
    if (a->size() != b->size()) {
      return false;
    }
    auto left = a->childAt(0);
    auto right = b->childAt(0);
    for (vector_size_t i = 0; i < a->size(); ++i) {
      if (!left->equalValueAt(right.get(), i, i)) {
        return false;
      }
    }
    return true;
  }

  // Runs the projection on GPU (cuDF registered) and CPU (cuDF unregistered)
  // under the same session timezone and asserts the results diverge. Logs both
  // results so the concrete values appear in the test output.
  void assertGpuDivergesFromCpu(
      const RowVectorPtr& input,
      const std::string& projection,
      const std::string& timezone) {
    auto gpu = project(input, projection, timezone);
    cudf_velox::unregisterCudf();
    auto cpu = project(input, projection, timezone);
    cudf_velox::registerCudf();
    LOG(INFO) << "[" << projection << " tz=" << timezone
              << "] CPU=" << cpu->childAt(0)->toString(0)
              << " GPU=" << gpu->childAt(0)->toString(0);
    EXPECT_FALSE(resultsEqual(gpu, cpu))
        << projection << " did not diverge under " << timezone
        << "; the timezone gap was not reproduced.";
  }

  // Runs the projection on GPU and CPU and asserts the results match. Used for
  // controls (UTC) and for fields that no timezone can affect.
  void assertGpuMatchesCpu(
      const RowVectorPtr& input,
      const std::string& projection,
      const std::string& timezone) {
    auto gpu = project(input, projection, timezone);
    cudf_velox::unregisterCudf();
    auto cpu = project(input, projection, timezone);
    cudf_velox::registerCudf();
    EXPECT_TRUE(resultsEqual(gpu, cpu))
        << projection << " unexpectedly diverged under " << timezone << ".";
  }
};

// America/Los_Angeles is UTC-8 in January. This instant is 2021-01-01 02:00:00
// UTC, which is 2020-12-31 18:00:00 local, so the local year/month/day/quarter/
// hour/day_of_week/day_of_year all land in the previous day, month, quarter and
// year.
constexpr int64_t kJan2021_0200Utc = 1'609'466'400;

// 2021-01-04 02:00:00 UTC is a Monday (ISO week 1 of 2021); 2021-01-03 18:00:00
// America/Los_Angeles is the preceding Sunday, which still belongs to ISO week
// 53 of week-year 2020. Used for week / year_of_week divergence.
constexpr int64_t kJan2021MondayUtc = 1'609'725'600;

// 2021-01-01 00:00:00 UTC. Asia/Kolkata is UTC+5:30, a half-hour offset, so the
// local minute differs from the UTC minute. Used for minute divergence, which a
// whole-hour offset zone like America/Los_Angeles cannot exercise.
constexpr int64_t kJan2021MidnightUtc = 1'609'459'200;

constexpr const char* kLosAngeles = "America/Los_Angeles";
constexpr const char* kKolkata = "Asia/Kolkata";

TEST_F(TimezoneExtractionTest, yearDivergesAtLocalBoundary) {
  // CPU: local year 2020. GPU: UTC year 2021.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "year(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, monthDivergesAtLocalBoundary) {
  // CPU: local month 12. GPU: UTC month 1.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "month(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dayDivergesAtLocalBoundary) {
  // CPU: local day 31. GPU: UTC day 1.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "day(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, quarterDivergesAtLocalBoundary) {
  // CPU: local quarter 4. GPU: UTC quarter 1.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "quarter(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, hourDivergesAtLocalBoundary) {
  // CPU: local hour 18. GPU: UTC hour 2.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "hour(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dayOfWeekDivergesAtLocalBoundary) {
  // CPU: local date 2020-12-31 (Thursday). GPU: UTC date 2021-01-01 (Friday).
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "day_of_week(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dowDivergesAtLocalBoundary) {
  // dow is an alias of day_of_week.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "dow(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dayOfYearDivergesAtLocalBoundary) {
  // CPU: local 2020-12-31 (day 366 of leap year 2020). GPU: UTC 2021-01-01
  // (day 1).
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "day_of_year(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, doyDivergesAtLocalBoundary) {
  // doy is an alias of day_of_year.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021_0200Utc), "doy(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, weekDivergesAtLocalBoundary) {
  // CPU: local 2021-01-03 (Sunday) is ISO week 53. GPU: UTC 2021-01-04
  // (Monday) is ISO week 1.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021MondayUtc), "week(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, weekOfYearDivergesAtLocalBoundary) {
  // week_of_year is an alias of week.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021MondayUtc), "week_of_year(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, yearOfWeekDivergesAtLocalBoundary) {
  // CPU: local 2021-01-03 belongs to week-year 2020. GPU: UTC 2021-01-04
  // belongs to week-year 2021.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021MondayUtc), "year_of_week(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, yowDivergesAtLocalBoundary) {
  // yow is an alias of year_of_week.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021MondayUtc), "yow(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, minuteDivergesForHalfHourOffsetZone) {
  // Asia/Kolkata is UTC+5:30. CPU: local minute 30. GPU: UTC minute 0.
  assertGpuDivergesFromCpu(
      timestampInput(kJan2021MidnightUtc), "minute(ts)", kKolkata);
}

// second and millisecond cannot diverge: every IANA timezone offset is a whole
// number of minutes, so sub-minute fields are identical in UTC and in any
// session timezone. On CPU, second/millisecond are additionally computed
// without applying the session timezone at all (getDateTime(timestamp,
// nullptr)). Assert that GPU still matches CPU under a non-UTC timezone to
// document the boundary of the gap.
TEST_F(TimezoneExtractionTest, secondUnaffectedByTimezone) {
  assertGpuMatchesCpu(
      timestampInput(kJan2021_0200Utc, 123'000'000), "second(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, millisecondUnaffectedByTimezone) {
  assertGpuMatchesCpu(
      timestampInput(kJan2021_0200Utc, 123'000'000),
      "millisecond(ts)",
      kLosAngeles);
}

// Control: under UTC the GPU (which always computes in UTC) must match the CPU
// for every extraction function, proving the divergences above are caused
// specifically by the session timezone and not by an unrelated extraction bug.
TEST_F(TimezoneExtractionTest, allComponentsMatchUnderUtc) {
  auto boundary = timestampInput(kJan2021_0200Utc, 123'000'000);
  auto monday = timestampInput(kJan2021MondayUtc);

  for (const auto& projection :
       {"year(ts)",
        "month(ts)",
        "day(ts)",
        "quarter(ts)",
        "hour(ts)",
        "minute(ts)",
        "second(ts)",
        "millisecond(ts)",
        "day_of_week(ts)",
        "dow(ts)",
        "day_of_year(ts)",
        "doy(ts)"}) {
    SCOPED_TRACE(projection);
    assertGpuMatchesCpu(boundary, projection, "UTC");
  }
  for (const auto& projection :
       {"week(ts)", "week_of_year(ts)", "year_of_week(ts)", "yow(ts)"}) {
    SCOPED_TRACE(projection);
    assertGpuMatchesCpu(monday, projection, "UTC");
  }
}

} // namespace
