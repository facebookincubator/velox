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

// Class A correctness tests: GPU (cuDF) date/time extraction functions must
// honor the session timezone exactly as CPU does. These reproduce the missing
// timezone support by failing while it is absent.
//
// On CPU, year/month/day/hour/... read the session timezone (via
// getTimeZoneFromConfig + getDateTime in velox/functions/lib/TimeUtils.h) and
// return the component of the *local* wall-clock time whenever
// adjust_timestamp_to_session_timezone is set. On GPU,
// ExtractComponentFunction::eval (velox/experimental/cudf/expression/
// ExpressionEvaluator.cpp) currently calls
// cudf::datetime::extract_datetime_component directly on the raw epoch column,
// so it always returns the UTC component and ignores the session timezone.
//
// Each test runs the same projection twice under a non-UTC session timezone --
// once with cuDF registered (GPU) and once without (CPU) -- and asserts the two
// results match. The timestamps are chosen so the local value lands in a
// different calendar field than the UTC value, so the assertion holds only once
// the GPU path applies the session-timezone offset before extracting the field.
// A matching control under UTC proves a failure is specifically timezone-driven
// and not a pre-existing extraction bug.
//
// IMPORTANT: these are test-driven-development tests for the *target* behavior,
// so they FAIL today and pass once the gap is closed. Each test below fails
// with the local-vs-UTC mismatch noted in its comment until the GPU path
// applies the session-timezone offset before extraction. A red test here means
// the timezone gap is still open; do not "fix" a failure by weakening the
// assertion. The UTC controls and the sub-minute fields already pass and guard
// the harness.
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
      std::string_view timezone) {
    auto plan = PlanBuilder().values({input}).project({projection}).planNode();
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kSessionTimezone, std::string(timezone))
        .config(core::QueryConfig::kAdjustTimestampToTimezone, "true")
        .copyResults(pool());
  }

  // Runs the projection on GPU (cuDF registered) and CPU (cuDF unregistered)
  // under the same session timezone and asserts the single output columns are
  // equal. This is the target behavior for every extraction function: the GPU
  // result must equal the CPU result regardless of the session timezone.
  void assertGpuMatchesCpu(
      const RowVectorPtr& input,
      const std::string& projection,
      std::string_view timezone) {
    auto gpu = project(input, projection, timezone);
    cudf_velox::unregisterCudf();
    auto cpu = project(input, projection, timezone);
    cudf_velox::registerCudf();
    SCOPED_TRACE(
        projection + " under session timezone " + std::string(timezone));
    facebook::velox::test::assertEqualVectors(cpu->childAt(0), gpu->childAt(0));
  }
};

// America/Los_Angeles is UTC-8 in January. This instant is 2021-01-01 02:00:00
// UTC, which is 2020-12-31 18:00:00 local, so the local year/month/day/quarter/
// hour/day_of_week/day_of_year all land in the previous day, month, quarter and
// year.
constexpr int64_t kJan2021At0200Utc = 1'609'466'400;

// 2021-01-04 02:00:00 UTC is a Monday (ISO week 1 of 2021); 2021-01-03 18:00:00
// America/Los_Angeles is the preceding Sunday, which still belongs to ISO week
// 53 of week-year 2020. Used for the week / year_of_week fields.
constexpr int64_t kJan2021MondayUtc = 1'609'725'600;

// 2021-01-01 00:00:00 UTC. Asia/Kolkata is UTC+5:30, a half-hour offset, so the
// local minute differs from the UTC minute. Used for the minute field, which a
// whole-hour offset zone like America/Los_Angeles cannot exercise.
constexpr int64_t kJan2021MidnightUtc = 1'609'459'200;

constexpr std::string_view kLosAngeles = "America/Los_Angeles";
constexpr std::string_view kKolkata = "Asia/Kolkata";

TEST_F(TimezoneExtractionTest, yearHonorsSessionTimezone) {
  // Expect local year 2020; GPU currently returns UTC year 2021.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "year(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, monthHonorsSessionTimezone) {
  // Expect local month 12; GPU currently returns UTC month 1.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "month(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dayHonorsSessionTimezone) {
  // Expect local day 31; GPU currently returns UTC day 1.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "day(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, quarterHonorsSessionTimezone) {
  // Expect local quarter 4; GPU currently returns UTC quarter 1.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "quarter(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, hourHonorsSessionTimezone) {
  // Expect local hour 18; GPU currently returns UTC hour 2.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "hour(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dayOfWeekHonorsSessionTimezone) {
  // Expect local 2020-12-31 (Thursday); GPU currently returns UTC 2021-01-01
  // (Friday).
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "day_of_week(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dowHonorsSessionTimezone) {
  // dow is an alias of day_of_week.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "dow(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, dayOfYearHonorsSessionTimezone) {
  // Expect local 2020-12-31 (day 366 of leap year 2020); GPU currently returns
  // UTC 2021-01-01 (day 1).
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "day_of_year(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, doyHonorsSessionTimezone) {
  // doy is an alias of day_of_year.
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc), "doy(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, weekHonorsSessionTimezone) {
  // Expect local 2021-01-03 (Sunday), ISO week 53; GPU currently returns UTC
  // 2021-01-04 (Monday), ISO week 1.
  assertGpuMatchesCpu(
      timestampInput(kJan2021MondayUtc), "week(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, weekOfYearHonorsSessionTimezone) {
  // week_of_year is an alias of week.
  assertGpuMatchesCpu(
      timestampInput(kJan2021MondayUtc), "week_of_year(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, yearOfWeekHonorsSessionTimezone) {
  // Expect local 2021-01-03 to belong to week-year 2020; GPU currently returns
  // UTC 2021-01-04, week-year 2021.
  assertGpuMatchesCpu(
      timestampInput(kJan2021MondayUtc), "year_of_week(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, yowHonorsSessionTimezone) {
  // yow is an alias of year_of_week.
  assertGpuMatchesCpu(
      timestampInput(kJan2021MondayUtc), "yow(ts)", kLosAngeles);
}

TEST_F(TimezoneExtractionTest, minuteHonorsHalfHourOffsetZone) {
  // Asia/Kolkata is UTC+5:30. Expect local minute 30; GPU currently returns UTC
  // minute 0.
  assertGpuMatchesCpu(
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
      timestampInput(kJan2021At0200Utc, 123'000'000),
      "second(ts)",
      kLosAngeles);
}

TEST_F(TimezoneExtractionTest, millisecondUnaffectedByTimezone) {
  assertGpuMatchesCpu(
      timestampInput(kJan2021At0200Utc, 123'000'000),
      "millisecond(ts)",
      kLosAngeles);
}

// Control: under UTC the GPU (which always computes in UTC) already matches the
// CPU for every extraction function. This passes today and guards the harness:
// a failure here would point to an extraction bug unrelated to the session
// timezone, isolating it from the timezone-driven failures above.
TEST_F(TimezoneExtractionTest, allComponentsMatchUnderUtc) {
  auto boundary = timestampInput(kJan2021At0200Utc, 123'000'000);
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
