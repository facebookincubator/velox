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

// Class B correctness tests: GPU (cuDF) evaluation of the TIMESTAMP WITH TIME
// ZONE function family must match CPU. These reproduce the missing timezone
// support by failing while it is absent.
//
// None of these functions are implemented on the GPU path today. Each works on
// CPU (it is registered by registerAllScalarFunctions), but forcing GPU
// evaluation currently throws from FunctionExpression::eval:
//
//   "Unsupported expression for recursive evaluation: <name>"
//   (velox/experimental/cudf/expression/ExpressionEvaluator.cpp)
//
// This is true both for functions that produce a TIMESTAMP WITH TIME ZONE from
// plain double/varchar inputs and for functions that consume a TIMESTAMP WITH
// TIME ZONE column. The input conversion (Velox -> cuDF) is *not* the failure
// site: cuDF has no TIMESTAMP WITH TIME ZONE type, but a TIMESTAMP WITH TIME
// ZONE column is carried as its physical BIGINT and round-trips without error
// (see timestampWithTimeZoneColumnPreservedThroughGpu). The failure surfaces
// only when an unsupported function is evaluated.
//
// CudfFunctionBaseTest::evaluate forces GPU execution (it does not consult
// allowCpuFallback), and assertExpressionMatchesCpu compares that GPU result to
// the CPU result -- so until a function is implemented the GPU evaluation
// throws and the test fails.
//
// IMPORTANT: these are test-driven-development tests for the *target* behavior,
// so they FAIL today and pass once the gap is closed. Each asserts the GPU
// result equals CPU; until the function (and a GPU TIMESTAMP WITH TIME ZONE
// representation) is implemented, GPU evaluation throws and the test is red. A
// red test here means the function is still unsupported on GPU; do not "fix" a
// failure by weakening the assertion. The one exception is
// timestampWithTimeZoneColumnPreservedThroughGpu, which passes today: a plain
// passthrough never touches timezone semantics and the physical BIGINT
// round-trips losslessly, so it serves as a baseline proving the function tests
// fail in the function and not in the column conversion.
//
// These tests require a GPU and are labeled cuda_driver; they will not run in a
// CPU-only environment.
//
// Note: Presto's with_timezone is intentionally not covered here. Velox does
// not register a with_timezone scalar function, so it cannot be compiled on CPU
// and is out of scope for a GPU-vs-CPU gap.

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <limits>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

namespace {

class TimezoneFunctionTest : public cudf_velox::CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    cudf_velox::registerCudf();
  }

  static void TearDownTestCase() {
    cudf_velox::unregisterCudf();
  }

  // Builds a single-row TIMESTAMP WITH TIME ZONE input column named c0, packing
  // the UTC millis with the given zone's key (same layout as
  // TimestampWithTimeZoneType: upper 52 bits millis, lower 12 bits zone key).
  RowVectorPtr timestampWithTimeZoneInput(int64_t millisUtc, const char* zone) {
    auto zoneId = tz::getTimeZoneID(zone);
    return makeRowVector({makeFlatVector<int64_t>(
        {pack(millisUtc, zoneId)}, TIMESTAMP_WITH_TIME_ZONE())});
  }

  // Builds a two-row TIMESTAMP WITH TIME ZONE column [value, NULL] in the given
  // zone, to check that a NULL row propagates as NULL through the GPU path.
  RowVectorPtr timestampWithTimeZoneAndNullInput(
      int64_t millisUtc,
      const char* zone) {
    auto zoneId = tz::getTimeZoneID(zone);
    return makeRowVector({makeNullableFlatVector<int64_t>(
        {pack(millisUtc, zoneId), std::nullopt}, TIMESTAMP_WITH_TIME_ZONE())});
  }

  // Builds a two-row, entirely-NULL TIMESTAMP WITH TIME ZONE column.
  RowVectorPtr allNullTimestampWithTimeZoneInput() {
    return makeRowVector({makeNullableFlatVector<int64_t>(
        {std::nullopt, std::nullopt}, TIMESTAMP_WITH_TIME_ZONE())});
  }

  // Builds a two-row TIMESTAMP WITH TIME ZONE column whose rows carry different
  // zone keys, to exercise per-row (non-uniform) zone handling.
  RowVectorPtr twoZoneTimestampWithTimeZoneInput(
      int64_t millisUtcA,
      const char* zoneA,
      int64_t millisUtcB,
      const char* zoneB) {
    return makeRowVector({makeFlatVector<int64_t>(
        {pack(millisUtcA, tz::getTimeZoneID(zoneA)),
         pack(millisUtcB, tz::getTimeZoneID(zoneB))},
        TIMESTAMP_WITH_TIME_ZONE())});
  }

  // Builds a single-row double input column named c0.
  RowVectorPtr doubleInput(double value) {
    return makeRowVector({makeFlatVector<double>({value})});
  }

  // Builds a single-row varchar input column named c0.
  RowVectorPtr varcharInput(const std::string& value) {
    return makeRowVector({makeFlatVector<std::string>({value})});
  }

  // Asserts the expression evaluates to the same result on GPU (forced by
  // CudfFunctionBaseTest::evaluate) and CPU. The input's own type is the
  // projection's row type.
  void assertMatchesCpu(const std::string& expr, const RowVectorPtr& input) {
    assertExpressionMatchesCpu(expr, input, asRowType(input->type()));
  }

  // Sets the session timezone for subsequent evaluate() calls, mirroring
  // DateTimeFunctionsTest::setQueryTimeZone, so a test can exercise the
  // session-timezone path the harness otherwise runs with an empty session.
  void setSessionTimezone(const std::string& zone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionTimezone, zone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  // Sets the session start time (consumed by now()/current_timestamp) and the
  // session timezone, with adjust-to-session-timezone on, for subsequent
  // evaluate() calls. testingOverrideConfigUnsafe replaces the whole config, so
  // all three keys are set together.
  void setSessionStartTimeAndTimeZone(
      int64_t startTimeMs,
      const std::string& zone) {
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kSessionStartTime, std::to_string(startTimeMs)},
        {core::QueryConfig::kSessionTimezone, zone},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }
};

// A TIMESTAMP WITH TIME ZONE column projected unchanged must round-trip through
// the GPU as TIMESTAMP WITH TIME ZONE and match CPU. cuDF has no native
// TIMESTAMP WITH TIME ZONE type and carries the column as its physical BIGINT;
// this asserts that representation preserves the packed millis+zone so the
// passthrough result is indistinguishable from CPU. It isolates the column
// conversion from the function evaluation exercised below.
TEST_F(TimezoneFunctionTest, timestampWithTimeZoneColumnPreservedThroughGpu) {
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("c0", input);
}

// Functions that consume a TIMESTAMP WITH TIME ZONE column. The column converts
// to cuDF (as its physical BIGINT) without error; the function is the work the
// GPU must learn to do.

TEST_F(TimezoneFunctionTest, toUnixtimeFromTimestampWithTimeZone) {
  // to_unixtime(timestamp with time zone) -> double.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("to_unixtime(c0)", input);
}

// Coverage: a pre-1970 (negative-millis) instant exercises the arithmetic right
// shift in unpackMillis, which differs from a logical shift only for negative
// packed values -- every other test here uses a positive 2021 instant.
// to_unixtime recovers the seconds and to_iso8601 unpacks then renders, so both
// must match CPU for the negative instant 1938-04-24T17:33:20 UTC.
TEST_F(TimezoneFunctionTest, toUnixtimePre1970Instant) {
  auto input =
      timestampWithTimeZoneInput(-1'000'000'000'000, "America/Los_Angeles");
  assertMatchesCpu("to_unixtime(c0)", input);
}

TEST_F(TimezoneFunctionTest, toIso8601Pre1970Instant) {
  auto input =
      timestampWithTimeZoneInput(-1'000'000'000'000, "America/Los_Angeles");
  assertMatchesCpu("to_iso8601(c0)", input);
}

TEST_F(TimezoneFunctionTest, atTimezone) {
  // at_timezone(timestamp with time zone, varchar) -> timestamp with time zone.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("at_timezone(c0, 'America/New_York')", input);
}

TEST_F(TimezoneFunctionTest, timezoneHour) {
  // timezone_hour(timestamp with time zone) -> bigint.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
}

TEST_F(TimezoneFunctionTest, timezoneMinute) {
  // timezone_minute(timestamp with time zone) -> bigint.
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("timezone_minute(c0)", input);
}

// Reproducers: timezone_hour/timezone_minute must return NULL for a NULL row,
// matching CPU (a plain call() -> NULL for NULL). The GPU offset primitive
// (utcOffsetSeconds) builds an all-valid column via make_column_from_scalar /
// gather and never re-applies the input mask (TimezoneConversion.h documents
// the all-valid contract), so the field functions' scalar DIV/MOD yield 0
// instead of NULL. Red until the input validity is carried onto the offset
// column. The single-row tests above use non-null inputs and so never exercise
// this.
TEST_F(TimezoneFunctionTest, timezoneHourPropagatesNull) {
  auto input =
      timestampWithTimeZoneAndNullInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("timezone_hour(c0)", input);
}

TEST_F(TimezoneFunctionTest, timezoneMinutePropagatesNull) {
  auto input =
      timestampWithTimeZoneAndNullInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("timezone_minute(c0)", input);
}

// Reproducers: a TSWTZ column mixing zone keys must be handled per row (CPU
// unpacks each row's own key). The GPU's uniformZoneKey VELOX_USER_CHECK-fails
// on mixed zones (the "one zone per column" limitation). Red until the per-row
// offset path lands.
TEST_F(TimezoneFunctionTest, timezoneHourMixedZones) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      1'609'466'400'000,
      "America/Los_Angeles",
      1'609'466'400'000,
      "Asia/Kolkata");
  assertMatchesCpu("timezone_hour(c0)", input);
}

TEST_F(TimezoneFunctionTest, timezoneMinuteMixedZones) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      1'609'466'400'000,
      "America/Los_Angeles",
      1'609'466'400'000,
      "Asia/Kolkata");
  assertMatchesCpu("timezone_minute(c0)", input);
}

// Mixed zones plus a null row: the null must stay null through the per-row path.
TEST_F(TimezoneFunctionTest, timezoneHourMixedZonesWithNull) {
  auto input = makeRowVector({makeNullableFlatVector<int64_t>(
      {pack(1'609'466'400'000, tz::getTimeZoneID("America/Los_Angeles")),
       pack(1'609'466'400'000, tz::getTimeZoneID("Asia/Kolkata")),
       std::nullopt},
      TIMESTAMP_WITH_TIME_ZONE())});
  assertMatchesCpu("timezone_hour(c0)", input);
}

// Mixed zones at a DST-varying instant: the per-row offset must be computed for
// each row's own instant, not a uniform one. 2021-07-01T02:00:00Z puts
// America/Los_Angeles in PDT (-07:00, not the -08:00 PST the January cases use)
// while Asia/Kolkata is fixed at +05:30. The existing single-zone timezone_hour
// test uses a January (PST) instant, so this is the only DST-active per-row
// case.
TEST_F(TimezoneFunctionTest, timezoneHourMixedZonesDst) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      1'625'104'800'000,
      "America/Los_Angeles",
      1'625'104'800'000,
      "Asia/Kolkata");
  assertMatchesCpu("timezone_hour(c0)", input);
}

TEST_F(TimezoneFunctionTest, toIso8601FromTimestampWithTimeZone) {
  // to_iso8601(timestamp with time zone) -> varchar.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("to_iso8601(c0)", input);
}

// Reproducer for the zero-offset divergence: to_iso8601 of a UTC/GMT instant
// must render a trailing 'Z', matching CPU (ToISO8601Function passes
// zeroOffsetText="Z"). The GPU's formatOffsetStrings has no zero-offset branch
// and emits '+00:00'. Red until the 'Z' branch is added. (The only other
// to_iso8601 test uses a non-zero offset, so it does not exercise this.)
TEST_F(TimezoneFunctionTest, toIso8601RendersZForZeroOffset) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "UTC");
  assertMatchesCpu("to_iso8601(c0)", input);
}

// to_iso8601 over mixed zones: each row renders its own offset (LA -08:00 vs
// Kolkata +05:30 on the same UTC instant). Red until the per-row offset lands.
TEST_F(TimezoneFunctionTest, toIso8601MixedZones) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      1'609'466'400'000,
      "America/Los_Angeles",
      1'609'466'400'000,
      "Asia/Kolkata");
  assertMatchesCpu("to_iso8601(c0)", input);
}

// Contract/regression test: an entirely-NULL TIMESTAMP WITH TIME ZONE column
// must yield an all-NULL result like CPU. uniformZoneKey reduces min/max over
// the (all-null) zone-key column; reduce excludes nulls, so its scalars come
// back invalid and value() would be a meaningless device read (UB) before
// VELOX_USER_CHECK_EQ(lo, hi). uniformZoneKey guards null_count() == size() and
// defaults to GMT (key 0), as the empty-column path does. This is not a
// differential RED for the UB -- the bad read happens to yield 0/GMT in this
// environment, so the output is already correct -- so it instead pins the
// all-null -> all-null contract and guards against the guard's removal.
TEST_F(TimezoneFunctionTest, toIso8601AllNullColumn) {
  assertMatchesCpu("to_iso8601(c0)", allNullTimestampWithTimeZoneInput());
}

TEST_F(TimezoneFunctionTest, formatDatetimeOfTimestampWithTimeZone) {
  // format_datetime(timestamp with time zone, varchar) -> varchar.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ')", input);
}

// format_datetime over mixed zones: the local wall clock and the numeric offset
// token ('ZZ' -> "+HH:MM") are both per-row. This exercises localAndOffset
// through the per-row offset path (LA -08:00 vs Kolkata +05:30 on the same UTC
// instant give different local times and different rendered offsets).
TEST_F(TimezoneFunctionTest, formatDatetimeMixedZones) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      1'609'466'400'000,
      "America/Los_Angeles",
      1'609'466'400'000,
      "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ')", input);
}

// Reproducers for the Joda zone-token divergences. CPU (DateTimeFormatter)
// distinguishes the run length and letter; the GPU collapses Z/z into one flag
// and always emits '+HH:MM'. Only the (correct) ZZ case is covered above. Each
// is red until jodaToStrftime threads the run length and letter.

// Single 'Z' renders the offset WITHOUT a colon (e.g. +0530) on CPU
// (appendTimezoneOffset, includeColon=false); the GPU emits +05:30.
TEST_F(TimezoneFunctionTest, formatDatetimeSingleZNoColon) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z')", input);
}

// 'ZZZ' (3+ repeats) renders the zone id (Asia/Kolkata) on CPU; the GPU emits
// the numeric offset.
TEST_F(TimezoneFunctionTest, formatDatetimeZoneIdToken) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZZ')", input);
}

// format_datetime zone-id token ('ZZZ' -> zone name) over mixed zones: each row
// renders its own zone name (America/Los_Angeles vs Asia/Kolkata) via
// perRowZoneName. formatDatetimeZoneIdToken above covers only a single zone;
// this pins the per-row name path the owner scoped into this PR. Red until
// perRowZoneName replaces the uniformZoneKey single-name render.
TEST_F(TimezoneFunctionTest, formatDatetimeZoneIdMixedZones) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      1'609'466'400'000,
      "America/Los_Angeles",
      1'609'466'400'000,
      "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZZ')", input);
}

// Lowercase 'z' is a distinct Joda specifier (zone abbreviation/name, e.g.
// IST). It is DST- and instant-dependent, so the GPU cannot render it on
// device; it rejects the token with VELOX_NYI rather than silently emit a wrong
// (numeric offset) result. Asserting the guard pins the scoped limitation.
TEST_F(TimezoneFunctionTest, formatDatetimeZoneNameTokenUnsupportedOnGpu) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  auto exprSet = compileExpression(
      "format_datetime(c0, 'yyyy-MM-dd HH:mm:ss z')", asRowType(input->type()));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Reproducers for the Joda fractional-second run length. CPU
// (formatFractionOfSecond) renders exactly <run length> digits: a single 'S'
// is 1 digit, 'SSSSSS' is 6. The GPU's jodaToStrftime maps any 'S' run to
// "%3f" (3 digits), so single-'S' and 6-'S' diverge while 'SSS' happens to
// match. Red until the run length feeds the "%<n>f" width. The 123 ms
// sub-second instant makes the fractional digits observable.
TEST_F(TimezoneFunctionTest, formatDatetimeFractionSingleDigit) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'123, "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss.S')", input);
}

// 'SSS' -> 3 digits; matches the GPU's current %3f (control case, stays green).
TEST_F(TimezoneFunctionTest, formatDatetimeFractionMillis) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'123, "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss.SSS')", input);
}

// 'SSSSSS' -> 6 digits; the millisecond value is right-padded with zeros.
TEST_F(TimezoneFunctionTest, formatDatetimeFractionMicros) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'123, "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss.SSSSSS')", input);
}

// Functions that produce a TIMESTAMP WITH TIME ZONE from plain inputs. The
// inputs convert to cuDF fine; the function is the work the GPU must learn.

TEST_F(TimezoneFunctionTest, fromUnixtimeWithZoneName) {
  // from_unixtime(double, varchar) -> timestamp with time zone.
  assertMatchesCpu(
      "from_unixtime(c0, 'America/Los_Angeles')", doubleInput(1'609'466'400.0));
}

TEST_F(TimezoneFunctionTest, fromUnixtimeWithHoursMinutes) {
  // from_unixtime(double, bigint, bigint) -> timestamp with time zone.
  assertMatchesCpu("from_unixtime(c0, 7, 30)", doubleInput(1'609'466'400.0));
}

// Reproducer: from_unixtime(double, bigint, bigint) computes the fixed offset as
// hours*60 + minutes. INT64_MAX hours overflows that int64 product. CPU
// (FromUnixtimeFunction) uses checkedMultiply/checkedPlus and throws; the GPU
// registration multiplies unchecked, then casts to int32 -- on this platform the
// UB wraps to -60, an in-range offset tz::getTimeZoneID happily accepts. Red
// until the GPU mirrors CPU's checked arithmetic. compileExpression succeeds on
// both (the CPU arithmetic error is a user error captured in initialize() and
// re-thrown at eval); both throws carry "overflow".
TEST_F(TimezoneFunctionTest, fromUnixtimeHoursMinutesOverflowRejectedLikeCpu) {
  auto input = doubleInput(0.0);
  auto exprSet = compileExpression(
      "from_unixtime(c0, 9223372036854775807, 0)", asRowType(input->type()));
  VELOX_ASSERT_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input), "overflow");
  VELOX_ASSERT_THROW(evaluate(*exprSet, input), "overflow");
}

// Reproducer: from_unixtime of an out-of-range instant must throw to match CPU.
// CPU pack() VELOX_USER_CHECKs the millis range and throws an overflow error;
// the CPU suite asserts from_unixtime(2251799813685.248, 'GMT') throws. The GPU
// shifts millis << 12 with no guard and silently overflows into the zone-key
// bits. Red until the range/NaN check is added.
TEST_F(TimezoneFunctionTest, fromUnixtimeOverflowRejectedLikeCpu) {
  auto input = doubleInput(2'251'799'813'685.248);
  auto exprSet =
      compileExpression("from_unixtime(c0, 'GMT')", asRowType(input->type()));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Reproducer: from_unixtime(NaN) must map to the epoch like CPU, which returns
// pack(0, zone) for a NaN unixtime, rather than reading a meaningless value out
// of a float->int cast of NaN. Red until NaN is mapped to 0 before packing.
TEST_F(TimezoneFunctionTest, fromUnixtimeNanMapsToEpochLikeCpu) {
  auto input = doubleInput(std::numeric_limits<double>::quiet_NaN());
  assertMatchesCpu("from_unixtime(c0, 'GMT')", input);
}

// Reproducer: from_unixtime(+/-Inf) must throw to match CPU. CPU saturates the
// millis to int64 min/max, which pack()'s range check then rejects as overflow.
// The GPU must throw too rather than rely on float->int cast behavior for Inf.
TEST_F(TimezoneFunctionTest, fromUnixtimeInfinityRejectedLikeCpu) {
  auto input = doubleInput(std::numeric_limits<double>::infinity());
  auto exprSet =
      compileExpression("from_unixtime(c0, 'GMT')", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Reproducer for the two-overload rounding split. The (double, hours, minutes)
// overload rounds via floor-seconds + a separate fractional llround (CPU's
// no-zone fromUnixtime), differing from the varchar overload's llround(x*1000)
// by up to 1 ms on negative-fractional input. For -0.0005 s the hours/minutes
// overload yields 0 ms while the varchar overload yields -1 ms. The GPU uses
// the varchar rounding for both, so the hours/minutes case is red.
TEST_F(
    TimezoneFunctionTest,
    fromUnixtimeHoursMinutesNegativeFractionalRounding) {
  auto input = doubleInput(-0.0005);
  assertMatchesCpu("from_unixtime(c0, 0, 0)", input);
}

// Control: the varchar overload's llround(x*1000) already matches CPU for the
// same negative-fractional input (-0.0005 -> -1 ms), so this stays green.
TEST_F(TimezoneFunctionTest, fromUnixtimeVarcharNegativeFractionalRounding) {
  auto input = doubleInput(-0.0005);
  assertMatchesCpu("from_unixtime(c0, 'GMT')", input);
}

TEST_F(TimezoneFunctionTest, parseDatetime) {
  // parse_datetime(varchar, varchar) -> timestamp with time zone.
  assertMatchesCpu(
      "parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss')",
      varcharInput("2021-01-01 02:00:00"));
}

// When the Joda format carries a colon offset token (ZZ), CPU folds the offset
// into the UTC instant AND packs the parsed fixed-offset zone key, so
// timezone_hour reports -9 and to_iso8601 prints -09:00. GPU currently packs
// GMT (timezone_hour = 0, to_iso8601 = Z). Compare through projections that read
// the zone key, since assertMatchesCpu on the TSWTZ value alone ignores it.
TEST_F(TimezoneFunctionTest, parseDatetimePreservesParsedOffset) {
  auto input = varcharInput("2021-01-01 02:00:00 -09:00");
  assertMatchesCpu(
      "timezone_hour(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ'))", input);
}

// Same, for the no-colon offset token (Z) matching -0900.
TEST_F(TimezoneFunctionTest, parseDatetimeNoColonOffset) {
  auto input = varcharInput("2021-01-01 02:00:00 -0900");
  assertMatchesCpu(
      "timezone_hour(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
}

TEST_F(TimezoneFunctionTest, fromIso8601Timestamp) {
  // from_iso8601_timestamp(varchar) -> timestamp with time zone.
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T02:00:00+05:30"));
}

// Reproducers: from_iso8601_timestamp must accept the ISO8601 shapes CPU does
// (see DateTimeFunctionsTest.fromIso8601Timestamp). The GPU's rigid regex
// requires a full yyyy-MM-ddTHH:mm:ss with a colon offset, so it rejects short
// forms (-> NULL), discards sub-second digits, rejects hours-only offsets, and
// loses the sign of offsets in (-1h, 0). Inputs without an embedded offset are
// interpreted as GMT under the default session, matching CPU. Each is red until
// the GPU parser matches CPU.

// Date-only: CPU -> midnight GMT; GPU regex needs a time component -> NULL.
TEST_F(TimezoneFunctionTest, fromIso8601DateOnly) {
  assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput("2021-01-01"));
}

// Minute precision (no seconds): CPU accepts; GPU regex needs seconds -> NULL.
TEST_F(TimezoneFunctionTest, fromIso8601MinutePrecision) {
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-02T11:38"));
}

// Sub-second digits: CPU preserves .123; GPU discards them (parses to seconds).
TEST_F(TimezoneFunctionTest, fromIso8601FractionalSeconds) {
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)",
      varcharInput("2021-01-01T02:00:00.123+05:30"));
}

// Hours-only offset: CPU expands +05 -> +05:00; GPU requires minutes -> NULL.
TEST_F(TimezoneFunctionTest, fromIso8601HoursOnlyOffset) {
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T02:00:00+05"));
}

// Offset in (-1h, 0): CPU keeps the sign (-00:30); GPU reads -00 as 0 and
// yields +30 -- wrong instant and wrong zone key.
TEST_F(TimezoneFunctionTest, fromIso8601NegativeHalfHourOffset) {
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T02:00:00-00:30"));
}

// Reproducer: an offset outside +/-14h must be rejected like CPU rather than
// silently corrupt the packed value. CPU normalizes "+99:00" to an unknown zone
// name and throws ("Unknown timezone value"); the +/-840-minute bound is the
// same one tz::getTimeZoneID enforces. The GPU parser has no bound -- +99:00 ->
// 5940 minutes -> zone key 6780, which overflows the 12-bit zone field and
// corrupts the packed millis (the key is not masked with kTimezoneMask). Red
// until the parsed offset magnitude is bounded with a user error.
TEST_F(TimezoneFunctionTest, fromIso8601OffsetOutOfRangeRejectedLikeCpu) {
  auto input = varcharInput("2021-01-01T02:00:00+99:00");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  // CPU rejects the out-of-range offset; confirm parity is "both throw".
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Reproducer: an offset-less ISO string is interpreted in the session timezone
// on CPU (the wall clock is that zone's local time, and the packed zone key is
// the session zone), not GMT. Asia/Kolkata has a fixed +05:30 offset (no DST),
// so the conversion is exact. The GPU treats offset-less input as GMT
// regardless of the session, so it produces both a wrong instant and a wrong
// zone key. Red until the session offset is applied.
TEST_F(TimezoneFunctionTest, fromIso8601OffsetlessUsesSessionZone) {
  setSessionTimezone("Asia/Kolkata");
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T02:00:00"));
}

// Reproducer: the offset-less session-zone conversion must match CPU even for a
// DST zone whose offset depends on the instant. America/Los_Angeles springs
// forward on 2021-03-14 at 02:00 PST (-08:00) to 03:00 PDT (-07:00), i.e. at
// 10:00:00 UTC. The wall clock 2021-03-14T03:30:00 is a valid post-gap local
// time (PDT), so CPU resolves it to 2021-03-14T10:30:00 UTC (to_unixtime
// 1615717800). The GPU uses the local->UTC approximation, which keys the wall
// clock as if it were UTC: 03:30 UTC precedes the 10:00 UTC transition, so it
// reads the pre-gap offset (-08:00) and yields 2021-03-14T11:30:00 UTC
// (to_unixtime 1615721400) -- one hour late. Red until an inverse (local-keyed)
// transition lookup replaces the approximation. The fixed-offset Kolkata case
// above stays green because its offset does not vary with the instant.
TEST_F(TimezoneFunctionTest, fromIso8601OffsetlessSessionZoneDstTransition) {
  setSessionTimezone("America/Los_Angeles");
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-03-14T03:30:00"));
}

// Reproducer: a wall clock inside the spring-forward gap is a nonexistent local
// time, so CPU's toGMT throws and from_iso8601_timestamp fails. America/
// Los_Angeles springs forward on 2021-03-14 from 02:00 PST to 03:00 PDT, so
// local times in [02:00, 03:00) never occur; 02:30:00 is one of them. The GPU
// local->UTC approximation does plain arithmetic and never throws, so it
// silently returns an instant. Asserting both paths throw is red until the
// inverse (local-keyed) transition lookup flags the gap and fails like CPU.
TEST_F(TimezoneFunctionTest, fromIso8601OffsetlessSessionZoneGapThrows) {
  setSessionTimezone("America/Los_Angeles");
  auto input = varcharInput("2021-03-14T02:30:00");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Reproducer: a wall clock in a fall-back overlap is ambiguous, and CPU's toGMT
// resolves it to the earliest instant (TChoose::kEarliest). Australia/Sydney
// falls back on 2021-04-04 from 03:00 AEDT (+11:00) to 02:00 AEST (+10:00), so
// local times in [02:00, 03:00) occur twice; 02:30:00 is one. CPU keeps the
// earlier AEDT reading -- 2021-04-03T15:30:00 UTC. The GPU approximation keys
// the wall clock as UTC, which lands after the 2021-04-03T16:00 UTC transition
// and reads the later AEST offset, yielding 2021-04-03T16:30:00 UTC -- one hour
// late. Red until the inverse transition lookup keeps the pre-transition offset
// over the overlap, matching kEarliest. A western-hemisphere zone like
// Los_Angeles cannot exercise this: its negative offsets place the overlap
// window before the UTC transition, where the approximation already reads the
// earlier offset.
TEST_F(
    TimezoneFunctionTest,
    fromIso8601OffsetlessSessionZoneAmbiguousPicksEarliest) {
  setSessionTimezone("Australia/Sydney");
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-04-04T02:30:00"));
}

// Control: an explicit numeric offset wins over the session zone on both paths,
// so this stays green and guards that the session change does not hijack rows
// that carry their own offset.
TEST_F(TimezoneFunctionTest, fromIso8601ExplicitOffsetIgnoresSessionZone) {
  setSessionTimezone("Asia/Kolkata");
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T02:00:00+09:00"));
}

// Control: an explicit "Z" designator is GMT on both paths (distinct from an
// absent zone), so this stays green and guards that "Z" is not mistaken for an
// offset-less input and rerouted through the session zone.
TEST_F(TimezoneFunctionTest, fromIso8601ZuluIgnoresSessionZone) {
  setSessionTimezone("Asia/Kolkata");
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T02:00:00Z"));
}

// now()/current_timestamp -> timestamp with time zone. now() is
// non-deterministic -- a live CPU now() and a separate GPU now() observe
// different instants -- so this cannot assert CPU == GPU against a live clock.
// Instead it pins the deterministic contract CPU's CurrentTimestampFunction
// implements: pack(sessionStartTimeMs, sessionZone). The GPU must emit a
// TIMESTAMP WITH TIME ZONE whose UTC millis are the session start time and
// whose zone key is the session zone. A dummy column sizes the batch.
TEST_F(TimezoneFunctionTest, nowUsesSessionStartTimeAndTimezone) {
  constexpr int64_t kStartMs = 1'609'466'400'000; // 2021-01-01T02:00:00 UTC.
  setSessionStartTimeAndTimeZone(kStartMs, "America/Los_Angeles");
  auto input = doubleInput(0.0);
  auto exprSet = compileExpression("now()", asRowType(input->type()));
  auto result = evaluate(*exprSet, input);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), input->size());
  ASSERT_TRUE(isTimestampWithTimeZoneType(result->type()))
      << "now() must produce TIMESTAMP WITH TIME ZONE, got "
      << result->type()->toString();
  const auto packed = result->as<SimpleVector<int64_t>>()->valueAt(0);
  EXPECT_EQ(unpackMillisUtc(packed), kStartMs);
  EXPECT_EQ(unpackZoneKeyId(packed), tz::getTimeZoneID("America/Los_Angeles"));
}

// now()/current_timestamp must be rejected exactly when CPU rejects it. CPU's
// CurrentTimestampFunction throws "Timezone cannot be null" when
// getTimeZoneFromConfig returns null -- i.e. when
// adjust_timestamp_to_session_timezone is off, or the session timezone is
// empty. The GPU previously honored neither condition (defaulting the zone key
// to GMT) and silently produced a value where CPU failed. Assert both paths
// throw for the two rejection configs. The exception is captured at
// initialize() and re-thrown at eval time, so compileExpression succeeds and
// the throw surfaces from evaluate().
TEST_F(TimezoneFunctionTest, nowWithoutAdjustedSessionTimezoneRejectedLikeCpu) {
  auto input = doubleInput(0.0);
  const auto rowType = asRowType(input->type());

  // Adjust on but no session timezone -> rejected on both paths.
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
  });
  {
    auto exprSet = compileExpression("now()", rowType);
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    EXPECT_ANY_THROW(evaluate(*exprSet, input));
  }

  // Session timezone set but adjust off -> rejected on both paths.
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kSessionTimezone, "America/Los_Angeles"},
      {core::QueryConfig::kAdjustTimestampToTimezone, "false"},
  });
  {
    auto exprSet = compileExpression("now()", rowType);
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    EXPECT_ANY_THROW(evaluate(*exprSet, input));
  }
}

} // namespace
