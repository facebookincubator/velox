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

#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/tz/TimeZoneMap.h"

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

TEST_F(TimezoneFunctionTest, parseDatetime) {
  // parse_datetime(varchar, varchar) -> timestamp with time zone.
  assertMatchesCpu(
      "parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss')",
      varcharInput("2021-01-01 02:00:00"));
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

// now()/current_timestamp -> timestamp with time zone. now() is
// non-deterministic -- a CPU evaluation and a separate GPU evaluation observe
// different instants -- so this cannot assert CPU == GPU. Instead it asserts
// the GPU can evaluate now() at all and produces a TIMESTAMP WITH TIME ZONE;
// today the GPU throws from the unsupported recursive-evaluation path. A dummy
// column sizes the batch.
TEST_F(TimezoneFunctionTest, now) {
  auto input = doubleInput(0.0);
  auto exprSet = compileExpression("now()", asRowType(input->type()));
  VectorPtr result;
  try {
    result = evaluate(*exprSet, input);
  } catch (const std::exception& e) {
    FAIL() << "now() must be evaluable on GPU but threw: " << e.what();
  }
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->size(), input->size());
  EXPECT_TRUE(isTimestampWithTimeZoneType(result->type()))
      << "now() must produce TIMESTAMP WITH TIME ZONE, got "
      << result->type()->toString();
}

} // namespace
