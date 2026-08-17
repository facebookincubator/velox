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
// ZONE function family must match CPU.
//
// The input conversion is not where the work is: cuDF has no TIMESTAMP WITH
// TIME ZONE type, but such a column is carried as its physical BIGINT and
// round-trips without error (see
// timestampWithTimeZoneColumnPreservedThroughGpu, which is the baseline proving
// a failure elsewhere is in the function rather than in the column conversion).
//
// CudfFunctionBaseTest::evaluate forces GPU execution -- it does not consult
// allowCpuFallback -- and assertExpressionMatchesCpu compares that result
// against CPU's, which is the oracle throughout. Two consequences worth
// knowing:
//
//   - These tests cannot see a CPU fallback. An expression the GPU declines is
//     covered by a selection test instead, in ToCudfSelectionTest.
//   - A comparison of TIMESTAMP WITH TIME ZONE values ignores the zone key.
//     That type's comparator orders on the unpacked UTC millis, so a case whose
//     visible effect is the offset must project through timezone_hour or
//     to_iso8601 for the assertion to see it.
//
// The tests near the end call TimezoneConversion.h directly. Every other case
// reaches those conversions through a SQL function, and each function narrows
// what can arrive, so the direct ones cover the round trip and the gap and
// overlap boundaries with arbitrary instants.
//
// These tests require a GPU and are labeled cuda_driver; they will not run in a
// CPU-only environment.
//
// Note: Presto's with_timezone is intentionally not covered here. Velox does
// not register a with_timezone scalar function, so it cannot be compiled on CPU
// and is out of scope for a GPU-vs-CPU gap.

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/TimezoneConversion.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gmock/gmock.h>

#include <limits>
#include <vector>

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

  // Builds a three-row TIMESTAMP WITH TIME ZONE column carrying two different
  // zone keys and a NULL, so one vector exercises per-row zone handling and
  // null propagation together.
  RowVectorPtr twoZoneAndNullTimestampWithTimeZoneInput(
      int64_t millisUtcA,
      const char* zoneA,
      int64_t millisUtcB,
      const char* zoneB) {
    return makeRowVector({makeNullableFlatVector<int64_t>(
        {pack(millisUtcA, tz::getTimeZoneID(zoneA)),
         pack(millisUtcB, tz::getTimeZoneID(zoneB)),
         std::nullopt},
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

// The regression reported in review: with the table ending at 2400, these two
// instants took the final interval's offset, so July lost daylight saving.
// America/Los_Angeles observes DST in July, so July 2401 is -07:00 and January
// 2401 is -08:00; reusing a single trailing interval gets one of them wrong.
// TIMESTAMP WITH TIME ZONE stores 52-bit millis, so both instants are
// representable and CPU answers correctly from the horizon-free tzdb rules.
//
// Asserted through timezone_hour rather than by comparing the TIMESTAMP WITH
// TIME ZONE column: that type's comparator orders on unpacked UTC millis and
// ignores the zone key, so a wrong offset is invisible to a direct comparison.
TEST_F(TimezoneFunctionTest, timezoneHourYear2401Summer) {
  // 2401-07-15T12:00:00Z.
  auto input =
      timestampWithTimeZoneInput(13'617'979'200'000, "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
}

TEST_F(TimezoneFunctionTest, timezoneHourYear2401Winter) {
  // 2401-01-15T12:00:00Z.
  auto input =
      timestampWithTimeZoneInput(13'602'340'800'000, "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
}

// Widening the materialized window moves its edges; it does not remove them. An
// instant outside the window in either direction must still get the offset CPU
// gives, since a 52-bit millis field reaches roughly year 71,000 and tzdb
// answers from recurring rules with no horizon.
//
// Above the end, the lookup used to fold onto the final interval and lose
// daylight saving. Below the start it is worse than wrong: the index is
// `upper_bound - 1`, so an instant before the first transition yields -1 and
// the gather runs with out_of_bounds_policy::DONT_CHECK. The existing pre-1970
// case uses 1938, which is inside the window, so nothing covered this.
TEST_F(TimezoneFunctionTest, timezoneHourBeyondWindowEndSummer) {
  // 12000-07-15T12:00:00Z.
  auto input =
      timestampWithTimeZoneInput(316'533'182'400'000, "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
}

TEST_F(TimezoneFunctionTest, timezoneHourBeyondWindowEndWinter) {
  // 12000-01-15T12:00:00Z.
  auto input =
      timestampWithTimeZoneInput(316'517'457'600'000, "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
}

// A fixed-offset zone has one interval covering all time. Keying that interval
// at the epoch would send every pre-1970 instant below the first key, and the
// index lookup subtracts one from upper_bound, so the gather would run out of
// bounds. Both these instants share a zone whose offset never changes, so the
// answer is the same either side of 1970 -- what is being pinned is that the
// earlier one is looked up at all.
TEST_F(TimezoneFunctionTest, timezoneMinuteFixedOffsetZonePre1970) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      -1'000'000'000'000, "+05:30", 1'609'466'400'000, "+05:30");
  assertMatchesCpu("timezone_minute(c0)", input);
}

TEST_F(TimezoneFunctionTest, toIso8601FixedOffsetZonePre1970) {
  auto input = timestampWithTimeZoneInput(-1'000'000'000'000, "+05:30");
  assertMatchesCpu("to_iso8601(c0)", input);
}

TEST_F(TimezoneFunctionTest, timezoneHourBeforeWindowStart) {
  // 1500-06-15T12:00:00Z, well before the first materialized transition. Los
  // Angeles had no daylight saving then, so this is local mean time.
  auto input =
      timestampWithTimeZoneInput(-14'817'470'400'000, "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
}

// The window has to widen more than once in a single process, and widening must
// not drop coverage it already had. Each step below needs a wider window than
// the one before, and the ordinary 2021 instant is re-checked at the end to
// prove the rebuilds did not lose the middle of the range.
TEST_F(TimezoneFunctionTest, timezoneHourWidensWindowRepeatedly) {
  const std::vector<int64_t> widening = {
      1'609'466'400'000, // 2021-01-01, inside the initial window
      316'533'182'400'000, // 12000-07-15, above it
      -14'817'470'400'000, // 1500-06-15, below both
      884'558'318'400'000, // 30000-07-15, above again
      -24'284'491'200'000, // 1200-06-15, below again
      1'609'466'400'000, // 2021 again: coverage must not have been lost
  };
  for (auto millisUtc : widening) {
    auto input = timestampWithTimeZoneInput(millisUtc, "America/Los_Angeles");
    assertMatchesCpu("timezone_hour(c0)", input);
  }
}

// A single batch spanning both extremes at once, so one window must cover the
// pair rather than being widened twice in sequence.
TEST_F(TimezoneFunctionTest, timezoneHourSpansFarPastAndFarFuture) {
  auto input = twoZoneTimestampWithTimeZoneInput(
      -14'817'470'400'000,
      "America/Los_Angeles",
      316'533'182'400'000,
      "America/Los_Angeles");
  assertMatchesCpu("timezone_hour(c0)", input);
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

// Mixed zones plus a null row: the null must stay null through the per-row
// path.
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

// Renders a Joda format as a SQL string literal, doubling each single quote as
// SQL requires. Quoted-literal formats are the subject of the tests below, and
// spelling them with SQL escaping applied by hand is unreadable: the Joda
// format 'he''llo' would have to be written '''he''''llo'''.
std::string jodaLiteral(const std::string& jodaFormat) {
  std::string sql{"'"};
  for (const char c : jodaFormat) {
    sql += c;
    if (c == '\'') {
      sql += c;
    }
  }
  sql += "'";
  return sql;
}

// Joda quotes a literal run with single quotes and escapes a literal quote by
// doubling it, and CPU implements that in numLiteralChars plus the literal
// branch of buildJodaDateTimeFormatter. The GPU's quote loop ends the literal
// at the first quote it sees, so a doubled quote terminates the run instead of
// producing one, and "''" alone produces nothing. Literal bytes also reach cuDF
// unescaped, where '%' starts a specifier: "'%d'" renders the day of month
// instead of the text, and a bare '%' escapes as a raw std::invalid_argument
// rather than a Velox user error. Red until literals follow CPU's rule and
// every literal byte is emitted through one escaping step.
//
// The formats use yyyy rather than y: Joda prints a single 'y' as the full year
// while the GPU maps a short run to "%y", which is T18's field-width gap and
// would make these red for an unrelated reason.
TEST_F(TimezoneFunctionTest, formatDatetimeQuotedLiteralsMatchCpu) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  for (const auto& jodaFormat : std::vector<std::string>{
           "'hello'", // a plain quoted run
           "''", // an escaped quote on its own
           "yyyy '' yyyy", // an escaped quote between specifiers
           "'he''llo'", // an escaped quote inside a run
           "'''he''llo'''", // runs and escaped quotes adjacent
           "'%d'", // a specifier inside a literal stays text
           "yyyy % MM", // a bare '%' is literal text on CPU
       }) {
    SCOPED_TRACE(jodaFormat);
    assertMatchesCpu(
        "format_datetime(c0, " + jodaLiteral(jodaFormat) + ")", input);
  }
}

// An unterminated literal is a user error on CPU ("No closing single quote for
// literal"); the GPU consumes the rest of the format as literal text.
TEST_F(TimezoneFunctionTest, formatDatetimeUnterminatedLiteralThrowsLikeCpu) {
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  auto exprSet = compileExpression(
      "format_datetime(c0, " + jodaLiteral("'abcd") + ")",
      asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  VELOX_ASSERT_THROW(
      evaluate(*exprSet, input), "No closing single quote for literal");
}

// jodaToStrftime serves parse_datetime too, so the same literal handling has to
// hold on the parse side, where a literal must match the input text.
TEST_F(TimezoneFunctionTest, parseDatetimeQuotedLiteralsMatchCpu) {
  for (const auto& [value, jodaFormat] :
       std::vector<std::pair<std::string, std::string>>{
           {"2021-01-01T02:00:00", "yyyy-MM-dd'T'HH:mm:ss"},
           {"2021-01-01 % 02:00:00", "yyyy-MM-dd '%' HH:mm:ss"},
       }) {
    SCOPED_TRACE(jodaFormat);
    assertMatchesCpu(
        "parse_datetime(c0, " + jodaLiteral(jodaFormat) + ")",
        varcharInput(value));
  }
}

// A literal run combined with a zone token, from the case suggested in review.
// Worth its own test for two reasons: the projection has to go through
// to_iso8601 because the parsed offset lands in the zone key, which a
// comparison of TIMESTAMP WITH TIME ZONE values ignores; and it is the one
// input that exercises the literal rule and the trailing-zone accept test
// together.
TEST_F(TimezoneFunctionTest, parseDatetimeQuotedLiteralWithOffsetMatchesCpu) {
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, " +
          jodaLiteral("'he''llo' yyyy-MM-dd HH:mm:ss Z") + "))",
      varcharInput("he'llo 2026-01-02 00:45:00 +0530"));
}

// Historical offsets are not whole minutes: before New York adopted standard
// time its offset was local mean time, -04:56:02, and Kathmandu's was
// +05:41:16. CPU emits the seconds whenever abs(offset) % 60 is non-zero
// (appendTimezoneOffset in functions/lib/DateTimeFormatter.cpp), and the colon
// before them is unconditional -- it sits outside the includeColon guard, which
// the buffer sizing corroborates by reserving 8 bytes for Joda 'Z' and 9 for
// 'ZZ'. So 'Z' renders "-0456:02", with a colon it would not otherwise have.
//
// The GPU computes hours and minutes but never the remaining seconds, so each
// of these renders a truncated offset. The instants are the ones CPU's own
// to_iso8601 tests use, at local 10:00 on 0022-11-01.
TEST_F(TimezoneFunctionTest, toIso8601SubMinuteOffsetKeepsSeconds) {
  assertMatchesCpu(
      "to_iso8601(c0)",
      timestampWithTimeZoneInput(-61'446'589'438'000, "America/New_York"));
  assertMatchesCpu(
      "to_iso8601(c0)",
      timestampWithTimeZoneInput(-61'446'627'676'000, "Asia/Kathmandu"));
}

// The colon before the seconds is present for Joda 'Z' as well, which otherwise
// separates nothing.
TEST_F(TimezoneFunctionTest, formatDatetimeSubMinuteOffsetKeepsSeconds) {
  auto input =
      timestampWithTimeZoneInput(-61'446'589'438'000, "America/New_York");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z')", input);
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ')", input);
}

// Control: a whole-minute offset must not grow a ":00", and a zero offset must
// still render as "Z" rather than "+00:00:00". Without these the suffix could
// be made unconditional and the tests above would still pass.
TEST_F(TimezoneFunctionTest, toIso8601WholeMinuteOffsetHasNoSeconds) {
  assertMatchesCpu(
      "to_iso8601(c0)",
      timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata"));
  assertMatchesCpu(
      "to_iso8601(c0)", timestampWithTimeZoneInput(1'609'466'400'000, "UTC"));
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z')", input);
}

// Joda renders a month or weekday as text once the letter run reaches three,
// and jodaToStrftime maps those to "%b"/"%B" and "%a"/"%A". cuDF writes nothing
// for those specifiers unless it is handed a table of the names, so "EEEE, MMMM
// dd" rendered as ", 02". The scaffolding uses yyyy/dd rather than single
// letters because a one-letter numeric run is T18's field-width gap.
TEST_F(TimezoneFunctionTest, formatDatetimeTextNamesMatchCpu) {
  // 2021-01-01 is a Friday; 2021-06-15 a Tuesday, to catch an off-by-one in the
  // weekday table that a single date could hide.
  for (const auto millisUtc : std::vector<int64_t>{
           1'609'466'400'000,
           1'623'760'000'000,
       }) {
    auto input = timestampWithTimeZoneInput(millisUtc, "Asia/Kolkata");
    SCOPED_TRACE(millisUtc);
    for (const auto& format : std::vector<std::string>{
             "E", // abbreviated weekday
             "EE",
             "EEE",
             "EEEE", // full weekday
             "MMM", // abbreviated month
             "MMMM", // full month
             "EEEE, MMMM dd", // the reported case
             "yyyy-MM-dd EEE MMM",
         }) {
      SCOPED_TRACE(format);
      assertMatchesCpu("format_datetime(c0, '" + format + "')", input);
    }
  }
}

// Supplying the names table changes where cuDF reads AM/PM from: with no table
// it uses a hardcoded pair, and with one it reads the table's first two
// entries. So a format that mixes a name with 'a' would regress if those
// entries were wrong.
TEST_F(TimezoneFunctionTest, formatDatetimeHalfdayWithTextNamesMatchesCpu) {
  for (const auto millisUtc : std::vector<int64_t>{
           1'609'466'400'000, // 05:30 IST, before noon
           1'609'509'600'000, // 17:30 IST, after noon
       }) {
    auto input = timestampWithTimeZoneInput(millisUtc, "Asia/Kolkata");
    SCOPED_TRACE(millisUtc);
    assertMatchesCpu("format_datetime(c0, 'hh:mm a')", input);
    assertMatchesCpu("format_datetime(c0, 'EEEE hh:mm a')", input);
  }
}

// Mixed zone keys and a NULL in one vector, run through both of the projections
// that read the zone key. The existing coverage uses a mixed-zone vector for
// one and a null-bearing vector for the other, so nothing asserted that a null
// row survives the per-zone select: that path computes the offset for each
// distinct key over the whole column and combines the results, which is where a
// null row could pick up a neighbour's value.
TEST_F(TimezoneFunctionTest, mixedZonesWithNullThroughBothProjections) {
  auto input = twoZoneAndNullTimestampWithTimeZoneInput(
      1'609'466'400'000,
      "America/Los_Angeles",
      1'609'466'400'000,
      "Asia/Kolkata");
  assertMatchesCpu("timezone_hour(c0)", input);
  assertMatchesCpu("to_iso8601(c0)", input);
}

// A null constant zone or format makes the whole expression null on CPU. The
// GPU read such a constant as the text "null", which for the integer arguments
// then reached std::stoll and threw while the expression was being built --
// before evaluation, so a try() could not catch it either.
TEST_F(TimezoneFunctionTest, nullConstantArgumentYieldsNullLikeCpu) {
  auto zoned = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  assertMatchesCpu("at_timezone(c0, cast(null as varchar))", zoned);
  assertMatchesCpu("format_datetime(c0, cast(null as varchar))", zoned);
  assertMatchesCpu(
      "parse_datetime(c0, cast(null as varchar))",
      varcharInput("2021-01-01 02:00:00"));
  auto seconds = doubleInput(1'609'466'400.0);
  assertMatchesCpu("from_unixtime(c0, cast(null as varchar))", seconds);
  assertMatchesCpu("from_unixtime(c0, cast(null as bigint), 0)", seconds);
  assertMatchesCpu("from_unixtime(c0, 5, cast(null as bigint))", seconds);
}

// CPU picks the zone the format parsed and falls back to the session zone only
// when the format carried none (ParseDateTimeFunction::call). The GPU instead
// rejected any non-UTC session outright, so a format stating its own offset --
// where the session cannot matter -- failed for a reason that did not apply.
//
// Asserted through projections that read the zone key, since comparing
// TIMESTAMP WITH TIME ZONE values ignores it.
TEST_F(TimezoneFunctionTest, parseDatetimeExplicitOffsetIgnoresSessionZone) {
  setSessionTimezone("America/Los_Angeles");
  auto input = varcharInput("2021-01-01 02:00:00 -0930");
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
}

// With no zone token the wall clock belongs to the session zone, and the result
// carries that zone -- the same rule from_iso8601_timestamp already follows for
// an offsetless string.
TEST_F(TimezoneFunctionTest, parseDatetimeOffsetlessUsesSessionZone) {
  setSessionTimezone("America/Los_Angeles");
  auto input = varcharInput("2021-01-01 02:00:00");
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss'))", input);
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss'))", input);
}

// The session-zone conversion inherits CPU's boundary rules from toGMT: a local
// time in a spring-forward gap has no instant and raises, and an ambiguous one
// resolves to the earlier instant.
TEST_F(TimezoneFunctionTest, parseDatetimeOffsetlessGapThrowsLikeCpu) {
  setSessionTimezone("America/Los_Angeles");
  auto input = varcharInput("2021-03-14 02:30:00");
  auto exprSet = compileExpression(
      "parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss')", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

TEST_F(TimezoneFunctionTest, parseDatetimeOffsetlessAmbiguousPicksEarliest) {
  setSessionTimezone("Australia/Sydney");
  auto input = varcharInput("2021-04-04 02:30:00");
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss'))", input);
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

// Reproducer: from_unixtime(double, bigint, bigint) computes the fixed offset
// as hours*60 + minutes. INT64_MAX hours overflows that int64 product. CPU
// (FromUnixtimeFunction) uses checkedMultiply/checkedPlus and throws; the GPU
// registration multiplies unchecked, then casts to int32 -- on this platform
// the UB wraps to -60, an in-range offset tz::getTimeZoneID happily accepts.
// Red until the GPU mirrors CPU's checked arithmetic. compileExpression
// succeeds on both (the CPU arithmetic error is a user error captured in
// initialize() and re-thrown at eval); both throws carry "overflow".
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
// GMT (timezone_hour = 0, to_iso8601 = Z). Compare through projections that
// read the zone key, since assertMatchesCpu on the TSWTZ value alone ignores
// it.
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

// A colon offset used to lose its minutes. cuDF's "%z" is fixed-width
// "+/-HHMM": it reads the two hour digits, then the two minute digits at a
// fixed position, so a colon (Joda ZZ, "+05:30") landed where a minute digit
// was expected and only "+05:00" reached the UTC instant, leaving it 30 minutes
// off while the trailing-offset regex still recovered "+05:30" for the zone
// key. The wall clock is now parsed without "%z" and the recovered offset
// subtracted instead, which also reads the hours-only form "+05" that a
// fixed-width specifier cannot. The -09:00 and -0900 tests above miss this
// because their minute component is zero.
TEST_F(TimezoneFunctionTest, parseDatetimeColonOffsetWithMinutes) {
  auto input = varcharInput("2026-01-02 00:45:00 +05:30");
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ'))", input);
}

// Control for the fix above: the no-colon form "+/-HHMM" with a non-zero minute
// component is what cuDF's "%z" parses correctly, so this should stay green.
// Pairs the offset with the single-Z token; both -09:00 (ZZ) and -0900 (Z) are
// accepted by CPU regardless of the token's Z-count.
TEST_F(TimezoneFunctionTest, parseDatetimeNoColonOffsetWithMinutes) {
  auto input = varcharInput("2026-01-02 00:45:00 -0930");
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
}

// CPU's parseTimezoneOffset expands an hours-only offset "+05" to "+05:00".
// cuDF's fixed-width "%z" expects five characters ("+/-HHMM") and the recovery
// regex requires two minute digits, so the GPU cannot parse a three-character
// offset. Red until the offset parsing accepts the hours-only form.
TEST_F(TimezoneFunctionTest, parseDatetimeHoursOnlyOffset) {
  auto input = varcharInput("2026-01-02 00:45:00 +05");
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
}

// CPU accepts a literal "Z" for the numeric-offset token (Z/ZZ) and maps it to
// GMT (offset 0). Confirms the fix keeps this form correct (the recovery regex
// finds no signed offset, so signedOffsetMinutes yields offset 0 / GMT).
TEST_F(TimezoneFunctionTest, parseDatetimeLiteralZUtc) {
  auto input = varcharInput("2026-01-02 00:45:00 Z");
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
}

// CPU also accepts UTC/UCT/GMT/GMT0 for the numeric-offset token (Z/ZZ), all
// mapping to GMT. Confirms the fix keeps a named-UTC alias correct (again
// offset 0 / GMT via the no-match path).
TEST_F(TimezoneFunctionTest, parseDatetimeNamedUtcAlias) {
  auto input = varcharInput("2026-01-02 00:45:00 GMT");
  assertMatchesCpu(
      "to_unixtime(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
  assertMatchesCpu(
      "to_iso8601(parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss Z'))", input);
}

// parse_datetime hands its input straight to cudf::strings::to_timestamps,
// which cuDF documents as undefined for a string that does not match the
// format: it reads whatever digits sit at each field position and computes a
// timestamp from them, so "2026-01-02 25:00:00" rolls into the next day rather
// than failing. CPU raises a user error for every form below. Red until eval
// validates its input.
TEST_F(TimezoneFunctionTest, parseDatetimeInvalidInputThrowsLikeCpu) {
  for (const auto& invalid : std::vector<std::string>{
           "not-a-date", // no timestamp at all
           "2026-01-02 25:00:00", // hour past 23
           "2026-01-02 00:70:00", // minute past 59
           "2026-01-02", // missing the time the format requires
       }) {
    SCOPED_TRACE(invalid);
    auto input = varcharInput(invalid);
    auto exprSet = compileExpression(
        "parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss')", asRowType(input->type()));
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    VELOX_ASSERT_THROW(
        evaluate(*exprSet, input), "Invalid format for parse_datetime");
  }
}

// Control for the validation above: a NULL row is not invalid input.
// cudf::strings::is_timestamp reports false for a null, so checking its result
// alone would throw on a legitimate SQL NULL; the check must exempt null rows
// and the result must stay null, as on CPU.
TEST_F(TimezoneFunctionTest, parseDatetimeNullRowStaysNull) {
  auto input = makeRowVector({makeNullableFlatVector<std::string>(
      {"2021-01-01 02:00:00", std::nullopt}, VARCHAR())});
  assertMatchesCpu("parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss')", input);
}

// When the format carries a zone token, CPU accepts exactly six forms for it --
// "+HH", "+HH:MM", "+HHMM", "Z", "UTC"/"UCT" and "GMT"/"GMT0" -- and resolves
// each through tz::locateZone, where a failed lookup is a parse error
// (DateTimeFormatter.cpp parseTimezoneOffset). The GPU recovered the offset
// with a regex whose minute group was [0-9]{2} and treated a non-match as
// offset 0, so "+05:99" became "+06:39" (399 minutes, inside the +/-840 the
// magnitude check allows) and a garbled or absent offset silently became GMT.
// Red until the trailing zone is required to be one of CPU's forms.
// The shape regex covers the trailing zone, so a zone outside the forms CPU
// accepts and a missing one are both reported as a format error rather than
// each tripping a separate check.
TEST_F(TimezoneFunctionTest, parseDatetimeInvalidOffsetThrowsLikeCpu) {
  for (const auto& invalid : std::vector<std::string>{
           "2021-01-01 02:00:00 +05:99", // offset minutes past 59
           "2021-01-01 02:00:00 +0599", // same, without the colon
           "2021-01-01 02:00:00 bogus", // neither an offset nor an alias
           "2021-01-01 02:00:00", // no zone where the format requires one
       }) {
    SCOPED_TRACE(invalid);
    auto input = varcharInput(invalid);
    auto exprSet = compileExpression(
        "parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ')",
        asRowType(input->type()));
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    VELOX_ASSERT_THROW(
        evaluate(*exprSet, input), "Invalid format for parse_datetime");
  }
}

// CPU requires the whole input to be consumed: after the last format token,
// anything left over is a parse failure (DateTimeFormatter::parse, "Ensure all
// input was consumed"). cudf::strings::is_timestamp instead stops at the last
// format item, so text beyond it is unchecked -- both trailing junk and junk
// between the time and a valid offset are accepted.
//
// Closing this needs a shape regex derived from the Joda format so the match
// can be anchored at both ends. That regex has to be emitted by the same pass
// that builds strptime_, or the two drift; jodaToStrftime is also where T19's
// quote and '%' handling lands, so both are done together there rather than
// restructuring that function twice.
TEST_F(TimezoneFunctionTest, parseDatetimeTrailingTextThrowsLikeCpu) {
  for (const auto& [invalid, format] :
       std::vector<std::pair<std::string, std::string>>{
           {"2021-01-01 02:00:00 junk", "yyyy-MM-dd HH:mm:ss"},
           {"2021-01-01 02:00:00 junk +05:30", "yyyy-MM-dd HH:mm:ss ZZ"},
       }) {
    SCOPED_TRACE(invalid);
    auto input = varcharInput(invalid);
    auto exprSet = compileExpression(
        "parse_datetime(c0, '" + format + "')", asRowType(input->type()));
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    EXPECT_ANY_THROW(evaluate(*exprSet, input));
  }
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

// Trailing 'T' with no time component: CPU treats it as the date at midnight;
// the current regex needs 2 digits after T -> NULL. (Oracle: DateTimeFunctions
// fromIso8601Timestamp accepts "1970-01-01T"/"1970-01T"/"1970T".)
TEST_F(TimezoneFunctionTest, fromIso8601TrailingT) {
  assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput("2021-01-01T"));
}

// Year-only and year-month: CPU -> start-of-period midnight GMT.
TEST_F(TimezoneFunctionTest, fromIso8601YearOnly) {
  assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput("2021"));
}
TEST_F(TimezoneFunctionTest, fromIso8601YearMonth) {
  assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput("2021-07"));
}

// No time, explicit offset ("<date>T+01:00", "<year>T+14:00"): CPU applies the
// offset to the start-of-period wall clock.
TEST_F(TimezoneFunctionTest, fromIso8601DateThenOffset) {
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("2021-01-01T+01:00"));
}

// Time-only ("Thh[:mm[:ss[.fff]]]" [offset]): CPU defaults the date to
// 1970-01-01; the date-anchored regex needs a leading year -> NULL today.
TEST_F(TimezoneFunctionTest, fromIso8601TimeOnly) {
  assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput("T11:38:56"));
}
TEST_F(TimezoneFunctionTest, fromIso8601TimeOnlyHourOnly) {
  assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput("T11"));
}
TEST_F(TimezoneFunctionTest, fromIso8601TimeOnlyWithOffset) {
  assertMatchesCpu(
      "from_iso8601_timestamp(c0)", varcharInput("T11:38:56.123-14:00"));
}

// Malformed input: CPU (util::fromTimestampWithTimezoneString) throws; the GPU
// must not silently return NULL. Red until the throw block lands.
// Out-of-range date, time and offset fields must be rejected, as CPU rejects
// them. The regex used [0-9]{2} for the hour, minute, second and offset
// minutes, so each form below parsed to a wrong value instead of failing:
// "T25:00" rolled into the next day, and "+05:99" passed the +/-840 magnitude
// check as "+06:39".
//
// The two date forms guard the interaction between the day and the offset:
// both groups are optional, so a day that fails its range could still match as
// a negative offset and be misclassified instead of rejected. On the
// extreme-year path that made "+12021-01-32" an unsupported year; offset hours
// are bounded to 00-14 so neither group accepts it.
//
// Raised independently by both reviewers; one regex change closes both threads.
TEST_F(TimezoneFunctionTest, fromIso8601OutOfRangeFieldsThrowLikeCpu) {
  for (const auto& invalid : std::vector<std::string>{
           "2021-13-01", // month past 12
           "2021-01-32", // day past 31
           "2021-01-02T25:00", // hour past 23
           "2021-01-02T12:60", // minute past 59
           "2021-01-02T12:30:61", // second past the leap second
           "2021-01-02T12:30:45+05:99", // offset minutes past 59
           "2021-01-02T:30", // minute without an hour
           "2021-01-02T12:30.5", // fraction not on the seconds field
           "2021-01-01.5", // fraction with no time at all
           "2021.5", // fraction on a year-only form
       }) {
    SCOPED_TRACE(invalid);
    auto input = varcharInput(invalid);
    auto exprSet = compileExpression(
        "from_iso8601_timestamp(c0)", asRowType(input->type()));
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    VELOX_ASSERT_THROW(
        evaluate(*exprSet, input),
        "Unable to parse timestamp value in from_iso8601_timestamp");
  }
}

// A malformed field in a string whose year is out of range must be reported as
// a parse error, not as an unsupported year. Both programs carry the same tail,
// so such a string matches neither and is malformed -- which is what CPU calls
// it. The extreme-year path raises before the calendar round-trip runs, so the
// ranges in the regex are the only thing that classifies this correctly.
TEST_F(TimezoneFunctionTest, fromIso8601ExtremeYearWithBadFieldIsMalformed) {
  for (const auto& invalid : std::vector<std::string>{
           "+12021-13-01", // month past 12
           "+12021-01-32", // day past 31
           "+12021-01-01T25:00", // hour past 23
       }) {
    SCOPED_TRACE(invalid);
    auto input = varcharInput(invalid);
    auto exprSet = compileExpression(
        "from_iso8601_timestamp(c0)", asRowType(input->type()));
    EXPECT_ANY_THROW(
        functions::test::FunctionBaseTest::evaluate(*exprSet, input));
    VELOX_ASSERT_THROW(
        evaluate(*exprSet, input),
        "Unable to parse timestamp value in from_iso8601_timestamp");
  }
}

// The forms next to those boundaries must keep parsing, so the tightened ranges
// cannot be over-tightened unnoticed.
//
// The seconds field still admits 60 on purpose: CPU accepts a leap second and
// normalises "T12:30:60" to 12:31:00, so rejecting it would trade one parity
// break for another.
TEST_F(TimezoneFunctionTest, fromIso8601InRangeFieldBoundariesMatchCpu) {
  for (const auto& valid : std::vector<std::string>{
           "2021-01-02T23:59:59",
           "2021-01-02T12:30:60", // leap second, normalised by CPU
           "2021-01-02T12:30:45+14:00", // largest legal offset
           "2021-01-02T12:30:45-00:30", // sub-hour negative offset
       }) {
    SCOPED_TRACE(valid);
    assertMatchesCpu("from_iso8601_timestamp(c0)", varcharInput(valid));
  }
}

TEST_F(TimezoneFunctionTest, fromIso8601MalformedThrowsLikeCpu) {
  auto input = varcharInput("not-a-timestamp");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Space separator: CPU rejects "yyyy-MM-dd HH:mm" (only 'T' is legal). GPU now
// rejects it too (regex tightened to 'T'; unmatched non-null row -> throw).
TEST_F(TimezoneFunctionTest, fromIso8601SpaceSeparatorThrowsLikeCpu) {
  auto input = varcharInput("2021-01-02 11:38");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Empty string and a bare "T": both malformed on CPU.
TEST_F(TimezoneFunctionTest, fromIso8601EmptyStringThrowsLikeCpu) {
  auto input = varcharInput("");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}
TEST_F(TimezoneFunctionTest, fromIso8601BareTThrowsLikeCpu) {
  auto input = varcharInput("T");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Well-formed shape but a nonexistent calendar date: CPU rejects both
// (isValidDate -- month > 12 / day past the month's length, leap year aware).
// The regex alone accepts the two-digit fields, and cudf::to_timestamps would
// silently normalize them (13 -> next year, Feb 30 -> March), so the GPU must
// detect the normalization and throw rather than return a wrong value.
TEST_F(TimezoneFunctionTest, fromIso8601InvalidMonthDayThrowsLikeCpu) {
  auto input = varcharInput("2021-13-45");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}
TEST_F(TimezoneFunctionTest, fromIso8601InvalidFebruaryThrowsLikeCpu) {
  auto input = varcharInput("2021-02-30");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_ANY_THROW(
      functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  EXPECT_ANY_THROW(evaluate(*exprSet, input));
}

// Extreme-but-valid years (5-digit / signed): CPU parses them; the GPU
// (to_timestamps is int16, <=4-digit %Y) cannot, so it throws VELOX_NYI rather
// than returning NULL or a wrong value -- the query stops (owner decision
// 2026-07-16, no silent NULL). CPU is asserted to succeed to document the
// divergence. Literals are the max/min still in CPU's non-overflow range.
TEST_F(TimezoneFunctionTest, fromIso8601FiveDigitYearNyiOnGpu) {
  auto input = varcharInput("73326-09-11T20:14:45.247");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_NO_THROW(functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  VELOX_ASSERT_THROW(evaluate(*exprSet, input), "does not support years");
}
TEST_F(TimezoneFunctionTest, fromIso8601NegativeYearNyiOnGpu) {
  auto input = varcharInput("-69387-04-22T03:45:14.752");
  auto exprSet =
      compileExpression("from_iso8601_timestamp(c0)", asRowType(input->type()));
  EXPECT_NO_THROW(functions::test::FunctionBaseTest::evaluate(*exprSet, input));
  VELOX_ASSERT_THROW(evaluate(*exprSet, input), "does not support years");
}

// Control: a genuine SQL NULL row stays NULL -- it must trip neither throw.
TEST_F(TimezoneFunctionTest, fromIso8601NullRowStaysNull) {
  auto input = makeRowVector(
      {makeNullableFlatVector<std::string>({std::nullopt}, VARCHAR())});
  assertMatchesCpu("from_iso8601_timestamp(c0)", input);
}

// Direct tests of the conversion API in TimezoneConversion.h. Every other test
// in this file reaches it through a SQL function, and each of those constrains
// what can arrive: date_trunc only ever supplies a truncation boundary, and
// from_iso8601_timestamp only a wall clock it parsed from text. These call the
// two functions with arbitrary instants, and assert the round trip, which no
// SQL path can express.
//
// The values are America/Los_Angeles in 2021. The clocks jump forward at
// 2021-03-14 10:00Z, so local 01:00 is PST (-08:00) and local 03:00 is PDT
// (-07:00) and local 02:30 never happens. They go back at 2021-11-07 09:00Z, so
// local 01:30 occurs twice, at 08:30Z and again at 09:30Z.

constexpr const char* kLosAngelesZone = "America/Los_Angeles";
constexpr int64_t kUtcBeforeSpringForward = 1'615'712'400'000; // 09:00Z
constexpr int64_t kUtcAfterSpringForward = 1'615'716'000'000; // 10:00Z
constexpr int64_t kLocalInGap = 1'615'689'000'000; // 02:30, nonexistent
constexpr int64_t kLocalInOverlap = 1'636'248'600'000; // 01:30, occurs twice

// CPU's counterpart of toLocalTimestamp: shifts a GMT instant to the wall clock
// at the zone. Used as the oracle so the expectations come from Velox rather
// than from arithmetic done here, as everywhere else in this file.
int64_t cpuToLocalMillis(int64_t utcMillis, const char* zone) {
  auto timestamp = Timestamp::fromMillis(utcMillis);
  timestamp.toTimezone(*tz::locateZone(zone));
  return timestamp.toMillis();
}

// CPU's counterpart of toUtcTimestamp, and the function toUtcTimestamp is
// documented as matching. It raises for a local time that does not exist.
int64_t cpuToUtcMillis(int64_t localMillis, const char* zone) {
  auto timestamp = Timestamp::fromMillis(localMillis);
  timestamp.toGMT(*tz::locateZone(zone));
  return timestamp.toMillis();
}

// Builds a TIMESTAMP_MILLISECONDS column from host values.
std::unique_ptr<cudf::column> millisColumn(
    const std::vector<int64_t>& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      column->mutable_view().data<int64_t>(),
      values.data(),
      values.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  stream.synchronize();
  return column;
}

std::vector<int64_t> millisToHost(
    const cudf::column_view& view,
    rmm::cuda_stream_view stream) {
  std::vector<int64_t> host(view.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      host.data(),
      view.data<int64_t>(),
      host.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return host;
}

TEST_F(TimezoneFunctionTest, toLocalTimestampShiftsAcrossTransition) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto utc = millisColumn(
      {kUtcBeforeSpringForward, kUtcAfterSpringForward}, stream, mr);

  auto local = toLocalTimestamp(utc->view(), kLosAngelesZone, stream, mr);

  EXPECT_THAT(
      millisToHost(local->view(), stream),
      testing::ElementsAre(
          cpuToLocalMillis(kUtcBeforeSpringForward, kLosAngelesZone),
          cpuToLocalMillis(kUtcAfterSpringForward, kLosAngelesZone)));
}

// The round trip is the property no SQL path states: converting to local and
// back has to land on the instant it started from, on both sides of a
// transition where the offset differs.
TEST_F(TimezoneFunctionTest, toUtcTimestampInvertsToLocalTimestamp) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const std::vector<int64_t> instants{
      kUtcBeforeSpringForward, kUtcAfterSpringForward};
  auto utc = millisColumn(instants, stream, mr);

  auto local = toLocalTimestamp(utc->view(), kLosAngelesZone, stream, mr);
  auto roundTripped =
      toUtcTimestamp(local->view(), kLosAngelesZone, stream, mr);

  EXPECT_THAT(
      millisToHost(roundTripped->view(), stream),
      testing::ElementsAre(instants[0], instants[1]));
}

TEST_F(TimezoneFunctionTest, toUtcTimestampGapRaises) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto local = millisColumn({kLocalInGap}, stream, mr);

  EXPECT_ANY_THROW(cpuToUtcMillis(kLocalInGap, kLosAngelesZone));
  VELOX_ASSERT_THROW(
      toUtcTimestamp(local->view(), kLosAngelesZone, stream, mr),
      "does not exist in the time zone");
}

TEST_F(TimezoneFunctionTest, toUtcTimestampOverlapPicksEarliest) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto local = millisColumn({kLocalInOverlap}, stream, mr);

  auto utc = toUtcTimestamp(local->view(), kLosAngelesZone, stream, mr);

  EXPECT_THAT(
      millisToHost(utc->view(), stream),
      testing::ElementsAre(cpuToUtcMillis(kLocalInOverlap, kLosAngelesZone)));
}

// A null row is not a gap, so an all-null column must convert without raising
// -- the property that lets a caller null out the rows it does not want
// checked.
TEST_F(TimezoneFunctionTest, toUtcTimestampNullRowIsNotAGap) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto local = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
      1,
      cudf::mask_state::ALL_NULL,
      stream,
      mr);

  auto utc = toUtcTimestamp(local->view(), kLosAngelesZone, stream, mr);

  EXPECT_EQ(utc->size(), 1);
  EXPECT_EQ(utc->null_count(), 1);
}

} // namespace
