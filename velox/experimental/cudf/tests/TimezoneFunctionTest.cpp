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

TEST_F(TimezoneFunctionTest, toIso8601FromTimestampWithTimeZone) {
  // to_iso8601(timestamp with time zone) -> varchar.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("to_iso8601(c0)", input);
}

TEST_F(TimezoneFunctionTest, formatDatetimeOfTimestampWithTimeZone) {
  // format_datetime(timestamp with time zone, varchar) -> varchar.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertMatchesCpu("format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ')", input);
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
