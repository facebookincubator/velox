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

// Class B reproducers: hard failures from the GPU (cuDF) expression evaluator
// for the TIMESTAMP WITH TIME ZONE function family.
//
// None of these functions are registered on the GPU path. Each works on CPU
// (it is registered by registerAllScalarFunctions), but forcing GPU evaluation
// throws from FunctionExpression::eval:
//
//   "Unsupported expression for recursive evaluation: <name>"
//   (velox/experimental/cudf/expression/ExpressionEvaluator.cpp)
//
// This is true both for functions that produce a TIMESTAMP WITH TIME ZONE from
// plain double/varchar inputs and for functions that consume a TIMESTAMP WITH
// TIME ZONE column. The input conversion (Velox -> cuDF) is *not* the failure
// site: cuDF has no TIMESTAMP WITH TIME ZONE type, but a TIMESTAMP WITH TIME
// ZONE column is silently reinterpreted as its physical BIGINT and converts
// without error (see timestampWithTimeZoneColumnSilentlyAcceptedAsBigint). The
// failure surfaces only when an unsupported function is evaluated.
//
// CudfFunctionBaseTest::evaluate forces GPU execution (it does not consult
// allowCpuFallback), so these unsupported expressions surface as real throws.
//
// IMPORTANT: these tests assert the *current* gap, so they pass only while the
// gap exists. They assert that GPU evaluation throws; once these functions (and
// a TIMESTAMP WITH TIME ZONE type) are implemented on GPU they will (correctly)
// start failing. At that point replace each assertGpuThrows with a correctness
// assertion that the GPU result matches CPU. A green throw assertion means the
// function is still unsupported on GPU.
//
// These tests require a GPU and are labeled cuda_driver; they will not run in a
// CPU-only environment.
//
// Note: Presto's with_timezone is intentionally not covered here. Velox does
// not register a with_timezone scalar function, so it cannot be compiled on CPU
// and is out of scope for a GPU-vs-CPU gap.

#include <gmock/gmock.h>

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/tz/TimeZoneMap.h"

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

namespace {

class TimezoneUnsupportedTest : public cudf_velox::CudfFunctionBaseTest {
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

  // Compiles the expression and forces GPU evaluation. Returns the error
  // message if it throws, or std::nullopt if it succeeds. Logs the outcome so
  // the concrete failure is visible in the test output.
  std::optional<std::string> gpuEvaluationError(
      const std::string& expr,
      const RowVectorPtr& input) {
    auto exprSet = compileExpression(expr, asRowType(input->type()));
    try {
      evaluate(*exprSet, input);
    } catch (const std::exception& e) {
      LOG(INFO) << "[" << expr << "] threw: " << e.what();
      return std::string(e.what());
    }
    LOG(INFO) << "[" << expr << "] did NOT throw on GPU";
    return std::nullopt;
  }

  // Asserts GPU evaluation throws and that the message contains substr.
  void assertGpuThrows(
      const std::string& expr,
      const RowVectorPtr& input,
      const std::string& substr) {
    auto error = gpuEvaluationError(expr, input);
    ASSERT_TRUE(error.has_value())
        << expr << " unexpectedly evaluated on GPU; expected a throw.";
    EXPECT_THAT(*error, testing::HasSubstr(substr));
  }

  // Substring of the failure raised by FunctionExpression::eval for functions
  // with no GPU implementation.
  static constexpr const char* kUnsupportedRecursive =
      "Unsupported expression for recursive evaluation";
};

// A TIMESTAMP WITH TIME ZONE column is silently reinterpreted as its physical
// BIGINT during Velox->cuDF conversion: a bare passthrough does not throw. This
// documents that the gap is the absence of any TIMESTAMP WITH TIME ZONE
// semantics on GPU, not an interop-level type rejection -- which is why every
// timezone-aware function below has to be evaluated and then fails.
TEST_F(TimezoneUnsupportedTest, timestampWithTimeZoneColumnSilentlyAcceptedAsBigint) {
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  EXPECT_FALSE(gpuEvaluationError("c0", input).has_value())
      << "Expected the TIMESTAMP WITH TIME ZONE column to be silently accepted "
         "as BIGINT (no throw) on the GPU passthrough path.";
}

// Functions that consume a TIMESTAMP WITH TIME ZONE column. The column converts
// to cuDF (as BIGINT) without error; the function itself is unsupported.

TEST_F(TimezoneUnsupportedTest, toUnixtimeFromTimestampWithTimeZone) {
  // CPU: to_unixtime(timestamp with time zone) -> double works.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertGpuThrows("to_unixtime(c0)", input, kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, atTimezone) {
  // CPU: at_timezone(timestamp with time zone, varchar) works.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertGpuThrows(
      "at_timezone(c0, 'America/New_York')", input, kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, timezoneHour) {
  // CPU: timezone_hour(timestamp with time zone) -> bigint works.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertGpuThrows("timezone_hour(c0)", input, kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, timezoneMinute) {
  // CPU: timezone_minute(timestamp with time zone) -> bigint works.
  auto input = timestampWithTimeZoneInput(1'609'466'400'000, "Asia/Kolkata");
  assertGpuThrows("timezone_minute(c0)", input, kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, toIso8601FromTimestampWithTimeZone) {
  // CPU: to_iso8601(timestamp with time zone) -> varchar works.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertGpuThrows("to_iso8601(c0)", input, kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, formatDatetimeOfTimestampWithTimeZone) {
  // CPU: format_datetime(timestamp with time zone, varchar) works.
  auto input =
      timestampWithTimeZoneInput(1'609'466'400'000, "America/Los_Angeles");
  assertGpuThrows(
      "format_datetime(c0, 'yyyy-MM-dd HH:mm:ss ZZ')",
      input,
      kUnsupportedRecursive);
}

// Functions that produce a TIMESTAMP WITH TIME ZONE from plain inputs. The
// inputs convert to cuDF fine; the unsupported function is the failure site.

TEST_F(TimezoneUnsupportedTest, fromUnixtimeWithZoneName) {
  // from_unixtime(double, varchar) -> timestamp with time zone.
  assertGpuThrows(
      "from_unixtime(c0, 'America/Los_Angeles')",
      doubleInput(1'609'466'400.0),
      kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, fromUnixtimeWithHoursMinutes) {
  // from_unixtime(double, bigint, bigint) -> timestamp with time zone.
  assertGpuThrows(
      "from_unixtime(c0, 7, 30)",
      doubleInput(1'609'466'400.0),
      kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, parseDatetime) {
  // parse_datetime(varchar, varchar) -> timestamp with time zone.
  assertGpuThrows(
      "parse_datetime(c0, 'yyyy-MM-dd HH:mm:ss')",
      varcharInput("2021-01-01 02:00:00"),
      kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, fromIso8601Timestamp) {
  // from_iso8601_timestamp(varchar) -> timestamp with time zone.
  assertGpuThrows(
      "from_iso8601_timestamp(c0)",
      varcharInput("2021-01-01T02:00:00+05:30"),
      kUnsupportedRecursive);
}

TEST_F(TimezoneUnsupportedTest, now) {
  // now()/current_timestamp -> timestamp with time zone. A dummy column sizes
  // the batch; the function itself is the unsupported expression.
  assertGpuThrows("now()", doubleInput(0.0), kUnsupportedRecursive);
}

} // namespace
