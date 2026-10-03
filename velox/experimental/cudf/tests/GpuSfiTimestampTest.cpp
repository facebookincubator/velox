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
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/external/tzdb/time_zone.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

// The wall-clock fields GPU SFI registers for TIMESTAMP, each compared with
// the CPU under the same session time zone.
const std::vector<std::string> kFields{
    "year",
    "quarter",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week",
    "year_of_week",
    "hour",
    "minute",
    "second",
    "millisecond",
};

// Fixed offsets, zones with and without daylight saving time, a 30-minute
// daylight shift (Lord_Howe), a 45-minute offset (Kathmandu), a zone that
// abolished daylight saving time (Sao_Paulo), and one that skipped a whole day
// (Apia).
const std::vector<std::string> kTimeZones{
    "UTC",
    "+05:30",
    "-08:00",
    "America/Los_Angeles",
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Kathmandu",
    "Australia/Lord_Howe",
    "America/Sao_Paulo",
    "Pacific/Apia",
};

class GpuSfiTimestampTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  static void TearDownTestCase() {
    unregisterCudf();
  }

  // Instants that stress the conversion: one second and one hour either side
  // of every offset change the zone records between `fromYear` and `toYear`,
  // sub-second values either side of the epoch, and a deterministic spread
  // over the whole span.
  static std::vector<Timestamp>
  instants(const std::string& timeZone, int32_t fromYear, int32_t toYear) {
    const auto from = date::sys_days{date::year{fromYear} / 1 / 1};
    const auto to = date::sys_days{date::year{toYear} / 1 / 1};
    const int64_t fromSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            from.time_since_epoch())
            .count();
    const int64_t toSeconds =
        std::chrono::duration_cast<std::chrono::seconds>(to.time_since_epoch())
            .count();

    std::vector<Timestamp> values{
        Timestamp(0, 0),
        Timestamp(0, 999'999'999),
        Timestamp(-1, 0),
        Timestamp(-1, 1),
        Timestamp(-1, 999'999'999),
        Timestamp(fromSeconds, 0),
        Timestamp(toSeconds - 1, 999'999'999),
    };

    if (const auto* zone = tz::locateZone(timeZone)->tz()) {
      auto info = zone->get_info(date::sys_seconds{from});
      while (info.end < to) {
        const int64_t change = info.end.time_since_epoch().count();
        for (const int64_t delta : {-3'600, -1, 0, 1, 3'599}) {
          values.emplace_back(change + delta, 123'000'000);
        }
        info = zone->get_info(info.end);
      }
    }

    uint64_t state = 0x9e37'79b9'7f4a'7c15;
    const auto span = static_cast<uint64_t>(toSeconds - fromSeconds);
    for (int32_t i = 0; i < 20'000; ++i) {
      state =
          state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
      values.emplace_back(
          fromSeconds + static_cast<int64_t>((state >> 11) % span),
          (state >> 7) % 1'000'000'000);
    }
    return values;
  }

  // Compares every field with the CPU, and checks that GPU SFI is the
  // evaluator that answered: with a session time zone in effect, nothing else
  // on the GPU may take these calls.
  void assertFieldsMatchCpu(
      const std::string& timeZone,
      int32_t fromYear,
      int32_t toYear) {
    setTimezone(timeZone);
    const auto input =
        makeRowVector({makeFlatVector(instants(timeZone, fromYear, toYear))});
    const auto rowType = asRowType(input->type());
    for (const auto& field : kFields) {
      const auto sql = field + "(c0)";
      SCOPED_TRACE(timeZone + " " + sql);
      const auto expr = test_utils::optimizeTypedExpr(
          sql, rowType, queryCtx_.get(), &execCtx_);
      const auto evaluator = createCudfExpression(
          expr, rowType, pool_.get(), queryCtx_->queryConfig());
      EXPECT_NE(dynamic_cast<GpuSfiExpression*>(evaluator.get()), nullptr);
      assertExpressionMatchesCpu(sql, input, rowType);
    }
  }
};

// A nanosecond column, cuDF's default, spans 1677 to 2262.
TEST_F(GpuSfiTimestampTest, fieldsMatchCpuInSessionTimeZone) {
  for (const auto& timeZone : kTimeZones) {
    assertFieldsMatchCpu(timeZone, 1678, 2262);
  }
}

// A microsecond column reaches past 2800, where the device table ends and the
// lookup folds an instant back by whole 400-year cycles.
TEST_F(GpuSfiTimestampTest, fieldsMatchCpuBeyondTheTabulatedRange) {
  auto& config = CudfConfig::getInstance();
  const auto unit = config.timestampUnit;
  config.timestampUnit = cudf::type_id::TIMESTAMP_MICROSECONDS;
  for (const auto& timeZone :
       {"America/Los_Angeles", "Australia/Lord_Howe", "Europe/London"}) {
    assertFieldsMatchCpu(timeZone, 2700, 4000);
  }
  config.timestampUnit = unit;
}

// Without adjust_timestamp_to_session_timezone, Velox reads TIMESTAMP as UTC
// whatever the session time zone, and so must GPU SFI.
TEST_F(GpuSfiTimestampTest, sessionTimeZoneIgnoredWithoutAdjustment) {
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kSessionTimezone, "America/Los_Angeles"},
      {core::QueryConfig::kAdjustTimestampToTimezone, "false"},
  });
  const auto input = makeRowVector(
      {makeFlatVector(instants("America/Los_Angeles", 1990, 2030))});
  const auto rowType = asRowType(input->type());
  for (const auto& field : kFields) {
    SCOPED_TRACE(field);
    assertExpressionMatchesCpu(field + "(c0)", input, rowType);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
