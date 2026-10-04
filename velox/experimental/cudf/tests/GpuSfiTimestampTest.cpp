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

#include "velox/experimental/cudf/tests/utils/GpuSfiParityTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <folly/ScopeGuard.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

using test_utils::GpuSfiParityTestBase;
using test_utils::nextRandom;

// The calendar fields GPU SFI registers over DATE, TIMESTAMP and TIMESTAMP
// WITH TIME ZONE, over a column c0 of that type. Every alias is named, since
// each is a registration of its own.
const std::vector<std::string> kFieldCalls{
    "year(c0)",
    "quarter(c0)",
    "month(c0)",
    "day(c0)",
    "day_of_month(c0)",
    "day_of_week(c0)",
    "dow(c0)",
    "day_of_year(c0)",
    "doy(c0)",
    "week(c0)",
    "week_of_year(c0)",
    "year_of_week(c0)",
    "yow(c0)",
    "hour(c0)",
    "minute(c0)",
    "second(c0)",
    "millisecond(c0)",
};

// The TIMESTAMP WITH TIME ZONE functions that read the instant or the
// embedded zone rather than a calendar field.
const std::vector<std::string> kTimestampWithTimeZoneInstantCalls{
    "to_unixtime(c0)",
    "timezone_hour(c0)",
    "timezone_minute(c0)",
    "at_timezone(c0, 'America/Los_Angeles')",
    "at_timezone(c0, '+05:30')",
    "at_timezone(c0, 'UTC')",
};

// The units date_trunc accepts for a TIMESTAMP, and the ones its DATE overload
// accepts.
const std::vector<std::string> kTimestampTruncationUnits{
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year"};
const std::vector<std::string> kDateUnits{
    "day",
    "week",
    "month",
    "quarter",
    "year"};

// The spans TIMESTAMP WITH TIME ZONE is compared over, not bound to a cuDF
// timestamp unit: past the nanosecond column's and the zones' own offset
// history, and past 2800, where the device table ends and the lookup folds.
const std::vector<std::pair<int32_t, int32_t>> kTimestampWithTimeZoneSpans{
    {1700, 2400},
    {2700, 4000},
};

std::vector<std::string> dateTruncCalls(const std::vector<std::string>& units) {
  std::vector<std::string> calls;
  for (const auto& unit : units) {
    calls.push_back(fmt::format("date_trunc('{}', c0)", unit));
  }
  return calls;
}

// `values` repeated up to `size` entries.
template <typename T>
std::vector<std::optional<T>> cycled(
    const std::vector<std::optional<T>>& values,
    size_t size) {
  std::vector<std::optional<T>> result;
  for (size_t i = 0; i < size; ++i) {
    result.push_back(values[i % values.size()]);
  }
  return result;
}

class GpuSfiTimestampTest : public GpuSfiParityTestBase {
 protected:
  // An instant and the zone a TIMESTAMP WITH TIME ZONE column packs it with; a
  // TIMESTAMP column holds the instant alone.
  struct ZonedInstant {
    int16_t zoneId;
    Timestamp instant;
  };

  // Every call over a column of the input's type, with the date_trunc rows
  // the CPU raises on left out: truncating to a local midnight an offset
  // increase skipped, as Europe/London's 1847-12-01 did, is a user error
  // there, and the GPU declines the row.
  void assertCallsMatchCpu(
      const std::vector<std::string>& calls,
      const std::vector<std::string>& truncationUnits,
      const RowVectorPtr& input) {
    for (const auto& sql : calls) {
      SCOPED_TRACE(sql);
      assertGpuMatchesCpu(sql, input);
    }
    for (const auto& sql : dateTruncCalls(truncationUnits)) {
      SCOPED_TRACE(sql);
      assertGpuMatchesCpu(sql, rowsTheCpuAnswers(sql, input));
    }
  }

  // Every TIMESTAMP WITH TIME ZONE call, under the session settings in force.
  void assertTimestampWithTimeZoneCallsMatchCpu(const RowVectorPtr& input) {
    auto calls = kFieldCalls;
    calls.insert(
        calls.end(),
        kTimestampWithTimeZoneInstantCalls.begin(),
        kTimestampWithTimeZoneInstantCalls.end());
    assertCallsMatchCpu(calls, kTimestampTruncationUnits, input);
  }

  // Runs `body` over every zone of each span as the session zone, with
  // TIMESTAMP columns in the span's unit: a nanosecond column, cuDF's default,
  // spans 1677 to 2262, entered two years in since truncating to the year in a
  // zone west of UTC moves the first instants a year back; a microsecond
  // column reaches past 2800, where the device table ends and the lookup folds
  // an instant back by whole 400-year cycles.
  template <typename Body>
  void forEachTimestampSpan(Body&& body) {
    struct Span {
      cudf::type_id unit;
      int32_t fromYear;
      int32_t toYear;
      std::vector<std::string> zones;
    };
    const std::vector<Span> spans{
        {cudf::type_id::TIMESTAMP_NANOSECONDS, 1680, 2262, timeZones()},
        {cudf::type_id::TIMESTAMP_MICROSECONDS,
         2700,
         4000,
         {"America/Los_Angeles", "Australia/Lord_Howe", "Europe/London"}},
    };
    auto& config = CudfConfig::getInstance();
    const auto unit = config.timestampUnit;
    SCOPE_EXIT {
      config.timestampUnit = unit;
    };
    for (const auto& span : spans) {
      config.timestampUnit = span.unit;
      for (const auto& timeZone : span.zones) {
        SCOPED_TRACE(
            fmt::format("{} {}-{}", timeZone, span.fromYear, span.toYear));
        setSession(
            timeZone,
            /*adjustTimestampToTimezone=*/true,
            /*legacyTimestampWithTimezone=*/true);
        body(timeZone, span.fromYear, span.toYear);
      }
    }
  }

  // instants() of every zone, each with the zone's id, and a null per zone.
  std::vector<std::optional<ZonedInstant>> zonedInstants(
      const std::vector<std::string>& zones,
      int32_t fromYear,
      int32_t toYear,
      int32_t numSpread) {
    std::vector<std::optional<ZonedInstant>> values;
    for (const auto& timeZone : zones) {
      const auto zoneId = tz::getTimeZoneID(timeZone);
      for (const auto& instant :
           instants(timeZone, fromYear, toYear, numSpread)) {
        values.push_back(ZonedInstant{zoneId, instant});
      }
      values.push_back(std::nullopt);
    }
    return values;
  }

  // A column of `type` over `values`: TIMESTAMP holds the instants, in whole
  // microseconds when the configured column unit is microseconds so that the
  // CPU is handed the same values; TIMESTAMP WITH TIME ZONE packs each
  // instant's milliseconds with its zone id.
  VectorPtr makeColumn(
      const TypePtr& type,
      const std::vector<std::optional<ZonedInstant>>& values) {
    if (type->isTimestamp()) {
      const bool microseconds = CudfConfig::getInstance().timestampUnit ==
          cudf::type_id::TIMESTAMP_MICROSECONDS;
      std::vector<std::optional<Timestamp>> column;
      for (const auto& value : values) {
        if (!value.has_value()) {
          column.push_back(std::nullopt);
        } else if (microseconds) {
          column.emplace_back(Timestamp(
              value->instant.getSeconds(),
              value->instant.getNanos() / 1'000 * 1'000));
        } else {
          column.push_back(value->instant);
        }
      }
      return makeNullableFlatVector<Timestamp>(column);
    }
    std::vector<std::optional<int64_t>> column;
    for (const auto& value : values) {
      column.push_back(
          value.has_value() ? std::optional<int64_t>(pack(
                                  value->instant.toMillis(), value->zoneId))
                            : std::nullopt);
    }
    return makeNullableFlatVector<int64_t>(column, TIMESTAMP_WITH_TIME_ZONE());
  }
};

// Each zone is the session zone in turn, so every TIMESTAMP call reads in it;
// a result the column cannot hold is declined rather than compared.
TEST_F(GpuSfiTimestampTest, timestampCallsMatchCpu) {
  auto calls = kFieldCalls;
  calls.push_back("to_unixtime(c0)");
  forEachTimestampSpan(
      [&](const std::string& timeZone, int32_t fromYear, int32_t toYear) {
        assertCallsMatchCpu(
            calls,
            kTimestampTruncationUnits,
            makeRowVector({makeColumn(
                TIMESTAMP(),
                zonedInstants({timeZone}, fromYear, toYear, 20'000))}));
      });
}

// Without adjust_timestamp_to_session_timezone, Velox reads TIMESTAMP as UTC
// whatever the session time zone, and so must GPU SFI.
TEST_F(GpuSfiTimestampTest, timestampCallsReadUtcWithoutAdjustment) {
  setSession(
      "America/Los_Angeles",
      /*adjustTimestampToTimezone=*/false,
      /*legacyTimestampWithTimezone=*/true);
  assertCallsMatchCpu(
      kFieldCalls,
      kTimestampTruncationUnits,
      makeRowVector({makeFlatVector(
          instants("America/Los_Angeles", 1990, 2030, 20'000))}));
}

// DATE has no time of day and no zone: the fields and the truncations read
// the calendar alone, whatever the session. The days are the epoch and its
// neighbours, both sides of a leap day, a century non-leap year, dates far
// outside any real query's range, and a spread.
TEST_F(GpuSfiTimestampTest, dateCallsMatchCpu) {
  setSession(
      "America/Los_Angeles",
      /*adjustTimestampToTimezone=*/true,
      /*legacyTimestampWithTimezone=*/true);
  std::vector<int32_t> days{
      0,
      -1,
      1,
      365,
      366,
      -365,
      8'035,
      10'592,
      19'000,
      -25'567,
      -700'000,
      700'000,
      250'000,
      -250'000};
  for (const auto& instant : instants("UTC", 1680, 2262, 5'000)) {
    days.push_back(
        static_cast<int32_t>(
            instant.getSeconds() / Timestamp::kSecondsInDay -
            (instant.getSeconds() % Timestamp::kSecondsInDay < 0 ? 1 : 0)));
  }
  assertCallsMatchCpu(
      kFieldCalls,
      kDateUnits,
      makeRowVector({makeFlatVector<int32_t>(days, DATE())}));
}

// Mixed-zone columns under both renderings and every session zone, over each
// of kTimestampWithTimeZoneSpans.
TEST_F(GpuSfiTimestampTest, timestampWithTimeZoneCallsMatchCpu) {
  for (const auto& [fromYear, toYear] : kTimestampWithTimeZoneSpans) {
    const auto input = makeRowVector({makeColumn(
        TIMESTAMP_WITH_TIME_ZONE(),
        zonedInstants(timeZones(), fromYear, toYear, 2'000))});
    forEachSessionZone(
        [&] { assertTimestampWithTimeZoneCallsMatchCpu(input); });
  }
}

// The render zone is the session time zone whether or not
// adjust_timestamp_to_session_timezone is set, unlike TIMESTAMP's, and an
// empty session zone renders in UTC.
TEST_F(GpuSfiTimestampTest, renderZoneFollowsTheSessionTimeZoneAlone) {
  const auto input = makeRowVector({makeColumn(
      TIMESTAMP_WITH_TIME_ZONE(),
      zonedInstants(timeZones(), 1990, 2030, 500))});
  const std::vector<std::pair<std::string, bool>> sessions{
      {"America/Los_Angeles", true},
      {"America/Los_Angeles", false},
      {"", true},
      {"", false},
  };
  for (const auto& [sessionTimeZone, legacy] : sessions) {
    SCOPED_TRACE(
        fmt::format("session='{}' legacy={}", sessionTimeZone, legacy));
    setSession(sessionTimeZone, /*adjustTimestampToTimezone=*/false, legacy);
    assertTimestampWithTimeZoneCallsMatchCpu(input);
  }
}

// from_unixtime builds TIMESTAMP WITH TIME ZONE values from doubles, with the
// zone named, given as a constant offset or as offset columns read per row on
// the device, and TIMESTAMP values with no zone. The doubles cover the
// sub-millisecond rounding, both signs and NaN, within the years a nanosecond
// TIMESTAMP column holds.
TEST_F(GpuSfiTimestampTest, fromUnixtimeMatchesCpu) {
  std::vector<std::optional<double>> values{
      0.0,
      -0.0,
      0.0004,
      0.0005,
      0.0015,
      -0.0005,
      -0.0015,
      -1.0,
      1.9999,
      -1.9999,
      1'700'000'000.123,
      -2'208'988'800.5,
      std::numeric_limits<double>::quiet_NaN(),
      std::nullopt,
  };
  uint64_t state = 0x9e37'79b9'7f4a'7c15;
  for (int32_t i = 0; i < 5'000; ++i) {
    // Roughly 1700 to 2239, with a fractional part.
    const auto random = nextRandom(state);
    values.push_back(
        static_cast<double>(static_cast<int64_t>(random) % 17'000'000'000) -
        8'500'000'000.0 + static_cast<double>(state & 0xffff) / 65'536.0);
  }
  const auto input = makeRowVector({
      makeNullableFlatVector<double>(values),
      makeNullableFlatVector<int64_t>(
          cycled<int64_t>({5, -8, 0, 14, 1, std::nullopt}, values.size())),
      makeNullableFlatVector<int64_t>(
          cycled<int64_t>({30, 0, 0, 0, std::nullopt, 15}, values.size())),
  });

  const std::vector<std::string> calls{
      "from_unixtime(c0, 'America/Los_Angeles')",
      "from_unixtime(c0, '+05:30')",
      "from_unixtime(c0, 'UTC')",
      "from_unixtime(c0, 5, 30)",
      "from_unixtime(c0, -8, 0)",
      "from_unixtime(c0, 0, 0)",
      "from_unixtime(c0, 14, 0)",
      "from_unixtime(c0, -14, 0)",
      "from_unixtime(c0, c1, c2)",
      "from_unixtime(c0)",
  };
  for (const auto& sessionTimeZone : sessionTimeZones()) {
    SCOPED_TRACE(sessionTimeZone);
    setSession(
        sessionTimeZone,
        /*adjustTimestampToTimezone=*/true,
        /*legacyTimestampWithTimezone=*/true);
    assertCallsMatchCpu(calls, {}, input);
  }
}

// An instant the output column's unit cannot hold is declined rather than
// written: from_unixtime(1e18) clamps to the last millisecond a Timestamp
// holds, far past a nanosecond column's range. An operator re-evaluates a
// declined row on the CPU; the bare evaluator here reports it instead.
TEST_F(GpuSfiTimestampTest, timestampResultBeyondTheColumnUnitIsDeclined) {
  const auto input = makeRowVector({makeFlatVector<double>({0.0, 1e18})});
  VELOX_ASSERT_THROW(
      evaluate(
          makeTypedExpr("from_unixtime(c0)", asRowType(input->type())), input),
      "A GPU SFI kernel declined a row");
}

// The comparisons order TIMESTAMP WITH TIME ZONE by instant: the same instant
// in two zones is equal, in both orders since the zone keys sort one way.
// Instants a millisecond apart, the extremes a packed value can hold, and
// nulls on either side complete the rows.
TEST_F(GpuSfiTimestampTest, timestampWithTimeZoneComparisonsMatchCpu) {
  std::vector<std::optional<int64_t>> left;
  std::vector<std::optional<int64_t>> right;
  const std::vector<int64_t> millisUtc{
      0,
      1,
      -1,
      1'700'000'000'000,
      -62'135'596'800'000,
      kMaxMillisUtc,
      kMinMillisUtc,
  };
  for (const auto& leftZone : timeZones()) {
    const auto leftId = tz::getTimeZoneID(leftZone);
    for (const auto& rightZone : timeZones()) {
      const auto rightId = tz::getTimeZoneID(rightZone);
      for (const int64_t millis : millisUtc) {
        left.push_back(pack(millis, leftId));
        right.push_back(pack(millis, rightId));
        if (millis < kMaxMillisUtc) {
          left.push_back(pack(millis, leftId));
          right.push_back(pack(millis + 1, rightId));
          left.push_back(pack(millis + 1, leftId));
          right.push_back(pack(millis, rightId));
        }
      }
    }
    left.push_back(std::nullopt);
    right.push_back(pack(0, leftId));
    left.push_back(pack(0, leftId));
    right.push_back(std::nullopt);
    left.push_back(std::nullopt);
    right.push_back(std::nullopt);
  }
  const auto input = makeRowVector({
      makeNullableFlatVector<int64_t>(left, TIMESTAMP_WITH_TIME_ZONE()),
      makeNullableFlatVector<int64_t>(right, TIMESTAMP_WITH_TIME_ZONE()),
  });

  const std::vector<std::string> calls{
      "c0 = c1",
      "c0 <> c1",
      "c0 < c1",
      "c0 <= c1",
      "c0 > c1",
      "c0 >= c1",
      "c0 between c1 and c1",
      "c0 between c1 and c0",
      "c1 between c0 and c0",
  };
  assertCallsMatchCpu(calls, {}, input);
}

} // namespace
} // namespace facebook::velox::cudf_velox
