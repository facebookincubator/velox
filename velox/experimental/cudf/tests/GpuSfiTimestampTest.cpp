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
#include "velox/functions/lib/DateTimeUnitArithmetic.h"
#include "velox/functions/lib/TimeUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <initializer_list>
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
// accepts; date_add and date_diff accept the same and, over a TIMESTAMP, the
// millisecond.
const std::vector<std::string> kTimestampUnits{
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

// The largest count the date_add rows add as a literal of each unit: twenty
// years, so that a result stays within the column's span.
const std::vector<std::pair<std::string, int64_t>> kLargestCounts{
    {"millisecond", std::numeric_limits<int32_t>::max()},
    {"second", 600'000'000},
    {"minute", 10'000'000},
    {"hour", 170'000},
    {"day", 7'000},
    {"week", 1'000},
    {"month", 240},
    {"quarter", 80},
    {"year", 20},
};

// The counts the date_add column c1 cycles through, within twenty years for
// every unit, and the months the INTERVAL YEAR TO MONTH literals add and
// subtract: within a year, whole years and beyond.
const std::vector<std::optional<int64_t>>
    kCounts{0, 1, -1, 7, -7, 13, -13, 20, -20, std::nullopt};
const std::vector<int32_t> kMonthCounts{0, 1, 11, 12, 13, 25, 36};

// Seconds between c0 and the second column c2 of its type: either side of a
// minute, an hour, a day, a week, a month, a year and a leap cycle.
const std::vector<int64_t> kPairDeltas{
    0,
    1,
    -1,
    59,
    3'599,
    3'600,
    -3'600,
    86'399,
    86'400,
    -86'400,
    7 * 86'400,
    -7 * 86'400,
    31 * 86'400,
    -31 * 86'400,
    366 * 86'400,
    -366 * 86'400,
    1'461 * 86'400,
};

// The milliseconds the INTERVAL DAY TO SECOND column c3 cycles through: either
// side of a second, an hour, a day, a month and a year, and a null.
const std::vector<std::optional<int64_t>> kIntervalMillis{
    0,
    1,
    -1,
    999,
    1'000,
    -1'000,
    3'600'000,
    -3'600'000,
    86'400'000,
    -86'400'000,
    31LL * 86'400'000,
    -31LL * 86'400'000,
    366LL * 86'400'000,
    -366LL * 86'400'000,
    std::nullopt,
};

// The spans TIMESTAMP WITH TIME ZONE is compared over, not bound to a cuDF
// timestamp unit: past the nanosecond column's and the zones' own offset
// history, and past 2800, where the device table ends and the lookup folds.
const std::vector<std::pair<int32_t, int32_t>> kTimestampWithTimeZoneSpans{
    {1700, 2400},
    {2700, 4000},
};

// date_trunc of each unit over c0.
std::vector<std::string> dateTruncCalls(const std::vector<std::string>& units) {
  std::vector<std::string> calls;
  for (const auto& unit : units) {
    calls.push_back(fmt::format("date_trunc('{}', c0)", unit));
  }
  return calls;
}

// date_add of each unit over c0 with the count in c1 and as the unit's
// largest count of each sign, and date_diff of each unit between c0 and c2 in
// both orders.
std::vector<std::string> arithmeticCalls(
    const std::vector<std::string>& units) {
  std::vector<std::string> calls;
  for (const auto& unit : units) {
    const auto largest = std::find_if(
        kLargestCounts.begin(), kLargestCounts.end(), [&](const auto& entry) {
          return entry.first == unit;
        });
    calls.push_back(fmt::format("date_add('{}', c1, c0)", unit));
    calls.push_back(
        fmt::format("date_add('{}', {}, c0)", unit, largest->second));
    calls.push_back(
        fmt::format("date_add('{}', -{}, c0)", unit, largest->second));
    calls.push_back(fmt::format("date_diff('{}', c0, c2)", unit));
    calls.push_back(fmt::format("date_diff('{}', c2, c0)", unit));
  }
  return calls;
}

// The interval operators over c0, the INTERVAL DAY TO SECOND column c3 and
// the second column c2 of c0's type, and INTERVAL YEAR TO MONTH literals of
// kMonthCounts, added in both operand orders and subtracted.
std::vector<std::string> intervalCalls() {
  std::vector<std::string> calls{
      "c0 + c3", "c3 + c0", "c0 - c3", "c0 - c2", "c2 - c0"};
  for (const int32_t months : kMonthCounts) {
    calls.push_back(fmt::format("c0 + INTERVAL {} MONTH", months));
    calls.push_back(fmt::format("INTERVAL {} MONTH + c0", months));
    calls.push_back(fmt::format("c0 - INTERVAL {} MONTH", months));
  }
  return calls;
}

// The calls of each table, one table after another.
std::vector<std::string> joined(
    std::initializer_list<std::vector<std::string>> tables) {
  std::vector<std::string> calls;
  for (const auto& table : tables) {
    calls.insert(calls.end(), table.begin(), table.end());
  }
  return calls;
}

// The arithmetic over a TIMESTAMP or TIMESTAMP WITH TIME ZONE column c0, with
// the columns it reads beside it: a count c1, a second column c2 of c0's type
// and an INTERVAL DAY TO SECOND column c3.
const std::vector<std::string> kArithmeticCalls = joined(
    {arithmeticCalls(joined({{"millisecond"}, kTimestampUnits})),
     intervalCalls()});

// Every call over a TIMESTAMP, a DATE and a TIMESTAMP WITH TIME ZONE column.
// The arithmetic over TIMESTAMP WITH TIME ZONE runs apart, over
// withoutCpuTruncatedFractions(); DATE plus or minus an interval stays with
// the function tier.
const std::vector<std::string> kTimestampCalls = joined(
    {kFieldCalls,
     {"to_unixtime(c0)"},
     dateTruncCalls(kTimestampUnits),
     kArithmeticCalls});
const std::vector<std::string> kDateCalls = joined(
    {kFieldCalls, dateTruncCalls(kDateUnits), arithmeticCalls(kDateUnits)});
const std::vector<std::string> kTimestampWithTimeZoneCalls = joined(
    {kFieldCalls,
     kTimestampWithTimeZoneInstantCalls,
     dateTruncCalls(kTimestampUnits)});

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

// `values` with each entry replaced by the one a third of the list further
// on, so that a pair of columns reads two instants, two zones in a mixed
// column, and a null against a value.
template <typename T>
std::vector<std::optional<T>> rotated(
    const std::vector<std::optional<T>>& values) {
  std::vector<std::optional<T>> result;
  for (size_t i = 0; i < values.size(); ++i) {
    result.push_back(values[(i + values.size() / 3) % values.size()]);
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

  // Compares every call over `input`, on the rows the CPU answers.
  void assertCallsMatchCpu(
      const std::vector<std::string>& calls,
      const RowVectorPtr& input) {
    for (const auto& sql : calls) {
      SCOPED_TRACE(sql);
      assertGpuMatchesCpu(sql, input);
    }
  }

  // Runs `body` over every zone of each span as the session zone, with
  // TIMESTAMP columns in the span's unit: a nanosecond column, cuDF's default,
  // spans 1677 to 2262, entered twenty years in at both ends, which the
  // date_add counts reach; a microsecond column reaches past 2800, where the
  // device table ends and the lookup folds an instant back by whole 400-year
  // cycles.
  template <typename Body>
  void forEachTimestampSpan(Body&& body) {
    struct Span {
      cudf::type_id unit;
      int32_t fromYear;
      int32_t toYear;
      std::vector<std::string> zones;
    };
    const std::vector<Span> spans{
        {cudf::type_id::TIMESTAMP_NANOSECONDS, 1700, 2240, timeZones()},
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
        setSession({timeZone, true, true});
        body(timeZone, span.fromYear, span.toYear);
      }
    }
  }

  // The columns the calls read. c0 holds instants() of every zone, each with
  // the zone's id, then for each calendar unit the instants a count of one
  // moves onto a skipped local time, built back from the start and the middle
  // of the gaps of up to twelve offset increases, then a null. c1 holds the
  // count: kCounts in turn, and the one each gap row was built for. c2 is c0
  // shifted by kPairDeltas with the fraction of a second moved on alternate
  // rows, under the zone of the instant a third of the list further on, that
  // instant itself, or a null. c3 cycles through kIntervalMillis.
  RowVectorPtr makeInput(
      const TypePtr& type,
      const std::vector<std::string>& zones,
      int32_t fromYear,
      int32_t toYear,
      int32_t numSpread) {
    std::vector<std::optional<ZonedInstant>> values;
    std::vector<std::optional<int64_t>> counts;
    for (const auto& timeZone : zones) {
      const auto zoneId = tz::getTimeZoneID(timeZone);
      for (const auto& instant :
           instants(timeZone, fromYear, toYear, numSpread)) {
        values.push_back(ZonedInstant{zoneId, instant});
        counts.push_back(kCounts[values.size() % kCounts.size()]);
      }
      std::vector<test_utils::OffsetChange> increases;
      for (const auto& change :
           test_utils::offsetChanges(timeZone, fromYear, toYear)) {
        if (change.offsetAfter > change.offsetBefore) {
          increases.push_back(change);
        }
      }
      const auto* zone = tz::locateZone(timeZone);
      const size_t step = std::max<size_t>(1, increases.size() / 12);
      for (size_t i = 0; i < increases.size(); i += step) {
        const auto& change = increases[i];
        const int64_t gapStart = change.utcSeconds + change.offsetBefore;
        const int64_t gapMiddle =
            gapStart + (change.offsetAfter - change.offsetBefore) / 2;
        for (const auto& unitName : kDateUnits) {
          const auto unit =
              functions::fromDateTimeUnitString(StringView(unitName), true)
                  .value();
          for (const int64_t skippedLocal : {gapStart, gapMiddle}) {
            for (const int32_t count : {1, -1}) {
              const int64_t localStart =
                  functions::addToEpochTime({skippedLocal, 0}, unit, -count)
                      .seconds;
              values.push_back(
                  ZonedInstant{
                      zoneId,
                      Timestamp(
                          zone->to_sys(
                                  std::chrono::seconds{localStart},
                                  tz::TimeZone::TChoose::kEarliest)
                              .count(),
                          123'000'000)});
              counts.push_back(count);
            }
          }
        }
      }
      values.push_back(std::nullopt);
      counts.push_back(1);
    }
    const auto others = rotated(values);
    std::vector<std::optional<ZonedInstant>> pairs;
    for (size_t i = 0; i < values.size(); ++i) {
      const auto slot = i % (kPairDeltas.size() + 2);
      if (!values[i].has_value() || slot == kPairDeltas.size()) {
        pairs.push_back(others[i]);
      } else if (slot == kPairDeltas.size() + 1) {
        pairs.push_back(std::nullopt);
      } else {
        pairs.push_back(
            ZonedInstant{
                others[i].has_value() ? others[i]->zoneId : values[i]->zoneId,
                Timestamp(
                    values[i]->instant.getSeconds() + kPairDeltas[slot],
                    (values[i]->instant.getNanos() + (i % 2) * 500'000'000) %
                        1'000'000'000)});
      }
    }
    return makeRowVector({
        makeColumn(type, values),
        makeNullableFlatVector<int64_t>(counts),
        makeColumn(type, pairs),
        makeNullableFlatVector<int64_t>(
            cycled(kIntervalMillis, values.size()), INTERVAL_DAY_TIME()),
    });
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

  // `input` with the fraction of a second dropped from every TIMESTAMP WITH
  // TIME ZONE value before 1990, so that no instant the arithmetic reads or
  // produces, reaching back twenty years at most, falls before 1970 with a
  // fraction. The CPU converts such a value for the calendar arithmetic
  // through tz::TimeZone::to_local(milliseconds) and to_sys(milliseconds), and
  // the vendored tzdb reduces a sub-second time point to seconds with
  // time_point_cast, which truncates a negative count toward zero: in the
  // last second before an offset change, in UTC or in local time, such an
  // instant is read with the offset after the change, where the GPU, rounding
  // toward negative infinity as the seconds paths do on both sides, reads the
  // one before it. The fields keep the fraction.
  RowVectorPtr withoutCpuTruncatedFractions(const RowVectorPtr& input) {
    // 1990-01-01T00:00:00Z.
    constexpr int64_t kFractionFrom = 631'152'000'000;
    std::vector<VectorPtr> children;
    for (const auto& child : input->children()) {
      if (!isTimestampWithTimeZoneType(child->type())) {
        children.push_back(child);
        continue;
      }
      const auto* values = child->as<FlatVector<int64_t>>();
      std::vector<std::optional<int64_t>> column;
      for (vector_size_t row = 0; row < child->size(); ++row) {
        if (child->isNullAt(row)) {
          column.push_back(std::nullopt);
          continue;
        }
        int64_t millis = unpackMillisUtc(values->valueAt(row));
        if (millis < kFractionFrom) {
          millis -= ((millis % 1'000) + 1'000) % 1'000;
        }
        column.push_back(pack(millis, unpackZoneKeyId(values->valueAt(row))));
      }
      children.push_back(
          makeNullableFlatVector<int64_t>(column, TIMESTAMP_WITH_TIME_ZONE()));
    }
    return makeRowVector(children);
  }
};

// Each zone is the session zone in turn, so every TIMESTAMP call reads in it:
// a calendar unit of the arithmetic moves the local calendar, past a skipped
// local time for date_add, and a time unit the instant. A result the column
// cannot hold is declined rather than compared.
TEST_F(GpuSfiTimestampTest, timestampCallsMatchCpu) {
  forEachTimestampSpan(
      [&](const std::string& timeZone, int32_t fromYear, int32_t toYear) {
        assertCallsMatchCpu(
            kTimestampCalls,
            makeInput(TIMESTAMP(), {timeZone}, fromYear, toYear, 5'000));
      });
}

// Without adjust_timestamp_to_session_timezone, Velox reads TIMESTAMP as UTC
// whatever the session time zone, and so must GPU SFI.
TEST_F(GpuSfiTimestampTest, timestampCallsReadUtcWithoutAdjustment) {
  setSession({"America/Los_Angeles", false, true});
  assertCallsMatchCpu(
      kTimestampCalls,
      makeInput(TIMESTAMP(), {"America/Los_Angeles"}, 1990, 2030, 5'000));
}

// DATE has no time of day and no zone: the fields, the truncations and the
// arithmetic read the calendar alone, whatever the session. The days are the
// epoch and its neighbours, both sides of a leap day, a century non-leap
// year, dates far outside any real query's range, the month ends and leap
// days of eight years, where the end-of-month rules act, and a spread; beside
// them a count and the day a third of the list further on.
TEST_F(GpuSfiTimestampTest, dateCallsMatchCpu) {
  setSession({"America/Los_Angeles", true, true});
  std::vector<std::optional<int32_t>> days{
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
  for (const int32_t year : {1582, 1900, 1969, 1970, 1999, 2000, 2024, 2100}) {
    for (int32_t month = 1; month <= 12; ++month) {
      std::tm firstOfMonth{};
      firstOfMonth.tm_mday = 1;
      firstOfMonth.tm_mon = month - 1;
      firstOfMonth.tm_year = year - 1900;
      const auto first = static_cast<int32_t>(
          Timestamp::calendarUtcToEpoch(firstOfMonth) /
          Timestamp::kSecondsInDay);
      days.push_back(first - 1);
      days.push_back(first);
      days.push_back(first + functions::daysInMonth(year, month) - 1);
    }
  }
  for (const auto& instant : instants("UTC", 1680, 2262, 5'000)) {
    days.push_back(
        static_cast<int32_t>(
            instant.getSeconds() / Timestamp::kSecondsInDay -
            (instant.getSeconds() % Timestamp::kSecondsInDay < 0 ? 1 : 0)));
  }
  days.push_back(std::nullopt);
  assertCallsMatchCpu(
      kDateCalls,
      makeRowVector({
          makeNullableFlatVector<int32_t>(days, DATE()),
          makeNullableFlatVector<int64_t>(cycled<int64_t>(
              {0, 1, -1, 13, -13, 30, -30, 100, -100, 400, -400, std::nullopt},
              days.size())),
          makeNullableFlatVector<int32_t>(rotated(days), DATE()),
      }));
}

// Mixed-zone columns over each of kTimestampWithTimeZoneSpans, under every
// session: the render zone is the embedded zone or the session zone as
// legacy_timestamp_with_timezone selects, whether or not
// adjust_timestamp_to_session_timezone is set, and UTC for an empty session
// zone. The calendar arithmetic moves in the render zone, keeping the zone
// key, and date_diff reads both operands in the first operand's render zone.
TEST_F(GpuSfiTimestampTest, timestampWithTimeZoneCallsMatchCpu) {
  for (const auto& [fromYear, toYear] : kTimestampWithTimeZoneSpans) {
    const auto input = makeInput(
        TIMESTAMP_WITH_TIME_ZONE(), timeZones(), fromYear, toYear, 500);
    const auto arithmeticInput = withoutCpuTruncatedFractions(input);
    forEachSession([&] {
      assertCallsMatchCpu(kTimestampWithTimeZoneCalls, input);
      assertCallsMatchCpu(kArithmeticCalls, arithmeticInput);
    });
  }
}

// from_unixtime builds TIMESTAMP WITH TIME ZONE values from doubles, with the
// zone named, given as a constant offset or as offset columns read per row on
// the device, and TIMESTAMP values with no zone, under every session. The
// doubles cover the sub-millisecond rounding, both signs and NaN, within the
// years a nanosecond TIMESTAMP column holds.
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
  forEachSession([&] { assertCallsMatchCpu(calls, input); });
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
  const auto add = [&](std::optional<int64_t> leftValue,
                       std::optional<int64_t> rightValue) {
    left.push_back(leftValue);
    right.push_back(rightValue);
  };
  const std::vector<int64_t> millisUtc{
      0,
      1,
      -1,
      1'700'000'000'000,
      -62'135'596'800'000,
      kMaxMillisUtc,
      kMinMillisUtc};
  for (const auto& leftZone : timeZones()) {
    for (const auto& rightZone : timeZones()) {
      const auto leftId = tz::getTimeZoneID(leftZone);
      const auto rightId = tz::getTimeZoneID(rightZone);
      for (const int64_t millis : millisUtc) {
        add(pack(millis, leftId), pack(millis, rightId));
        if (millis < kMaxMillisUtc) {
          add(pack(millis, leftId), pack(millis + 1, rightId));
          add(pack(millis + 1, leftId), pack(millis, rightId));
        }
      }
      add(std::nullopt, pack(0, rightId));
      add(pack(0, leftId), std::nullopt);
      add(std::nullopt, std::nullopt);
    }
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
  assertCallsMatchCpu(calls, input);
}

} // namespace
} // namespace facebook::velox::cudf_velox
