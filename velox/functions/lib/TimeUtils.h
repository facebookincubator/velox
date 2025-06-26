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
#pragma once

#include <boost/algorithm/string/case_conv.hpp>
#include <velox/type/Timestamp.h>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/external/date/date.h"
#include "velox/external/date/iso_week.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions {

inline constexpr int64_t kSecondsInDay = 86'400;
inline constexpr int64_t kDaysInWeek = 7;
extern const folly::F14FastMap<std::string, int8_t> kDayOfWeekNames;

FOLLY_ALWAYS_INLINE const tz::TimeZone* getTimeZoneFromConfig(
    const core::QueryConfig& config) {
  if (config.adjustTimestampToTimezone()) {
    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      return tz::locateZone(sessionTzName);
    }
  }
  return nullptr;
}

FOLLY_ALWAYS_INLINE int64_t
getSeconds(Timestamp timestamp, const tz::TimeZone* timeZone) {
  if (timeZone != nullptr) {
    timestamp.toTimezone(*timeZone);
    return timestamp.getSeconds();
  } else {
    return timestamp.getSeconds();
  }
}

FOLLY_ALWAYS_INLINE
std::tm getDateTime(Timestamp timestamp, const tz::TimeZone* timeZone) {
  int64_t seconds = getSeconds(timestamp, timeZone);
  std::tm dateTime;
  VELOX_USER_CHECK(
      Timestamp::epochToCalendarUtc(seconds, dateTime),
      "Timestamp is too large: {} seconds since epoch",
      seconds);
  return dateTime;
}

// days is the number of days since Epoch.
FOLLY_ALWAYS_INLINE
std::tm getDateTime(int32_t days) {
  int64_t seconds = days * kSecondsInDay;
  std::tm dateTime;
  VELOX_USER_CHECK(
      Timestamp::epochToCalendarUtc(seconds, dateTime),
      "Date is too large: {} days",
      days);
  return dateTime;
}

FOLLY_ALWAYS_INLINE int getYear(const std::tm& time) {
  // tm_year: years since 1900.
  return 1900 + time.tm_year;
}

FOLLY_ALWAYS_INLINE int getMonth(const std::tm& time) {
  // tm_mon: months since January – [0, 11].
  return 1 + time.tm_mon;
}

FOLLY_ALWAYS_INLINE int getDay(const std::tm& time) {
  return time.tm_mday;
}

FOLLY_ALWAYS_INLINE int32_t getQuarter(const std::tm& time) {
  return time.tm_mon / 3 + 1;
}

FOLLY_ALWAYS_INLINE int32_t getDayOfYear(const std::tm& time) {
  return time.tm_yday + 1;
}

FOLLY_ALWAYS_INLINE uint32_t getWeek(
    const Timestamp& timestamp,
    const tz::TimeZone* timezone,
    bool allowOverflow) {
  // The computation of ISO week from date follows the algorithm here:
  // https://en.wikipedia.org/wiki/ISO_week_date
  Timestamp t = timestamp;
  if (timezone) {
    t.toTimezone(*timezone);
  }
  const auto timePoint = t.toTimePointMs(allowOverflow);
  const auto daysTimePoint = date::floor<date::days>(timePoint);
  const date::year_month_day calDate(daysTimePoint);
  auto weekNum = date::iso_week::year_weeknum_weekday{calDate}.weeknum();
  return (uint32_t)weekNum;
}

template <typename T>
struct InitSessionTimezone {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  const tz::TimeZone* timeZone_{nullptr};

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Timestamp>* /*timestamp*/) {
    timeZone_ = getTimeZoneFromConfig(config);
  }
};

/// Converts string as date time unit. Throws for invalid input string.
///
/// @param unitString The input string to represent date time unit.
/// @param throwIfInvalid Whether to throw an exception for invalid input
/// string.
/// @param allowMicro Whether to allow microsecond.
/// @param allowAbbreviated Whether to allow abbreviated unit string.
std::optional<DateTimeUnit> fromDateTimeUnitString(
    StringView unitString,
    bool throwIfInvalid,
    bool allowMicro = false,
    bool allowAbbreviated = false);

/// Adjusts the given date time object to the start of the specified date time
/// unit (e.g., year, quarter, month, week, day, hour, minute).
void adjustDateTime(std::tm& dateTime, const DateTimeUnit& unit);

/// Returns timestamp with seconds adjusted to the nearest lower multiple of the
/// specified interval. If the given seconds is negative and not an exact
/// multiple of the interval, it adjusts further down.
FOLLY_ALWAYS_INLINE Timestamp
adjustEpoch(int64_t seconds, int64_t intervalSeconds) {
  int64_t s = seconds / intervalSeconds;
  if (seconds < 0 && seconds % intervalSeconds) {
    s = s - 1;
  }
  int64_t truncatedSeconds = s * intervalSeconds;
  return Timestamp(truncatedSeconds, 0);
}

// Returns timestamp truncated to the specified unit.
Timestamp truncateTimestamp(
    Timestamp timestamp,
    DateTimeUnit unit,
    const tz::TimeZone* timeZone);

// Returns toTimestamp - fromTimestamp expressed in terms of unit.
FOLLY_ALWAYS_INLINE int64_t diffTimestamp(
    const DateTimeUnit unit,
    const Timestamp& fromTimestamp,
    const Timestamp& toTimestamp) {
  // TODO(gaoge): Handle overflow and underflow with 64-bit representation
  if (fromTimestamp == toTimestamp) {
    return 0;
  }

  const int8_t sign = fromTimestamp < toTimestamp ? 1 : -1;

  // fromTimepoint is less than or equal to toTimepoint
  const std::chrono::
      time_point<std::chrono::system_clock, std::chrono::milliseconds>
          fromTimepoint(std::chrono::milliseconds(
              std::min(fromTimestamp, toTimestamp).toMillis()));
  const std::chrono::
      time_point<std::chrono::system_clock, std::chrono::milliseconds>
          toTimepoint(std::chrono::milliseconds(
              std::max(fromTimestamp, toTimestamp).toMillis()));

  // Millisecond, second, minute, hour and day have fixed conversion ratio
  switch (unit) {
    case DateTimeUnit::kMillisecond: {
      return sign *
          std::chrono::duration_cast<std::chrono::milliseconds>(
              toTimepoint - fromTimepoint)
              .count();
    }
    case DateTimeUnit::kSecond: {
      return sign *
          std::chrono::duration_cast<std::chrono::seconds>(
              toTimepoint - fromTimepoint)
              .count();
    }
    case DateTimeUnit::kMinute: {
      return sign *
          std::chrono::duration_cast<std::chrono::minutes>(
              toTimepoint - fromTimepoint)
              .count();
    }
    case DateTimeUnit::kHour: {
      return sign *
          std::chrono::duration_cast<std::chrono::hours>(
              toTimepoint - fromTimepoint)
              .count();
    }
    case DateTimeUnit::kDay: {
      return sign *
          std::chrono::duration_cast<date::days>(toTimepoint - fromTimepoint)
              .count();
    }
    case DateTimeUnit::kWeek: {
      return sign *
          std::chrono::duration_cast<date::days>(toTimepoint - fromTimepoint)
              .count() /
          7;
    }
    default:
      break;
  }

  // Spark support microsecond unit.
  if (unit == DateTimeUnit::kMicrosecond) {
    const std::chrono::time_point<std::chrono::system_clock>
        fromMicrosecondpoint(std::chrono::microseconds(
            std::min(fromTimestamp, toTimestamp).toMicros()));
    const std::chrono::time_point<std::chrono::system_clock> toMicrosecondpoint(
        std::chrono::microseconds(
            std::max(fromTimestamp, toTimestamp).toMicros()));
    return sign *
        std::chrono::duration_cast<std::chrono::microseconds>(
            toMicrosecondpoint - fromMicrosecondpoint)
            .count();
  }

  // Month, quarter and year do not have fixed conversion ratio. Ex. a month can
  // have 28, 29, 30 or 31 days. A year can have 365 or 366 days.
  const std::chrono::time_point<std::chrono::system_clock, date::days>
      fromDaysTimepoint = std::chrono::floor<date::days>(fromTimepoint);
  const std::chrono::time_point<std::chrono::system_clock, date::days>
      toDaysTimepoint = std::chrono::floor<date::days>(toTimepoint);
  const date::year_month_day fromCalDate(fromDaysTimepoint);
  const date::year_month_day toCalDate(toDaysTimepoint);
  const uint64_t fromTimeInstantOfDay =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          fromTimepoint - fromDaysTimepoint)
          .count();

  uint64_t toTimeInstantOfDay = 0;
  uint64_t toTimePointMillis = toTimepoint.time_since_epoch().count();
  uint64_t toDaysTimepointMillis =
      std::chrono::
          time_point<std::chrono::system_clock, std::chrono::milliseconds>(
              toDaysTimepoint)
              .time_since_epoch()
              .count();
  bool overflow = __builtin_sub_overflow(
      toTimePointMillis, toDaysTimepointMillis, &toTimeInstantOfDay);
  VELOX_USER_CHECK_EQ(
      overflow,
      false,
      "{} - {} Causes arithmetic overflow: {} - {}",
      fromTimestamp.toString(),
      toTimestamp.toString(),
      toTimePointMillis,
      toDaysTimepointMillis);
  const uint8_t fromDay = static_cast<unsigned>(fromCalDate.day()),
                fromMonth = static_cast<unsigned>(fromCalDate.month());
  const uint8_t toDay = static_cast<unsigned>(toCalDate.day()),
                toMonth = static_cast<unsigned>(toCalDate.month());
  const date::year_month_day toCalLastYearMonthDay(
      toCalDate.year() / toCalDate.month() / date::last);
  const uint8_t toLastYearMonthDay =
      static_cast<unsigned>(toCalLastYearMonthDay.day());

  if (unit == DateTimeUnit::kMonth || unit == DateTimeUnit::kQuarter) {
    int64_t diff =
        (int64_t(toCalDate.year()) - int64_t(fromCalDate.year())) * 12 +
        int(toMonth) - int(fromMonth);

    if ((toDay != toLastYearMonthDay && fromDay > toDay) ||
        (fromDay == toDay && fromTimeInstantOfDay > toTimeInstantOfDay)) {
      diff--;
    }

    diff = (unit == DateTimeUnit::kMonth) ? diff : diff / 3;
    return sign * diff;
  }

  if (unit == DateTimeUnit::kYear) {
    int64_t diff = (toCalDate.year() - fromCalDate.year()).count();

    if (fromMonth > toMonth ||
        (fromMonth == toMonth && fromDay > toDay &&
         toDay != toLastYearMonthDay) ||
        (fromMonth == toMonth && fromDay == toDay &&
         fromTimeInstantOfDay > toTimeInstantOfDay)) {
      diff--;
    }
    return sign * diff;
  }

  VELOX_UNREACHABLE("Unsupported datetime unit");
}

inline bool isTimeUnit(const DateTimeUnit unit) {
  return unit == DateTimeUnit::kMillisecond || unit == DateTimeUnit::kSecond ||
      unit == DateTimeUnit::kMinute || unit == DateTimeUnit::kHour ||
      unit == DateTimeUnit::kMicrosecond;
}
} // namespace facebook::velox::functions
