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

enum class DateTimeUnit {
  kMicrosecond,
  kMillisecond,
  kSecond,
  kMinute,
  kHour,
  kDay,
  kWeek,
  kMonth,
  kQuarter,
  kYear
};

FOLLY_ALWAYS_INLINE std::string dateTimeUnitString(DateTimeUnit unit) {
  switch (unit) {
    case DateTimeUnit::kMicrosecond:
      return "microsecond";
    case DateTimeUnit::kMillisecond:
      return "millisecond";
    case DateTimeUnit::kSecond:
      return "second";
    case DateTimeUnit::kMinute:
      return "minute";
    case DateTimeUnit::kHour:
      return "hour";
    case DateTimeUnit::kDay:
      return "day";
    case DateTimeUnit::kWeek:
      return "week";
    case DateTimeUnit::kMonth:
      return "month";
    case DateTimeUnit::kQuarter:
      return "quarter";
    case DateTimeUnit::kYear:
      return "year";
    default:
      return fmt::format("UNKNOWN: {}", static_cast<int>(unit));
  }
}

FOLLY_ALWAYS_INLINE std::optional<DateTimeUnit> fromDateTimeUnitString(
    const StringView& unitString,
    bool isSparkUnitString) {
  static const StringView kMicrosecond("microsecond");
  static const StringView kMillisecond("millisecond");
  static const StringView kSecond("second");
  static const StringView kMinute("minute");
  static const StringView kHour("hour");
  static const StringView kDay("day");
  static const StringView kWeek("week");
  static const StringView kMonth("month");
  static const StringView kQuarter("quarter");
  static const StringView kYear("year");

  const auto unit = boost::algorithm::to_lower_copy(unitString.str());

  if (unit == kMillisecond) {
    return DateTimeUnit::kMillisecond;
  }
  if (unit == kSecond) {
    return DateTimeUnit::kSecond;
  }
  if (unit == kMinute) {
    return DateTimeUnit::kMinute;
  }
  if (unit == kHour) {
    return DateTimeUnit::kHour;
  }
  if (unit == kDay) {
    return DateTimeUnit::kDay;
  }
  if (unit == kWeek) {
    return DateTimeUnit::kWeek;
  }
  if (unit == kMonth) {
    return DateTimeUnit::kMonth;
  }
  if (unit == kQuarter) {
    return DateTimeUnit::kQuarter;
  }
  if (unit == kYear) {
    return DateTimeUnit::kYear;
  }
  if (isSparkUnitString) {
    if (unit == kMicrosecond) {
      return DateTimeUnit::kMicrosecond;
    }
    if (unit == "dd") {
      return DateTimeUnit::kDay;
    }
    if (unit == "mon" || unit == "mm") {
      return DateTimeUnit::kMonth;
    }
    if (unit == "yyyy" || unit == "yy") {
      return DateTimeUnit::kYear;
    }
  }
  return std::nullopt;
}

FOLLY_ALWAYS_INLINE std::optional<DateTimeUnit> getDateTimeUnit(
    const StringView& unitString,
    bool throwIfInvalid,
    const DateTimeUnit& minUnit = DateTimeUnit::kMillisecond,
    bool isSparkUnitString = false) {
  std::optional<DateTimeUnit> unit =
      fromDateTimeUnitString(unitString, isSparkUnitString);
  if (unit.has_value() && unit.value() >= minUnit) {
    return unit;
  }
  if (throwIfInvalid) {
    if (unit.has_value()) {
      VELOX_USER_FAIL(
          "{} is not a valid datetime unit field, minimum unit is {}",
          unitString,
          dateTimeUnitString(minUnit));
    } else {
      VELOX_UNSUPPORTED("Unsupported datetime unit: {}", unitString);
    }
  }
  return std::nullopt;
}

FOLLY_ALWAYS_INLINE void adjustDateTime(
    std::tm& dateTime,
    const DateTimeUnit& unit) {
  switch (unit) {
    case DateTimeUnit::kYear:
      dateTime.tm_mon = 0;
      dateTime.tm_yday = 0;
      FMT_FALLTHROUGH;
    case DateTimeUnit::kQuarter:
      dateTime.tm_mon = dateTime.tm_mon / 3 * 3;
      FMT_FALLTHROUGH;
    case DateTimeUnit::kMonth:
      dateTime.tm_mday = 1;
      dateTime.tm_hour = 0;
      dateTime.tm_min = 0;
      dateTime.tm_sec = 0;
      break;
    case DateTimeUnit::kWeek:
      // Subtract the truncation
      dateTime.tm_mday -= dateTime.tm_wday == 0 ? 6 : dateTime.tm_wday - 1;
      // Setting the day of the week to Monday
      dateTime.tm_wday = 1;

      // If the adjusted day of the month falls in the previous month
      // Move to the previous month
      if (dateTime.tm_mday < 1) {
        dateTime.tm_mon -= 1;

        // If the adjusted month falls in the previous year
        // Set to December and Move to the previous year
        if (dateTime.tm_mon < 0) {
          dateTime.tm_mon = 11;
          dateTime.tm_year -= 1;
        }

        // Calculate the correct day of the month based on the number of days
        // in the adjusted month
        static const int daysInMonth[] = {
            31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        int daysInPrevMonth = daysInMonth[dateTime.tm_mon];

        // Adjust for leap year if February
        if (dateTime.tm_mon == 1 && (dateTime.tm_year + 1900) % 4 == 0 &&
            ((dateTime.tm_year + 1900) % 100 != 0 ||
             (dateTime.tm_year + 1900) % 400 == 0)) {
          daysInPrevMonth = 29;
        }
        // Set to the correct day in the previous month
        dateTime.tm_mday += daysInPrevMonth;
      }
      dateTime.tm_hour = 0;
      dateTime.tm_min = 0;
      dateTime.tm_sec = 0;
      break;
    case DateTimeUnit::kDay:
      dateTime.tm_hour = 0;
      FMT_FALLTHROUGH;
    case DateTimeUnit::kHour:
      dateTime.tm_min = 0;
      FMT_FALLTHROUGH;
    case DateTimeUnit::kMinute:
      dateTime.tm_sec = 0;
      break;
    default:
      VELOX_UNREACHABLE();
  }
}

/// For fixed interval like second, minute, hour, day and week
/// we can truncate date by a simple arithmetic expression:
/// floor(seconds / intervalSeconds) * intervalSeconds.
FOLLY_ALWAYS_INLINE Timestamp
adjustEpoch(int64_t seconds, int64_t intervalSeconds) {
  int64_t s = seconds / intervalSeconds;
  if (seconds < 0 && seconds % intervalSeconds) {
    s = s - 1;
  }
  int64_t truncedSeconds = s * intervalSeconds;
  return Timestamp(truncedSeconds, 0);
}

FOLLY_ALWAYS_INLINE Timestamp truncateTimestamp(
    const Timestamp& timestamp,
    const DateTimeUnit& unit,
    const tz::TimeZone* timeZone) {
  Timestamp result;
  switch (unit) {
    case DateTimeUnit::kMicrosecond:
      return Timestamp(
          timestamp.getSeconds(), timestamp.getNanos() / 1000 * 1000);

    case DateTimeUnit::kMillisecond:
      return Timestamp(
          timestamp.getSeconds(), timestamp.getNanos() / 1000000 * 1000000);

    // For seconds, we just truncate the nanoseconds part of the timestamp; no
    // timezone conversion required.
    case DateTimeUnit::kSecond:
      return Timestamp(timestamp.getSeconds(), 0);

    // Same for minutes; timezones and daylight savings time are at least in
    // the granularity of 30 mins, so we can just truncate the epoch directly.
    case DateTimeUnit::kMinute:
      return adjustEpoch(timestamp.getSeconds(), 60);

    // Hour truncation has to handle the corner case of daylight savings time
    // boundaries. Since conversions from local timezone to UTC may be
    // ambiguous, we need to be carefull about the roundtrip of converting to
    // local time and back. So what we do is to calculate the truncation delta
    // in UTC, then applying it to the input timestamp.
    case DateTimeUnit::kHour: {
      auto epochToAdjust = getSeconds(timestamp, timeZone);
      auto secondsDelta =
          epochToAdjust - adjustEpoch(epochToAdjust, 60 * 60).getSeconds();
      return Timestamp(timestamp.getSeconds() - secondsDelta, 0);
    }

    // For the truncations below, we may first need to convert to the local
    // timestamp, truncate, then convert back to GMT.
    case DateTimeUnit::kDay:
      result = adjustEpoch(getSeconds(timestamp, timeZone), 24 * 60 * 60);
      break;

    default:
      auto dateTime = getDateTime(timestamp, timeZone);
      adjustDateTime(dateTime, unit);
      result = Timestamp(Timestamp::calendarUtcToEpoch(dateTime), 0);
      break;
  }

  if (timeZone != nullptr) {
    result.toGMT(*timeZone);
  }
  return result;
}
} // namespace facebook::velox::functions
