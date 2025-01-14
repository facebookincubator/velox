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

#include <velox/type/Timestamp.h>
#include "velox/core/QueryConfig.h"
#include "velox/external/date/date.h"
#include "velox/external/date/iso_week.h"
#include "velox/functions/Macros.h"
#include "velox/type/TimestampConversion.h"
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

/// Return day-of-year (DOY) of the first `dayOfWeek` in the year.
///
/// `dayOfWeek` is a 1-based weekday number starting with Sunday.
///   (1 = Sunday, 2 = Monday, ..., 7 = Saturday).
///
/// If the `dayOfWeek` is Monday, it returns DOY of first Monday in
/// the year. The returned DOY is a number from 1 to 7.
FOLLY_ALWAYS_INLINE
uint32_t getDayOfFirstDayOfWeek(int32_t y, uint32_t dayOfWeek) {
  auto firstDay =
      date::year_month_day(date::year(y), date::month(1), date::day(1));
  auto weekday = date::weekday(firstDay).c_encoding() + 1;

  int32_t delta = dayOfWeek - weekday;
  if (delta < 0) {
    delta += 7;
  }

  return delta + 1;
}

/// Return the week year represented by Gregorian calendar for the given year,
/// month and day.
///
/// getWeekYear only works with Gregorian calendar due to limitations in the
/// date library. As a result, dates before the Gregorian calendar
/// (1582-10-15) yields mismatched results.
///
/// The week that includes January 1st and has 'minimalDaysInFirstWeek' or more
/// days is referred to as week 1. The starting day of the week is decided by
/// the `firstDayOfWeek`, which is a 1-based weekday number starting with
/// Sunday.
///
/// For ISO 8601, `firstDayOfWeek` is 2 (Monday) and `minimalDaysInFirstWeek`
/// is 4. For legacy Spark, `firstDayOfWeek` is 1 (Sunday) and
/// `minimalDaysInFirstWeek` is 1.
///
/// The algorithm refers to the getWeekYear algorithm in openjdk:
/// https://github.com/openjdk/jdk/blob/d9c67443f7d7f03efb2837b63ee2acc6113f737f/src/java.base/share/classes/java/util/GregorianCalendar.java#L2058
FOLLY_ALWAYS_INLINE
int32_t getWeekYear(
    int32_t y,
    uint32_t m,
    uint32_t d,
    uint32_t firstDayOfWeek,
    uint32_t minimalDaysInFirstWeek) {
  const auto ymd =
      date::year_month_day(date::year(y), date::month(m), date::day(d));
  const auto firstDayOfTheYear =
      date::year_month_day(ymd.year(), date::month(1), date::day(1));
  const auto dayOfYear =
      (date::sys_days{ymd} - date::sys_days{firstDayOfTheYear}).count() + 1;
  const auto maxDayOfYear = util::isLeapYear(y) ? 366 : 365;

  // If this week does not cross the years (`7 < dayOfYear && dayOfYear <
  // (maxDayOfYear - 6)`), the weekyear must be equal to the year.
  //
  // If some days of this week fall in the last year and `minimalDaysInFirstWeek
  // < dayOfYear`, the number of days in this week in this year must be greater
  // than minimalDaysInFirstWeek, so the weekyear must be equal to the year.
  //
  // Since minimalDaysInFirstWeek always no more than 7, these two conditions
  // can be reduced to the following code.
  if (dayOfYear > minimalDaysInFirstWeek && dayOfYear < (maxDayOfYear - 6)) {
    return y;
  }

  auto year = y;
  // Day of beginning of first complete week of this year.
  auto minDayOfYear = getDayOfFirstDayOfWeek(y, firstDayOfWeek);
  if (dayOfYear >= minDayOfYear) {
    // Day of ending of first week of the last year.
    auto minDayOfYear = getDayOfFirstDayOfWeek(y + 1, firstDayOfWeek) - 1;
    if (minDayOfYear == 0) {
      minDayOfYear = 7;
    }

    // If that week belongs to the next weekyear.
    if (minDayOfYear >= minimalDaysInFirstWeek) {
      // If dayOfYear is in that week.
      int days = maxDayOfYear - dayOfYear + 1;
      if (days <= (7 - minDayOfYear)) {
        ++year;
      }
    }
  } else if (minDayOfYear <= minimalDaysInFirstWeek) {
    // Days of the first week in this year less then minimalDaysInFirstWeek
    --year;
  }

  return year;
}
} // namespace facebook::velox::functions
