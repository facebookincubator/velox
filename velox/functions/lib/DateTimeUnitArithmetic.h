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

#include <cstdint>
#include <ctime>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"
#include "velox/functions/lib/DateTimeUnit.h"
#include "velox/functions/lib/TimeUtilsCore.h"
#include "velox/type/TimestampCalendar.h"

/// Arithmetic in date/time units with no time zone involved: adding a count of
/// units to a day count or to a timestamp, the difference between two values
/// in a unit, and truncation to the start of a unit. Callers convert to local
/// time before these calls and back to UTC after them. Header-only and free of
/// host-only facilities so that CUDA translation units can use it.
namespace facebook::velox::functions {

/// Seconds since 1970-01-01 00:00:00 on the caller's clock, UTC or a local
/// wall clock, plus the nanoseconds within that second. Mirrors the fields of
/// Timestamp, which device code cannot use.
struct EpochTime {
  int64_t seconds;
  uint64_t nanos;
};

/// Returns the number of days in 'month' (1 to 12) of 'year'.
VELOX_GPU_COMPATIBLE inline int32_t daysInMonth(int64_t year, int32_t month) {
  // Function-local so that device code can index it with a runtime value.
  constexpr int32_t kDays[12] = {
      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month == 2 && calendar::isLeap(year)) {
    return 29;
  }
  return kDays[month - 1];
}

namespace detail {

inline constexpr int64_t kNanosecondsInMicrosecond = 1'000;
inline constexpr int64_t kNanosecondsInMillisecond = 1'000'000;
inline constexpr int64_t kNanosecondsInSecond = 1'000'000'000;
inline constexpr int64_t kMillisecondsInSecond = 1'000;
inline constexpr int64_t kMicrosecondsInSecond = 1'000'000;

// Division that rounds toward negative infinity. 'divisor' is positive.
VELOX_GPU_COMPATIBLE inline int64_t floorDivide(
    int64_t value,
    int64_t divisor) {
  const int64_t quotient = value / divisor;
  return value % divisor < 0 ? quotient - 1 : quotient;
}

// Returns 'time' shifted by 'nanos', carrying whole seconds.
VELOX_GPU_COMPATIBLE inline EpochTime addNanos(EpochTime time, int64_t nanos) {
  const int64_t total = static_cast<int64_t>(time.nanos) + nanos;
  const int64_t carry = floorDivide(total, kNanosecondsInSecond);
  return {
      time.seconds + carry,
      static_cast<uint64_t>(total - carry * kNanosecondsInSecond),
  };
}

// Months in a month, quarter or year.
VELOX_GPU_COMPATIBLE inline int64_t monthsPerUnit(DateTimeUnit unit) {
  return unit == DateTimeUnit::kYear   ? 12
      : unit == DateTimeUnit::kQuarter ? 3
                                       : 1;
}

// Moves the calendar date in 'dateTime' by 'months'. A day of the month that
// the target month does not have becomes that month's last day.
VELOX_GPU_COMPATIBLE inline void addMonths(std::tm& dateTime, int64_t months) {
  const int64_t totalMonths =
      (static_cast<int64_t>(dateTime.tm_year) + calendar::kTmYearBase) * 12 +
      dateTime.tm_mon + months;
  const int64_t year = floorDivide(totalMonths, 12);
  dateTime.tm_year = static_cast<int>(year - calendar::kTmYearBase);
  dateTime.tm_mon = static_cast<int>(totalMonths - year * 12);
  const int32_t lastDay = daysInMonth(year, dateTime.tm_mon + 1);
  if (dateTime.tm_mday > lastDay) {
    dateTime.tm_mday = lastDay;
  }
}

// Milliseconds since the start of the day of 'dateTime'.
VELOX_GPU_COMPATIBLE inline int64_t millisOfDay(
    const std::tm& dateTime,
    uint64_t nanos) {
  const int64_t seconds = dateTime.tm_hour * kSecondsInHour +
      dateTime.tm_min * kSecondsInMinute + dateTime.tm_sec;
  return seconds * kMillisecondsInSecond +
      static_cast<int64_t>(nanos / kNanosecondsInMillisecond);
}

} // namespace detail

/// Adds 'value' units to a count of days since the epoch. Supports the day
/// unit and longer ones. Month, quarter and year arithmetic keeps the day of
/// the month when the target month has it and otherwise moves to that month's
/// last day: 2022-01-30 plus one month is 2022-02-28, and 2020-02-29 plus one
/// year is 2021-02-28.
VELOX_GPU_COMPATIBLE inline int32_t
addToDays(int32_t days, DateTimeUnit unit, int32_t value) {
  if (value == 0) {
    return days;
  }
  int64_t result{0};
  switch (unit) {
    case DateTimeUnit::kDay:
      result = static_cast<int64_t>(days) + value;
      break;
    case DateTimeUnit::kWeek:
      result = days + kDaysInWeek * static_cast<int64_t>(value);
      break;
    case DateTimeUnit::kMonth:
    case DateTimeUnit::kQuarter:
    case DateTimeUnit::kYear: {
      std::tm dateTime = getDateTime(days);
      detail::addMonths(
          dateTime, detail::monthsPerUnit(unit) * static_cast<int64_t>(value));
      result = calendar::calendarUtcToEpoch(dateTime) / kSecondsInDay;
      break;
    }
    default:
      VELOX_UNREACHABLE();
  }
  return static_cast<int32_t>(result);
}

/// Adds 'value' units to 'time'. Units shorter than a day shift by a fixed
/// duration and keep the nanoseconds below that unit. Longer units keep the
/// time of day and move the calendar date the way addToDays does.
VELOX_GPU_COMPATIBLE inline EpochTime
addToEpochTime(EpochTime time, DateTimeUnit unit, int32_t value) {
  if (value == 0) {
    return time;
  }
  switch (unit) {
    case DateTimeUnit::kMicrosecond:
      return detail::addNanos(time, value * detail::kNanosecondsInMicrosecond);
    case DateTimeUnit::kMillisecond:
      return detail::addNanos(time, value * detail::kNanosecondsInMillisecond);
    case DateTimeUnit::kSecond:
      return {time.seconds + value, time.nanos};
    case DateTimeUnit::kMinute:
      return {
          time.seconds + value * kSecondsInMinute,
          time.nanos,
      };
    case DateTimeUnit::kHour:
      return {
          time.seconds + value * kSecondsInHour,
          time.nanos,
      };
    case DateTimeUnit::kDay:
      return {
          time.seconds + value * kSecondsInDay,
          time.nanos,
      };
    case DateTimeUnit::kWeek:
      return {
          time.seconds + value * kDaysInWeek * kSecondsInDay,
          time.nanos,
      };
    case DateTimeUnit::kMonth:
    case DateTimeUnit::kQuarter:
    case DateTimeUnit::kYear: {
      std::tm dateTime = getDateTimeUtc(time.seconds);
      detail::addMonths(
          dateTime, detail::monthsPerUnit(unit) * static_cast<int64_t>(value));
      return {calendar::calendarUtcToEpoch(dateTime), time.nanos};
    }
    default:
      VELOX_UNREACHABLE("Unsupported datetime unit");
  }
}

/// Returns 'to' minus 'from' in whole units, with the sign of the difference.
/// Fixed-length units compare the two values at millisecond resolution, or
/// microsecond resolution for the microsecond unit, and round toward zero.
/// Month, quarter and year count the calendar months between the two dates
/// and drop the last one when 'to' falls earlier within its month than 'from'
/// does within its month, compared by day and then by time of day. With
/// 'respectLastDay', a 'to' on the last day of its month completes the month
/// whatever the day of 'from': 2020-01-31 to 2020-02-29 is one month with it
/// and zero months without it.
VELOX_GPU_COMPATIBLE inline int64_t diffEpochTime(
    DateTimeUnit unit,
    EpochTime from,
    EpochTime to,
    bool respectLastDay) {
  if (from.seconds == to.seconds && from.nanos == to.nanos) {
    return 0;
  }
  const bool forward = from.seconds < to.seconds ||
      (from.seconds == to.seconds && from.nanos < to.nanos);
  const EpochTime low = forward ? from : to;
  const EpochTime high = forward ? to : from;
  const int64_t sign = forward ? 1 : -1;
  const int64_t seconds = high.seconds - low.seconds;

  if (unit == DateTimeUnit::kMicrosecond) {
    const int64_t micros =
        static_cast<int64_t>(high.nanos / detail::kNanosecondsInMicrosecond) -
        static_cast<int64_t>(low.nanos / detail::kNanosecondsInMicrosecond);
    return sign * (seconds * detail::kMicrosecondsInSecond + micros);
  }

  const int64_t millis =
      static_cast<int64_t>(high.nanos / detail::kNanosecondsInMillisecond) -
      static_cast<int64_t>(low.nanos / detail::kNanosecondsInMillisecond);
  // The last second counts only when 'high' reaches the millisecond of 'low'.
  const int64_t wholeSeconds = millis < 0 ? seconds - 1 : seconds;
  switch (unit) {
    case DateTimeUnit::kMillisecond:
      return sign * (seconds * detail::kMillisecondsInSecond + millis);
    case DateTimeUnit::kSecond:
      return sign * wholeSeconds;
    case DateTimeUnit::kMinute:
      return sign * (wholeSeconds / kSecondsInMinute);
    case DateTimeUnit::kHour:
      return sign * (wholeSeconds / kSecondsInHour);
    case DateTimeUnit::kDay:
      return sign * (wholeSeconds / kSecondsInDay);
    case DateTimeUnit::kWeek:
      return sign * (wholeSeconds / (kDaysInWeek * kSecondsInDay));
    default:
      break;
  }

  const std::tm lowDate = getDateTimeUtc(low.seconds);
  const std::tm highDate = getDateTimeUtc(high.seconds);
  const int64_t lowYear = getYear(lowDate);
  const int64_t highYear = getYear(highDate);
  const bool highOnLastDay =
      highDate.tm_mday == daysInMonth(highYear, highDate.tm_mon + 1);
  const bool partialMonth = ((!respectLastDay || !highOnLastDay) &&
                             lowDate.tm_mday > highDate.tm_mday) ||
      (lowDate.tm_mday == highDate.tm_mday &&
       detail::millisOfDay(lowDate, low.nanos) >
           detail::millisOfDay(highDate, high.nanos));

  if (unit == DateTimeUnit::kYear) {
    int64_t years = highYear - lowYear;
    if (lowDate.tm_mon > highDate.tm_mon ||
        (lowDate.tm_mon == highDate.tm_mon && partialMonth)) {
      --years;
    }
    return sign * years;
  }
  if (unit != DateTimeUnit::kMonth && unit != DateTimeUnit::kQuarter) {
    VELOX_UNREACHABLE();
  }
  int64_t months = (highYear - lowYear) * 12 + highDate.tm_mon - lowDate.tm_mon;
  if (partialMonth) {
    --months;
  }
  return sign * (unit == DateTimeUnit::kQuarter ? months / 3 : months);
}

/// Returns 'toDays' minus 'fromDays' in whole units, as diffEpochTime does
/// for two midnights with the last day of a month respected.
VELOX_GPU_COMPATIBLE inline int64_t
diffDays(DateTimeUnit unit, int32_t fromDays, int32_t toDays) {
  return diffEpochTime(
      unit,
      {static_cast<int64_t>(fromDays) * kSecondsInDay, 0},
      {static_cast<int64_t>(toDays) * kSecondsInDay, 0},
      /*respectLastDay=*/true);
}

/// Moves 'dateTime' back to the start of 'unit', from a minute up to a year.
/// Weeks start on Monday and quarters on January, April, July and October.
/// Maintains the fields that calendarUtcToEpoch reads; tm_wday and tm_yday
/// may be left stale.
VELOX_GPU_COMPATIBLE inline void adjustDateTime(
    std::tm& dateTime,
    DateTimeUnit unit) {
  switch (unit) {
    case DateTimeUnit::kYear:
      dateTime.tm_mon = 0;
      dateTime.tm_yday = 0;
      [[fallthrough]];
    case DateTimeUnit::kQuarter:
      dateTime.tm_mon = dateTime.tm_mon / 3 * 3;
      [[fallthrough]];
    case DateTimeUnit::kMonth:
      dateTime.tm_mday = 1;
      dateTime.tm_hour = 0;
      dateTime.tm_min = 0;
      dateTime.tm_sec = 0;
      break;
    case DateTimeUnit::kWeek:
      // Sunday is tm_wday 0, so it moves back six days to Monday.
      dateTime.tm_mday -= dateTime.tm_wday == 0 ? 6 : dateTime.tm_wday - 1;
      dateTime.tm_wday = 1;
      // A day of the month below 1 belongs to the previous month, which may
      // be December of the previous year.
      if (dateTime.tm_mday < 1) {
        dateTime.tm_mon -= 1;
        if (dateTime.tm_mon < 0) {
          dateTime.tm_mon = 11;
          dateTime.tm_year -= 1;
        }
        dateTime.tm_mday += daysInMonth(getYear(dateTime), dateTime.tm_mon + 1);
      }
      dateTime.tm_hour = 0;
      dateTime.tm_min = 0;
      dateTime.tm_sec = 0;
      break;
    case DateTimeUnit::kDay:
      dateTime.tm_hour = 0;
      [[fallthrough]];
    case DateTimeUnit::kHour:
      dateTime.tm_min = 0;
      [[fallthrough]];
    case DateTimeUnit::kMinute:
      dateTime.tm_sec = 0;
      break;
    default:
      VELOX_UNREACHABLE();
  }
}

/// Returns 'time' moved back to the start of 'unit'. Weeks start on Monday
/// and quarters on January, April, July and October.
VELOX_GPU_COMPATIBLE inline EpochTime truncateEpochTime(
    EpochTime time,
    DateTimeUnit unit) {
  switch (unit) {
    case DateTimeUnit::kMicrosecond:
      return {
          time.seconds,
          time.nanos / detail::kNanosecondsInMicrosecond *
              detail::kNanosecondsInMicrosecond,
      };
    case DateTimeUnit::kMillisecond:
      return {
          time.seconds,
          time.nanos / detail::kNanosecondsInMillisecond *
              detail::kNanosecondsInMillisecond,
      };
    case DateTimeUnit::kSecond:
      return {time.seconds, 0};
    case DateTimeUnit::kMinute:
      return {
          detail::floorDivide(time.seconds, kSecondsInMinute) *
              kSecondsInMinute,
          0,
      };
    case DateTimeUnit::kHour:
      return {
          detail::floorDivide(time.seconds, kSecondsInHour) * kSecondsInHour,
          0,
      };
    case DateTimeUnit::kDay:
      return {
          detail::floorDivide(time.seconds, kSecondsInDay) * kSecondsInDay,
          0,
      };
    case DateTimeUnit::kWeek:
    case DateTimeUnit::kMonth:
    case DateTimeUnit::kQuarter:
    case DateTimeUnit::kYear: {
      std::tm dateTime = getDateTimeUtc(time.seconds);
      adjustDateTime(dateTime, unit);
      return {calendar::calendarUtcToEpoch(dateTime), 0};
    }
    default:
      VELOX_UNREACHABLE();
  }
}

} // namespace facebook::velox::functions
