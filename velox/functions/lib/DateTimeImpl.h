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
#include "velox/external/date/date.h"
#include "velox/functions/lib/DateTimeFormatter.h"
#include "velox/functions/lib/DateTimeImpl.h"
#include "velox/functions/lib/TimeUtils.h"

namespace facebook::velox::functions {

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

  // Spark support microsecond unit, if convert by .toMicros() for all the
  // unit, may cause overflow, so treat it dependently.
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

  // fromTimepoint is less than or equal to toTimepoint.
  const std::chrono::
      time_point<std::chrono::system_clock, std::chrono::milliseconds>
          fromTimepoint(std::chrono::milliseconds(
              std::min(fromTimestamp, toTimestamp).toMillis()));
  const std::chrono::
      time_point<std::chrono::system_clock, std::chrono::milliseconds>
          toTimepoint(std::chrono::milliseconds(
              std::max(fromTimestamp, toTimestamp).toMillis()));

  // Millisecond, second, minute, hour and day have fixed conversion ratio.
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

FOLLY_ALWAYS_INLINE int64_t diffTimestamp(
    const DateTimeUnit unit,
    const Timestamp& fromTimestamp,
    const Timestamp& toTimestamp,
    const tz::TimeZone* timeZone) {
  if (LIKELY(timeZone != nullptr)) {
    // sessionTimeZone not null means that the config
    // adjust_timestamp_to_timezone is on.
    Timestamp fromZonedTimestamp = fromTimestamp;
    fromZonedTimestamp.toTimezone(*timeZone);

    Timestamp toZonedTimestamp = toTimestamp;
    if (isTimeUnit(unit)) {
      const int64_t offset =
          static_cast<Timestamp>(fromTimestamp).getSeconds() -
          fromZonedTimestamp.getSeconds();
      toZonedTimestamp = Timestamp(
          toZonedTimestamp.getSeconds() - offset, toZonedTimestamp.getNanos());
    } else {
      toZonedTimestamp.toTimezone(*timeZone);
    }
    return diffTimestamp(unit, fromZonedTimestamp, toZonedTimestamp);
  }
  return diffTimestamp(unit, fromTimestamp, toTimestamp);
}
} // namespace facebook::velox::functions
