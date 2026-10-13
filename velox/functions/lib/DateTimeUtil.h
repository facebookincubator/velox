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

#include <chrono>

#include "velox/common/base/Macros.h"
#include "velox/functions/lib/DateTimeUnitArithmetic.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions {

/// Returns toTimestamp - fromTimestamp expressed in terms of unit. See
/// diffEpochTime for the rounding rules.
/// @param respectLastDay If true, a toTimestamp on the last day of its month
/// completes the month whatever the day of fromTimestamp: '2020-01-31' to
/// '2020-02-29' is 1 month. If false it is 0 months, which is what Spark
/// expects.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int64_t diffTimestamp(
    DateTimeUnit unit,
    const Timestamp& fromTimestamp,
    const Timestamp& toTimestamp,
    bool respectLastDay = true) {
  return diffEpochTime(
      unit,
      {fromTimestamp.getSeconds(), fromTimestamp.getNanos()},
      {toTimestamp.getSeconds(), toTimestamp.getNanos()},
      respectLastDay);
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int64_t diffTimestamp(
    DateTimeUnit unit,
    const Timestamp& fromTimestamp,
    const Timestamp& toTimestamp,
    const tz::TimeZone* timeZone,
    bool respectLastDay = true) {
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
    return diffTimestamp(
        unit, fromZonedTimestamp, toZonedTimestamp, respectLastDay);
  }
  return diffTimestamp(unit, fromTimestamp, toTimestamp, respectLastDay);
}

/// Returns toDate - fromDate expressed in terms of unit.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int64_t diffDate(
    const DateTimeUnit unit,
    const int32_t fromDate,
    const int32_t toDate) {
  return diffDays(unit, fromDate, toDate);
}

/// Adds value units to a DATE. See addToDays for the end-of-month rules.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int32_t
addToDate(int32_t input, DateTimeUnit unit, int32_t value) {
  return addToDays(input, unit, value);
}

/// Adds value units to a timestamp with no time zone applied. See
/// addToEpochTime.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE Timestamp
addToTimestamp(const Timestamp& timestamp, DateTimeUnit unit, int32_t value) {
  const auto result = addToEpochTime(
      {timestamp.getSeconds(), timestamp.getNanos()}, unit, value);
  return Timestamp(result.seconds, result.nanos);
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE Timestamp addToTimestamp(
    DateTimeUnit unit,
    int32_t value,
    const Timestamp& timestamp,
    const tz::TimeZone* timeZone) {
  Timestamp result;
  if (LIKELY(timeZone != nullptr)) {
    // timeZone not null means that the config
    // adjust_timestamp_to_timezone is on.
    Timestamp zonedTimestamp = timestamp;
    zonedTimestamp.toTimezone(*timeZone);

    Timestamp resultTimestamp = addToTimestamp(zonedTimestamp, unit, value);

    if (isTimeUnit(unit)) {
      const int64_t offset =
          timestamp.getSeconds() - zonedTimestamp.getSeconds();
      result = Timestamp(
          resultTimestamp.getSeconds() + offset, resultTimestamp.getNanos());
    } else {
      result = Timestamp(
          timeZone
              ->correct_nonexistent_time(
                  std::chrono::seconds(resultTimestamp.getSeconds()))
              .count(),
          resultTimestamp.getNanos());
      result.toGMT(*timeZone);
    }
  } else {
    result = addToTimestamp(timestamp, unit, value);
  }
  return result;
}

/// Adds the specified unit and value to a TIME value, handling 24-hour
/// wraparound. TIME represents milliseconds since midnight (0 to 86399999ms).
/// For units < DAY, the time of day changes.
/// For units >= DAY, the time of day doesn't change.
FOLLY_ALWAYS_INLINE int64_t addToTime(int64_t time, int64_t valueInMillis) {
  VELOX_USER_CHECK(
      time >= 0 && time < kMillisInDay,
      "TIME value {} is out of range [0, 86400000)",
      time);

  if (FOLLY_UNLIKELY(valueInMillis == 0)) {
    return time;
  }

  // Use std::chrono for safe duration arithmetic with overflow protection
  const auto timeDuration = std::chrono::milliseconds(time);
  const auto valueDuration = std::chrono::milliseconds(valueInMillis);
  const auto resultDuration = timeDuration + valueDuration;

  // Handle 24-hour wraparound using modulo
  const auto dayDuration = std::chrono::milliseconds(kMillisInDay);
  auto newTime = resultDuration % dayDuration;

  // Ensure result is positive (C++ modulo can return negative values)
  if (FOLLY_UNLIKELY(newTime.count() < 0)) {
    newTime += dayDuration;
  }

  return newTime.count();
}

/// Truncates a TIME value to the specified unit. TIME represents milliseconds
/// since midnight (0 to 86399999ms). Only time-related units (millisecond,
/// second, minute, hour) are supported.
FOLLY_ALWAYS_INLINE int64_t truncateTime(int64_t time, DateTimeUnit unit) {
  VELOX_USER_CHECK(
      time >= 0 && time < kMillisInDay,
      "TIME value {} is out of range [0, 86400000)",
      time);

  // Validate that the unit is appropriate for TIME type
  VELOX_USER_CHECK(isTimeUnit(unit), "Unsupported time unit for TIME type");

  const auto duration = std::chrono::milliseconds(time);

  switch (unit) {
    case DateTimeUnit::kMillisecond:
      return time;
    case DateTimeUnit::kSecond: {
      const auto seconds =
          std::chrono::duration_cast<std::chrono::seconds>(duration);
      return std::chrono::duration_cast<std::chrono::milliseconds>(seconds)
          .count();
    }
    case DateTimeUnit::kMinute: {
      const auto minutes =
          std::chrono::duration_cast<std::chrono::minutes>(duration);
      return std::chrono::duration_cast<std::chrono::milliseconds>(minutes)
          .count();
    }
    case DateTimeUnit::kHour: {
      const auto hours =
          std::chrono::duration_cast<std::chrono::hours>(duration);
      return std::chrono::duration_cast<std::chrono::milliseconds>(hours)
          .count();
    }
    default:
      VELOX_UNREACHABLE();
  }
}
} // namespace facebook::velox::functions
