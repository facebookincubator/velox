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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/container/F14Map.h>

#include "velox/common/base/Macros.h"
#include "velox/external/date/date.h"
#include "velox/external/date/iso_week.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/DateTimeUnit.h"
#include "velox/functions/lib/DateTimeUnitArithmetic.h"
#include "velox/functions/lib/TimeUtilsCore.h"
#include "velox/type/StringView.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::core {
class QueryConfig;
}

namespace facebook::velox::functions {

extern const folly::F14FastMap<std::string, int8_t> kDayOfWeekNames;

/// Returns the configured session time zone, or GMT when it is unset.
FOLLY_ALWAYS_INLINE const tz::TimeZone* getSessionTimeZone(
    std::string_view sessionTimeZoneName) {
  return sessionTimeZoneName.empty() ? tz::locateZone(0)
                                     : tz::locateZone(sessionTimeZoneName);
}

/// Returns the session time zone of the query, or GMT when it is unset.
const tz::TimeZone* getSessionTimeZone(const core::QueryConfig& config);

/// Returns the session time zone when the query adjusts timestamps to it, and
/// null otherwise.
const tz::TimeZone* getTimeZoneFromConfig(const core::QueryConfig& config);

/// Returns the epoch seconds of 'timestamp' as seen in 'timeZone', or as
/// stored when 'timeZone' is null.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int64_t
getSeconds(Timestamp timestamp, const tz::TimeZone* timeZone) {
  if (timeZone != nullptr) {
    timestamp.toTimezone(*timeZone);
  }
  return timestamp.getSeconds();
}

/// Returns the broken-down time of 'timestamp' as seen in 'timeZone', or in
/// UTC when 'timeZone' is null.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE std::tm getDateTime(
    Timestamp timestamp,
    const tz::TimeZone* timeZone) {
  return getDateTimeUtc(getSeconds(timestamp, timeZone));
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

/// Returns timestamp with seconds adjusted to the nearest lower multiple of the
/// specified interval. If the given seconds is negative and not an exact
/// multiple of the interval, it adjusts further down.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE Timestamp
adjustEpoch(int64_t seconds, int64_t intervalSeconds) {
  int64_t s = seconds / intervalSeconds;
  if (seconds < 0 && seconds % intervalSeconds) {
    s = s - 1;
  }
  int64_t truncatedSeconds = s * intervalSeconds;
  return Timestamp(truncatedSeconds, 0);
}

/// Returns 'timestamp' truncated to the start of 'unit' as seen in 'timeZone',
/// or in UTC when 'timeZone' is null.
VELOX_GPU_COMPATIBLE inline Timestamp truncateTimestamp(
    Timestamp timestamp,
    DateTimeUnit unit,
    const tz::TimeZone* timeZone) {
  switch (unit) {
    // Units up to a minute truncate the UTC value directly: time zone offsets
    // and daylight saving shifts are whole minutes.
    case DateTimeUnit::kMicrosecond:
    case DateTimeUnit::kMillisecond:
    case DateTimeUnit::kSecond:
    case DateTimeUnit::kMinute: {
      const auto truncated = truncateEpochTime(
          {timestamp.getSeconds(), timestamp.getNanos()}, unit);
      return Timestamp(truncated.seconds, truncated.nanos);
    }

    // Hour truncation has to handle the corner case of daylight savings time
    // boundaries. Since conversions from local timezone to UTC may be
    // ambiguous, we need to be carefull about the roundtrip of converting to
    // local time and back. So what we do is to calculate the truncation delta
    // in UTC, then applying it to the input timestamp.
    case DateTimeUnit::kHour: {
      const int64_t localSeconds = getSeconds(timestamp, timeZone);
      const int64_t secondsDelta =
          localSeconds - truncateEpochTime({localSeconds, 0}, unit).seconds;
      return Timestamp(timestamp.getSeconds() - secondsDelta, 0);
    }

    // For the truncations below, we may first need to convert to the local
    // timestamp, truncate, then convert back to GMT.
    default: {
      const EpochTime local{
          getSeconds(timestamp, timeZone), timestamp.getNanos()};
      Timestamp result(truncateEpochTime(local, unit).seconds, 0);
      if (timeZone != nullptr) {
        result.toGMT(*timeZone);
      }
      return result;
    }
  }
}

} // namespace facebook::velox::functions
