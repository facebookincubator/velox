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

#include "velox/functions/lib/TimeUtils.h"

namespace facebook::velox::functions {

const folly::F14FastMap<std::string, int8_t> kDayOfWeekNames{
    {"th", 0},       {"fr", 1},     {"sa", 2},       {"su", 3},
    {"mo", 4},       {"tu", 5},     {"we", 6},       {"thu", 0},
    {"fri", 1},      {"sat", 2},    {"sun", 3},      {"mon", 4},
    {"tue", 5},      {"wed", 6},    {"thursday", 0}, {"friday", 1},
    {"saturday", 2}, {"sunday", 3}, {"monday", 4},   {"tuesday", 5},
    {"wednesday", 6}};

std::optional<DateTimeUnit> fromDateTimeUnitString(
    StringView unitString,
    bool throwIfInvalid,
    bool allowMicro,
    bool allowAbbreviated) {
  const auto unit = boost::algorithm::to_lower_copy(unitString.str());

  if (unit == "microsecond" && allowMicro) {
    return DateTimeUnit::kMicrosecond;
  }
  if (unit == "millisecond") {
    return DateTimeUnit::kMillisecond;
  }
  if (unit == "second") {
    return DateTimeUnit::kSecond;
  }
  if (unit == "minute") {
    return DateTimeUnit::kMinute;
  }
  if (unit == "hour") {
    return DateTimeUnit::kHour;
  }
  if (unit == "day") {
    return DateTimeUnit::kDay;
  }
  if (unit == "week") {
    return DateTimeUnit::kWeek;
  }
  if (unit == "month") {
    return DateTimeUnit::kMonth;
  }
  if (unit == "quarter") {
    return DateTimeUnit::kQuarter;
  }
  if (unit == "year") {
    return DateTimeUnit::kYear;
  }
  if (allowAbbreviated) {
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
  if (throwIfInvalid) {
    VELOX_UNSUPPORTED("Unsupported datetime unit: {}", unitString);
  }
  return std::nullopt;
}

Timestamp truncateTimestamp(
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
