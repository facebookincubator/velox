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

#include <boost/algorithm/string/case_conv.hpp>

#include "velox/core/QueryConfig.h"

namespace facebook::velox::functions {

const folly::F14FastMap<std::string, int8_t> kDayOfWeekNames{
    {"th", 0},       {"fr", 1},     {"sa", 2},       {"su", 3},
    {"mo", 4},       {"tu", 5},     {"we", 6},       {"thu", 0},
    {"fri", 1},      {"sat", 2},    {"sun", 3},      {"mon", 4},
    {"tue", 5},      {"wed", 6},    {"thursday", 0}, {"friday", 1},
    {"saturday", 2}, {"sunday", 3}, {"monday", 4},   {"tuesday", 5},
    {"wednesday", 6}};

const tz::TimeZone* getSessionTimeZone(const core::QueryConfig& config) {
  return getSessionTimeZone(config.sessionTimezone());
}

const tz::TimeZone* getTimeZoneFromConfig(const core::QueryConfig& config) {
  if (config.adjustTimestampToTimezone()) {
    auto sessionTzName = config.sessionTimezone();
    if (!sessionTzName.empty()) {
      return tz::locateZone(sessionTzName);
    }
  }
  return nullptr;
}

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

// The device-compatible arithmetic restates the Timestamp bounds; keep the two
// in step.
static_assert(kMaxEpochSeconds == Timestamp::kMaxSeconds);
static_assert(kMinEpochSeconds == Timestamp::kMinSeconds);

} // namespace facebook::velox::functions
