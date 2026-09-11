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

#include "velox/functions/Macros.h"
#include "velox/functions/sparksql/AnsiMode.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/TimestampUtils.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::functions::sparksql {

namespace detail {

inline std::optional<Timestamp> buildTimestampFromFields(
    int32_t year,
    int32_t month,
    int32_t day,
    int32_t hour,
    int32_t minute,
    int64_t micros,
    bool ansiEnabled) {
  if (hour < 0 || hour >= 24) {
    return nullOrUserFail(
        ansiEnabled, "Invalid value for hour, must be in [0, 24): {}", hour);
  }
  if (minute < 0 || minute >= 60) {
    return nullOrUserFail(
        ansiEnabled,
        "Invalid value for minute, must be in [0, 60): {}",
        minute);
  }
  if (micros < 0) {
    return nullOrUserFail(
        ansiEnabled,
        "Invalid value for second microseconds, must be non-negative: {}",
        micros);
  }
  auto seconds = micros / util::kMicrosPerSec;
  if (micros > 60 * util::kMicrosPerSec) {
    return nullOrUserFail(
        ansiEnabled,
        "Invalid value for second, must be in [0, 60] with 0 microseconds at 60: {}.{:06d}",
        seconds,
        micros % util::kMicrosPerSec);
  }

  Expected<int64_t> daysSinceEpoch =
      util::daysSinceEpochFromDate(year, month, day);
  if (daysSinceEpoch.hasError()) {
    VELOX_DCHECK(daysSinceEpoch.error().isUserError());
    return nullOrUserFail(ansiEnabled, "{}", daysSinceEpoch.error().message());
  }

  // micros <= 60,000,000, which fits in int32_t.
  const auto localMicros =
      util::fromTime(hour, minute, 0, static_cast<int32_t>(micros));
  return util::fromDatetime(daysSinceEpoch.value(), localMicros);
}

} // namespace detail

// make_timestamp / try_make_timestamp / make_timestamp_ntz /
// try_make_timestamp_ntz. The 7-argument (explicit timezone) overload is
// only registered for TTimestamp == Timestamp. When kTry, invalid input
// always returns NULL, regardless of ANSI mode.
template <typename T, typename TTimestamp, bool kTry>
struct MakeTimestampFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const int32_t* /*year*/,
      const int32_t* /*month*/,
      const int32_t* /*day*/,
      const int32_t* /*hour*/,
      const int32_t* /*minute*/,
      const int64_t* /*micros*/) {
    if constexpr (kTry) {
      ansiEnabled_ = false;
    } else {
      ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
    }
    if constexpr (std::is_same_v<TTimestamp, TimestampUtc>) {
      // TIMESTAMP UTC represents a timestamp in UTC, not subject to session
      // timezone adjustment.
      sessionTimeZone_ = nullptr;
    } else {
      const auto sessionTzName = config.sessionTimezone();
      VELOX_USER_CHECK(
          !sessionTzName.empty(),
          "make_timestamp requires session time zone to be set.");
      sessionTimeZone_ = tz::locateZone(sessionTzName);
    }
  }

  // Caches the timezone once when it's constant; null means it varies per
  // row, resolved in call() instead.
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const int32_t* /*year*/,
      const int32_t* /*month*/,
      const int32_t* /*day*/,
      const int32_t* /*hour*/,
      const int32_t* /*minute*/,
      const int64_t* /*micros*/,
      const arg_type<Varchar>* timezone) {
    if constexpr (kTry) {
      ansiEnabled_ = false;
    } else {
      ansiEnabled_ = SparkQueryConfig{config}.ansiEnabled();
    }
    if (timezone != nullptr) {
      constantTimeZoneName_ = std::string(*timezone);
      constantTimeZone_ = tz::locateZone(
          std::string_view(constantTimeZoneName_), /*failOnError=*/false);
      hasConstantTimeZone_ = true;
    }
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<TTimestamp>& result,
      int32_t year,
      int32_t month,
      int32_t day,
      int32_t hour,
      int32_t minute,
      int64_t micros) {
    auto timestamp = detail::buildTimestampFromFields(
        year, month, day, hour, minute, micros, ansiEnabled_);
    if (!timestamp.has_value()) {
      return false;
    }
    if (sessionTimeZone_ != nullptr) {
      toGMTWithGapCorrection(*timestamp, *sessionTimeZone_);
    }
    result = *timestamp;
    return true;
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Timestamp>& result,
      int32_t year,
      int32_t month,
      int32_t day,
      int32_t hour,
      int32_t minute,
      int64_t micros,
      const arg_type<Varchar>& timezone) {
    auto timestamp = detail::buildTimestampFromFields(
        year, month, day, hour, minute, micros, ansiEnabled_);
    if (!timestamp.has_value()) {
      return false;
    }
    const tz::TimeZone* zone;
    std::string_view zoneName;
    if (hasConstantTimeZone_) {
      zone = constantTimeZone_;
      zoneName = constantTimeZoneName_;
    } else {
      zoneName = std::string_view(timezone);
      zone = tz::locateZone(zoneName, /*failOnError=*/false);
    }
    if (zone == nullptr) {
      nullOrUserFail(ansiEnabled_, "Unknown time zone: '{}'", zoneName);
      return false;
    }
    toGMTWithGapCorrection(*timestamp, *zone);
    result = *timestamp;
    return true;
  }

 private:
  bool ansiEnabled_{false};
  // Null when TTimestamp == TimestampUtc.
  const tz::TimeZone* sessionTimeZone_{nullptr};
  // Set when the 7-argument form's timezone is constant.
  bool hasConstantTimeZone_{false};
  std::string constantTimeZoneName_;
  const tz::TimeZone* constantTimeZone_{nullptr};
};

} // namespace facebook::velox::functions::sparksql
