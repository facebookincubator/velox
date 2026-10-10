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

#include <folly/CPortability.h>

#include <cstdint>
#include <ctime>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"
#include "velox/type/TimestampCalendar.h"

#ifndef __CUDACC__
#include "velox/type/Timestamp.h"
#endif

/// Time-zone-free calendar-field accessors that year(), month(), day(),
/// quarter() and day_of_year() are built from. Kept apart from TimeUtils.h,
/// which includes this, so that CUDA translation units can use them without
/// the time zone database and the datetime formatter.
namespace facebook::velox::functions {

inline constexpr int64_t kSecondsInMinute = 60;
inline constexpr int64_t kMinutesInHour = 60;
inline constexpr int64_t kSecondsInHour = kSecondsInMinute * kMinutesInHour;
inline constexpr int64_t kSecondsInDay = 86'400;
inline constexpr int64_t kDaysInWeek = 7;

/// Broken-down UTC time for an epoch-second count. Device code has no
/// wide-range fallback; the fast path covers roughly three million years.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE std::tm getDateTimeUtc(
    int64_t seconds) {
  std::tm dateTime{};
#ifdef __CUDACC__
  const bool converted = calendar::epochToCalendarUtc(seconds, dateTime);
#else
  const bool converted = Timestamp::epochToCalendarUtc(seconds, dateTime);
#endif
  VELOX_USER_CHECK(
      converted, "Timestamp is too large: {} seconds since epoch", seconds);
  return dateTime;
}

/// Broken-down UTC time for a count of days since the epoch.
VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE std::tm getDateTime(int32_t days) {
  const int64_t seconds = days * kSecondsInDay;
  std::tm dateTime{};
#ifdef __CUDACC__
  const bool converted = calendar::epochToCalendarUtc(seconds, dateTime);
#else
  const bool converted = Timestamp::epochToCalendarUtc(seconds, dateTime);
#endif
  VELOX_USER_CHECK(converted, "Date is too large: {} days", days);
  return dateTime;
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int getYear(const std::tm& time) {
  // tm_year: years since 1900.
  return 1900 + time.tm_year;
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int getMonth(const std::tm& time) {
  // tm_mon: months since January – [0, 11].
  return 1 + time.tm_mon;
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int getDay(const std::tm& time) {
  return time.tm_mday;
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int32_t
getQuarter(const std::tm& time) {
  return time.tm_mon / 3 + 1;
}

VELOX_GPU_COMPATIBLE FOLLY_ALWAYS_INLINE int32_t
getDayOfYear(const std::tm& time) {
  return time.tm_yday + 1;
}

} // namespace facebook::velox::functions
