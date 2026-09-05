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
#include "velox/type/TimestampCalendar.h"

#ifndef __CUDACC__
#include "velox/type/Timestamp.h"
#endif

/// The timezone-free half of TimeUtils.h: the calendar-field accessors that
/// year(), month(), day(), quarter() and day_of_year() are built from.
///
/// Split out because TimeUtils.h reaches Boost, QueryConfig, the timezone
/// database and the datetime formatter, none of which a CUDA translation unit
/// can parse -- and none of which are involved in reading the month out of a
/// std::tm. TimeUtils.h includes this and adds everything timezone-aware on
/// top, so existing callers see no change.
namespace facebook::velox::functions {

inline constexpr int64_t kSecondsInMinute = 60;
inline constexpr int64_t kMinutesInHour = 60;
inline constexpr int64_t kSecondsInHour = kSecondsInMinute * kMinutesInHour;
inline constexpr int64_t kSecondsInDay = 86'400;
inline constexpr int64_t kDaysInWeek = 7;

/// Broken-down UTC time for an epoch-second count.
///
/// On the host this goes through Timestamp::epochToCalendarUtc, which falls
/// back to the wide-range conversion outside the Neri-Schneider window. Device
/// code has no such fallback and uses the fast path directly; the window spans
/// roughly three million years, so nothing representable as a Velox DATE or
/// TIMESTAMP falls outside it.
VELOX_GPU_COMPATIBLE inline std::tm getDateTimeUtc(int64_t seconds) {
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

/// days is the number of days since Epoch.
VELOX_GPU_COMPATIBLE inline std::tm getDateTime(int32_t days) {
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

VELOX_GPU_COMPATIBLE inline int getYear(const std::tm& time) {
  // tm_year: years since 1900.
  return 1900 + time.tm_year;
}

VELOX_GPU_COMPATIBLE inline int getMonth(const std::tm& time) {
  // tm_mon: months since January – [0, 11].
  return 1 + time.tm_mon;
}

VELOX_GPU_COMPATIBLE inline int getDay(const std::tm& time) {
  return time.tm_mday;
}

VELOX_GPU_COMPATIBLE inline int32_t getQuarter(const std::tm& time) {
  return time.tm_mon / 3 + 1;
}

VELOX_GPU_COMPATIBLE inline int32_t getDayOfYear(const std::tm& time) {
  return time.tm_yday + 1;
}

} // namespace facebook::velox::functions
