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

#include "velox/common/base/Macros.h"
#include "velox/type/FastDate.h"

/// Conversion between a UTC epoch-second count and a broken-down std::tm.
///
/// This is the core every date/time extractor funnels through: year, month,
/// day, quarter, day_of_week, day_of_year, year_of_week, hour, minute and
/// second all reach it via getDateTime(). It lives in a header, separate from
/// Timestamp.cpp, so that callers which only need calendar arithmetic -- and
/// in particular a CUDA translation unit -- do not have to link against the
/// rest of the Timestamp implementation.
///
/// Timestamp::epochToCalendarUtc delegates here for the common range and
/// falls back to WideRangeDateConversion outside it.
namespace facebook::velox {

namespace calendar {

/// std::tm counts years from 1900.
inline constexpr int kTmYearBase = 1900;
inline constexpr int64_t kSecondsPerHour = 3600;
inline constexpr int64_t kSecondsPerDay = 24 * kSecondsPerHour;

/// Offset applied before the leap-year division so that the arithmetic stays
/// correct for negative years.
inline constexpr int64_t kLeapYearOffset = 4000000000ll;

VELOX_GPU_COMPATIBLE inline bool isLeap(int64_t y) {
  return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

VELOX_GPU_COMPATIBLE inline int64_t leapThroughEndOf(int64_t y) {
  y += kLeapYearOffset;
  return y / 4 - y / 100 + y / 400;
}

VELOX_GPU_COMPATIBLE inline int64_t daysBetweenYears(int64_t y1, int64_t y2) {
  return 365 * (y2 - y1) + leapThroughEndOf(y2 - 1) - leapThroughEndOf(y1 - 1);
}

/// Days elapsed before the first day of `month` (0-based) within a leap or a
/// common year.
///
/// The table is inside the function rather than at namespace scope because
/// nvcc cannot index an object with static storage duration from device code
/// when the index is only known at run time.
VELOX_GPU_COMPATIBLE inline int16_t daysBeforeMonth(bool leap, int month) {
  constexpr int16_t kTable[2][12] = {
      {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
      {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335},
  };
  return kTable[leap ? 1 : 0][month];
}

/// Fills `tm` from a UTC epoch-second count.
///
/// Returns false when `epoch` falls outside the Neri-Schneider range, or when
/// the year does not fit tm_year; the caller is then expected to fall back to
/// the wide-range conversion. The supported range covers roughly three million
/// years centred on the epoch, so the fallback is unreachable for any
/// practical DATE or TIMESTAMP.
VELOX_GPU_COMPATIBLE inline bool epochToCalendarUtc(
    int64_t epoch,
    std::tm& tm) {
  int64_t days = epoch / kSecondsPerDay;
  int64_t rem = epoch % kSecondsPerDay;
  if (rem < 0) {
    rem += kSecondsPerDay;
    --days;
  }
  if (days < fast_date::kRataDieMin || days > fast_date::kRataDieMax) {
    return false;
  }
  tm.tm_hour = rem / kSecondsPerHour;
  rem = rem % kSecondsPerHour;
  tm.tm_min = rem / 60;
  tm.tm_sec = rem % 60;
  tm.tm_wday = (4 + days) % 7;
  if (tm.tm_wday < 0) {
    tm.tm_wday += 7;
  }
  const auto ymd = daysToYmd(static_cast<int32_t>(days));
  const int64_t y = static_cast<int64_t>(ymd.year) - kTmYearBase;
  // std::numeric_limits is not reachable from device code; tm_year is int.
  if (y > INT32_MAX || y < INT32_MIN) {
    return false;
  }
  tm.tm_year = static_cast<int>(y);
  tm.tm_mon = static_cast<int>(ymd.month) - 1;
  tm.tm_mday = static_cast<int>(ymd.day);
  tm.tm_yday = daysBeforeMonth(isLeap(ymd.year), tm.tm_mon) + tm.tm_mday - 1;
  tm.tm_isdst = 0;
  return true;
}

/// Inverse of epochToCalendarUtc. Total, so it needs no fallback. Month values
/// outside [0, 11] are normalised into the year, matching timegm.
VELOX_GPU_COMPATIBLE inline int64_t calendarUtcToEpoch(const std::tm& tm) {
  // Widen before adding: tm_year can be close to INT32_MAX, and int + int
  // would wrap before the result reached the int64_t.
  int64_t year = tm.tm_year + static_cast<int64_t>(kTmYearBase);
  int64_t month = tm.tm_mon;
  if (month > 11) {
    year += month / 12;
    month %= 12;
  } else if (month < 0) {
    const auto yearsDiff = (-month + 11) / 12;
    year -= yearsDiff;
    month += 12 * yearsDiff;
  }
  const auto dayOfYear = -1ll +
      daysBeforeMonth(isLeap(year), static_cast<int>(month)) + tm.tm_mday;
  const auto daysSinceEpoch = daysBetweenYears(1970, year) + dayOfYear;
  return kSecondsPerDay * daysSinceEpoch + kSecondsPerHour * tm.tm_hour +
      60ll * tm.tm_min + tm.tm_sec;
}

} // namespace calendar

} // namespace facebook::velox
