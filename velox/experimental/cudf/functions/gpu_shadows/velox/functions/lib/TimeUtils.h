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

// GPU shadow for velox/functions/lib/TimeUtils.h, which it includes: the
// conversions, InitSessionTimezone and truncateTimestamp() are the real ones,
// already device-callable. Four names are redirected around the real header:
//
//   getTimeZoneFromConfig() and getSessionTimeZone() return pointers to host
//   tz::TimeZone objects, which a kernel cannot follow. The shadow resolver
//   returns the zone's device object, built by GpuTimeZone.cpp from the same
//   Velox zone. getSessionTimeZone() has no shadow and is left undeclared: no
//   registered struct calls it, and a new one that does fails to compile here
//   rather than hand a kernel a host pointer.
//
//   fromDateTimeUnitString() is host-only on the CPU. The shadow forwards to
//   it on the host; on the device it declines the row, which only a unit
//   column could reach, and no registered signature accepts one.
//
//   getWeek() reads the week through the date library's iso_week, which is
//   host-only. The shadow reads it off the calendar fields.
//
// Everything defined here carries a name no real translation unit defines: an
// inline definition under a real name would be interchangeable with the real
// one when the program links, and host code would follow whichever the linker
// kept. The macros give the redirected names to the real header's own uses
// (InitSessionTimezone calls getTimeZoneFromConfig()) and to every struct
// parsed after it.
#pragma once

#include "velox/experimental/cudf/functions/GpuDateTimeUnit.h"
#include "velox/experimental/cudf/functions/GpuTimeZone.h"

// Redirected before the real header is read, so that its InitSessionTimezone
// binds to the shadow resolver and its own declaration declares the shadow.
#define getTimeZoneFromConfig gpuShadowGetTimeZoneFromConfig
// Renamed while the real header is read, so that its declarations go under
// names nothing calls: the real getSessionTimeZone(std::string_view) and
// getWeek() are definitions, which the shadow cannot replace in place.
#define getSessionTimeZone gpuShadowRealGetSessionTimeZone
#define fromDateTimeUnitString gpuShadowRealFromDateTimeUnitString
#define getWeek gpuShadowRealGetWeek
#include_next "velox/functions/lib/TimeUtils.h"
#undef getSessionTimeZone
#undef fromDateTimeUnitString
#undef getWeek
#define fromDateTimeUnitString gpuShadowFromDateTimeUnitString
#define getWeek gpuShadowGetWeek

namespace facebook::velox::functions {

/// The session zone's device object when adjust_timestamp_to_session_timezone
/// is set, null otherwise, as getTimeZoneFromConfig() selects it. Host only,
/// like the real one.
inline const tz::TimeZone* gpuShadowGetTimeZoneFromConfig(
    const core::QueryConfig& config) {
  return static_cast<const tz::TimeZone*>(
      cudf_velox::gpu_sfi::gpuDeviceTimeZoneFromConfig(config));
}

/// fromDateTimeUnitString() for the device. On the host the real parser
/// answers. On the device the row is declined, and the host re-evaluates it
/// for the error. The unit returned is one the DATE and TIMESTAMP filters
/// around the parser accept, so that no caller reaches `.value()` on the empty
/// optional a rejected unit yields: that read is undefined on the device, and
/// the compiler may drop the raise before it as unreachable. The TIME filter
/// rejects it, and no TIME function is registered.
VELOX_GPU_COMPATIBLE inline std::optional<DateTimeUnit>
gpuShadowFromDateTimeUnitString(
    StringView unitString,
    bool throwIfInvalid,
    bool allowMicro = false,
    bool allowAbbreviated = false) {
#ifdef __CUDA_ARCH__
  (void)throwIfInvalid;
  (void)allowMicro;
  (void)allowAbbreviated;
  VELOX_USER_FAIL("Unsupported datetime unit: {}", unitString);
  return DateTimeUnit::kDay;
#else
  return cudf_velox::gpu_sfi::gpuFromDateTimeUnitString(
      std::string_view(unitString.data(), unitString.size()),
      throwIfInvalid,
      allowMicro,
      allowAbbreviated);
#endif
}

/// The ISO 8601 week of 'timestamp' in 'timezone', or in UTC when it is null,
/// as getWeek() computes it. A week belongs to the year of its Thursday and is
/// numbered by that Thursday's day of the year. Years the date library rejects
/// are the user error the CPU raises.
VELOX_GPU_COMPATIBLE inline uint32_t gpuShadowGetWeek(
    const Timestamp& timestamp,
    const tz::TimeZone* timezone,
    bool /*allowOverflow*/) {
  // Day numbers of -32767-01-01 and 32767-12-31, the years date::year spans.
  constexpr int64_t kMinDays = -12'687'429;
  constexpr int64_t kMaxDays = 11'248'737;
  const int64_t days =
      detail::floorDivide(getSeconds(timestamp, timezone), kSecondsInDay);
  VELOX_USER_CHECK(
      days >= kMinDays && days <= kMaxDays,
      "Timepoint is outside of supported year range: [-32767, 32767]");
  // 1970-01-01 was a Thursday, ISO weekday 4.
  const int64_t isoWeekday = days - detail::floorDivide(days + 3, 7) * 7 + 4;
  const int64_t thursday = days - isoWeekday + 4;
  return getDateTimeUtc(thursday * kSecondsInDay).tm_yday / 7 + 1;
}

} // namespace facebook::velox::functions
