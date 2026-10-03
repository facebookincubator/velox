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

#include "velox/common/base/Macros.h"

#include <cstdint>

namespace facebook::velox::core {
class QueryConfig;
} // namespace facebook::velox::core

/// The session time zone in a form device code can apply. Names only POD
/// types, so it is safe on both sides of the shadow include path.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// UTC-to-local conversion for one time zone. A fixed-offset zone has no table
/// and applies fixedOffset. A named zone points at device memory holding every
/// offset change up to kTableEndSeconds, built on the host from the Velox time
/// zone database; later instants fold back by whole 400-year Gregorian cycles,
/// over which the day-of-week daylight saving rules repeat.
///
/// Trivially copyable, so a function struct can carry it to the kernel. The
/// table is owned by a process-wide cache and outlives every instance.
struct GpuTimeZone {
  /// Seconds in 400 Gregorian years.
  static constexpr int64_t kCycleSeconds = 146'097LL * 86'400;
  /// 2800-01-01T00:00:00Z: a whole cycle after the database's last rule.
  static constexpr int64_t kTableEndSeconds = 26'192'246'400LL;

  /// UTC instants, ascending, at which offsets[i] starts to apply. The first is
  /// INT64_MIN. Null for a fixed-offset zone.
  const int64_t* transitions{nullptr};
  /// Seconds east of UTC, one per transition.
  const int32_t* offsets{nullptr};
  int32_t numTransitions{0};
  /// Seconds east of UTC when there is no table.
  int32_t fixedOffset{0};

  /// Local wall-clock seconds for a UTC instant, as tz::TimeZone::to_local()
  /// computes them.
  VELOX_GPU_COMPATIBLE int64_t toLocal(int64_t utcSeconds) const {
    if (transitions == nullptr) {
      return utcSeconds + fixedOffset;
    }
    int64_t instant = utcSeconds;
    if (instant >= kTableEndSeconds) {
      const int64_t cycleStart = kTableEndSeconds - kCycleSeconds;
      const int64_t sinceCycleStart = (instant - cycleStart) % kCycleSeconds;
      instant = cycleStart + sinceCycleStart;
    }
    // Last transition at or before the instant. transitions[0] is INT64_MIN,
    // so there always is one.
    int32_t low = 0;
    int32_t high = numTransitions - 1;
    while (low < high) {
      const int32_t middle = low + (high - low + 1) / 2;
      if (transitions[middle] <= instant) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    return utcSeconds + offsets[low];
  }
};

/// The time zone getTimeZoneFromConfig() selects: the session time zone when
/// adjust_timestamp_to_session_timezone is set, UTC otherwise. Defined on the
/// host; initialize() calls it from the shadow-compiled side.
GpuTimeZone gpuSessionTimeZone(const core::QueryConfig& config);

} // namespace facebook::velox::cudf_velox::gpu_sfi
