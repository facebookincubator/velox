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

namespace facebook::velox::tz {
class TimeZone;
} // namespace facebook::velox::tz

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

  /// A UTC instant for a local wall-clock time, or the report that the local
  /// time was skipped.
  struct UtcInstant {
    /// Seconds since the epoch. In a gap, the instant the offset in force
    /// before the change gives, which is also what correct_nonexistent_time()
    /// followed by to_sys() produces.
    int64_t utcSeconds;
    /// False for a local time that an offset increase skipped.
    /// Timestamp::toGMT() raises a user error there; a device caller does the
    /// same through VELOX_USER_FAIL, since this header cannot throw.
    bool exists;
  };

  /// UTC instant for a local wall-clock time, as Timestamp::toGMT() computes
  /// it: a local time that an offset decrease repeats maps to the earlier
  /// instant, and one that an offset increase skipped has exists == false.
  VELOX_GPU_COMPATIBLE UtcInstant toUtc(int64_t localSeconds) const {
    if (transitions == nullptr) {
      return {localSeconds - fixedOffset, true};
    }
    // Folded by whole cycles exactly as toLocal() folds the instant, so the
    // two stay inverse of each other across the table's end.
    int64_t local = localSeconds;
    int64_t foldedCycles = 0;
    if (local >= kTableEndSeconds) {
      const int64_t cycleStart = kTableEndSeconds - kCycleSeconds;
      const int64_t sinceCycleStart = (local - cycleStart) % kCycleSeconds;
      foldedCycles = local - cycleStart - sinceCycleStart;
      local = cycleStart + sinceCycleStart;
    }
    // Last interval whose local start, transitions[i] + offsets[i], is at or
    // before the local time. Interval 0 starts at INT64_MIN, which must not be
    // offset, and always qualifies; the search never reads it.
    int32_t low = 0;
    int32_t high = numTransitions - 1;
    while (low < high) {
      const int32_t middle = low + (high - low + 1) / 2;
      if (transitions[middle] + offsets[middle] <= local) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    // After an offset decrease the previous interval still covers the local
    // time; its instant is the earlier one.
    if (low > 0 && local < transitions[low] + offsets[low - 1]) {
      return {local - offsets[low - 1] + foldedCycles, true};
    }
    // An offset increase ends an interval, in local time, before the next one
    // starts; local times in between belong to no interval.
    const bool exists = low == numTransitions - 1 ||
        local < transitions[low + 1] + offsets[low];
    return {local - offsets[low] + foldedCycles, exists};
  }
};

/// The time zone getTimeZoneFromConfig() selects: the session time zone when
/// adjust_timestamp_to_session_timezone is set, UTC otherwise. Defined on the
/// host; initialize() calls it from the shadow-compiled side.
GpuTimeZone gpuSessionTimeZone(const core::QueryConfig& config);

/// The device form of one Velox time zone. A named zone's table is built on
/// first use and cached per device for the life of the process. Defined on the
/// host.
GpuTimeZone gpuTimeZone(const tz::TimeZone& zone);

} // namespace facebook::velox::cudf_velox::gpu_sfi
