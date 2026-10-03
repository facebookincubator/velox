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

#include "velox/experimental/cudf/functions/GpuTimeZone.h"

#include "velox/common/base/Macros.h"

#include <cstdint>

/// Every Velox time zone in a form device code can apply, indexed by the time
/// zone id. Names only POD types, so it is safe on both sides of the shadow
/// include path.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// A view of a device table holding one GpuTimeZone per Velox time zone id.
/// A TIMESTAMP WITH TIME ZONE row carries its own zone id, so a kernel reading
/// such a column needs every zone at hand rather than one chosen at
/// initialize() time.
///
/// Trivially copyable, so a function struct can carry it to the kernel. The
/// table is owned by a process-wide cache and outlives every instance.
struct GpuTimeZoneDatabase {
  /// Marks an id with no zone: an entry with no table and an offset no real
  /// zone has, since fixed offsets stay within fourteen hours of UTC.
  static constexpr int32_t kAbsentOffset = INT32_MIN;

  /// One entry per id in [0, numZones). Ids the Velox database leaves empty
  /// hold an absent entry.
  const GpuTimeZone* zones{nullptr};
  /// One past the largest id the table covers.
  int32_t numZones{0};

  /// The zone with the given Velox time zone id, or null when there is none: an
  /// id outside [0, numZones) or one the Velox database leaves empty. Never
  /// fails, so the caller decides how to report a bad id.
  VELOX_GPU_COMPATIBLE const GpuTimeZone* zone(int32_t timeZoneId) const {
    if (timeZoneId < 0 || timeZoneId >= numZones) {
      return nullptr;
    }
    const GpuTimeZone* candidate = zones + timeZoneId;
    if (candidate->transitions == nullptr &&
        candidate->fixedOffset == kAbsentOffset) {
      return nullptr;
    }
    return candidate;
  }
};

/// The database for the current device, built on first use from every zone
/// tz::locateZone() resolves and kept for the life of the process. Building it
/// tabulates every named zone, which takes tens of milliseconds. Defined on the
/// host.
GpuTimeZoneDatabase gpuTimeZoneDatabase();

} // namespace facebook::velox::cudf_velox::gpu_sfi
