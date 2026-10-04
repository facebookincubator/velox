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

// GPU shadow for
// velox/functions/prestosql/types/TimestampWithTimeZoneRenderZone.h.
//
// The real class keeps the session zone's name in a std::string, to resolve it
// on demand through the host time zone index, and holds a host tz::TimeZone
// pointer. A function instance reaches the kernel as a trivially copyable
// argument and cannot resolve a name there, so this one resolves the zone when
// it is built, on the host in initialize(), and keeps only the device object.
// get() runs on the device and selects the zone as the real one does.
#pragma once

#include "velox/experimental/cudf/functions/GpuTimeZone.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cstdint>

namespace facebook::velox::core {
class QueryConfig;
} // namespace facebook::velox::core

namespace facebook::velox {

class TimestampWithTimeZoneRenderZone {
 public:
  enum class SessionZoneResolution { kEager, kOnDemand };

  explicit TimestampWithTimeZoneRenderZone(const core::QueryConfig& config)
      : TimestampWithTimeZoneRenderZone(config, SessionZoneResolution::kEager) {
  }

  /// An on-demand session zone is resolved here as well, since the device
  /// cannot resolve one per row, but without failing: get() raises for a name
  /// the host could not resolve, where the CPU raises on first use.
  TimestampWithTimeZoneRenderZone(
      const core::QueryConfig& config,
      SessionZoneResolution resolution)
      : legacyTimestampWithTimeZone_(
            cudf_velox::gpu_sfi::gpuLegacyTimestampWithTimezone(config)),
        sessionRenderZone_(
            legacyTimestampWithTimeZone_
                ? nullptr
                : static_cast<const tz::TimeZone*>(
                      cudf_velox::gpu_sfi::gpuDeviceSessionTimeZone(
                          config,
                          resolution == SessionZoneResolution::kEager))) {}

  /// The value's embedded zone in legacy mode and the session zone otherwise.
  VELOX_GPU_COMPATIBLE const tz::TimeZone* get(
      int64_t timestampWithTimeZone) const {
    if (legacyTimestampWithTimeZone_) {
      return tz::locateZone(unpackZoneKeyId(timestampWithTimeZone));
    }
    if (sessionRenderZone_ == nullptr) {
      VELOX_USER_FAIL("Unknown time zone");
      return tz::locateZone(0);
    }
    return sessionRenderZone_;
  }

 private:
  // Preserves embedded-zone rendering when true.
  bool legacyTimestampWithTimeZone_;
  // The session zone's device object; null under legacy rendering, and for a
  // name the host could not resolve.
  const tz::TimeZone* sessionRenderZone_;
};

} // namespace facebook::velox
