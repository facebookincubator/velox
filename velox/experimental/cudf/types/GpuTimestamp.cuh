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

namespace facebook::velox::gpu {

/// The device type of a Velox Timestamp, which a call() receives through
/// GpuExec. Holds the same fields as Timestamp and orders the same way: by
/// seconds, then by nanoseconds.
struct GpuTimestamp {
  /// Seconds since the epoch, 1970-01-01 00:00:00 UTC.
  int64_t seconds{0};
  /// Nanoseconds within the second.
  uint64_t nanos{0};

  // Defaulted constructors are implicitly callable from both host and device
  // code, so VELOX_GPU_COMPATIBLE is unnecessary here, and nvcc warns about it.
  GpuTimestamp() = default;

  VELOX_GPU_COMPATIBLE GpuTimestamp(int64_t seconds, uint64_t nanos)
      : seconds(seconds), nanos(nanos) {}

  VELOX_GPU_COMPATIBLE bool operator==(const GpuTimestamp& other) const {
    return seconds == other.seconds && nanos == other.nanos;
  }

  VELOX_GPU_COMPATIBLE bool operator!=(const GpuTimestamp& other) const {
    return !(*this == other);
  }

  VELOX_GPU_COMPATIBLE bool operator<(const GpuTimestamp& other) const {
    return seconds < other.seconds ||
        (seconds == other.seconds && nanos < other.nanos);
  }

  VELOX_GPU_COMPATIBLE bool operator<=(const GpuTimestamp& other) const {
    return *this == other || *this < other;
  }

  VELOX_GPU_COMPATIBLE bool operator>(const GpuTimestamp& other) const {
    return !(*this <= other);
  }

  VELOX_GPU_COMPATIBLE bool operator>=(const GpuTimestamp& other) const {
    return !(*this < other);
  }
};

} // namespace facebook::velox::gpu
