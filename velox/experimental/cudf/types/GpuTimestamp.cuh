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

#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"
#include <cstdint>

namespace facebook::velox::gpu {

struct GpuTimestamp {
  int64_t seconds{0};
  uint64_t nanos{0};

  GPU_HOST_DEVICE GpuTimestamp() = default;
  GPU_HOST_DEVICE GpuTimestamp(int64_t s, uint64_t n) : seconds(s), nanos(n) {}

  GPU_HOST_DEVICE bool operator==(const GpuTimestamp& o) const {
    return seconds == o.seconds && nanos == o.nanos;
  }
  GPU_HOST_DEVICE bool operator!=(const GpuTimestamp& o) const {
    return !(*this == o);
  }
  GPU_HOST_DEVICE bool operator<(const GpuTimestamp& o) const {
    return seconds < o.seconds ||
        (seconds == o.seconds && nanos < o.nanos);
  }
  GPU_HOST_DEVICE bool operator<=(const GpuTimestamp& o) const {
    return *this == o || *this < o;
  }
  GPU_HOST_DEVICE bool operator>(const GpuTimestamp& o) const {
    return !(*this <= o);
  }
  GPU_HOST_DEVICE bool operator>=(const GpuTimestamp& o) const {
    return !(*this < o);
  }
};

} // namespace facebook::velox::gpu
