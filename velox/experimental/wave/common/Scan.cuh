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

#include "velox/experimental/wave/common/BitUtil.cuh"

namespace facebook::velox::wave {

template <typename T, int32_t kNumLanes = kWarpThreads>
struct WarpScan {
  enum {
    /// Whether the logical warp size and the PTX warp size coincide
    IS_ARCH_WARP = (kNumLanes == kWarpThreads),

    /// The number of warp scan steps
    STEPS = Log2<kNumLanes>::VALUE,

    /// The 5-bit SHFL mask for logically splitting warps into sub-segments
    /// starts 8-bits up
    SHFL_C = (kWarpThreads - kNumLanes) << 8

  };

  int laneId;

  static constexpr unsigned int member_mask =
      kNumLanes == 32 ? 0xffffffff : (1 << kNumLanes) - 1;

  __device__ WarpScan() : laneId(LaneId()) {}

  __device__ __forceinline__ void exclusiveSum(
      T input, ///< [in] Calling thread's input item.
      T& exclusive_output) ///< [out] Calling thread's output item.  May be
                           ///< aliased with \p input.
  {
    T initial_value = 0;
    exclusiveSum(input, exclusive_output, initial_value);
  }

  __device__ __forceinline__ void exclusiveSum(
      T input, ///< [in] Calling thread's input item.
      T& exclusive_output, ///< [out] Calling thread's output item.  May be
                           ///< aliased with \p input.
      T initial_value) {
    T inclusive_output;
    inclusiveSum(input, inclusive_output);

    exclusive_output = initial_value + inclusive_output - input;
  }

  __device__ __forceinline__ void ExclusiveSum(
      T input,
      T& exclusive_output,
      T initial_value,
      T& warp_aggregate) {
    T inclusive_output;
    Inclusivesum(input, inclusive_output);
    warp_aggregate = __shfl_sync(member_mask, inclusive_output, kNumLanes - 1);
    exclusive_output = initial_value + inclusive_output - input;
  }

  __device__ __forceinline__ void inclusiveSum(T input, T& inclusive_output) {
    inclusive_output = input;
#pragma unroll
    for (int STEP = 0; STEP < STEPS; STEP++) {
      int offset = (1 << STEP);
      T other = __shfl_up_sync(member_mask, inclusive_output, offset);
      if (laneId >= offset) {
        inclusive_output += other;
      }
    }
  }
};

template <typename T, int32_t kNumLanes = kWarpThreads>
struct WarpReduce {
  static constexpr int32_t STEPS = Log2<kNumLanes>::VALUE;

  int laneId;

  /// 32-thread physical warp member mask of logical warp

  static constexpr unsigned int member_mask =
      kNumLanes == 32 ? 0xffffffff : (1 << kNumLanes) - 1;

  __device__ WarpReduce() : laneId(LaneId()) {}

  template <typename Func>
  __device__ __forceinline__ T reduce(T val, Func func) {
    for (int32_t offset = kNumLanes / 2; offset > 0; offset = offset >> 1) {
      T other = __shfl_down_sync(0xffffffff, val, offset);
      if (laneId + offset < kNumLanes) {
        val = func(val, other);
      }
    }
    return val;
  }
};

} // namespace facebook::velox::wave
