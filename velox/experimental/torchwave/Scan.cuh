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

#include "velox/experimental/wave/common/Scan.cuh"

namespace torch::wave {

template <int32_t kBlockSize, typename T, typename Func>
__device__ T reduce(T input, Func func, T* temp) {
  constexpr int32_t kNumWarps = kBlockSize / kWarpThreads;

  // Warp-level reduce.
  T warpResult = facebook::velox::wave::WarpReduce<T>().reduce(input, func);

  // Lane 0 of each warp stores its result in temp.
  if (threadIdx.x % kWarpThreads == 0) {
    temp[threadIdx.x / kWarpThreads] = warpResult;
  }
  __syncthreads();

  // First warp reduces the warp-level results.
  if (threadIdx.x < kWarpThreads) {
    T val = threadIdx.x < kNumWarps ? temp[threadIdx.x] : T{};
    warpResult =
        facebook::velox::wave::WarpReduce<T, kNumWarps>().reduce(val, func);
  }
  return warpResult;
}

template <int32_t kBlockSize, typename T>
__device__ void masked_select(
    T input,
    bool flag,
    Tensor* output,
    Int32X32& temp,
    uint32_t& counter,
    uint32_t idx,
    uint32_t size,
    BlockInfo& block) {
  T* out = storage<T>(output);
  if (idx == 0) {
    // there is a syncthreads in exclusiveSum before use of counter.
    counter = 0;
  }
  if (idx >= size) {
    flag = false;
  }
  int32_t sum = facebook::velox::wave::inclusiveSum<uint32_t, kBlockSize>(
      threadIdx.x == 0 ? static_cast<uint32_t>(flag) + counter
                       : static_cast<uint32_t>(flag),
      &counter,
      &temp[0]);
  if (flag) {
    out[sum - 1] = input;
  }
  if (idx == size - 1) {
    output->dims[0] = sum;
  }
  __syncthreads();
}

template <int32_t kBlockSize, typename T>
__device__ void masked_select(
    Tensor* input,
    Tensor* mask,
    Tensor* output,
    Int32X32& temp,
    uint32_t& counter,
    uint32_t idx,
    uint32_t size,
    BlockInfo& block) {
  bool* flags = storage<bool>(mask);
  T* in = storage<T>(input);
  masked_select<kBlockSize, T>(
      idx < size ? in[idx] : T(),
      idx < size ? flags[idx] : false,
      output,
      temp,
      counter,
      idx,
      size,
      block);
}

template <int32_t kBlockSize, typename T>
__device__ void masked_select_head(
    Tensor* input,
    Tensor* mask,
    Tensor* output,
    Int32X32& temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  bool* flags = storage<bool>(mask);
  int32_t* out = storage<int32_t>(output);
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    uint32_t flag = (idx < size) ? static_cast<uint32_t>(flags[idx]) : 0;
    auto count = reduce<kBlockSize, uint32_t>(
        flag, [](uint32_t a, uint32_t b) { return a + b; }, &temp[0]);
    if (threadIdx.x == 0) {
      out[blockIdx] = count;
      blockIdx += block.numBlocksInOp;
    }
  }
}

template <int32_t kBlockSize>
__device__ void add_sizes(
    Tensor* input,
    int32_t* output,
    Int32X32& temp,
    uint32_t& size,
    uint32_t& rounded,
    uint32_t& counter,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    counter = 0;
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  int32_t* in = storage<int32_t>(input);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    int32_t val = (idx < size) ? in[idx] : 0;
    if (threadIdx.x == 0) {
      val += counter;
    }
    int32_t sum = facebook::velox::wave::inclusiveSum<int32_t, kBlockSize>(
        val, nullptr, reinterpret_cast<int32_t*>(&temp[0]));
    if (idx < size) {
      in[idx] = sum;
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = counter;
  }
}

template <int32_t kBlockSize, typename T>
__device__ void masked_select_final(
    Tensor* input,
    Tensor* mask,
    Tensor* counts,
    int32_t* total,
    Tensor* output,
    Int32X32& temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  T* in = storage<T>(input);
  bool* flags = storage<bool>(mask);
  int32_t* cnt = storage<int32_t>(counts);
  T* out = storage<T>(output);
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    bool flag = (idx < size) ? flags[idx] : false;
    int32_t base = blockIdx == 0 ? 0 : cnt[blockIdx - 1];
    int32_t val = static_cast<int32_t>(flag);
    if (threadIdx.x == 0) {
      val += base;
    }
    int32_t sum = facebook::velox::wave::inclusiveSum<int32_t, kBlockSize>(
        val, nullptr, reinterpret_cast<int32_t*>(&temp[0]));
    if (flag) {
      out[sum - 1] = in[idx];
    }
    if (idx == size - 1) {
      output->dims[0] = sum;
    }
    blockIdx += block.numBlocksInOp;
  }
}

} // namespace torch::wave
