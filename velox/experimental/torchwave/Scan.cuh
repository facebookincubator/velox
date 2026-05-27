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

// GPU reduction and scan algorithms. Operations like cumsum, exclusive_sum,
// masked_select, and sum combine adjacent array elements via three strategies:
//
// Single-block: one thread block processes the entire array sequentially,
// carrying an intermediate sum between block-wide iterations. Low overhead
// but limited parallelism.
//
// Multi-kernel: stage 1 computes per-block partial results in parallel,
// stage 2 prefix-sums the partials to produce per-block starting offsets,
// stage 3 produces final output using those offsets. Higher throughput than
// single-block but more overhead per element.
//
// Cooperative-grid: same three-stage logic as multi-kernel but uses
// device-side inter-block barriers instead of kernel boundaries, avoiding
// the launch overhead between stages.

namespace torch::wave {

template <int32_t kBlockSize, typename T, typename Func>
__device__ T reduce(T input, Func func, void* temp) {
  constexpr int32_t kNumWarps = kBlockSize / kWarpThreads;

  auto* tempT = static_cast<T*>(temp);
  // Warp-level reduce.
  T warpResult = facebook::velox::wave::WarpReduce<T>().reduce(input, func);

  // Lane 0 of each warp stores its result in temp.
  if (threadIdx.x % kWarpThreads == 0) {
    tempT[threadIdx.x / kWarpThreads] = warpResult;
  }
  __syncthreads();

  // First warp reduces the warp-level results.
  if (threadIdx.x < kWarpThreads) {
    T val = threadIdx.x < kNumWarps ? tempT[threadIdx.x] : T{};
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
    void* temp,
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
      static_cast<uint32_t*>(temp));
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
    void* temp,
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
    void* temp,
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
        flag, [](uint32_t a, uint32_t b) { return a + b; }, temp);
    if (threadIdx.x == 0) {
      out[blockIdx] = count;
      blockIdx += block.numBlocksInOp;
    }
  }
}

template <int32_t kBlockSize, typename T = uint32_t, typename T2>
__device__ void add_sizes(
    Tensor* input,
    T2* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    T& counter,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    counter = 0;
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  T* in = storage<T>(input);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    T val = (idx < size) ? in[idx] : T(0);
    if (threadIdx.x == 0) {
      val += counter;
    }
    T sum = facebook::velox::wave::inclusiveSum<T, kBlockSize>(
        val, nullptr, static_cast<T*>(temp));
    if (idx < size) {
      in[idx] = sum;
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && output) {
    *output = counter;
  }
}

// Overload without output argument for use as middle stage of cumsum.
template <int32_t kBlockSize, typename T = uint32_t>
__device__ void add_sizes(
    Tensor* input,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    T& counter,
    BlockInfo& block) {
  add_sizes<kBlockSize, T>(
      input, static_cast<T*>(nullptr), temp, size, rounded, counter, block);
}

template <int32_t kBlockSize, typename T>
__device__ void masked_select_final(
    Tensor* input,
    Tensor* mask,
    Tensor* counts,
    int32_t* total,
    Tensor* output,
    void* temp,
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
    auto base = blockIdx == 0 ? 0 : cnt[blockIdx - 1];
    int32_t val = static_cast<int32_t>(flag);
    if (threadIdx.x == 0) {
      val += base;
    }
    int32_t sum = facebook::velox::wave::inclusiveSum<int32_t, kBlockSize>(
        val, nullptr, static_cast<int32_t*>(temp));
    if (flag) {
      out[sum - 1] = in[idx];
    }
    if (idx == size - 1) {
      output->dims[0] = sum;
    }
    blockIdx += block.numBlocksInOp;
  }
}

// Single-block cumulative sum. Writes inclusive prefix sum of
// input into output. Iterates over the full length of the input.
// TIn is the input element type, TOut is the accumulation/output type.
// Calling convention: (input, output, dim_attr, shared..., blockInfo).
template <int32_t kBlockSize, typename TIn, typename TOut, int32_t dim = 0>
__device__ void cumsum(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counter,
    BlockInfo& /*block*/) {
  static_assert(dim == 0, "Only dim 0 is supported");
  if (threadIdx.x == 0) {
    size = numEl(*input);
    counter = 0;
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  TOut* out = storage<TOut>(output);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    if (threadIdx.x == 0) {
      val += counter;
    }
    TOut sum = facebook::velox::wave::inclusiveSum<TOut, kBlockSize>(
        val, nullptr, static_cast<TOut*>(temp));
    if (idx < size) {
      out[idx] = sum;
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
}

// Three-stage cumsum, stage 1: compute per-block sums.
// TIn is the input element type, TOut is the accumulation type.
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void cumsum_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  TOut* out = storage<TOut>(output);
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    auto sum = reduce<kBlockSize, TOut>(
        val, [](TOut a, TOut b) { return a + b; }, temp);
    if (threadIdx.x == 0) {
      out[blockIdx] = sum;
      blockIdx += block.numBlocksInOp;
    }
  }
}

// Three-stage cumsum, stage 3: compute final inclusive prefix sum using
// per-block prefix sums from add_sizes.
// TIn is the input element type, TOut is the accumulation/output type.
// Calling convention: (input, counts, output, dim_attr, shared..., blockInfo).
template <int32_t kBlockSize, typename TIn, typename TOut, int32_t dim = 0>
__device__ void cumsum_final(
    Tensor* input,
    Tensor* counts,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  static_assert(dim == 0, "Only dim 0 is supported");
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  TOut* cnt = storage<TOut>(counts);
  TOut* out = storage<TOut>(output);
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    TOut base = blockIdx == 0 ? TOut(0) : cnt[blockIdx - 1];
    if (threadIdx.x == 0) {
      val += base;
    }
    TOut sum = facebook::velox::wave::inclusiveSum<TOut, kBlockSize>(
        val, nullptr, static_cast<TOut*>(temp));
    if (idx < size) {
      out[idx] = sum;
    }
    blockIdx += block.numBlocksInOp;
  }
}

// Single-block exclusive prefix sum. Output has size+1 elements.
// out[0] = 0, out[i+1] = sum(in[0..i]).
template <int32_t kBlockSize, typename TIn, typename TOut, int32_t dim = 0>
__device__ void exclusive_sum(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counter,
    BlockInfo& /*block*/) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    counter = 0;
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  TOut* out = storage<TOut>(output);
  if (threadIdx.x == 0) {
    out[0] = TOut(0);
  }
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    if (threadIdx.x == 0) {
      val += counter;
    }
    TOut sum = facebook::velox::wave::inclusiveSum<TOut, kBlockSize>(
        val, nullptr, static_cast<TOut*>(temp));
    if (idx < size) {
      out[idx + 1] = sum;
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
}

// Multi-block exclusive sum, stage 1: per-block sums (same as cumsum_head).
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void exclusive_sum_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  cumsum_head<kBlockSize, TIn, TOut>(input, output, temp, size, rounded, block);
}

// Multi-block exclusive sum, stage 3: final exclusive prefix sum.
// Output has size+1 elements: out[0]=0, out[i+1]=sum(in[0..i]).
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void exclusive_sum_final(
    Tensor* input,
    Tensor* counts,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  TOut* cnt = storage<TOut>(counts);
  TOut* out = storage<TOut>(output);
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    TOut base = blockIdx == 0 ? TOut(0) : cnt[blockIdx - 1];
    if (threadIdx.x == 0) {
      val += base;
      out[idx] = base;
    }
    TOut sum = facebook::velox::wave::inclusiveSum<TOut, kBlockSize>(
        val, nullptr, static_cast<TOut*>(temp));
    if (idx < size) {
      out[idx + 1] = sum;
    }
    blockIdx += block.numBlocksInOp;
  }
}

// repeat_interleave head stage: compute inclusive prefix sums of repeats into
// a separate int32 prefix tensor and write the total to *total.
// TRepeats is the element type of the repeats tensor (int32_t or int64_t).
template <int32_t kBlockSize, typename TRepeats>
__device__ void repeat_interleave_head(
    Tensor* repeats,
    Tensor* prefix,
    int32_t* total,
    Int32X32& warpSums,
    uint32_t& size,
    uint32_t& rounded,
    uint32_t& counter,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*repeats);
    counter = 0;
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TRepeats* in = storage<TRepeats>(repeats);
  int32_t* out = storage<int32_t>(prefix);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    uint32_t val = (idx < size) ? static_cast<uint32_t>(in[idx]) : 0;
    if (threadIdx.x == 0) {
      val += counter;
    }
    uint32_t sum = facebook::velox::wave::inclusiveSum<uint32_t, kBlockSize>(
        val, nullptr, reinterpret_cast<uint32_t*>(&warpSums));
    if (idx < size) {
      out[idx] = static_cast<int32_t>(sum);
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && total) {
    *total = static_cast<int32_t>(counter);
  }
}

// repeat_interleave final stage: for each element i in self, writes self[i]
// into output[prefix[i-1]..prefix[i]). prefix is the inclusive prefix sum of
// the repeats tensor. T is the element type of input and output.
template <int32_t kBlockSize, typename T>
__device__ void repeat_interleave_final(
    Tensor* input,
    Tensor* prefix,
    int32_t* /*total*/,
    Tensor* output,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  T* in = storage<T>(input);
  int32_t* pfx = storage<int32_t>(prefix);
  T* out = storage<T>(output);
  auto outputSize = numEl(*output);
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    if (idx < size) {
      int32_t start = idx == 0 ? 0 : pfx[idx - 1];
      int32_t end = pfx[idx];
      T val = in[idx];
      for (int32_t j = start; j < end; ++j) {
        assert(static_cast<uint32_t>(j) < outputSize);
        out[j] = val;
      }
    }
  }
}

// Multi-block variant of repeat_interleave_final with an op barrier at the
// end for cooperative/multi-kernel grids.
template <int32_t kBlockSize, typename T>
__device__ void repeat_interleave_final_cg(
    Tensor* input,
    Tensor* prefix,
    int32_t* total,
    Tensor* output,
    uint32_t& size,
    uint32_t& rounded,
    int32_t bar0,
    BlockInfo& block) {
  repeat_interleave_final<kBlockSize, T>(
      input, prefix, total, output, size, rounded, block);
  opBarrier(block, bar0);
}

// Multi-block sum reduction, stage 1: per-block reduction.
// Each block reduces its chunk of the input and writes one TOut value.
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  TOut* out = storage<TOut>(output);
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    auto sum = reduce<kBlockSize, TOut>(
        val, [](TOut a, TOut b) { return a + b; }, temp);
    if (threadIdx.x == 0) {
      out[blockIdx] = sum;
      blockIdx += block.numBlocksInOp;
    }
  }
}

// Single-block sum reduction over a tensor. Reduces all elements of input
// into a scalar stored in output. counterT accumulates intermediate sums
// across iterations. TIn is the input element type, TOut is the
// accumulation/output type.
// Calling convention: (input, output, shared..., blockInfo).
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum_tensor(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counterT,
    BlockInfo& /*block*/) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
    counterT = TOut(0);
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    TOut val = (idx < size) ? static_cast<TOut>(in[idx]) : TOut(0);
    auto blockSum = reduce<kBlockSize, TOut>(
        val, [](TOut a, TOut b) { return a + b; }, temp);
    if (threadIdx.x == 0) {
      counterT += blockSum;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *storage<TOut>(output) = counterT;
  }
}

// Register-input overload: called per-lane from the caller's loop.
// Accumulates block-reduced sums across iterations. On the last iteration
// (idx at the last stride-aligned position), lane 0 writes the total to output.
// TIn is the input element type, TOut is the accumulation/output type.
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum(
    TIn in,
    Tensor* output,
    void* temp,
    TOut& counterT,
    uint32_t idx,
    uint32_t size,
    BlockInfo& /*block*/) {
  if (idx == 0) {
    counterT = TOut(0);
  }
  __syncthreads();
  TOut val = static_cast<TOut>(in);
  auto blockSum =
      reduce<kBlockSize, TOut>(val, [](TOut a, TOut b) { return a + b; }, temp);
  if (threadIdx.x == 0) {
    counterT += blockSum;
  }
  __syncthreads();
  if (threadIdx.x == 0 && idx + kBlockSize >= size) {
    *storage<TOut>(output) = counterT;
  }
}

// --- Cooperative grid variants ---

template <int32_t kBlockSize, typename TIn, typename TOut, int32_t dim = 0>
__device__ void cumsum_cg(
    Tensor* input,
    Tensor* output,
    Tensor* counts,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counter,
    int32_t bar0,
    int32_t bar1,
    BlockInfo& block) {
  cumsum_head<kBlockSize, TIn, TOut>(input, counts, temp, size, rounded, block);
  opBarrier(block, bar0);
  if (block.blockInOp == 0) {
    add_sizes<kBlockSize, TOut>(counts, temp, size, rounded, counter, block);
  }
  opBarrier(block, bar1);
  cumsum_final<kBlockSize, TIn, TOut, dim>(
      input, counts, output, temp, size, rounded, block);
}

template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void exclusive_sum_cg(
    Tensor* input,
    Tensor* output,
    Tensor* counts,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counter,
    int32_t bar0,
    int32_t bar1,
    BlockInfo& block) {
  exclusive_sum_head<kBlockSize, TIn, TOut>(
      input, counts, temp, size, rounded, block);
  opBarrier(block, bar0);
  if (block.blockInOp == 0) {
    add_sizes<kBlockSize, TOut>(counts, temp, size, rounded, counter, block);
  }
  opBarrier(block, bar1);
  exclusive_sum_final<kBlockSize, TIn, TOut>(
      input, counts, output, temp, size, rounded, block);
}

template <int32_t kBlockSize, typename T>
__device__ void masked_select_cg(
    Tensor* input,
    Tensor* mask,
    Tensor* output,
    Tensor* counts,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    uint32_t& counter,
    int32_t bar0,
    int32_t bar1,
    int32_t bar2,
    BlockInfo& block) {
  masked_select_head<kBlockSize, T>(
      input, mask, counts, temp, size, rounded, block);
  opBarrier(block, bar0);
  if (block.blockInOp == 0) {
    add_sizes<kBlockSize>(
        counts,
        static_cast<int32_t*>(nullptr),
        temp,
        size,
        rounded,
        counter,
        block);
  }
  opBarrier(block, bar1);
  masked_select_final<kBlockSize, T>(
      input,
      mask,
      counts,
      static_cast<int32_t*>(nullptr),
      output,
      temp,
      size,
      rounded,
      block);
  opBarrier(block, bar2);
}

template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum_cg(
    Tensor* input,
    Tensor* output,
    Tensor* partials,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counterT,
    int32_t bar0,
    int32_t bar1,
    BlockInfo& block) {
  tw_sum_head<kBlockSize, TIn, TOut>(
      input, partials, temp, size, rounded, block);
  opBarrier(block, bar0);
  if (block.blockInOp == 0) {
    tw_sum_tensor<kBlockSize, TOut, TOut>(
        partials, output, temp, size, rounded, counterT, block);
  }
  opBarrier(block, bar1);
}

} // namespace torch::wave
