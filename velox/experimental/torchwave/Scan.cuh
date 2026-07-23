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

  // First warp reduces the warp-level results. Lanes past kNumWarps hold a
  // neutral T{}, but WarpReduce<T, kNumWarps> only combines lanes [0,
  // kNumWarps) (kNumWarps is a power of two), so that padding never enters the
  // result -- the reduction's identity therefore only matters at the caller's
  // out-of-range lane padding, not here.
  if (threadIdx.x < kWarpThreads) {
    T val = threadIdx.x < kNumWarps ? tempT[threadIdx.x] : T{};
    warpResult =
        facebook::velox::wave::WarpReduce<T, kNumWarps>().reduce(val, func);
  }
  return warpResult;
}

// Reduction operator policies for the generic tw_reduce* kernels. Each provides
// the neutral element (identity), used both to pad out-of-range lanes before
// the block reduce and to initialize the cross-iteration accumulator, and
// combine(), applied by the block/warp reduce. Sum pads with 0; max/min pad
// with the lowest/highest representable value so out-of-range lanes never win.
// Limits are written as bit-pattern intrinsics / literals rather than <cmath> /
// <cstdint> macros because NVRTC compiles this header without the C++ standard
// library headers available.
template <typename T>
struct ReduceLimits {
  // Only the specializations below are supported; instantiating with any other
  // type is a clear compile error rather than a cryptic missing-symbol link
  // failure on the undefined lowest()/highest() below.
  static_assert(sizeof(T) == 0, "ReduceLimits: unsupported reduction type");
  static __device__ T lowest();
  static __device__ T highest();
};
template <>
struct ReduceLimits<float> {
  static __device__ float lowest() {
    return -__int_as_float(0x7f800000); // -inf
  }
  static __device__ float highest() {
    return __int_as_float(0x7f800000); // +inf
  }
};
template <>
struct ReduceLimits<double> {
  static __device__ double lowest() {
    return -__longlong_as_double(0x7ff0000000000000LL); // -inf
  }
  static __device__ double highest() {
    return __longlong_as_double(0x7ff0000000000000LL); // +inf
  }
};
template <>
struct ReduceLimits<int32_t> {
  static __device__ int32_t lowest() {
    return -2147483647 - 1;
  }
  static __device__ int32_t highest() {
    return 2147483647;
  }
};
template <>
struct ReduceLimits<int64_t> {
  static __device__ int64_t lowest() {
    return -9223372036854775807LL - 1;
  }
  static __device__ int64_t highest() {
    return 9223372036854775807LL;
  }
};

template <typename T>
struct SumOp {
  static __device__ T identity() {
    return T(0);
  }
  static __device__ T combine(T a, T b) {
    return a + b;
  }
};
template <typename T>
struct MaxOp {
  static __device__ T identity() {
    return ReduceLimits<T>::lowest();
  }
  static __device__ T combine(T a, T b) {
    return a > b ? a : b;
  }
};
template <typename T>
struct MinOp {
  static __device__ T identity() {
    return ReduceLimits<T>::highest();
  }
  static __device__ T combine(T a, T b) {
    return a < b ? a : b;
  }
};

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
  if (size == 0) {
    // Empty input: the count loop below runs zero iterations, leaving this
    // block's per-block count slot uninitialized. add_sizes would then sum
    // garbage into the total (and reserve the final output to it). Write 0 so
    // the total is 0 and the output is reserved empty.
    if (threadIdx.x == 0) {
      out[block.blockInOp] = 0;
    }
    return;
  }
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
  if (in == nullptr) {
    // Empty/unproduced input (e.g. an isolated stage under --debug_single_ops,
    // or a degenerate empty scan): the running total is zero.
    if (threadIdx.x == 0 && output) {
      *output = T2(0);
    }
    return;
  }
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
  if (size == 0) {
    // Empty input: the element loop below runs no iterations, so no thread
    // writes the output length. Set it to zero so the output does not keep its
    // reserved (upper-bound) size.
    if (threadIdx.x == 0 && block.blockInOp == 0) {
      output->dims[0] = 0;
    }
    return;
  }
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
    // Honor the input's stride: a non-contiguous input (e.g. a select column
    // view) must be read through indexToOffset, not as flat storage.
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : TOut(0);
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
    // Honor the input's stride (non-contiguous select-column views).
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : TOut(0);
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
  if (in == nullptr || out == nullptr) {
    // Degenerate/isolated stage with no real input or output buffer.
    return;
  }
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    // Honor the input's stride (non-contiguous select-column views).
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : TOut(0);
    TOut base = (blockIdx == 0 || cnt == nullptr) ? TOut(0) : cnt[blockIdx - 1];
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
    // Honor the input's stride (non-contiguous select-column views), matching
    // the single-block cumsum.
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : TOut(0);
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

// Single-block exclusive prefix sum truncated to the input length (fb
// lengths_to_offsets with include_last_offset=False). Identical to
// exclusive_sum except the output has 'size' elements instead of size+1: out[0]
// = 0, out[i] = sum(in[0..i-1]); the trailing total (out[size]) is not written.
// Reuses the same block-wide inclusive scan.
template <int32_t kBlockSize, typename TIn, typename TOut, int32_t dim = 0>
__device__ void lengths_to_offsets(
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
  if (threadIdx.x == 0 && size > 0) {
    // Unlike exclusive_sum (output size + 1), this output has exactly 'size'
    // elements, so an empty input yields an empty output; guard the out[0]
    // write to avoid an out-of-bounds store on a zero-length output.
    out[0] = TOut(0);
  }
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    // Honor the input's stride (non-contiguous select-column views), matching
    // the single-block exclusive_sum.
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : TOut(0);
    if (threadIdx.x == 0) {
      val += counter;
    }
    TOut sum = facebook::velox::wave::inclusiveSum<TOut, kBlockSize>(
        val, nullptr, static_cast<TOut*>(temp));
    // Drop the final total: write out[i+1] only while i+1 stays in range.
    if (idx + 1 < size) {
      out[idx + 1] = sum;
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
}

// fb.offsets_to_lengths: per-segment lengths from a 1-D offsets tensor of N
// segments (N == offsets->dims[0]). lengths[i] = offsets[i+1] - offsets[i],
// with the last length extending to values->numEl (the total element count),
// matching the eager kernel (offsets_to_lengths_kernel in
// offsets_to_lengths_cuda.cu). Output shape and dtype match offsets.
// Embarrassingly parallel grid-stride loop, one thread per segment; the grid
// may be sized larger than N (kMax over inputs includes values), so the loop is
// bounded by N.
template <int32_t kBlockSize, typename T>
__device__ void offsets_to_lengths(
    Tensor* offsets,
    Tensor* values,
    Tensor* output,
    BlockInfo& block) {
  int64_t n = offsets->dims[0];
  T* off = storage<T>(offsets);
  T* out = storage<T>(output);
  int64_t total = values->numEl;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x;
       idx < static_cast<uint32_t>(n);
       idx += block.numBlocksInOp * blockDim.x) {
    int64_t start = static_cast<int64_t>(off[idx]);
    int64_t end = (static_cast<int64_t>(idx) + 1 < n)
        ? static_cast<int64_t>(off[idx + 1])
        : total;
    out[idx] = static_cast<T>(end - start);
  }
}

// fb.offsets_to_ranges: builds a fresh [N, 1, 2] int32 ranges tensor from a 1-D
// offsets tensor of N segments. Row i is (start, length) = (offsets[i],
// offsets[i+1] - offsets[i]); the last length extends to values->numEl,
// matching the eager kernel (offsets_to_ranges_kernel in
// offsets_to_ranges_cuda.cu). The output is contiguous, so segment i writes
// out[2*i] (start) and out[2*i+1] (length). Output is always int32 regardless
// of the offsets dtype; TOff is the offsets element type. One thread per
// segment, grid-stride, bounded by N.
template <int32_t kBlockSize, typename TOff>
__device__ void offsets_to_ranges(
    Tensor* offsets,
    Tensor* values,
    Tensor* output,
    BlockInfo& block) {
  int64_t n = offsets->dims[0];
  TOff* off = storage<TOff>(offsets);
  int32_t* out = storage<int32_t>(output);
  int64_t total = values->numEl;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x;
       idx < static_cast<uint32_t>(n);
       idx += block.numBlocksInOp * blockDim.x) {
    int64_t start = static_cast<int64_t>(off[idx]);
    int64_t end = (static_cast<int64_t>(idx) + 1 < n)
        ? static_cast<int64_t>(off[idx + 1])
        : total;
    out[2 * idx] = static_cast<int32_t>(start);
    out[2 * idx + 1] = static_cast<int32_t>(end - start);
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
  if (in == nullptr || out == nullptr) {
    // Degenerate/isolated stage with no real input or output buffer.
    return;
  }
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    // Honor the input's stride (non-contiguous select-column views), matching
    // cumsum_final. exclusive_sum_head reads through complexIdx, so a flat read
    // here would sum the wrong storage for a strided input.
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : TOut(0);
    TOut base = (blockIdx == 0 || cnt == nullptr) ? TOut(0) : cnt[blockIdx - 1];
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

// repeat_interleave index-generating final stage: for each segment i, writes
// the segment index i into output[prefix[i-1]..prefix[i]). Mirrors
// repeat_interleave_final's position->segment mapping but emits the index i
// instead of a gathered self[i]. The repeats tensor is passed only for its
// element count (number of segments) and dtype; eager
// repeat_interleave(repeats) returns the same dtype as repeats, so T is the
// repeats element type.
template <int32_t kBlockSize, typename T>
__device__ void repeat_interleave_index_final(
    Tensor* repeats,
    Tensor* prefix,
    int32_t* /*total*/,
    Tensor* output,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*repeats);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  int32_t* pfx = storage<int32_t>(prefix);
  T* out = storage<T>(output);
  auto outputSize = numEl(*output);
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    if (idx < size) {
      int32_t start = idx == 0 ? 0 : pfx[idx - 1];
      int32_t end = pfx[idx];
      T val = static_cast<T>(idx);
      for (int32_t j = start; j < end; ++j) {
        assert(static_cast<uint32_t>(j) < outputSize);
        out[j] = val;
      }
    }
  }
}

// Multi-block reduction, stage 1: per-block reduction.
// Each block reduces its chunk of the input and writes one TOut value. Op
// selects the reduction (sum/max/min); out-of-range lanes pad with
// Op::identity.
template <int32_t kBlockSize, typename TIn, typename TOut, typename Op>
__device__ void tw_reduce_head(
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
    // Honor the input's stride so a non-contiguous view (e.g. a select-column
    // reduced by max/min) reads the right elements, matching exclusive_sum.
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : Op::identity();
    auto res = reduce<kBlockSize, TOut>(
        val, [](TOut a, TOut b) { return Op::combine(a, b); }, temp);
    if (threadIdx.x == 0) {
      out[blockIdx] = res;
      blockIdx += block.numBlocksInOp;
    }
  }
}

// Single-block reduction over a tensor. Reduces all elements of input into a
// scalar stored in output. counterT accumulates the intermediate result across
// block-wide iterations. TIn is the input element type, TOut the
// accumulation/output type; Op selects the reduction.
// Calling convention: (input, output, shared..., blockInfo).
template <int32_t kBlockSize, typename TIn, typename TOut, typename Op>
__device__ void tw_reduce_tensor(
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
    counterT = Op::identity();
  }
  __syncthreads();
  TIn* in = storage<TIn>(input);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    // Honor the input's stride so a non-contiguous view (e.g. a select-column
    // reduced by max/min) reads the right elements, matching exclusive_sum.
    TOut val = (idx < size)
        ? static_cast<TOut>(in[complexIdx(input->contiguous, input, idx)])
        : Op::identity();
    auto blockRes = reduce<kBlockSize, TOut>(
        val, [](TOut a, TOut b) { return Op::combine(a, b); }, temp);
    if (threadIdx.x == 0) {
      counterT = Op::combine(counterT, blockRes);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *storage<TOut>(output) = counterT;
  }
}

// Register-input overload: called per-lane from the caller's loop.
// Accumulates the block-reduced result across iterations. On the last iteration
// (idx at the last stride-aligned position), lane 0 writes the total to output.
// Out-of-range lanes (idx >= size) pad with Op::identity so a fused
// expression's stale/garbage value never enters the reduction.
template <int32_t kBlockSize, typename TIn, typename TOut, typename Op>
__device__ void tw_reduce(
    TIn in,
    Tensor* output,
    void* temp,
    TOut& counterT,
    uint32_t idx,
    uint32_t size,
    BlockInfo& /*block*/) {
  if (idx == 0) {
    counterT = Op::identity();
  }
  __syncthreads();
  TOut val = (idx < size) ? static_cast<TOut>(in) : Op::identity();
  auto blockRes = reduce<kBlockSize, TOut>(
      val, [](TOut a, TOut b) { return Op::combine(a, b); }, temp);
  if (threadIdx.x == 0) {
    counterT = Op::combine(counterT, blockRes);
  }
  __syncthreads();
  if (threadIdx.x == 0 && idx + kBlockSize >= size) {
    *storage<TOut>(output) = counterT;
  }
}

// Named entry points the codegen references by deviceFunc name. Each binds the
// generic tw_reduce* to a reduction Op; the codegen appends <kBlockSize, TIn,
// TOut> so the Op (which depends on TOut) is baked in here, not
// codegen-supplied.
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  tw_reduce_head<kBlockSize, TIn, TOut, SumOp<TOut>>(
      input, output, temp, size, rounded, block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_max_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  tw_reduce_head<kBlockSize, TIn, TOut, MaxOp<TOut>>(
      input, output, temp, size, rounded, block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_min_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  tw_reduce_head<kBlockSize, TIn, TOut, MinOp<TOut>>(
      input, output, temp, size, rounded, block);
}

template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum_tensor(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counterT,
    BlockInfo& block) {
  tw_reduce_tensor<kBlockSize, TIn, TOut, SumOp<TOut>>(
      input, output, temp, size, rounded, counterT, block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_max_tensor(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counterT,
    BlockInfo& block) {
  tw_reduce_tensor<kBlockSize, TIn, TOut, MaxOp<TOut>>(
      input, output, temp, size, rounded, counterT, block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_min_tensor(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    TOut& counterT,
    BlockInfo& block) {
  tw_reduce_tensor<kBlockSize, TIn, TOut, MinOp<TOut>>(
      input, output, temp, size, rounded, counterT, block);
}

template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_sum(
    TIn in,
    Tensor* output,
    void* temp,
    TOut& counterT,
    uint32_t idx,
    uint32_t size,
    BlockInfo& block) {
  tw_reduce<kBlockSize, TIn, TOut, SumOp<TOut>>(
      in, output, temp, counterT, idx, size, block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_max(
    TIn in,
    Tensor* output,
    void* temp,
    TOut& counterT,
    uint32_t idx,
    uint32_t size,
    BlockInfo& block) {
  tw_reduce<kBlockSize, TIn, TOut, MaxOp<TOut>>(
      in, output, temp, counterT, idx, size, block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_min(
    TIn in,
    Tensor* output,
    void* temp,
    TOut& counterT,
    uint32_t idx,
    uint32_t size,
    BlockInfo& block) {
  tw_reduce<kBlockSize, TIn, TOut, MinOp<TOut>>(
      in, output, temp, counterT, idx, size, block);
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

// Single-block nonzero for 1D input. Returns indices where input != 0.
template <int32_t kBlockSize, typename T>
__device__ void nonzero1d(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& counter,
    BlockInfo& block) {
  int64_t* out = storage<int64_t>(output);
  T* in = storage<T>(input);
  if (threadIdx.x == 0) {
    size = numEl(*input);
    counter = 0;
  }
  __syncthreads();
  auto rounded = roundUpPwr2(size, kBlockSize);
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    bool flag = (idx < size) ? (in[idx] != T(0)) : false;
    int32_t val = static_cast<int32_t>(flag);
    if (threadIdx.x == 0) {
      val += counter;
    }
    int32_t sum = facebook::velox::wave::inclusiveSum<int32_t, kBlockSize>(
        val, nullptr, static_cast<int32_t*>(temp));
    if (flag) {
      out[sum - 1] = static_cast<int64_t>(idx);
    }
    if (threadIdx.x == blockDim.x - 1) {
      counter = sum;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output->dims[0] = counter;
    output->dims[1] = 1;
  }
}

// Multi-block nonzero head: counts non-zero elements per block.
template <int32_t kBlockSize, typename T>
__device__ void nonzero1d_head(
    Tensor* input,
    Tensor* output,
    void* temp,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  T* in = storage<T>(input);
  int32_t* out = storage<int32_t>(output);
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    uint32_t flag = (idx < size) ? static_cast<uint32_t>(in[idx] != T(0)) : 0;
    auto count = reduce<kBlockSize, uint32_t>(
        flag, [](uint32_t a, uint32_t b) { return a + b; }, temp);
    if (threadIdx.x == 0) {
      out[blockIdx] = count;
      blockIdx += block.numBlocksInOp;
    }
  }
}

// Multi-block nonzero final: scatters indices using prefix-summed counts.
template <int32_t kBlockSize, typename T>
__device__ void nonzero1d_final(
    Tensor* input,
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
  int32_t* cnt = storage<int32_t>(counts);
  int64_t* out = storage<int64_t>(output);
  uint32_t blockIdx = block.blockInOp;
  for (uint32_t idx = block.blockInOp * blockDim.x + threadIdx.x; idx < rounded;
       idx += block.numBlocksInOp * blockDim.x) {
    bool flag = (idx < size) ? (in[idx] != T(0)) : false;
    auto base = blockIdx == 0 ? 0 : cnt[blockIdx - 1];
    int32_t val = static_cast<int32_t>(flag);
    if (threadIdx.x == 0) {
      val += base;
    }
    int32_t sum = facebook::velox::wave::inclusiveSum<int32_t, kBlockSize>(
        val, nullptr, static_cast<int32_t*>(temp));
    if (flag) {
      out[sum - 1] = static_cast<int64_t>(idx);
    }
    if (idx == size - 1) {
      output->dims[0] = sum;
      output->dims[1] = 1;
    }
    blockIdx += block.numBlocksInOp;
  }
}

// CG nonzero: 3-pass nonzero in a single cooperative kernel.
template <int32_t kBlockSize, typename T>
__device__ void nonzero1d_cg(
    Tensor* input,
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
  nonzero1d_head<kBlockSize, T>(input, counts, temp, size, rounded, block);
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
  nonzero1d_final<kBlockSize, T>(
      input,
      counts,
      static_cast<int32_t*>(nullptr),
      output,
      temp,
      size,
      rounded,
      block);
  opBarrier(block, bar2);
}

// Cooperative-grid reduction: stage 1 per-block partials, an inter-block
// barrier, then block 0 reduces the partials -- the multi-kernel head+final
// fused into one cooperative launch. Op selects the reduction.
template <int32_t kBlockSize, typename TIn, typename TOut, typename Op>
__device__ void tw_reduce_cg(
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
  tw_reduce_head<kBlockSize, TIn, TOut, Op>(
      input, partials, temp, size, rounded, block);
  opBarrier(block, bar0);
  if (block.blockInOp == 0) {
    tw_reduce_tensor<kBlockSize, TOut, TOut, Op>(
        partials, output, temp, size, rounded, counterT, block);
  }
  opBarrier(block, bar1);
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
  tw_reduce_cg<kBlockSize, TIn, TOut, SumOp<TOut>>(
      input,
      output,
      partials,
      temp,
      size,
      rounded,
      counterT,
      bar0,
      bar1,
      block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_max_cg(
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
  tw_reduce_cg<kBlockSize, TIn, TOut, MaxOp<TOut>>(
      input,
      output,
      partials,
      temp,
      size,
      rounded,
      counterT,
      bar0,
      bar1,
      block);
}
template <int32_t kBlockSize, typename TIn, typename TOut>
__device__ void tw_min_cg(
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
  tw_reduce_cg<kBlockSize, TIn, TOut, MinOp<TOut>>(
      input,
      output,
      partials,
      temp,
      size,
      rounded,
      counterT,
      bar0,
      bar1,
      block);
}

// --- bincount ---

// atomicAdd dispatch for bincount: counts accumulate as int64 (no CUDA int64
// atomicAdd, so route through unsigned long long -- counts are non-negative and
// fit), weighted sums as their float/double dtype.
__device__ inline void binAtomicAdd(int64_t* p, int64_t v) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(p),
      static_cast<unsigned long long>(v));
}
__device__ inline void binAtomicAdd(int32_t* p, int32_t v) {
  atomicAdd(p, v);
}
__device__ inline void binAtomicAdd(float* p, float v) {
  atomicAdd(p, v);
}
__device__ inline void binAtomicAdd(double* p, double v) {
  atomicAdd(p, v);
}

// aten.bincount head: max over the (non-negative integer) input. The result
// sizes the output on the host: output length = max(maxval + 1, minlength). An
// empty input reduces to the identity (INT_MIN), so the host falls back to
// minlength. Single block; TIdx is the input dtype.
template <int32_t kBlockSize, typename TIdx>
__device__ void bincount_head(
    Tensor* input,
    int32_t* maxOut,
    Int32X32& warpSums,
    uint32_t& size,
    uint32_t& rounded,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
    rounded = roundUpPwr2(size, kBlockSize);
  }
  __syncthreads();
  TIdx* in = storage<TIdx>(input);
  int32_t localMax = ReduceLimits<int32_t>::lowest();
  for (uint32_t idx = threadIdx.x; idx < rounded; idx += blockDim.x) {
    if (idx < size) {
      int32_t v =
          static_cast<int32_t>(in[complexIdx(input->contiguous, input, idx)]);
      localMax = v > localMax ? v : localMax;
    }
  }
  int32_t blockMax = reduce<kBlockSize, int32_t>(
      localMax, [](int32_t a, int32_t b) { return a > b ? a : b; }, warpSums);
  if (threadIdx.x == 0 && maxOut) {
    *maxOut = blockMax;
  }
}

// aten.bincount final (no weights): clear the output bins, sync, then scatter
// one atomic add of 1 per input element. The output is already sized by the
// host from the head's max; out-of-range indices are skipped. 'maxval' is
// passed only to order this kernel after the head. TIdx is the index dtype,
// TOut the output dtype (int64 counts). Single block.
template <int32_t kBlockSize, typename TIdx, typename TOut>
__device__ void bincount_final(
    Tensor* input,
    int32_t* /*maxval*/,
    Tensor* output,
    uint32_t& size,
    BlockInfo& /*block*/) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
  }
  __syncthreads();
  TIdx* in = storage<TIdx>(input);
  TOut* out = storage<TOut>(output);
  auto outSize = numEl(*output);
  for (uint32_t i = threadIdx.x; i < outSize; i += blockDim.x) {
    out[i] = TOut(0);
  }
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < size; j += blockDim.x) {
    int64_t idx =
        static_cast<int64_t>(in[complexIdx(input->contiguous, input, j)]);
    if (idx >= 0 && idx < static_cast<int64_t>(outSize)) {
      binAtomicAdd(&out[idx], TOut(1));
    }
  }
}

// Cooperative-grid variant: each block clears its slice of the output, an
// inter-block barrier makes the clears visible, all blocks scatter their slice
// of the input via atomic adds, then a final barrier makes the counts visible.
template <int32_t kBlockSize, typename TIdx, typename TOut>
__device__ void bincount_final_cg(
    Tensor* input,
    int32_t* /*maxval*/,
    Tensor* output,
    uint32_t& size,
    int32_t bar0,
    int32_t bar1,
    BlockInfo& block) {
  if (threadIdx.x == 0) {
    size = numEl(*input);
  }
  __syncthreads();
  TIdx* in = storage<TIdx>(input);
  TOut* out = storage<TOut>(output);
  auto outSize = numEl(*output);
  for (uint32_t i = block.blockInOp * blockDim.x + threadIdx.x; i < outSize;
       i += block.numBlocksInOp * blockDim.x) {
    out[i] = TOut(0);
  }
  opBarrier(block, bar0);
  for (uint32_t j = block.blockInOp * blockDim.x + threadIdx.x; j < size;
       j += block.numBlocksInOp * blockDim.x) {
    int64_t idx =
        static_cast<int64_t>(in[complexIdx(input->contiguous, input, j)]);
    if (idx >= 0 && idx < static_cast<int64_t>(outSize)) {
      binAtomicAdd(&out[idx], TOut(1));
    }
  }
  opBarrier(block, bar1);
}

} // namespace torch::wave
