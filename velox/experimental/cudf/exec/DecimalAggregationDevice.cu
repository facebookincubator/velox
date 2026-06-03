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

#include "velox/experimental/cudf/exec/DecimalAggregationDevice.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_for.cuh>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

// Mirrors the CPU LongDecimalWithOverflowState layout so serialized SUM state
// is interchangeable between CPU and GPU aggregation.
// TODO: Track int128 overflow as the CPU does (DecimalUtil::addWithOverflow);
// the `overflow` field is reserved for that and is always 0 until then.
struct DecimalSumState {
  int64_t count; // count of non-null input rows aggregated
  int64_t overflow; // net int128 carries (CPU parity); always 0 on GPU for now
  uint64_t lower; // lower 64 bits of the decimal sum
  int64_t upper; // upper 64 bits of the decimal sum (signed)
};

static_assert(sizeof(DecimalSumState) == detail::kDecimalSumStateSize);

__device__ __forceinline__ void
splitToWords(int64_t value, int64_t& upper, uint64_t& lower) {
  lower = static_cast<uint64_t>(value);
  upper = value < 0 ? -1 : 0;
}

__device__ __forceinline__ void
splitToWords(__int128_t value, int64_t& upper, uint64_t& lower) {
  lower = static_cast<uint64_t>(value);
  upper = static_cast<int64_t>(value >> 64);
}

template <typename OffsetT>
struct FillOffsetsFunctor {
  cuda::std::span<OffsetT> offsets;

  __device__ void operator()(int32_t idx) const {
    int64_t offset = static_cast<int64_t>(idx) * detail::kDecimalSumStateSize;
    offsets[idx] = static_cast<OffsetT>(offset);
  }
};

template <typename SumT, typename OffsetT>
struct PackStateFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  cuda::std::span<const OffsetT> offsets;
  uint8_t* chars;

  __device__ void operator()(int32_t idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<DecimalSumState*>(chars + offset);
    int64_t upper;
    uint64_t lower;
    splitToWords(sums[idx], upper, lower);
    state->count = counts[idx];
    state->overflow = 0;
    state->lower = lower;
    state->upper = upper;
  }
};

template <typename OffsetT>
struct UnpackStateFunctor {
  cuda::std::span<const OffsetT> offsets;
  const uint8_t* chars;
  cuda::std::span<__int128_t> sums;
  cuda::std::span<int64_t> counts;

  __device__ void operator()(int32_t idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<const DecimalSumState*>(chars + offset);
    counts[idx] = state->count;
    sums[idx] = (static_cast<__int128_t>(state->upper) << 64) | state->lower;
  }
};

template <typename SumT>
struct AvgRoundFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  cuda::std::span<SumT> out;

  __device__ void operator()(int32_t idx) const {
    auto count = counts[idx];
    if (count == 0) {
      out[idx] = SumT{0};
      return;
    }
    auto sum = sums[idx];
    using U = cuda::std::make_unsigned_t<SumT>;
    U absSum = sum < 0 ? -static_cast<U>(sum) : static_cast<U>(sum);
    U half = static_cast<U>(count / 2);
    U rounded = (absSum + half) / static_cast<U>(count);
    // Use `U{0} - rounded` below to avoid signed overflow
    out[idx] = static_cast<SumT>(sum < 0 ? U{0} - rounded : rounded);
  }
};

template <typename OffsetT>
void launchFillOffsets(
    cuda::std::span<OffsetT> offsets,
    rmm::cuda_stream_view stream) {
  FillOffsetsFunctor<OffsetT> op{offsets};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      static_cast<int32_t>(offsets.size()),
      op,
      stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename SumT, typename OffsetT>
void launchPackState(
    cuda::std::span<const SumT> sums,
    cuda::std::span<const int64_t> counts,
    cuda::std::span<const OffsetT> offsets,
    uint8_t* chars,
    rmm::cuda_stream_view stream) {
  PackStateFunctor<SumT, OffsetT> op{sums, counts, offsets, chars};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      static_cast<int32_t>(sums.size()),
      op,
      stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename OffsetT>
void launchUnpackState(
    cuda::std::span<const OffsetT> offsets,
    const uint8_t* chars,
    cuda::std::span<__int128_t> sums,
    cuda::std::span<int64_t> counts,
    rmm::cuda_stream_view stream) {
  UnpackStateFunctor<OffsetT> op{offsets, chars, sums, counts};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      static_cast<int32_t>(sums.size()),
      op,
      stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename SumT>
void launchAvgRound(
    cuda::std::span<const SumT> sums,
    cuda::std::span<const int64_t> counts,
    cuda::std::span<SumT> out,
    rmm::cuda_stream_view stream) {
  AvgRoundFunctor<SumT> op{sums, counts, out};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0),
      static_cast<int32_t>(out.size()),
      op,
      stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

struct StateValidPredicate {
  cudf::column_device_view sum;
  cudf::column_device_view count;

  __device__ bool operator()(cudf::size_type idx) const {
    if (sum.is_null(idx) || count.is_null(idx)) {
      return false;
    }
    return count.element<int64_t>(idx) != 0;
  }
};

std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMaskImpl(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = sumCol.size();
  if (numRows == 0) {
    return {rmm::device_buffer{}, 0};
  }
  auto sumDeviceView = cudf::column_device_view::create(sumCol, stream);
  auto countDeviceView = cudf::column_device_view::create(countCol, stream);
  StateValidPredicate pred{*sumDeviceView, *countDeviceView};
  // Build a BOOL8 column of per-row validity, then convert via the public API.
  auto bools = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(numRows),
      bools->mutable_view().begin<bool>(),
      pred);
  auto [mask, nullCount] = cudf::bools_to_mask(bools->view(), stream, mr);
  return {std::move(*mask), nullCount};
}

} // namespace

namespace detail {

void fillOffsetsForDecimalSumState(
    bool use64BitOffsets,
    void* offsetsMutable,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  // The offsets buffer holds numRows + 1 entries.
  auto const n = static_cast<size_t>(numRows) + 1;
  if (use64BitOffsets) {
    launchFillOffsets(
        cuda::std::span<int64_t>{static_cast<int64_t*>(offsetsMutable), n},
        stream);
  } else {
    launchFillOffsets(
        cuda::std::span<int32_t>{static_cast<int32_t*>(offsetsMutable), n},
        stream);
  }
}

void packDecimalSumState(
    cudf::type_id sumType,
    bool use64BitOffsets,
    const void* sumPtr,
    const int64_t* countPtr,
    const void* offsetsPtr,
    uint8_t* chars,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  auto const n = static_cast<size_t>(numRows);
  cuda::std::span<const int64_t> counts{countPtr, n};
  if (use64BitOffsets) {
    cuda::std::span<const int64_t> offsets{
        static_cast<const int64_t*>(offsetsPtr), n};
    if (sumType == cudf::type_id::DECIMAL64) {
      launchPackState(
          cuda::std::span<const int64_t>{
              static_cast<const int64_t*>(sumPtr), n},
          counts,
          offsets,
          chars,
          stream);
    } else {
      launchPackState(
          cuda::std::span<const __int128_t>{
              static_cast<const __int128_t*>(sumPtr), n},
          counts,
          offsets,
          chars,
          stream);
    }
  } else {
    cuda::std::span<const int32_t> offsets{
        static_cast<const int32_t*>(offsetsPtr), n};
    if (sumType == cudf::type_id::DECIMAL64) {
      launchPackState(
          cuda::std::span<const int64_t>{
              static_cast<const int64_t*>(sumPtr), n},
          counts,
          offsets,
          chars,
          stream);
    } else {
      launchPackState(
          cuda::std::span<const __int128_t>{
              static_cast<const __int128_t*>(sumPtr), n},
          counts,
          offsets,
          chars,
          stream);
    }
  }
}

void unpackDecimalSumState(
    bool offsets64,
    const void* offsetsPtr,
    const uint8_t* chars,
    __int128_t* sums,
    int64_t* counts,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  auto const n = static_cast<size_t>(numRows);
  cuda::std::span<__int128_t> sumsSpan{sums, n};
  cuda::std::span<int64_t> countsSpan{counts, n};
  if (offsets64) {
    launchUnpackState(
        cuda::std::span<const int64_t>{
            static_cast<const int64_t*>(offsetsPtr), n},
        chars,
        sumsSpan,
        countsSpan,
        stream);
  } else {
    launchUnpackState(
        cuda::std::span<const int32_t>{
            static_cast<const int32_t*>(offsetsPtr), n},
        chars,
        sumsSpan,
        countsSpan,
        stream);
  }
}

void averageRoundDecimalSum(
    cudf::type_id sumType,
    const void* sums,
    const int64_t* counts,
    void* out,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  auto const n = static_cast<size_t>(numRows);
  cuda::std::span<const int64_t> countsSpan{counts, n};
  if (sumType == cudf::type_id::DECIMAL64) {
    launchAvgRound(
        cuda::std::span<const int64_t>{static_cast<const int64_t*>(sums), n},
        countsSpan,
        cuda::std::span<int64_t>{static_cast<int64_t*>(out), n},
        stream);
  } else {
    launchAvgRound(
        cuda::std::span<const __int128_t>{
            static_cast<const __int128_t*>(sums), n},
        countsSpan,
        cuda::std::span<__int128_t>{static_cast<__int128_t*>(out), n},
        stream);
  }
}

std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return buildStateValidityMaskImpl(sumCol, countCol, stream, mr);
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
