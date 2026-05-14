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

#include "velox/experimental/cudf/exec/DecimalAggregationKernelsGpu.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/utilities/error.hpp>

#include <cub/device/device_for.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

// TODO: Handle overflow as in CPU.
struct DecimalSumState {
  int64_t count;    // count of non-null input rows aggregated
  int64_t overflow; // overflow/extension field, not used by GPU
  uint64_t lower;   // lower 64 bits of the decimal sum
  int64_t upper;    // upper 64 bits of the decimal sum (signed)
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
  OffsetT* offsets;

  __device__ void operator()(int32_t idx) const {
    int64_t offset = static_cast<int64_t>(idx) * detail::kDecimalSumStateSize;
    offsets[idx] = static_cast<OffsetT>(offset);
  }
};

template <typename SumT, typename OffsetT>
struct PackStateFunctor {
  const SumT* sums;
  const int64_t* counts;
  const OffsetT* offsets;
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
  const OffsetT* offsets;
  const uint8_t* chars;
  __int128_t* sums;
  int64_t* counts;

  __device__ void operator()(int32_t idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state =
        reinterpret_cast<const DecimalSumState*>(chars + offset);
    counts[idx] = state->count;
    sums[idx] = (static_cast<__int128_t>(state->upper) << 64) | state->lower;
  }
};

template <typename SumT>
struct AvgRoundFunctor {
  const SumT* sums;
  const int64_t* counts;
  SumT* out;

  __device__ void operator()(int32_t idx) const {
    auto count = counts[idx];
    if (count == 0) {
      out[idx] = SumT{0};
      return;
    }
    auto sum = sums[idx];
    SumT absSum = sum < 0 ? -sum : sum;
    SumT half = static_cast<SumT>(count / 2);
    SumT rounded = (absSum + half) / static_cast<SumT>(count);
    out[idx] = sum < 0 ? -rounded : rounded;
  }
};

template <typename OffsetT>
void launchFillOffsets(
    OffsetT* offsets,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  FillOffsetsFunctor<OffsetT> op{offsets};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), numRows + 1, op, stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename SumT, typename OffsetT>
void launchPackState(
    const SumT* sums,
    const int64_t* counts,
    const OffsetT* offsets,
    uint8_t* chars,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  PackStateFunctor<SumT, OffsetT> op{sums, counts, offsets, chars};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), numRows, op, stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename OffsetT>
void launchUnpackState(
    const OffsetT* offsets,
    const uint8_t* chars,
    __int128_t* sums,
    int64_t* counts,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  UnpackStateFunctor<OffsetT> op{offsets, chars, sums, counts};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), numRows, op, stream.value());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename SumT>
void launchAvgRound(
    const SumT* sums,
    const int64_t* counts,
    SumT* out,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  AvgRoundFunctor<SumT> op{sums, counts, out};
  cub::DeviceFor::ForEachN(
      thrust::counting_iterator<int32_t>(0), numRows, op, stream.value());
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
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end = begin + numRows;
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

} // namespace

namespace detail {

void fillOffsetsForDecimalSumState(
    bool use64BitOffsets,
    void* offsetsMutable,
    int32_t numRows,
    rmm::cuda_stream_view stream) {
  if (use64BitOffsets) {
    launchFillOffsets(static_cast<int64_t*>(offsetsMutable), numRows, stream);
  } else {
    launchFillOffsets(static_cast<int32_t*>(offsetsMutable), numRows, stream);
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
  if (use64BitOffsets) {
    auto offsets = static_cast<const int64_t*>(offsetsPtr);
    if (sumType == cudf::type_id::DECIMAL64) {
      launchPackState(
          static_cast<const int64_t*>(sumPtr),
          countPtr,
          offsets,
          chars,
          numRows,
          stream);
    } else {
      launchPackState(
          static_cast<const __int128_t*>(sumPtr),
          countPtr,
          offsets,
          chars,
          numRows,
          stream);
    }
  } else {
    auto offsets = static_cast<const int32_t*>(offsetsPtr);
    if (sumType == cudf::type_id::DECIMAL64) {
      launchPackState(
          static_cast<const int64_t*>(sumPtr),
          countPtr,
          offsets,
          chars,
          numRows,
          stream);
    } else {
      launchPackState(
          static_cast<const __int128_t*>(sumPtr),
          countPtr,
          offsets,
          chars,
          numRows,
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
  if (offsets64) {
    launchUnpackState(
        static_cast<const int64_t*>(offsetsPtr),
        chars,
        sums,
        counts,
        numRows,
        stream);
  } else {
    launchUnpackState(
        static_cast<const int32_t*>(offsetsPtr),
        chars,
        sums,
        counts,
        numRows,
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
  if (sumType == cudf::type_id::DECIMAL64) {
    launchAvgRound(
        static_cast<const int64_t*>(sums),
        counts,
        static_cast<int64_t*>(out),
        numRows,
        stream);
  } else {
    launchAvgRound(
        static_cast<const __int128_t*>(sums),
        counts,
        static_cast<__int128_t*>(out),
        numRows,
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
