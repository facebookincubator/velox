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
#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
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

  __device__ void operator()(cudf::size_type idx) const {
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

  __device__ void operator()(cudf::size_type idx) const {
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

  __device__ void operator()(cudf::size_type idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<const DecimalSumState*>(chars + offset);
    counts[idx] = state->count;
    sums[idx] = (static_cast<__int128_t>(state->upper) << 64) | state->lower;
  }
};

// Half-up sum/count divide for AVG.
template <typename SumT>
struct AvgRoundFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  cuda::std::span<SumT> out;

  __device__ void operator()(cudf::size_type idx) const {
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
      cuda::counting_iterator{0},
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
      cuda::counting_iterator{0},
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
      cuda::counting_iterator{0},
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
      cuda::counting_iterator{0},
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
  auto iter = cuda::counting_iterator{0};
  thrust::transform(
      rmm::exec_policy(stream),
      iter,
      iter + numRows,
      bools->mutable_view().begin<bool>(),
      pred);
  auto [mask, nullCount] = cudf::bools_to_mask(bools->view(), stream, mr);
  return {std::move(*mask), nullCount};
}

} // namespace

namespace detail {

template <typename OffsetT, std::enable_if_t<isOffsetStorageType<OffsetT>, int>>
void fillOffsetsForDecimalSumState::operator()(
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const {
  launchFillOffsets(
      cuda::std::span<OffsetT>{
          offsetsView.data<OffsetT>(), static_cast<size_t>(numRows) + 1},
      stream);
}

template void fillOffsetsForDecimalSumState::operator()<int32_t, 0>(
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;
template void fillOffsetsForDecimalSumState::operator()<int64_t, 0>(
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;

template <typename OffsetT, std::enable_if_t<isOffsetStorageType<OffsetT>, int>>
void unpackDecimalSumState::operator()(
    cudf::column_view offsetsView,
    const uint8_t* chars,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const {
  auto const n = static_cast<size_t>(numRows);
  launchUnpackState(
      cuda::std::span<const OffsetT>{offsetsView.data<OffsetT>(), n},
      chars,
      cuda::std::span<__int128_t>{sumView.data<__int128_t>(), n},
      cuda::std::span<int64_t>{countView.data<int64_t>(), n},
      stream);
}

template void unpackDecimalSumState::operator()<int32_t, 0>(
    cudf::column_view offsetsView,
    const uint8_t* chars,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;
template void unpackDecimalSumState::operator()<int64_t, 0>(
    cudf::column_view offsetsView,
    const uint8_t* chars,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;

template <typename SumT, std::enable_if_t<isDecimalSumStorageType<SumT>, int>>
void packDecimalSumState::operator()(
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const {
  auto const n = static_cast<size_t>(numRows);
  auto const sums = sumCol.data<SumT>();
  if (offsetsView.type().id() == cudf::type_id::INT32) {
    launchPackState(
        cuda::std::span<const SumT>{sums, n},
        cuda::std::span<const int64_t>{counts, n},
        cuda::std::span<const int32_t>{offsetsView.data<int32_t>(), n},
        chars,
        stream);
  } else {
    launchPackState(
        cuda::std::span<const SumT>{sums, n},
        cuda::std::span<const int64_t>{counts, n},
        cuda::std::span<const int64_t>{offsetsView.data<int64_t>(), n},
        chars,
        stream);
  }
}

template void packDecimalSumState::operator()<int64_t, 0>(
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;
template void packDecimalSumState::operator()<__int128_t, 0>(
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;

template <typename SumT, std::enable_if_t<isDecimalSumStorageType<SumT>, int>>
void averageRoundDecimalSum::operator()(
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const {
  auto const n = static_cast<size_t>(numRows);
  launchAvgRound(
      cuda::std::span<const SumT>{sumCol.data<SumT>(), n},
      cuda::std::span<const int64_t>{counts, n},
      cuda::std::span<SumT>{outView.data<SumT>(), n},
      stream);
}

template void averageRoundDecimalSum::operator()<int64_t, 0>(
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;
template void averageRoundDecimalSum::operator()<__int128_t, 0>(
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream) const;

std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return buildStateValidityMaskImpl(sumCol, countCol, stream, mr);
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
