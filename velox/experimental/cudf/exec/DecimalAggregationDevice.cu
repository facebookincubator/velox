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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_for.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/transform.h>

#include <concepts>
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

struct DecimalSumCount {
  __int128_t sum;
  int64_t count;
};

struct Decimal64ToSumCount {
  int64_t const* values;
  cudf::bitmask_type const* nullMask;

  __device__ DecimalSumCount operator()(cudf::size_type idx) const {
    return nullMask && !cudf::bit_is_set(nullMask, idx)
        ? DecimalSumCount{0, 0}
        : DecimalSumCount{static_cast<__int128_t>(values[idx]), 1};
  }
};

struct AddDecimalSumCount {
  __device__ DecimalSumCount operator()(
      DecimalSumCount lhs,
      DecimalSumCount rhs) const {
    return {lhs.sum + rhs.sum, lhs.count + rhs.count};
  }
};

struct StoreDecimalSumCount {
  DecimalSumCount const* result;
  __int128_t* sum;
  int64_t* count;

  __device__ void operator()(cudf::size_type) const {
    *sum = result->sum;
    *count = result->count;
  }
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
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(cudf::size_type idx) const {
    if (nullMask && !cudf::bit_is_set(nullMask, idx)) {
      return;
    }
    assert(
        offsets[idx + 1] - offsets[idx] ==
        static_cast<OffsetT>(detail::kDecimalSumStateSize));
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

template <typename BuildOp>
void launchDeviceFor(
    cudf::size_type size,
    BuildOp buildOp,
    cuda::stream_ref stream) {
  if (size == 0) {
    return;
  }
  auto op = buildOp();
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0}, size, op, stream.get());
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

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMaskImpl(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = sumCol.size();
  if (numRows == 0) {
    return {
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
        0};
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

void reduceDecimal64SumCount(
    cudf::column_view input,
    cudf::mutable_column_view sum,
    cudf::mutable_column_view count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      input.type().id() == cudf::type_id::DECIMAL64,
      "Direct decimal reduction requires DECIMAL64 input");
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 && sum.size() == 1,
      "Direct decimal reduction requires one DECIMAL128 sum output");
  CUDF_EXPECTS(
      count.type().id() == cudf::type_id::INT64 && count.size() == 1,
      "Direct decimal reduction requires one INT64 count output");

  auto indices = cuda::counting_iterator<cudf::size_type>{0};
  auto transform = Decimal64ToSumCount{
      input.data<int64_t>(), input.nullable() ? input.null_mask() : nullptr};
  auto result = rmm::device_uvector<DecimalSumCount>(1, stream, mr);
  size_t tempStorageBytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(
      nullptr,
      tempStorageBytes,
      indices,
      result.data(),
      input.size(),
      AddDecimalSumCount{},
      transform,
      DecimalSumCount{0, 0},
      stream.get()));
  auto tempStorage = rmm::device_buffer(tempStorageBytes, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(
      tempStorage.data(),
      tempStorageBytes,
      indices,
      result.data(),
      input.size(),
      AddDecimalSumCount{},
      transform,
      DecimalSumCount{0, 0},
      stream.get()));
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0},
      1,
      StoreDecimalSumCount{
          result.data(), sum.data<__int128_t>(), count.data<int64_t>()},
      stream.get());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename T>
concept OffsetStorageType =
    std::same_as<T, int32_t> || std::same_as<T, int64_t>;

template <typename T>
concept DecimalSumStorageType =
    std::same_as<T, int64_t> || std::same_as<T, __int128_t>;

template <typename SumT, typename OffsetT>
concept ValidDecimalPackStorageTypes =
    DecimalSumStorageType<SumT> && OffsetStorageType<OffsetT>;

struct fillOffsetsForDecimalSumStateKernel {
  cudf::mutable_column_view offsetsView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename OffsetT>
    requires OffsetStorageType<OffsetT>
  void operator()() const {
    launchDeviceFor(
        numRows + 1,
        [&] {
          return FillOffsetsFunctor<OffsetT>{cuda::std::span<OffsetT>{
              offsetsView.data<OffsetT>(), static_cast<size_t>(numRows) + 1}};
        },
        stream);
  }

  template <typename OffsetT>
    requires(!OffsetStorageType<OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid offset type for decimal sum state");
  }
};

struct unpackDecimalSumStateKernel {
  cudf::column_view offsetsView;
  const uint8_t* chars;
  cudf::mutable_column_view sumView;
  cudf::mutable_column_view countView;
  cudf::size_type numRows;
  cudf::bitmask_type const* nullMask;
  cuda::stream_ref stream;

  template <typename OffsetT>
    requires OffsetStorageType<OffsetT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    launchDeviceFor(
        numRows,
        [&] {
          return UnpackStateFunctor<OffsetT>{
              cuda::std::span<const OffsetT>{
                  offsetsView.data<OffsetT>(), n + 1},
              chars,
              cuda::std::span<__int128_t>{sumView.data<__int128_t>(), n},
              cuda::std::span<int64_t>{countView.data<int64_t>(), n},
              nullMask};
        },
        stream);
  }

  template <typename OffsetT>
    requires(!OffsetStorageType<OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid offset type for decimal sum state");
  }
};

struct averageRoundDecimalSumKernel {
  cudf::column_view sumCol;
  const int64_t* counts;
  cudf::mutable_column_view outView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT>
    requires DecimalSumStorageType<SumT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    launchDeviceFor(
        numRows,
        [&] {
          return AvgRoundFunctor<SumT>{
              cuda::std::span<const SumT>{sumCol.data<SumT>(), n},
              cuda::std::span<const int64_t>{counts, n},
              cuda::std::span<SumT>{outView.data<SumT>(), n}};
        },
        stream);
  }

  template <typename SumT>
    requires(!DecimalSumStorageType<SumT>)
  void operator()() const {
    CUDF_FAIL("Invalid sum type for decimal average");
  }
};

struct packDecimalSumStateKernel {
  cudf::column_view sumCol;
  const int64_t* counts;
  cudf::column_view offsetsView;
  uint8_t* chars;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT, typename OffsetT>
    requires ValidDecimalPackStorageTypes<SumT, OffsetT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    auto const sums = sumCol.data<SumT>();
    launchDeviceFor(
        numRows,
        [&] {
          return PackStateFunctor<SumT, OffsetT>{
              cuda::std::span<const SumT>{sums, n},
              cuda::std::span<const int64_t>{counts, n},
              cuda::std::span<const OffsetT>{offsetsView.data<OffsetT>(), n},
              chars};
        },
        stream);
  }

  template <typename SumT, typename OffsetT>
    requires(!ValidDecimalPackStorageTypes<SumT, OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid types for decimal sum state pack");
  }
};

void fillOffsetsForDecimalSumState(
    cudf::type_id offsetType,
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher(
      cudf::data_type{offsetType},
      fillOffsetsForDecimalSumStateKernel{offsetsView, numRows, stream});
}

void unpackDecimalSumState(
    cudf::type_id offsetType,
    cudf::column_view offsetsView,
    const uint8_t* chars,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    cudf::size_type numRows,
    cudf::bitmask_type const* nullMask,
    cuda::stream_ref stream) {
  cudf::type_dispatcher(
      cudf::data_type{offsetType},
      unpackDecimalSumStateKernel{
          offsetsView, chars, sumView, countView, numRows, nullMask, stream});
}

void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      averageRoundDecimalSumKernel{sumCol, counts, outView, numRows, stream});
}

void packDecimalSumState(
    cudf::type_id sumType,
    cudf::type_id offsetType,
    cudf::column_view sumCol,
    const int64_t* counts,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      cudf::data_type{offsetType},
      packDecimalSumStateKernel{
          sumCol, counts, offsetsView, chars, numRows, stream});
}

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return buildStateValidityMaskImpl(sumCol, countCol, stream, mr);
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
