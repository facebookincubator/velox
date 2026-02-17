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
#include "velox/experimental/cudf/exec/DecimalAggregationKernels.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

constexpr int32_t kStateSize = 32;

struct DecimalSumStateDevice {
  int64_t count;
  int64_t overflow;
  uint64_t lower;
  int64_t upper;
};

static_assert(sizeof(DecimalSumStateDevice) == kStateSize);

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

__global__ void fillOffsetsKernel(int32_t* offsets, int32_t numRows) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= numRows) {
    offsets[idx] = idx * kStateSize;
  }
}

template <typename SumT>
__global__ void packStateKernel(
    const SumT* sums,
    const int64_t* counts,
    const int32_t* offsets,
    uint8_t* chars,
    int32_t numRows) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  int32_t offset = offsets[idx];
  auto* state = reinterpret_cast<DecimalSumStateDevice*>(chars + offset);
  int64_t upper;
  uint64_t lower;
  splitToWords(sums[idx], upper, lower);
  state->count = counts[idx];
  state->overflow = 0;
  state->lower = lower;
  state->upper = upper;
}

__global__ void unpackStateKernel(
    const int32_t* offsets,
    const uint8_t* chars,
    __int128_t* sums,
    int64_t* counts,
    int32_t numRows) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  int32_t offset = offsets[idx];
  auto* state = reinterpret_cast<const DecimalSumStateDevice*>(chars + offset);
  counts[idx] = state->count;
  sums[idx] = (static_cast<__int128_t>(state->upper) << 64) | state->lower;
}

template <typename SumT>
__global__ void avgRoundKernel(
    const SumT* sums,
    const int64_t* counts,
    SumT* out,
    int32_t numRows) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
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

std::pair<rmm::device_buffer, cudf::size_type> buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream) {
  auto numRows = sumCol.size();
  if (numRows == 0) {
    return {rmm::device_buffer{}, 0};
  }
  auto sumDeviceView = cudf::column_device_view::create(sumCol, stream);
  auto countDeviceView = cudf::column_device_view::create(countCol, stream);
  StateValidPredicate pred{*sumDeviceView, *countDeviceView};
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end = begin + numRows;
  return cudf::detail::valid_if(
      begin, end, pred, stream, cudf::get_current_device_resource_ref());
}

} // namespace

DecimalSumStateColumns deserializeDecimalSumStateWithCount(
    const cudf::column_view& stateCol,
    int32_t scale,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(
      stateCol.type().id() == cudf::type_id::STRING,
      "Decimal sum state requires STRING/VARBINARY column");
  auto numRows = stateCol.size();
  if (numRows == 0) {
    DecimalSumStateColumns empty;
    empty.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        0,
        cudf::mask_state::UNALLOCATED,
        stream);
    empty.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        0,
        cudf::mask_state::UNALLOCATED,
        stream);
    return empty;
  }

  // For fully-null state columns there is nothing to deserialize. Avoid
  // launching unpack kernels over string payload buffers that may be empty.
  if (stateCol.nullable() && stateCol.null_count() == numRows) {
    DecimalSumStateColumns allNull;
    allNull.sum = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::DECIMAL128, -scale},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream);
    allNull.count = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream);
    return allNull;
  }

  cudf::strings_column_view strings(stateCol);
  numRows = strings.size();

  auto offsetsView = strings.offsets();
  auto offsetsCol = offsetsView.data<int32_t>();
  auto charsPtr = reinterpret_cast<const uint8_t*>(strings.chars_begin(stream));

  auto sumCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, -scale},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream);
  auto countCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream);

  auto sumView = sumCol->mutable_view();
  auto countView = countCol->mutable_view();

  if (numRows > 0) {
    int32_t blockSize = 256;
    int32_t gridSize = (numRows + blockSize - 1) / blockSize;
    unpackStateKernel<<<gridSize, blockSize, 0, stream.value()>>>(
        offsetsCol,
        charsPtr,
        sumView.data<__int128_t>(),
        countView.data<int64_t>(),
        numRows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (stateCol.nullable()) {
    auto nullMask = cudf::detail::copy_bitmask(
        stateCol, stream, cudf::get_current_device_resource_ref());
    auto nullCount = stateCol.null_count();
    sumCol->set_null_mask(std::move(nullMask), nullCount);
    auto countMask = cudf::detail::copy_bitmask(
        stateCol, stream, cudf::get_current_device_resource_ref());
    countCol->set_null_mask(std::move(countMask), nullCount);
  }

  DecimalSumStateColumns result;
  result.sum = std::move(sumCol);
  result.count = std::move(countCol);
  return result;
}

std::unique_ptr<cudf::column> deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    rmm::cuda_stream_view stream) {
  auto decoded = deserializeDecimalSumStateWithCount(stateCol, scale, stream);
  return std::move(decoded.sum);
}

std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal sum state requires INT64 count column");
  auto numRows = sumCol.size();
  CUDF_EXPECTS(
      numRows == countCol.size(),
      "Decimal sum state requires sum and count to be same size");
  CUDF_EXPECTS(
      numRows <= std::numeric_limits<int32_t>::max(),
      "Too many rows to serialize decimal sum state");

  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream);
  auto offsetsView = offsetsCol->mutable_view();

  rmm::device_buffer charsBuf(
      static_cast<size_t>(numRows) * kStateSize, stream);

  int32_t blockSize = 256;
  int32_t offsetGridSize = (numRows + 1 + blockSize - 1) / blockSize;
  fillOffsetsKernel<<<offsetGridSize, blockSize, 0, stream.value()>>>(
      offsetsView.data<int32_t>(), numRows);
  CUDF_CUDA_TRY(cudaGetLastError());

  if (numRows > 0) {
    int32_t gridSize = (numRows + blockSize - 1) / blockSize;
    auto offsetsPtr = offsetsView.data<int32_t>();
    auto charsPtr = reinterpret_cast<uint8_t*>(charsBuf.data());
    if (sumCol.type().id() == cudf::type_id::DECIMAL64) {
      packStateKernel<int64_t><<<gridSize, blockSize, 0, stream.value()>>>(
          sumCol.data<int64_t>(),
          countCol.data<int64_t>(),
          offsetsPtr,
          charsPtr,
          numRows);
    } else {
      CUDF_EXPECTS(
          sumCol.type().id() == cudf::type_id::DECIMAL128,
          "Unsupported decimal sum column type");
      packStateKernel<__int128_t><<<gridSize, blockSize, 0, stream.value()>>>(
          sumCol.data<__int128_t>(),
          countCol.data<int64_t>(),
          offsetsPtr,
          charsPtr,
          numRows);
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto [nullMask, nullCount] = buildStateValidityMask(sumCol, countCol, stream);
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(numRows),
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));
}

std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal average requires INT64 count column");
  CUDF_EXPECTS(
      sumCol.type().id() == cudf::type_id::DECIMAL64 ||
          sumCol.type().id() == cudf::type_id::DECIMAL128,
      "Decimal average requires DECIMAL64 or DECIMAL128 sum column");
  CUDF_EXPECTS(
      sumCol.size() == countCol.size(),
      "Decimal average requires sum and count to be same size");

  auto numRows = sumCol.size();
  auto out = cudf::make_fixed_width_column(
      sumCol.type(), numRows, cudf::mask_state::UNALLOCATED, stream);

  if (numRows > 0) {
    int32_t blockSize = 256;
    int32_t gridSize = (numRows + blockSize - 1) / blockSize;
    if (sumCol.type().id() == cudf::type_id::DECIMAL64) {
      avgRoundKernel<<<gridSize, blockSize, 0, stream.value()>>>(
          sumCol.data<int64_t>(),
          countCol.data<int64_t>(),
          out->mutable_view().data<int64_t>(),
          numRows);
    } else {
      avgRoundKernel<<<gridSize, blockSize, 0, stream.value()>>>(
          sumCol.data<__int128_t>(),
          countCol.data<int64_t>(),
          out->mutable_view().data<__int128_t>(),
          numRows);
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto [nullMask, nullCount] = buildStateValidityMask(sumCol, countCol, stream);
  if (nullCount > 0) {
    out->set_null_mask(std::move(nullMask), nullCount);
  } else if (nullMask.size() > 0) {
    out->set_null_mask(std::move(nullMask), 0);
  }
  return out;
}

} // namespace facebook::velox::cudf_velox
