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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/prestosql/VectorDistanceKernels.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace facebook::velox::cudf_velox::prestosql {
namespace {

constexpr int kWarpSize{32};
constexpr int kBlockSize{256};
constexpr unsigned kFullWarpMask{0xffffffffu};

template <typename T>
__device__ T quietNan() {
  if constexpr (std::is_same_v<T, float>) {
    return nanf("");
  } else {
    return nan("");
  }
}

template <typename T>
__device__ T squareRoot(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return sqrtf(value);
  } else {
    return sqrt(value);
  }
}

// One row's array, or a null array.
template <typename T>
struct RowView {
  bool isNull{false};
  cudf::size_type size{0};
  const T* values{nullptr};
  // Null when no element can be null.
  const cudf::bitmask_type* nullMask{nullptr};
  cudf::size_type maskOffset{0};

  __device__ bool isElementNull(cudf::size_type index) const {
    return nullMask != nullptr &&
        !cudf::bit_is_set(nullMask, maskOffset + index);
  }
};

template <typename T>
struct ColumnOperand {
  cudf::lists_column_device_view lists;

  __device__ RowView<T> row(cudf::size_type rowIndex) const {
    if (lists.is_null(rowIndex)) {
      return RowView<T>{.isNull = true};
    }
    const auto start = lists.offset_at(rowIndex);
    const auto child = lists.child();
    return RowView<T>{
        .size = lists.offset_at(rowIndex + 1) - start,
        .values = child.data<T>() + start,
        .nullMask = child.nullable() ? child.null_mask() : nullptr,
        .maskOffset = child.offset() + start,
    };
  }
};

template <typename T>
struct ConstantOperand {
  const T* values;
  cudf::size_type size;

  __device__ RowView<T> row(cudf::size_type /*rowIndex*/) const {
    return RowView<T>{.size = size, .values = values};
  }
};

template <typename T>
__device__ T warpSum(T value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

// Assigns one warp per row. Lane 0 writes the row's value and validity.
template <typename T, typename Left, typename Right>
__global__ void vectorDistanceKernel(
    VectorDistanceKind kind,
    cudf::size_type numRows,
    Left left,
    Right right,
    T* out,
    bool* valid,
    int32_t* sizeMismatch) {
  const auto lane = static_cast<int>(threadIdx.x % kWarpSize);
  const auto firstRow =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const auto rowStride =
      static_cast<int64_t>(gridDim.x) * blockDim.x / kWarpSize;
  bool mismatch{false};
  for (auto row = firstRow; row < numRows; row += rowStride) {
    const auto rowIndex = static_cast<cudf::size_type>(row);
    const auto l = left.row(rowIndex);
    const auto r = right.row(rowIndex);
    if (l.isNull || r.isNull) {
      if (lane == 0) {
        valid[rowIndex] = false;
      }
      continue;
    }
    // The CPU returns null for nested nulls before it checks sizes.
    bool hasNullElement{false};
    for (auto i = lane; i < l.size; i += kWarpSize) {
      hasNullElement |= l.isElementNull(i);
    }
    for (auto i = lane; i < r.size; i += kWarpSize) {
      hasNullElement |= r.isElementNull(i);
    }
    if (__any_sync(kFullWarpMask, hasNullElement)) {
      if (lane == 0) {
        valid[rowIndex] = false;
      }
      continue;
    }
    if (l.size != r.size) {
      mismatch = true;
      if (lane == 0) {
        valid[rowIndex] = false;
      }
      continue;
    }

    T dot{0};
    T leftSquares{0};
    T rightSquares{0};
    T squaredDistance{0};
    for (auto i = lane; i < l.size; i += kWarpSize) {
      const auto x = l.values[i];
      const auto y = r.values[i];
      dot += x * y;
      leftSquares += x * x;
      rightSquares += y * y;
      const auto difference = x - y;
      squaredDistance += difference * difference;
    }
    dot = warpSum(dot);
    leftSquares = warpSum(leftSquares);
    rightSquares = warpSum(rightSquares);
    squaredDistance = warpSum(squaredDistance);
    if (lane != 0) {
      continue;
    }
    valid[rowIndex] = true;
    if (l.size == 0) {
      out[rowIndex] = quietNan<T>();
      continue;
    }
    switch (kind) {
      case VectorDistanceKind::kCosineSimilarity: {
        const auto leftNorm = squareRoot(leftSquares);
        const auto rightNorm = squareRoot(rightSquares);
        out[rowIndex] = (leftNorm == 0 || rightNorm == 0)
            ? quietNan<T>()
            : dot / (leftNorm * rightNorm);
        break;
      }
      case VectorDistanceKind::kL2Squared:
        out[rowIndex] = squaredDistance;
        break;
      case VectorDistanceKind::kDotProduct:
        out[rowIndex] = dot;
        break;
    }
  }
  if (lane == 0 && mismatch) {
    atomicOr(sizeMismatch, 1);
  }
}

template <typename T, typename Left, typename Right>
void launch(
    VectorDistanceKind kind,
    cudf::size_type numRows,
    Left left,
    Right right,
    T* out,
    bool* valid,
    int32_t* sizeMismatch,
    cuda::stream_ref stream) {
  constexpr int64_t kRowsPerBlock{kBlockSize / kWarpSize};
  constexpr int64_t kMaxBlocks{1 << 16};
  const auto numBlocks = static_cast<unsigned>(std::min<int64_t>(
      (static_cast<int64_t>(numRows) + kRowsPerBlock - 1) / kRowsPerBlock,
      kMaxBlocks));
  vectorDistanceKernel<T, Left, Right>
      <<<numBlocks, kBlockSize, 0, stream.get()>>>(
          kind, numRows, left, right, out, valid, sizeMismatch);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename T>
cudf::data_type resultType() {
  if constexpr (std::is_same_v<T, double>) {
    return cudf::data_type{cudf::type_id::FLOAT64};
  } else {
    return cudf::data_type{cudf::type_id::FLOAT32};
  }
}

} // namespace

template <typename T>
VectorDistanceResult vectorDistance(
    VectorDistanceKind kind,
    const VectorDistanceOperand<T>& left,
    const VectorDistanceOperand<T>& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
  CUDF_EXPECTS(
      left.column.has_value() || right.column.has_value(),
      "vectorDistance needs a column operand");
  const auto numRows =
      left.column.has_value() ? left.column->size() : right.column->size();
  CUDF_EXPECTS(
      !left.column.has_value() || !right.column.has_value() ||
          left.column->size() == right.column->size(),
      "vectorDistance column operands must have the same size");

  auto result = cudf::make_numeric_column(
      resultType<T>(), numRows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (numRows == 0) {
    return {std::move(result), false};
  }

  rmm::device_uvector<bool> valid(numRows, stream, get_temp_mr());
  cudf::detail::device_scalar<int32_t> sizeMismatch{0, stream, get_temp_mr()};

  const auto copyConstant = [&](const VectorDistanceOperand<T>& operand) {
    rmm::device_uvector<T> values(
        operand.constant.size(), stream, get_temp_mr());
    if (!operand.constant.empty()) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          values.data(),
          operand.constant.data(),
          operand.constant.size_bytes(),
          cudaMemcpyHostToDevice,
          stream.get()));
    }
    return values;
  };
  const auto columnOperand = [&](const cudf::column_view& column) {
    return cudf::column_device_view::create(column, stream, mr);
  };

  auto* out = result->mutable_view().template data<T>();
  if (left.column.has_value() && right.column.has_value()) {
    const auto leftView = columnOperand(*left.column);
    const auto rightView = columnOperand(*right.column);
    launch<T>(
        kind,
        numRows,
        ColumnOperand<T>{cudf::lists_column_device_view(*leftView)},
        ColumnOperand<T>{cudf::lists_column_device_view(*rightView)},
        out,
        valid.data(),
        sizeMismatch.data(),
        stream);
  } else if (left.column.has_value()) {
    const auto leftView = columnOperand(*left.column);
    const auto constant = copyConstant(right);
    launch<T>(
        kind,
        numRows,
        ColumnOperand<T>{cudf::lists_column_device_view(*leftView)},
        ConstantOperand<T>{
            constant.data(), static_cast<cudf::size_type>(constant.size())},
        out,
        valid.data(),
        sizeMismatch.data(),
        stream);
  } else {
    const auto rightView = columnOperand(*right.column);
    const auto constant = copyConstant(left);
    launch<T>(
        kind,
        numRows,
        ConstantOperand<T>{
            constant.data(), static_cast<cudf::size_type>(constant.size())},
        ColumnOperand<T>{cudf::lists_column_device_view(*rightView)},
        out,
        valid.data(),
        sizeMismatch.data(),
        stream);
  }

  const bool mismatch = sizeMismatch.value(stream) != 0;
  const cudf::column_view validView(
      cudf::data_type{cudf::type_id::BOOL8}, numRows, valid.data(), nullptr, 0);
  auto [nullMask, nullCount] = cudf::bools_to_mask(validView, stream, mr);
  if (nullCount > 0) {
    result->set_null_mask(std::move(*nullMask), nullCount);
  }
  return {std::move(result), mismatch};
}

template VectorDistanceResult vectorDistance<float>(
    VectorDistanceKind kind,
    const VectorDistanceOperand<float>& left,
    const VectorDistanceOperand<float>& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

template VectorDistanceResult vectorDistance<double>(
    VectorDistanceKind kind,
    const VectorDistanceOperand<double>& left,
    const VectorDistanceOperand<double>& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox::prestosql
