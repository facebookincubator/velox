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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"

#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/type_traits>
#include <thrust/count.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

// Applies bitmap membership to each row's absolute index. A row is deleted if
// it was already deleted or if the bitmap bit is set.
struct IsDeletedRow {
  const cudf::bitmask_type* bitmask;
  const size_t* rowIndex;
  std::size_t startRow;
  std::size_t numRows;

  __device__ bool operator()(cudf::size_type index, bool wasDeleted)
      const noexcept {
    const auto row = rowIndex[index];
    if (row < startRow) {
      return wasDeleted;
    }
    const auto bitmapIndex = row - startRow;
    return wasDeleted or
        (bitmapIndex < numRows and
         cudf::bit_is_set(bitmask, static_cast<cudf::size_type>(bitmapIndex)));
  }
};

} // namespace

void applyBitmapToMask(
    cudf::device_span<const cudf::bitmask_type> bitmap,
    std::size_t startRow,
    std::size_t numRows,
    cudf::column_view const& rowIndex,
    cudf::mutable_column_view const& deleteMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  auto iter = cuda::counting_iterator{0};
  thrust::transform(
      rmm::exec_policy_nosync(stream, temp_mr),
      iter,
      iter + deleteMask.size(),
      deleteMask.begin<bool>(),
      deleteMask.begin<bool>(),
      IsDeletedRow{bitmap.data(), rowIndex.begin<size_t>(), startRow, numRows});
}

cudf::size_type countDeletedRows(
    cudf::column_view const& deleteMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  return static_cast<cudf::size_type>(thrust::count(
      rmm::exec_policy_nosync(stream, temp_mr),
      deleteMask.begin<bool>(),
      deleteMask.end<bool>(),
      true));
}

void scatterDeletesToMask(
    cudf::mutable_column_view const& deleteMask,
    cudf::device_span<const cudf::size_type> indices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  // Alternate: Use `cudf::scatter` but it produces a new column.
  auto iter = cuda::constant_iterator<bool>(true);
  thrust::scatter(
      rmm::exec_policy_nosync(stream, temp_mr),
      iter,
      iter + indices.size(),
      indices.begin(),
      deleteMask.begin<bool>());
}

template <typename ValueType>
void fillSequence(
    cudf::mutable_column_view const& rowIndices,
    ValueType startRow,
    int64_t numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr) {
  auto rowIndexIter = rowIndices.begin<ValueType>();
  thrust::sequence(
      rmm::exec_policy_nosync(stream, temp_mr),
      rowIndexIter,
      rowIndexIter + numRows,
      static_cast<ValueType>(startRow));
}

template void fillSequence<uint32_t>(
    cudf::mutable_column_view const&,
    uint32_t,
    int64_t,
    rmm::cuda_stream_view,
    rmm::device_async_resource_ref);

template void fillSequence<uint64_t>(
    cudf::mutable_column_view const&,
    uint64_t,
    int64_t,
    rmm::cuda_stream_view,
    rmm::device_async_resource_ref);

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
