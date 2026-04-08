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
#include <thrust/for_each.h>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

/// Functor that returns true when the bit at `index` is clear. i.e., the row
/// is NOT deleted and should survive.
struct IsSurvivingRow {
  const cudf::bitmask_type* bitmask;
  __device__ bool operator()(cudf::size_type index) const noexcept {
    return not cudf::bit_is_set(bitmask, index);
  }
};

} // namespace

std::unique_ptr<cudf::table> applyDeleteBitmap(
    cudf::table_view input,
    const uint8_t* hostBitmap,
    std::shared_ptr<rmm::device_buffer> deviceBitmap,
    std::shared_ptr<rmm::device_buffer> rowMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temp_mr,
    rmm::device_async_resource_ref output_mr) {
  const auto numRows = input.num_rows();
  const auto numWords = cudf::num_bitmask_words(numRows);
  const auto numBitmaskBytes = numWords * sizeof(cudf::bitmask_type);

  // Copy the deletion bitmap to device
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      deviceBitmap->data(),
      hostBitmap,
      numBitmaskBytes,
      cudaMemcpyHostToDevice,
      stream.value()));

  // Transform the deletion bitmap to the surviving row mask
  thrust::transform(
      rmm::exec_policy_nosync(stream, temp_mr),
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::counting_iterator<cudf::size_type>{numRows},
      static_cast<bool*>(rowMask->data()),
      IsSurvivingRow{
          static_cast<const cudf::bitmask_type*>(deviceBitmap->data())});

  // Convert the surviving row mask to a column view
  auto rowMaskCol = cudf::column_view(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      rowMask->data(),
      nullptr,
      0,
      0);

  // Apply the boolean mask to the input table
  return cudf::apply_boolean_mask(input, rowMaskCol, stream, output_mr);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
