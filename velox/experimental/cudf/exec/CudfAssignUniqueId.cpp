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
#include "velox/experimental/cudf/exec/CudfAssignUniqueId.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/lists/filling.hpp>

#include <utility>

namespace facebook::velox::cudf_velox {

CudfAssignUniqueId::CudfAssignUniqueId(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::AssignUniqueIdNode>& planNode,
    int32_t uniqueTaskId,
    std::shared_ptr<std::atomic_int64_t> rowIdPool)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfAssignUniqueId"),
      NvtxHelper(
          nvtx3::rgb{160, 82, 45}, // Sienna
          operatorId,
          fmt::format("[{}]", planNode->id())),
      rowIdPool_(std::move(rowIdPool)) {
  VELOX_USER_CHECK_LT(
      uniqueTaskId,
      kTaskUniqueIdLimit,
      "Unique 24-bit ID specified for CudfAssignUniqueId exceeds the limit");
  uniqueValueMask_ = static_cast<int64_t>(uniqueTaskId) << 40;

  rowIdCounter_ = 0;
  maxRowIdCounterValue_ = 0;
}

void CudfAssignUniqueId::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto numInput = input->size();
  VELOX_CHECK_NE(
      numInput, 0, "CudfAssignUniqueId::addInput received empty set of rows");
  input_ = std::move(input);
}

RowVectorPtr CudfAssignUniqueId::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (input_ == nullptr) {
    return nullptr;
  }

  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK(cudfVector, "Input must be a CudfVector");
  auto stream = cudfVector->stream();
  auto uniqueIdColumn =
      generateIdColumn(input_->size(), stream, cudf_velox::get_output_mr());
  auto size = cudfVector->size();
  auto columns = cudfVector->release()->release();
  columns.push_back(std::move(uniqueIdColumn));
  auto output = std::make_shared<CudfVector>(
      input_->pool(),
      outputType_,
      size,
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
  input_ = nullptr;
  return output;
}

bool CudfAssignUniqueId::isFinished() {
  return noMoreInput_ && input_ == nullptr;
}

std::unique_ptr<cudf::column> CudfAssignUniqueId::generateIdColumn(
    vector_size_t size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<int64_t> starts, sizes;
  starts.reserve(size / kRowIdsPerRequest + 1);
  sizes.reserve(size / kRowIdsPerRequest + 1);

  vector_size_t start = 0;
  while (start < size) {
    if (rowIdCounter_ >= maxRowIdCounterValue_) {
      requestRowIds();
    }

    const auto numAvailableIds =
        std::min(maxRowIdCounterValue_ - rowIdCounter_, kRowIdsPerRequest);
    const vector_size_t end =
        std::min(static_cast<int64_t>(size), start + numAvailableIds);
    VELOX_CHECK_EQ(
        (rowIdCounter_ + (end - start)) & uniqueValueMask_,
        0,
        "Ran out of unique IDs at {}. Need {} more.",
        rowIdCounter_,
        (end - start));
    starts.push_back(uniqueValueMask_ | rowIdCounter_);
    sizes.push_back(end - start);

    rowIdCounter_ += (end - start);
    start = end;
  }

  // Copy starts and sizes to device.
  rmm::device_buffer d_starts_buffer(
      starts.data(), starts.size() * sizeof(int64_t), stream, mr);
  rmm::device_buffer d_sizes_buffer(
      sizes.data(), sizes.size() * sizeof(int64_t), stream, mr);
  auto d_starts_column_view = cudf::column_view(
      cudf::data_type(cudf::type_id::INT64),
      starts.size(),
      d_starts_buffer.data(),
      nullptr,
      0,
      0);
  auto d_sizes_column_view = cudf::column_view(
      cudf::data_type(cudf::type_id::INT64),
      sizes.size(),
      d_sizes_buffer.data(),
      nullptr,
      0,
      0);

  auto list_sequence = cudf::lists::sequences(
      d_starts_column_view, d_sizes_column_view, stream, mr);
  // Discard offsets.
  return std::move(list_sequence->release().children[1]);
}

void CudfAssignUniqueId::requestRowIds() {
  rowIdCounter_ = rowIdPool_->fetch_add(kRowIdsPerRequest);
  maxRowIdCounterValue_ =
      std::min(rowIdCounter_ + kRowIdsPerRequest, kMaxRowId);
}
} // namespace facebook::velox::cudf_velox
