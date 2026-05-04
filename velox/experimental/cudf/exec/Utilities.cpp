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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <limits>

namespace facebook::velox::cudf_velox {

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });
  return cudf::concatenate(tableViews, stream, mr);
}

std::unique_ptr<cudf::table> makeEmptyTable(TypePtr const& inputType) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  for (size_t i = 0; i < inputType->size(); ++i) {
    if (auto const& childType = inputType->childAt(i);
        childType->kind() == TypeKind::ROW) {
      auto tbl = makeEmptyTable(childType);
      auto structColumn = std::make_unique<cudf::column>(
          cudf::data_type(cudf::type_id::STRUCT),
          0,
          rmm::device_buffer(),
          rmm::device_buffer(),
          0,
          tbl->release());
      emptyColumns.push_back(std::move(structColumn));
    } else {
      auto emptyColumn = cudf::make_empty_column(
          cudf_velox::veloxToCudfDataType(inputType->childAt(i)));
      emptyColumns.push_back(std::move(emptyColumn));
    }
  }
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  if (tables.size() == 0) {
    return makeEmptyTable(tableType);
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  // Even for a single input table we must concatenate (copy) rather than
  // release in-place: the output is owned by `stream` but the input buffer was
  // allocated on a different stream, so releasing it would bind deallocation to
  // the wrong stream.
  auto output = cudf::concatenate(tableViews, stream, mr);

  // Order input deallocations after the concatenate read.
  // Since memory resources are stream-ordered, deallocations
  // on inputStreams will be ordered after the concatenate completes.
  CudaEvent event(cudaEventDisableTiming);
  streamsWaitForStream(event, inputStreams, stream);
  // Input tables are deallocated here when 'tables' goes out of scope.
  return output;
}

std::vector<std::unique_ptr<cudf::table>> getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::table>> concatTables;
  // Check for empty vector
  if (tables.size() == 0) {
    concatTables.push_back(makeEmptyTable(tableType));
    return concatTables;
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  std::vector<std::unique_ptr<cudf::table>> outputTables;
  const auto& cudfConfig = CudfConfig::getInstance();
  auto const maxRows = cudfConfig.batchSizeMaxThreshold
      ? static_cast<size_t>(cudfConfig.batchSizeMaxThreshold.value())
      : static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
  size_t startpos = 0;
  size_t runningRows = 0;
  for (size_t i = 0; i < tableViews.size(); ++i) {
    auto const numRows = static_cast<size_t>(tableViews[i].num_rows());
    // If adding this table would exceed the limit, flush current batch
    // [startpos, i).
    if (runningRows > 0 && runningRows + numRows > maxRows) {
      outputTables.push_back(
          cudf::concatenate(
              std::vector<cudf::table_view>(
                  tableViews.begin() + startpos, tableViews.begin() + i),
              stream,
              mr));
      startpos = i;
      runningRows = 0;
    }
    runningRows += numRows;
  }
  // Flush the final batch [startpos, end).
  if (startpos < tableViews.size()) {
    outputTables.push_back(
        cudf::concatenate(
            std::vector<cudf::table_view>(
                tableViews.begin() + startpos, tableViews.end()),
            stream,
            mr));
  }
  // Order input deallocations after the concatenate reads by making all input
  // streams wait for the output stream.
  // Since memory resources are stream-ordered, deallocations
  // on inputStreams will be ordered after the concatenate completes.
  CudaEvent event(cudaEventDisableTiming);
  streamsWaitForStream(event, inputStreams, stream);

  // Input tables are deallocated here when 'tables' goes out of scope.
  return outputTables;
}

void streamsWaitForStream(
    CudaEvent& event,
    const std::vector<rmm::cuda_stream_view>& streams,
    rmm::cuda_stream_view stream) {
  event.recordFrom(stream);
  for (const auto& strm : streams) {
    event.waitOn(strm);
  }
}

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEvent_t ev{};
  CUDF_CUDA_TRY(cudaEventCreateWithFlags(&ev, flags));
  event_ = ev;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

const CudaEvent& CudaEvent::recordFrom(rmm::cuda_stream_view stream) const {
  CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
  return *this;
}

const CudaEvent& CudaEvent::waitOn(rmm::cuda_stream_view stream) const {
  CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event_, 0));
  return *this;
}

} // namespace facebook::velox::cudf_velox
