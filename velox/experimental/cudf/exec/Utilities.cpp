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
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

int getNumCudaDevices() {
  int numDevices{};
  CUDF_CUDA_TRY(cudaGetDeviceCount(&numDevices));
  return numDevices;
}

int getCurrentCudaDevice() {
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  return device;
}

CudaEvent& eventForThread() {
  // Intentionally leak per-thread, per-device events to avoid CUDA calls from
  // thread-local destructors after CUDA context teardown.
  thread_local static std::vector<CudaEvent*> events(getNumCudaDevices());
  auto const device = getCurrentCudaDevice();
  VELOX_CHECK_GE(device, 0);
  auto const deviceIndex = static_cast<size_t>(device);
  VELOX_CHECK_LT(deviceIndex, events.size());

  if (events[deviceIndex] == nullptr) {
    events[deviceIndex] = new CudaEvent(cudaEventDisableTiming);
  }
  return *events[deviceIndex];
}

size_t maxBatchRows() {
  const auto& cudfConfig = CudfConfig::getInstance();
  if (cudfConfig.batchSizeMaxThreshold) {
    VELOX_CHECK_GT(
        cudfConfig.batchSizeMaxThreshold.value(),
        0,
        "cuDF max batch size must be positive");
    return static_cast<size_t>(cudfConfig.batchSizeMaxThreshold.value());
  }
  return static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
}

vector_size_t checkedVectorSize(size_t rowCount) {
  VELOX_CHECK_LE(
      rowCount,
      static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
      "cuDF vector row count exceeds Velox vector size limit");
  return static_cast<vector_size_t>(rowCount);
}
} // namespace

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
    auto const& childType = inputType->childAt(i);
    if (childType->kind() == TypeKind::ROW) {
      auto tbl = makeEmptyTable(childType);
      emptyColumns.push_back(
          cudf::make_structs_column(
              0, tbl->release(), 0, rmm::device_buffer()));
    } else if (childType->kind() == TypeKind::ARRAY) {
      // LIST: empty offsets + recursively empty child.
      auto emptyChild = makeEmptyTable(ROW({"e"}, {childType->childAt(0)}));
      auto offsets = cudf::make_empty_column(cudf::type_id::INT32);
      emptyColumns.push_back(
          cudf::make_lists_column(
              0,
              std::move(offsets),
              std::move(emptyChild->release()[0]),
              0,
              rmm::device_buffer()));
    } else {
      auto emptyColumn =
          cudf::make_empty_column(cudf_velox::veloxToCudfDataType(childType));
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

  orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);
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
  auto const maxRows = maxBatchRows();
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
  orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);

  // Input tables are deallocated here when 'tables' goes out of scope.
  return outputTables;
}

std::vector<CudfVectorPtr> getConcatenatedCudfVectorsBatched(
    memory::MemoryPool* pool,
    std::vector<CudfVectorPtr>&& vectors,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(pool);

  std::vector<CudfVectorPtr> outputVectors;
  if (tableType->size() > 0) {
    auto tables =
        getConcatenatedTableBatched(std::move(vectors), tableType, stream, mr);
    outputVectors.reserve(tables.size());
    for (auto& table : tables) {
      VELOX_CHECK_NOT_NULL(table);
      const auto rowCount =
          checkedVectorSize(static_cast<size_t>(table->num_rows()));
      outputVectors.push_back(
          std::make_shared<CudfVector>(
              pool, tableType, rowCount, std::move(table), stream));
    }
    return outputVectors;
  }

  size_t remainingRows = 0;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    VELOX_CHECK_EQ(vector->getTableView().num_columns(), 0);
    const auto rowCount = static_cast<size_t>(vector->size());
    VELOX_CHECK_LE(
        rowCount,
        std::numeric_limits<size_t>::max() - remainingRows,
        "zero-column cuDF vector row count overflow");
    remainingRows += rowCount;
  }

  const auto maxRows = maxBatchRows();
  do {
    const auto chunkRows = std::min(remainingRows, maxRows);
    outputVectors.push_back(
        std::make_shared<CudfVector>(
            pool,
            tableType,
            checkedVectorSize(chunkRows),
            makeEmptyTable(tableType),
            stream));
    remainingRows -= chunkRows;
  } while (remainingRows > 0);

  return outputVectors;
}

void streamsWaitForStream(
    CudaEvent& event,
    std::span<const rmm::cuda_stream_view> streams,
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

std::string getBaseFunctionName(const std::string& fullName) {
  auto pos = fullName.rfind('.');
  return pos == std::string::npos ? fullName : fullName.substr(pos + 1);
}

std::string stripFunctionPrefix(
    const std::string& name,
    const std::string& prefix) {
  auto base = getBaseFunctionName(name);
  if (!prefix.empty() && base.find(prefix) == 0) {
    return base.substr(prefix.size());
  }
  return base;
}

void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    std::span<const rmm::cuda_stream_view> inputStreams,
    rmm::cuda_stream_view stream) {
  bool allRebound = true;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    allRebound &= vector->rebindStream(stream);
  }

  if (!allRebound) {
    streamsWaitForStream(eventForThread(), inputStreams, stream);
  }
}

std::unique_ptr<cudf::column> makeAllNullColumn(
    const TypePtr& type,
    cudf::size_type numRows,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (type->kind()) {
    case TypeKind::ARRAY: {
      // LIST: zeroed offsets (numRows + 1 zeros) + empty child + ALL_NULL mask.
      auto zeroScalar = cudf::make_default_constructed_scalar(
          cudf::data_type{cudf::type_id::INT32}, stream, mr);
      zeroScalar->set_valid_async(true, stream);
      auto offsets =
          cudf::make_column_from_scalar(*zeroScalar, numRows + 1, stream, mr);
      auto child = makeAllNullColumn(type->childAt(0), 0, stream, mr);
      auto nullMask = cudf::create_null_mask(
          numRows, cudf::mask_state::ALL_NULL, stream, mr);
      return cudf::make_lists_column(
          numRows,
          std::move(offsets),
          std::move(child),
          numRows,
          std::move(nullMask));
    }
    case TypeKind::ROW: {
      // STRUCT: recursively create all-null children + ALL_NULL mask.
      std::vector<std::unique_ptr<cudf::column>> children;
      children.reserve(type->size());
      for (auto i = 0; i < type->size(); ++i) {
        children.push_back(
            makeAllNullColumn(type->childAt(i), numRows, stream, mr));
      }
      auto nullMask = cudf::create_null_mask(
          numRows, cudf::mask_state::ALL_NULL, stream, mr);
      return cudf::make_structs_column(
          numRows,
          std::move(children),
          numRows,
          std::move(nullMask),
          stream,
          mr);
    }
    default: {
      // Flat types (including STRING): use the scalar approach.
      auto cudfType = veloxToCudfDataType(type);
      auto nullScalar =
          cudf::make_default_constructed_scalar(cudfType, stream, mr);
      return cudf::make_column_from_scalar(*nullScalar, numRows, stream, mr);
    }
  }
}

} // namespace facebook::velox::cudf_velox
