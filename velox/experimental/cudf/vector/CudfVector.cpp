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
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/buffer/Buffer.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/process/StackTrace.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {
namespace {

// Rebinds every buffer in 'table' to 'stream' for deallocation.
void rebindTable(std::unique_ptr<cudf::table>& table, cuda::stream_ref stream) {
  auto columns = table->release();
  for (auto& column : columns) {
    column = cudf::rebind_stream(std::move(*column), stream);
  }
  table = std::make_unique<cudf::table>(std::move(columns));
}

void logDefaultStreamIfNeeded(
    cuda::stream_ref stream,
    const char* constructorName) {
  if (stream != cuda::stream_ref{cudaStream_t{cudaStreamDefault}}) {
    return;
  }
  LOG(WARNING) << constructorName
               << " constructed with default CUDA stream. Backtrace:\n"
               << process::StackTrace().toString();
}

// Returns false once release() has taken the storage.
template <typename TableStorage>
bool hasStorage(const TableStorage& storage) {
  return std::visit(
      [](const auto& table) { return table != nullptr; }, storage);
}

} // namespace

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::table>&& table,
    cuda::stream_ref stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(table)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(table)");
  auto& tablePtr = std::get<std::unique_ptr<cudf::table>>(tableStorage_);
  rebindTable(tablePtr, stream_);
  flatSize_ = tablePtr->alloc_size();
  tabView_ = tablePtr->view();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    cuda::stream_ref stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(packedTable)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(packed_table)");
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  packedPtr->data.gpu_data->set_stream(stream_);
  tabView_ = packedPtr->table;
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
}

cudf::table_view CudfVector::getTableView() const {
  VELOX_CHECK(hasStorage(tableStorage_), "CudfVector was already released");
  return tabView_;
}

std::unique_ptr<cudf::table> CudfVector::release() {
  VELOX_CHECK(hasStorage(tableStorage_), "CudfVector was already released");
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // Constructed from owned table - just move it out
    flatSize_ = 0;
    return std::move(*tablePtr);
  }
  // Constructed from packed_table - materialize a table from the view.
  // This copies the data since the view references the packed buffer.
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  // Using same memory resource as packed_table
  auto mr = packedPtr->data.gpu_data->memory_resource();
  auto materializedTable = std::make_unique<cudf::table>(tabView_, stream_, mr);
  stream_.sync();
  // Clear the packed table since we've materialized
  packedPtr.reset();
  flatSize_ = 0;
  return materializedTable;
}

bool CudfVector::rebindStream(cuda::stream_ref stream) {
  if (!hasStorage(tableStorage_)) {
    return false;
  }

  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // The constructor and every rebind bind the buffers to 'stream_', so they
    // already free on 'stream'.
    if (stream_.get() == stream.get()) {
      return true;
    }

    rebindTable(*tablePtr, stream);
    tabView_ = (*tablePtr)->view();
    stream_ = stream;
    return true;
  }

  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  packedPtr->data.gpu_data->set_stream(stream);
  stream_ = stream;
  return true;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

uint64_t CudfVector::retainedSizeImpl(
    uint64_t& /*totalStringBufferSize*/) const {
  return flatSize_;
}

void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    cuda::stream_ref stream) {
  // Rebind every input before checking, so a released one does not leave the
  // rest freeing on their producer streams while 'stream' may still read them.
  bool allRebound = true;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    allRebound &= vector->rebindStream(stream);
  }
  VELOX_CHECK(allRebound, "Cannot rebind a released CudfVector");
}

} // namespace facebook::velox::cudf_velox
