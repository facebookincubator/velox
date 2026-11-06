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
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/process/StackTrace.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/table/table.hpp>

#include <exception>

namespace facebook::velox::cudf_velox {
namespace {

/// Calculates the total memory size in bytes of a cudf column and reconstructs
/// it.
///
/// This function disassembles a cudf column to access its underlying memory
/// buffers, calculates the total size including children columns (for nested
/// types), and then reassembles the column.
///
/// @return A pair containing the total size in bytes and the reconstructed
/// column
std::pair<uint64_t, std::unique_ptr<cudf::column>> getColumnSize(
    std::unique_ptr<cudf::column> column) {
  // Store column metadata (type, null count, and size) before releasing it,
  // as the release() operation transfers ownership of the underlying buffers
  // and invalidates access to these properties.
  auto type = column->type();
  auto nullCount = column->null_count();
  auto size = column->size();

  auto contents = column->release();
  auto bytes = contents.data->size() + contents.null_mask->size();

  // Recursively get the size of the children columns.
  std::vector<std::unique_ptr<cudf::column>> children;
  for (auto& child : contents.children) {
    auto [childBytes, childColumn] = getColumnSize(std::move(child));
    bytes += childBytes;
    children.push_back(std::move(childColumn));
  }

  // Reassemble the column with the original metadata.
  auto reconstitutedColumn = std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data.release()),
      std::move(*contents.null_mask.release()),
      nullCount,
      std::move(children));

  return std::make_pair(bytes, std::move(reconstitutedColumn));
}

/// Calculates the total memory size in bytes of a cudf table and reconstructs
/// it.
///
/// This function disassembles a cudf table to access its underlying columns,
/// calculates the total size, and then reassembles the table.
///
/// @note This is a workaround because cudf::table doesn't have an API to get
/// this information without involving estimation and d->h copies.
/// @see https://github.com/rapidsai/cudf/issues/18462
///
/// @return A pair containing the total size in bytes and the reconstructed
/// table
std::pair<uint64_t, std::unique_ptr<cudf::table>> getTableSize(
    std::unique_ptr<cudf::table>&& table) {
  auto columns = table->release();
  std::vector<std::unique_ptr<cudf::column>> columnsOut;
  uint64_t totalBytes = 0;

  for (auto& column : columns) {
    auto [bytes, columnOut] = getColumnSize(std::move(column));
    totalBytes += bytes;
    columnsOut.push_back(std::move(columnOut));
  }
  return std::make_pair(
      totalBytes, std::make_unique<cudf::table>(std::move(columnsOut)));
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
  auto [bytes, tableOut] = getTableSize(std::move(tablePtr));
  flatSize_ = bytes;
  tablePtr = std::move(tableOut);
  tabView_ = tablePtr->view();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    cuda::stream_ref stream,
    ReleaseCallback releaseCallback)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(packedTable)},
      stream_{stream},
      releaseCallback_{std::move(releaseCallback)} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(packed_table)");
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  tabView_ = packedPtr->table;
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
}

CudfVector::~CudfVector() {
  // Release the GPU allocation first. device_buffer deallocation is ordered on
  // its associated stream; runReleaseCallback() synchronizes that stream before
  // reporting the bytes as available to exchange flow control.
  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_);
      packedPtr && *packedPtr) {
    // The logical stream can differ from the original packing stream even if
    // the vector is dropped before its first rebindStream().
    (*packedPtr)->data.gpu_data->set_stream(stream_);
    packedPtr->reset();
  }
  runReleaseCallback();
}

void CudfVector::runReleaseCallback(
    bool synchronizeStream,
    std::exception_ptr synchronizeError) noexcept {
  ReleaseCallback callback;
  callback.swap(releaseCallback_);
  if (!callback) {
    return;
  }

  // rebindStream() may move ownership to a downstream operator's stream. Use
  // stream_ at release time rather than capturing the receive stream when the
  // callback is installed.
  if (synchronizeStream) {
    try {
      stream_.sync();
    } catch (const std::exception& e) {
      synchronizeError = std::current_exception();
      LOG(ERROR) << "Failed to synchronize CudfVector release stream: "
                 << e.what();
    } catch (...) {
      synchronizeError = std::current_exception();
      LOG(ERROR) << "Failed to synchronize CudfVector release stream";
    }
  }

  try {
    callback(std::move(synchronizeError));
  } catch (const std::exception& e) {
    LOG(ERROR) << "CudfVector release callback failed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "CudfVector release callback failed";
  }
}

std::unique_ptr<cudf::table> CudfVector::release() {
  flatSize_ = 0;
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // Constructed from owned table - just move it out
    return std::move(*tablePtr);
  }
  // Constructed from packed_table - materialize a table from the view.
  // This copies the data since the view references the packed buffer.
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  // Using same memory resource as packed_table
  auto mr = packedPtr->data.gpu_data->memory_resource();
  packedPtr->data.gpu_data->set_stream(stream_);
  auto materializedTable = std::make_unique<cudf::table>(tabView_, stream_, mr);
  // Clear the packed table once its materialization is ordered on stream_. The
  // following synchronization waits for both the copy and the stream-ordered
  // deallocation before returning its receive credit.
  packedPtr.reset();
  try {
    stream_.sync();
  } catch (...) {
    auto error = std::current_exception();
    runReleaseCallback(/*synchronizeStream=*/false, error);
    std::rethrow_exception(error);
  }
  runReleaseCallback(/*synchronizeStream=*/false);
  return materializedTable;
}

bool CudfVector::rebindStream(cuda::stream_ref stream) {
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    if (!*tablePtr) {
      return false;
    }

    if (stream_.get() == stream.get()) {
      return true;
    }

    auto columns = (*tablePtr)->release();
    for (auto& column : columns) {
      column = cudf::rebind_stream(std::move(*column), stream);
    }

    *tablePtr = std::make_unique<cudf::table>(std::move(columns));
    tabView_ = (*tablePtr)->view();
    stream_ = stream;
    return true;
  }

  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    if (!*packedPtr) {
      return false;
    }

    (*packedPtr)->data.gpu_data->set_stream(stream);
    stream_ = stream;
    return true;
  }

  return false;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

uint64_t CudfVector::retainedSizeImpl(
    uint64_t& /*totalStringBufferSize*/) const {
  return flatSize_;
}

} // namespace facebook::velox::cudf_velox
