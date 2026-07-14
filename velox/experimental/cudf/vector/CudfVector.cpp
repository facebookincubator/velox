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

namespace facebook::velox::cudf_velox {
namespace {

/// Returns total device allocation bytes for a table without disassembling
/// columns. The previous release/reconstitute path corrupted dictionary columns
/// (e.g. dictionary-encoded decimals from Parquet) and could segfault.
uint64_t tableAllocSize(std::unique_ptr<cudf::table>& table) {
  auto columns = table->release();
  uint64_t totalBytes = 0;
  for (auto& column : columns) {
    totalBytes += column->alloc_size();
  }
  table = std::make_unique<cudf::table>(std::move(columns));
  return totalBytes;
}

void logDefaultStreamIfNeeded(
    rmm::cuda_stream_view stream,
    const char* constructorName) {
  if (stream.value() != rmm::cuda_stream_default.value()) {
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
    rmm::cuda_stream_view stream)
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
  flatSize_ = tableAllocSize(tablePtr);
  tabView_ = tablePtr->view();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    rmm::cuda_stream_view stream)
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
  tabView_ = packedPtr->table;
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
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
  stream_.synchronize();
  // Clear the packed table since we've materialized
  packedPtr.reset();
  return materializedTable;
}

bool CudfVector::rebindStream(rmm::cuda_stream_view stream) {
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    if (!*tablePtr) {
      return false;
    }

    if (stream_.value() == stream.value()) {
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

} // namespace facebook::velox::cudf_velox
