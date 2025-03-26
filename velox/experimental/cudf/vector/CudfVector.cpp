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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cmath>

namespace facebook::velox::cudf_velox {
namespace {

// get size in bytes of a column from it's contents and return that and the
// column put back together
std::pair<uint64_t, std::unique_ptr<cudf::column>> getColumnSize(
    std::unique_ptr<cudf::column> column) {
  // when releasing a column, we lose the type, null count, and size so we save
  // it first
  auto type = column->type();
  auto null_count = column->null_count();
  auto size = column->size();

  auto contents = column->release();
  auto bytes = contents.data->size() + contents.null_mask->size();

  // Recursively get the size of the children
  std::vector<std::unique_ptr<cudf::column>> children;
  for (auto& child : contents.children) {
    auto [child_bytes, child_column] = getColumnSize(std::move(child));
    bytes += child_bytes;
    children.push_back(std::move(child_column));
  }

  // put the column back together
  auto reconstituted_column = std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data.release()),
      std::move(*contents.null_mask.release()),
      null_count,
      std::move(children));

  return std::make_pair(bytes, std::move(reconstituted_column));
}

std::pair<uint64_t, std::unique_ptr<cudf::table>> getTableSize(
    std::unique_ptr<cudf::table>&& table) {
  // break apart the table to get to the juicy bits
  auto columns = table->release();
  std::vector<std::unique_ptr<cudf::column>> columns_out;
  uint64_t total_bytes = 0;

  for (auto& column : columns) {
    auto [bytes, column_out] = getColumnSize(std::move(column));
    total_bytes += bytes;
    columns_out.push_back(std::move(column_out));
  }
  return std::make_pair(
      total_bytes, std::make_unique<cudf::table>(std::move(columns_out)));
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
      table_{std::move(table)},
      stream_{stream} {
  auto [bytes, table_out] = getTableSize(std::move(table_));
  flatSize_ = bytes;
  table_ = std::move(table_out);
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

} // namespace facebook::velox::cudf_velox
