/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#pragma once

#include "velox/common/memory/MemoryPool.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>
#include <variant>

namespace facebook::velox::cudf_velox {

// Vector class which holds GPU data from cuDF.
// Can be constructed either from an owned cudf::table or from packed_table.
// When constructed from packed_table, the data remains packed and tabView_
// references the table view inside packed_table without copying the underlying
// GPU data.
class CudfVector : public RowVector {
 public:
  /// Constructs a CudfVector from an owned cudf::table.
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::table>&& table,
      rmm::cuda_stream_view stream);

  /// Constructs a CudfVector from packed_table.
  /// The packed data is retained and tabView_ references the table view inside
  /// packed_table. This avoids copying the underlying GPU data.
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::packed_table>&& packedTable,
      rmm::cuda_stream_view stream);

  rmm::cuda_stream_view stream() const {
    return stream_;
  }

  cudf::table_view getTableView() const {
    return tabView_;
  }

  /// Releases ownership of the underlying table.
  /// If constructed from packed_table, materializes a table from the view
  /// first (which copies the data).
  std::unique_ptr<cudf::table> release();

  uint64_t estimateFlatSize() const override;

 private:
  // Storage for either an owned table or packed table.
  // Only one is active at a time - using variant enforces this at compile time.
  using TableStorage = std::variant<
      std::unique_ptr<cudf::table>,
      std::unique_ptr<cudf::packed_table>>;
  TableStorage tableStorage_;

  // Table view - always valid, points to either table_->view() or
  // packedTable_->table.
  cudf::table_view tabView_;

  rmm::cuda_stream_view stream_;
  uint64_t flatSize_;
};

using CudfVectorPtr = std::shared_ptr<CudfVector>;

} // namespace facebook::velox::cudf_velox
