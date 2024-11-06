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

#pragma once

#include "velox/common/memory/Memory.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

std::unique_ptr<cudf::table> to_cudf_table(
    const facebook::velox::RowVectorPtr& leftBatch);
facebook::velox::VectorPtr to_velox_column(
    const cudf::column_view& col,
    facebook::velox::memory::MemoryPool* pool);
facebook::velox::RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string name_prefix = "c");

namespace with_arrow {
std::unique_ptr<cudf::table> to_cudf_table(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool);

facebook::velox::RowVectorPtr to_velox_column(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string name_prefix);
} // namespace with_arrow

} // namespace facebook::velox::cudf_velox
