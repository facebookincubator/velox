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

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <folly/Executor.h>
#include <memory>
#include <vector>
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestData.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace facebook::velox::cudf_exchange {

static const uint64_t FOUR_GBYTES = 4294967296;

/*const std::vector<std::string> kTestColumnNames = {"c0", "c1","c2"};
const std::vector<TypePtr> kTestColumnTypes = {INTEGER(), DOUBLE(),VARCHAR()};
const facebook::velox::RowTypePtr kTestRowType =
    ROW(kTestColumnNames, kTestColumnTypes);*/

/// @brief Helper function to create a source Task for testing purposes.
/// Creates a simple task associated with a plan fragment that consists of a
/// single value node. The output type is the given row type. Memory pool is
/// only needed to initialize a value node in the plan fragment.
///
/// @param taskId The unique identifier for the task
/// @param pool Shared pointer to the memory pool to use by the value source
/// node.
/// @param rowType The row type to use for the task
/// @return Shared pointer to the created Task
std::shared_ptr<facebook::velox::exec::Task> createSourceTask(
    const std::string& taskId,
    std::shared_ptr<facebook::velox::memory::MemoryPool> pool,
    facebook::velox::RowTypePtr rowType,
    uint64_t kMaxOutputBufferSize = FOUR_GBYTES);

/// @brief Helper function to create a sink task for testing.
/// Creates a simple task associated with a plan fragment that consists fo a
/// single exchange node. The input type is the given row type.
/// @param taskId The unique identifier for the task
/// @param rowType The row type to use for the task
/// @param partitionId The partition that this task is retrieving from upstream
/// task(s).
/// @param exchangeNodeId Reference parameter that returns the id of the
/// exchange node in this task's plan fragment.
/// @return Shared pointer to the created Task
std::shared_ptr<facebook::velox::exec::Task> createExchangeTask(
    const std::string& taskId,
    facebook::velox::RowTypePtr rowType,
    int partitionId,
    core::PlanNodeId& exchangeNodeId);

/// @brief Helper function to create a source task with PartitionedOutput for
/// testing. Creates a task with a plan fragment: Values -> PartitionedOutput.
/// This is used by SourceDriverMock to drive real CudfPartitionedOutput
/// operators.
/// @param taskId The unique identifier for the task
/// @param pool Shared pointer to the memory pool
/// @param rowType The row type to use for the task
/// @param numPartitions The number of output partitions
/// @param partitionKeys The keys to use for hash partitioning (empty for
/// round-robin)
/// @param kMaxOutputBufferSize Maximum output buffer size
/// @return Shared pointer to the created Task
std::shared_ptr<facebook::velox::exec::Task> createPartitionedOutputTask(
    const std::string& taskId,
    std::shared_ptr<facebook::velox::memory::MemoryPool> pool,
    facebook::velox::RowTypePtr rowType,
    int numPartitions,
    const std::vector<std::string>& partitionKeys = {},
    uint64_t kMaxOutputBufferSize = FOUR_GBYTES);

/// @brief Helper function to create a CudfVector for testing.
/// Uses makeTable when tableGenerator is null, or tableGenerator->makeTable()
/// when provided.
/// @param pool The memory pool to use for the CudfVector
/// @param numRows Number of rows to create
/// @param rowType The row type for the vector
/// @param tableGenerator Optional table generator to create the table data
/// @param stream The CUDA stream to use
/// @return Shared pointer to the created CudfVector
std::shared_ptr<facebook::velox::cudf_velox::CudfVector> makeCudfVector(
    facebook::velox::memory::MemoryPool* pool,
    size_t numRows,
    facebook::velox::RowTypePtr rowType,
    std::shared_ptr<BaseTableGenerator> tableGenerator,
    rmm::cuda_stream_view stream);

/// Helper function to create cudf::table for testing.
/// Creates a table with columns based on the given rowType.
///
/// @param numRows Number of rows to create in the table
/// @param rowType The row type specifying the columns
/// @param stream The CUDA stream to use
/// @return Unique pointer to the created table
///
std::unique_ptr<cudf::table> makeTable(
    std::size_t numRows,
    facebook::velox::RowTypePtr rowType,
    rmm::cuda_stream_view stream);

/// @brief testing utility for dumping the contents of a string column.
/// @param str_column_view The string column view to be dumped.
/// @param stream The cuda stream.
std::vector<std::string> getStringCol(
    const cudf::strings_column_view& str_column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream);

/// @brief Helper function to create a strings column from a vector of host
/// strings.
/// @param host_strings The vector of strings to use for creating the column.
/// @return A unique pointer to the created strings column.
std::unique_ptr<cudf::column> make_strings_column_from_host(
    const std::vector<std::string>& host_strings);

/// @brief Template function for retrieving the contents of a fixed-size column.
/// @param column_view The column view to be dumped.
/// @param max_rows The maximum number of rows to be retrieved.
/// @param stream The cude stream.

template <typename T>
std::vector<T> getColVector(
    const cudf::column_view& column_view,
    cudf::size_type max_rows,
    rmm::cuda_stream_view stream) {
  max_rows = column_view.size() < max_rows ? column_view.size() : max_rows;
  const T* ptr_data = column_view.template data<T>();
  auto host_vec = cudf::detail::make_host_vector_async(
      cudf::device_span<T const>(ptr_data, max_rows), stream);
  std::vector<T> vec(max_rows);
  std::copy(host_vec.begin(), host_vec.end(), vec.begin());
  return vec;
}

} // namespace facebook::velox::cudf_exchange
