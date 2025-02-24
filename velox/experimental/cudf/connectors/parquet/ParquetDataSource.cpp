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

#include <filesystem>
#include <limits>
#include <memory>
#include <string>

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

namespace facebook::velox::cudf_velox::connector::parquet {

using namespace facebook::velox::connector;

ParquetDataSource::ParquetDataSource(
    const std::shared_ptr<const RowType>& outputType,
    const std::shared_ptr<ConnectorTableHandle>& tableHandle,
    const std::unordered_map<std::string, std::shared_ptr<ColumnHandle>>&
        columnHandles,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<ParquetConfig>& ParquetConfig)
    : ParquetConfig_(ParquetConfig),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      pool_(connectorQueryCtx->memoryPool()),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const ParquetColumnHandle*>(it->second.get());
    readColumnNames_.emplace_back(handle->name());
  }

  // Dynamic cast tableHandle to ParquetTableHandle
  tableHandle_ = std::dynamic_pointer_cast<ParquetTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of ParquetTableHandle");

  // Create empty IOStats for later use
  ioStats_ = std::make_shared<io::IoStatistics>();

  // Create remaining filter
  auto remainingFilter = tableHandle_->remainingFilter();
  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    // auto& remainingFilterExpr = remainingFilterExprSet_->expr(0);
    // Get column names and subfields from remaining filter? required?
  }
}

std::optional<RowVectorPtr> ParquetDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /* future */) {
  nvtx3::scoped_range r{std::string("ParquetDataSource::") + __func__};
  // Basic sanity checks
  VELOX_CHECK_NOT_NULL(split_, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(splitReader_, "No split reader present");

  if (not splitReader_->has_next()) {
    return nullptr;
  }

  // Vector to store read tables
  auto readTables = std::vector<std::unique_ptr<cudf::table>>{};
  // Read chunks until num_rows > size or no more chunks left.
  if (splitReader_->has_next()) {
    auto [table, metadata] = splitReader_->read_chunk();
    cudfTable_ = std::move(table);
    // Fill in the column names if reading the first chunk.
    if (columnNames.empty()) {
      for (auto schema : metadata.schema_info) {
        columnNames.emplace_back(schema.name);
      }
    }
  }

  // cudfTable_ = concatenateTables(std::move(readTables));
  auto stream = cudfGlobalStreamPool().get_stream();

  // Apply remaining filter if present
  if (remainingFilterExprSet_) {
    auto cudf_table_columns = cudfTable_->release();
    auto const original_num_columns = cudf_table_columns.size();
    auto& remainingFilterExpr = remainingFilterExprSet_->expr(0);
    std::vector<std::unique_ptr<cudf::scalar>> scalars_;
    std::vector<PrecomputeInstruction> precompute_instructions_;
    cudf::ast::tree tree;
    create_ast_tree(
        remainingFilterExpr,
        tree,
        scalars_,
        outputType_,
        precompute_instructions_);
    addPrecomputedColumns(
        cudf_table_columns, precompute_instructions_, scalars_, stream);
    cudfTable_ = std::make_unique<cudf::table>(std::move(cudf_table_columns));
    auto cudf_table_view = cudfTable_->view();
    std::unique_ptr<cudf::column> col;
    if (auto col_ref_ptr =
            dynamic_cast<cudf::ast::column_reference const*>(&tree.back())) {
      col = std::make_unique<cudf::column>(
          cudf_table_view.column(col_ref_ptr->get_column_index()),
          stream,
          cudf::get_current_device_resource_ref());
    } else {
      col = cudf::compute_column(
          cudf_table_view,
          tree.back(),
          stream,
          cudf::get_current_device_resource_ref());
    }
    std::vector<std::unique_ptr<cudf::column>> original_columns;
    original_columns.reserve(original_num_columns);
    cudf_table_columns = cudfTable_->release();
    for (size_t i = 0; i < original_num_columns; ++i) {
      original_columns.push_back(std::move(cudf_table_columns[i]));
    }
    auto original_table =
        std::make_unique<cudf::table>(std::move(original_columns));
    cudfTable_ = cudf::apply_boolean_mask(
        *original_table, *col, stream, cudf::get_current_device_resource_ref());
  }

  // Output RowVectorPtr
  currentCudfTableView_ = cudfTable_->view();
  auto sz = cudfTable_->num_rows();
  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, sz, std::move(cudfTable_), stream)
      : with_arrow::to_velox_column(
            currentCudfTableView_, pool_, columnNames, stream);
  stream.synchronize();

  // Reset internal tables
  resetCudfTableAndView();

  // Check if conversion yielded a nullptr
  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  // Update completedRows_.
  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

void ParquetDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  // Dynamic cast split to `ParquetConnectorSplit`
  split_ = std::dynamic_pointer_cast<ParquetConnectorSplit>(split);
  VLOG(1) << "Adding split " << split_->toString();

  // Split reader already exists, reset
  if (splitReader_) {
    splitReader_.reset();
  }

  // Clear columnNames if not empty
  if (not columnNames.empty()) {
    columnNames.clear();
  }

  // Reset cudfTable and views if not already reset
  if (cudfTable_) {
    resetCudfTableAndView();
  }

  // Create a `cudf::io::chunked_parquet_reader` SplitReader
  splitReader_ = createSplitReader();

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  const auto& filePaths = split_->getCudfSourceInfo().filepaths();
  for (const auto& filePath : filePaths) {
    completedBytes_ += std::filesystem::file_size(filePath);
  }
}

std::unique_ptr<cudf::io::chunked_parquet_reader>
ParquetDataSource::createSplitReader() {
  // Reader options
  auto readerOptions =
      cudf::io::parquet_reader_options::builder(split_->getCudfSourceInfo())
          .skip_rows(ParquetConfig_->skipRows())
          .use_pandas_metadata(ParquetConfig_->isUsePandasMetadata())
          .use_arrow_schema(ParquetConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              ParquetConfig_->isAllowMismatchedParquetSchemas())
          .timestamp_type(ParquetConfig_->timestampType())
          .build();

  // Set num_rows only if available
  if (ParquetConfig_->numRows().has_value()) {
    readerOptions.set_num_rows(ParquetConfig_->numRows().value());
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions.set_columns(readColumnNames_);
  }

  // Create a parquet reader
  return std::make_unique<cudf::io::chunked_parquet_reader>(
      ParquetConfig_->maxChunkReadLimit(),
      ParquetConfig_->maxPassReadLimit(),
      readerOptions);
}

void ParquetDataSource::resetSplit() {
  split_.reset();
  splitReader_.reset();
  columnNames.clear();
}

void ParquetDataSource::resetCudfTableAndView() {
  cudfTable_.reset();
  currentCudfTableView_ = cudf::table_view{};
}

} // namespace facebook::velox::cudf_velox::connector::parquet
