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

#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetReaderConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

namespace {

// Concatenate a vector of cuDF tables into a single table
std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables) {
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
      [&](auto const& tbl) { return tbl->view(); });
  return cudf::concatenate(tableViews, cudf::get_default_stream());
}

} // namespace

namespace facebook::velox::cudf_velox::connector::parquet {

using namespace facebook::velox::connector;

ParquetDataSource::ParquetDataSource(
    const std::shared_ptr<const RowType>& outputType,
    const std::shared_ptr<ConnectorTableHandle>& tableHandle,
    const std::unordered_map<std::string, std::shared_ptr<ColumnHandle>>&
        columnHandles,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<ParquetReaderConfig>& ParquetReaderConfig)
    : ParquetReaderConfig_(ParquetReaderConfig),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      pool_(connectorQueryCtx->memoryPool()),
      outputType_(outputType) {
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
}

std::optional<RowVectorPtr> ParquetDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /* future */) {
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
  currentCudfTableView_ = cudfTable_->view();

  // Output RowVectorPtr
  auto sz = cudfTable_->num_rows();
  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, sz, std::move(cudfTable_))
      : with_arrow::to_velox_column(currentCudfTableView_, pool_, columnNames);

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
          .skip_rows(ParquetReaderConfig_->skipRows())
          .use_pandas_metadata(ParquetReaderConfig_->isUsePandasMetadata())
          .use_arrow_schema(ParquetReaderConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              ParquetReaderConfig_->isAllowMismatchedParquetSchemas())
          .timestamp_type(ParquetReaderConfig_->timestampType())
          .build();

  // Set num_rows only if available
  if (ParquetReaderConfig_->numRows().has_value()) {
    readerOptions.set_num_rows(ParquetReaderConfig_->numRows().value());
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions.set_columns(readColumnNames_);
  }

  // Create a parquet reader
  return std::make_unique<cudf::io::chunked_parquet_reader>(
      ParquetReaderConfig_->maxChunkReadLimit(),
      ParquetReaderConfig_->maxPassReadLimit(),
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
