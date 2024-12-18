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
#include <memory>

#include "velox/experimental/cudf/connectors/parquet/ParquetConnectorSplit.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetReaderConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"

#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

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
    uint64_t /* size */,
    velox::ContinueFuture& /* future */) {
  VELOX_CHECK(split_ != nullptr, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(splitReader_, "No split reader present");

  // TODO: Implement a cudf::partition and cudf::concatenate based algorithm to
  // cater for `size` argument

  // TODO: MH: Enable this some other way
  // if (splitReader_->emptySplit()) {
  //  resetSplit();
  //  return nullptr;
  //}

  // cudf parquet reader returns has_next() = true if no chunk has yet been
  // read.
  if (splitReader_->has_next()) {
    // Read a chunk of table.
    auto [table, metadata] = splitReader_->read_chunk();

    // Check if the chunk is empty
    const auto rowsScanned = table->num_rows();
    if (rowsScanned == 0) {
      // TODO: Update runtime stats here
      return nullptr;
    }

    // update completedRows
    completedRows_ += table->num_rows();

    // TODO: Get `completedBytes_` from elsewhere instead of this hacky method
    const auto& filePaths = split_->getCudfSourceInfo().filepaths();
    for (const auto& filePath : filePaths) {
      completedBytes_ += std::filesystem::file_size(filePath);
    }

    // Convert to velox RowVectorPtr with_arrow to support more rowTypes
    RowVectorPtr output =
        with_arrow::to_velox_column(table->view(), pool_, "c");

    // Return output
    return output;

  } else {
    return nullptr;
  }
}

void ParquetDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  split_ = std::dynamic_pointer_cast<ParquetConnectorSplit>(split);

  VLOG(1) << "Adding split " << split_->toString();

  // Split reader already exists, reset
  if (splitReader_) {
    splitReader_.reset();
  }

  splitReader_ = createSplitReader();
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
  // Simply reset the split and the reader
  split_.reset();
  splitReader_.reset();
}

} // namespace facebook::velox::cudf_velox::connector::parquet
