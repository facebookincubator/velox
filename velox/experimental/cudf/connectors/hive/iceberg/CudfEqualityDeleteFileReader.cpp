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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfEqualityDeleteFileReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/ReaderFactory.h"

#include <cudf/io/parquet.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/types.hpp>

#include <limits>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

CudfEqualityDeleteFileReader::CudfEqualityDeleteFileReader(
    const velox_iceberg::IcebergDeleteFile& deleteFile,
    const std::vector<std::string>& equalityColumnNames,
    const std::vector<TypePtr>& equalityColumnTypes,
    const std::string& /*baseFilePath*/,
    FileHandleFactory* fileHandleFactory,
    const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
    const std::shared_ptr<::facebook::velox::io::IoStatistics>& ioStatistics,
    const std::shared_ptr<::facebook::velox::IoStats>& ioStats,
    ::facebook::velox::dwio::common::RuntimeStatistics& runtimeStats,
    const std::string& connectorId)
    : equalityColumnNames_(equalityColumnNames) {
  VELOX_CHECK(
      deleteFile.content == velox_iceberg::FileContent::kEqualityDeletes,
      "Expected equality delete file but got content type: {}",
      static_cast<int>(deleteFile.content));
  VELOX_CHECK_GT(deleteFile.recordCount, 0, "Empty equality delete file.");
  VELOX_CHECK(
      !equalityColumnNames_.empty(),
      "Equality delete file must specify at least one column.");
  VELOX_CHECK_EQ(
      equalityColumnNames.size(),
      equalityColumnTypes.size(),
      "Equality column names and types must have the same size.");

  pool_ = connectorQueryCtx->memoryPool();

  auto deleteFileSchema =
      ROW(std::vector<std::string>(equalityColumnNames),
          std::vector<TypePtr>(equalityColumnTypes));

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  for (size_t i = 0; i < equalityColumnNames.size(); ++i) {
    scanSpec->addField(equalityColumnNames[i], static_cast<int>(i));
  }

  auto deleteSplit = std::make_shared<velox_hive::HiveConnectorSplit>(
      connectorId,
      deleteFile.filePath,
      deleteFile.fileFormat,
      0,
      deleteFile.fileSizeInBytes);

  dwio::common::ReaderOptions deleteReaderOpts(pool_);
  velox_hive::configureReaderOptions(
      hiveConfig,
      connectorQueryCtx,
      deleteFileSchema,
      deleteSplit,
      /*tableParameters=*/{},
      deleteReaderOpts);

  const FileHandleKey fileHandleKey{
      .filename = deleteFile.filePath,
      .tokenProvider = connectorQueryCtx->fsTokenProvider()};
  auto deleteFileHandleCachePtr = fileHandleFactory->generate(fileHandleKey);
  auto deleteFileInput =
      velox_hive::BufferedInputBuilder::getInstance()->create(
          *deleteFileHandleCachePtr,
          deleteReaderOpts,
          connectorQueryCtx,
          ioStatistics,
          ioStats,
          executor);

  // Directly read the Parquet-format equality delete file to the
  // deleteKeyTable_ using cuDF
  if (deleteFile.fileFormat == dwio::common::FileFormat::PARQUET) {
    directReadEqualityDeleteFile(deleteFile, std::move(deleteFileInput));
    return;
  }

  auto deleteReader =
      dwio::common::getReaderFactory(deleteReaderOpts.fileFormat())
          ->createReader(std::move(deleteFileInput), deleteReaderOpts);

  if (!velox_hive::testFilters(
          scanSpec.get(),
          deleteReader.get(),
          deleteSplit->filePath,
          deleteSplit->partitionKeys,
          {},
          hiveConfig->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx->sessionProperties()))) {
    runtimeStats.skippedSplitBytes += static_cast<int64_t>(deleteSplit->length);
    return;
  }

  dwio::common::RowReaderOptions deleteRowReaderOpts;
  velox_hive::configureRowReaderOptions(
      {},
      scanSpec,
      nullptr,
      deleteFileSchema,
      deleteSplit,
      nullptr,
      nullptr,
      nullptr,
      deleteRowReaderOpts);

  auto deleteRowReader = deleteReader->createRowReader(deleteRowReaderOpts);

  // Read the entire delete file in a single batch.
  VectorPtr output = BaseVector::create(deleteFileSchema, 0, pool_);
  auto rowsRead =
      deleteRowReader->next(std::numeric_limits<uint64_t>::max(), output);
  if (rowsRead == 0 or output->size() == 0) {
    return;
  }
  output->loadedVector();
  deleteRows_ = std::dynamic_pointer_cast<RowVector>(output);
  VELOX_CHECK_NOT_NULL(deleteRows_);
  numDeleteKeys_ = deleteRows_->size();
}

void CudfEqualityDeleteFileReader::directReadEqualityDeleteFile(
    const velox_iceberg::IcebergDeleteFile& deleteFile,
    std::shared_ptr<dwio::common::BufferedInput> bufferedInput) {
  using cudf_velox::connector::hive::BufferedInputDataSource;

  // Create a cuDF data source
  std::shared_ptr<cudf::io::datasource> dataSource;
  auto sourceInfo = [&]() {
    if (bufferedInput) {
      dataSource =
          std::make_shared<BufferedInputDataSource>(std::move(bufferedInput));
      return cudf::io::source_info{dataSource.get()};
    }
    return cudf::io::source_info{deleteFile.filePath};
  }();

  // Read the equality delete file
  auto options =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo)).build();
  options.set_column_names(equalityColumnNames_);
  auto stream = cudfGlobalStreamPool().get_stream();
  deleteKeyTable_ =
      cudf::io::read_parquet(options, stream, get_output_mr()).tbl;
  stream.synchronize();

  VELOX_CHECK_NOT_NULL(deleteKeyTable_);
  numDeleteKeys_ = deleteKeyTable_->num_rows();
}

void CudfEqualityDeleteFileReader::buildHashJoin(rmm::cuda_stream_view stream) {
  if (deleteHashJoin_) {
    return;
  }

  // Convert host rows to a GPU table if we came through the non-Parquet path.
  if (!deleteKeyTable_) {
    VELOX_CHECK_NOT_NULL(deleteRows_);
    deleteKeyTable_ =
        with_arrow::toCudfTable(deleteRows_, pool_, stream, get_temp_mr());
    VELOX_CHECK_NOT_NULL(deleteKeyTable_);
    deleteRows_.reset();
  }

  // null_equality::EQUAL per Iceberg spec (NULL == NULL for equality deletes).
  // distinct_hash_join treats all NaNs as equal by default.
  deleteHashJoin_ = std::make_unique<cudf::distinct_hash_join>(
      deleteKeyTable_->view(), cudf::null_equality::EQUAL, 0.5, stream);
}

void CudfEqualityDeleteFileReader::buildEqualityColumnIndices(
    const std::vector<std::string>& inputColumnNames) {
  if (not equalityColumnIndices_.empty()) {
    return;
  }

  equalityColumnIndices_.reserve(equalityColumnNames_.size());
  for (const auto& eqColName : equalityColumnNames_) {
    bool found = false;
    for (cudf::size_type i = 0;
         i < static_cast<cudf::size_type>(inputColumnNames.size());
         ++i) {
      if (inputColumnNames[i] == eqColName) {
        equalityColumnIndices_.push_back(i);
        found = true;
        break;
      }
    }
    VELOX_CHECK(
        found,
        "Equality delete column not found in input table: {}",
        eqColName);
  }
}

void CudfEqualityDeleteFileReader::applyDeletes(
    cudf::table_view table,
    const std::vector<std::string>& inputColumnNames,
    cudf::mutable_column_view const& deleteMask,
    rmm::cuda_stream_view stream) {
  const auto numRows = table.num_rows();
  if (empty() or numRows == 0) {
    return;
  }

  // Lazily build hash join and equality column indices if needed
  buildHashJoin(stream);
  buildEqualityColumnIndices(inputColumnNames);

  // Use hash join to find probe indices that match any delete key.
  // inner_join returns (probeIndices, buildindices); we only need the probe
  // side to know which data rows to delete.
  const auto probeTable = table.select(equalityColumnIndices_);
  const auto probeIndices =
      deleteHashJoin_->inner_join(probeTable, stream, get_temp_mr()).first;

  // Clear `deleteMask` at probe positions if any
  if (not probeIndices->is_empty()) {
    scatterDeletesToMask(deleteMask, *probeIndices, stream, get_temp_mr());
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
