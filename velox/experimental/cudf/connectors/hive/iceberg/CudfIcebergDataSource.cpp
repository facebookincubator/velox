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
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDataSource.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"

#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

CudfIcebergDataSource::CudfIcebergDataSource(
    const RowTypePtr& outputType,
    const velox_connector::ConnectorTableHandlePtr& tableHandle,
    const velox_connector::ColumnHandleMap& columnHandles,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig)
    : CudfHiveDataSource(
          outputType,
          tableHandle,
          columnHandles,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig),
      hiveConfig_(hiveConfig) {}

CudfIcebergDataSource::~CudfIcebergDataSource() = default;

void CudfIcebergDataSource::addSplit(
    std::shared_ptr<velox_connector::ConnectorSplit> split) {
  auto icebergSplit =
      std::dynamic_pointer_cast<const velox_iceberg::HiveIcebergSplit>(split);

  if (!icebergSplit) {
    auto hiveSplit =
        std::dynamic_pointer_cast<velox_hive::HiveConnectorSplit>(split);
    if (hiveSplit) {
      VELOX_CHECK(
          hiveSplit->fileFormat == dwio::common::FileFormat::PARQUET,
          "CudfIcebergDataSource only supports PARQUET format.");
      icebergSplit =
          std::make_shared<velox_iceberg::HiveIcebergSplit>(
              hiveSplit->connectorId,
              hiveSplit->filePath,
              hiveSplit->fileFormat,
              hiveSplit->start,
              hiveSplit->length,
              hiveSplit->partitionKeys,
              hiveSplit->tableBucketNumber,
              hiveSplit->customSplitInfo,
              hiveSplit->extraFileInfo,
              hiveSplit->cacheable,
              std::vector<velox_iceberg::IcebergDeleteFile>{},
              hiveSplit->infoColumns,
              hiveSplit->properties);
    } else {
      VELOX_FAIL("Unsupported split type: {}", split->toString());
    }
  }

  VLOG(1) << "CudfIcebergDataSource: adding split "
          << icebergSplit->filePath;

  splitReader_ = std::make_unique<CudfIcebergSplitReader>(
      icebergSplit,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      hiveConfig_,
      ioStatistics_,
      ioStats_);
  splitReader_->prepareSplit();

  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = icebergSplit->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    auto const fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    if (fileHandleCachePtr.get() && fileHandleCachePtr.get()->file) {
      completedBytes_ += fileHandleCachePtr->file->size();
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to get file size for "
                 << icebergSplit->filePath << ": " << e.what();
  }
}

std::optional<RowVectorPtr> CudfIcebergDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /*future*/) {
  VELOX_CHECK_NOT_NULL(
      splitReader_, "No split to process. Call addSplit first.");
  auto result = splitReader_->next(size);
  if (!result.has_value()) {
    return RowVectorPtr(nullptr);
  }
  completedRows_ += result.value()->size();
  return result;
}

uint64_t CudfIcebergDataSource::getCompletedRows() {
  return completedRows_;
}

std::unordered_map<std::string, RuntimeMetric>
CudfIcebergDataSource::getRuntimeStats() {
  auto res = runtimeStats_.toRuntimeMetricMap();
  if (splitReader_) {
    auto splitStats = splitReader_->runtimeStats().toRuntimeMetricMap();
    res.insert(splitStats.begin(), splitStats.end());
  }
  res.insert(
      {std::string(velox_hive::HiveDataSource::kTotalScanTime),
       RuntimeMetric(
           ioStatistics_->totalScanTime(), RuntimeCounter::Unit::kNanos)});
  const auto& ioStatsMap = ioStats_->stats();
  for (const auto& storageStats : ioStatsMap) {
    res.emplace(storageStats.first, storageStats.second);
  }
  return res;
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
