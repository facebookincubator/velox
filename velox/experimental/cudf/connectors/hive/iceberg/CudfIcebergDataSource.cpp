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

#include "velox/common/Casts.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
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

void CudfIcebergDataSource::convertSplit(
    std::shared_ptr<velox_connector::ConnectorSplit> split) {
  // Convert ConnectorSplit to `HiveIcebergSplit`
  icebergSplit_ =
      std::dynamic_pointer_cast<const velox_iceberg::HiveIcebergSplit>(split);

  if (!icebergSplit_) {
    auto hiveSplit = checkedPointerCast<velox_hive::HiveConnectorSplit>(split);

    VELOX_CHECK(
        hiveSplit->fileFormat == dwio::common::FileFormat::PARQUET,
        "CudfIcebergDataSource only supports PARQUET format.");
    icebergSplit_ = std::make_shared<velox_iceberg::HiveIcebergSplit>(
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
  }

  VLOG(1) << "Converted split to HiveIcebergSplit: "
          << icebergSplit_->toString();

  // Convert `HiveIcebergSplit` to `CudfHiveConnectorSplit`
  CudfHiveDataSource::convertSplit(
      std::const_pointer_cast<velox_iceberg::HiveIcebergSplit>(icebergSplit_));
}

std::unique_ptr<CudfSplitReader>
CudfIcebergDataSource::createCudfSplitReader() {
  return std::make_unique<CudfIcebergSplitReader>(
      split_,
      icebergSplit_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      hiveConfig_,
      ioStatistics_,
      ioStats_,
      useExperimentalCudfReader_,
      subfieldFilterExpr_,
      &remainingFilterExprSet_,
      cudfExpressionEvaluator_,
      &totalRemainingFilterTime_);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
