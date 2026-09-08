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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
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

std::unique_ptr<CudfSplitReader>
CudfIcebergDataSource::createBatchedSplitReader(
    const std::vector<
        std::shared_ptr<facebook::velox::connector::ConnectorSplit>>& batch,
    dwio::common::RuntimeStats& runtimeStats) {
  if (!useExperimentalCudfReader_) {
    return nullptr;
  }

  std::vector<std::string> paths;
  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> first;
  for (const auto& child : batch) {
    auto split =
        std::dynamic_pointer_cast<const velox_iceberg::HiveIcebergSplit>(child);
    if (!split || !split->deleteFiles.empty() ||
        !split->partitionKeys.empty() ||
        !split->identityPartitionKeys.empty() || split->start != 0 ||
        split->fileFormat != dwio::common::FileFormat::PARQUET ||
        split->tableBucketNumber || split->bucketConversion ||
        split->rowIdProperties || split->extraFileInfo ||
        split->batchSizeHint != 0 || !split->cacheable || split->properties) {
      return nullptr;
    }
    // Per-file information columns cannot be injected as one batch constant.
    for (const auto& name : readColumnNames_) {
      if (split->infoColumns.contains(name)) {
        return nullptr;
      }
    }
    if (!first) {
      first = split;
    } else if (
        split->customSplitInfo != first->customSplitInfo ||
        split->columnMappingMode != first->columnMappingMode) {
      return nullptr;
    }
    paths.push_back(split->filePath);
  }

  icebergSplit_ = first;
  split_ = CudfHiveConnectorSplit::makeBatch(first->connectorId, paths, 0);
  auto reader = createCudfSplitReader();
  if (!checkedPointerCast<CudfIcebergSplitReader>(reader.get())
           ->tryPrepareBatch(batch, runtimeStats)) {
    return nullptr;
  }
  return reader;
}

void CudfIcebergDataSource::convertSplit(
    std::shared_ptr<velox_connector::ConnectorSplit> split) {
  // Convert `ConnectorSplit` to `HiveIcebergSplit`
  icebergSplit_ =
      checkedPointerCast<const velox_iceberg::HiveIcebergSplit>(split);

  // Convert `ConnectorSplit` to `CudfHiveConnectorSplit`
  CudfHiveDataSource::convertSplit(split);
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
      subfieldFilterExpr_);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
