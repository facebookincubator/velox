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

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

/// Iceberg-specific data source that extends CudfHiveDataSource.
///
/// Provides Iceberg table format support by creating CudfIcebergSplitReader
/// instances that handle:
/// - GPU-accelerated positional delete files and vectors for row-level deletes.
/// - GPU-accelerated equality delete files for column-level deletes.
/// - Schema evolution with column adaptation.
/// - Iceberg-specific metadata columns.
class CudfIcebergDataSource : public CudfHiveDataSource {
 public:
  CudfIcebergDataSource(
      const RowTypePtr& outputType,
      const velox_connector::ConnectorTableHandlePtr& tableHandle,
      const velox_connector::ColumnHandleMap& columnHandles,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig);

 protected:
  /// Override to create `CudfIcebergSplitReader` instead of
  /// `CudfHiveSplitReader`
  std::unique_ptr<CudfSplitReader> createCudfSplitReader() override;

  /// Override to convert ConnectorSplit to `HiveIcebergSplit` and then to
  /// `CudfHiveConnectorSplit`
  void convertSplit(std::shared_ptr<ConnectorSplit> split) override;

 private:
  std::shared_ptr<const velox_hive::HiveConfig> hiveConfig_;
  std::shared_ptr<const velox_iceberg::HiveIcebergSplit> icebergSplit_;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
