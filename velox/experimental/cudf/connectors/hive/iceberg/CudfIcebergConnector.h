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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

using namespace facebook::velox::connector;
using namespace facebook::velox::config;

/// Provides GPU-accelerated Iceberg table format support.
/// - Creates CudfIcebergDataSource instances for reading Iceberg tables with
///   support for delete files. Schema evolution is not yet supported.
class CudfIcebergConnector final
    : public ::facebook::velox::connector::hive::HiveConnector {
 public:
  CudfIcebergConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* executor);

  /// Creates a CudfIcebergDataSource when reading from Iceberg tables when cudf
  /// is registered, otherwise falls back to the base IcebergDataSource.
  ///
  /// @param outputType The schema of the output data to read.
  /// @param tableHandle The table handle containing table metadata.
  /// @param columnHandles Map of column names to column handles.
  /// @param connectorQueryCtx Query context for the read operation.
  /// @return IcebergDataSource instance configured for the read operation.
  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override;

  bool canAddDynamicFilter() const override {
    return false;
  }

  bool supportsSplitPreload() const override {
    return false;
  }

 private:
  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  const std::shared_ptr<velox::connector::hive::iceberg::IcebergConfig>
      icebergConfig_;
};

/// Factory for creating CudfIcebergConnector instances.
class CudfIcebergConnectorFactory final : public ConnectorFactory {
 public:
  static constexpr const char* kCudfIcebergConnectorName = "cudf-iceberg";

  CudfIcebergConnectorFactory() : ConnectorFactory(kCudfIcebergConnectorName) {}

  explicit CudfIcebergConnectorFactory(const char* connectorName)
      : ConnectorFactory(connectorName) {}

  /// Creates a new CudfIcebergConnector instance.
  ///
  /// @param id Unique identifier for this connector instance (typically the
  /// catalog name).
  /// @param config Connector configuration properties
  /// @param ioExecutor Optional executor for asynchronous I/O operations such
  /// as split preloading and file prefetching. When provided, enables
  /// background file operations off the main driver thread. If nullptr, I/O
  /// operations run synchronously.
  /// @param cpuExecutor ConnectorFactory interface to support other connector
  /// types that may need CPU-bound async work. Currently unused by
  /// IcebergConnector.
  /// @return Shared pointer to the newly created IcebergConnector instance
  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
