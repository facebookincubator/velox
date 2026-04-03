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

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

using namespace facebook::velox::connector;
using namespace facebook::velox::config;

/// GPU-accelerated Iceberg connector that delegates parquet data file reading
/// and deletion vector application to libcudf.
///
/// Derives from HiveConnector (since IcebergConnector is final) and mirrors
/// the Iceberg-specific setup (function registration, config) while overriding
/// createDataSource to produce a CudfIcebergDataSource that reads via the cudf
/// chunked_parquet_reader.
class CudfIcebergConnector final
    : public ::facebook::velox::connector::hive::HiveConnector {
 public:
  CudfIcebergConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* executor);

  /// Creates a CudfIcebergDataSource when cudf is registered, otherwise
  /// falls back to the base IcebergDataSource.
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

 protected:
  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  const std::shared_ptr<
      ::facebook::velox::connector::hive::iceberg::IcebergConfig>
      icebergConfig_;
};

/// Factory for creating CudfIcebergConnector instances.
class CudfIcebergConnectorFactory final : public ConnectorFactory {
 public:
  static constexpr const char* kCudfIcebergConnectorName = "cudf-iceberg";

  CudfIcebergConnectorFactory()
      : ConnectorFactory(kCudfIcebergConnectorName) {}

  explicit CudfIcebergConnectorFactory(const char* connectorName)
      : ConnectorFactory(connectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override;
};

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
