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

namespace facebook::velox::connector::hive::iceberg {

/// TODO Add IcebergConfig class and Move these configuration properties to
/// IcebergConfig.h
extern const std::string_view kIcebergFunctionPrefixConfig;
extern const std::string_view kDefaultIcebergFunctionPrefix;

/// Provides Iceberg table format support.
/// - Creates HiveDataSource instances that use IcebergSplitReader for reading
///   Iceberg tables with support for delete files and schema evolution.
/// - Creates IcebergDataSink instances for writing data with Iceberg-specific
///   partition transforms and commit metadata.
class IcebergConnector final : public HiveConnector {
 public:
  IcebergConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor);

  /// Creates IcebergDataSink for writing to Iceberg tables.
  ///
  /// @param inputType The schema of the input data to write.
  /// @param connectorInsertTableHandle Must be an IcebergInsertTableHandle
  ///        containing Iceberg-specific write configuration.
  /// @param connectorQueryCtx Query context for the write operation.
  /// @param commitStrategy Strategy for committing the write operation. Only
  ///        CommitStrategy::kNoCommit is supported for Iceberg tables. Files
  ///        are written directly with their final names and commit metadata is
  ///        returned for the coordinator to update the Iceberg metadata tables.
  /// @return IcebergDataSink instance configured for the write operation.
  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      ConnectorInsertTableHandlePtr connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) override;

 private:
  const std::string functionPrefix_;
};

class IcebergConnectorFactory final : public ConnectorFactory {
 public:
  static constexpr const char* kIcebergConnectorName = "iceberg";

  IcebergConnectorFactory() : ConnectorFactory(kIcebergConnectorName) {}

  /// Creates a new IcebergConnector instance.
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
      [[maybe_unused]] folly::Executor* cpuExecutor = nullptr) override {
    return std::make_shared<IcebergConnector>(id, config, ioExecutor);
  }
};

} // namespace facebook::velox::connector::hive::iceberg
