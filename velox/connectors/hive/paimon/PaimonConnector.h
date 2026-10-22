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
#include "velox/connectors/hive/paimon/PaimonConfig.h"

namespace facebook::velox::connector::hive::paimon {

/// Provides Paimon table format support by extending HiveConnector.
///
/// Creates PaimonDataSource instances that handle Paimon's multi-file splits
/// (one split = one bucket with multiple data files across LSM-tree levels).
/// Reuses HiveConnector's ORC/Parquet readers directly — no Arrow bridge.
class PaimonConnector final : public HiveConnector {
 public:
  PaimonConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor);

  /// Creates PaimonDataSource. Requires a PaimonTableHandle belonging to this
  /// connector instance.
  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override;

  bool supportsSplitPreload() const override {
    return false;
  }

  bool supportsIndexLookup() const override {
    return false;
  }

  /// Paimon writing and index lookup have no execution implementation.
  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      ConnectorInsertTableHandlePtr connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) override;

  std::shared_ptr<IndexSource> createIndexSource(
      const RowTypePtr& inputType,
      const std::vector<std::shared_ptr<core::IndexLookupCondition>>&
          joinConditions,
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override;

  /// Registers Paimon table/split serde and the reused HiveColumnHandle. Type,
  /// Filter and expression serde are registered by the query plan owner.
  static void registerSerDe();

 private:
  const std::shared_ptr<PaimonConfig> paimonConfig_;
};

class PaimonConnectorFactory final : public ConnectorFactory {
 public:
  static constexpr const char* kPaimonConnectorName = "paimon";

  PaimonConnectorFactory() : ConnectorFactory(kPaimonConnectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      [[maybe_unused]] folly::Executor* cpuExecutor = nullptr) override {
    return std::make_shared<PaimonConnector>(id, config, ioExecutor);
  }
};

} // namespace facebook::velox::connector::hive::paimon
