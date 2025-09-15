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

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::dwio::common {
class DataSink;
class DataSource;
} // namespace facebook::velox::dwio::common

namespace facebook::velox::connector::hive {

class HiveConnector : public Connector {
 public:
  HiveConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor);

  bool canAddDynamicFilter() const override {
    return true;
  }

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override;

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
  bool supportsSplitPreload() override {
    return true;
  }
#else
  bool supportsSplitPreload() const override {
    return true;
  }
#endif

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      ConnectorInsertTableHandlePtr connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) override;

  folly::Executor* ioExecutor() const override {
    return ioExecutor_;
  }

  FileHandleCacheStats fileHandleCacheStats() {
    return fileHandleFactory_.cacheStats();
  }

  // NOTE: this is to clear file handle cache which might affect performance,
  // and is only used for operational purposes.
  FileHandleCacheStats clearFileHandleCache() {
    return fileHandleFactory_.clearCache();
  }

  static void registerSerDe();

 protected:
  const std::shared_ptr<HiveConfig> hiveConfig_;
  FileHandleFactory fileHandleFactory_;
  folly::Executor* ioExecutor_;
};

class HiveConnectorFactory : public ConnectorFactory {
 public:
  static constexpr const char* kHiveConnectorName = "hive";

  HiveConnectorFactory() : ConnectorFactory(kHiveConnectorName) {}

  explicit HiveConnectorFactory(const char* connectorName)
      : ConnectorFactory(connectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override {
    return std::make_shared<HiveConnector>(id, config, ioExecutor);
  }

  std::shared_ptr<connector::ConnectorSplit> makeConnectorSplit(
      const std::string& connectorId,
      const std::string& filePath,
      uint64_t start,
      uint64_t length,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<connector::ColumnHandle> makeColumnHandle(
      const std::string& connectorId,
      const std::string& name,
      const TypePtr& type,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<ConnectorTableHandle> makeTableHandle(
      const std::string& connectorId,
      const std::string& tableName,
      std::vector<std::shared_ptr<const connector::ColumnHandle>> columnHandles,
      const folly::dynamic& options) const override;

  std::shared_ptr<ConnectorInsertTableHandle> makeInsertTableHandle(
      const std::string& connectorId,
      std::vector<std::shared_ptr<const connector::ColumnHandle>> inputColumns,
      std::shared_ptr<const ConnectorLocationHandle> locationHandle,
      const folly::dynamic& options = {}) const override;

  std::shared_ptr<connector::ConnectorLocationHandle> makeLocationHandle(
      const std::string& connectorId,
      connector::ConnectorLocationHandle::TableType tableType =
          connector::ConnectorLocationHandle::TableType::kNew,
      const folly::dynamic& options = {}) const override;

  core::PartitionFunctionSpecPtr makePartitionFunctionSpec(
      const std::string& connectorId,
      const folly::dynamic& options = {}) const override;

 private:
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  dwio::common::FileFormat defaultFileFormat_{
      dwio::common::FileFormat::PARQUET};
};

class HivePartitionFunctionSpec : public core::PartitionFunctionSpec {
 public:
  HivePartitionFunctionSpec(
      int numBuckets,
      std::vector<int> bucketToPartition,
      std::vector<column_index_t> channels,
      std::vector<VectorPtr> constValues)
      : numBuckets_(numBuckets),
        bucketToPartition_(std::move(bucketToPartition)),
        channels_(std::move(channels)),
        constValues_(std::move(constValues)) {}

  /// The constructor without 'bucketToPartition' input is used in case that
  /// we don't know the actual number of partitions until we create the
  /// partition function instance. The hive partition function spec then builds
  /// a bucket to partition map based on the actual number of partitions with
  /// round-robin partitioning scheme to create the function instance. For
  /// instance, when we create the local partition node with hive bucket
  /// function to support multiple table writer drivers, we don't know the the
  /// actual number of table writer drivers until start the task.
  HivePartitionFunctionSpec(
      int numBuckets,
      std::vector<column_index_t> channels,
      std::vector<VectorPtr> constValues)
      : HivePartitionFunctionSpec(
            numBuckets,
            {},
            std::move(channels),
            std::move(constValues)) {}

  std::unique_ptr<core::PartitionFunction> create(
      int numPartitions,
      bool localExchange) const override;

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static core::PartitionFunctionSpecPtr deserialize(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 private:
  const int numBuckets_;
  const std::vector<int> bucketToPartition_;
  const std::vector<column_index_t> channels_;
  const std::vector<VectorPtr> constValues_;
};

} // namespace facebook::velox::connector::hive
