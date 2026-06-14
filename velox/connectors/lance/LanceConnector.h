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
#include "velox/connectors/lance/LanceColumnHandle.h"
#include "velox/connectors/lance/LanceConnectorSplit.h"
#include "velox/connectors/lance/LanceTableHandle.h"
#include "velox/connectors/lance/lance_ffi.h"

namespace facebook::velox::connector::lance {

/// Reads data from a Lance dataset via the Lance FFI bridge.
class LanceDataSource : public DataSource {
 public:
  LanceDataSource(
      const RowTypePtr& outputType,
      const connector::ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx);

  ~LanceDataSource() override;

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<common::Filter>& /*filter*/) final {
    VELOX_NYI("Dynamic filters not supported by LanceConnector.");
  }

  /// Returns the next batch of rows. Returns nullopt when the split is
  /// exhausted.
  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& future) override;

  /// Returns the number of rows read so far.
  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  /// Returns the number of bytes read so far.
  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

 private:
  // Re-orders imported Arrow columns to match the expected output schema.
  RowVectorPtr projectOutputColumns(RowVectorPtr vector);

  // Closes and frees the active scanner and dataset handles.
  void closeScannerAndDataset();

  RowTypePtr outputType_;
  // Names of columns to project from the Lance dataset, in output order.
  std::vector<std::string> columnNames_;

  // Open Lance dataset handle; null when no split is active.
  LanceDataset* dataset_{nullptr};
  // Active scanner handle; null when no split is active.
  LanceScanner* scanner_{nullptr};
  bool splitProcessed_{true};

  uint64_t completedRows_{0};
  uint64_t completedBytes_{0};

  memory::MemoryPool* pool_;
};

/// Velox connector for reading Lance datasets.
class LanceConnector final : public Connector {
 public:
  LanceConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor);

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) final;

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      ConnectorInsertTableHandlePtr connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) final;
};

/// Factory for creating LanceConnector instances.
class LanceConnectorFactory : public ConnectorFactory {
 public:
  static constexpr const char* kLanceConnectorName = "lance";

  LanceConnectorFactory() : ConnectorFactory(kLanceConnectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override;
};

} // namespace facebook::velox::connector::lance
