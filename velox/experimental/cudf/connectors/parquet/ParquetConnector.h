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

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"

#include "velox/connectors/Connector.h"

namespace facebook::velox::cudf_velox::connector::parquet {

using namespace facebook::velox::connector;
using namespace facebook::velox::config;

class ParquetConnector final : public Connector {
 public:
  ParquetConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* executor);

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override final;

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr inputType,
      ConnectorInsertTableHandlePtr connectorInsertTableHandle,
      ConnectorQueryCtx* connectorQueryCtx,
      CommitStrategy commitStrategy) override final;

  folly::Executor* executor() const override {
    return executor_;
  }

 protected:
  const std::shared_ptr<ParquetConfig> parquetConfig_;
  folly::Executor* executor_;
};

class ParquetConnectorFactory : public ConnectorFactory {
 public:
  static constexpr const char* kParquetConnectorName = "parquet";

  ParquetConnectorFactory() : ConnectorFactory(kParquetConnectorName) {}

  explicit ParquetConnectorFactory(const char* connectorName)
      : ConnectorFactory(connectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
