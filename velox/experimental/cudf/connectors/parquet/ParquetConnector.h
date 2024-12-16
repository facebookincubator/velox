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
#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox::connector::parquet {

class ParquetConnector final : public facebook::velox::connector::Connector {
 public:
  ParquetConnector(
      const std::string& id,
      std::shared_ptr<const facebook::velox::config::ConfigBase> config,
      folly::Executor* executor);

  std::unique_ptr<facebook::velox::connector::DataSource> createDataSource(
      const std::shared_ptr<const RowType>& outputType,
      const std::shared_ptr<facebook::velox::connector::ConnectorTableHandle>&
          tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<facebook::velox::connector::ColumnHandle>>&
          columnHandles,
      facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx)
      override final;

  const std::shared_ptr<const facebook::velox::config::ConfigBase>&
  connectorConfig() const override {
    return parquetConfig_->config();
  }

  std::unique_ptr<facebook::velox::connector::DataSink> createDataSink(
      RowTypePtr /*inputType*/,
      std::shared_ptr<
          facebook::velox::connector::
              ConnectorInsertTableHandle> /*connectorInsertTableHandle*/,
      facebook::velox::connector::ConnectorQueryCtx* /*connectorQueryCtx*/,
      facebook::velox::connector::CommitStrategy /*commitStrategy*/)
      override final {
    VELOX_NYI("cudf::ParquetConnector does not yet support data sink.");
  }

  folly::Executor* executor() const override {
    return executor_;
  }

 protected:
  const std::shared_ptr<ParquetConfig> parquetConfig_;
  folly::Executor* executor_;
};

class ParquetConnectorFactory
    : public facebook::velox::connector::ConnectorFactory {
 public:
  static constexpr const char* kParquetConnectorName = "parquet";

  ParquetConnectorFactory() : ConnectorFactory(kParquetConnectorName) {}

  explicit ParquetConnectorFactory(const char* connectorName)
      : ConnectorFactory(connectorName) {}

  std::shared_ptr<facebook::velox::connector::Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const facebook::velox::config::ConfigBase> config,
      folly::Executor* executor = nullptr) override;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
