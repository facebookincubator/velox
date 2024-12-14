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

#include "velox/common/config/Config.h"
#include "velox/connectors/Connector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::cudf_velox::connector::parquet {

class ParquetConfig {
 public:
  ParquetConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for parquetConfig initialization");
    config_ = std::move(config);
  }

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

  // [[nodiscard]] cudf::io::source_info const& get_source() const = delete;

 private:
  std::shared_ptr<const config::ConfigBase> config_;
};

class ParquetConnector final : public Connector {
 public:
  ParquetConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor);

  std::unique_ptr<DataSource> createDataSource(
      const std::shared_ptr<const RowType>& outputType,
      const std::shared_ptr<ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override final;

  const std::shared_ptr<const config::ConfigBase>& connectorConfig()
      const override {
    return parquetConfig_->config();
  }

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr /*inputType*/,
      std::shared_ptr<
          ConnectorInsertTableHandle> /*connectorInsertTableHandle*/,
      ConnectorQueryCtx* /*connectorQueryCtx*/,
      CommitStrategy /*commitStrategy*/) override final {
    // cudf::ParquetConnector::DataSink not yet implemented
    VELOX_NYI("cudf::ParquetConnector does not yet support data sink.");
  }

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
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* executor = nullptr) override;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
