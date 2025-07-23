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
#include "velox/connectors/clp/ClpConfig.h"

namespace facebook::velox::connector::clp {

class ClpConnector : public Connector {
 public:
  ClpConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config);

  [[nodiscard]] const std::shared_ptr<const config::ConfigBase>&
  connectorConfig() const override {
    return config_->config();
  }

  [[nodiscard]] bool canAddDynamicFilter() const override {
    return false;
  }

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const connector::ColumnHandleMap& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override;

  bool supportsSplitPreload() override {
    return false;
  }

  std::unique_ptr<DataSink> createDataSink(
    RowTypePtr inputType,
    ConnectorInsertTableHandlePtr connectorInsertTableHandle,
    ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy) override;

 private:
  std::shared_ptr<const ClpConfig> config_;
};

class ClpConnectorFactory : public ConnectorFactory {
 public:
  static constexpr const char* kClpConnectorName = "clp";

  ClpConnectorFactory();
  explicit ClpConnectorFactory(const char* connectorName);

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* /*ioExecutor*/,
      folly::Executor* /*cpuExecutor*/) override {
    return std::make_shared<ClpConnector>(id, config);
  }
};

} // namespace facebook::velox::connector::clp
