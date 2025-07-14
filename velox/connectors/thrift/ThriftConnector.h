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

#include <memory>
#include "velox/connectors/Connector.h"

namespace facebook::velox::connector::thrift {

class ThriftColumnHandle final : public ColumnHandle {};

class ThriftTableHandle final : public ConnectorTableHandle {
 public:
  explicit ThriftTableHandle(std::string connectorId)
      : ConnectorTableHandle(std::move(connectorId)) {}
};

struct ThriftConnectorSplit final : public connector::ConnectorSplit {
  /// 'configName' is the name of the thrift config to read
  /// (for example "presto/testing/thrift_connector_testing" to read the
  /// contents of the config defined in https://fburl.com/code/12klb8fo).
  ThriftConnectorSplit(const std::string& connectorId, std::string configName)
      : ConnectorSplit(connectorId), configName(std::move(configName)) {}
  const std::string configName;
};

/// An thin abstraction layer over the facebook::thrift::ThriftApi
/// to allow easy mocking if ever necessary
class ThriftClient {
 public:
  virtual ~ThriftClient() = default;
};

class ThriftClientImpl final : public ThriftClient {
 public:
  ThriftClientImpl() = default;
};

class ThriftConnector final : public Connector {
 public:
  ThriftConnector(
      const std::string& id,
      std::shared_ptr<const ThriftClient> client)
      : Connector(id), client_(std::move(client)) {}

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& /* unused */,
      const std::shared_ptr<ConnectorTableHandle>&,
      const std::
          unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>&,
      ConnectorQueryCtx* FOLLY_NONNULL /* unused */) override final {
    VELOX_NYI("Data source not supported.");
  }

  std::unique_ptr<DataSink> createDataSink(
      RowTypePtr,
      std::shared_ptr<ConnectorInsertTableHandle>,
      ConnectorQueryCtx*,
      CommitStrategy) override final {
    VELOX_NYI("Data sink not supported.");
  }

 private:
  const std::shared_ptr<const ThriftClient> client_;
};

class ThriftConnectorFactory : public ConnectorFactory {
 public:
  static constexpr const char* FOLLY_NONNULL kThriftConnectorName{
      "presto-thrift"};

  ThriftConnectorFactory() : ConnectorFactory(kThriftConnectorName) {}

  explicit ThriftConnectorFactory(const char* FOLLY_NONNULL connectorName)
      : ConnectorFactory(connectorName) {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const config::ConfigBase> config,
      folly::Executor* ioExecutor = nullptr,
      folly::Executor* cpuExecutor = nullptr) override;
};

} // namespace facebook::velox::connector::thrift
