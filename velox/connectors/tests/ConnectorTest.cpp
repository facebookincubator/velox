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

#include "velox/connectors/Connector.h"
#include "velox/common/config/Config.h"

#include <gtest/gtest.h>

namespace facebook::velox::connector {
namespace {

class TestConnector : public connector::Connector {
 public:
  TestConnector(const std::string& id) : connector::Connector(id) {}

  std::unique_ptr<connector::DataSource> createDataSource(
      const RowTypePtr& /* outputType */,
      const ConnectorTableHandlePtr& /* tableHandle */,
      const connector::ColumnHandleMap& /* columnHandles */,
      connector::ConnectorQueryCtx* connectorQueryCtx) override {
    VELOX_NYI();
  }

  std::unique_ptr<connector::DataSink> createDataSink(
      RowTypePtr /*inputType*/,
      ConnectorInsertTableHandlePtr /*connectorInsertTableHandle*/,
      ConnectorQueryCtx* /*connectorQueryCtx*/,
      CommitStrategy /*commitStrategy*/) override final {
    VELOX_NYI();
  }
};

class TestConnectorFactory : public connector::ConnectorFactory {
 public:
  TestConnectorFactory() : ConnectorFactory("test-factory") {}

  std::shared_ptr<Connector> newConnector(
      const std::string& id,
      config::ConfigPtr /*config*/,
      folly::Executor* /*ioExecutor*/ = nullptr,
      folly::Executor* /*cpuExecutor*/ = nullptr) override {
    return std::make_shared<TestConnector>(id);
  }
};

TEST(ConnectorTest, getAllConnectors) {
  TestConnectorFactory factory;

  const int32_t numConnectors = 10;
  for (int32_t i = 0; i < numConnectors; i++) {
    registerConnector(factory.newConnector(
        fmt::format("connector-{}", i),
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>())));
  }

  const auto& connectors = getAllConnectors();
  EXPECT_EQ(connectors.size(), numConnectors);
  for (int32_t i = 0; i < numConnectors; i++) {
    EXPECT_EQ(connectors.count(fmt::format("connector-{}", i)), 1);
  }

  for (int32_t i = 0; i < numConnectors; i++) {
    unregisterConnector(fmt::format("connector-{}", i));
  }
  EXPECT_EQ(getAllConnectors().size(), 0);
}

TEST(ConnectorTest, connectorSplit) {
  {
    const ConnectorSplit split("test", 100, true);
    ASSERT_EQ(split.connectorId, "test");
    ASSERT_EQ(split.splitWeight, 100);
    ASSERT_EQ(split.cacheable, true);
    ASSERT_EQ(
        split.toString(),
        "[split: connector id test, weight 100, cacheable true]");
  }
  {
    const ConnectorSplit split("test", 50, false);
    ASSERT_EQ(split.connectorId, "test");
    ASSERT_EQ(split.splitWeight, 50);
    ASSERT_EQ(split.cacheable, false);
    ASSERT_EQ(
        split.toString(),
        "[split: connector id test, weight 50, cacheable false]");
  }
}
} // namespace
} // namespace facebook::velox::connector
