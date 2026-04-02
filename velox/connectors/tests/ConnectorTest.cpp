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
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/core/QueryCtx.h"

#include <gmock/gmock.h>
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

TEST(ConnectorTest, registryOperations) {
  const int32_t numConnectors = 10;
  for (int32_t i = 0; i < numConnectors; i++) {
    auto connector =
        std::make_shared<TestConnector>(fmt::format("connector-{}", i));
    auto connectorId = connector->connectorId();
    ConnectorRegistry::global().insert(
        std::move(connectorId), std::move(connector));
  }

  for (int32_t i = 0; i < numConnectors; i++) {
    EXPECT_NE(
        ConnectorRegistry::tryGet(fmt::format("connector-{}", i)), nullptr);
  }
  EXPECT_EQ(ConnectorRegistry::tryGet("nonexistent"), nullptr);

  auto allTestConnectors = ConnectorRegistry::findAll<TestConnector>();
  EXPECT_EQ(allTestConnectors.size(), numConnectors);

  ConnectorRegistry::unregisterAll();
  EXPECT_EQ(ConnectorRegistry::findAll<TestConnector>().size(), 0);
}

class ConnectorRegistryTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }
};

TEST_F(ConnectorRegistryTest, queryScopedOverride) {
  auto globalConnector = std::make_shared<TestConnector>("global");
  ConnectorRegistry::global().insert("catalog", globalConnector);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = ConnectorRegistry::create(&ConnectorRegistry::global());
  auto queryConnector = std::make_shared<TestConnector>("query-override");
  queryRegistry->insert("catalog", queryConnector);
  queryCtx->setRegistry(ConnectorRegistry::kRegistryKey, queryRegistry);

  // Query-scoped lookup returns the override.
  EXPECT_EQ(ConnectorRegistry::tryGet(*queryCtx, "catalog"), queryConnector);
  // Global lookup returns the global connector.
  EXPECT_EQ(ConnectorRegistry::tryGet("catalog"), globalConnector);

  ConnectorRegistry::unregisterAll();
}

TEST_F(ConnectorRegistryTest, queryScopedFallbackToGlobal) {
  auto globalConnector = std::make_shared<TestConnector>("global");
  ConnectorRegistry::global().insert("catalog", globalConnector);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = ConnectorRegistry::create(&ConnectorRegistry::global());
  queryCtx->setRegistry(ConnectorRegistry::kRegistryKey, queryRegistry);

  // Empty per-query registry falls back to global.
  EXPECT_EQ(ConnectorRegistry::tryGet(*queryCtx, "catalog"), globalConnector);

  ConnectorRegistry::unregisterAll();
}

TEST_F(ConnectorRegistryTest, noQueryRegistryFallsBackToGlobal) {
  auto globalConnector = std::make_shared<TestConnector>("global");
  ConnectorRegistry::global().insert("catalog", globalConnector);

  // QueryCtx with no per-query registry set.
  auto queryCtx = core::QueryCtx::create();
  EXPECT_EQ(ConnectorRegistry::tryGet(*queryCtx, "catalog"), globalConnector);

  ConnectorRegistry::unregisterAll();
}

TEST_F(ConnectorRegistryTest, queryScopedUnregisterAll) {
  auto globalConnector = std::make_shared<TestConnector>("global");
  ConnectorRegistry::global().insert("catalog", globalConnector);

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = ConnectorRegistry::create(&ConnectorRegistry::global());
  queryRegistry->insert("catalog", std::make_shared<TestConnector>("query"));
  queryCtx->setRegistry(ConnectorRegistry::kRegistryKey, queryRegistry);

  ConnectorRegistry::unregisterAll(*queryCtx);

  // Query-scoped registry cleared; falls back to global.
  EXPECT_EQ(ConnectorRegistry::tryGet(*queryCtx, "catalog"), globalConnector);
  // Global is untouched.
  EXPECT_EQ(ConnectorRegistry::tryGet("catalog"), globalConnector);

  ConnectorRegistry::unregisterAll();
}

// Verify that unregisterAll on a queryCtx without a per-query registry does
// not clear the global registry.
TEST_F(ConnectorRegistryTest, unregisterAllNoQueryRegistry) {
  auto globalConnector = std::make_shared<TestConnector>("global");
  ConnectorRegistry::global().insert("catalog", globalConnector);

  auto queryCtx = core::QueryCtx::create();
  ConnectorRegistry::unregisterAll(*queryCtx);

  // Global registry is untouched.
  EXPECT_EQ(ConnectorRegistry::tryGet("catalog"), globalConnector);

  ConnectorRegistry::unregisterAll();
}

TEST_F(ConnectorRegistryTest, queryScopedFindAll) {
  ConnectorRegistry::global().insert(
      "global-cat", std::make_shared<TestConnector>("global-cat"));

  auto queryCtx = core::QueryCtx::create();
  auto queryRegistry = ConnectorRegistry::create(&ConnectorRegistry::global());
  queryRegistry->insert(
      "query-cat", std::make_shared<TestConnector>("query-cat"));
  queryCtx->setRegistry(ConnectorRegistry::kRegistryKey, queryRegistry);

  // findAll with queryCtx sees both query-scoped and global connectors.
  auto all = ConnectorRegistry::findAll<TestConnector>(*queryCtx);
  EXPECT_EQ(all.size(), 2);

  // findAll without queryCtx sees only global.
  auto globalOnly = ConnectorRegistry::findAll<TestConnector>();
  EXPECT_EQ(globalOnly.size(), 1);

  ConnectorRegistry::unregisterAll();
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
