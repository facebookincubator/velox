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

#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include <gtest/gtest.h>
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

class IcebergConnectorTest : public test::IcebergTestBase {
 protected:
  static void resetIcebergConnector(
      const std::shared_ptr<const config::ConfigBase>& config) {
    ConnectorRegistry::global().erase(test::kIcebergConnectorId);

    IcebergConnectorFactory factory;
    auto icebergConnector =
        factory.newConnector(test::kIcebergConnectorId, config);
    ConnectorRegistry::global().insert(
        icebergConnector->connectorId(), icebergConnector);
  }
};

TEST_F(IcebergConnectorTest, connectorConfiguration) {
  auto customConfig = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {hive::HiveConfig::kEnableFileHandleCache, "true"},
          {hive::HiveConfig::kNumCacheFileHandles, "1000"}});

  resetIcebergConnector(customConfig);

  // Verify connector was registered successfully with custom config.
  auto icebergConnector = ConnectorRegistry::tryGet(test::kIcebergConnectorId);
  ASSERT_NE(icebergConnector, nullptr);

  auto config = icebergConnector->connectorConfig();
  ASSERT_NE(config, nullptr);

  hive::HiveConfig hiveConfig(config);
  ASSERT_TRUE(hiveConfig.isFileHandleCacheEnabled());
  ASSERT_EQ(hiveConfig.numCacheFileHandles(), 1000);
}

TEST_F(IcebergConnectorTest, connectorProperties) {
  auto icebergConnector = ConnectorRegistry::tryGet(test::kIcebergConnectorId);
  ASSERT_NE(icebergConnector, nullptr);

  ASSERT_TRUE(icebergConnector->canAddDynamicFilter());
  ASSERT_TRUE(icebergConnector->supportsSplitPreload());
  ASSERT_NE(icebergConnector->ioExecutor(), nullptr);
}

TEST_F(IcebergConnectorTest, columnHandleForwardsPostProcessor) {
  auto called = std::make_shared<bool>(false);
  std::function<void(VectorPtr&)> postProcessor = [called](VectorPtr&) {
    *called = true;
  };

  IcebergColumnHandle handle(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      parquet::ParquetFieldId{1, {}},
      /*requiredSubfields=*/{},
      /*initialDefaultValue=*/std::nullopt,
      /*icebergMetadata=*/{},
      postProcessor);

  ASSERT_TRUE(handle.postProcessor());
  VectorPtr unused;
  handle.postProcessor()(unused);
  EXPECT_TRUE(*called);
}

TEST_F(IcebergConnectorTest, columnHandleDefaultsToNoPostProcessor) {
  IcebergColumnHandle handle(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      parquet::ParquetFieldId{1, {}});

  EXPECT_FALSE(handle.postProcessor());
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
