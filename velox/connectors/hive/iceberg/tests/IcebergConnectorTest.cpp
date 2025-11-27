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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

class IcebergConnectorTest : public test::IcebergTestBase {
 protected:
  static void resetIcebergConnector(
      const std::shared_ptr<const config::ConfigBase>& config) {
    unregisterConnector(test::kIcebergConnectorId);

    IcebergConnectorFactory factory;
    auto icebergConnector =
        factory.newConnector(test::kIcebergConnectorId, config);
    registerConnector(icebergConnector);
  }
};

TEST_F(IcebergConnectorTest, connectorConfiguration) {
  auto customConfig = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {hive::HiveConfig::kEnableFileHandleCache, "true"},
          {hive::HiveConfig::kNumCacheFileHandles, "1000"}});

  resetIcebergConnector(customConfig);

  // Verify connector was registered successfully with custom config.
  auto icebergConnector = getConnector(test::kIcebergConnectorId);
  ASSERT_NE(icebergConnector, nullptr);

  auto config = icebergConnector->connectorConfig();
  ASSERT_NE(config, nullptr);

  hive::HiveConfig hiveConfig(config);
  ASSERT_TRUE(hiveConfig.isFileHandleCacheEnabled());
  ASSERT_EQ(hiveConfig.numCacheFileHandles(), 1000);
}

TEST_F(IcebergConnectorTest, connectorProperties) {
  auto icebergConnector = getConnector(test::kIcebergConnectorId);
  ASSERT_NE(icebergConnector, nullptr);

  ASSERT_TRUE(icebergConnector->canAddDynamicFilter());
  ASSERT_TRUE(icebergConnector->supportsSplitPreload());
  ASSERT_NE(icebergConnector->ioExecutor(), nullptr);
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
