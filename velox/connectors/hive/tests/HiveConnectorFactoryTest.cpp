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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/Connector.h"

namespace facebook::velox::connector::hive {
namespace {

TEST(HiveConnectorFactoryTest, aliases) {
  auto factory = connector::getConnectorFactory("hive");
  ASSERT_EQ(factory->connectorName(), "hive");

  registerConnectorFactoryAlias("hive-hadoop2", factory->connectorName());
  registerConnectorFactoryAlias("iceberg", factory->connectorName());

  ASSERT_EQ(
      connector::getConnectorFactory("hive-hadoop2")->connectorName(),
      factory->connectorName());
  ASSERT_EQ(
      connector::getConnectorFactory("iceberg")->connectorName(),
      factory->connectorName());

  VELOX_ASSERT_THROW(
      registerConnectorFactoryAlias("iceberg", factory->connectorName()),
      "Alias 'iceberg' is already registered for factory 'hive'");
  VELOX_ASSERT_THROW(
      registerConnectorFactoryAlias("foo", "hivefoo"),
      "ConnectorFactory with name 'hivefoo' not registered");
}

} // namespace
} // namespace facebook::velox::connector::hive
