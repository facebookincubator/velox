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

#include "velox/connectors/lance/LanceConnector.h"
#include "velox/connectors/lance/LanceConnectorSplit.h"
#include "velox/connectors/lance/LanceTableHandle.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::lance;
using namespace facebook::velox::exec::test;

static const std::string kLanceConnectorId = "lance-test";

class LanceConnectorTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    LanceConnectorFactory factory;
    auto lanceConnector = factory.newConnector(
        kLanceConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    connector::registerConnector(lanceConnector);
  }

  void TearDown() override {
    connector::unregisterConnector(kLanceConnectorId);
    OperatorTestBase::TearDown();
  }
};

TEST_F(LanceConnectorTest, registrationWorks) {
  ASSERT_TRUE(connector::hasConnector(kLanceConnectorId));
}

TEST_F(LanceConnectorTest, tableHandleFields) {
  auto handle = std::make_shared<LanceTableHandle>(
      kLanceConnectorId, "/tmp/test.lance");
  ASSERT_EQ(handle->datasetPath(), "/tmp/test.lance");
  ASSERT_EQ(handle->name(), "/tmp/test.lance");
  ASSERT_EQ(handle->connectorId(), kLanceConnectorId);
}

TEST_F(LanceConnectorTest, splitFields) {
  auto split = std::make_shared<LanceConnectorSplit>(
      kLanceConnectorId, "/tmp/test.lance");
  ASSERT_EQ(split->datasetPath, "/tmp/test.lance");
  ASSERT_EQ(split->connectorId, kLanceConnectorId);
}

TEST_F(LanceConnectorTest, columnHandleName) {
  LanceColumnHandle handle("my_column");
  ASSERT_EQ(handle.name(), "my_column");
}
