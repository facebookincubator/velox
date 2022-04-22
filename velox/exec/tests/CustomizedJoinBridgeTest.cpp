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
#include "velox/exec/JoinBridge.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gmock/gmock.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class MockJoinBridge : public JoinBridge {
 public:
  bool isMockJoinBridge() {
    return true;
  }
};

class MockJoinNode : public core::PlanNode {
 public:
  MockJoinNode(const core::PlanNodeId& id) : PlanNode(id) {}

  MOCK_METHOD(const RowTypePtr&, outputType, (), (const, override));
  MOCK_METHOD(
      const std::vector<std::shared_ptr<const PlanNode>>&,
      sources,
      (),
      (const, override));
  MOCK_METHOD(std::string_view, name, (), (const, override));
  MOCK_METHOD(
      void,
      addDetails,
      (std::stringstream & stream),
      (const, override));
};

class CustomizedJoinBridgeTranslator : public Task::JoinBridgeTranslator {
  std::shared_ptr<JoinBridge> translate(
      const std::shared_ptr<const core::PlanNode>& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const MockJoinNode>(node)) {
      return std::shared_ptr<JoinBridge>(std::make_shared<MockJoinBridge>());
    }
    return nullptr;
  }
};

class MockJoinBridgeTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    registerJoinBridge();
  }

  void registerJoinBridge() {
    Task::registerJoinBridgeTranslator(
        std::make_unique<CustomizedJoinBridgeTranslator>());
  }
};

TEST_F(MockJoinBridgeTest, singleJoinBridge) {
  auto planNode = std::make_shared<MockJoinNode>("0");

  auto& joinTranslators = Task::translators();
  auto joinBridge = joinTranslators[0]->translate(planNode);
  auto mockJoinBridge = std::dynamic_pointer_cast<MockJoinBridge>(joinBridge);
  EXPECT_TRUE(mockJoinBridge->isMockJoinBridge());
}

TEST_F(MockJoinBridgeTest, MultiJoinBridge) {
  registerJoinBridge();
  auto planNode = std::make_shared<MockJoinNode>("0");

  auto& translators = Task::translators();
  for (const auto& t : translators) {
    if (auto joinBridge = t->translate(planNode)) {
      auto mockJoinBridge =
          std::dynamic_pointer_cast<MockJoinBridge>(joinBridge);
      EXPECT_TRUE(mockJoinBridge->isMockJoinBridge());
    }
  }
}
