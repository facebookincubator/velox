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
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gmock/gmock.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

using facebook::velox::test::BatchMaker;

class CustomJoinNode : public core::PlanNode {
 public:
  CustomJoinNode(
      const core::PlanNodeId& id,
      std::shared_ptr<const core::PlanNode> left,
      std::shared_ptr<const core::PlanNode> right)
      : PlanNode(id), sources_{std::move(left), std::move(right)} {}

  const std::shared_ptr<const RowType>& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "custom join";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<std::shared_ptr<const core::PlanNode>> sources_;
};

class CustomJoinBridge : public JoinBridge {
 public:
  void setRowNum(int32_t rowNum) {
    rowNum_ = rowNum;
  }

  int32_t rowNumOrFuture(ContinueFuture* future) {
    if (rowNum_ > 0) {
      return rowNum_;
    }
    return -1;
  }

 private:
  int32_t rowNum_;
};

class CustomJoinBuild : public Operator {
 public:
  CustomJoinBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const CustomJoinNode> joinNode)
      : Operator(
            driverCtx,
            nullptr,
            operatorId,
            joinNode->id(),
            "CustomJoinBuild") {}

  void addInput(RowVectorPtr input) {
    rowNum_ += input->size();
  }

  bool needsInput() const {
    return !noMoreInput_;
  }

  RowVectorPtr getOutput() {
    return nullptr;
  }

  void noMoreInput() {
    Operator::noMoreInput();
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto customJoinBridge =
        std::dynamic_pointer_cast<CustomJoinBridge>(joinBridge);
    customJoinBridge->setRowNum(rowNum_);
  }

  BlockingReason isBlocked(ContinueFuture* future) {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() {
    return noMoreInput_;
  }

 private:
  int32_t rowNum_ = 0;
};

class CustomJoinProbe : public Operator {
 public:
  CustomJoinProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const CustomJoinNode> joinNode)
      : Operator(
            driverCtx,
            nullptr,
            operatorId,
            joinNode->id(),
            "CustomJoinProbe") {}

  bool needsInput() const {
    return !finished_ && input_ == nullptr;
  }

  void addInput(RowVectorPtr input) {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() {
    if (input_ == nullptr || rowNum_ == 0) {
      return nullptr;
    }

    auto output = std::make_shared<RowVector>(
        input_->pool(),
        input_->type(),
        input_->nulls(),
        rowNum_,
        input_->children());
    input_.reset();
    finished_ = true;
    return output;
  }

  BlockingReason isBlocked(ContinueFuture* future) {
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());

    auto rowNum = std::dynamic_pointer_cast<CustomJoinBridge>(joinBridge)
                      ->rowNumOrFuture(future);
    if (rowNum <= 0) {
      return BlockingReason::kWaitForJoinBuild;
    }
    rowNum_ = rowNum;
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() {
    return finished_;
  }

 private:
  int32_t rowNum_ = 0;

  bool finished_{false};
};

class CustomJoinBridgeTranslator : public Operator::PlanNodeTranslator {
  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const core::PlanNode>& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CustomJoinNode>(node)) {
      return std::make_unique<CustomJoinProbe>(id, ctx, joinNode);
    }
    return nullptr;
  }

  std::unique_ptr<JoinBridge> toJoinBridge(
      const std::shared_ptr<const core::PlanNode>& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CustomJoinNode>(node)) {
      auto joinBridge = std::make_unique<CustomJoinBridge>();
      return joinBridge;
    }
    return nullptr;
  }

  OperatorSupplier toOperatorSupplier(
      const std::shared_ptr<const core::PlanNode>& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CustomJoinNode>(node)) {
      return [joinNode](int32_t operatorId, DriverCtx* ctx) {
        return std::make_unique<CustomJoinBuild>(operatorId, ctx, joinNode);
      };
    }
    return nullptr;
  }
};

class CustomJoinTest : public OperatorTestBase {};

// This test will show the CustomJoinBuild passing count of input rows to
// CustomJoinProbe via CustomJoinBridge, then probe will emit corresponding
// number of rows, like a Limit operator.
TEST_F(CustomJoinTest, fakeLimit) {
  Operator::registerOperator(std::make_unique<CustomJoinBridgeTranslator>());

  std::shared_ptr<const RowType> rowType{
      ROW({"c0", "c1"}, {BIGINT(), BIGINT()})};

  // create batches for custom join node, the batch size is only designed for
  // this test.
  auto leftBatch = std::dynamic_pointer_cast<RowVector>(
      BatchMaker::createBatch(rowType, 100, *pool_));
  auto rightBatch = std::dynamic_pointer_cast<RowVector>(
      BatchMaker::createBatch(rowType, 10, *pool_));

  createDuckDbTable("t", {leftBatch});
  createDuckDbTable("u", {rightBatch});

  auto planNodeIdGenerator = std::make_shared<PlanNodeIdGenerator>();
  auto leftNode =
      PlanBuilder(planNodeIdGenerator).values({leftBatch}, true).planNode();
  auto rightNode =
      PlanBuilder(planNodeIdGenerator).values({rightBatch}, true).planNode();

  CursorParameters params;
  // This test only support single driver
  params.maxDrivers = 1;
  params.planNode =
      PlanBuilder(planNodeIdGenerator)
          .values({leftBatch}, true)
          .addNode(
              [&leftNode, &rightNode](
                  std::string id, std::shared_ptr<const core::PlanNode> input) {
                return std::make_shared<CustomJoinNode>(
                    id, std::move(leftNode), std::move(rightNode));
              })
          .project({"c0", "c1"})
          .planNode();

  OperatorTestBase::assertQuery(params, "select c0, c1 from t limit 10");
}
