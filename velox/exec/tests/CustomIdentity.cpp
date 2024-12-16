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
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/arrow/Bridge.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class CustomIdentityNode : public core::PlanNode {
 public:
  CustomIdentityNode(const core::PlanNodeId& id, core::PlanNodePtr input)
      : PlanNode(id), sources_{std::move(input)} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "identity using Arrow";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  std::vector<core::PlanNodePtr> sources_;
};

class CustomIdentity : public Operator {
 public:
  CustomIdentity(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const CustomIdentityNode> identityNode)
      : Operator(
            driverCtx,
            nullptr,
            operatorId,
            identityNode->id(),
            "CustomIdentity") {}

  bool needsInput() const override {
    return !finished_ and input_ == nullptr;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    if (finished_) { //|| !noMoreInput_
      return nullptr;
    }
    finished_ = noMoreInput_;

    if (!input_) {
      return nullptr;
    }
    const auto inputSize = input_->size();
    // Return nullptr if there is no data to return.
    if (inputSize == 0) {
      input_.reset();
      return nullptr;
    }

    // TO ARROW
    // flattenDictionary and flattenConstant
    ArrowOptions arrowOptions{true, true};
    ArrowArray arrowArray;
    facebook::velox::exportToArrow(
        std::dynamic_pointer_cast<facebook::velox::BaseVector>(input_),
        arrowArray,
        pool(),
        arrowOptions);
    ArrowSchema arrowSchema;
    facebook::velox::exportToArrow(
        std::dynamic_pointer_cast<facebook::velox::BaseVector>(input_),
        arrowSchema,
        arrowOptions);
    // FROM ARROW
    auto veloxTable = facebook::velox::importFromArrowAsOwner(arrowSchema, arrowArray, pool());

    // Release Arrow resources
    if (arrowArray.release) {
      arrowArray.release(&arrowArray);
    }
    if (arrowSchema.release) {
      arrowSchema.release(&arrowSchema);
    }
    // Convert to Arrow and Arrow to Velox
    auto outputTable_ = std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
    VELOX_CHECK_NOT_NULL(outputTable_);
    //
    input_.reset();
    return outputTable_;
  }

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_ || (noMoreInput_ && input_ == nullptr);
  }

 private:
  bool finished_{false};
};

class CustomIdentityNodeTranslator : public Operator::PlanNodeTranslator {
  std::unique_ptr<Operator>
  toOperator(DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node) {
    if (auto identityNode =
            std::dynamic_pointer_cast<const CustomIdentityNode>(node)) {
      return std::make_unique<CustomIdentity>(id, ctx, identityNode);
    }
    return nullptr;
  }
};

/// This test will show the CustomIdentity works for a simple case.
class CustomIdentityTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    Operator::registerOperator(std::make_unique<CustomIdentityNodeTranslator>());
  }

  RowVectorPtr makeSimpleRowVector(vector_size_t size) {
    return makeRowVector(
        {makeFlatVector<int32_t>(size, [](auto row) { return row; })});
  }
  RowVectorPtr makeConstantRowVector(vector_size_t size) {
    return makeRowVector(
        {makeConstant<int32_t>(size, size)});
  }

  void testCustomIdentity(
      int32_t numThreads,
      const std::vector<RowVectorPtr>& leftBatch,
      const std::string& referenceQuery) {
    createDuckDbTable("t", {leftBatch});

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto leftNode =
        PlanBuilder(planNodeIdGenerator).values({leftBatch}, true).planNode();

    CursorParameters params;
    params.maxDrivers = numThreads;
    params.planNode =
        PlanBuilder(planNodeIdGenerator)
            .values({leftBatch}, true)
            .addNode([&leftNode](
                         std::string id, core::PlanNodePtr /* input */) {
              return std::make_shared<CustomIdentityNode>(
                  id, std::move(leftNode));
            })
            .project({"c0"})
            .planNode();

    OperatorTestBase::assertQuery(params, referenceQuery);
  }
};

TEST_F(CustomIdentityTest, simple) {
  auto leftBatch = {makeSimpleRowVector(100)};
  testCustomIdentity(1, leftBatch, "SELECT c0 FROM t");
}

TEST_F(CustomIdentityTest, constant) {
  auto leftBatch = {makeConstantRowVector(100)};
  testCustomIdentity(1, leftBatch, "SELECT c0 FROM t");
}

TEST_F(CustomIdentityTest, parallelism) {
  auto leftBatch = {
      makeSimpleRowVector(30),
      makeSimpleRowVector(40),
      makeSimpleRowVector(50)};

  testCustomIdentity(
      3,
      leftBatch,
      "(SELECT c0 FROM t) "
      "UNION ALL (SELECT c0 FROM t) "
      "UNION ALL (SELECT c0 FROM t)");
}
