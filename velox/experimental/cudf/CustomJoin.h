#include "velox/exec/JoinBridge.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

using CustomJoinNode = typename facebook::velox::core::HashJoinNode;

class CustomJoinBridge : public JoinBridge {
 public:
  void setNumRows(std::optional<int32_t> numRows) {
    std::vector<ContinuePromise> promises;
    {
      std::lock_guard<std::mutex> l(mutex_);
      VELOX_CHECK(!numRows_.has_value(), "setNumRows may be called only once");
      numRows_ = numRows;
      promises = std::move(promises_);
    }
    notify(std::move(promises));
  }

  std::optional<int32_t> numRowsOrFuture(ContinueFuture* future) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
    if (numRows_.has_value()) {
      return numRows_;
    }
    promises_.emplace_back("CustomJoinBridge::numRowsOrFuture");
    *future = promises_.back().getSemiFuture();
    return std::nullopt;
  }

 private:
  std::optional<int32_t> numRows_;
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
  CustomJoinBuild(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::PlanNodeId& joinNodeid)
      : Operator(
            driverCtx,
            nullptr,
            operatorId,
            joinNodeid,
            "CustomJoinBuild") {}

  void addInput(RowVectorPtr input) override {
    auto inputSize = input->size();
    if (inputSize > 0) {
      numRows_ += inputSize;
    }
  }

  bool needsInput() const override {
    return !noMoreInput_;
  }

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  void noMoreInput() override {
    Operator::noMoreInput();
    std::vector<ContinuePromise> promises;
    std::vector<std::shared_ptr<Driver>> peers;
    // The last Driver to hit CustomJoinBuild::finish gathers the data from
    // all build Drivers and hands it over to the probe side. At this
    // point all build Drivers are continued and will free their
    // state. allPeersFinished is true only for the last Driver of the
    // build pipeline.
    if (!operatorCtx_->task()->allPeersFinished(
            planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
      return;
    }

    for (auto& peer : peers) {
      auto op = peer->findOperator(planNodeId());
      auto* build = dynamic_cast<CustomJoinBuild*>(op);
      VELOX_CHECK(build);
      numRows_ += build->numRows_;
    }

    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }

    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto customJoinBridge =
        std::dynamic_pointer_cast<CustomJoinBridge>(joinBridge);
    customJoinBridge->setNumRows(std::make_optional(numRows_));
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    if (!future_.valid()) {
      return BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return BlockingReason::kWaitForJoinBuild;
  }

  bool isFinished() override {
    return !future_.valid() && noMoreInput_;
  }

 private:
  int32_t numRows_ = 0;

  ContinueFuture future_{ContinueFuture::makeEmpty()};
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
  CustomJoinProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::PlanNodeId& joinNodeid)
      : Operator(
            driverCtx,
            nullptr,
            operatorId,
            joinNodeid,
            "CustomJoinProbe") {}

  bool needsInput() const override {
    return !finished_ && input_ == nullptr;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    if (!input_) {
      return nullptr;
    }

    const auto inputSize = input_->size();
    if (remainingLimit_ <= inputSize) {
      finished_ = true;
    }

    if (remainingLimit_ >= inputSize) {
      remainingLimit_ -= inputSize;
      auto output = input_;
      input_.reset();
      return output;
    }

    // Return nullptr if there is no data to return.
    if (remainingLimit_ == 0) {
      input_.reset();
      return nullptr;
    }

    auto output = std::make_shared<RowVector>(
        input_->pool(),
        input_->type(),
        input_->nulls(),
        remainingLimit_,
        input_->children());
    input_.reset();
    remainingLimit_ = 0;
    return output;
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    if (numRows_.has_value()) {
      return BlockingReason::kNotBlocked;
    }

    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto numRows = std::dynamic_pointer_cast<CustomJoinBridge>(joinBridge)
                       ->numRowsOrFuture(future);

    if (!numRows.has_value()) {
      return BlockingReason::kWaitForJoinBuild;
    }
    numRows_ = std::move(numRows);
    remainingLimit_ = numRows_.value();

    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_ || (noMoreInput_ && input_ == nullptr);
  }

 private:
  int32_t remainingLimit_;
  std::optional<int32_t> numRows_;

  bool finished_{false};
};

class CustomJoinBridgeTranslator : public Operator::PlanNodeTranslator {
  std::unique_ptr<Operator>
  toOperator(DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CustomJoinNode>(node)) {
        std::cout<<"CustomJoinProbe replaced\n";
      return std::make_unique<CustomJoinProbe>(id, ctx, joinNode);
    }
    return nullptr;
  }

  std::unique_ptr<JoinBridge> toJoinBridge(const core::PlanNodePtr& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CustomJoinNode>(node)) {
        std::cout<<"CustomJoinBridge replaced\n";
      auto joinBridge = std::make_unique<CustomJoinBridge>();
      return joinBridge;
    }
    return nullptr;
  }

  OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node) {
    if (auto joinNode = std::dynamic_pointer_cast<const CustomJoinNode>(node)) {
      return [joinNode](int32_t operatorId, DriverCtx* ctx) {
        std::cout<<"CustomJoinBuild replaced\n";
        return std::make_unique<CustomJoinBuild>(operatorId, ctx, joinNode);
      };
    }
    return nullptr;
  }
};

void registerTranslator() {
    Operator::registerOperator(std::make_unique<CustomJoinBridgeTranslator>());
}