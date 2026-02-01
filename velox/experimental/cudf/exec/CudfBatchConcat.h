#pragma once

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

#include <queue>

namespace facebook::velox::cudf_velox {

class CudfBatchConcat : public exec::Operator, public CudfOperator {
 public:
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode);

  bool needsInput() const override {
    return !noMoreInput_ && outputQueue_.empty() &&
        currentNumRows_ < targetRows_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override {
    noMoreInput_ = true;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  exec::DriverCtx* const driverCtx_;
  std::vector<CudfVectorPtr> buffer_;
  std::queue<std::unique_ptr<cudf::table>> outputQueue_;
  size_t currentNumRows_{0};
  const size_t targetRows_{0};
  bool noMoreInput_{false};
};

class CudfBatchConcatTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
