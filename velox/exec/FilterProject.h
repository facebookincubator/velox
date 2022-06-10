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
#pragma once

#include <folly/executors/ThreadedExecutor.h>
#include <velox/exec/Driver.h>
#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::exec {
class FilterProject : public Operator {
 public:
  FilterProject(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project);

  bool isFilter() const override {
    return true;
  }

  bool preservesOrder() const override {
    return true;
  }

  bool needsInput() const override {
    return !input_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* continueFuture) override {
    if (!isBlocked_) {
      return BlockingReason::kNotBlocked;
    } else {
      // If is blocked then the future shall be valid
      VELOX_CHECK(continueFuture_.valid());
      *continueFuture = std::move(continueFuture_);
      return BlockingReason::kOperatorAsync;
    }
  }

  bool isFinished() override;

  void close() override {
    Operator::close();
    exprs_->clear();
  }

 private:
  // Tests if 'numProcessedRows_' equals to the length of input_ and clears
  // outstanding references to input_ if done. Returns true if getOutput
  // should return nullptr.
  bool allInputProcessed();

  // Evaluate filter on all rows. Return number of rows that passed the
  // filter. Populate filterEvalCtx_.selectedBits and selectedIndices with the
  // indices of the passing rows if only some rows pass the filter. If all or
  // no rows passed the filter filterEvalCtx_.selectedBits and selectedIndices
  // are not updated.
  vector_size_t filter(EvalCtx* evalCtx, const SelectivityVector& allRows);

  // Evaluate projections on the specified rows and populate results_.
  // pre-condition: !isIdentityProjection_
  void project(const SelectivityVector& rows, EvalCtx* evalCtx);

  // If true exprs_[0] is a filter and the other expressions are projections
  const bool hasFilter_{false};
  std::unique_ptr<ExprSet> exprs_;
  int32_t numExprs_;

  FilterEvalCtx filterEvalCtx_;

  vector_size_t numProcessedInputRows_{0};

  bool isAsync_ = false;
  bool noYield_ = false;

  bool isBlocked_ = false;

  folly::Promise<bool> promise_;

  ContinueFuture continueFuture_{ContinueFuture::makeEmpty()};
  bool asyncProjectedNotConsumed = false;
  std::shared_ptr<LocalSelectivityVector> localRows_;
  std::shared_ptr<EvalCtx> evalCtx_;
};
} // namespace facebook::velox::exec
