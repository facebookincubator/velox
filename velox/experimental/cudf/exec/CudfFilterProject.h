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

#include "velox/experimental/cudf/exec/ExpressionEvaluator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::cudf_velox {

// TODO: Does not support Filter yet.
class CudfFilterProject : public exec::Operator, public NvtxHelper {
 public:
  CudfFilterProject(
      int32_t operatorId,
      velox::exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project);

  void initialize() override;

  bool needsInput() const override {
    return !input_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void filter(
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
      rmm::cuda_stream_view stream);

  std::vector<std::unique_ptr<cudf::column>> project(
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
      rmm::cuda_stream_view stream);

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  void close() override {
    Operator::close();
    projectEvaluator_.close();
    filterEvaluator_.close();
  }

 private:
  // Copied from operator FilterProject.
  void initializeFilterProject();

  /// Data for accelerator conversion.
  struct Export {
    const exec::ExprSet* exprs;
    bool hasFilter;
    const std::vector<exec::IdentityProjection>* resultProjections;
  };

  Export exprsAndProjection() const {
    return Export{exprs_.get(), hasFilter_, &resultProjections_};
  }
  
  bool allInputProcessed();

  // If true exprs_[0] is a filter and the other expressions are projections
  const bool hasFilter_{false};
  const bool lazyDereference_;

  std::unique_ptr<exec::ExprSet> exprs_;
  int32_t numExprs_;

  // Cached filter and project node for lazy initialization. After
  // initialization, they will be reset, and initialized_ will be set to true.
  std::shared_ptr<const core::ProjectNode> project_;
  std::shared_ptr<const core::FilterNode> filter_;

  ExpressionEvaluator projectEvaluator_;
  ExpressionEvaluator filterEvaluator_;

  std::vector<velox::exec::IdentityProjection> resultProjections_;
  std::vector<velox::exec::IdentityProjection> identityProjections_;

  // Indices for fields/input columns that are both an identity projection and
  // are referenced by either a filter or project expression. This is used to
  // identify fields that need to be preloaded before evaluating filters or
  // projections.
  // Consider projection with 2 expressions: f(c0) AND g(c1), c1
  // If c1 is a LazyVector and f(c0) AND g(c1) expression is evaluated first, it
  // will load c1 only for rows where f(c0) is true. However, c1 identity
  // projection needs all rows.
  std::vector<column_index_t> multiplyReferencedFieldIndices_;
};

} // namespace facebook::velox::cudf_velox
