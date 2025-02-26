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

#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/expression/Expr.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>

namespace facebook::velox::cudf_velox {

// TODO: Does not support Filter yet.
class CudfFilterProject : public exec::Operator {
 public:
  CudfFilterProject(
      int32_t operatorId,
      velox::exec::DriverCtx* driverCtx,
      const velox::exec::FilterProject::Export& info,
      std::vector<velox::exec::IdentityProjection> identityProjections,
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project);

  bool needsInput() const override {
    return !input_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  void close() override {
    Operator::close();
    projectAst_.clear();
    scalars_.clear();
    precompute_instructions_.clear();
  }

 private:
  bool allInputProcessed();
  // If true exprs_[0] is a filter and the other expressions are projections
  const bool hasFilter_{false};
  // Cached filter and project node for lazy initialization. After
  // initialization, they will be reset, and initialized_ will be set to true.
  std::shared_ptr<const core::ProjectNode> project_;
  std::shared_ptr<const core::FilterNode> filter_;
  std::vector<cudf::ast::tree> projectAst_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  // instruction on dependent column to get new column index on non-ast
  // supported operations in expressions
  // <dependent_column_index, "instruction", new_column_index>
  std::vector<std::tuple<int, std::string, int>> precompute_instructions_;

  std::vector<velox::exec::IdentityProjection> resultProjections_;
  std::vector<velox::exec::IdentityProjection> identityProjections_;

  nvtx3::color color_{nvtx3::rgb{220, 20, 60}}; // Crimson
};

} // namespace facebook::velox::cudf_velox
