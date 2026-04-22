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

// Bridge between the existing CudfExpression evaluator pipeline and the
// GPU SFI engine (PRs 1-11).  GpuSfiExpression converts a Velox exec::Expr
// tree into a GpuExprNode tree at compile time (create()), then evaluates
// it on GPU using GpuExprEvaluator at runtime (eval()).
#pragma once

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/functions/GpuExprEvaluator.h"

namespace facebook::velox::cudf_velox {

class GpuSfiExpression : public CudfExpression {
 public:
  static bool canEvaluate(std::shared_ptr<velox::exec::Expr> expr);

  static std::shared_ptr<CudfExpression> create(
      std::shared_ptr<velox::exec::Expr> expr,
      const RowTypePtr& inputRowSchema);

  ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false) override;

  void close() override;

 private:
  std::unique_ptr<gpu::GpuExprNode> root_;
  gpu::GpuExprEvaluator evaluator_;
};

void registerGpuSfiEvaluator(int priority);

} // namespace facebook::velox::cudf_velox
