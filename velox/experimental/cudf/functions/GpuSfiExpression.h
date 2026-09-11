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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"

namespace facebook::velox::cudf_velox {

inline constexpr const char* kGpuSfiEvaluatorName = "gpu_sfi";

/// Evaluates Velox simple functions that were compiled to CUDA kernels.
///
/// A peer of ASTExpression and JitExpression rather than a CudfFunction, for
/// two reasons. The function tier is column-in column-out, so a tree of N nodes
/// costs N kernels and N intermediate columns, which is precisely the cost that
/// having row-level device code should let us avoid. And it drops constant
/// arguments before eval(), obliging every function to re-handle literals by
/// hand; Velox's own adapter has no such limitation, reading each argument
/// through a DecodedVector so a constant is simply an index mapping. Sitting at
/// the evaluator tier keeps both properties available.
///
/// Today one node is evaluated per instance and foreign children are delegated
/// through createCudfExpression, the way AST delegates a subtree it cannot
/// represent. The shape leaves room to claim a whole subtree of simple
/// functions and, once cuDF can link LTO IR at runtime, to emit it as a single
/// fused kernel -- the property that would make this strictly better than the
/// AST tier rather than merely broader.
class GpuSfiExpression : public CudfExpression {
 public:
  /// Where one argument's values come from. Resolved once at compile time so
  /// eval() only has to form device pointers.
  struct Argument {
    enum class Source {
      /// A column of the operator's input table.
      kInputColumn,
      /// A literal, held as a one-row column and read by every row.
      kConstant,
      /// A nested expression this evaluator does not handle itself.
      kSubexpression,
    };
    Source source;
    /// Index into the input table, constants_, or subexpressions_ per source.
    int32_t index;
  };

  GpuSfiExpression(
      gpu_sfi::GpuLaunchFn launch,
      cudf::data_type outputType,
      std::vector<Argument> arguments,
      std::vector<std::unique_ptr<cudf::column>> constants,
      std::vector<std::shared_ptr<CudfExpression>> subexpressions);

  /// True when the call names a simple function registered for these argument
  /// types. Only the node itself is examined; children pick their own
  /// evaluator.
  static bool canEvaluate(const core::TypedExprPtr& expr);

  static std::shared_ptr<CudfExpression> create(
      const core::TypedExprPtr& expr,
      const RowTypePtr& inputRowSchema,
      memory::MemoryPool* pool);

  ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false) override;

  void close() override;

 private:
  const gpu_sfi::GpuLaunchFn launch_;
  const cudf::data_type outputType_;
  const std::vector<Argument> arguments_;
  const std::vector<std::unique_ptr<cudf::column>> constants_;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions_;
};

/// Registers the evaluator at `priority`, alongside AST and JIT.
void registerGpuSfiEvaluator(int priority);

} // namespace facebook::velox::cudf_velox
