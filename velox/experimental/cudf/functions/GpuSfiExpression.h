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

#include <cstddef>
#include <vector>

namespace facebook::velox::cudf_velox {

inline constexpr const char* kGpuSfiEvaluatorName = "gpu_sfi";

/// Evaluates Velox simple functions compiled to CUDA kernels, as a peer of
/// ASTExpression and JitExpression. Each instance evaluates one call node and
/// delegates children it does not handle through createCudfExpression.
/// Constant arguments reach the kernel as one-row columns.
///
/// TODO: Claim whole subtrees of simple functions and fuse them into one
/// kernel.
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
      std::vector<std::byte> instance,
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
      memory::MemoryPool* pool,
      const core::QueryConfig& config);

  ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false,
      gpu_sfi::GpuSfiErrors* errors = nullptr) override;

  void close() override;

 private:
  const gpu_sfi::GpuLaunchFn launch_;
  // The function's initialized instance, built once at compile time. Opaque
  // because only the shadow-compiled side can name the type.
  const std::vector<std::byte> instance_;
  const cudf::data_type outputType_;
  const std::vector<Argument> arguments_;
  const std::vector<std::unique_ptr<cudf::column>> constants_;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions_;
};

/// Registers the evaluator at `priority`, alongside AST and JIT.
void registerGpuSfiEvaluator(int priority);

} // namespace facebook::velox::cudf_velox
