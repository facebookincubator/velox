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

// Host side of the GPU simple-function evaluator. Compiled against real Velox
// with no shadow include path; it reaches device code only through GpuLaunchFn.

#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"
#include "velox/experimental/cudf/functions/GpuFunctionLookup.h"
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"

#include "velox/expression/SignatureBinder.h"
#include "velox/type/TypeCoercer.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {
namespace {

using gpu_sfi::GpuArgView;
using gpu_sfi::GpuLaunchFn;

// A constant is a one-row column read at element 0, which cannot represent a
// null literal, so a call with one is declined.
bool hasNullLiteralArgument(const core::TypedExprPtr& expr) {
  for (const auto& input : expr->inputs()) {
    if (input->isConstantKind() &&
        input->asUnchecked<core::ConstantTypedExpr>()->isNull()) {
      return true;
    }
  }
  return false;
}

const gpu_sfi::GpuFunctionEntry* resolve(const core::TypedExprPtr& expr) {
  if (expr->kind() != core::ExprKind::kCall) {
    return nullptr;
  }
  // Declined here so a lower-priority evaluator can take the node; create()
  // throwing would abort the operator.
  if (hasNullLiteralArgument(expr)) {
    return nullptr;
  }
  const auto& name = expr->asUnchecked<core::CallTypedExpr>()->name();

  const auto& registry = gpu_sfi::gpuFunctionRegistry();
  auto entries = registry.find(name);
  if (entries == registry.end()) {
    return nullptr;
  }

  std::vector<TypePtr> argumentTypes;
  argumentTypes.reserve(expr->inputs().size());
  for (const auto& input : expr->inputs()) {
    argumentTypes.push_back(input->type());
  }

  // Resolves overloads with exec::SignatureBinder, as SimpleFunctionRegistry
  // does. No coercions: widening an argument could make the GPU result differ
  // from the CPU one.
  for (const auto& entry : entries->second) {
    exec::SignatureBinder binder(
        *entry.signature, argumentTypes, TypeCoercer::defaults());
    if (!binder.tryBind()) {
      continue;
    }
    // Binding proves the arguments fit; the return type still has to be the one
    // the plan expects, since a bound generic could resolve to something else.
    const auto returnType = binder.tryResolveReturnType();
    if (returnType != nullptr && returnType->equivalent(*expr->type())) {
      return &entry;
    }
  }
  return nullptr;
}

// Forms the descriptor the kernel reads an argument through.
GpuArgView toArgView(const cudf::column_view& column, bool isConstant) {
  return GpuArgView{
      static_cast<const void*>(column.head<uint8_t>()),
      column.null_mask(),
      column.offset(),
      isConstant};
}

} // namespace

GpuSfiExpression::GpuSfiExpression(
    GpuLaunchFn launch,
    cudf::data_type outputType,
    std::vector<Argument> arguments,
    std::vector<std::unique_ptr<cudf::column>> constants,
    std::vector<std::shared_ptr<CudfExpression>> subexpressions)
    : launch_(launch),
      outputType_(outputType),
      arguments_(std::move(arguments)),
      constants_(std::move(constants)),
      subexpressions_(std::move(subexpressions)) {}

bool GpuSfiExpression::canEvaluate(const core::TypedExprPtr& expr) {
  return resolve(expr) != nullptr;
}

std::shared_ptr<CudfExpression> GpuSfiExpression::create(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool,
    const core::QueryConfig& config) {
  const auto* resolved = resolve(expr);
  VELOX_CHECK_NOT_NULL(
      resolved, "No GPU simple function for {}", expr->toString());

  std::vector<Argument> arguments;
  std::vector<std::unique_ptr<cudf::column>> constants;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions;
  arguments.reserve(expr->inputs().size());

  // No operator stream exists at compile time, and cuDF requires the stream
  // and memory resource to be named.
  const auto stream = cudf::get_default_stream(cudf::allow_default_stream);
  const auto mr = get_output_mr();

  for (const auto& input : expr->inputs()) {
    if (input->isConstantKind()) {
      // One row is enough: the kernel reads element 0 for a constant.
      auto scalar =
          makeScalarFromConstantExpr(input, pool, std::nullopt, stream);
      // canEvaluate() declines null literals.
      VELOX_CHECK(
          scalar->is_valid(stream),
          "Null literal argument to {} reached create(); canEvaluate() should "
          "have declined it",
          expr->toString());
      constants.push_back(
          cudf::make_column_from_scalar(*scalar, 1, stream, mr));
      arguments.push_back(
          Argument{
              Argument::Source::kConstant,
              static_cast<int32_t>(constants.size() - 1)});
      continue;
    }

    if (auto field =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(input);
        field != nullptr && field->isInputColumn()) {
      arguments.push_back(
          Argument{
              Argument::Source::kInputColumn,
              static_cast<int32_t>(
                  inputRowSchema->getChildIdx(field->name()))});
      continue;
    }

    // Delegates any child this evaluator does not handle itself.
    subexpressions.push_back(
        createCudfExpression(input, inputRowSchema, pool, config));
    arguments.push_back(
        Argument{
            Argument::Source::kSubexpression,
            static_cast<int32_t>(subexpressions.size() - 1)});
  }

  return std::make_shared<GpuSfiExpression>(
      resolved->launch,
      veloxToCudfDataType(expr->type()),
      std::move(arguments),
      std::move(constants),
      std::move(subexpressions));
}

ColumnOrView GpuSfiExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    bool /*finalize*/,
    gpu_sfi::GpuSfiErrors* errors) {
  // Results of delegated children have to outlive the launch.
  std::vector<ColumnOrView> subexpressionResults;
  subexpressionResults.reserve(subexpressions_.size());
  for (const auto& subexpression : subexpressions_) {
    subexpressionResults.push_back(subexpression->eval(
        inputColumnViews, stream, mr, /*finalize=*/false, errors));
  }

  std::vector<GpuArgView> argViews;
  argViews.reserve(arguments_.size());
  cudf::size_type numRows = 0;

  for (const auto& argument : arguments_) {
    switch (argument.source) {
      case Argument::Source::kInputColumn: {
        const auto& column = inputColumnViews.at(argument.index);
        numRows = std::max(numRows, column.size());
        argViews.push_back(toArgView(column, /*isConstant=*/false));
        break;
      }
      case Argument::Source::kSubexpression: {
        auto column = asView(subexpressionResults.at(argument.index));
        numRows = std::max(numRows, column.size());
        argViews.push_back(toArgView(column, /*isConstant=*/false));
        break;
      }
      case Argument::Source::kConstant:
        argViews.push_back(toArgView(
            constants_.at(argument.index)->view(), /*isConstant=*/true));
        break;
    }
  }

  // An all-constant call still needs a row count; fall back to the table's.
  if (numRows == 0 && !inputColumnViews.empty()) {
    numRows = inputColumnViews.front().size();
  }

  if (errors == nullptr) {
    // No owner can act on a declined row, so the launch does not collect:
    // nulling the row would turn the error into a different answer.
    return launch_(
        argViews, numRows, outputType_, /*declinedRows=*/nullptr, stream, mr);
  }

  // Every launch in this evaluation records into the owner's buffer, which the
  // owner reads once; reading a device scalar here would synchronize the
  // stream.
  auto* const declinedRows = errors->declinedRows(numRows);
  return launch_(argViews, numRows, outputType_, declinedRows, stream, mr);
}

void GpuSfiExpression::close() {
  for (const auto& subexpression : subexpressions_) {
    subexpression->close();
  }
  subexpressions_.clear();
}

void registerGpuSfiEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kGpuSfiEvaluatorName,
      priority,
      [](const core::TypedExprPtr& expr) {
        return GpuSfiExpression::canEvaluate(expr);
      },
      [](const core::TypedExprPtr& expr,
         const RowTypePtr& row,
         memory::MemoryPool* pool,
         const core::QueryConfig& config) {
        return GpuSfiExpression::create(expr, row, pool, config);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
