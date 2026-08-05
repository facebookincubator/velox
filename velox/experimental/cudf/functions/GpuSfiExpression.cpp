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

#include "velox/experimental/cudf/functions/GpuSfiExpression.h"

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"

#include <cudf/column/column_factories.hpp>

#include <algorithm>
#include <cctype>

namespace facebook::velox::cudf_velox {
namespace {

using gpu_sfi::GpuArgView;
using gpu_sfi::GpuLaunchFn;

/// Velox type name in the form registration recorded it. Registration derives
/// its strings from SimpleTypeTrait<T>::name lowercased, which is the same
/// source Velox's own TypeAnalysis reads, so the two agree by construction.
///
/// Direct comparison suffices while every registered signature is concrete. If
/// generic signatures arrive -- Variadic, Generic<T>, decimal precision
/// variables -- this becomes exec::SignatureBinder, which is what the CPU
/// registry uses.
std::string typeName(const TypePtr& type) {
  std::string name{type->kindName()};
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return name;
}

const gpu_sfi::GpuFunctionEntry* resolve(const core::TypedExprPtr& expr) {
  if (expr->kind() != core::ExprKind::kCall) {
    return nullptr;
  }
  const auto& name = expr->asUnchecked<core::CallTypedExpr>()->name();

  const auto& registry = gpu_sfi::gpuFunctionRegistry();
  auto entries = registry.find(name);
  if (entries == registry.end()) {
    return nullptr;
  }

  for (const auto& entry : entries->second) {
    const auto& signature = entry.signature;
    if (signature.argumentTypes.size() != expr->inputs().size() ||
        signature.returnType != typeName(expr->type())) {
      continue;
    }
    bool matches = true;
    for (size_t i = 0; i < signature.argumentTypes.size(); ++i) {
      if (signature.argumentTypes[i] != typeName(expr->inputs()[i]->type())) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return &entry;
    }
  }
  return nullptr;
}

/// Forms the descriptor the kernel indexes through. `isConstant` makes every
/// row read element 0, so a literal is stored once rather than broadcast to
/// numRows elements.
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
    memory::MemoryPool* pool) {
  const auto* resolved = resolve(expr);
  VELOX_CHECK_NOT_NULL(
      resolved, "No GPU simple function for {}", expr->toString());

  std::vector<Argument> arguments;
  std::vector<std::unique_ptr<cudf::column>> constants;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions;
  arguments.reserve(expr->inputs().size());

  for (const auto& input : expr->inputs()) {
    if (input->isConstantKind()) {
      // One row is enough: the kernel reads element 0 for a constant.
      auto scalar = makeScalarFromConstantExpr(input, pool);
      VELOX_USER_CHECK(
          scalar->is_valid(),
          "Null literal argument to {} is not supported on GPU",
          expr->toString());
      constants.push_back(cudf::make_column_from_scalar(*scalar, 1));
      arguments.push_back(
          Argument{Argument::Source::kConstant,
                   static_cast<int32_t>(constants.size() - 1)});
      continue;
    }

    if (auto field =
            std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(input);
        field != nullptr && field->isInputColumn()) {
      arguments.push_back(
          Argument{Argument::Source::kInputColumn,
                   static_cast<int32_t>(inputRowSchema->getChildIdx(
                       field->name()))});
      continue;
    }

    // Anything this evaluator does not model itself is delegated, the way AST
    // delegates a subtree it cannot represent.
    subexpressions.push_back(
        createCudfExpression(input, inputRowSchema, pool));
    arguments.push_back(
        Argument{Argument::Source::kSubexpression,
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
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool /*finalize*/) {
  // Results of delegated children have to outlive the launch.
  std::vector<ColumnOrView> subexpressionResults;
  subexpressionResults.reserve(subexpressions_.size());
  for (const auto& subexpression : subexpressions_) {
    subexpressionResults.push_back(
        subexpression->eval(inputColumnViews, stream, mr));
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
        argViews.push_back(
            toArgView(constants_.at(argument.index)->view(), /*isConstant=*/true));
        break;
    }
  }

  // An all-constant call still needs a row count; fall back to the table's.
  if (numRows == 0 && !inputColumnViews.empty()) {
    numRows = inputColumnViews.front().size();
  }

  return launch_(argViews, numRows, outputType_, stream, mr);
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
         memory::MemoryPool* pool) {
        return GpuSfiExpression::create(expr, row, pool);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
