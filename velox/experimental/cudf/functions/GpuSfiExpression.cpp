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
#include "velox/vector/SimpleVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <deque>

namespace facebook::velox::cudf_velox {
namespace {

using gpu_sfi::GpuArgView;
using gpu_sfi::GpuConstantArgument;
using gpu_sfi::GpuFunctionInstance;
using gpu_sfi::GpuFunctionInstanceSpec;
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

// True when `entry`'s kernel was compiled for exactly these physical types. A
// variadic tail repeats its element kind.
bool physicalTypesMatch(
    const gpu_sfi::GpuFunctionEntry& entry,
    const std::vector<TypePtr>& argumentTypes,
    const TypePtr& returnType) {
  if (entry.returnKind != returnType->kind()) {
    return false;
  }
  if (entry.argumentKinds.empty()) {
    return argumentTypes.empty();
  }
  const auto fixed = entry.signature->variableArity()
      ? entry.argumentKinds.size() - 1
      : entry.argumentKinds.size();
  if (entry.signature->variableArity() ? argumentTypes.size() < fixed
                                       : argumentTypes.size() != fixed) {
    return false;
  }
  for (std::size_t i = 0; i < argumentTypes.size(); ++i) {
    const auto expected =
        i < fixed ? entry.argumentKinds[i] : entry.argumentKinds.back();
    if (argumentTypes[i]->kind() != expected) {
      return false;
    }
  }
  return true;
}

// True when every argument the signature declares constant is a literal. The
// kernel reads such an argument only through what initialize() derived from
// it, so a column there would be silently ignored.
bool constantArgumentsAreLiterals(
    const exec::FunctionSignature& signature,
    const core::TypedExprPtr& expr) {
  const auto& constants = signature.constantArguments();
  const auto& inputs = expr->inputs();
  for (std::size_t i = 0; i < inputs.size() && i < constants.size(); ++i) {
    if (constants[i] && !inputs[i]->isConstantKind()) {
      return false;
    }
  }
  return true;
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
    if (returnType == nullptr || !returnType->equivalent(*expr->type())) {
      continue;
    }
    if (!constantArgumentsAreLiterals(*entry.signature, expr)) {
      continue;
    }
    // A decimal signature cannot tell a short-decimal kernel from a long one.
    if (physicalTypesMatch(entry, argumentTypes, returnType)) {
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

// The physical value of a non-null constant as the bytes GpuConstantArgument
// describes.
template <TypeKind Kind>
std::vector<std::byte> constantBytes(const BaseVector& vector) {
  using T = typename TypeTraits<Kind>::NativeType;
  const T value = vector.as<SimpleVector<T>>()->valueAt(0);
  if constexpr (std::is_same_v<T, StringView>) {
    const auto* begin = reinterpret_cast<const std::byte*>(value.data());
    return {begin, begin + value.size()};
  } else {
    const auto* begin = reinterpret_cast<const std::byte*>(&value);
    return {begin, begin + sizeof(T)};
  }
}

// The constant arguments of a call as initialize() receives them: a value for
// each non-null literal of a primitive type, nothing for any other input.
class ConstantArguments {
 public:
  ConstantArguments(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    for (const auto& input : expr->inputs()) {
      if (!input->isConstantKind() || !input->type()->isPrimitiveType()) {
        descriptors_.push_back(GpuConstantArgument{nullptr, 0});
        continue;
      }
      const auto* constant = input->asUnchecked<core::ConstantTypedExpr>();
      const auto vector = constant->hasValueVector()
          ? constant->valueVector()
          : constant->toConstantVector(pool);
      if (vector->isNullAt(0)) {
        descriptors_.push_back(GpuConstantArgument{nullptr, 0});
        continue;
      }
      values_.push_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          constantBytes, input->type()->kind(), *vector));
      descriptors_.push_back(
          GpuConstantArgument{
              values_.back().data(),
              static_cast<int32_t>(values_.back().size())});
    }
  }

  const std::vector<GpuConstantArgument>& descriptors() const {
    return descriptors_;
  }

 private:
  // Owns the bytes the descriptors point at.
  std::deque<std::vector<std::byte>> values_;
  std::vector<GpuConstantArgument> descriptors_;
};

// Runs the function's initialize() once, at compile time, with this call
// site's argument types and constant values, as SimpleFunctionAdapter does in
// its constructor.
std::vector<std::byte> makeInstance(
    const GpuFunctionInstanceSpec& spec,
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool,
    const core::QueryConfig& config) {
  if (spec.initialize == nullptr) {
    // No initialize(): the kernel default-constructs the instance.
    return {};
  }

  std::vector<TypePtr> inputTypes;
  inputTypes.reserve(expr->inputs().size());
  for (const auto& input : expr->inputs()) {
    inputTypes.push_back(input->type());
  }

  // A byte vector cannot hold overaligned state.
  VELOX_CHECK_LE(
      spec.alignment,
      static_cast<int32_t>(alignof(std::max_align_t)),
      "GPU function instance for {} needs {}-byte alignment, which exceeds "
      "what a std::vector<std::byte> guarantees",
      expr->toString(),
      spec.alignment);

  const ConstantArguments constants(expr, pool);
  std::vector<std::byte> instance(spec.size);
  spec.initialize(instance.data(), inputTypes, config, constants.descriptors());
  return instance;
}

} // namespace

GpuSfiExpression::GpuSfiExpression(
    GpuLaunchFn launch,
    std::vector<std::byte> instance,
    cudf::data_type outputType,
    std::vector<Argument> arguments,
    std::vector<std::unique_ptr<cudf::column>> constants,
    std::vector<std::shared_ptr<CudfExpression>> subexpressions)
    : launch_(launch),
      instance_(std::move(instance)),
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
      makeInstance(resolved->instanceSpec, expr, pool, config),
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

  const GpuFunctionInstance instance{
      instance_.empty() ? nullptr : instance_.data(),
      static_cast<int32_t>(instance_.size())};
  if (errors == nullptr) {
    // No owner can act on a declined row, so the launch does not collect:
    // nulling the row would turn the error into a different answer.
    return launch_(
        argViews,
        instance,
        numRows,
        outputType_,
        /*declinedRows=*/nullptr,
        stream,
        mr);
  }

  // Every launch in this evaluation records into the owner's buffer, which the
  // owner reads once; reading a device scalar here would synchronize the
  // stream.
  auto* const declinedRows = errors->declinedRows(numRows);
  return launch_(
      argViews, instance, numRows, outputType_, declinedRows, stream, mr);
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
