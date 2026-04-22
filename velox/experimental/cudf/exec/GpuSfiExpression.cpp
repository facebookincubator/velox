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
#include "velox/experimental/cudf/exec/GpuSfiExpression.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/functions/CudfFallbackFunction.h"
#include "velox/experimental/cudf/functions/GpuFunctionDispatch.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FieldReference.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/unary.hpp>

#include <cudf/table/table.hpp>

#include <fmt/format.h>
#include <glog/logging.h>

#include <atomic>

namespace facebook::velox::gpu {
void registerAllPrestoGpuFunctions() __attribute__((weak));
} // namespace facebook::velox::gpu

namespace facebook::velox::cudf_velox {

namespace {

constexpr const char* kGpuSfiEvaluatorName = "gpu_sfi";

cudf::type_id veloxTypeToCudfTypeId(const TypePtr& type) {
  return veloxToCudfDataType(type).id();
}

bool canDispatchOnGpu(
    const std::string& name,
    cudf::type_id returnType,
    const std::vector<cudf::type_id>& argTypes) {
  auto result = gpu::dispatchGpuFunction(name, returnType, argTypes);
  return result.function != nullptr;
}

// Reconstruct a Velox uncompiled TypedExprPtr from a compiled exec::Expr.
// Handles field references, constants, cast, and generic function calls.
core::TypedExprPtr exprToTypedExpr(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (auto fieldRef =
          std::dynamic_pointer_cast<velox::exec::FieldReference>(expr)) {
    return std::make_shared<core::FieldAccessTypedExpr>(
        fieldRef->type(), fieldRef->field());
  }

  if (auto constExpr =
          std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr)) {
    return std::make_shared<core::ConstantTypedExpr>(constExpr->value());
  }

  // Recursively reconstruct children.
  std::vector<core::TypedExprPtr> childTypedExprs;
  childTypedExprs.reserve(expr->inputs().size());
  for (const auto& child : expr->inputs()) {
    childTypedExprs.push_back(exprToTypedExpr(child));
  }

  // Handle cast / try_cast via CastTypedExpr.
  if (expr->name() == "cast" || expr->name() == "try_cast") {
    bool isTryCast = (expr->name() == "try_cast");
    return std::make_shared<core::CastTypedExpr>(
        expr->type(), childTypedExprs.at(0), isTryCast);
  }

  // Generic function call (covers regular functions, special forms like
  // and, or, if, switch, coalesce, in, between, etc.).
  return std::make_shared<core::CallTypedExpr>(
      expr->type(), std::move(childTypedExprs), expr->name());
}

std::unique_ptr<gpu::GpuExprNode> convertExpr(
    const std::shared_ptr<velox::exec::Expr>& expr,
    const RowTypePtr& schema) {
  // Field access: leaf input column.
  if (auto fieldRef =
          std::dynamic_pointer_cast<velox::exec::FieldReference>(expr)) {
    auto fieldName = fieldRef->field();
    auto idx = schema->getChildIdx(fieldName);
    auto typeId = veloxTypeToCudfTypeId(fieldRef->type());
    return gpu::makeFieldAccess(idx, typeId);
  }

  // Constant / literal.
  if (auto constExpr =
          std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr)) {
    auto value = constExpr->value();
    auto typeId = veloxTypeToCudfTypeId(expr->type());

    if (value->isNullAt(0)) {
      return gpu::makeLiteralNull(typeId);
    }

    switch (expr->type()->kind()) {
      case TypeKind::DOUBLE:
        return gpu::makeLiteralDouble(
            value->as<SimpleVector<double>>()->valueAt(0));
      case TypeKind::REAL:
        return gpu::makeLiteralDouble(
            static_cast<double>(
                value->as<SimpleVector<float>>()->valueAt(0)));
      case TypeKind::BIGINT:
        return gpu::makeLiteralInt64(
            value->as<SimpleVector<int64_t>>()->valueAt(0));
      case TypeKind::INTEGER:
        return gpu::makeLiteralInt32(
            value->as<SimpleVector<int32_t>>()->valueAt(0));
      case TypeKind::SMALLINT:
        return gpu::makeLiteralInt64(
            static_cast<int64_t>(
                value->as<SimpleVector<int16_t>>()->valueAt(0)));
      case TypeKind::TINYINT:
        return gpu::makeLiteralInt64(
            static_cast<int64_t>(
                value->as<SimpleVector<int8_t>>()->valueAt(0)));
      case TypeKind::BOOLEAN:
        return gpu::makeLiteralBool(
            value->as<SimpleVector<bool>>()->valueAt(0));
      case TypeKind::VARCHAR: {
        auto sv = value->as<SimpleVector<StringView>>()->valueAt(0);
        return gpu::makeLiteralString(std::string(sv.data(), sv.size()));
      }
      default:
        LOG(WARNING) << "[GPU SFI] CPU fallback (compile): unsupported "
                     << "literal type " << expr->type()->toString()
                     << " in '" << expr->toString() << "'";
        auto typedExpr = exprToTypedExpr(expr);
        return gpu::makeCpuFallback(
            typeId,
            std::shared_ptr<void>(
                std::make_shared<core::TypedExprPtr>(std::move(typedExpr))),
            std::static_pointer_cast<void>(
                std::const_pointer_cast<RowType>(schema)));
    }
  }

  auto resultTypeId = veloxTypeToCudfTypeId(expr->type());
  const auto& inputs = expr->inputs();
  const auto& name = expr->name();

  // Lambda to recursively convert all children.
  auto convertChildren =
      [&]() -> std::vector<std::unique_ptr<gpu::GpuExprNode>> {
    std::vector<std::unique_ptr<gpu::GpuExprNode>> ch;
    ch.reserve(inputs.size());
    for (const auto& input : inputs) {
      ch.push_back(convertExpr(input, schema));
    }
    return ch;
  };

  // Lambda to build a CPU-fallback node for this expression.
  auto buildCpuFallback = [&]() -> std::unique_ptr<gpu::GpuExprNode> {
    LOG(WARNING) << "[GPU SFI] CPU fallback (compile): '" << name
                 << "' -- " << expr->toString();
    auto typedExpr = exprToTypedExpr(expr);
    return gpu::makeCpuFallback(
        resultTypeId,
        std::shared_ptr<void>(
            std::make_shared<core::TypedExprPtr>(std::move(typedExpr))),
        std::static_pointer_cast<void>(
            std::const_pointer_cast<RowType>(schema)));
  };

  // -- Special forms handled natively on GPU --

  if (name == "and") {
    return gpu::makeAnd(convertChildren());
  }
  if (name == "or") {
    return gpu::makeOr(convertChildren());
  }
  if (name == "not") {
    return gpu::makeNot(convertExpr(inputs.at(0), schema));
  }
  if (name == "switch" || name == "if") {
    return gpu::makeSwitch(resultTypeId, convertChildren());
  }
  if (name == "coalesce") {
    return gpu::makeCoalesce(resultTypeId, convertChildren());
  }
  if (name == "cast" || name == "try_cast") {
    auto childTypeId = veloxTypeToCudfTypeId(inputs.at(0)->type());
    cudf::data_type from{childTypeId};
    cudf::data_type to{resultTypeId};
    if (cudf::is_supported_cast(from, to)) {
      return gpu::makeCast(
          resultTypeId, convertExpr(inputs.at(0), schema));
    }
    return buildCpuFallback();
  }

  // -- Regular function call: check GPU dispatch availability --

  std::vector<cudf::type_id> argTypes;
  argTypes.reserve(inputs.size());
  for (const auto& input : inputs) {
    argTypes.push_back(veloxTypeToCudfTypeId(input->type()));
  }

  if (!canDispatchOnGpu(name, resultTypeId, argTypes)) {
    return buildCpuFallback();
  }

  return gpu::makeFunctionCall(name, resultTypeId, convertChildren());
}

} // namespace

bool GpuSfiExpression::canEvaluate(
    std::shared_ptr<velox::exec::Expr> /*expr*/) {
  // Always claim the expression -- unsupported sub-trees become kCpuFallback
  // nodes that are evaluated on CPU transparently.
  return true;
}

std::shared_ptr<CudfExpression> GpuSfiExpression::create(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema) {
  auto node = std::make_shared<GpuSfiExpression>();
  node->root_ = convertExpr(expr, inputRowSchema);

  node->evaluator_.setCpuFallbackHandler(
      [](const gpu::GpuExprNode& fbNode,
         const cudf::table_view& input,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) -> std::unique_ptr<cudf::column> {
        auto& typedExpr = *std::static_pointer_cast<core::TypedExprPtr>(
            fbNode.fallbackExpr);
        auto fbSchema = std::static_pointer_cast<const velox::RowType>(
            fbNode.fallbackSchema);

        LOG(WARNING) << "[GPU SFI] CPU fallback (eval): evaluating '"
                     << typedExpr->toString() << "' on Velox CPU engine "
                     << "(D->H->eval->H->D, " << input.num_rows()
                     << " rows)";

        static std::atomic<uint64_t> poolCounter{0};
        auto poolName =
            fmt::format("gpu_sfi_fallback_{}", poolCounter.fetch_add(1));
        auto pool =
            velox::memory::memoryManager()->addLeafPool(poolName);

        // D-to-H: GPU columns -> Velox RowVector. Uses the TypePtr overload
        // which preserves original field names from the schema so that
        // ExprSet can resolve FieldReference nodes.
        auto veloxInput = with_arrow::toVeloxColumn(
            input,
            pool.get(),
            std::static_pointer_cast<const Type>(fbSchema),
            stream,
            mr);
        stream.synchronize();

        // Evaluate on Velox CPU engine using ExprSet.
        auto queryCtx = core::QueryCtx::create();
        core::ExecCtx execCtx(pool.get(), queryCtx.get());

        exec::ExprSet exprSet({typedExpr}, &execCtx);
        exec::EvalCtx evalCtx(&execCtx, &exprSet, veloxInput.get());

        SelectivityVector allRows(veloxInput->size());
        std::vector<VectorPtr> results(1);
        exprSet.eval(allRows, evalCtx, results);

        // Wrap result into a single-column RowVector for H-to-D transfer.
        auto resultRow = std::make_shared<RowVector>(
            pool.get(),
            ROW({"_result"}, {results[0]->type()}),
            nullptr,
            results[0]->size(),
            std::vector<VectorPtr>{results[0]});

        // H-to-D: Velox result -> cuDF column.
        auto resultTable =
            with_arrow::toCudfTable(resultRow, pool.get(), stream, mr);
        stream.synchronize();

        // Extract the single result column from the table.
        auto columns = resultTable->release();
        VELOX_CHECK_EQ(columns.size(), 1);
        return std::move(columns[0]);
      });

  return node;
}

ColumnOrView GpuSfiExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool /*finalize*/) {
  std::vector<cudf::column_view> views(
      inputColumnViews.begin(), inputColumnViews.end());
  cudf::table_view input(views);

  auto result = evaluator_.evaluate(*root_, input, stream, mr);
  return std::move(result);
}

void GpuSfiExpression::close() {
  root_.reset();
}

void registerGpuSfiEvaluator(int priority) {
  if (!&gpu::registerAllPrestoGpuFunctions) {
    LOG(WARNING) << "GPU SFI evaluator skipped: "
                    "velox_cudf_gpu_presto_functions not linked";
    return;
  }
  gpu::registerAllPrestoGpuFunctions();
  gpu::CudfFallbackRegistry::instance().registerDefaults();
  registerCudfExpressionEvaluator(
      kGpuSfiEvaluatorName,
      priority,
      [](std::shared_ptr<velox::exec::Expr> expr) {
        return GpuSfiExpression::canEvaluate(std::move(expr));
      },
      [](std::shared_ptr<velox::exec::Expr> expr, const RowTypePtr& row) {
        return GpuSfiExpression::create(std::move(expr), row);
      },
      /*overwrite=*/false);
}

} // namespace facebook::velox::cudf_velox
