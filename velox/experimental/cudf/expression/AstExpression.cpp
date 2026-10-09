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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/AstPrinter.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"
#include "velox/experimental/cudf/vector/TableViewPrinter.h"

#include "velox/expression/ExprConstants.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/ast/detail/operators.hpp>

namespace facebook::velox::cudf_velox {

cudf::ast::expression const& createAstTree(
    const core::TypedExprPtr& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    std::vector<PrecomputeInstruction>& precomputeInstructions,
    memory::MemoryPool* pool) {
  AstContext context{
      tree, scalars, {inputRowSchema}, {precomputeInstructions}, pool, expr};
  return context.pushExprToTree(expr);
}

cudf::ast::expression const& createAstTree(
    const core::TypedExprPtr& expr,
    cudf::ast::tree& tree,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& leftRowSchema,
    const RowTypePtr& rightRowSchema,
    std::vector<PrecomputeInstruction>& leftPrecomputeInstructions,
    std::vector<PrecomputeInstruction>& rightPrecomputeInstructions,
    memory::MemoryPool* pool) {
  AstContext context{
      tree,
      scalars,
      {leftRowSchema, rightRowSchema},
      {leftPrecomputeInstructions, rightPrecomputeInstructions},
      pool,
      expr};
  return context.pushExprToTree(expr);
}

ASTExpression::ASTExpression(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool)
    : expr_(expr), inputRowSchema_(inputRowSchema), pool_(pool) {
  createAstTree(
      expr,
      cudfTree_,
      scalars_,
      inputRowSchema,
      precomputeInstructions_,
      pool_);
}

void ASTExpression::close() {
  cudfTree_ = {};
  scalars_.clear();
  precomputeInstructions_.clear();
}

ColumnOrView ASTExpression::eval(
    std::vector<cudf::column_view> inputColumnViews,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    bool finalize) {
  auto precomputedColumns = precomputeSubexpressions(
      inputColumnViews,
      precomputeInstructions_,
      scalars_,
      inputRowSchema_,
      stream);

  // Make table_view from input columns and precomputed columns
  std::vector<cudf::column_view> allColumnViews(inputColumnViews);
  allColumnViews.reserve(inputColumnViews.size() + precomputedColumns.size());
  for (auto& precomputedCol : precomputedColumns) {
    allColumnViews.push_back(asView(precomputedCol));
  }

  cudf::table_view astInputTableView(allColumnViews);

  auto result = [&]() -> ColumnOrView {
    if (auto colRefPtr = dynamic_cast<cudf::ast::column_reference const*>(
            &cudfTree_.back())) {
      auto columnIndex = colRefPtr->get_column_index();
      if (columnIndex < inputColumnViews.size()) {
        return inputColumnViews[columnIndex];
      } else {
        // Referencing a precomputed column return as it is (view or owned)
        return std::move(
            precomputedColumns[columnIndex - inputColumnViews.size()]);
      }
    } else {
      if (CudfConfig::getInstance().debugEnabled) {
        LOG(INFO) << cudf::ast::expression_to_string(cudfTree_.back());
        LOG(INFO) << cudf::table_schema_to_string(astInputTableView);
      }
      return cudf::compute_column(
          astInputTableView, cudfTree_.back(), stream, mr);
    }
  }();
  if (finalize) {
    const auto requestedType = cudf_velox::veloxToCudfDataType(expr_->type());
    auto resultView = asView(result);
    if (resultView.type() != requestedType) {
      result = cudf::cast(resultView, requestedType, stream, mr);
    }
  }
  return result;
}

bool ASTExpression::canEvaluate(const core::TypedExprPtr& expr) {
  // Keep this in sync with pushExprToTree(); otherwise unsupported field types
  // can recursively select AST/JIT while trying to precompute themselves.
  return detail::isAstExprSupported(expr);
}

namespace {

using SignatureMap =
    std::unordered_map<std::string, std::vector<exec::FunctionSignaturePtr>>;

// Types the AST operations are probed with. DECIMAL takes precision and scale
// parameters and is not probed. Interval and custom types reuse the cuDF type
// of a type listed here, so probing them would only repeat its operations.
const std::vector<TypePtr>& astProbeTypes() {
  static const std::vector<TypePtr> kTypes{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      DATE(),
      TIMESTAMP(),
  };
  return kTypes;
}

std::string typeSignature(const TypePtr& type) {
  return exec::sanitizeName(type->toString());
}

// Returns true if ASTExpression accepts a call to 'name' over input columns of
// 'argumentTypes'. canEvaluate() reads the call's result type only to reject
// TIMESTAMP and DECIMAL, and the signatures built from accepted calls return
// BOOLEAN or the argument type, so the first argument type stands in for the
// result type.
bool astAccepts(
    const std::string& name,
    const std::vector<TypePtr>& argumentTypes) {
  std::vector<core::TypedExprPtr> inputs;
  inputs.reserve(argumentTypes.size());
  for (size_t i = 0; i < argumentTypes.size(); ++i) {
    inputs.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(
            argumentTypes[i], fmt::format("c{}", i)));
  }
  return ASTExpression::canEvaluate(
      std::make_shared<core::CallTypedExpr>(
          argumentTypes.front(), std::move(inputs), name));
}

// Returns the type 'op' produces over 'arity' operands of 'operandType' when it
// is BOOLEAN or the operand type, and nullptr when cuDF produces another type
// (e.g. INTEGER from adding TINYINTs, or a duration from subtracting dates).
// Requires cuDF to support 'op' over these operands.
TypePtr astResultType(
    cudf::ast::ast_operator op,
    const TypePtr& operandType,
    size_t arity) {
  const auto operandCudfType = veloxToCudfDataType(operandType);
  const auto resultCudfType = cudf::ast::detail::ast_operator_return_type(
      op, std::vector<cudf::data_type>(arity, operandCudfType));
  if (resultCudfType.id() == cudf::type_id::BOOL8) {
    return BOOLEAN();
  }
  return resultCudfType == operandCudfType ? operandType : nullptr;
}

exec::FunctionSignaturePtr makeSignature(
    const TypePtr& resultType,
    const std::vector<TypePtr>& argumentTypes) {
  exec::FunctionSignatureBuilder builder;
  builder.returnType(typeSignature(resultType));
  for (const auto& argumentType : argumentTypes) {
    builder.argumentType(typeSignature(argumentType));
  }
  return builder.build();
}

// Lists the calls ASTExpression accepts, keyed by the operation name without
// the function name prefix. Operations are probed with operands of a single
// type because cuDF rejects AST expressions whose operand types differ.
SignatureMap buildAstSignatures() {
  SignatureMap result;
  const auto addOperation =
      [&](const std::string& name, cudf::ast::ast_operator op, size_t arity) {
        for (const auto& type : astProbeTypes()) {
          const std::vector<TypePtr> argumentTypes(arity, type);
          if (!astAccepts(name, argumentTypes)) {
            continue;
          }
          if (auto resultType = astResultType(op, type, arity)) {
            result[name].push_back(makeSignature(resultType, argumentTypes));
          }
        }
      };
  for (const auto& [name, op] : unaryOps) {
    addOperation(name, op, 1);
  }
  for (const auto& [name, op] : binaryOps) {
    addOperation(name, op, 2);
  }

  // Calls that detail::isAstExprSupported() handles by name rather than
  // through the operator maps. All of them return BOOLEAN.
  for (const auto& type : astProbeTypes()) {
    if (astAccepts("isnotnull", {type})) {
      result["isnotnull"].push_back(makeSignature(BOOLEAN(), {type}));
    }
    if (astAccepts("between", {type, type, type})) {
      result["between"].push_back(makeSignature(BOOLEAN(), {type, type, type}));
    }
    if (astAccepts(expression::kIn, {type, ARRAY(type)})) {
      // The IN list must be a constant array.
      result[expression::kIn].push_back(
          exec::FunctionSignatureBuilder()
              .returnType("boolean")
              .argumentType(typeSignature(type))
              .constantArgumentType(
                  fmt::format("array({})", typeSignature(type)))
              .build());
    }
  }
  return result;
}

} // namespace

SignatureMap ASTExpression::signatures() {
  // Built once: what the AST supports does not depend on the configuration,
  // and the signatures must outlive the evaluator registration.
  static const auto kSignatures = buildAstSignatures();

  // The AST matches names with the function name prefix stripped. AND and OR
  // are special forms, which calls name without the prefix.
  const auto& prefix = CudfConfig::getInstance().functionNamePrefix;
  SignatureMap result;
  for (const auto& [name, signatures] : kSignatures) {
    const bool isSpecialForm =
        name == expression::kAnd || name == expression::kOr;
    result[isSpecialForm ? name : prefix + name] = signatures;
  }
  return result;
}

void registerAstEvaluator(int priority) {
  registerCudfExpressionEvaluator(
      kAstEvaluatorName,
      priority,
      [](const core::TypedExprPtr& expr) {
        return ASTExpression::canEvaluate(expr);
      },
      [](const core::TypedExprPtr& expr,
         const RowTypePtr& row,
         memory::MemoryPool* pool) {
        return std::make_shared<ASTExpression>(expr, row, pool);
      },
      /*overwrite=*/false,
      [] { return ASTExpression::signatures(); });
}

} // namespace facebook::velox::cudf_velox
