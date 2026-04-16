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

#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"

#include "velox/core/Expressions.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace facebook::velox::cudf_velox {

// Holds either a non-owning cudf::column_view (zero-copy) or an owning
// cudf::column (materialised result).
using ColumnOrView =
    std::variant<cudf::column_view, std::unique_ptr<cudf::column>>;

// Helper to always obtain a column_view.
inline cudf::column_view asView(ColumnOrView& holder) {
  return std::visit(
      [](auto& h) -> cudf::column_view {
        using T = std::decay_t<decltype(h)>;
        if constexpr (std::is_same_v<T, cudf::column_view>) {
          return h;
        } else {
          return h->view();
        }
      },
      holder);
}

// Helper to convert a table_view to a vector of column_views.
inline std::vector<cudf::column_view> tableViewToColumnViews(
    cudf::table_view tableView) {
  std::vector<cudf::column_view> result;
  result.reserve(tableView.num_columns());
  for (cudf::size_type i = 0; i < tableView.num_columns(); ++i) {
    result.push_back(tableView.column(i));
  }
  return result;
}

class CudfFunction {
 public:
  virtual ~CudfFunction() = default;
  virtual ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const = 0;
};

using CudfFunctionFactory = std::function<std::shared_ptr<CudfFunction>(
    const std::string& name,
    const core::TypedExprPtr& expr)>;

struct CudfFunctionSpec {
  CudfFunctionFactory factory;
  std::vector<exec::FunctionSignaturePtr> signatures;
};

bool registerCudfFunction(
    const std::string& name,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true);

void registerCudfFunctions(
    const std::vector<std::string>& aliases,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true);

/// Create a CudfFunction for the given name and expression.
/// Returns nullptr if no registered function matches the expression's
/// signature.
std::shared_ptr<CudfFunction> createCudfFunction(
    const std::string& name,
    const core::TypedExprPtr& expr);

bool registerBuiltinFunctions(const std::string& prefix);

void unregisterFunctions();

class CudfExpression {
 public:
  virtual ~CudfExpression() = default;
  virtual void close() = 0;

  virtual ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false) = 0;
};

using CudfExpressionPtr = std::shared_ptr<CudfExpression>;

class FunctionExpression : public CudfExpression {
 public:
  static std::shared_ptr<FunctionExpression> create(
      const core::TypedExprPtr& expr,
      const RowTypePtr& inputRowSchema,
      CudfExprCtx exprCtx);

  ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false) override;

  void close() override;

  /// Check if this specific operation can be evaluated by FunctionExpression.
  /// Does not recursively check children.
  static bool canEvaluate(const core::TypedExprPtr& expr);

 private:
  core::TypedExprPtr expr_;
  std::shared_ptr<CudfFunction> function_;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions_;

  RowTypePtr inputRowSchema_;
};

/// Create a CudfExpression from a TypedExpr, selecting the best evaluator.
/// Delegates to CudfExpressionCompiler::compileSubExpression and does not
/// optimize the expression. Prefer constructing a CudfExpressionCompiler
/// directly in operator code and calling compile() at top-level entry points.
std::shared_ptr<CudfExpression> createCudfExpression(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    CudfExprCtx exprCtx);

/// Lightweight check if an expression tree is supported by any CUDF evaluator
/// without initializing CudfExpression objects.
/// \param expr Expression to check
/// \param deep If true, recursively check all children in the expression tree;
///             if false, only check if the top-level operation is supported
///             (useful when delegating to subexpressions)
bool canBeEvaluatedByCudf(const core::TypedExprPtr& expr, bool deep = true);

bool canBeEvaluatedByCudf(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool deep = true);

/// Return the best CudfExpressionEvaluatorEntry for the given expression,
/// or nullptr if no evaluator can handle it.
const CudfExpressionEvaluatorEntry* findBestEvaluator(
    const core::TypedExprPtr& expr);

/// Extract the full field path from a field access / dereference chain.
/// Returns nullopt for non-field expressions.
inline std::optional<std::vector<std::string>> extractFieldPath(
    const core::TypedExprPtr& expr) {
  if (expr == nullptr) {
    return std::nullopt;
  }

  if (expr->isFieldAccessKind()) {
    const auto* field = expr->asUnchecked<core::FieldAccessTypedExpr>();
    if (field->inputs().empty() || field->inputs()[0]->isInputKind()) {
      return std::vector<std::string>{field->name()};
    }

    auto path = extractFieldPath(field->inputs()[0]);
    if (!path.has_value()) {
      return std::nullopt;
    }
    path->push_back(field->name());
    return path;
  }

  if (expr->isDereferenceKind()) {
    const auto* dereference = expr->asUnchecked<core::DereferenceTypedExpr>();
    auto path = extractFieldPath(dereference->inputs()[0]);
    if (!path.has_value()) {
      return std::nullopt;
    }
    path->push_back(dereference->name());
    return path;
  }

  return std::nullopt;
}

/// Return the root (top-level) field name, or nullopt.
inline std::optional<std::string> rootFieldName(
    const core::TypedExprPtr& expr) {
  auto path = extractFieldPath(expr);
  if (!path.has_value() || path->empty()) {
    return std::nullopt;
  }
  return path->front();
}

/// True if the expression is a direct input field reference (possibly nested).
inline bool isInputFieldReference(const core::TypedExprPtr& expr) {
  return rootFieldName(expr).has_value();
}

/// Collect all top-level input field names referenced by an expression tree.
inline void collectReferencedInputFields(
    const core::TypedExprPtr& expr,
    std::unordered_set<std::string>& fields,
    const std::unordered_set<std::string>& lambdaInputs = {}) {
  if (expr == nullptr) {
    return;
  }

  if (auto root = rootFieldName(expr);
      root.has_value() && !lambdaInputs.count(*root)) {
    fields.insert(*root);
  }

  if (expr->isLambdaKind()) {
    const auto* lambda = expr->asUnchecked<core::LambdaTypedExpr>();
    auto scopedLambdaInputs = lambdaInputs;
    for (const auto& name : lambda->signature()->names()) {
      scopedLambdaInputs.insert(name);
    }
    collectReferencedInputFields(lambda->body(), fields, scopedLambdaInputs);
    return;
  }

  for (const auto& input : expr->inputs()) {
    collectReferencedInputFields(input, fields, lambdaInputs);
  }
}

/// Return the set of top-level input field names referenced by the expression.
inline std::unordered_set<std::string> referencedInputFields(
    const core::TypedExprPtr& expr) {
  std::unordered_set<std::string> fields;
  collectReferencedInputFields(expr, fields);
  return fields;
}

} // namespace facebook::velox::cudf_velox
