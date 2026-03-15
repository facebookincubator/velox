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

#include "velox/expression/Expr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <functional>
#include <memory>
#include <string>
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
    const std::shared_ptr<velox::exec::Expr>& expr)>;

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

bool registerBuiltinFunctions(const std::string& prefix);

// Get the cudf function registry (exposed for metadata collection)
std::unordered_map<std::string, CudfFunctionSpec>& getCudfFunctionRegistry();

// Get function signatures map from the CUDF registry
// Returns a map of function names to their function signatures
std::unordered_map<std::string, std::vector<const exec::FunctionSignature*>>
getCudfFunctionSignatureMap();

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

using CudfExpressionEvaluatorCanEvaluate =
    std::function<bool(std::shared_ptr<velox::exec::Expr> expr)>;
using CudfExpressionEvaluatorCreate =
    std::function<std::shared_ptr<CudfExpression>(
        std::shared_ptr<velox::exec::Expr> expr,
        const RowTypePtr& inputRowSchema)>;

// Register a CudfExpression evaluator.
// - name: unique identifier (e.g., "ast", "function", "my_custom").
// - priority: higher number = higher priority.
// - canEvaluate: shallow check whether evaluator can handle current expr root.
// - create: factory to build the evaluator node.
// - overwrite: replace existing registration with the same name if true.
bool registerCudfExpressionEvaluator(
    const std::string& name,
    int priority,
    CudfExpressionEvaluatorCanEvaluate canEvaluate,
    CudfExpressionEvaluatorCreate create,
    bool overwrite = true);

class FunctionExpression : public CudfExpression {
 public:
  static std::shared_ptr<FunctionExpression> create(
      const std::shared_ptr<velox::exec::Expr>& expr,
      const RowTypePtr& inputRowSchema);

  // TODO (dm): A storage for keeping results in case this is a multiply
  // referenced subexpression (to do CSE)

  ColumnOrView eval(
      std::vector<cudf::column_view> inputColumnViews,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool finalize = false) override;

  void close() override;

  // Check if this specific operation can be evaluated by FunctionExpression
  // (does not recursively check children)
  static bool canEvaluate(std::shared_ptr<velox::exec::Expr> expr);

 private:
  std::shared_ptr<velox::exec::Expr> expr_;
  std::shared_ptr<CudfFunction> function_;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions_;

  RowTypePtr inputRowSchema_;
};

std::shared_ptr<CudfExpression> createCudfExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema,
    std::optional<std::string> except = std::nullopt);

/// Lightweight check if an expression tree is supported by any CUDF evaluator
/// without initializing CudfExpression objects.
/// \param expr Expression to check
/// \param deep If true, recursively check all children in the expression tree;
///             if false, only check if the top-level operation is supported
///             (useful when delegating to subexpressions)
bool canBeEvaluatedByCudf(
    std::shared_ptr<velox::exec::Expr> expr,
    bool deep = true);

} // namespace facebook::velox::cudf_velox
