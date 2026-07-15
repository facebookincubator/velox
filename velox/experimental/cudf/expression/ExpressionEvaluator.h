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

namespace facebook::velox::core {
class QueryConfig;
}

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

/// Carries query-scoped evaluation settings that individual GPU functions need
/// but that are not part of the expression tree, most notably the session
/// timezone. Populated from the QueryConfig at expression-creation time and
/// attached to every CudfFunction so timezone-aware functions can match the CPU
/// path. Defaults represent "no session timezone" (UTC/GMT), matching the CPU
/// behavior when adjust_timestamp_to_session_timezone is off.
struct CudfExpressionContext {
  /// Session timezone name (QueryConfig::sessionTimezone), e.g.
  /// "America/Los_Angeles". Empty means none.
  std::string sessionTimezone;
  /// Whether timezone-less timestamp conversions honor the session timezone
  /// (QueryConfig::adjustTimestampToTimezone).
  bool adjustTimestampToTimezone{false};
  /// Session start time in milliseconds since epoch
  /// (QueryConfig::sessionStartTimeMs); used by now()/current_timestamp.
  int64_t sessionStartTimeMs{0};

  /// Returns true when extraction functions must convert the instant to the
  /// session-local wall clock before reading a calendar field.
  bool appliesSessionTimezone() const {
    return adjustTimestampToTimezone && !sessionTimezone.empty();
  }
};

/// Builds a CudfExpressionContext from the query config, copying the session
/// timezone, the adjust-to-session-timezone flag, and the session start time.
/// Operators that construct cuDF expressions build the context here so the
/// derivation lives in one place and timezone-aware functions match the CPU
/// path.
CudfExpressionContext contextFromConfig(const core::QueryConfig& config);

class CudfFunction {
 public:
  virtual ~CudfFunction() = default;
  virtual ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      [[maybe_unused]] cudf::size_type numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const = 0;

  /// Attaches the query-scoped evaluation context. Called once after the
  /// function is created. Functions that do not need it simply ignore context_.
  void setContext(const CudfExpressionContext& context) {
    context_ = context;
  }

 protected:
  // Query-scoped evaluation context (session timezone and start time), attached
  // via setContext. Timezone-aware functions read it; others ignore it.
  CudfExpressionContext context_;
};

using CudfFunctionFactory = std::function<std::shared_ptr<CudfFunction>(
    const std::string& name,
    const std::shared_ptr<velox::exec::Expr>& expr)>;

// Optional function-specific eligibility check applied after signature
// matching. Use this for semantic restrictions that cannot be expressed by a
// FunctionSignature. Both FunctionExpression::canEvaluate and
// createCudfFunction apply this filter.
using CudfCanEvaluate =
    std::function<bool(const std::shared_ptr<velox::exec::Expr>& expr)>;

struct CudfFunctionSpec {
  CudfFunctionFactory factory;
  std::vector<exec::FunctionSignaturePtr> signatures;
  // If set, this must return true before the factory is selected.
  CudfCanEvaluate canEvaluate;
};

bool registerCudfFunction(
    const std::string& name,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true,
    CudfCanEvaluate canEvaluate = nullptr);

void registerCudfFunctions(
    const std::vector<std::string>& aliases,
    CudfFunctionFactory factory,
    const std::vector<exec::FunctionSignaturePtr>& signatures,
    bool overwrite = true,
    CudfCanEvaluate canEvaluate = nullptr);

/// Create a CudfFunction for the given name and expression.
/// Returns nullptr if no registered function matches the expression's
/// signature. The context is attached to the created function so
/// timezone-aware functions can read the session timezone.
std::shared_ptr<CudfFunction> createCudfFunction(
    const std::string& name,
    const std::shared_ptr<velox::exec::Expr>& expr,
    const CudfExpressionContext& context = {});

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

using CudfExpressionEvaluatorCanEvaluate =
    std::function<bool(std::shared_ptr<velox::exec::Expr> expr)>;
using CudfExpressionEvaluatorCreate =
    std::function<std::shared_ptr<CudfExpression>(
        std::shared_ptr<velox::exec::Expr> expr,
        const RowTypePtr& inputRowSchema,
        const CudfExpressionContext& context)>;

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
      const RowTypePtr& inputRowSchema,
      const CudfExpressionContext& context = {});

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
  static std::unique_ptr<cudf::column> makeStructChildColumn(
      ColumnOrView& structColumn,
      cudf::size_type childIndex,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  std::shared_ptr<velox::exec::Expr> expr_;
  std::shared_ptr<CudfFunction> function_;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions_;
  // TODO: Remove once FieldReference can resolve index directly from RowType.
  int32_t fieldIndex_{-1};

  RowTypePtr inputRowSchema_;
};

std::shared_ptr<CudfExpression> createCudfExpression(
    std::shared_ptr<velox::exec::Expr> expr,
    const RowTypePtr& inputRowSchema,
    const CudfExpressionContext& context = {});

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
