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
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace facebook::velox::core {
class QueryConfig;
class QueryCtx;
} // namespace facebook::velox::core


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

// Throws a VeloxUserError with userMessage if any non-null entry of cond is
// false. cond must be a BOOL8 column. Does nothing for empty or all-null
// columns.
void checkAllTrue(
    cudf::column_view cond,
    std::string_view userMessage,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Carries query-scoped evaluation settings that individual GPU functions need
/// but that are not part of the expression tree, most notably the session
/// timezone. Populated from the QueryConfig at expression-creation time and
/// attached to every CudfFunction so timezone-aware functions can match the CPU
/// path. Defaults represent "no session timezone" (UTC/GMT), matching the CPU
/// behavior when adjust_timestamp_to_session_timezone is off.
struct CudfDateTimeContext {
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

/// Builds a CudfDateTimeContext from the query config, copying the session
/// timezone, the adjust-to-session-timezone flag, and the session start time.
/// Operators that construct cuDF expressions build the context here so the
/// derivation lives in one place and timezone-aware functions match the CPU
/// path.
CudfDateTimeContext contextFromConfig(const core::QueryConfig& config);

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
  void setContext(const CudfDateTimeContext& context) {
    context_ = context;
  }

 protected:
  // Query-scoped evaluation context (session timezone and start time), attached
  // via setContext. Timezone-aware functions read it; others ignore it.
  CudfDateTimeContext context_;
};

using CudfFunctionFactory = std::function<std::shared_ptr<CudfFunction>(
    const std::string& name,
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool)>;

// Optional function-specific eligibility check applied after signature
// matching. Use this for semantic restrictions that cannot be expressed by a
// FunctionSignature. Both FunctionExpression::canEvaluate and
// createCudfFunction apply this filter.
using CudfCanEvaluate = std::function<bool(const core::TypedExprPtr& expr)>;

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
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool,
    const CudfDateTimeContext& context = {});

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
      memory::MemoryPool* pool,
      const CudfDateTimeContext& context = {});

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
  static std::unique_ptr<cudf::column> makeStructChildColumn(
      ColumnOrView& structColumn,
      cudf::size_type childIndex,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  core::TypedExprPtr expr_;
  std::shared_ptr<CudfFunction> function_;
  std::vector<std::shared_ptr<CudfExpression>> subexpressions_;
  // Index of the dereferenced field inside its parent ROW for nested
  // FieldAccess/Dereference expressions. -1 for non-nested or non-field
  // expressions.
  int32_t fieldIndex_{-1};

  RowTypePtr inputRowSchema_;
};

/// Create a CudfExpression from a TypedExpr, selecting the best evaluator.
/// Does not apply expression-level optimization; callers that need
/// optimization should run expression::optimize at the top-level entry point
/// first.
std::shared_ptr<CudfExpression> createCudfExpression(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool,
    const CudfDateTimeContext& context = {});

/// Plan-time GPU eligibility for a top-level operator expression, as invoked by
/// the OperatorAdapters and the aggregation validators. Optimizes the
/// expression (constant folding and rewrites) with `pool` so the check sees the
/// same form the operator compiles at runtime; e.g. cast(<literal> as DECIMAL)
/// folds to a plain decimal constant that the structural check accepts, rather
/// than a live decimal-target cast that it would reject. Then applies a
/// structural support check on the optimized expression. When
/// `queryCtx` or `pool` is null, skips optimization and checks `expr` directly.
/// \param pool Leaf pool used for constant folding during optimization,
///             typically the operator's own pool.
bool canExprRunOnGpu(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool);

/// Extract the full field path from a field access / dereference chain.
/// Returns nullopt for non-field expressions.
std::optional<std::vector<std::string>> extractFieldPath(
    const core::TypedExprPtr& expr);

/// Return the set of top-level input field names referenced by the expression.
std::unordered_set<std::string> referencedInputFields(
    const core::TypedExprPtr& expr);

} // namespace facebook::velox::cudf_velox
