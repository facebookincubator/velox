/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <string>
#include <variant>
#include "velox/common/Enums.h"
#include "velox/parse/IExpr.h"

namespace facebook::velox::parse {

/// Represents a single ORDER BY key with sort direction and nulls ordering.
struct OrderByClause {
  core::ExprPtr expr;
  bool ascending;
  bool nullsFirst;

  std::string toString() const;
};

/// Parsed aggregate function call, e.g. "sum(a)", "array_agg(x ORDER BY y)",
/// "count(DISTINCT x) FILTER (WHERE y > 0)".
struct AggregateExpr {
  /// Aggregate function call expression.
  core::ExprPtr expr;

  /// Optional ORDER BY clause within the aggregate, e.g.
  /// array_agg(x ORDER BY y DESC).
  std::vector<OrderByClause> orderBy;

  /// True if DISTINCT keyword is present.
  bool distinct{false};

  /// Optional FILTER clause expression, e.g.
  /// count(*) FILTER (WHERE x > 0).
  core::ExprPtr filter;
};

/// Window frame type: ROWS or RANGE.
enum class WindowType { kRows, kRange };

VELOX_DECLARE_ENUM_NAME(WindowType);

/// Window frame bound type.
enum class BoundType {
  kCurrentRow,
  kUnboundedPreceding,
  kUnboundedFollowing,
  kPreceding,
  kFollowing,
};

VELOX_DECLARE_ENUM_NAME(BoundType);

/// Window frame specification, e.g. ROWS BETWEEN 1 PRECEDING AND CURRENT ROW.
struct WindowFrame {
  WindowType type;
  BoundType startType;
  /// Optional bound value expression for kPreceding or kFollowing start.
  core::ExprPtr startValue;
  BoundType endType;
  /// Optional bound value expression for kPreceding or kFollowing end.
  core::ExprPtr endValue;
};

/// Parsed window function expression, e.g.
/// "row_number() OVER (PARTITION BY a ORDER BY b)".
struct WindowExpr {
  /// Window function call expression.
  core::ExprPtr functionCall;

  /// Window frame specification.
  WindowFrame frame;

  /// True if IGNORE NULLS is specified.
  bool ignoreNulls;

  /// PARTITION BY expressions.
  std::vector<core::ExprPtr> partitionBy;

  /// ORDER BY clause within the OVER specification.
  std::vector<OrderByClause> orderBy;
};

class SqlExpressionsParser {
 public:
  virtual ~SqlExpressionsParser() = default;

  /// Parses a single expression. Throws on error.
  virtual core::ExprPtr parseExpr(const std::string& expr) = 0;

  /// Parses a comma-separated list of expressions. Throws on error.
  virtual std::vector<core::ExprPtr> parseExprs(const std::string& expr) = 0;

  /// Parses an expression that represents an ORDER BY clause. Throws on error.
  virtual OrderByClause parseOrderByExpr(const std::string& expr) = 0;

  /// Parses an aggregate function call with optional FILTER, ORDER BY,
  /// and DISTINCT. Throws on error.
  virtual AggregateExpr parseAggregateExpr(const std::string& expr) = 0;

  /// Parses a window function expression. Throws on error.
  virtual WindowExpr parseWindowExpr(const std::string& expr) = 0;

  /// Parses an expression that can be either a scalar expression or a window
  /// function. Returns the appropriate type based on the parsed result.
  virtual std::variant<core::ExprPtr, WindowExpr> parseScalarOrWindowExpr(
      const std::string& expr) = 0;
};

} // namespace facebook::velox::parse
