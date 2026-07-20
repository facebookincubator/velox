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

#include <string>
#include "velox/parse/Expressions.h"
#include "velox/parse/IExpr.h"

namespace facebook::velox::parse {

/// Represents a single ORDER BY key with sort direction and nulls ordering.
struct OrderByClause {
  core::ExprPtr expr;
  bool ascending;
  bool nullsFirst;

  std::string toString() const;
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
  /// and DISTINCT. Always returns an AggregateCallExpr. Throws on error.
  virtual core::AggregateCallExprPtr parseAggregateExpr(
      const std::string& expr) = 0;

  /// Parses a window function expression. Returns a WindowCallExpr.
  /// Throws on error.
  virtual core::WindowCallExprPtr parseWindowExpr(const std::string& expr) = 0;

  /// Parses a SQL expression that can be either a scalar expression or a
  /// window function. Returns a WindowCallExpr (kWindow kind) for window
  /// functions, or a regular ExprPtr for scalar expressions.
  virtual core::ExprPtr parseScalarOrWindowExpr(const std::string& expr) = 0;
};

} // namespace facebook::velox::parse
