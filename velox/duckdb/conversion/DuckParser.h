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
#include "velox/parse/SqlExpressionsParser.h"

namespace facebook::velox::duckdb {

/// Hold parsing options.
struct ParseOptions {
  // Retain legacy behavior by default.
  bool parseDecimalAsDouble = true;
  bool parseIntegerAsBigint = true;
  // Whether to parse the values in an IN list as separate arguments or as a
  // single array argument.
  bool parseInListAsArray = true;

  // DuckDB defaults the window frame end bound to CURRENT ROW even when ORDER
  // BY is absent. The SQL standard requires UNBOUNDED FOLLOWING in that case.
  // When true, corrects this default. Cannot distinguish defaulted from
  // explicit frames, so an explicit CURRENT ROW may be incorrectly overridden.
  bool correctWindowFrameDefault = false;

  /// SQL functions could be registered with different prefixes by the user.
  /// This parameter is the registered prefix of presto or spark functions,
  /// which helps generate the correct Velox expression.
  std::string functionPrefix;
};

// Parses an input expression using DuckDB's internal postgresql-based parser,
// converting it to an IExpr tree. Takes a single expression as input.
//
// One caveat to keep in mind when using DuckDB's parser is that all identifiers
// are lower-cased, what prevents you to use functions and column names
// containing upper case letters (e.g: "concatRow" will be parsed as
// "concatrow").
core::ExprPtr parseExpr(
    const std::string& exprString,
    const ParseOptions& options);

std::vector<core::ExprPtr> parseMultipleExpressions(
    const std::string& exprString,
    const ParseOptions& options);

/// Parses aggregate function call expression with optional ORDER by clause.
/// Always returns an AggregateCallExpr.
/// Examples:
///     sum(a)
///     sum(a) as s
///     array_agg(x ORDER BY y DESC)
core::AggregateCallExprPtr parseAggregateExpr(
    const std::string& exprString,
    const ParseOptions& options);

// Parses an ORDER BY clause using DuckDB's internal postgresql-based parser.
// Uses ASC NULLS LAST as the default sort order.
parse::OrderByClause parseOrderByExpr(const std::string& exprString);

/// Parses a WINDOW function SQL string. Returns a WindowCallExpr.
core::WindowCallExprPtr parseWindowExpr(
    const std::string& windowString,
    const ParseOptions& options);

/// Parses a SQL expression that can be either a scalar expression or a window
/// function. Returns a WindowCallExpr (kWindow kind) for window functions,
/// or a regular ExprPtr for scalar expressions.
core::ExprPtr parseScalarOrWindowExpr(
    const std::string& exprString,
    const ParseOptions& options);

} // namespace facebook::velox::duckdb
