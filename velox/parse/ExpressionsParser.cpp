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
#include "velox/parse/ExpressionsParser.h"

namespace facebook::velox::parse {
namespace {

facebook::velox::duckdb::ParseOptions toDuckOptions(
    const ParseOptions& options) {
  facebook::velox::duckdb::ParseOptions duckOptions;
  duckOptions.parseDecimalAsDouble = options.parseDecimalAsDouble;
  duckOptions.parseIntegerAsBigint = options.parseIntegerAsBigint;
  duckOptions.parseInListAsArray = options.parseInListAsArray;
  duckOptions.functionPrefix = options.functionPrefix;
  return duckOptions;
}
} // namespace

DuckSqlExpressionsParser::DuckSqlExpressionsParser(const ParseOptions& options)
    : options_{toDuckOptions(options)} {}

core::ExprPtr DuckSqlExpressionsParser::parseExpr(const std::string& expr) {
  return facebook::velox::duckdb::parseExpr(expr, options_);
}

std::vector<core::ExprPtr> DuckSqlExpressionsParser::parseExprs(
    const std::string& expr) {
  return facebook::velox::duckdb::parseMultipleExpressions(expr, options_);
}

OrderByClause DuckSqlExpressionsParser::parseOrderByExpr(
    const std::string& expr) {
  return facebook::velox::duckdb::parseOrderByExpr(expr);
}

AggregateExpr DuckSqlExpressionsParser::parseAggregateExpr(
    const std::string& expr) {
  return facebook::velox::duckdb::parseAggregateExpr(expr, options_);
}

WindowExpr DuckSqlExpressionsParser::parseWindowExpr(const std::string& expr) {
  return facebook::velox::duckdb::parseWindowExpr(expr, options_);
}

std::variant<core::ExprPtr, WindowExpr>
DuckSqlExpressionsParser::parseScalarOrWindowExpr(const std::string& expr) {
  return facebook::velox::duckdb::parseScalarOrWindowExpr(expr, options_);
}

} // namespace facebook::velox::parse
