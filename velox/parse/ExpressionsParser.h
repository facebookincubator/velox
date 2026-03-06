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

#include "velox/duckdb/conversion/DuckParser.h"
#include "velox/parse/SqlExpressionsParser.h"

namespace facebook::velox::parse {

/// Hold parsing options.
struct ParseOptions {
  // Retain legacy behavior by default.
  bool parseDecimalAsDouble = true;
  bool parseIntegerAsBigint = true;
  // Controls whether to parse the values in an IN list as separate arguments or
  // as a single array argument.
  bool parseInListAsArray = true;

  /// SQL functions could be registered with different prefixes by the user.
  /// This parameter is the registered prefix of presto or spark functions,
  /// which helps generate the correct Velox expression.
  std::string functionPrefix;
};

/// Parses SQL expressions using DuckDB SQL dialect:
/// https://duckdb.org/docs/stable/sql/dialect/overview
class DuckSqlExpressionsParser : public SqlExpressionsParser {
 public:
  DuckSqlExpressionsParser(const ParseOptions& options = {});

  core::ExprPtr parseExpr(const std::string& expr) override;

  std::vector<core::ExprPtr> parseExprs(const std::string& expr) override;

  OrderByClause parseOrderByExpr(const std::string& expr) override;

  AggregateExpr parseAggregateExpr(const std::string& expr) override;

  WindowExpr parseWindowExpr(const std::string& expr) override;

  std::variant<core::ExprPtr, WindowExpr> parseScalarOrWindowExpr(
      const std::string& expr) override;

 private:
  const facebook::velox::duckdb::ParseOptions options_;
};

} // namespace facebook::velox::parse
