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
#include "velox/duckdb/conversion/DuckParser.h"
#include "velox/external/duckdb/duckdb.hpp"

namespace facebook::velox::parse {

std::shared_ptr<const core::IExpr> parseExpr(
    const std::string& expr,
    const ParseOptions& options) {
  ::duckdb::ParserOptions duckdbOptions;
  duckdbOptions.parse_decimal_as_double = options.parseDecimalAsDouble;
  duckdbOptions.preserve_identifier_case = options.preserveIdentifierCase;
  duckdbOptions.max_expression_depth = options.maxExpressionDepth;
  return facebook::velox::duckdb::parseExpr(expr, duckdbOptions);
}

std::pair<std::shared_ptr<const core::IExpr>, core::SortOrder> parseOrderByExpr(
    const std::string& expr) {
  return facebook::velox::duckdb::parseOrderByExpr(expr);
}

} // namespace facebook::velox::parse
