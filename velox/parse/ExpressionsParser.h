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

#include <set>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/parse/IExpr.h"

namespace facebook::velox::parse {

/// Hold parsing options.
struct ParseOptions {
  // Retain legacy behavior by default.
  bool parseDecimalAsDouble = true;
  bool parseIntegerAsBigint = true;
};

std::shared_ptr<const core::IExpr> parseExpr(
    const std::string& expr,
    const ParseOptions& options);

std::pair<std::shared_ptr<const core::IExpr>, core::SortOrder> parseOrderByExpr(
    const std::string& expr);

std::vector<std::shared_ptr<const core::IExpr>> parseMultipleExpressions(
    const std::string& expr,
    const ParseOptions& options);

} // namespace facebook::velox::parse
