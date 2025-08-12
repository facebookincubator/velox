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

#include "velox/expression/Expr.h"
#include "velox/expression/ExprRewrite.h"
#include "velox/expression/ExprUtils.h"

namespace facebook::velox::expression {

std::vector<core::TypedExprPtr> optimizeExpressions(
    const std::vector<core::TypedExprPtr>& expressions,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  std::vector<core::TypedExprPtr> optimizedExpressions;
  optimizedExpressions.reserve(expressions.size());

  for (auto& expression : expressions) {
    auto rewritten = expression::rewriteExpression(expression, queryCtx, pool);
    auto optimizedExpression =
        utils::constantFold(rewritten ? rewritten : expression, queryCtx, pool);
    optimizedExpressions.push_back(std::move(optimizedExpression));
  }
  return optimizedExpressions;
}

} // namespace facebook::velox::expression
