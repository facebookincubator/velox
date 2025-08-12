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

#include <set>

#include "velox/expression/ExprRewrite.h"
#include "velox/expression/ExprRewriteRegistry.h"

namespace facebook::velox::expression {

core::TypedExprPtr rewriteExpression(
    const core::TypedExprPtr& expr,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool) {
  const auto& rewriteRegistry = exec::expressionRewriteRegistry();
  const auto& rewriteNames = rewriteRegistry.getExpressionRewriteNames();
  for (const auto& name : rewriteNames) {
    auto expressionRewriteFunc = *rewriteRegistry.getExpressionRewrite(name);
    if (auto rewritten = expressionRewriteFunc(expr, queryCtx, pool)) {
      return rewritten;
    }
  }
  return nullptr;
}

} // namespace facebook::velox::expression
