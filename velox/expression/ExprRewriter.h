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
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::exec {

/// An expression re-writer that takes an expression and returns an equivalent
/// expression or nullptr if re-write is not possible.
using ExpressionRewrite = std::function<core::TypedExprPtr(
    const core::TypedExprPtr,
    const std::shared_ptr<core::QueryCtx>&,
    memory::MemoryPool*)>;

/// Returns a list of registered re-writes.
std::vector<ExpressionRewrite>& expressionRewrites();

/// Appends a 'rewrite' to 'expressionRewrites'.
///
/// The logic that applies re-writes is very simple and assumes that all
/// rewrites are independent. Re-writes are applied to all expressions starting
/// at the root and going down the hierarchy. For each expression, rewrites are
/// applied in the order they were registered. The first rewrite that returns
/// non-null result terminates the re-write for this particular expression.
void registerExpressionRewrite(ExpressionRewrite rewrite);

/// Clears all registered expression re-writes.
void unregisterExpressionRewrites();

/// Applies all registered expression rewrites to `expr` sequentially. Returns a
/// rewritten TypedExpr. Expression rewrites can be used to perform logical
/// optimizations, such as simplifying `AND(orderkey, false)` to `false`.
/// Expression rewrites are applied during expression compilation in
/// `ExprCompiler` and during expression optimization in `ExprOptimizer`.
core::TypedExprPtr rewriteExpression(
    const core::TypedExprPtr& expr,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool);
} // namespace facebook::velox::exec
