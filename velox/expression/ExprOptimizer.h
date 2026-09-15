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

#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::core {
class ExpressionEvaluator;
}

namespace facebook::velox::expression {

/// A callback invoked if execution failure is encountered during constant
/// folding. The callback receives an error message and the result type of the
/// expression that fails. The callback is expected to produce a replacement
/// expression with the same result type. For example, Presto returns
/// `cast(fail(errorMessage) as resultType)`.
using MakeFailExpr = std::function<core::TypedExprPtr(
    const std::string& errorMessage,
    const TypePtr& resultType)>;

/// Optimizes an expression through a combination of constant folding and
/// rewrites; all possible subtrees are constant folded bottom-up, then the
/// folded expression is rewritten. Constant subtrees are folded with
/// `exec::tryEvaluateConstantExpression` using `queryCtx` and `pool`. If a
/// `VeloxUserError` is encountered while folding a subtree and `makeFailExpr`
/// is provided, the failing subexpression is replaced by the result of applying
/// `makeFailExpr`; otherwise exceptions are ignored and the subexpression is
/// left unchanged.
core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    const MakeFailExpr& makeFailExpr = nullptr);

/// Variant of `optimize` that folds constant subtrees with `evaluator` (e.g.
/// the ExpressionEvaluator a connector exposes for pushed down filters), for
/// callers that have an ExpressionEvaluator but no QueryCtx.
core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    const MakeFailExpr& makeFailExpr = nullptr);
} // namespace facebook::velox::expression
