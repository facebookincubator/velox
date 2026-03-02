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

namespace facebook::velox::expression {

/// A callback invoked if execution failure is encountered during constant
/// folding. The callback receives an error message and the result type of the
/// expression that fails. The callback is expected to produce a replacement
/// expression with the same result type. For example, Presto returns
/// `cast(fail(errorMessage) as resultType)`.
using MakeFailExpr = std::function<core::TypedExprPtr(
    const std::string& errorMessage,
    const TypePtr& resultType)>;

/// Optimizes expression through a combination of constant folding and rewrites;
/// all possible subtrees of the expression are constant folded first in a
/// bottom up manner, then the folded expression is rewritten. If an exception
/// (i.e VeloxUserError) is encountered during constant folding of any subtree
/// of the expression and the function `makeFailExpr` is provided, the failing
/// subexpression is replaced by the result of applying `makeFailExpr`, allowing
/// users to define custom logic for handling constant folding errors during
/// expression optimization. When `makeFailExpr` is not provided, exceptions
/// encountered while constant folding subexpressions are ignored and failing
/// subexpressions are left unchanged in the expression tree.
core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    const MakeFailExpr& makeFailExpr = nullptr);
} // namespace facebook::velox::expression
