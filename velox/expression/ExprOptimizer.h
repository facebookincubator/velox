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

/// Optimizes expression through a combination of constant folding and rewrites.
/// Constant folds all possible subtrees of the expression and the expression
/// itself, if possible, then rewrites the folded expression. If an exception
/// (i.e VeloxUserError) is encountered during constant folding, a fail
/// expression with the error message will be returned instead of the original
/// expression when 'replaceEvalErrorWithFailExpr' is set to 'true'.
core::TypedExprPtr optimize(
    const core::TypedExprPtr& expr,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    bool replaceEvalErrorWithFailExpr);
} // namespace facebook::velox::expression
