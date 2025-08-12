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

namespace facebook::velox::expression {

/// An expression re-writer that takes an expression and returns an equivalent
/// expression or nullptr if re-write is not possible.
using ExpressionRewrite = std::function<core::TypedExprPtr(
    const core::TypedExprPtr&,
    const std::shared_ptr<core::QueryCtx>&,
    memory::MemoryPool*)>;

/// Applies all registered expression rewrites to `expr` sequentially and
/// constant folds the result. Returns a rewritten TypedExpr.
/// Specifically, the function returns:
/// - a new constant expression (if constant folded)
/// - a fail function call expression due to an error during constant
///   evaluation (for example, constant folding the expression 5/0 would result
///   in an error)
/// - the original expression (if it cannot be constant folded, for example,
///   due to presence of a field expression)
/// - nullptr if no rewrite was applied
///
/// Additional fail function details:
/// When the expression is attempted to be constant folded and an error occurs,
/// 5/0 or a BIGINT value can't be cast to TINYINT due to overflow, the fail
/// function may be returned. If that is the case, no further evaluation of the
/// arguments is performed and the fail function is returned for the expression.
/// Expression rewrites are applied during expression compilation in
/// `ExprCompiler` and during expression optimization in `ExprOptimizer`.
core::TypedExprPtr rewriteExpression(
    const core::TypedExprPtr& expr,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    memory::MemoryPool* pool);
} // namespace facebook::velox::expression
