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

#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

namespace facebook::velox::cudf_velox {

/// Selects the best cuDF evaluator for the given (already-optimized) expression
/// and creates the corresponding CudfExpression.  Use for recursive evaluator
/// hand-off, and for callers that have already optimized the expression because
/// they also need the optimized tree (e.g. hash join's two-table AST checks and
/// the data source's read-column collection).
std::shared_ptr<CudfExpression> compile(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    memory::MemoryPool* pool);

/// Optimizes the expression (rewrites + constant folding through `queryCtx`)
/// and compiles the result.  Use for top-level operator expressions that do not
/// need the optimized tree separately.
std::shared_ptr<CudfExpression> optimizeAndCompile(
    const core::TypedExprPtr& expr,
    const RowTypePtr& inputRowSchema,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool);

} // namespace facebook::velox::cudf_velox
