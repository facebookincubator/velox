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

namespace facebook::velox::expression {

/// An expression re-writer that takes an expression and returns an equivalent
/// expression or nullptr if re-write is not possible.
using ExpressionRewrite =
    std::function<core::TypedExprPtr(const core::TypedExprPtr)>;

/// Applies all registered expression rewrites to `expr`, returns the rewritten
/// expression once a rewrite is successfully applied. Returns `expr` is none
/// of the registered rewrites could be applied.
core::TypedExprPtr rewriteExpression(const core::TypedExprPtr& expr);

} // namespace facebook::velox::expression
