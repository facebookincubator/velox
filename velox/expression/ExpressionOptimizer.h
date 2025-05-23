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

#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::expression {

bool allInputTypesEquivalent(const core::TypedExprPtr& expr);

void flattenInput(
    const core::TypedExprPtr& input,
    const std::string& flattenCall,
    std::vector<core::TypedExprPtr>& flat);

/// Register expression optimizations for AND, OR, IF, COALESCE.
void registerExpressionOptimizations();

/// Constant fold subtrees of the input expression and return the folded
/// expression.
core::TypedExprPtr constantFold(
    const core::TypedExprPtr& expr,
    const core::QueryConfig& config,
    memory::MemoryPool* pool);
} // namespace facebook::velox::expression
