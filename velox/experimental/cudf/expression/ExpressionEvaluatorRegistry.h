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
#include "velox/type/Type.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

class CudfExpression;
struct CudfDateTimeContext;

using CudfExpressionEvaluatorCanEvaluate =
    std::function<bool(const core::TypedExprPtr& expr)>;
// The date/time context is threaded through creation so that evaluators which
// build child expressions (AST precompute) pass the session timezone down
// instead of dropping it; a child created with a default context would evaluate
// in UTC while its sibling honored the session zone.
using CudfExpressionEvaluatorCreate =
    std::function<std::shared_ptr<CudfExpression>(
        const core::TypedExprPtr& expr,
        const RowTypePtr& inputRowSchema,
        memory::MemoryPool* pool,
        const CudfDateTimeContext& context)>;

struct CudfExpressionEvaluatorEntry {
  int priority;
  CudfExpressionEvaluatorCanEvaluate canEvaluate;
  CudfExpressionEvaluatorCreate create;
};

/// Ensure that built-in expression evaluators are registered.
void ensureBuiltinExpressionEvaluatorsRegistered();

/// Get the registry of expression evaluators.
std::unordered_map<std::string, CudfExpressionEvaluatorEntry>&
getCudfExpressionEvaluatorRegistry();

/// Register a CudfExpression evaluator.
/// Internal API used by expression evaluators to self-register.
bool registerCudfExpressionEvaluator(
    const std::string& name,
    int priority,
    CudfExpressionEvaluatorCanEvaluate canEvaluate,
    CudfExpressionEvaluatorCreate create,
    bool overwrite = true);

} // namespace facebook::velox::cudf_velox
