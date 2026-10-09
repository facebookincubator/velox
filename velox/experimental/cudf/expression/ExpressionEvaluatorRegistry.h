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
#include "velox/expression/FunctionSignature.h"
#include "velox/type/Type.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfExpression;

using CudfExpressionEvaluatorCanEvaluate =
    std::function<bool(const core::TypedExprPtr& expr)>;
using CudfExpressionEvaluatorCreate =
    std::function<std::shared_ptr<CudfExpression>(
        const core::TypedExprPtr& expr,
        const RowTypePtr& inputRowSchema,
        memory::MemoryPool* pool)>;

/// Returns the function signatures an evaluator can evaluate, keyed by the
/// function name calls carry. A name mapped to an empty list is supported, but
/// whether a call is accepted depends on more than its argument types (e.g.
/// cast). The evaluator must keep the signatures alive while it is registered.
using CudfExpressionEvaluatorSignatures = std::function<
    std::unordered_map<std::string, std::vector<exec::FunctionSignaturePtr>>()>;

struct CudfExpressionEvaluatorEntry {
  int priority;
  CudfExpressionEvaluatorCanEvaluate canEvaluate;
  CudfExpressionEvaluatorCreate create;
  // Null when the evaluator does not describe what it supports.
  CudfExpressionEvaluatorSignatures signatures;
};

/// Ensure that built-in expression evaluators are registered.
void ensureBuiltinExpressionEvaluatorsRegistered();

/// Get the registry of expression evaluators.
std::unordered_map<std::string, CudfExpressionEvaluatorEntry>&
getCudfExpressionEvaluatorRegistry();

/// Register a CudfExpression evaluator.
/// Internal API used by expression evaluators to self-register.
/// \param signatures Optional. Lists what the evaluator supports for
///                   getCudfFunctionSignatureMap().
bool registerCudfExpressionEvaluator(
    const std::string& name,
    int priority,
    CudfExpressionEvaluatorCanEvaluate canEvaluate,
    CudfExpressionEvaluatorCreate create,
    bool overwrite = true,
    CudfExpressionEvaluatorSignatures signatures = nullptr);

} // namespace facebook::velox::cudf_velox
