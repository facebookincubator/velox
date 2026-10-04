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

namespace facebook::velox::core {
// Named only by reference in the create signature below, so a declaration is
// enough and QueryConfig.h stays out of every evaluator's include graph.
class QueryConfig;
} // namespace facebook::velox::core

namespace facebook::velox::cudf_velox {

class CudfExpression;

using CudfExpressionEvaluatorCanEvaluate =
    std::function<bool(const core::TypedExprPtr& expr)>;
using CudfExpressionEvaluatorCreate =
    std::function<std::shared_ptr<CudfExpression>(
        const core::TypedExprPtr& expr,
        const RowTypePtr& inputRowSchema,
        memory::MemoryPool* pool,
        const core::QueryConfig& config)>;

struct CudfExpressionEvaluatorEntry {
  int priority;
  CudfExpressionEvaluatorCanEvaluate canEvaluate;
  CudfExpressionEvaluatorCreate create;
  /// True when the evaluator applies the session time zone to TIMESTAMP
  /// arguments as Velox does; calls that depend on it go only to these.
  bool honorsSessionTimeZone;
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
    bool honorsSessionTimeZone,
    bool overwrite = true);

/// Registers, under `name`, a predicate reporting calls whose result depends
/// on the session time zone, as a set of GPU function registrations declares
/// it. Replaces a predicate of the same name.
///
/// The predicates live apart from the evaluator entries so that the knowledge
/// outlives the evaluator: which calls read the zone is a property of Velox's
/// functions, which GPU SFI's registrations mirror, not of the evaluator that
/// runs them. With that evaluator disabled, the predicate still keeps such a
/// call away from the evaluators that read TIMESTAMP as UTC, and it stays on
/// the CPU rather than return a UTC answer.
void registerSessionTimeZoneSensitivity(
    const std::string& name,
    CudfExpressionEvaluatorCanEvaluate predicate);

/// True when a registered predicate reports the call's result to depend on
/// the session time zone.
bool isSessionTimeZoneSensitiveCall(const core::TypedExprPtr& expr);

} // namespace facebook::velox::cudf_velox
