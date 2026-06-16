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

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "velox/expression/FunctionMetadata.h"

namespace facebook::velox {
class BaseVector;
class Type;
class SelectivityVector;
} // namespace facebook::velox

namespace facebook::velox::core {
class QueryConfig;
} // namespace facebook::velox::core

/// Listener infrastructure for observing VectorFunction::apply calls during
/// expression evaluation.
///
/// Listeners are registered globally via registerVectorFunctionListenerFactory.
/// During expression compilation, ExprCompiler iterates all registered
/// factories, calling create() once per resolved scalar function with the
/// function name, metadata, and QueryConfig. Each factory may return a
/// VectorFunctionListeners struct containing pre and/or post listeners, or
/// std::nullopt to skip that function. Returned listeners are stored on the
/// Expr node and invoked around every VectorFunction::apply call in both the
/// standard (applyFunction) and simplified (evalSimplifiedImpl) evaluation
/// paths. Special forms (AND, OR, CAST, etc.) are not subject to listening.
///
/// Multiple factories can be registered independently, each observing
/// different concerns (monitoring, access control, auditing, etc.) without
/// coordination.

namespace facebook::velox::exec {

class EvalCtx;

/// Callback invoked before VectorFunction::apply.
using PreApplyListener = std::function<void(
    std::string_view functionName,
    const SelectivityVector& rows,
    const std::vector<std::shared_ptr<BaseVector>>& args,
    const std::shared_ptr<const Type>& outputType,
    const EvalCtx& context)>;

/// Callback invoked after VectorFunction::apply, regardless of whether apply
/// succeeded or threw. When apply throws, 'error' holds the exception;
/// otherwise it is nullptr. The 'error' exception is always rethrown by the
/// framework after all post-listeners have executed. If a post-listener itself
/// throws, the exception is caught, logged periodically, and remaining
/// post-listeners still execute.
using PostApplyListener = std::function<void(
    std::string_view functionName,
    const SelectivityVector& rows,
    const std::vector<std::shared_ptr<BaseVector>>& args,
    const std::shared_ptr<const Type>& outputType,
    const EvalCtx& context,
    const std::shared_ptr<BaseVector>& result,
    std::exception_ptr error)>;

/// Pre/post listeners returned by VectorFunctionListenerFactory. Stored as
/// shared_ptrs so multiple Expr instances can reference the same listener
/// without copying.
struct VectorFunctionListeners {
  /// Called before vectorFunction_->apply(); may be null.
  std::shared_ptr<const PreApplyListener> pre;
  /// Called after vectorFunction_->apply() regardless of outcome; may be null.
  std::shared_ptr<const PostApplyListener> post;
};

/// Factory that optionally creates pre/post listeners for a given function.
/// Called once per function during expression compilation. Returns
/// std::nullopt to skip listening for that function.
class VectorFunctionListenerFactory {
 public:
  virtual ~VectorFunctionListenerFactory() = default;

  virtual std::optional<VectorFunctionListeners> create(
      std::string_view name,
      const VectorFunctionMetadata& metadata,
      const core::QueryConfig& queryConfig) = 0;
};

/// Registers a listener factory globally. Returns true if the factory was
/// newly added, false if it was already registered.
bool registerVectorFunctionListenerFactory(
    std::shared_ptr<VectorFunctionListenerFactory> factory);

/// Unregisters a previously registered listener factory. Returns true if the
/// factory was found and removed.
bool unregisterVectorFunctionListenerFactory(
    const std::shared_ptr<VectorFunctionListenerFactory>& factory);

/// Iterates all registered factories and collects listeners for the given
/// function. Called by ExprCompiler during expression compilation.
std::vector<VectorFunctionListeners> createVectorFunctionListeners(
    std::string_view name,
    const VectorFunctionMetadata& metadata,
    const core::QueryConfig& queryConfig);

} // namespace facebook::velox::exec
