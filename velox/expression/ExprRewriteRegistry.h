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

#include "folly/Synchronized.h"
#include "velox/expression/ExprRewrite.h"

namespace facebook::velox::expression {

using ExprRewriteRegistryType = std::
    unordered_map<std::string, std::unique_ptr<expression::ExpressionRewrite>>;
class ExpressionRewriteRegistry {
 public:
  /// Appends a 'rewrite' to 'expressionRewrites'.
  ///
  /// The logic that applies re-writes is very simple and assumes that all
  /// rewrites are independent. Re-writes are applied to all expressions
  /// starting at the root and going down the hierarchy. For each expression,
  /// rewrites are applied in the order they were registered. The first rewrite
  /// that returns non-null result terminates the re-write for this particular
  /// expression.
  void registerExpressionRewrite(
      const std::string& name,
      std::unique_ptr<expression::ExpressionRewrite> expressionRewrite);

  /// Clears all registered expression re-writes.
  void unregisterExpressionRewrites();

  /// Retrieves an `ExpressionRewrite` with given name from the registry.
  expression::ExpressionRewrite* FOLLY_NULLABLE
  getExpressionRewrite(const std::string& name) const;

  /// Returns a list of all registered `ExpressionRewrite` names.
  std::vector<std::string> getExpressionRewriteNames() const;

 private:
  folly::Synchronized<ExprRewriteRegistryType> registry_;
};

/// Returns a static instance of `ExpressionRewriteRegistry`.
const ExpressionRewriteRegistry& expressionRewriteRegistry();

/// Registers an `ExpressionRewrite` with given name in the static instance of
/// `ExpressionRewriteRegistry`.
void registerExpressionRewrite(
    const std::string& name,
    std::unique_ptr<expression::ExpressionRewrite> expressionRewrite);

/// Clears all registered `ExpressionRewrite`s from the static instance of
/// `ExpressionRewriteRegistry`.
void unregisterExpressionRewrites();
} // namespace facebook::velox::expression
