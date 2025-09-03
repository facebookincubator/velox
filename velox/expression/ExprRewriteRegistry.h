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

#include <folly/Synchronized.h>
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::expression {

/// An expression re-writer that takes an expression and returns an equivalent
/// expression or nullptr if re-write is not possible.
using ExpressionRewrite =
    std::function<core::TypedExprPtr(const core::TypedExprPtr)>;

class ExprRewriteRegistry {
 public:
  /// Adds a 'rewrite' to registry.
  void registerRewrite(ExpressionRewrite rewrite);

  /// Clears the registry to remove all registered rewrites.
  void clear();

  /// Applies re-writes to expression with a simple logic that assumes all
  /// rewrites are independent. Rewrites are applied to the expression in order
  /// they were registered. The first rewrite that returns non-null result
  /// terminates the re-write for the expression. After rewrites are applied,
  /// the expression is constant folded recursively in a bottom-up manner when
  /// 'enableConstantFolding' is 'true'. If a VeloxUserError is encountered
  /// during constant folding, a fail expression with the error message can be
  /// returned instead of the original expression by setting the argument
  /// 'replaceEvalErrorWithFailExpr' to 'true'.
  core::TypedExprPtr rewrite(
      const core::TypedExprPtr& expr,
      core::QueryCtx* queryCtx,
      memory::MemoryPool* pool,
      bool enableConstantFolding = true,
      bool replaceEvalErrorWithFailExpr = false);

  static ExprRewriteRegistry& instance() {
    static ExprRewriteRegistry kInstance;
    return kInstance;
  }

 private:
  folly::Synchronized<std::vector<ExpressionRewrite>> registry_;
};
} // namespace facebook::velox::expression
