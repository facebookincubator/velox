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

#include "velox/parse/IExpr.h"

namespace facebook::velox::core {

/// Rewrites an IExpr tree bottom-up. Recurses into all constituent expressions
/// of each node (args, filter, orderBy, partitionKeys, frame values), rebuilds
/// the node from the rewritten parts, then applies the transform to the result.
///
/// Usage:
///   // Rename field accesses.
///   auto result = ExprRewriter::rewrite(expr, [&](const ExprPtr& e) {
///     if (auto* field = e->as<FieldAccessExpr>()) {
///       if (field->isRootColumn()) {
///         auto it = mapping.find(field->name());
///         if (it != mapping.end()) {
///           return std::make_shared<FieldAccessExpr>(
///               it->second, field->alias());
///         }
///       }
///     }
///     return e;
///   });
class ExprRewriter {
 public:
  /// Called on each node after its children have been rewritten. Receives
  /// either a leaf node or a node rebuilt from already-transformed children.
  /// Returns the node unchanged or a replacement.
  using Transform = std::function<ExprPtr(const ExprPtr&)>;

  /// Rewrites 'expr' bottom-up: first rewrites all children, rebuilds the
  /// node, then applies 'transform'.
  ///
  /// Lambda and subquery expressions are not recursed into. Lambda arguments
  /// may shadow outer names, requiring scope tracking that this rewriter does
  /// not provide. The transform can handle these directly if needed.
  static ExprPtr rewrite(const ExprPtr& expr, const Transform& transform);
};

} // namespace facebook::velox::core
