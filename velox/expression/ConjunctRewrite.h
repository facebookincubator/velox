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
#include "velox/core/Expressions.h"

namespace facebook::velox::expression {

class ConjunctRewrite {
 public:
  /// Rewrites special forms AND and OR. First, nested AND, OR special forms
  /// are flattened recursively, eg: a AND (b AND (c AND d)) -> AND(a, b, c, d).
  /// Then, tries to short circuit expression evaluation by rewriting AND and OR
  /// special forms to `false` and `true` respectively if this is one of
  /// the inputs. If expression could not be short-circuited, inputs to AND and
  /// OR conjuncts that are `true` and `false` respectively are pruned. If there
  /// is only one input to the conjunct after its inputs are pruned, rewrites
  /// the conjunct expression to this input. Otherwise, returns an optimized
  /// conjunct expression with pruned inputs.
  static core::TypedExprPtr rewrite(const core::TypedExprPtr& expr);

  static void registerRewrite();
};

} // namespace facebook::velox::expression
