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

class CoalesceRewrite {
 public:
  /// Rewrites COALESCE special form expression. Nested COALESCE expressions are
  /// first flattened in a recursive manner, eg: COALESCE(a, COALESCE(b, c)) ->
  /// COALESCE(a, b, c). The inputs to COALESCE are then traversed in order and:
  ///  1. If the input is constant:
  ///     a. and it is NULL, the input is removed.
  ///     b. and if there are no inputs before this, the COALESCE expression is
  ///        simplified to this constant input.
  ///     c. otherwise, all subsequent inputs after this are removed.
  ///  2. Otherwise, if the same input is seen before, it is removed i.e non
  ///     constant inputs to COALESCE are deduplicated.
  /// If there is a single input to COALESCE after pruning inputs, COALESCE is
  /// simplified to this expression. Otherwise, returns an optimized COALESCE
  /// expression with pruned inputs.
  static core::TypedExprPtr rewrite(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* /*pool*/);

  static void registerRewrite();
};

} // namespace facebook::velox::expression
