/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

class SwitchRewrite {
 public:
  /// Rewrites SWITCH and IF special form expressions which have the shape
  /// `SWITCH(condition1, value1, condition2, value2, .., conditionN, valueN, ..
  /// .., optional<elseValue>)`, and `IF(condition, trueValue, falseValue)`
  /// respectively. IF expression is rewritten to `trueValue` or `falseValue`
  /// if `condition` is either true or false respectively. The conditions of
  /// SWITCH expression are traversed in order and if the condition is:
  ///  1. false, it is removed from the inputs along with its corresponding
  ///     value.
  ///  2. true: If there are no inputs before this, the corresponding value
  ///     is returned. Otherwise, the corresponding value is the new else value
  ///     and all subsequent inputs are ignored.
  ///  3. not a constant boolean and it cannot be evaluated without input data,
  ///     the condition and its corresponding value are retained in the inputs.
  /// If there are no conditions in the expression now, the else value is
  /// returned if it is present; otherwise NULL is returned. The else value is
  /// added to the inputs otherwise and the optimized SWITCH expression with
  /// pruned inputs is returned.
  static core::TypedExprPtr rewrite(const core::TypedExprPtr& expr);

  static void registerRewrite();
};

} // namespace facebook::velox::expression
