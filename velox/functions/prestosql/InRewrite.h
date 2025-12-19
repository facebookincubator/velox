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

namespace facebook::velox::functions {

class InRewrite {
 public:
  /// Rewrites IN special form function when the `value` being searched for is
  /// a non-NULL constant and the IN-list is not constant. Returns `true` if
  /// `value` is in the IN-list. Removes constant expressions from IN-list that
  /// are not equal to `value`.
  static core::TypedExprPtr rewrite(
      const core::TypedExprPtr& expr,
      memory::MemoryPool* pool);

  static void registerRewrite();
};

} // namespace facebook::velox::functions
