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

#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

/// array_append(array(E), element) -> array(E)
/// Given an array and another element, append the element at the end of the
/// array.
template <typename TExec>
struct ArrayAppendFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Array<Generic<T1>>>* array,
      const arg_type<Generic<T1>>* element) {
    if (array == nullptr) {
      return false;
    }
    out.reserve(array->size() + 1);
    out.add_items(*array);
    element ? out.push_back(*element) : out.add_null();
    return true;
  }
};

} // namespace facebook::velox::functions::sparksql
