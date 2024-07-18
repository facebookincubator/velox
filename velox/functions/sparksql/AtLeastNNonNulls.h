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

template <typename T>
struct AtLeastNNonNullsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void callNullable(
      out_type<bool>& out,
      const arg_type<int32_t>* n,
      const arg_type<Variadic<Any>>* inputs) {
    out = getNonNullAndNanCount(inputs) >= *n;
  }

 private:
  int32_t getNonNullAndNanCount(const arg_type<Variadic<Any>>* inputs) {
    int32_t result = 0;
    for (const auto& input : *inputs) {
      if (input.has_value()) {
        switch (input.value().kind()) {
          case TypeKind::REAL:
            if (!std::isnan(input.value().template castTo<float>())) {
              result++;
            }
            break;
          case TypeKind::DOUBLE:
            if (!std::isnan(input.value().template castTo<double>())) {
              result++;
            }
            break;
          default:
            result++;
        }
      }
    }
    return result;
  }
};
} // namespace facebook::velox::functions::sparksql
