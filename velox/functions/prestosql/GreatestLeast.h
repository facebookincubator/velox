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

#include <functions/Macros.h>
#include <cmath>
#include <type_traits>
#include "velox/common/base/Exceptions.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions {

template <typename TExec, typename T, bool isLeast>
struct ExtremeValueFunction;

template <typename TExec, typename T>
using LeastFunction = ExtremeValueFunction<TExec, T, true>;

template <typename TExec, typename T>
using GreatestFunction = ExtremeValueFunction<TExec, T, false>;

/**
 * This class implements two functions:
 *
 * greatest(value1, value2, ..., valueN) → [same as input]
 * Returns the largest of the provided values.
 *
 * least(value1, value2, ..., valueN) → [same as input]
 * Returns the smallest of the provided values.
 **/
template <typename TExec, typename T, bool isLeast>
struct ExtremeValueFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE void call(
      out_type<T>& result,
      const arg_type<T>& firstElement,
      const arg_type<Variadic<T>>& remainingList) {
    auto currentValue = firstElement;

    for (auto i = 0; i < remainingList.size(); ++i) {
      VELOX_USER_CHECK(remainingList[i].has_value());
      auto candidateValue = remainingList[i].value();

      if constexpr (isLeast) {
        if (candidateValue < currentValue) {
          currentValue = candidateValue;
        }
      } else {
        if (candidateValue > currentValue) {
          currentValue = candidateValue;
        }
      }
    }

    result = currentValue;
  }
};

} // namespace facebook::velox::functions
