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

template <typename TExec, typename TInput, bool isLeast>
struct ExtremeValueFunction;

template <typename TExec, typename TInput>
using LeastFunction = ExtremeValueFunction<TExec, TInput, true>;

template <typename TExec, typename TInput>
using GreatestFunction = ExtremeValueFunction<TExec, TInput, false>;

/**
 * This class implements two functions:
 *
 * greatest(value1, value2, ..., valueN) → [same as input]
 * Returns the largest of the provided values.
 *
 * least(value1, value2, ..., valueN) → [same as input]
 * Returns the smallest of the provided values.
 **/
template <typename TExec, typename TInput, bool isLeast>
struct ExtremeValueFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  // For double, presto should throw error if input is Nan
  // void checkNan(const TInput& value) const {
  //   if constexpr (std::is_same_v<
  //                     TInput,
  //                     TypeTraits<TypeKind::DOUBLE>::NativeType>) {
  //     if (std::isnan(value)) {
  //       VELOX_USER_FAIL(
  //           "Invalid argument to {}: NaN", isLeast ? "least()" :
  //           "greatest()");
  //     }
  //   }
  // }

  // expect all input to be not null, else the result is null
  FOLLY_ALWAYS_INLINE bool callNullFree(
      out_type<TInput>& result,
      const null_free_arg_type<Variadic<TInput>>& inputs) {
    // ensure that input size is greater than 0
    if (inputs.size() == 0) {
      VELOX_USER_FAIL(
          "Invalid number of argument to {}",
          isLeast ? "least()" : "greatest()");
    }

    auto currentValue = inputs[0];
    // checkNan(currentValue);
    if constexpr (std::is_same_v<
                      TInput,
                      TypeTraits<TypeKind::DOUBLE>::NativeType>) {
      if (std::isnan(currentValue)) {
        VELOX_USER_FAIL(
            "Invalid argument to {}: NaN", isLeast ? "least()" : "greatest()");
      }
    }

    for (auto i = 1; i < inputs.size(); ++i) {
      auto candidateValue = inputs[i];
      // checkNan(candidateValue);
      if constexpr (std::is_same_v<
                        TInput,
                        TypeTraits<TypeKind::DOUBLE>::NativeType>) {
        if (std::isnan(candidateValue)) {
          VELOX_USER_FAIL(
              "Invalid argument to {}: NaN",
              isLeast ? "least()" : "greatest()");
        }
      }

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

    return true;
  }
};

} // namespace facebook::velox::functions
