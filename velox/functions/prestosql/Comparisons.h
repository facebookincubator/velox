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

namespace facebook::velox::functions {
#define VELOX_GEN_BINARY_EXPR(Name, Expr, TResult)                \
  template <typename T>                                           \
  struct Name {                                                   \
    VELOX_DEFINE_FUNCTION_TYPES(T);                               \
    template <typename TInput>                                    \
    FOLLY_ALWAYS_INLINE bool                                      \
    call(TResult& result, const TInput& lhs, const TInput& rhs) { \
      result = (Expr);                                            \
      return true;                                                \
    }                                                             \
  };

VELOX_GEN_BINARY_EXPR(EqFunction, lhs == rhs, bool);
VELOX_GEN_BINARY_EXPR(NeqFunction, lhs != rhs, bool);
VELOX_GEN_BINARY_EXPR(LtFunction, lhs < rhs, bool);
VELOX_GEN_BINARY_EXPR(GtFunction, lhs > rhs, bool);
VELOX_GEN_BINARY_EXPR(LteFunction, lhs <= rhs, bool);
VELOX_GEN_BINARY_EXPR(GteFunction, lhs >= rhs, bool);

#undef VELOX_GEN_BINARY_EXPR

template <typename T>
struct BetweenFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool call(
      bool& result,
      const TInput& value,
      const TInput& low,
      const TInput& high) {
    result = value >= low && value <= high;
    return true;
  }
};

} // namespace facebook::velox::functions
