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
#include "folly/container/F14Set.h"
#include "velox/functions/Udf.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct ConcatWsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Variadic<Varchar>>& inputs) {
    concatWsImpl(result, inputs);
  }
  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Array<Varchar>>& inputs) {
    concatWsImpl(result, inputs);
  }
  template <class Tout, class Tin>
  void concatWsImpl(Tout& result, const Tin& inputs) {
    if (!inputs[0]) {
      return;
    }
    bool first = true;
    for (int i = 1; i < inputs.size(); ++i) {
      if (inputs[i]) {
        if (first) {
          first = false;
        } else {
          result.append(inputs[0].value());
        }
        result.append(inputs[i].value());
      }
    }
  }
};
} // namespace facebook::velox::functions::sparksql
