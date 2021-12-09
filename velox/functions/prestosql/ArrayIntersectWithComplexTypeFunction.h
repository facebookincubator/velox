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
#include "velox/expression/EvalCtx.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
struct ArrayIntersectWithComplexTypeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<int64_t>>& out,
      const arg_type<Array<Array<int64_t>>>& in) {
    return callByType<int64_t>(out, in);
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<double>>& out,
      const arg_type<Array<Array<double>>>& in) {
    return callByType<double>(out, in);
  }

  template <typename U>
  FOLLY_ALWAYS_INLINE bool callByType(
      out_type<Array<U>>& out,
      const arg_type<Array<Array<U>>>& in) {
    std::unordered_set<U> intersectedSet;
    bool hasNullFinal = false;
    bool isInitialized = false;

    if (in.size() == 1) {
      // special case: if the size of input's innermost array is 1, no need to
      // intersect, just return the only innermost array
      for (const auto& element : *in[0]) {
        out.append(std::optional<U>(element));
      }
      return true;
    }

    for (const auto& innerArrayViewPtr : in) {
      if (!isInitialized) {
        // 1. initialize the intersectedSet with values from the first innermost
        // array
        for (const auto& element : *innerArrayViewPtr) {
          if (element) { // non-null, aka. has_value() is True
            intersectedSet.insert(*element);
          } else {
            hasNullFinal = true;
          }
        }
        isInitialized = true;
      } else {
        // 2. intersect each innermost array into the intersectedSet
        bool hasNullTemp = false;
        std::unordered_set<U> setToIntersect;
        for (const auto& element : *innerArrayViewPtr) {
          if (element) {
            if (intersectedSet.count(*element) > 0) {
              setToIntersect.insert(*element);
            }
          } else {
            hasNullTemp = true;
          }
        }
        intersectedSet = setToIntersect;
        hasNullFinal &= hasNullTemp;
      }
    }

    // 3. populate the result elements from set to the final output Array
    if (hasNullFinal) {
      out.append(std::nullopt);
    }
    for (const auto& element : intersectedSet) {
      out.append(element);
    }
    // 3.1 ensuring output ordering
    std::sort(out.begin(), out.end());

    return true;
  }
};

} // namespace
} // namespace facebook::velox::functions
