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

#include "velox/functions/Udf.h"

namespace facebook::velox::functions::sparksql {

/// This class implements the array insert function.
///
/// DEFINITION:
/// array_insert(array(E), pos, E) → array(E)
/// Places new element into index pos of the input array.
template <typename T>
struct ArrayInsertFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T)

  // INT_MAX - 15, keep the same limit with spark.
  static constexpr int32_t kMaxNumberOfElements = 2147483632;

  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Array<Generic<T1>>>* srcArray,
      const arg_type<int32_t>* position,
      const arg_type<Generic<T1>>* element,
      const arg_type<bool>* legacyNegativeIndex) {
    VELOX_USER_CHECK_NOT_NULL(
      legacyNegativeIndex, "Parameter legacyNegativeIndex should not be NULL.")
    if (srcArray == nullptr || position == nullptr) {
      return false;
    }
    VELOX_USER_CHECK(
      *position != 0, "Array insert position should not be 0.")

    if (*position > 0) {
      int64_t newArrayLength = std::max(srcArray->size() + 1, *position);
      VELOX_USER_CHECK_LE(
        newArrayLength,
        kMaxNumberOfElements,
        "Array insert result exceeds the max array size limit {}",
        kMaxNumberOfElements);

      out.reserve(newArrayLength);
      int32_t posIdx = *position - 1;
      int32_t nextIdx = 0;
      for (const auto& item : *srcArray) {
        if (nextIdx == posIdx) {
          element ? out.push_back(*element) : out.add_null();
          nextIdx++;
        }
        out.push_back(item);
        nextIdx++;
      }
      while(nextIdx < newArrayLength) {
        if (nextIdx == posIdx) {
          element ? out.push_back(*element) : out.add_null();
        } else {
          out.add_null();
        }
        nextIdx++;
      }
    } else {
      bool newPosExtendsArrayLeft = (*position < 0) && (-*position > srcArray->size());
      if (newPosExtendsArrayLeft) {
        int32_t baseOffset = *legacyNegativeIndex ? 1 : 0;
        int64_t newArrayLength = -*position + baseOffset;
        VELOX_USER_CHECK_LE(
          newArrayLength,
          kMaxNumberOfElements,
          "Array insert result exceeds the max array size limit {}",
          kMaxNumberOfElements);
        out.reserve(newArrayLength);
        element ? out.push_back(*element) : out.add_null();
        int64_t nullsToFill = newArrayLength - 1 - srcArray->size();
        while (nullsToFill > 0)
        {
          out.add_null();
          nullsToFill--;
        }
        for (const auto& item : *srcArray) {
          out.push_back(item);
        }
      } else {
        int64_t posIdx = *position + srcArray->size() + (*legacyNegativeIndex ? 0 : 1);
        int64_t newArrayLength = std::max(int64_t(srcArray->size() + 1), posIdx + 1);
        VELOX_USER_CHECK_LE(
          newArrayLength,
          kMaxNumberOfElements,
          "Array insert result exceeds the max array size limit {}",
          kMaxNumberOfElements);
        out.reserve(newArrayLength);

        int32_t nextIdx = 0;
        for (const auto& item : *srcArray) {
          if (nextIdx == posIdx) {
            element ? out.push_back(*element) : out.add_null();
            nextIdx++;
          }
          out.push_back(item);
          nextIdx++;
        }
        if (nextIdx < newArrayLength) {
          element ? out.push_back(*element) : out.add_null();
        }
      }
    }

    return true;
  }
};
} // namespace facebook::velox::functions::sparksql
