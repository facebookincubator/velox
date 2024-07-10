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

/// array_insert(array(E), pos, E, bool) → array(E)
/// Places new element into index pos of the input array.
template <typename T>
struct ArrayInsertFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T)

  // INT_MAX - 15, keep the same limit with spark.
  static constexpr int32_t kMaxNumberOfElements = 2'147'483'632;

  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<Array<Generic<T1>>>& out,
      const arg_type<Array<Generic<T1>>>* srcArray,
      const arg_type<int32_t>* pos,
      const arg_type<Generic<T1>>* item,
      const arg_type<bool>* legacyNegativeIndex) {
    VELOX_USER_CHECK_NOT_NULL(
        legacyNegativeIndex,
        "Parameter legacyNegativeIndex should not be NULL.")
    if (srcArray == nullptr || pos == nullptr) {
      return false;
    }
    VELOX_USER_CHECK(*pos != 0, "Array insert position should not be 0.")

    if (*pos > 0) {
      // Insert element into index *pos of the input array, append nulls after the original
      // elements if target postion is above the input array size.
      int64_t newArrayLength = std::max((int64_t)srcArray->size() + 1, (int64_t)*pos);
      VELOX_USER_CHECK_LE(
          newArrayLength,
          kMaxNumberOfElements,
          "Array insert result exceeds the max array size limit {}",
          kMaxNumberOfElements);

      out.reserve(newArrayLength);
      int32_t posIdx = *pos - 1;
      int32_t nextIdx = 0;
      for (int32_t i = 0; i < newArrayLength; i++) {
        if (i == posIdx) {
          item ? out.push_back(*item) : out.add_null();
        } else {
          bool inserted = i > posIdx;
          int32_t srcIdx = i - inserted;
          if (srcIdx < srcArray->size()) {
            out.push_back((*srcArray)[srcIdx]);
          } else {
            out.add_null();
          }
        }
      }
    } else {
      bool newPosExtendsArrayLeft = -(int64_t)(*pos) > srcArray->size();
      if (newPosExtendsArrayLeft) {
        // Insert element at the beginning of the array followed by nulls and the original array.
        // The new array size depends on legacyNegativeIndex, if legacyNegativeIndex is true the
        // index is 0-based and the new array size is (-*pos + 1), otherwise it's 1-based and the
        // new array size is -*pos. 
        int64_t newArrayLength = -(int64_t)(*pos) + *legacyNegativeIndex;
        VELOX_USER_CHECK_LE(
            newArrayLength,
            kMaxNumberOfElements,
            "Array insert result exceeds the max array size limit {}",
            kMaxNumberOfElements);

        out.reserve(newArrayLength);
        item ? out.push_back(*item) : out.add_null();
        int64_t nullsToFill = newArrayLength - 1 - srcArray->size();
        while (nullsToFill > 0) {
          out.add_null();
          nullsToFill--;
        }
        for (const auto& element : *srcArray) {
          out.push_back(element);
        }
      } else {
        // Insert element into the array based on the negative index *pos, if legacyNegativeIndex
        // is true the index is 0-based otherwise 1-based.
        int64_t posIdx = *pos + srcArray->size() + !*legacyNegativeIndex;
        int64_t newArrayLength =
            std::max((int64_t)(srcArray->size()) + 1, posIdx + 1);
        VELOX_USER_CHECK_LE(
            newArrayLength,
            kMaxNumberOfElements,
            "Array insert result exceeds the max array size limit {}",
            kMaxNumberOfElements);

        out.reserve(newArrayLength);
        int32_t nextIdx = 0;
        for (const auto& element : *srcArray) {
          if (nextIdx == posIdx) {
            item ? out.push_back(*item) : out.add_null();
            nextIdx++;
          }
          out.push_back(element);
          nextIdx++;
        }
        if (nextIdx < newArrayLength) {
          item ? out.push_back(*item) : out.add_null();
        }
      }
    }

    return true;
  }
};
} // namespace facebook::velox::functions::sparksql
