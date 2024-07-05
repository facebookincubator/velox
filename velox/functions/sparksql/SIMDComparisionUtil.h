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

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

template <bool allSelected>
inline void setBoolTypeResultVectorByWord(
    const int8_t* rows,
    uint8_t* result,
    int8_t* tempRes,
    vector_size_t index) {
  if constexpr (allSelected) {
    *(reinterpret_cast<uint32_t*>(result + index / 8)) = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes)));
    *(reinterpret_cast<uint32_t*>(result + index / 8 + 4)) = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes + 32)));
  } else {
    uint32_t mask1 = *(reinterpret_cast<const uint32_t*>(rows + index / 8));
    uint32_t* addr1 = reinterpret_cast<uint32_t*>(result + index / 8);
    uint32_t res1 = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes)));
    // Set results only for selected rows.
    *addr1 = (*addr1 & ~mask1) | (res1 & mask1);
    uint32_t mask2 = *(reinterpret_cast<const uint32_t*>(rows + index / 8 + 4));
    uint32_t* addr2 = reinterpret_cast<uint32_t*>(result + index / 8 + 4);
    uint32_t res2 = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes + 32)));
    *addr2 = (*addr2 & ~mask2) | (res2 & mask2);
  }
}

template <typename A, typename B, typename Compare>
void applySimdComparison(
    const SelectivityVector& rows,
    const A* __restrict rawA,
    const B* __restrict rawB,
    Compare cmp,
    VectorPtr& result) {
  vector_size_t begin = rows.begin();
  vector_size_t end = rows.end();
  int8_t tempBuffer[64];
  auto* __restrict tempRes = tempBuffer;
  auto* rowsData = reinterpret_cast<const int8_t*>(rows.allBits());
  auto* resultVector = result->asUnchecked<FlatVector<bool>>();
  auto* rawResult = resultVector->mutableRawValues<uint8_t>();
  if (rows.isAllSelected()) {
    auto i = 0;
    for (; i + 64 <= end; i += 64) {
      // Do 64 comparisons in a batch, set results by SIMD.
      for (auto j = 0; j < 64; ++j) {
        tempRes[j] = cmp(rawA, rawB, i + j) ? -1 : 0;
      }
      setBoolTypeResultVectorByWord<true>(rowsData, rawResult, tempRes, i);
    }
    for (; i < end; ++i) {
      bits::setBit(rawResult, i, cmp(rawA, rawB, i));
    }
  } else {
    static constexpr uint64_t kAllSet = -1ULL;
    bits::forEachWord(
        begin,
        end,
        [&](int32_t idx, uint64_t mask) {
          auto word = rowsData[idx] & mask;
          if (!word) {
            return;
          }
          while (word) {
            auto index = idx * 64 + __builtin_ctzll(word);
            bits::setBit(rawResult, index, cmp(rawA, rawB, index));
            word &= word - 1;
          }
        },
        [&](int32_t idx) {
          auto word = rowsData[idx];
          if (kAllSet == word) {
            const size_t start = idx * 64;
            const size_t end = (idx + 1) * 64;
            // Do 64 comparisons in a batch, set results by SIMD.
            for (size_t row = start; row < end; ++row) {
              tempRes[row - start] = cmp(rawA, rawB, row) ? -1 : 0;
            }
            setBoolTypeResultVectorByWord<true>(
                rowsData, rawResult, tempRes, start);
          } else {
            // Do 64 comparisons in a batch, set results by SIMD.
            while (word) {
              auto index = __builtin_ctzll(word);
              tempRes[index] = cmp(rawA, rawB, idx * 64 + index) ? -1 : 0;
              word &= word - 1;
            }
            setBoolTypeResultVectorByWord<false>(
                rowsData, rawResult, tempRes, idx * 64);
          }
        });
  }
}
} // namespace facebook::velox::functions::sparksql
