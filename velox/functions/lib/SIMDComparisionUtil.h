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

namespace facebook::velox::functions {

namespace {

template <typename T, bool isConstant, typename Arch = xsimd::default_arch>
inline auto loadSimdData(const T* rawData, vector_size_t offset) {
  using d_type = xsimd::batch<T>;
  if constexpr (isConstant) {
    return xsimd::broadcast<T>(rawData[0]);
  }
  return d_type::load_unaligned(rawData + offset);
}

template <bool allSelected>
inline void setBoolTypeResultVectorByWord(
    const uint64_t* rows,
    uint8_t* result,
    int8_t* tempRes,
    vector_size_t index) {
  if constexpr (allSelected) {
    *(reinterpret_cast<uint32_t*>(result + index / 8)) = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes)));
    *(reinterpret_cast<uint32_t*>(result + index / 8 + 4)) = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes + 32)));
  } else {
    uint32_t mask1 = *(reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const int8_t*>(rows) + index / 8));
    uint32_t* addr1 = reinterpret_cast<uint32_t*>(result + index / 8);
    uint32_t res1 = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes)));
    // Set results only for selected rows.
    *addr1 = (*addr1 & ~mask1) | (res1 & mask1);
    uint32_t mask2 = *(reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const int8_t*>(rows) + index / 8 + 4));
    uint32_t* addr2 = reinterpret_cast<uint32_t*>(result + index / 8 + 4);
    uint32_t res2 = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes + 32)));
    *addr2 = (*addr2 & ~mask2) | (res2 & mask2);
  }
}

} // namespace

template <
    typename T,
    bool isLeftConstant,
    bool isRightConstant,
    typename ComparisonOp,
    typename Arch = xsimd::default_arch>
void applySimdComparison(
    const vector_size_t begin,
    const vector_size_t end,
    const T* rawLhs,
    const T* rawRhs,
    uint8_t* rawResult) {
  using d_type = xsimd::batch<T>;
  constexpr auto numScalarElements = d_type::size;
  const auto vectorEnd = (end - begin) - (end - begin) % numScalarElements;

  if constexpr (numScalarElements == 2 || numScalarElements == 4) {
    for (auto i = begin; i < vectorEnd; i += 8) {
      rawResult[i / 8] = 0;
      for (auto j = 0; j < 8 && (i + j) < vectorEnd; j += numScalarElements) {
        auto left = loadSimdData<T, isLeftConstant>(rawLhs, i + j);
        auto right = loadSimdData<T, isRightConstant>(rawRhs, i + j);

        uint8_t res = simd::toBitMask(ComparisonOp()(left, right));
        rawResult[i / 8] |= res << j;
      }
    }
  } else {
    for (auto i = begin; i < vectorEnd; i += numScalarElements) {
      auto left = loadSimdData<T, isLeftConstant>(rawLhs, i);
      auto right = loadSimdData<T, isRightConstant>(rawRhs, i);

      auto res = simd::toBitMask(ComparisonOp()(left, right));
      if constexpr (numScalarElements == 8) {
        rawResult[i / 8] = res;
      } else if constexpr (numScalarElements == 16) {
        uint16_t* addr = reinterpret_cast<uint16_t*>(rawResult + i / 8);
        *addr = res;
      } else if constexpr (numScalarElements == 32) {
        uint32_t* addr = reinterpret_cast<uint32_t*>(rawResult + i / 8);
        *addr = res;
      } else {
        VELOX_FAIL("Unsupported number of scalar elements");
      }
    }
  }

  // Evaluate remaining values.
  for (auto i = vectorEnd; i < end; i++) {
    if constexpr (isRightConstant && isLeftConstant) {
      bits::setBit(rawResult, i, ComparisonOp()(rawLhs[0], rawRhs[0]));
    } else if constexpr (isRightConstant) {
      bits::setBit(rawResult, i, ComparisonOp()(rawLhs[i], rawRhs[0]));
    } else if constexpr (isLeftConstant) {
      bits::setBit(rawResult, i, ComparisonOp()(rawLhs[0], rawRhs[i]));
    } else {
      bits::setBit(rawResult, i, ComparisonOp()(rawLhs[i], rawRhs[i]));
    }
  }
}

template <
    typename T,
    typename ComparisonOp,
    typename Arch = xsimd::default_arch>
void applySimdComparison(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    VectorPtr& result) {
  auto resultVector = result->asUnchecked<FlatVector<bool>>();
  auto rawResult = resultVector->mutableRawValues<uint8_t>();
  if (args[0]->isConstantEncoding() && args[1]->isConstantEncoding()) {
    auto l = args[0]->asUnchecked<ConstantVector<T>>()->valueAt(0);
    auto r = args[1]->asUnchecked<ConstantVector<T>>()->valueAt(0);
    applySimdComparison<T, true, true, ComparisonOp>(
        rows.begin(), rows.end(), &l, &r, rawResult);
  } else if (args[0]->isConstantEncoding()) {
    auto l = args[0]->asUnchecked<ConstantVector<T>>()->valueAt(0);
    auto rawRhs = args[1]->asUnchecked<FlatVector<T>>()->rawValues();
    applySimdComparison<T, true, false, ComparisonOp>(
        rows.begin(), rows.end(), &l, rawRhs, rawResult);
  } else if (args[1]->isConstantEncoding()) {
    auto rawLhs = args[0]->asUnchecked<FlatVector<T>>()->rawValues();
    auto r = args[1]->asUnchecked<ConstantVector<T>>()->valueAt(0);
    applySimdComparison<T, false, true, ComparisonOp>(
        rows.begin(), rows.end(), rawLhs, &r, rawResult);
  } else {
    auto rawLhs = args[0]->asUnchecked<FlatVector<T>>()->rawValues();
    auto rawRhs = args[1]->asUnchecked<FlatVector<T>>()->rawValues();
    applySimdComparison<T, false, false, ComparisonOp>(
        rows.begin(), rows.end(), rawLhs, rawRhs, rawResult);
  }
}

template <typename A, typename B, typename Compare>
void applyAutoSimdComparison(
    const SelectivityVector& rows,
    const A* __restrict rawA,
    const B* __restrict rawB,
    Compare cmp,
    VectorPtr& result) {
  vector_size_t begin = rows.begin();
  vector_size_t end = rows.end();
  int8_t tempBuffer[64];
  auto* __restrict tempRes = tempBuffer;
  auto* rowsData = reinterpret_cast<const uint64_t*>(rows.allBits());
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
          const size_t start = idx * 64;
          while (word) {
            auto index = start + __builtin_ctzll(word);
            bits::setBit(rawResult, index, cmp(rawA, rawB, index));
            word &= word - 1;
          }
        },
        [&](int32_t idx) {
          auto word = rowsData[idx];
          const size_t start = idx * 64;
          if (kAllSet == word) {
            const size_t end = start + 64;
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
              tempRes[index] = cmp(rawA, rawB, start + index) ? -1 : 0;
              word &= word - 1;
            }
            setBoolTypeResultVectorByWord<false>(
                rowsData, rawResult, tempRes, start);
          }
        });
  }
}
} // namespace facebook::velox::functions
