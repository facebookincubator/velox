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

template <typename A, typename B, typename Compare>
void applySimdComparison(
    const SelectivityVector& rows,
    const A* __restrict rawA,
    const B* __restrict rawB,
    Compare cmp,
    VectorPtr& result,
    std::vector<int8_t>& tempBuffer) {
  vector_size_t begin = rows.begin();
  vector_size_t end = rows.end();
  tempBuffer.reserve(end);
  auto* __restrict tempRes = tempBuffer.data();
  rows.applyToSelected(
      [&](auto i) { tempRes[i] = cmp(rawA, rawB, i) ? -1 : 0; });

  auto* rowsData = reinterpret_cast<const int8_t*>(rows.allBits());
  auto* resultVector = result->asUnchecked<FlatVector<bool>>();
  auto* rawResult = resultVector->mutableRawValues<uint8_t>();
  auto vectorBegin = begin;
  if (vectorBegin % 32 != 0) {
    auto modCnt = 32 - begin % 32;
    vectorBegin = std::min(begin + modCnt, end);
    for (auto i = begin; i < vectorBegin; i++) {
      if (rows.isValid(i)) {
        bits::setBit(rawResult, i, tempRes[i]);
      }
    }
  }
  const auto vectorEnd = std::max(end - end % 32, vectorBegin);
  for (auto i = vectorBegin; i < vectorEnd; i += 32) {
    auto res = simd::toBitMask(
        xsimd::batch_bool<int8_t>(xsimd::load_unaligned(tempRes + i)));
    uint32_t mask = *(reinterpret_cast<const uint32_t*>(rowsData + i / 8));
    uint32_t* addr = reinterpret_cast<uint32_t*>(rawResult + i / 8);
    *addr = (*addr & ~mask) | (res & mask);
  }
  for (auto i = vectorEnd; i < end; i++) {
    if (rows.isValid(i)) {
      bits::setBit(rawResult, i, tempRes[i]);
    }
  }
}
} // namespace facebook::velox::functions::sparksql
