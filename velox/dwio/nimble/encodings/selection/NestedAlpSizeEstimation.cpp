/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

#include <folly/hash/Hash.h>

#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"

namespace facebook::nimble::detail {

uint32_t NestedAlpSizeEstimation::sampledRowIndex(
    uint32_t sampleIndex,
    uint32_t numSamples,
    uint32_t numRows) {
  NIMBLE_DCHECK_LT(sampleIndex, numSamples);
  NIMBLE_DCHECK_LE(numSamples, numRows);
  const auto begin = uint64_t{sampleIndex} * numRows / numSamples;
  const auto end = uint64_t{sampleIndex + 1} * numRows / numSamples;
  return begin + folly::hash::twang_mix64(sampleIndex + 1) % (end - begin);
}

template <typename T>
uint64_t NestedAlpSizeEstimation::estimateChildSize(
    uint32_t numRows,
    uint64_t minValue,
    uint64_t maxValue,
    const Encoding::Options& options) {
  NIMBLE_DCHECK_GT(numRows, 0);
  NIMBLE_DCHECK_LE(minValue, maxValue);
  if (minValue == maxValue) {
    return EncodingPrefix::serializedSize(numRows, options.useVarintRowCount) +
        sizeof(T);
  }
  return std::min(
      FixedBitWidthEncoding<T>::estimateSize(
          numRows, minValue, maxValue, options),
      TrivialEncoding<T>::estimateSize(numRows));
}

template uint64_t NestedAlpSizeEstimation::estimateChildSize<uint16_t>(
    uint32_t,
    uint64_t,
    uint64_t,
    const Encoding::Options&);
template uint64_t NestedAlpSizeEstimation::estimateChildSize<uint32_t>(
    uint32_t,
    uint64_t,
    uint64_t,
    const Encoding::Options&);
template uint64_t NestedAlpSizeEstimation::estimateChildSize<uint64_t>(
    uint32_t,
    uint64_t,
    uint64_t,
    const Encoding::Options&);

} // namespace facebook::nimble::detail
