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
#pragma once

#include <optional>
#include <span>
#include <vector>
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble::detail {

/// Estimates the nested ALP size from all physical values when `T` is a
/// floating-point type and nested ALP selection is enabled. Returns
/// `std::nullopt` when nested ALP is not eligible.
template <typename T>
std::optional<uint64_t> nestedAlpSize(
    std::span<const typename TypeTraits<T>::physicalType> values,
    const Encoding::Options& options) {
  static_assert(
      isFloatingPointType<T>(),
      "nestedAlpSize only supports floating-point logical types.");
  if (!options.allowNestedAlpSelection) {
    return std::nullopt;
  }
  return ALPEncoding<T>::estimateSize(values, options);
}

/// Estimates the nested ALP size by sampling the distinct physical values in
/// `uniqueCounts`. Returns `std::nullopt` when nested ALP is not eligible or
/// there are no values to sample.
template <typename T, typename UniqueCounts>
std::optional<uint64_t> uniqueValuesNestedAlpSize(
    const UniqueCounts& uniqueCounts,
    const Encoding::Options& options) {
  static_assert(
      isFloatingPointType<T>(),
      "uniqueValuesNestedAlpSize only supports floating-point logical types.");
  if (!options.allowNestedAlpSelection) {
    return std::nullopt;
  }
  if (uniqueCounts.size() == 0) {
    return std::nullopt;
  }

  using physicalType = typename TypeTraits<T>::physicalType;
  const uint64_t uniqueCount = uniqueCounts.size();
  const uint32_t sampleSize = ALPEncoding<T>::estimateSampleSize(uniqueCount);
  std::vector<physicalType> sampledValues;
  sampledValues.reserve(sampleSize);
  for (const auto& [value, count] : uniqueCounts) {
    (void)count;
    sampledValues.push_back(value);
    if (sampledValues.size() == sampleSize) {
      break;
    }
  }
  return ALPEncoding<T>::estimateSizeFromSample(
      uniqueCount, sampledValues, options);
}

} // namespace facebook::nimble::detail
