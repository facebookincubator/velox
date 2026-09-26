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

#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"

namespace facebook::nimble::detail {

/// Chooses a deterministic point within an evenly sized sampling interval.
/// Varies offsets to avoid repeatedly observing one phase of periodic input.
uint32_t
sampledRowIndex(uint32_t sampleIndex, uint32_t numSamples, uint32_t numRows);

/// Estimates a floating-point container using its actual value-child policy.
/// Returns nullopt for encodings other than Dictionary, RLE and MainlyConstant.
template <typename T>
std::optional<uint64_t> estimateNestedFloatingPointSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> values,
    uint32_t numRows,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    EncodingSelectionPolicyBase& policy,
    const Encoding::Options& options);

/// Projects selection costs from a representative sample. Scalar range and
/// constant estimates use the full row count; other existing estimates scale
/// their sampled payload. A policy enables ALPRD's child-aware model and the
/// corresponding floating-point container estimates. No candidate is encoded.
template <typename T>
std::optional<uint64_t> estimateSampledEncodingSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> values,
    uint32_t numRows,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALPRD) {
      return ALPRDEncodingBase::estimateSize(values, numRows, options, policy);
    }
    if (policy != nullptr) {
      if (auto size = estimateNestedFloatingPointSize<T>(
              encodingType, values, numRows, statistics, *policy, options)) {
        return size;
      }
    }
  }
  if (numRows == values.size()) {
    return EncodingSizeEstimation<T>::estimateSize(
        encodingType, values, statistics, options);
  }
  NIMBLE_CHECK(!values.empty(), "Size estimation requires a non-empty sample.");
  NIMBLE_CHECK_LE(values.size(), numRows);
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
  const auto samplePrefixSize =
      EncodingPrefix::serializedSize(values.size(), options.useVarintRowCount);
  if (encodingType == EncodingType::Constant) {
    auto size = EncodingSizeEstimation<T>::estimateSize(
        encodingType, values, statistics, options);
    return size ? std::optional<uint64_t>{*size - samplePrefixSize + prefixSize}
                : std::nullopt;
  }
  if constexpr (!isStringType<T>()) {
    if (encodingType == EncodingType::Trivial ||
        encodingType == EncodingType::FixedBitWidth ||
        encodingType == EncodingType::SimdForBitpack) {
      return EncodingSizeEstimation<T>::estimateSize(
          encodingType, numRows, statistics, options);
    }
  }
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALP) {
      return ALPEncoding<T>::estimateSizeFromSample(numRows, values, options);
    }
  }
  auto size = EncodingSizeEstimation<T>::estimateSize(
      encodingType, values, statistics, options);
  if (!size) {
    return std::nullopt;
  }
  // Existing composite estimates are heuristics. Scaling their inner metadata
  // along with the payload is conservative; it avoids assuming a different
  // child codec merely because the sample is small.
  // Varint's existing estimator uses a fixed prefix. Keep that convention for
  // policy scoring, then correct it for the selected child's serialized size.
  const auto estimatedPrefixSize = encodingType == EncodingType::Varint
      ? EncodingPrefix::kFixedPrefixSize
      : prefixSize;
  const auto estimatedSamplePrefixSize = encodingType == EncodingType::Varint
      ? EncodingPrefix::kFixedPrefixSize
      : samplePrefixSize;
  return estimatedPrefixSize +
      (*size - std::min<uint64_t>(*size, estimatedSamplePrefixSize)) * numRows /
      values.size();
}

/// Estimates the bytes written by the child chosen by policy, before generic
/// compression. Corrects FBW's padding and scalar prefix sizes without changing
/// the policy's established scoring of existing codecs.
template <typename T>
uint64_t estimateSelectedChildSize(
    EncodingSelectionPolicyBase& policy,
    std::span<const typename TypeTraits<T>::physicalType> values,
    uint32_t numRows,
    const Encoding::Options& options) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  const auto statistics = Statistics<PhysicalType>::create(values);
  auto result =
      static_cast<EncodingSelectionPolicy<T>&>(policy).selectFromSample(
          values, numRows, statistics, options);
  auto size = result.estimatedSize;
  if (!size) {
    size = estimateSampledEncodingSize<T>(
        result.encodingType, values, numRows, statistics, options, &policy);
  }
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
  if (!size) {
    // Custom policies can select codecs without estimators. Keep their layout
    // binding and use an uncompressed size as the training approximation.
    return prefixSize + 1 + uint64_t{numRows} * sizeof(PhysicalType);
  }
  if (result.encodingType == EncodingType::Trivial ||
      result.encodingType == EncodingType::FixedBitWidth ||
      result.encodingType == EncodingType::Varint) {
    *size = *size - EncodingPrefix::kFixedPrefixSize + prefixSize;
    if (result.encodingType == EncodingType::FixedBitWidth) {
      *size += FixedBitArray::bufferSize(0, 0);
    }
  }
  return *size;
}

} // namespace facebook::nimble::detail
