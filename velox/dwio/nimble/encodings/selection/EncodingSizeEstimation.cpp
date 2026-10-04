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
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"

#include <folly/hash/Hash.h>
#include <array>

namespace facebook::nimble::detail {

uint32_t
sampledRowIndex(uint32_t sampleIndex, uint32_t numSamples, uint32_t numRows) {
  NIMBLE_DCHECK_LT(sampleIndex, numSamples);
  NIMBLE_DCHECK_LE(numSamples, numRows);
  const auto begin = uint64_t{sampleIndex} * numRows / numSamples;
  const auto end = uint64_t{sampleIndex + 1} * numRows / numSamples;
  return begin + folly::hash::twang_mix64(sampleIndex + 1) % (end - begin);
}

namespace {

// Samples the derived sequence without allocating a full run-value or uncommon
// value stream. The caller already knows its length from the cached statistics.
template <typename T, typename Predicate>
std::span<const T> sampleFilteredValues(
    std::span<const T> values,
    uint32_t numValues,
    Predicate include,
    std::array<T, ALPRDEncodingBase::kSampleSize>& storage) {
  const auto sampleSize = std::min<uint32_t>(numValues, storage.size());
  if (sampleSize == 0) {
    return {};
  }
  uint32_t ordinal{0};
  uint32_t sampled{0};
  auto nextIndex = sampledRowIndex(0, sampleSize, numValues);
  for (const auto value : values) {
    if (!include(value)) {
      continue;
    }
    if (ordinal++ == nextIndex) {
      storage[sampled++] = value;
      if (sampled == sampleSize) {
        break;
      }
      nextIndex = sampledRowIndex(sampled, sampleSize, numValues);
    }
  }
  return {storage.data(), sampled};
}

// Estimates a floating-point container using its actual value-child policy.
// Returns nullopt for encodings other than Dictionary, RLE and MainlyConstant.
template <typename T>
std::optional<uint64_t> estimateNestedFloatingPointSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> values,
    uint32_t numRows,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    EncodingSelectionPolicyBase& policy,
    const Encoding::Options& options) {
  static_assert(isFloatingPointType<T>());
  using PhysicalType = typename TypeTraits<T>::physicalType;
  if (encodingType != EncodingType::Dictionary &&
      encodingType != EncodingType::RLE &&
      encodingType != EncodingType::MainlyConstant) {
    return std::nullopt;
  }
  const auto scaleCount = [&](uint64_t count) -> uint32_t {
    return (count * numRows + values.size() - 1) / values.size();
  };
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
  std::array<PhysicalType, ALPRDEncodingBase::kSampleSize> storage;
  std::span<const PhysicalType> sample;
  uint32_t numChildRows{0};
  NestedEncodingIdentifier identifier;
  uint64_t otherSize{0};
  switch (encodingType) {
    case EncodingType::Dictionary: {
      const auto& counts = statistics.uniqueCounts().value();
      // A dictionary alphabet is a set, so count each sampled key once rather
      // than retaining the original values' frequency weights.
      uint32_t sampled{0};
      for (const auto& [value, count] : counts) {
        storage[sampled++] = value;
        if (sampled == storage.size()) {
          break;
        }
      }
      sample = {storage.data(), sampled};
      numChildRows = scaleCount(counts.size());
      identifier = EncodingIdentifiers::Dictionary::Alphabet;
      otherSize = prefixSize + sizeof(uint32_t) +
          FixedBitWidthEncoding<uint32_t>::estimateSize(
                      numRows, 0, numChildRows - 1, options);
      break;
    }
    case EncodingType::RLE: {
      const auto numRuns = statistics.consecutiveRepeatCount();
      std::optional<PhysicalType> previous;
      sample = sampleFilteredValues(
          values,
          numRuns,
          [&](PhysicalType value) {
            if (previous == value) {
              return false;
            }
            previous = value;
            return true;
          },
          storage);
      numChildRows = scaleCount(numRuns);
      identifier = EncodingIdentifiers::RunLength::RunValues;
      otherSize = prefixSize + sizeof(uint32_t) +
          FixedBitWidthEncoding<uint32_t>::estimateSize(
                      numChildRows,
                      statistics.minRepeat(),
                      statistics.maxRepeat(),
                      options);
      break;
    }
    case EncodingType::MainlyConstant: {
      const auto common = statistics.uniqueCounts()->mostFrequent().value();
      const auto numUncommon = values.size() - common.second;
      if (numUncommon != 0) {
        sample = sampleFilteredValues(
            values,
            numUncommon,
            [&](PhysicalType value) { return value != common.first; },
            storage);
      }
      numChildRows = scaleCount(numUncommon);
      identifier = EncodingIdentifiers::MainlyConstant::OtherValues;
      otherSize = prefixSize + 2 * sizeof(uint32_t) + sizeof(PhysicalType) +
          SparseBoolEncoding::estimateSize(numRows, numChildRows, options);
      break;
    }
    default:
      NIMBLE_UNREACHABLE("Unexpected floating-point container.");
  }
  auto childPolicy = policy.create<T>(encodingType, identifier);
  return otherSize +
      EncodingSizeEstimation<T>::estimateSelectedSize(
             *childPolicy, sample, numChildRows, options);
}

} // namespace

template <typename T>
std::optional<uint64_t> EncodingSizeEstimation<T>::estimateSize(
    EncodingType encodingType,
    std::span<const physicalType> sampleValues,
    uint32_t numTotalRows,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase& policy,
    bool isSample) {
  // Refine floating-point containers for full input when the policy requests
  // logical selection. Sampled child selection keeps existing container
  // heuristics, bounding repeated training across candidate trees. ALPRD's
  // bounded split training always uses the supplied child policies.
  auto* nestedPolicy = encodingType == EncodingType::ALPRD ||
          (!isSample && policy.useLogicalTypeForNestedEncoding())
      ? &policy
      : nullptr;
  return estimateSize(
      encodingType,
      sampleValues,
      numTotalRows,
      statistics,
      options,
      nestedPolicy);
}

template <typename T>
std::optional<uint64_t> EncodingSizeEstimation<T>::estimateSize(
    EncodingType encodingType,
    std::span<const physicalType> sampleValues,
    uint32_t numTotalRows,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase* policy) {
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALPRD) {
      return ALPRDEncodingBase::estimateSize(
          sampleValues, numTotalRows, options, policy);
    }
    if (policy != nullptr) {
      if (auto size = estimateNestedFloatingPointSize<T>(
              encodingType,
              sampleValues,
              numTotalRows,
              statistics,
              *policy,
              options)) {
        return size;
      }
    }
  }
  if (numTotalRows == sampleValues.size()) {
    return estimateSize(encodingType, sampleValues, statistics, options);
  }
  NIMBLE_CHECK(
      !sampleValues.empty(), "Size estimation requires a non-empty sample.");
  NIMBLE_CHECK_LE(sampleValues.size(), numTotalRows);
  const auto prefixSize =
      EncodingPrefix::serializedSize(numTotalRows, options.useVarintRowCount);
  const auto samplePrefixSize = EncodingPrefix::serializedSize(
      sampleValues.size(), options.useVarintRowCount);
  if (encodingType == EncodingType::Constant) {
    auto size = estimateSize(encodingType, sampleValues, statistics, options);
    return size ? std::optional<uint64_t>{*size - samplePrefixSize + prefixSize}
                : std::nullopt;
  }
  if constexpr (!isStringType<T>()) {
    if (encodingType == EncodingType::Trivial ||
        encodingType == EncodingType::FixedBitWidth ||
        encodingType == EncodingType::SimdForBitpack) {
      return estimateSize(encodingType, numTotalRows, statistics, options);
    }
  }
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALP) {
      return ALPEncoding<T>::estimateSizeFromSample(
          numTotalRows, sampleValues, options);
    }
  }
  auto size = estimateSize(encodingType, sampleValues, statistics, options);
  if (!size) {
    return std::nullopt;
  }
  // Project the sampled bytes after the outer prefix to the full row count:
  //
  //   estimatedSize = fullPrefixSize
  //       + (sampleSizeBytes - samplePrefixSize)
  //           * numTotalRows / numSampleRows
  //
  // Here numSampleRows is sampleValues.size(). Count the outer prefix once;
  // its varint length can depend on the row count.
  //
  // Existing composite estimates are heuristics. Scaling their inner metadata
  // along with the payload is conservative; it avoids assuming a different
  // child codec merely because the sample is small.
  //
  // Varint's existing estimator uses a fixed prefix. Keep that convention for
  // policy scoring, then correct it for the selected child's serialized size.
  const auto estimatedPrefixSize = encodingType == EncodingType::Varint
      ? EncodingPrefix::kFixedPrefixSize
      : prefixSize;
  const auto estimatedSamplePrefixSize = encodingType == EncodingType::Varint
      ? EncodingPrefix::kFixedPrefixSize
      : samplePrefixSize;
  return estimatedPrefixSize +
      (*size - std::min<uint64_t>(*size, estimatedSamplePrefixSize)) *
      numTotalRows / sampleValues.size();
}

template <typename T>
uint64_t EncodingSizeEstimation<T>::estimateSelectedSize(
    EncodingSelectionPolicyBase& policy,
    std::span<const physicalType> sampleValues,
    uint32_t numTotalRows,
    const Encoding::Options& options) {
  const auto statistics = Statistics<physicalType>::create(sampleValues);
  auto result = static_cast<EncodingSelectionPolicy<T>&>(policy).select(
      sampleValues, numTotalRows, statistics, options);
  auto size = result.estimatedSize;
  if (!size) {
    size = estimateSize(
        result.encodingType,
        sampleValues,
        numTotalRows,
        statistics,
        options,
        &policy);
  }
  const auto prefixSize =
      EncodingPrefix::serializedSize(numTotalRows, options.useVarintRowCount);
  if (!size) {
    // Custom policies can select codecs without estimators. Keep their layout
    // binding and use an uncompressed size as the training approximation.
    return prefixSize + 1 + uint64_t{numTotalRows} * sizeof(physicalType);
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

#define INSTANTIATE_SIZE_ESTIMATION(T)                                      \
  template std::optional<uint64_t> EncodingSizeEstimation<T>::estimateSize( \
      EncodingType,                                                         \
      std::span<const TypeTraits<T>::physicalType>,                         \
      uint32_t,                                                             \
      const Statistics<TypeTraits<T>::physicalType>&,                       \
      const Encoding::Options&,                                             \
      EncodingSelectionPolicyBase&,                                         \
      bool);                                                                \
  template uint64_t EncodingSizeEstimation<T>::estimateSelectedSize(        \
      EncodingSelectionPolicyBase&,                                         \
      std::span<const TypeTraits<T>::physicalType>,                         \
      uint32_t,                                                             \
      const Encoding::Options&)

INSTANTIATE_SIZE_ESTIMATION(int8_t);
INSTANTIATE_SIZE_ESTIMATION(uint8_t);
INSTANTIATE_SIZE_ESTIMATION(int16_t);
INSTANTIATE_SIZE_ESTIMATION(uint16_t);
INSTANTIATE_SIZE_ESTIMATION(int32_t);
INSTANTIATE_SIZE_ESTIMATION(uint32_t);
INSTANTIATE_SIZE_ESTIMATION(int64_t);
INSTANTIATE_SIZE_ESTIMATION(uint64_t);
INSTANTIATE_SIZE_ESTIMATION(float);
INSTANTIATE_SIZE_ESTIMATION(double);
INSTANTIATE_SIZE_ESTIMATION(bool);
INSTANTIATE_SIZE_ESTIMATION(std::string_view);

#undef INSTANTIATE_SIZE_ESTIMATION

} // namespace facebook::nimble::detail
