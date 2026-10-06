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

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"

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

uint64_t NestedAlpSizeEstimation::serializedSize(
    EncodingType encodingType,
    uint64_t estimatedSize,
    uint32_t numRows,
    const Encoding::Options& options) {
  if (encodingType == EncodingType::Trivial ||
      encodingType == EncodingType::FixedBitWidth ||
      encodingType == EncodingType::Varint ||
      encodingType == EncodingType::SimdForBitpack) {
    // Preserve the built-in selection scores used by child writers. ALP and
    // ALPRD count the selected child's actual prefix and padding in their size.
    NIMBLE_DCHECK_GE(estimatedSize, EncodingPrefix::kFixedPrefixSize);
    estimatedSize = estimatedSize - EncodingPrefix::kFixedPrefixSize +
        EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
    if (encodingType == EncodingType::FixedBitWidth) {
      estimatedSize += FixedBitArray::bufferSize(0, 0);
    }
  }
  return estimatedSize;
}

namespace {

// Samples the derived sequence without allocating a full run-value or uncommon
// value stream. The caller already knows its length from the cached statistics.
template <typename T, typename Predicate>
std::span<const T> sampleFilteredValues(
    std::span<const T> values,
    uint32_t numValues,
    Predicate include,
    std::array<T, NestedAlpSizeEstimation::kSampleSize>& storage) {
  const auto sampleSize = std::min<uint32_t>(numValues, storage.size());
  if (sampleSize == 0) {
    return {};
  }
  uint32_t ordinal{0};
  uint32_t sampled{0};
  auto nextIndex =
      NestedAlpSizeEstimation::sampledRowIndex(0, sampleSize, numValues);
  for (const auto value : values) {
    if (!include(value)) {
      continue;
    }
    if (ordinal++ == nextIndex) {
      storage[sampled++] = value;
      if (sampled == sampleSize) {
        break;
      }
      nextIndex = NestedAlpSizeEstimation::sampledRowIndex(
          sampled, sampleSize, numValues);
    }
  }
  return {storage.data(), sampled};
}

// Estimates a floating-point container using its actual value-child policy.
// Returns nullopt for encodings other than Dictionary, RLE and MainlyConstant.
// Uses sampleValues to estimate the target stream of numRows values.
template <typename T>
std::optional<uint64_t> estimateNestedFloatingPointSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> sampleValues,
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
  // Project an observed child count to numRows, rounding up to a whole row.
  const auto scaleCount = [&](uint64_t count) -> uint32_t {
    return (count * numRows + sampleValues.size() - 1) / sampleValues.size();
  };
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows, options.useVarintRowCount);
  std::array<PhysicalType, NestedAlpSizeEstimation::kSampleSize> storage;
  std::span<const PhysicalType> sample;
  uint32_t numChildRows{0};
  NestedEncodingIdentifier nestedIdentifier;
  uint64_t otherSize{0};
  switch (encodingType) {
    case EncodingType::Dictionary: {
      const auto& counts = statistics.uniqueCounts().value();
      // A dictionary alphabet is a set, so count each sampled key once rather
      // than retaining the observed frequency weights.
      uint32_t sampled{0};
      for (const auto& [value, count] : counts) {
        storage[sampled++] = value;
        if (sampled == storage.size()) {
          break;
        }
      }
      sample = {storage.data(), sampled};
      numChildRows = scaleCount(counts.size());
      nestedIdentifier = EncodingIdentifiers::Dictionary::Alphabet;
      otherSize = prefixSize + sizeof(uint32_t) +
          FixedBitWidthEncoding<uint32_t>::estimateSize(
                      numRows, 0, numChildRows - 1, options);
      break;
    }
    case EncodingType::RLE: {
      const auto numRuns = statistics.consecutiveRepeatCount();
      std::optional<PhysicalType> previous;
      sample = sampleFilteredValues(
          sampleValues,
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
      nestedIdentifier = EncodingIdentifiers::RunLength::RunValues;
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
      const auto numUncommon = sampleValues.size() - common.second;
      if (numUncommon != 0) {
        sample = sampleFilteredValues(
            sampleValues,
            numUncommon,
            [&](PhysicalType value) { return value != common.first; },
            storage);
      }
      numChildRows = scaleCount(numUncommon);
      nestedIdentifier = EncodingIdentifiers::MainlyConstant::OtherValues;
      otherSize = prefixSize + 2 * sizeof(uint32_t) + sizeof(PhysicalType) +
          SparseBoolEncoding::estimateSize(numRows, numChildRows, options);
      break;
    }
    default:
      NIMBLE_UNREACHABLE(
          "Unexpected floating-point container: {}.", encodingType);
  }
  auto childPolicy = policy.create<T>(encodingType, nestedIdentifier);
  return otherSize +
      NestedAlpSizeEstimation::estimateChildSize<T>(
             sample, numChildRows, options, *childPolicy);
}

// Estimates ALP and ALPRD children at the size of their target streams.
// Sampling state and policy adaptation stay local to these encoding models.
template <typename T>
class SampledCost {
 public:
  using PhysicalType = typename TypeTraits<T>::physicalType;

  // Statistics describe the observed sample; numRows is the target size.
  SampledCost(
      EncodingSelectionPolicyBase& policy,
      std::span<const PhysicalType> sampleValues,
      uint32_t numRows,
      const Encoding::Options& options);

  // Scores manual candidates at full size or honors a policy-selected layout.
  uint64_t selectedSize();

 private:
  // Uses the writer's selection-size convention for candidate comparison.
  // Container heuristics bound recursive training across candidates.
  std::optional<uint64_t> estimateSelectionSize(EncodingType encodingType);

  // Honors child policies when estimating a policy-selected container.
  std::optional<uint64_t> estimateBoundSize(EncodingType encodingType);

  // Observed values and the full stream length they represent.
  const std::span<const PhysicalType> sampleValues_;
  const uint32_t numRows_;
  // Builds statistics once for all candidate estimates.
  const Statistics<PhysicalType> statistics_;
  // Prefix and bit-packing options must match the subsequent writer.
  const Encoding::Options& options_;
  // Borrows the child policy that carries candidate and layout restrictions.
  EncodingSelectionPolicy<T>& policy_;
};

template <typename T>
SampledCost<T>::SampledCost(
    EncodingSelectionPolicyBase& policy,
    std::span<const PhysicalType> sampleValues,
    uint32_t numRows,
    const Encoding::Options& options)
    : sampleValues_{sampleValues},
      numRows_{numRows},
      statistics_{Statistics<PhysicalType>::create(sampleValues)},
      options_{options},
      policy_{*velox::checkedPointerCast<EncodingSelectionPolicy<T>>(&policy)} {
}

template <typename T>
uint64_t SampledCost<T>::selectedSize() {
  EncodingType selectedEncoding{EncodingType::Trivial};
  if (auto* manual =
          dynamic_cast<ManualEncodingSelectionPolicy<T>*>(&policy_)) {
    // The policy compares candidates using their full-stream costs. Sampled
    // candidates retain the existing container heuristics to bound recursive
    // training, even when the sample contains every input row.
    NIMBLE_CHECK_LE(sampleValues_.size(), numRows_);
    const auto result = manual->select(sampleValues_, [&](EncodingType type) {
      return estimateSelectionSize(type);
    });
    selectedEncoding = result.encodingType;
    if (result.estimatedSize) {
      return NestedAlpSizeEstimation::serializedSize(
          selectedEncoding, *result.estimatedSize, numRows_, options_);
    }
  } else {
    const auto result = policy_.select(sampleValues_, statistics_, options_);
    selectedEncoding = result.encodingType;
    // A policy's estimate describes its input, so reuse it only when the
    // sample covers the full stream. The selected layout remains binding.
    if (numRows_ == sampleValues_.size() && result.estimatedSize) {
      return *result.estimatedSize;
    }
  }
  const auto size = estimateBoundSize(selectedEncoding);
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows_, options_.useVarintRowCount);
  if (!size) {
    // Custom policies can select codecs without estimators. Keep their layout
    // binding and use an uncompressed size as the training approximation.
    return prefixSize + 1 + uint64_t{numRows_} * sizeof(PhysicalType);
  }
  return NestedAlpSizeEstimation::serializedSize(
      selectedEncoding, *size, numRows_, options_);
}

template <typename T>
std::optional<uint64_t> SampledCost<T>::estimateSelectionSize(
    EncodingType encodingType) {
  if constexpr (isFloatingPointType<T>()) {
    if (encodingType == EncodingType::ALPRD) {
      return ALPRDEncodingBase::estimateSize(
          sampleValues_, numRows_, options_, &policy_);
    }
    if (encodingType == EncodingType::ALP) {
      if (sampleValues_.empty()) {
        return std::nullopt;
      }
      return ALPEncoding<T>::estimateSizeFromSample(
          numRows_, sampleValues_, options_, &policy_);
    }
  }
  if (numRows_ == sampleValues_.size()) {
    return detail::EncodingSizeEstimation<T>::estimateSize(
        encodingType, sampleValues_, statistics_, options_);
  }
  NIMBLE_CHECK(
      !sampleValues_.empty(), "Size estimation requires a non-empty sample.");
  NIMBLE_CHECK_LE(sampleValues_.size(), numRows_);
  const auto prefixSize =
      EncodingPrefix::serializedSize(numRows_, options_.useVarintRowCount);
  const auto samplePrefixSize = EncodingPrefix::serializedSize(
      sampleValues_.size(), options_.useVarintRowCount);
  if (encodingType == EncodingType::Constant) {
    auto size = detail::EncodingSizeEstimation<T>::estimateSize(
        encodingType, sampleValues_, statistics_, options_);
    return size ? std::optional<uint64_t>{*size - samplePrefixSize + prefixSize}
                : std::nullopt;
  }
  // Use the framework's row-count estimator when the observed statistics
  // suffice. Other codecs need the sample values before extrapolation.
  if constexpr (!isStringType<T>()) {
    if (encodingType == EncodingType::Trivial ||
        encodingType == EncodingType::FixedBitWidth ||
        encodingType == EncodingType::SimdForBitpack) {
      return detail::EncodingSizeEstimation<T>::estimateSize(
          encodingType, numRows_, statistics_, options_);
    }
  }
  auto size = detail::EncodingSizeEstimation<T>::estimateSize(
      encodingType, sampleValues_, statistics_, options_);
  if (!size) {
    return std::nullopt;
  }
  if (encodingType == EncodingType::Varint) {
    // Varint stores one baseline per stream. Project only the histogram's
    // payload bytes, preserving the fixed-prefix convention for selection.
    constexpr auto kHeaderSize =
        EncodingPrefix::kFixedPrefixSize + sizeof(PhysicalType);
    NIMBLE_DCHECK_GE(*size, kHeaderSize);
    return kHeaderSize +
        (*size - kHeaderSize) * numRows_ / sampleValues_.size();
  }
  // Project the sampled bytes after the outer prefix to the full row count:
  //
  //   estimatedSize = fullPrefixSize
  //       + (sampleSizeBytes - samplePrefixSize)
  //           * numRows / numSampleRows
  //
  // Here numSampleRows is sampleValues.size(). Count the outer prefix once;
  // its varint length can depend on the row count.
  //
  // Existing composite estimates are heuristics. Scaling their inner metadata
  // along with the payload is conservative; it avoids assuming a different
  // child codec merely because the sample is small.
  return prefixSize +
      (*size - std::min<uint64_t>(*size, samplePrefixSize)) * numRows_ /
      sampleValues_.size();
}

template <typename T>
std::optional<uint64_t> SampledCost<T>::estimateBoundSize(
    EncodingType encodingType) {
  if constexpr (isFloatingPointType<T>()) {
    if (auto size = estimateNestedFloatingPointSize<T>(
            encodingType,
            sampleValues_,
            numRows_,
            statistics_,
            policy_,
            options_)) {
      return size;
    }
  }
  return estimateSelectionSize(encodingType);
}

} // namespace

template <typename T>
uint64_t NestedAlpSizeEstimation::estimateChildSize(
    std::span<const typename TypeTraits<T>::physicalType> sampleValues,
    uint32_t numRows,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase& policy) {
  return SampledCost<T>{policy, sampleValues, numRows, options}.selectedSize();
}

template <typename T>
std::optional<uint64_t> NestedAlpSizeEstimation::estimateSize(
    EncodingType encodingType,
    std::span<const typename TypeTraits<T>::physicalType> values,
    const Statistics<typename TypeTraits<T>::physicalType>& statistics,
    const Encoding::Options& options,
    EncodingSelectionPolicyBase& policy) {
  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  return estimateNestedFloatingPointSize<T>(
      encodingType, values, values.size(), statistics, policy, options);
}

template uint64_t NestedAlpSizeEstimation::estimateChildSize<uint16_t>(
    std::span<const uint16_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);
template uint64_t NestedAlpSizeEstimation::estimateChildSize<uint32_t>(
    std::span<const uint32_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);
template uint64_t NestedAlpSizeEstimation::estimateChildSize<uint64_t>(
    std::span<const uint64_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);
template uint64_t NestedAlpSizeEstimation::estimateChildSize<float>(
    std::span<const uint32_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);
template uint64_t NestedAlpSizeEstimation::estimateChildSize<double>(
    std::span<const uint64_t>,
    uint32_t,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);

template std::optional<uint64_t> NestedAlpSizeEstimation::estimateSize<float>(
    EncodingType,
    std::span<const uint32_t>,
    const Statistics<uint32_t>&,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);
template std::optional<uint64_t> NestedAlpSizeEstimation::estimateSize<double>(
    EncodingType,
    std::span<const uint64_t>,
    const Statistics<uint64_t>&,
    const Encoding::Options&,
    EncodingSelectionPolicyBase&);

} // namespace facebook::nimble::detail
