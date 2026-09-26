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
#include "velox/dwio/nimble/encodings/selection/SampledEncodingSizeEstimation.h"

#include <folly/hash/Hash.h>

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

} // namespace

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
      estimateSelectedChildSize<T>(*childPolicy, sample, numChildRows, options);
}

template std::optional<uint64_t> estimateNestedFloatingPointSize<float>(
    EncodingType,
    std::span<const uint32_t>,
    uint32_t,
    const Statistics<uint32_t>&,
    EncodingSelectionPolicyBase&,
    const Encoding::Options&);
template std::optional<uint64_t> estimateNestedFloatingPointSize<double>(
    EncodingType,
    std::span<const uint64_t>,
    uint32_t,
    const Statistics<uint64_t>&,
    EncodingSelectionPolicyBase&,
    const Encoding::Options&);

} // namespace facebook::nimble::detail
