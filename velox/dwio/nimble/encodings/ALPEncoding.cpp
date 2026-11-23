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

#include "velox/dwio/nimble/encodings/ALPEncoding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <tuple>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

namespace facebook::nimble {

namespace {

// Describes how a sample encodes under one (exponent, factor) pair.
struct EncodedSampleStats {
  // Values the pair cannot represent exactly.
  uint32_t exceptionCount{0};
  // Smallest ZigZag integer of a representable value.
  uint64_t minZigZag{std::numeric_limits<uint64_t>::max()};
  // Largest ZigZag integer of a representable value.
  uint64_t maxZigZag{0};
};

// Returns the stats of 'logicalValues' under (exponent, factor), from the
// vectorized transform with a scalar tail.
template <typename T>
EncodedSampleStats exponentFactorStats(
    std::span<const T> logicalValues,
    const typename ALPEncoding<T>::physicalType* physicalValues,
    int exponent,
    int factor) {
  using Alp = ALPEncoding<T>;
  const double exponentMultiplier = Alp::kPow10Double[exponent];
  const double factorMultiplier = Alp::kPow10Double[factor];
  EncodedSampleStats stats;
  const auto add = [&](bool representable, uint64_t zigZag) {
    if (!representable) {
      ++stats.exceptionCount;
      return;
    }
    stats.minZigZag = std::min(stats.minZigZag, zigZag);
    stats.maxZigZag = std::max(stats.maxZigZag, zigZag);
  };
  alignas(64) uint64_t zigZagLanes[Alp::kBatchSize];
  alignas(64) bool okLanes[Alp::kBatchSize];
  size_t i{0};
  for (; i + Alp::kBatchSize <= logicalValues.size(); i += Alp::kBatchSize) {
    Alp::batchTransform(
        logicalValues.data() + i,
        physicalValues + i,
        exponentMultiplier,
        factorMultiplier,
        zigZagLanes,
        okLanes);
    for (size_t lane = 0; lane < Alp::kBatchSize; ++lane) {
      add(okLanes[lane], zigZagLanes[lane]);
    }
  }
  for (; i < logicalValues.size(); ++i) {
    uint64_t zigZag{0};
    const bool representable = Alp::scalarTransformOne(
        logicalValues[i],
        physicalValues[i],
        exponentMultiplier,
        factorMultiplier,
        zigZag);
    add(representable, zigZag);
  }
  return stats;
}

// Estimates the encoded bits of 'numValues' values from their stats under one
// pair. Each value costs the bit width of the integer range, and each
// exception costs 'exceptionBits'.
uint64_t estimateBits(
    const EncodedSampleStats& stats,
    uint32_t numValues,
    uint64_t exceptionBits) {
  // Exceptions encode as zero, which widens the integer range.
  const uint64_t minEncoded = stats.exceptionCount == 0 ? stats.minZigZag : 0;
  const uint64_t bitWidth = 64 - std::countl_zero(stats.maxZigZag - minEncoded);
  return bitWidth * numValues + stats.exceptionCount * exceptionBits;
}

} // namespace

template <typename T>
std::pair<uint8_t, uint8_t> ALPEncoding<T>::findBestExponentFactor(
    std::span<const cppDataType> values,
    uint32_t rowCount,
    const Encoding::Options& options) {
  NIMBLE_CHECK(!values.empty(), "ALP parameter selection requires values.");
  NIMBLE_CHECK_LE(values.size(), rowCount);
  const auto numValues = static_cast<uint32_t>(values.size());
  const uint32_t sampleSize = estimateSampleSize(numValues);
  std::vector<cppDataType> samples;
  samples.reserve(sampleSize);
  for (uint32_t i = 0; i < sampleSize; ++i) {
    samples.emplace_back(
        values[detail::NestedAlpSizeEstimation::sampledRowIndex(
            i, sampleSize, numValues)]);
  }
  // Search every pair on a small subset of the sample, then pick among the
  // best few scales on the whole sample, as the ALP paper does. Searching the
  // whole sample costs about as much as encoding a stripe.
  constexpr uint32_t kSearchSampleSize = 64;
  constexpr size_t kNumCandidates = 5;
  const uint32_t searchSize = std::min(sampleSize, kSearchSampleSize);
  std::vector<cppDataType> searchSamples;
  searchSamples.reserve(searchSize);
  for (uint32_t i = 0; i < searchSize; ++i) {
    searchSamples.emplace_back(samples[uint64_t{i} * sampleSize / searchSize]);
  }
  const auto physicalValues =
      [](const std::vector<cppDataType>& logicalValues) {
        return EncodingPhysicalType<cppDataType>::asEncodingPhysicalTypeSpan(
                   std::span<const cppDataType>{logicalValues})
            .data();
      };

  // An exception stores its raw value and its position. The size estimate
  // prices the positions as FixedBitWidth over 'rowCount' rows: the bits of
  // the last row, rounded up to a byte unless 'options' asks for exact bits.
  int positionBits = velox::bits::bitsRequired(rowCount - 1);
  if (!options.fixedBitWidthUseExactBits) {
    positionBits = velox::bits::roundUp(positionBits, 8);
  }
  const uint64_t exceptionBits = sizeof(physicalType) * 8 + positionBits;
  struct Candidate {
    uint64_t numBits{std::numeric_limits<uint64_t>::max()};
    uint8_t exponent{0};
    uint8_t factor{0};
  };
  // The pair with the fewest bits on the search subset for each scale,
  // exponent minus factor. Pairs of one scale make the same integers, so the
  // candidates differ in scale; equal bits keep the pair found first.
  std::array<std::optional<Candidate>, kMaxExponent + 1> bestPerScale;
  // Smallest scale that represents every value of the subset. A scale at
  // least as large makes integers at least as large and has no fewer
  // exceptions, so it cannot encode smaller.
  int fullyRepresentedScale{kMaxExponent + 1};
  for (int exponent = 0; exponent <= kMaxExponent; ++exponent) {
    for (int factor = 0; factor <= std::min(exponent, kMaxFactor); ++factor) {
      const int scale = exponent - factor;
      if (scale >= fullyRepresentedScale) {
        continue;
      }
      const auto stats = exponentFactorStats<cppDataType>(
          searchSamples, physicalValues(searchSamples), exponent, factor);
      const uint64_t numBits = estimateBits(stats, searchSize, exceptionBits);
      auto& best = bestPerScale[scale];
      if (!best.has_value() || numBits < best->numBits) {
        best = Candidate{
            numBits,
            static_cast<uint8_t>(exponent),
            static_cast<uint8_t>(factor)};
      }
      if (stats.exceptionCount == 0) {
        fullyRepresentedScale = std::min(fullyRepresentedScale, scale);
      }
    }
  }
  std::vector<Candidate> candidates;
  candidates.reserve(bestPerScale.size());
  for (const auto& best : bestPerScale) {
    if (best.has_value()) {
      candidates.emplace_back(*best);
    }
  }
  const auto numCandidates = std::min(candidates.size(), kNumCandidates);
  std::partial_sort(
      candidates.begin(),
      candidates.begin() + numCandidates,
      candidates.end(),
      [](const Candidate& lhs, const Candidate& rhs) {
        return std::tie(lhs.numBits, lhs.exponent, lhs.factor) <
            std::tie(rhs.numBits, rhs.exponent, rhs.factor);
      });
  candidates.resize(numCandidates);

  // Equal bits on the whole sample keep the smaller exponent, then factor.
  Candidate best;
  for (const auto& candidate : candidates) {
    const auto stats = exponentFactorStats<cppDataType>(
        samples, physicalValues(samples), candidate.exponent, candidate.factor);
    const uint64_t numBits = estimateBits(stats, sampleSize, exceptionBits);
    if (std::tie(numBits, candidate.exponent, candidate.factor) <
        std::tie(best.numBits, best.exponent, best.factor)) {
      best = {numBits, candidate.exponent, candidate.factor};
    }
  }
  return {best.exponent, best.factor};
}

template <typename T>
std::optional<uint64_t> ALPEncoding<T>::estimateSize(
    std::span<const physicalType> values,
    const Encoding::Options& options) {
  if (values.empty()) {
    return std::nullopt;
  }

  NIMBLE_CHECK_LE(values.size(), std::numeric_limits<uint32_t>::max());
  const uint64_t rowCount = values.size();
  const uint32_t sampleSize = estimateSampleSize(rowCount);

  std::vector<physicalType> sampledValues;
  sampledValues.reserve(sampleSize);
  // Vary the offset within each interval to retain periodic value ranges.
  for (uint32_t i = 0; i < sampleSize; ++i) {
    const auto inputIndex = detail::NestedAlpSizeEstimation::sampledRowIndex(
        i, sampleSize, rowCount);
    sampledValues.push_back(values[inputIndex]);
  }

  return estimateSizeFromSample(rowCount, sampledValues, options);
}

template <typename T>
std::optional<uint64_t> ALPEncoding<T>::estimateSizeFromSample(
    uint64_t rowCount,
    std::span<const physicalType> sampledValues,
    const Encoding::Options& options) {
  NIMBLE_CHECK_GT(rowCount, 0, "ALP estimation requires non-empty input.");
  NIMBLE_CHECK(
      !sampledValues.empty(), "ALP estimation requires a non-empty sample.");
  NIMBLE_CHECK_LE(
      sampledValues.size(),
      rowCount,
      "ALP sample size cannot exceed the input row count.");

  NIMBLE_CHECK_LE(rowCount, std::numeric_limits<uint32_t>::max());
  const uint64_t sampleSize = sampledValues.size();

  std::vector<cppDataType> logicalValues;
  logicalValues.reserve(sampleSize);
  for (const auto value : sampledValues) {
    logicalValues.push_back(detail::alp::toLogical<cppDataType>(value));
  }

  const auto [exponent, factor] = findBestExponentFactor(
      std::span<const cppDataType>{logicalValues.data(), logicalValues.size()},
      static_cast<uint32_t>(rowCount),
      options);

  std::vector<uint64_t> encodedValues;
  encodedValues.reserve(sampleSize);
  std::vector<physicalType> exceptionValues;
  exceptionValues.reserve(sampleSize);
  uint64_t sampleExceptionCount{0};
  for (auto i = 0; i < sampleSize; ++i) {
    if (!canRepresentExactly(
            logicalValues[i],
            sampledValues[static_cast<size_t>(i)],
            exponent,
            factor)) {
      encodedValues.push_back(0);
      exceptionValues.push_back(sampledValues[static_cast<size_t>(i)]);
      ++sampleExceptionCount;
      continue;
    }

    const auto encoded =
        encodeValue(static_cast<double>(logicalValues[i]), exponent, factor);
    encodedValues.push_back(velox::ZigZag::encode(encoded));
  }

  using Cost = detail::NestedAlpSizeEstimation;
  const auto [minEncoded, maxEncoded] =
      std::minmax_element(encodedValues.begin(), encodedValues.end());
  const uint64_t nestedEncodedValuesSize = Cost::estimateChildSize<uint64_t>(
      rowCount, *minEncoded, *maxEncoded, options);
  const uint64_t exceptionCount =
      (sampleExceptionCount * rowCount + sampleSize - 1) / sampleSize;
  uint64_t exceptionPositionsSize{0};
  uint64_t exceptionValuesSize{0};
  if (exceptionCount > 0) {
    // Multiple exceptions have distinct positions in the full row range,
    // including when only one exception was observed in the sample.
    exceptionPositionsSize = Cost::estimateChildSize<uint32_t>(
        exceptionCount, 0, exceptionCount == 1 ? 0 : rowCount - 1, options);
    const auto [minException, maxException] =
        std::minmax_element(exceptionValues.begin(), exceptionValues.end());
    exceptionValuesSize = Cost::estimateChildSize<physicalType>(
        exceptionCount, *minException, *maxException, options);
  }
  const uint64_t metadataSize = kHeaderSize +
      (exceptionCount > 0 ? varint::varintSize(exceptionCount) : 0) +
      varint::varintSize(nestedEncodedValuesSize) +
      (exceptionCount > 0 ? varint::varintSize(exceptionPositionsSize) +
               varint::varintSize(exceptionValuesSize)
                          : 0);
  return Encoding::serializePrefixSize(
             static_cast<uint32_t>(rowCount), options.useVarintRowCount) +
      metadataSize + nestedEncodedValuesSize + exceptionPositionsSize +
      exceptionValuesSize;
}

template <typename T>
void ALPEncoding<T>::decodeBulkValues(
    const uint64_t* encodedValues,
    vector_size_t numValues,
    int exponent,
    int factor,
    cppDataType* output) {
  using UnsignedBatch = xsimd::batch<uint64_t>;
  using SignedBatch = xsimd::batch<int64_t>;
  using DoubleBatch = xsimd::batch<double>;
  constexpr auto kDecodeBatchSize =
      static_cast<vector_size_t>(DoubleBatch::size);
  const DoubleBatch exponentMultiplier(kPow10Double[exponent]);
  const DoubleBatch factorMultiplier(kPow10Double[factor]);
  vector_size_t row = 0;
  for (; row <= numValues - kDecodeBatchSize; row += kDecodeBatchSize) {
    // ALP stores transformed values as ZigZag-encoded uint64_t lanes for both
    // float and double inputs. Shifting removes the sign bit; XOR with the
    // zero or all-ones sign mask restores each signed integer.
    const auto zigZag = UnsignedBatch::load_unaligned(encodedValues + row);
    const auto signedBits =
        (zigZag >> 1) ^ (UnsignedBatch(0) - (zigZag & UnsignedBatch(1)));
    const auto integers = xsimd::bitwise_cast<SignedBatch>(signedBits);
    const auto restored = xsimd::batch_cast<double>(integers) *
        factorMultiplier / exponentMultiplier;
    restored.store_unaligned(output + row);
  }
  for (; row < numValues; ++row) {
    output[row] = decodeValue(
        velox::ZigZag::decode(encodedValues[row]), exponent, factor);
  }
}

template void ALPEncoding<float>::decodeBulkValues(
    const uint64_t*,
    vector_size_t,
    int,
    int,
    float*);
template void ALPEncoding<double>::decodeBulkValues(
    const uint64_t*,
    vector_size_t,
    int,
    int,
    double*);

template std::optional<uint64_t> ALPEncoding<float>::estimateSize(
    std::span<const uint32_t>,
    const Encoding::Options&);
template std::optional<uint64_t> ALPEncoding<float>::estimateSizeFromSample(
    uint64_t,
    std::span<const uint32_t>,
    const Encoding::Options&);
template std::optional<uint64_t> ALPEncoding<double>::estimateSize(
    std::span<const uint64_t>,
    const Encoding::Options&);
template std::optional<uint64_t> ALPEncoding<double>::estimateSizeFromSample(
    uint64_t,
    std::span<const uint64_t>,
    const Encoding::Options&);
template std::pair<uint8_t, uint8_t> ALPEncoding<float>::findBestExponentFactor(
    std::span<const float>,
    uint32_t,
    const Encoding::Options&);
template std::pair<uint8_t, uint8_t>
ALPEncoding<double>::findBestExponentFactor(
    std::span<const double>,
    uint32_t,
    const Encoding::Options&);

} // namespace facebook::nimble
