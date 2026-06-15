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

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"

// Per-segment cost models for SubIntSplitEncoding's DP selector.
//
// Each function estimates the compressed size in bits for encoding numValues
// items from a bit-range sub-stream of logical width bitWidth. The models
// correspond to nimble's present encodings and are derived from the same
// assumptions used in EncodingSizeEstimation.h (common prefix 6 bytes, nested
// encoding overhead, etc.).
//
// Deliberately avoids HLL cardinality estimation, entropy, and frame-residual
// tracking — the simplified SegmentMetrics provides enough signal for the DP
// to make directionally correct split decisions.

namespace facebook::nimble::detail::subintsplit {

// Smallest storage width in bits for a logical value of `bw` bits.
// Matches nimble's physical type selection: uint8/16/32/64.
inline constexpr uint8_t storageWidthBits(int bw) noexcept {
  if (bw <= 8) {
    return 8;
  }
  if (bw <= 16) {
    return 16;
  }
  if (bw <= 32) {
    return 32;
  }
  return 64;
}

// Union of MetricFlags needed across all cost models below.
inline MetricFlags allCostModelRequiredFlags() noexcept {
  return MetricFlag::MinMax | MetricFlag::RunStats | MetricFlag::UniqueCount |
      MetricFlag::DominantValue | MetricFlag::BitWidthHistogram;
}

// Trivial: store each value at its native storage width.
inline double trivialCostBits(
    const SegmentMetrics& /*m*/,
    size_t numValues,
    int bitWidth) noexcept {
  constexpr double kHeaderBits = 7.0 * 8.0; // prefix(6) + compressionType(1)
  return kHeaderBits +
      static_cast<double>(numValues) *
      static_cast<double>(storageWidthBits(bitWidth));
}

// FixedBitWidth: bit-pack using observed range, rounded to byte boundary.
// Required: MinMax
inline double fixedBitWidthCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  // prefix(6) + compressionType(1) + baseline(storageBytes) + bitWidth(1)
  const double baselineBytes =
      static_cast<double>(storageWidthBits(bitWidth)) / 8.0;
  const double headerBits = (7.0 + baselineBytes + 1.0) * 8.0;

  const uint8_t rangeWidth =
      m.range == 0 ? uint8_t{0} : static_cast<uint8_t>(std::bit_width(m.range));
  const uint8_t packedBits =
      std::min<uint8_t>(static_cast<uint8_t>(bitWidth), rangeWidth);
  // Sections are packed at their exact bit width (SubIntSplit encodes nested
  // sections with fixedBitWidthUseExactBits), so there is no byte-boundary
  // rounding: a 12-bit section costs 12 bits/value, not 16.

  return headerBits +
      static_cast<double>(packedBits) * static_cast<double>(numValues);
}

// Constant: zero cost when all values are equal, infinity otherwise.
// Required: MinMax
inline double constantCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  if (m.min != m.max) {
    return std::numeric_limits<double>::infinity();
  }
  // prefix(6) + stored value
  const double headerBits =
      (6.0 + static_cast<double>(storageWidthBits(bitWidth)) / 8.0) * 8.0;
  return headerBits;
}

// MainlyConstant: store one dominant value, a SparseBool mask marking the
// exception rows, and the exception values as a FixedBitWidth child. Effective
// when one value dominates the segment (say >=50%).
// Required: DominantValue, MinMax
inline double mainlyConstantCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  // Without a reliable dominant-value count (cardinality exceeded the cap)
  // there is no dominant value and MainlyConstant cannot help.
  if (m.dominantCountCapped || m.dominantCount == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const double storageBits = static_cast<double>(storageWidthBits(bitWidth));
  const double storageBytes = storageBits / 8.0;
  const size_t uncommonCount =
      m.dominantCount >= numValues ? 0 : numValues - m.dominantCount;

  // Outer: prefix(6) + two child-size fields(4 each) + the common value.
  const double outerBits = (6.0 + 4.0 + 4.0) * 8.0 + storageBits;

  // otherValues: FixedBitWidth over the uncommon values (observed range),
  // packed bits rounded up to a byte boundary.
  const uint8_t rangeWidth =
      m.range == 0 ? uint8_t{0} : static_cast<uint8_t>(std::bit_width(m.range));
  const uint8_t packedBits = static_cast<uint8_t>(
      (std::min<uint8_t>(static_cast<uint8_t>(bitWidth), rangeWidth) + 7u) &
      ~7u);

  const double otherHeaderBits = (7.0 + storageBytes + 1.0) * 8.0;
  const double otherValuesBits = otherHeaderBits +
      static_cast<double>(packedBits) * static_cast<double>(uncommonCount);

  const uint32_t indexWidth =
      numValues <= 1 ? 1u : static_cast<uint32_t>(std::bit_width(numValues));
  const double roundedIndexBits = static_cast<double>((indexWidth + 7u) & ~7u);
  // SparseBool: prefix(6) + tag(1) + FixedBitWidth header(7 + uint32 baseline +
  // 1)
  const double sparseHeaderBits = (6.0 + 1.0 + 7.0 + 4.0 + 1.0) * 8.0;
  const double isCommonBits = sparseHeaderBits +
      roundedIndexBits * static_cast<double>(uncommonCount + 1);

  return outerBits + otherValuesBits + isCommonBits;
}

// Dictionary: unique value table + bit-packed indices.
// Uses observed uniqueCount directly (no HLL blending). Directionally correct
// for the DP's purposes.
// Required: UniqueCount, MinMax
inline double dictionaryCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (m.uniqueCount == 0 || numValues == 0) {
    return 0.0;
  }
  const size_t uniques = m.uniqueCount;
  const uint8_t valueBits = storageWidthBits(bitWidth);
  const double dictBits =
      static_cast<double>(uniques) * static_cast<double>(valueBits);
  // Index width: ceil(log2(uniques)), clamped to 32
  const uint32_t indexWidth = (uniques <= 1)
      ? 1u
      : std::min(32u, static_cast<uint32_t>(std::bit_width(uniques - 1)));
  // Bit-packed indices rounded up to byte boundary
  const double roundedIndexBits = static_cast<double>((indexWidth + 7u) & ~7u);
  const double indexBits = roundedIndexBits * static_cast<double>(numValues);
  // prefix(6) + alphabetSize(4) + nested header overhead (~15 bytes)
  const double headerBits = (6.0 + 4.0 + 15.0) * 8.0;

  // Penalise when index width ≥ value width (dictionary doesn't compress).
  const double penalty = (indexWidth >= valueBits)
      ? 1.0 + 0.15 * (static_cast<double>(indexWidth) / valueBits - 1.0)
      : 1.0;

  return headerBits + penalty * (dictBits + indexBits);
}

// RLE: run values + bit-packed run lengths.
// Required: RunStats, MinMax
inline double
rleCostBits(const SegmentMetrics& m, size_t numValues, int bitWidth) noexcept {
  if (numValues == 0 || m.avgRunLength <= 0.0) {
    return 0.0;
  }
  // avgRunLength is scale-invariant, so extrapolate run count to full stream.
  const double estimatedRuns = static_cast<double>(numValues) / m.avgRunLength;
  // prefix(6) + runLengthsSize(4) + nested encoding overhead (~16 bytes)
  const double headerBits = (6.0 + 4.0 + 16.0) * 8.0;
  const double runValuesBits =
      estimatedRuns * static_cast<double>(storageWidthBits(bitWidth));
  // Assume 16-bit run lengths (conservative)
  const double runLengthsBits = estimatedRuns * 16.0;
  return headerBits + runValuesBits + runLengthsBits;
}

// Varint: variable-length integer storage. Only useful for ≥32-bit sections.
// Required: MinMax (max value determines byte width)
inline double varintCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0 || bitWidth < 32) {
    return std::numeric_limits<double>::infinity();
  }
  const uint64_t maxVal = m.max;
  uint8_t varintBytes;
  if (maxVal < (1ULL << 7)) {
    varintBytes = 1;
  } else if (maxVal < (1ULL << 14)) {
    varintBytes = 2;
  } else if (maxVal < (1ULL << 21)) {
    varintBytes = 3;
  } else if (maxVal < (1ULL << 28)) {
    varintBytes = 4;
  } else if (maxVal < (1ULL << 35)) {
    varintBytes = 5;
  } else if (maxVal < (1ULL << 42)) {
    varintBytes = 6;
  } else if (maxVal < (1ULL << 49)) {
    varintBytes = 7;
  } else if (maxVal < (1ULL << 56)) {
    varintBytes = 8;
  } else {
    varintBytes = 9;
  }
  // prefix(6) + baseline
  const double headerBits =
      (6.0 + static_cast<double>(storageWidthBits(bitWidth)) / 8.0) * 8.0;
  return headerBits +
      static_cast<double>(varintBytes) * 8.0 * static_cast<double>(numValues);
}

// SimdForBitpack: SIMD-friendly bit-packing of the observed [min, max] range.
// Required: MinMax
inline double simdForBitpackCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  const auto rowCount = static_cast<uint64_t>(numValues);
  uint64_t bytes;
  switch (storageWidthBits(bitWidth)) {
    case 8:
      bytes = SimdForBitpackEncoding<uint8_t>::estimateSize(
          rowCount, static_cast<uint8_t>(m.min), static_cast<uint8_t>(m.max));
      break;
    case 16:
      bytes = SimdForBitpackEncoding<uint16_t>::estimateSize(
          rowCount,
          static_cast<uint16_t>(m.min),
          static_cast<uint16_t>(m.max));
      break;
    case 32:
      bytes = SimdForBitpackEncoding<uint32_t>::estimateSize(
          rowCount,
          static_cast<uint32_t>(m.min),
          static_cast<uint32_t>(m.max));
      break;
    default:
      bytes = SimdForBitpackEncoding<uint64_t>::estimateSize(
          rowCount, m.min, m.max);
      break;
  }
  return static_cast<double>(bytes) * 8.0;
}

// PFOR: bit-packed "base" region sized to cover ~90% of values (by observed
// bit width), plus exception side-channels (positions + residual values) for
// the remainder. Mirrors PFOREncoding<T>::selectBaseBitWidth /
// PFOREncoding<T>::estimateSize, but operates on `m.bitWidthBuckets` directly
// instead of constructing a real Statistics<T>.
// Required: BitWidthHistogram
inline double
pforCostBits(const SegmentMetrics& m, size_t numValues, int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  if (bitWidth < 4) {
    // PFOR's fixed per-segment header (baseline + baseBitWidth +
    // numExceptions) dominates for very narrow segments.
    return std::numeric_limits<double>::infinity();
  }

  constexpr double kCoverageThreshold = 0.9;
  const uint64_t threshold = static_cast<uint64_t>(
      static_cast<double>(numValues) * kCoverageThreshold);

  uint8_t baseBitWidth = static_cast<uint8_t>(bitWidth);
  uint64_t numExceptions = 0;
  uint64_t cumulative = 0;
  for (size_t k = 0; k < m.bitWidthBuckets.size(); ++k) {
    cumulative += m.bitWidthBuckets[k];
    if (cumulative >= threshold) {
      const uint8_t bucketEndBitWidth =
          static_cast<uint8_t>(std::min<size_t>((k + 1) * 7, 64));
      baseBitWidth =
          std::min<uint8_t>(bucketEndBitWidth, static_cast<uint8_t>(bitWidth));
      numExceptions = static_cast<uint64_t>(numValues) - cumulative;
      break;
    }
  }

  const double storageBytes =
      static_cast<double>(storageWidthBits(bitWidth)) / 8.0;
  // prefix(6) + baseline(storageBytes) + baseBitWidth(1) + numExceptions(4)
  const double headerBits = (6.0 + storageBytes + 1.0 + 4.0) * 8.0;
  const double baseValuesBits =
      static_cast<double>(baseBitWidth) * static_cast<double>(numValues);

  // Exception side-channels are nested encodings; approximate as Trivial
  // sub-encodings (prefix(6) + 1 + raw values), as PFOREncoding::estimateSize
  // does.
  constexpr double kNestedHeaderBits = 7.0 * 8.0;
  const double positionsBits = numExceptions == 0
      ? 0.0
      : kNestedHeaderBits + static_cast<double>(numExceptions) * 32.0;
  const double valuesBits = numExceptions == 0
      ? 0.0
      : kNestedHeaderBits +
          static_cast<double>(numExceptions) *
              static_cast<double>(storageWidthBits(bitWidth));

  return headerBits + baseValuesBits + positionsBits + valuesBits;
}

// BlockBitPacking: per-block bit-packing with local baselines/widths.
// Calls the encoding's own estimateSize directly on the raw sample (always
// instantiated at uint64_t, regardless of `bitWidth` -- this overestimates
// per-block metadata for narrower sections, but keeps the DP directionally
// correct without per-width sample copies).
// Required: none beyond `segValues` itself.
inline double blockBitPackingCostBits(
    const std::vector<uint64_t>& segValues,
    size_t numValues,
    uint16_t blockSize = kBlockBitPackingBlockSize) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  const uint64_t bytes =
      BlockBitPackingEncoding<uint64_t>::estimateSize(segValues, blockSize);
  return static_cast<double>(bytes) * 8.0;
}

// Evaluate all cost models and return the minimum cost in bits.
// Also sets `bestEncoding` to the winning EncodingType.
inline double bestCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth,
    const std::vector<uint64_t>& segValues,
    EncodingType& bestEncoding) noexcept {
  double best = std::numeric_limits<double>::infinity();
  auto consider = [&](double cost, EncodingType type) noexcept {
    if (cost < best) {
      best = cost;
      bestEncoding = type;
    }
  };

  consider(trivialCostBits(m, numValues, bitWidth), EncodingType::Trivial);
  consider(
      fixedBitWidthCostBits(m, numValues, bitWidth),
      EncodingType::FixedBitWidth);
  consider(constantCostBits(m, numValues, bitWidth), EncodingType::Constant);
  consider(
      mainlyConstantCostBits(m, numValues, bitWidth),
      EncodingType::MainlyConstant);
  consider(rleCostBits(m, numValues, bitWidth), EncodingType::RLE);
  consider(varintCostBits(m, numValues, bitWidth), EncodingType::Varint);
  // Dictionary only when cardinality << numValues
  if (m.uniqueCount > 0 &&
      (m.uniqueCountCapped || m.uniqueCount < numValues / 2)) {
    consider(
        dictionaryCostBits(m, numValues, bitWidth), EncodingType::Dictionary);
  }
  consider(
      simdForBitpackCostBits(m, numValues, bitWidth),
      EncodingType::SimdForBitpack);
  consider(pforCostBits(m, numValues, bitWidth), EncodingType::PFOR);
  consider(
      blockBitPackingCostBits(segValues, numValues),
      EncodingType::BlockBitPacking);
  return best;
}

} // namespace facebook::nimble::detail::subintsplit
