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
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

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
      MetricFlag::DominantValue | MetricFlag::BitWidthHistogram | MetricFlag::DeltaStats |
      MetricFlag::FrequencyTiers;
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
// Delegates the otherValues and isCommon sub-costs directly to
// FixedBitWidthEncoding::estimateSize / SparseBoolEncoding::estimateSize --
// the same calls MainlyConstantEncoding::estimateSize itself makes (see
// MainlyConstantEncoding.h) -- instead of a separate hand-rolled formula, so
// the two estimators cannot drift apart.
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
  const auto rowCount = static_cast<uint64_t>(numValues);
  const uint64_t uncommonCount =
      m.dominantCount >= numValues ? 0 : rowCount - m.dominantCount;

  // Outer: prefix(6) + two child-size fields(4 each) + the common value.
  const double outerBits = (6.0 + 4.0 + 4.0) * 8.0 + storageBits;

  uint64_t otherValuesBytes;
  switch (storageWidthBits(bitWidth)) {
    case 8:
      otherValuesBytes = FixedBitWidthEncoding<uint8_t>::estimateSize(
          uncommonCount, m.min, m.max, Encoding::Options{});
      break;
    case 16:
      otherValuesBytes = FixedBitWidthEncoding<uint16_t>::estimateSize(
          uncommonCount, m.min, m.max, Encoding::Options{});
      break;
    case 32:
      otherValuesBytes = FixedBitWidthEncoding<uint32_t>::estimateSize(
          uncommonCount, m.min, m.max, Encoding::Options{});
      break;
    default:
      otherValuesBytes = FixedBitWidthEncoding<uint64_t>::estimateSize(
          uncommonCount, m.min, m.max, Encoding::Options{});
      break;
  }

  const uint64_t isCommonBytes =
      SparseBoolEncoding::estimateSize(rowCount, uncommonCount, Encoding::Options{});

  return outerBits + static_cast<double>(otherValuesBytes) * 8.0 +
      static_cast<double>(isCommonBytes) * 8.0;
}

// Dictionary: unique value table + bit-packed indices.
// Delegates both nested streams directly to the same estimators
// DictionaryEncoding::estimateSize uses (see DictionaryEncoding.h): the
// indices stream via FixedBitWidthEncoding<uint32_t>::estimateSize(rowCount,
// 0, uniqueCount-1), and the alphabet via
// min(TrivialEncoding, FixedBitWidthEncoding)::estimateSize(uniqueCount, min,
// max). Only the numeric path is modeled; SIS operates on raw uint64_t
// bit-range slices, never strings.
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
  const uint64_t uniques = static_cast<uint64_t>(m.uniqueCount);
  const auto rowCount = static_cast<uint64_t>(numValues);
  const Encoding::Options options{};

  const uint64_t indicesBytes = FixedBitWidthEncoding<uint32_t>::estimateSize(
      rowCount, /*minValue=*/0, uniques - 1, options);

  uint64_t alphabetBytes;
  switch (storageWidthBits(bitWidth)) {
    case 8:
      alphabetBytes = std::min(
          TrivialEncoding<uint8_t>::estimateSize(uniques),
          FixedBitWidthEncoding<uint8_t>::estimateSize(
              uniques, m.min, m.max, options));
      break;
    case 16:
      alphabetBytes = std::min(
          TrivialEncoding<uint16_t>::estimateSize(uniques),
          FixedBitWidthEncoding<uint16_t>::estimateSize(
              uniques, m.min, m.max, options));
      break;
    case 32:
      alphabetBytes = std::min(
          TrivialEncoding<uint32_t>::estimateSize(uniques),
          FixedBitWidthEncoding<uint32_t>::estimateSize(
              uniques, m.min, m.max, options));
      break;
    default:
      alphabetBytes = std::min(
          TrivialEncoding<uint64_t>::estimateSize(uniques),
          FixedBitWidthEncoding<uint64_t>::estimateSize(
              uniques, m.min, m.max, options));
      break;
  }

  // Outer: prefix(6) + alphabetSize(4), matching DictionaryEncoding's layout.
  const double outerBits = (6.0 + 4.0) * 8.0;
  return outerBits + static_cast<double>(alphabetBytes) * 8.0 +
      static_cast<double>(indicesBytes) * 8.0;
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
  const auto bytes =
      BlockBitPackingEncoding<uint64_t>::estimateSize(segValues, blockSize);
  if (!bytes.has_value()) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(bytes.value()) * 8.0;
}

// Huffman: canonical Huffman coding over the observed value alphabet.
// Delegates to HuffmanEncoding<uint64_t>::estimateSize directly (rather than
// approximating from SegmentMetrics) since the exact per-symbol code length
// depends on the full frequency distribution, which SegmentMetrics does not
// retain. Only evaluated when the segment's cardinality is low enough for
// Huffman to apply (see the `bestCostBits` guard below), keeping the
// Statistics<uint64_t>::create() cost bounded.
// Required: none beyond `segValues` itself.
inline double huffmanCostBits(
    const std::vector<uint64_t>& segValues,
    size_t numValues) noexcept {
  if (numValues < 2) {
    return std::numeric_limits<double>::infinity();
  }
  const std::span<const uint64_t> values(segValues.data(), numValues);
  const auto statistics = Statistics<uint64_t>::create(values);
  const auto bytes = HuffmanEncoding<uint64_t>::estimateSize(
      values, statistics, Encoding::Options{});
  if (!bytes.has_value()) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(bytes.value()) * 8.0;
}

// DeltaBlock: fixed-size blocks, each storing a base value plus bit-packed
// non-decreasing deltas from that base. Delegates directly to
// DeltaBlockEncoding<uint64_t>::estimateSize on the raw sample, mirroring
// blockBitPackingCostBits -- DeltaBlockEncoding's own estimator is already
// exact (it walks real block boundaries), so approximating from
// SegmentMetrics would only lose precision. Returns infinity for any block
// containing a decrease, matching DeltaBlockEncoding's non-decreasing-only
// design.
// Required: none beyond `segValues` itself.
inline double deltaBlockCostBits(
    const std::vector<uint64_t>& segValues,
    size_t numValues,
    const Encoding::Options& options = {}) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  const std::span<const uint64_t> values(segValues.data(), numValues);
  const auto bytes = DeltaBlockEncoding<uint64_t>::estimateSize(
      values, options);
  if (!bytes.has_value()) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(bytes.value()) * 8.0;
}

// Delta: positive-delta encoding with restatements for non-monotonic steps.
// Infinity unless at least 90% of consecutive steps are non-decreasing
// (matches DeltaEncoding's positive-delta-only design — frequent decreases
// force expensive restatements).
// Required: DeltaStats
inline double deltaCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues < 2) {
    return std::numeric_limits<double>::infinity();
  }
  const double monotonicFraction = static_cast<double>(m.monotonicCount) /
      static_cast<double>(numValues - 1);
  constexpr double kMonotonicThreshold = 0.9;
  if (monotonicFraction < kMonotonicThreshold) {
    return std::numeric_limits<double>::infinity();
  }

  // Sized to the largest kept (non-decreasing) delta, not the average: a
  // fixed-width packed array must cover every value it stores, and a
  // right-skewed delta distribution makes the average a severe
  // underestimate of the width actually required (see SegmentMetrics::
  // maxDelta). Matches FixedBitWidthEncoding's own exact-bits sizing.
  const uint8_t deltaBitWidth = m.maxDelta == 0
      ? uint8_t{0}
      : static_cast<uint8_t>(std::bit_width(m.maxDelta));

  const double restatementFraction = 1.0 - monotonicFraction;
  // At least one restatement (the leading value) is always present.
  const double numRestatements =
      std::max(1.0, restatementFraction * static_cast<double>(numValues));

  // Three nested sub-encodings (deltas, restatements, isRestatements), each
  // with its own ~7-byte header, plus the outer prefix(6) + two 4-byte
  // relative offsets.
  constexpr double kNestedHeaderBits = 7.0 * 8.0;
  constexpr double kOuterHeaderBits = (6.0 + 4.0 + 4.0) * 8.0;

  const double deltasBits = kNestedHeaderBits +
      static_cast<double>(numValues) * static_cast<double>(deltaBitWidth);
  const double restatementsBits = kNestedHeaderBits +
      numRestatements * static_cast<double>(storageWidthBits(bitWidth));
  // isRestatements is a bool stream that is true only at restatement
  // positions -- nested-encoded, so delegate to SparseBoolEncoding's own
  // estimator (as the MainlyConstant/Dictionary fixes above do) instead of
  // charging a flat 1 bit/value, which drastically overestimates when
  // restatements are rare (the common case for mostly-monotonic data).
  const double isRestatementsBits =
      static_cast<double>(SparseBoolEncoding::estimateSize(
          static_cast<uint64_t>(numValues),
          static_cast<uint64_t>(numRestatements),
          Encoding::Options{})) *
      8.0;

  return kOuterHeaderBits + deltasBits + restatementsBits +
      isRestatementsBits;
}

// FOR (Frame of Reference): fixed-size frames, each bit-packed against a
// local minimum (reference). The local bit width is estimated from the
// average step size scaled to the frame size -- a random-walk heuristic
// where the local range over a frame of `kForFrameSize` steps grows roughly
// with avgAbsDelta -- capped by the segment's overall range.
// Required: MinMax, DeltaStats
inline double
forCostBits(const SegmentMetrics& m, size_t numValues, int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  constexpr uint32_t kForFrameSize = 128;
  const uint32_t numFrames =
      static_cast<uint32_t>((numValues + kForFrameSize - 1) / kForFrameSize);

  const double avgAbsDelta = numValues > 1
      ? static_cast<double>(m.sumAbsDelta) /
          static_cast<double>(numValues - 1)
      : 0.0;
  const double localRange = std::min(
      static_cast<double>(m.range),
      avgAbsDelta * static_cast<double>(kForFrameSize) / 2.0);
  const uint8_t localBits = localRange < 1.0
      ? uint8_t{0}
      : static_cast<uint8_t>(
            std::bit_width(static_cast<uint64_t>(localRange)));

  // prefix(6) + compressionType(1) + frameSize(4) + numFrames(4) +
  // enableBitOffsets(1)
  constexpr double kOuterHeaderBits = (6.0 + 1.0 + 4.0 + 4.0 + 1.0) * 8.0;
  // Per-frame metadata streams (bitWidths, references, bitOffsets), each a
  // nested encoding with its own ~7-byte header.
  constexpr double kNestedHeaderBits = 7.0 * 8.0;
  const double bitWidthsBits =
      kNestedHeaderBits + static_cast<double>(numFrames) * 8.0;
  const double referencesBits = kNestedHeaderBits +
      static_cast<double>(numFrames) *
          static_cast<double>(storageWidthBits(bitWidth));
  const double bitOffsetsBits =
      kNestedHeaderBits + static_cast<double>(numFrames) * 64.0;
  const double packedBits = static_cast<double>(numValues) * localBits;

  return kOuterHeaderBits + bitWidthsBits + referencesBits + bitOffsetsBits +
      packedBits;
}

// Number of PerTierBitmaps tiers FrequencyPartitionEncoding::encode would
// create for `uniqueCount` distinct values, mirroring its tier-capacity table
// (FrequencyPartitionEncoding.h: keyBitOptions = {1,2,4,8,16,32} bits with
// capacities 2/4/16/256/65280/~4.29B). Each tier created needs its own N-bit
// bitmap in the index payload (FrequencyPartitionEncoding.h's "Index payload
// layout (PerTierBitmaps)"), so this directly drives indexBits below.
// Assuming a fixed 2 tiers regardless of uniqueCount -- rather than the 3-5
// tiers typical for a segment near the 1024-unique cap -- was the single
// largest source of FPE's cost underestimate.
inline uint32_t frequencyPartitionNumTiers(uint64_t uniqueCount) noexcept {
  constexpr uint64_t kCapacities[] = {2, 4, 16, 256, 65280, 4294901760ull};
  uint64_t assigned = 0;
  uint32_t tiers = 0;
  for (uint64_t capacity : kCapacities) {
    if (assigned >= uniqueCount) {
      break;
    }
    ++tiers;
    assigned += capacity;
  }
  return std::max(tiers, 1u);
}

// FrequencyPartition: tier-based dictionary encoding where the top-K
// most-frequent values are stored with narrow (1/2-bit) keys. Requires an
// indexed mode (PerTierBitmaps) that preserves original row order; without an
// index, materialize() would desync sibling SubIntSplit segments.
// Returns infinity when the unique count is unknown, capped, or > 1024
// (FPE's overhead dominates for high-cardinality segments).
// Required: UniqueCount, FrequencyTiers (topKCoverage populated)
inline double frequencyPartitionCostBits(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0 || m.uniqueCount == 0 || m.uniqueCountCapped ||
      m.uniqueCount > 1024) {
    return std::numeric_limits<double>::infinity();
  }

  const double n = static_cast<double>(numValues);

  // Tier 0 (1-bit keys, capacity 2): top-2 most frequent values.
  // Tier 1 (2-bit keys, capacity 4): next tier up to top-8 (proxy).
  // Remainder: fallback at full storage width.
  const double tier0Coverage = m.topKCoverage[1]; // top-2 values → 1-bit keys
  const double tier1Coverage =
      std::max(0.0, m.topKCoverage[3] - m.topKCoverage[1]); // next → 2-bit
  const double fallbackCoverage = std::max(0.0, 1.0 - m.topKCoverage[3]);

  const double keyCostBits = tier0Coverage * n * 1.0 +
      tier1Coverage * n * 2.0 +
      fallbackCoverage * n * static_cast<double>(storageWidthBits(bitWidth));

  // PerTierBitmaps index: one N-bit bitmap per active tier (4-byte
  // bitmapByteCount prefix + numValues bits rounded to a 64-bit word), not a
  // fixed 2 tiers -- see frequencyPartitionNumTiers.
  const uint32_t numTiers = frequencyPartitionNumTiers(m.uniqueCount);
  const double bitmapBits = std::ceil(n / 64.0) * 64.0 + 32.0;
  const double indexBits = static_cast<double>(numTiers) * bitmapBits;

  // One dictionary + one key stream per active tier, each a nested
  // sub-encoding with a ~7-byte header.
  const double kTierOverheadBits =
      static_cast<double>(numTiers) * 2.0 * 7.0 * 8.0;

  // Outer prefix + numPartitions + nested partitionOffsets/partitionSizes.
  constexpr double kOuterHeaderBits = (6.0 + 4.0 + 4.0 + 4.0 + 2.0 * 7.0) * 8.0;

  return kOuterHeaderBits + keyCostBits + indexBits + kTierOverheadBits;
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
  consider(deltaCostBits(m, numValues, bitWidth), EncodingType::Delta);
  consider(forCostBits(m, numValues, bitWidth), EncodingType::FOR);
  // FrequencyPartition is only viable for low-cardinality segments.
  if (m.uniqueCount > 0 && !m.uniqueCountCapped && m.uniqueCount <= 1024) {
    consider(
        frequencyPartitionCostBits(m, numValues, bitWidth),
        EncodingType::FrequencyPartition);
  }
  // Huffman is only viable within its supported alphabet size; skip the
  // Statistics<uint64_t>::create() call entirely otherwise.
  if (m.uniqueCount > 0 && !m.uniqueCountCapped &&
      m.uniqueCount <= HuffmanEncoding<uint64_t>::kMaxSymbols) {
    consider(huffmanCostBits(segValues, numValues), EncodingType::Huffman);
  }
  consider(
      deltaBlockCostBits(segValues, numValues), EncodingType::DeltaBlock);
  return best;
}

} // namespace facebook::nimble::detail::subintsplit
