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
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"

#include <algorithm>
#include <bit>
#include <limits>

#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"

namespace facebook::nimble::subintsplit {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Every encoding writes the 6-byte common Encoding prefix.
constexpr double kPrefixBytes = 6.0;

double toBits(double bytes) noexcept {
  return bytes * 8.0;
}

double storageBits(int bitWidth) noexcept {
  return static_cast<double>(sectionStorageBits(bitWidth));
}

double storageBytes(int bitWidth) noexcept {
  return static_cast<double>(sectionStorageBytes(bitWidth));
}

// FixedBitWidth header: prefix + compressionType + baseline + bitWidth byte.
double fixedBitWidthHeaderBits(int bitWidth) noexcept {
  return toBits(kPrefixBytes + 1.0 + storageBytes(bitWidth) + 1.0);
}

// Bits needed to distinguish `count` values, rounded up to a byte boundary,
// which is how the bit-packed index children are laid out.
double roundedIndexBits(size_t count) noexcept {
  const uint32_t width =
      count <= 1 ? 1u : static_cast<uint32_t>(std::bit_width(count));
  return static_cast<double>((width + 7u) & ~7u);
}

// Bits a FixedBitWidth child spends per value given the observed range.
uint8_t packedBitsForRange(uint64_t range, int bitWidth) noexcept {
  const uint8_t rangeWidth =
      range == 0 ? uint8_t{0} : static_cast<uint8_t>(std::bit_width(range));
  return std::min<uint8_t>(static_cast<uint8_t>(bitWidth), rangeWidth);
}

uint8_t varintBytesFor(uint64_t maxValue) noexcept {
  for (uint8_t bytes = 1; bytes < 9; ++bytes) {
    if (maxValue < (uint64_t{1} << (7 * bytes))) {
      return bytes;
    }
  }
  return 9;
}

} // namespace

MetricFlags allCostModelRequiredFlags() noexcept {
  return MetricFlag::MinMax | MetricFlag::RunStats | MetricFlag::UniqueCount |
      MetricFlag::DominantValue;
}

double trivialCostBits(
    const SectionMetrics& /*metrics*/,
    size_t numValues,
    int bitWidth) noexcept {
  const double headerBits = toBits(kPrefixBytes + 1.0);
  return headerBits + static_cast<double>(numValues) * storageBits(bitWidth);
}

double fixedBitWidthCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  const double packedBits =
      static_cast<double>(packedBitsForRange(metrics.range, bitWidth));
  return fixedBitWidthHeaderBits(bitWidth) +
      packedBits * static_cast<double>(numValues);
}

double constantCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }
  if (metrics.min != metrics.max) {
    return kInfinity;
  }
  return toBits(kPrefixBytes + storageBytes(bitWidth));
}

double mainlyConstantCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0) {
    return 0.0;
  }

  // Without a reliable dominant-value count (cardinality exceeded the cap)
  // there is no dominant value and MainlyConstant cannot help.
  if (metrics.dominantCountCapped || metrics.dominantCount == 0) {
    return kInfinity;
  }

  const size_t numUncommon = metrics.dominantCount >= numValues
      ? 0
      : numValues - metrics.dominantCount;

  // Outer: prefix + two child-size fields (4 bytes each) + the common value.
  const double outerBits =
      toBits(kPrefixBytes + 4.0 + 4.0) + storageBits(bitWidth);

  // otherValues: a FixedBitWidth child over the uncommon values, whose packed
  // width rounds up to a byte boundary.
  const uint8_t packed = packedBitsForRange(metrics.range, bitWidth);
  const double otherPackedBits = static_cast<double>((packed + 7u) & ~7u);
  const double otherValuesBits = fixedBitWidthHeaderBits(bitWidth) +
      otherPackedBits * static_cast<double>(numUncommon);

  // isCommon: a SparseBool child, itself prefix + tag + a FixedBitWidth index
  // child with a uint32 baseline.
  const double sparseHeaderBits = toBits(kPrefixBytes + 1.0 + 7.0 + 4.0 + 1.0);
  const double isCommonBits = sparseHeaderBits +
      roundedIndexBits(numValues) * static_cast<double>(numUncommon + 1);

  return outerBits + otherValuesBits + isCommonBits;
}

double dictionaryCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept {
  if (metrics.uniqueCount == 0 || numValues == 0) {
    return 0.0;
  }

  const double valueBits = storageBits(bitWidth);
  const double alphabetBits =
      static_cast<double>(metrics.uniqueCount) * valueBits;

  const uint32_t indexWidth = metrics.uniqueCount <= 1
      ? 1u
      : std::min(
            32u,
            static_cast<uint32_t>(std::bit_width(metrics.uniqueCount - 1)));
  const double indexBits = static_cast<double>((indexWidth + 7u) & ~7u) *
      static_cast<double>(numValues);

  // prefix + alphabetSize(4) + nested header overhead (~15 bytes).
  const double headerBits = toBits(kPrefixBytes + 4.0 + 15.0);

  // Penalise when the index is no narrower than the value: the dictionary is
  // not compressing, it is just adding a level of indirection.
  const double penalty = (static_cast<double>(indexWidth) >= valueBits)
      ? 1.0 + 0.15 * (static_cast<double>(indexWidth) / valueBits - 1.0)
      : 1.0;

  return headerBits + penalty * (alphabetBits + indexBits);
}

double rleCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0 || metrics.avgRunLength <= 0.0) {
    return 0.0;
  }

  // avgRunLength is scale-invariant, so the run count extrapolates from the
  // sample to the full stream.
  const double numRuns = static_cast<double>(numValues) / metrics.avgRunLength;

  // prefix + runLengthsSize(4) + nested encoding overhead (~16 bytes).
  const double headerBits = toBits(kPrefixBytes + 4.0 + 16.0);
  const double runValuesBits = numRuns * storageBits(bitWidth);

  // Run lengths are assumed 16-bit, which is conservative.
  constexpr double kRunLengthBits = 16.0;
  return headerBits + runValuesBits + numRuns * kRunLengthBits;
}

double varintCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept {
  if (numValues == 0 || bitWidth < 32) {
    return kInfinity;
  }
  const double headerBits = toBits(kPrefixBytes + storageBytes(bitWidth));
  return headerBits +
      toBits(static_cast<double>(varintBytesFor(metrics.max))) *
      static_cast<double>(numValues);
}

double bestCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth,
    EncodingType& bestEncoding) noexcept {
  double best = kInfinity;
  const auto consider = [&](double cost, EncodingType type) noexcept {
    if (cost < best) {
      best = cost;
      bestEncoding = type;
    }
  };

  consider(
      trivialCostBits(metrics, numValues, bitWidth), EncodingType::Trivial);
  consider(
      fixedBitWidthCostBits(metrics, numValues, bitWidth),
      EncodingType::FixedBitWidth);
  consider(
      constantCostBits(metrics, numValues, bitWidth), EncodingType::Constant);
  consider(
      mainlyConstantCostBits(metrics, numValues, bitWidth),
      EncodingType::MainlyConstant);
  consider(rleCostBits(metrics, numValues, bitWidth), EncodingType::RLE);
  consider(varintCostBits(metrics, numValues, bitWidth), EncodingType::Varint);

  // A dictionary only pays off when cardinality is well below the value count.
  if (metrics.uniqueCount > 0 &&
      (metrics.uniqueCountCapped || metrics.uniqueCount < numValues / 2)) {
    consider(
        dictionaryCostBits(metrics, numValues, bitWidth),
        EncodingType::Dictionary);
  }
  return best;
}

} // namespace facebook::nimble::subintsplit
