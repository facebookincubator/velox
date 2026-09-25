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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"

// DP-based bit-range split selector for SubIntSplitEncoding.

namespace facebook::nimble::subintsplit {

struct SelectorConfig {
  int minSectionWidth{1};
  double splitPenalty{10.0}; // extra bits charged per additional split boundary
  // Withdrawing this moves the boundaries the DP picks, not merely the
  // encoding it names for a segment, since cost is what the DP minimises over.
  bool allowHuffman{true};
  // Same effect as allowHuffman: withdrawing it moves the boundaries the DP
  // picks, not only the segment encoding.
  bool allowDeltaBlock{true};
  // Weight of a segment's decode cost against its size. Zero reproduces
  // size-only selection exactly (see DecodeCostWeighting).
  DecodeCostWeighting decodeWeighting{};

  /// Trims bit planes constant across the whole sample into a single
  /// Constant section at each edge before scoring. Off by default because
  /// trimming changes the plan. Ignored below minSectionWidth or when
  /// Constant is disallowed.
  bool trimConstantPlanes{false};
  /// Relative set-rate change below which a position is not considered a
  /// split boundary. 0.0 considers every position.
  double boundaryPruneThreshold{0.0};
  /// Ceiling on candidate boundaries, the strongest by set-rate change kept.
  /// 0 is unlimited.
  size_t maxCandidateBoundaries{0};
  /// Widest section the grid scores, beyond the whole range, which is always
  /// scored so the DP keeps a fallback. 0 is unlimited.
  int maxSectionWidth{0};
  /// Widest section for which unique and dominant-value counts are collected.
  /// Wider sections leave the frequency-driven models unusable. 0 is
  /// unlimited.
  int frequencyMetricsMaxWidth{0};
  /// Decode cost charged per additional section, in bits per value of the
  /// full stream, on top of splitPenalty. 0.0 disables the term.
  double decodeCostBitsPerValue{0.0};
  /// Rows in the full stream, which decodeCostBitsPerValue is scaled by. Set
  /// by the entry points that are handed the row count.
  size_t streamRowCount{0};
};

/// Default set-rate threshold for boundary pruning.
constexpr double kBoundaryPruneThreshold = 0.001;

/// Extra bits the DP charges each section after the first.
inline double sectionPenaltyBits(const SelectorConfig& cfg) noexcept {
  return cfg.splitPenalty +
      cfg.decodeCostBitsPerValue * static_cast<double>(cfg.streamRowCount);
}

/// Whether a section starting at `start` over `cell` is charged no penalty:
/// the first section, a trimmed constant edge, and the first section after a
/// trimmed low edge.
inline bool startsFreeSection(
    const std::vector<SectionCost>& grid,
    int start,
    const SectionCost& cell) noexcept {
  return start == 0 || cell.trimmedEdge || grid[start - 1].trimmedEdge;
}

/// The contiguous range of bit positions that actually vary across the sample.
struct ActiveBitRange {
  int lo{0};
  int hi{-1};

  bool allConstant() const noexcept {
    return hi < lo;
  }
};

/// Finds the lowest and highest bit that is not identical across every sample.
ActiveBitRange findActiveBitRange(
    const std::vector<uint64_t>& samples,
    int numBits);

/// Bit positions in [lo, hi + 1] where a split is worth considering: lo and
/// hi + 1, plus every interior position whose set rate differs from the one
/// below it by at least `threshold`, capped at `maxCount` (0 is unlimited),
/// keeping the largest changes.
std::vector<int> candidateBoundaries(
    const std::vector<uint64_t>& samples,
    int lo,
    int hi,
    double threshold,
    size_t maxCount = 0);

/// A constant bit-plane run, stored as a single Constant section.
SectionPlan makeConstantSection(int bitStart, int bitEnd);

/// Which cells of the split grid are scored, derived once from the sample and
/// the configuration. Unrestricted, the default, scores every cell exactly as
/// the grid always has.
struct GridLayout {
  bool unrestricted{true};
  /// Every plane is constant; only the whole-range cell is kept, as Constant.
  bool allConstant{false};
  /// Varying range. Planes below lo and above hi form Constant edge cells.
  int lo{0};
  int hi{0};
  /// isBoundary[p]: a section may start or end (exclusive) at bit p.
  std::vector<uint8_t> isBoundary;
};

/// Builds the grid layout for `cfg` over `samples`, `sz` bits wide.
GridLayout makeGridLayout(
    const std::vector<uint64_t>& samples,
    int sz,
    const SelectorConfig& cfg,
    bool constantAllowed);

inline SelectorConfig defaultSelectorConfig() noexcept {
  return SelectorConfig{
      .minSectionWidth = 1,
      .splitPenalty = 10.0,
      .allowHuffman = true,
      .allowDeltaBlock = true,
      .decodeWeighting = DecodeCostWeighting{}};
}

// Incremental bit-range value extractor.
// Builds values[i] = bits [bitStart..bitEnd] of sample[i], extending one bit
// at a time to reuse work across the inner loop of the segment-evaluation grid.
class BitRangeExtractor {
 public:
  explicit BitRangeExtractor(const std::vector<uint64_t>& samples)
      : samples_(samples),
        values_(samples.size(), uint64_t{0}),
        bitStart_(-1),
        bitEnd_(-1) {}

  void reset(int bitStart) {
    bitStart_ = bitStart;
    bitEnd_ = bitStart;
    const size_t n = samples_.size();
    for (size_t i = 0; i < n; ++i) {
      values_[i] = (samples_[i] >> bitStart_) & uint64_t{1};
    }
  }

  void extend(int bitEnd) {
    if (bitEnd <= bitEnd_) {
      return;
    }
    const size_t n = samples_.size();
    for (int b = bitEnd_ + 1; b <= bitEnd; ++b) {
      const int shift = b - bitStart_;
      const uint64_t maskShift = uint64_t{1} << shift;
      for (size_t i = 0; i < n; ++i) {
        const uint64_t bit = (samples_[i] >> b) & uint64_t{1};
        values_[i] |= bit * maskShift;
      }
    }
    bitEnd_ = bitEnd;
  }

  const std::vector<uint64_t>& values() const noexcept {
    return values_;
  }

 private:
  const std::vector<uint64_t>& samples_;
  std::vector<uint64_t> values_;
  int bitStart_;
  int bitEnd_;
};

// Counts, for every bit range [bitStart, bitEnd] of a sample, the metrics
// that need no pass over the range's extracted values (value frequencies,
// runs, bit-width histogram), each prepared once per left edge or per sample
// and read per range rather than recomputed for every one of the grid's
// ranges. Frequencies come from one sort of the reversed, bit-shifted
// samples, since samples equal on a range are then contiguous and group
// boundaries follow from adjacent-pair prefix lengths.
class BitRangeCounter {
 public:
  explicit BitRangeCounter(const std::vector<uint64_t>& samples)
      : samples_{samples} {
    const size_t numSamples = samples_.size();
    // setBitCounts_[high * 64 + low]: samples whose highest set bit at or
    // below `high` is at or above `low`.
    setBitCounts_.assign(64 * 64, 0);
    std::array<uint32_t, 65> widthCounts{};
    for (int high = 0; high < 64; ++high) {
      const uint64_t mask =
          high == 63 ? ~uint64_t{0} : ((uint64_t{1} << (high + 1)) - 1);
      widthCounts.fill(0);
      for (size_t i = 0; i < numSamples; ++i) {
        ++widthCounts[std::bit_width(samples_[i] & mask)];
      }
      uint32_t atLeast = 0;
      for (int low = high; low >= 0; --low) {
        atLeast += widthCounts[low + 1];
        setBitCounts_[high * 64 + low] = atLeast;
      }
    }
  }

  // Starts the left edge at `bitStart`. Cheapest when called for consecutive
  // left edges in ascending order, which is how the grid walks them.
  void reset(int bitStart) {
    const size_t numSamples = samples_.size();
    if (bitStart_ >= 0 && bitStart == bitStart_ + 1) {
      const auto firstOne =
          std::partition_point(keys_.begin(), keys_.end(), [](uint64_t key) {
            return (key >> 63) == 0;
          });
      for (auto& key : keys_) {
        key <<= 1;
      }
      mergeBuffer_.resize(numSamples);
      std::merge(
          keys_.begin(), firstOne, firstOne, keys_.end(), mergeBuffer_.begin());
      keys_.swap(mergeBuffer_);
    } else {
      keys_.resize(numSamples);
      for (size_t i = 0; i < numSamples; ++i) {
        keys_[i] = reverseBits(samples_[i]) << bitStart;
      }
      std::sort(keys_.begin(), keys_.end());
    }
    bitStart_ = bitStart;

    // Adjacent sorted pairs ordered by common prefix length, so that the group
    // boundaries of each width are a contiguous run of this order.
    std::array<uint32_t, 65> prefixCounts{};
    prefixLengths_.resize(numSamples);
    for (size_t i = 1; i < numSamples; ++i) {
      const auto length =
          static_cast<uint8_t>(std::countl_zero(keys_[i - 1] ^ keys_[i]));
      prefixLengths_[i] = length;
      ++prefixCounts[length];
    }
    boundaryStarts_[0] = 0;
    for (int length = 0; length <= 64; ++length) {
      boundaryStarts_[length + 1] =
          boundaryStarts_[length] + prefixCounts[length];
    }
    boundariesByWidth_.resize(numSamples > 0 ? numSamples - 1 : 0);
    std::array<uint32_t, 66> next = boundaryStarts_;
    for (size_t i = 1; i < numSamples; ++i) {
      boundariesByWidth_[next[prefixLengths_[i]]++] = static_cast<uint32_t>(i);
    }

    // Bit i set where a group starts at sorted position i. Position 0 always
    // starts one, and position numSamples is set as a sentinel end.
    groupStarts_.assign(numSamples / 64 + 1, 0);
    setGroupStart(0);
    setGroupStart(numSamples);
    boundariesAdded_ = 0;
    frequenciesCurrent_ = false;

    // Runs.
    std::array<uint32_t, 65> transitionCounts{};
    for (size_t i = 1; i < numSamples; ++i) {
      const uint64_t differing = (samples_[i] ^ samples_[i - 1]) >> bitStart;
      ++transitionCounts[std::countr_zero(differing)];
    }
    transitionsBelow_[0] = 0;
    for (int width = 0; width < 64; ++width) {
      transitionsBelow_[width + 1] =
          transitionsBelow_[width] + transitionCounts[width];
    }
  }

  // Frequencies for [bitStart, bitEnd]. `bitEnd` must not fall between calls
  // on one left edge.
  const FrequencyCounts& frequencies(int bitEnd) {
    const int width = bitEnd - bitStart_ + 1;
    const uint32_t boundaries = boundaryStarts_[width];
    if (frequenciesCurrent_ && boundaries == boundariesAdded_) {
      return frequencies_;
    }
    for (; boundariesAdded_ < boundaries; ++boundariesAdded_) {
      setGroupStart(boundariesByWidth_[boundariesAdded_]);
    }
    recomputeFrequencies();
    frequenciesCurrent_ = true;
    return frequencies_;
  }

  RangeCounts counts(int bitEnd) {
    RangeCounts result = countsWithoutFrequencies(bitEnd);
    result.frequencies = frequencies(bitEnd);
    return result;
  }

  // The run count and bit-width histogram for [bitStart, bitEnd], leaving the
  // frequencies unset, for a range whose frequency-driven models are not
  // wanted.
  RangeCounts countsWithoutFrequencies(int bitEnd) const {
    RangeCounts result;
    const int width = bitEnd - bitStart_ + 1;
    result.runCount = transitionsBelow_[width] + 1;

    // valuesAtLeast[k]: samples whose bit width on the range is at least 7k.
    const auto numSamples = static_cast<uint32_t>(samples_.size());
    std::array<uint32_t, 11> valuesAtLeast{};
    valuesAtLeast[0] = numSamples;
    for (int bucket = 1; bucket <= 9; ++bucket) {
      if (7 * bucket <= width) {
        valuesAtLeast[bucket] =
            setBitCounts_[bitEnd * 64 + bitStart_ + 7 * bucket - 1];
      }
    }
    for (int bucket = 0; bucket < 10; ++bucket) {
      result.bitWidthBuckets[bucket] =
          valuesAtLeast[bucket] - valuesAtLeast[bucket + 1];
    }
    return result;
  }

 private:
  static uint64_t reverseBits(uint64_t value) noexcept {
    value = ((value >> 1) & 0x5555'5555'5555'5555) |
        ((value & 0x5555'5555'5555'5555) << 1);
    value = ((value >> 2) & 0x3333'3333'3333'3333) |
        ((value & 0x3333'3333'3333'3333) << 2);
    value = ((value >> 4) & 0x0F0F'0F0F'0F0F'0F0F) |
        ((value & 0x0F0F'0F0F'0F0F'0F0F) << 4);
    return __builtin_bswap64(value);
  }

  void setGroupStart(size_t position) noexcept {
    groupStarts_[position / 64] |= uint64_t{1} << (position % 64);
  }

  // Position of the first bit at or after `from` that is set in the group
  // starts, or clear when `wantClear`.
  size_t findBit(size_t from, bool wantClear) const noexcept {
    const size_t numSamples = samples_.size();
    size_t word = from / 64;
    uint64_t bits = wantClear ? ~groupStarts_[word] : groupStarts_[word];
    bits &= ~uint64_t{0} << (from % 64);
    while (bits == 0) {
      ++word;
      if (word >= groupStarts_.size()) {
        return numSamples;
      }
      bits = wantClear ? ~groupStarts_[word] : groupStarts_[word];
    }
    return std::min(word * 64 + std::countr_zero(bits), numSamples);
  }

  // Group sizes from the boundaries: a group of size s >= 2 is a run of s - 1
  // clear bits after its start, and every group not found that way holds one
  // sample. So the walk costs the groups with repeats, not every group.
  void recomputeFrequencies() {
    const size_t numSamples = samples_.size();
    frequencies_ = FrequencyCounts{};
    const size_t groupCount = boundariesAdded_ + 1;
    frequencies_.uniqueCount = groupCount;
    LargestFrequencies largest;
    uint32_t dominant = 1;
    size_t repeatedGroups = 0;
    size_t position = 1;
    while (position < numSamples) {
      const size_t runStart = findBit(position, /*wantClear=*/true);
      if (runStart >= numSamples) {
        break;
      }
      const size_t runEnd = findBit(runStart, /*wantClear=*/false);
      const auto size = static_cast<uint32_t>(runEnd - runStart + 1);
      dominant = std::max(dominant, size);
      frequencies_.doubletonCount += (size == 2) ? 1 : 0;
      largest.offer(size);
      ++repeatedGroups;
      position = runEnd + 1;
    }
    frequencies_.singletonCount = groupCount - repeatedGroups;
    for (size_t i = 0; i < std::min<size_t>(frequencies_.singletonCount, 8);
         ++i) {
      largest.offer(1);
    }
    frequencies_.dominantCount = dominant;
    frequencies_.largest = largest.values();
  }

  const std::vector<uint64_t>& samples_;
  std::vector<uint32_t> setBitCounts_;
  int bitStart_{-1};
  // Samples' bits from bitStart_ up, reversed and sorted.
  std::vector<uint64_t> keys_;
  std::vector<uint64_t> mergeBuffer_;
  // prefixLengths_[i]: common prefix length of keys_[i - 1] and keys_[i].
  std::vector<uint8_t> prefixLengths_;
  // Sorted positions i >= 1, ascending by prefix length; those with a prefix
  // shorter than w are the first boundaryStarts_[w].
  std::vector<uint32_t> boundariesByWidth_;
  std::array<uint32_t, 66> boundaryStarts_{};
  std::vector<uint64_t> groupStarts_;
  uint32_t boundariesAdded_{0};
  FrequencyCounts frequencies_;
  bool frequenciesCurrent_{false};
  // transitionsBelow_[w]: adjacent row pairs that differ within the first w
  // bits of the range.
  std::array<uint32_t, 65> transitionsBelow_{};
};

struct SelectorResult {
  std::vector<SectionPlan> sections;
  double totalCost{0.0};
  // Estimated size alone, in bits: what the sections store, with no split
  // penalty and no decode term. Answers what the plan costs the file, as
  // opposed to totalCost, what it costs the DP.
  double totalSizeBits{0.0};
  // Estimated decode cost composed over sections by
  // combineSectionDecodeNanos, including the per-section assembly term.
  double totalDecodeNanosPerRow{0.0};
};

// Runs the DP split selector on `samples` (uint64_t values drawn from a
// physical-type stream of `kBits` width). `fullCount` scales cost model
// scores from the sample size to the full stream's. `costFn` scores a single
// segment and returns weighted cost, size, and decode cost together, since
// the DP minimises the weighted figure while callers need to report the
// other two. buildSectionCostGrid costs the bit ranges [l..r] that `layout`
// admits as a row-major sz*sz grid indexed l * sz + r; selectSplitsImpl runs
// the DP over it. `samples` must be non-empty and `sz` in [1, 64].
template <typename CostFn>
inline std::vector<SectionCost> buildSectionCostGrid(
    const std::vector<uint64_t>& samples,
    int sz,
    size_t fullCount,
    const SelectorConfig& cfg,
    const GridLayout& layout,
    CostFn&& costFn) {
  std::vector<SectionCost> bestCost(sz * sz);
  const auto constantCell = [] {
    SectionCost cell;
    cell.weightedBits = 0.0;
    cell.sizeBits = 0.0;
    cell.decodeNanosPerRow = 0.0;
    cell.encoding = EncodingType::Constant;
    cell.trimmedEdge = true;
    return cell;
  };
  if (!layout.unrestricted && layout.allConstant) {
    bestCost[sz - 1] = constantCell();
    return bestCost;
  }
  if (!layout.unrestricted && layout.lo > 0) {
    bestCost[layout.lo - 1] = constantCell();
  }
  if (!layout.unrestricted && layout.hi < sz - 1) {
    bestCost[(layout.hi + 1) * sz + (sz - 1)] = constantCell();
  }

  const MetricFlags allFlags = allCostModelRequiredFlags();
  const MetricFlags withoutFrequency = allFlags &
      ~static_cast<MetricFlags>(MetricFlag::UniqueCount) &
      ~static_cast<MetricFlags>(MetricFlag::DominantValue) &
      ~static_cast<MetricFlags>(MetricFlag::FrequencyTiers);
  MetricCollector collector;
  BitRangeExtractor extractor(samples);
  const size_t numSamples = samples.size();
  // BitRangeCounter cannot describe a capped count, since capping depends on
  // arrival order and this path sorts. Falls back to the uncapped counting
  // path rather than asserting, since a caller sampling more than the cap is
  // not doing anything wrong.
  const bool rangeCounts = numSamples <= MetricCollector::kUniqueCountCap;
  std::optional<BitRangeCounter> counter;
  if (rangeCounts) {
    counter.emplace(samples);
  }
  const double scale =
      static_cast<double>(fullCount) / static_cast<double>(numSamples);

  for (int l = 0; l < sz; ++l) {
    const bool activeStart = layout.unrestricted ||
        (l >= layout.lo && l <= layout.hi &&
         layout.isBoundary[static_cast<size_t>(l)] != 0);
    if (!activeStart && l != 0) {
      continue;
    }
    extractor.reset(l);
    if (rangeCounts) {
      counter->reset(l);
    }
    for (int r = l; r < sz; ++r) {
      const bool wholeRange = l == 0 && r == sz - 1;
      const bool isActiveRange = l == layout.lo && r == layout.hi;
      const int bitWidth = r - l + 1;
      if (!layout.unrestricted && !wholeRange) {
        if (!activeStart || r > layout.hi ||
            layout.isBoundary[static_cast<size_t>(r + 1)] == 0) {
          continue;
        }
        if (cfg.maxSectionWidth > 0 && bitWidth > cfg.maxSectionWidth &&
            !isActiveRange) {
          continue;
        }
      }
      extractor.extend(r);
      const std::vector<uint64_t>& segValues = extractor.values();
      const bool wantFrequency = layout.unrestricted ||
          cfg.frequencyMetricsMaxWidth <= 0 ||
          bitWidth <= cfg.frequencyMetricsMaxWidth;
      const MetricFlags flags = wantFrequency ? allFlags : withoutFrequency;
      SectionMetrics metrics;
      if (rangeCounts) {
        metrics = collector.compute(
            segValues,
            flags,
            wantFrequency ? counter->counts(r)
                          : counter->countsWithoutFrequencies(r));
      } else {
        metrics = collector.compute(segValues, flags);
      }
      SectionCost cell =
          costFn(metrics, numSamples, fullCount, bitWidth, segValues);
      cell.weightedBits *= scale;
      cell.sizeBits *= scale;
      bestCost[l * sz + r] = cell;
    }
  }
  return bestCost;
}

/// As above, over every cell.
template <typename CostFn>
inline std::vector<SectionCost> buildSectionCostGrid(
    const std::vector<uint64_t>& samples,
    int sz,
    size_t fullCount,
    CostFn&& costFn) {
  return buildSectionCostGrid(
      samples,
      sz,
      fullCount,
      SelectorConfig{},
      GridLayout{},
      std::forward<CostFn>(costFn));
}

// Runs the split DP over a grid buildSectionCostGrid already costed, so a
// caller pricing the same sample more than once pays for the grid once.
inline SelectorResult selectSplitsOverGrid(
    const std::vector<SectionCost>& bestCost,
    int sz,
    const SelectorConfig& cfg) {
  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  std::vector<EncodingType> chosen(sz + 1, EncodingType::Trivial);
  dp[0] = 0.0;

  for (int i = 1; i <= sz; ++i) {
    for (int j = 0; j < i; ++j) {
      const int width = i - j;
      if (width < cfg.minSectionWidth) {
        continue;
      }
      const auto& choice = bestCost[j * sz + (i - 1)];
      if (!std::isfinite(choice.weightedBits)) {
        continue;
      }
      const double splitCost = startsFreeSection(bestCost, j, choice)
          ? 0.0
          : sectionPenaltyBits(cfg);
      const double candidate = dp[j] + choice.weightedBits + splitCost;
      if (candidate < dp[i]) {
        dp[i] = candidate;
        prev[i] = j;
        chosen[i] = choice.encoding;
      }
    }
  }

  SelectorResult result;
  result.totalCost = dp[sz];

  if (!std::isfinite(result.totalCost)) {
    SectionPlan fallback;
    fallback.bitStart = 0;
    fallback.bitEnd = sz - 1;
    fallback.encoding = EncodingType::Trivial;
    const SectionCost& whole = bestCost[0 * sz + (sz - 1)];
    fallback.cost = whole.weightedBits;
    fallback.sizeCostBits = whole.sizeBits;
    fallback.decodeNanosPerRow = whole.decodeNanosPerRow;
    result.sections.push_back(fallback);
    result.totalCost = fallback.cost;
    result.totalSizeBits = fallback.sizeCostBits;
    result.totalDecodeNanosPerRow = combineSectionDecodeNanos(
        cfg.decodeWeighting.accessPattern,
        cfg.decodeWeighting.readPath,
        std::span<const double>(&result.sections.back().decodeNanosPerRow, 1));
    return result;
  }

  int idx = sz;
  while (idx > 0) {
    const int start = prev[idx];
    if (start < 0) {
      break;
    }
    SectionPlan plan;
    plan.bitStart = start;
    plan.bitEnd = idx - 1;
    plan.encoding = chosen[idx];
    const SectionCost& cell = bestCost[start * sz + (idx - 1)];
    plan.cost = cell.weightedBits;
    plan.sizeCostBits = cell.sizeBits;
    plan.decodeNanosPerRow = cell.decodeNanosPerRow;
    result.sections.push_back(plan);
    idx = start;
  }

  std::reverse(result.sections.begin(), result.sections.end());

  std::vector<double> sectionNanos;
  sectionNanos.reserve(result.sections.size());
  for (const auto& segment : result.sections) {
    result.totalSizeBits += segment.sizeCostBits;
    sectionNanos.push_back(segment.decodeNanosPerRow);
  }
  result.totalDecodeNanosPerRow = combineSectionDecodeNanos(
      cfg.decodeWeighting.accessPattern,
      cfg.decodeWeighting.readPath,
      sectionNanos);
  return result;
}

template <typename CostFn>
inline SelectorResult selectSplitsImpl(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg,
    CostFn&& costFn,
    bool constantAllowed = true) {
  if (samples.empty() || kBits <= 0) {
    return {};
  }
  const int sz = std::min(kBits, 64);
  SelectorConfig sized = cfg;
  if (sized.streamRowCount == 0) {
    sized.streamRowCount = fullCount;
  }
  return selectSplitsOverGrid(
      buildSectionCostGrid(
          samples,
          sz,
          fullCount,
          sized,
          makeGridLayout(samples, sz, sized, constantAllowed),
          std::forward<CostFn>(costFn)),
      sz,
      sized);
}

/// Whether a section may be encoded as Constant under `allowed`, where an
/// empty set allows everything.
inline bool allowsConstant(const AllowedEncodings& allowed) {
  return allowed.empty() || allowed.contains(EncodingType::Constant);
}

// The per-range cost function the split DP minimises, over `allowed` only.
// Holds a reference to `allowed`, which must outlive it.
inline auto restrictedSectionCostFn(
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg) {
  return [&allowed,
          allowHuffman = cfg.allowHuffman,
          allowDeltaBlock = cfg.allowDeltaBlock,
          weighting = cfg.decodeWeighting](
             const SectionMetrics& m,
             size_t numValues,
             size_t streamCount,
             int bitWidth,
             const std::vector<uint64_t>& segValues) noexcept {
    return bestSectionCost(
        m,
        numValues,
        streamCount,
        bitWidth,
        segValues,
        allowed,
        allowHuffman,
        allowDeltaBlock,
        weighting);
  };
}

// Selects splits costing segments against `allowed` only. An empty set costs
// every encoding, so a caller can pass one through unconditionally.
inline SelectorResult selectSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  return selectSplitsImpl(
      samples,
      kBits,
      fullCount,
      cfg,
      restrictedSectionCostFn(allowed, cfg),
      allowsConstant(allowed));
}

/// Costs the split grid over `allowed` under every planner restriction `cfg`
/// asks for. The grid the encoder plans on, and the one its shortlist is taken
/// from.
inline std::vector<SectionCost> buildRestrictedCostGrid(
    const std::vector<uint64_t>& samples,
    int sz,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg) {
  return buildSectionCostGrid(
      samples,
      sz,
      fullCount,
      cfg,
      makeGridLayout(samples, sz, cfg, allowsConstant(allowed)),
      restrictedSectionCostFn(allowed, cfg));
}

// Selects splits over the full encoding inventory. Forwards to
// selectSplitsRestricted with an empty allowed set so that cfg.allowHuffman
// and cfg.allowDeltaBlock are still respected.
inline SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  static const AllowedEncodings kAll;
  return selectSplitsRestricted(samples, kBits, fullCount, kAll, cfg);
}

// The k cheapest segmentations of [0, sz) over `grid`, cheapest first, under
// the same split penalty and minimum segment width as the DP. Lets a planner
// hand a shortlist to a more accurate, more expensive scorer instead of
// trusting the DP's single argmin. When `cuts` is non-empty a range may only
// start and end at positions it marks.
inline std::vector<std::vector<SectionPlan>> kBestSplits(
    const std::vector<SectionCost>& grid,
    int sz,
    const SelectorConfig& cfg,
    size_t k,
    const std::vector<bool>& cuts = {}) {
  struct Entry {
    double cost;
    int prevPosition;
    size_t prevRank;
  };
  const auto isCut = [&cuts](int position) {
    return cuts.empty() || cuts[position];
  };
  std::vector<std::vector<Entry>> best(sz + 1);
  best[0].push_back({0.0, -1, 0});
  for (int end = 1; end <= sz; ++end) {
    if (!isCut(end)) {
      continue;
    }
    std::vector<Entry> candidates;
    for (int start = 0; start < end; ++start) {
      if (end - start < cfg.minSectionWidth || !isCut(start)) {
        continue;
      }
      const double rangeCost = grid[start * sz + (end - 1)].weightedBits;
      if (!std::isfinite(rangeCost)) {
        continue;
      }
      const double penalty =
          startsFreeSection(grid, start, grid[start * sz + (end - 1)])
          ? 0.0
          : sectionPenaltyBits(cfg);
      for (size_t rank = 0; rank < best[start].size(); ++rank) {
        candidates.push_back(
            {best[start][rank].cost + rangeCost + penalty, start, rank});
      }
    }
    const size_t keep = std::min(k, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + keep,
        candidates.end(),
        [](const Entry& a, const Entry& b) { return a.cost < b.cost; });
    candidates.resize(keep);
    best[end] = std::move(candidates);
  }

  std::vector<std::vector<SectionPlan>> plans;
  for (size_t rank = 0; rank < best[sz].size(); ++rank) {
    std::vector<SectionPlan> plan;
    int position = sz;
    size_t atRank = rank;
    while (position > 0) {
      const Entry& entry = best[position][atRank];
      const SectionCost& cell = grid[entry.prevPosition * sz + (position - 1)];
      SectionPlan segment;
      segment.bitStart = entry.prevPosition;
      segment.bitEnd = position - 1;
      segment.encoding = cell.encoding;
      segment.cost = cell.weightedBits;
      segment.sizeCostBits = cell.sizeBits;
      segment.decodeNanosPerRow = cell.decodeNanosPerRow;
      plan.push_back(segment);
      position = entry.prevPosition;
      atRank = entry.prevRank;
    }
    std::reverse(plan.begin(), plan.end());
    plans.push_back(std::move(plan));
  }
  return plans;
}

// Shortlists split plans for the hybrid planner from one costing of the
// grid: the k cheapest plans, then the k cheapest cut only at `cuts`. Plans
// may repeat across the two lists; callers that re-price them cache by
// range.
inline std::vector<std::vector<SectionPlan>> shortlistSplitsOverGrid(
    const std::vector<SectionCost>& grid,
    int sz,
    const SelectorConfig& cfg,
    size_t k,
    const std::vector<bool>& cuts) {
  auto plans = kBestSplits(grid, sz, cfg, k);
  if (!cuts.empty()) {
    for (auto& plan : kBestSplits(grid, sz, cfg, k, cuts)) {
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

// shortlistSplitsOverGrid, costing the grid over `allowed` first.
inline std::vector<std::vector<SectionPlan>> shortlistSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg,
    size_t k,
    const std::vector<bool>& cuts) {
  if (samples.empty() || kBits <= 0) {
    return {};
  }
  const int sz = std::min(kBits, 64);
  SelectorConfig sized = cfg;
  if (sized.streamRowCount == 0) {
    sized.streamRowCount = fullCount;
  }
  return shortlistSplitsOverGrid(
      buildRestrictedCostGrid(samples, sz, fullCount, allowed, sized),
      sz,
      sized,
      k,
      cuts);
}

} // namespace facebook::nimble::subintsplit
