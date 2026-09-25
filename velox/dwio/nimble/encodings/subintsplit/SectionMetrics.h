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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h" // @manual=fbsource//third-party/abseil-cpp:container__flat_hash_map
#include "velox/common/base/SimdUtil.h"

// Lightweight per-segment metric collection for SubIntSplitEncoding's DP
// planner. Deliberately minimal, unlike Nimble's Statistics<T>, since it
// runs over the 64x64 bit-range grid on a small sample.

namespace facebook::nimble::subintsplit {

enum class MetricFlag : uint32_t {
  None = 0,
  MinMax = 1u << 0, // min, max, range
  RunStats = 1u << 1, // runCount, avgRunLength
  UniqueCount = 1u << 2, // uniqueCount (raw, capped)
  DominantValue = 1u << 3, // dominantCount (most-frequent value's frequency)
  BitWidthHistogram = 1u << 4, // bitWidthBuckets
  DeltaStats = 1u << 5, // sumAbsDelta, monotonicCount, maxDelta
  FrequencyTiers = 1u << 6, // topKCoverage (requires UniqueCount)
  All = (1u << 7) - 1,
};
using MetricFlags = uint32_t;

inline constexpr MetricFlags operator|(MetricFlag a, MetricFlag b) noexcept {
  return static_cast<MetricFlags>(a) | static_cast<MetricFlags>(b);
}
inline constexpr MetricFlags operator|(MetricFlags a, MetricFlag b) noexcept {
  return a | static_cast<MetricFlags>(b);
}
inline constexpr bool hasFlag(MetricFlags flags, MetricFlag f) noexcept {
  return (flags & static_cast<MetricFlags>(f)) != 0;
}

struct SectionMetrics {
  uint64_t min{0};
  uint64_t max{0};
  uint64_t range{0};

  size_t uniqueCount{0};
  bool uniqueCountCapped{false};

  // Frequency of the most common value. Used by the MainlyConstant cost model.
  // Unreliable (and flagged capped) once cardinality exceeds the unique cap.
  size_t dominantCount{0};
  bool dominantCountCapped{false};

  size_t runCount{0};
  double avgRunLength{0.0};

  // bitWidthBuckets[i] counts values v with bit_width(v) in [7*i, 7*i+6],
  // for i in [0, 8]; bucket 9 catches bit_width(v) >= 63. Approximates
  // PFOREncoding<T>::selectBaseBitWidth's histogram directly on bit_width(v)
  // rather than bit_width(v - min), since segment values are already small.
  std::array<uint32_t, 10> bitWidthBuckets{};

  // Sum of |v[i] - v[i-1]| and count of non-decreasing pairs, used to
  // estimate average step size and monotonic fraction for delta/FOR costs.
  uint64_t sumAbsDelta{0};
  size_t monotonicCount{0};

  // Max delta over non-decreasing pairs only, since decreasing pairs are
  // restated rather than delta-encoded. A fixed-width packed array must be
  // sized to this max, not the average, or a skewed distribution overflows it.
  uint64_t maxDelta{0};

  // Distinct values seen exactly once and exactly twice, over the rows
  // actually counted. Feeds estimatedStreamUniqueCount in CostModel.h to
  // extrapolate the stream's distinct count from the sample.
  size_t singletonCount{0};
  size_t doubletonCount{0};
  // Rows counted into singletonCount/doubletonCount. Equal to the segment
  // length unless capping stopped the count early, which is exactly when the
  // estimate matters most, so the two must travel together.
  size_t countedRows{0};

  // Cumulative coverage fraction for the top-1/2/4/8 most-frequent distinct
  // values, valid only when FrequencyTiers was requested and
  // uniqueCountCapped is false.
  std::array<double, 4> topKCoverage{};
};

/// Frequency metrics a caller may supply instead of having MetricCollector
/// count them, for a caller holding a structure that already knows them.
struct FrequencyCounts {
  size_t uniqueCount{0};
  uint32_t dominantCount{0};
  // The eight largest frequencies, descending, zero-padded.
  std::array<uint32_t, 8> largest{};
  // Distinct values seen exactly once and exactly twice. See the fields of the
  // same name on SectionMetrics.
  size_t singletonCount{0};
  size_t doubletonCount{0};
};

/// Everything about a segment that does not depend on the order of its rows
/// beyond adjacency, for a caller that counts these for a bit range without
/// scanning the range's extracted values.
struct RangeCounts {
  FrequencyCounts frequencies;
  // Runs of equal adjacent values, at least one.
  size_t runCount{0};
  // See SectionMetrics::bitWidthBuckets.
  std::array<uint32_t, 10> bitWidthBuckets{};
};

// Keeps the eight largest frequencies offered to it, descending and
// zero-padded, since coverage never needs more than eight and the zero
// padding lets a short alphabet's coverage sum correctly past its end.
class LargestFrequencies {
 public:
  void offer(uint32_t frequency) noexcept {
    if (frequency <= largest_[7]) {
      return;
    }
    size_t i = 7;
    while (i > 0 && largest_[i - 1] < frequency) {
      largest_[i] = largest_[i - 1];
      --i;
    }
    largest_[i] = frequency;
  }

  const std::array<uint32_t, 8>& values() const noexcept {
    return largest_;
  }

 private:
  std::array<uint32_t, 8> largest_{};
};

// Single-pass metric collector for extracted bit-range values. Counts unique
// and dominant values two ways -- a direct-indexed histogram where values are
// narrow enough to index, a frequency map otherwise (capped at
// kUniqueCountCap). Its counting structures are reusable members so that
// compute(), called once per bit-range in an O(kBits^2) grid, avoids
// reallocating them each time.
class MetricCollector {
 public:
  static constexpr size_t kUniqueCountCap = 1
      << 14; // 16K cap (lighter than full HLL)

  // Segments whose values fit in this many bits are counted in a
  // direct-indexed array rather than a hash map. Only touched slots are
  // cleared afterward, so the table's resident cost tracks the segment's
  // cardinality rather than its full size.
  static constexpr int kDirectHistogramBits = 16;

  // Maps bit_width(v) to a bucket index in [0, 9], grouping every 7 bits.
  static constexpr size_t bitWidthBucket(uint64_t v) noexcept {
    return std::min<size_t>(static_cast<size_t>(std::bit_width(v)) / 7, 9);
  }

  SectionMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags = static_cast<MetricFlags>(MetricFlag::All)) {
    return computeImpl(values, flags, nullptr);
  }

  /// Computes the metrics that come from scanning the segment, and takes the
  /// ones that come from counting it rather than counting it again. For a
  /// caller that already holds the frequencies, such as one maintaining an
  /// equality partition across a grid of bit ranges.
  SectionMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags,
      const FrequencyCounts& counts) {
    return computeImpl(values, flags, &counts);
  }

  /// As above, and takes the run count and the bit-width histogram as well,
  /// leaving only the order-dependent scan metrics to compute from `values`.
  SectionMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags,
      const RangeCounts& counts) {
    const size_t count = values.size();
    if (count == 0 || !hasFlag(flags, MetricFlag::MinMax) ||
        !hasFlag(flags, MetricFlag::RunStats) ||
        !hasFlag(flags, MetricFlag::BitWidthHistogram) ||
        !hasFlag(flags, MetricFlag::DeltaStats)) {
      return computeImpl(values, flags, &counts.frequencies);
    }
    SectionMetrics out = scanMinMaxAndDeltas(values);
    out.runCount = counts.runCount;
    out.avgRunLength =
        static_cast<double>(count) / static_cast<double>(out.runCount);
    out.bitWidthBuckets = counts.bitWidthBuckets;
    // The same fields, and only those, that computeImpl's supplied-counts
    // path fills, so that the two agree on every field and the planner sees
    // identical metrics whichever the grid takes.
    if (hasFlag(flags, MetricFlag::UniqueCount) ||
        hasFlag(flags, MetricFlag::FrequencyTiers)) {
      out.uniqueCount = counts.frequencies.uniqueCount;
      out.singletonCount = counts.frequencies.singletonCount;
      out.doubletonCount = counts.frequencies.doubletonCount;
      out.countedRows = count;
    }
    if (hasFlag(flags, MetricFlag::DominantValue)) {
      out.dominantCount = counts.frequencies.dominantCount;
    }
    if (hasFlag(flags, MetricFlag::FrequencyTiers)) {
      fillCoverage(counts.frequencies.largest, count, out.topKCoverage);
    }
    return out;
  }

 private:
  // The scan metrics that depend on row order beyond adjacency: extremes and
  // delta statistics. Vectorised over signed 64-bit lanes, which is exact
  // only while every value stays below 2^63; the loop checks that bound
  // rather than assuming it, falling back to an unsigned rescan otherwise.
  static SectionMetrics scanMinMaxAndDeltas(
      const std::vector<uint64_t>& values) {
    using Batch = xsimd::batch<int64_t>;
    const size_t count = values.size();
    const auto* signedValues = reinterpret_cast<const int64_t*>(values.data());
    uint64_t minimum = values[0];
    uint64_t maximum = values[0];
    uint64_t sumAbsDelta = 0;
    uint64_t monotonic = 0;
    uint64_t maxDelta = 0;
    size_t next = 1;
    if (count > Batch::size) {
      const auto zero = Batch::broadcast(0);
      const auto one = Batch::broadcast(1);
      auto orOfValues = Batch::broadcast(signedValues[0]);
      auto minimumBatch = orOfValues;
      auto maximumBatch = orOfValues;
      auto sumBatch = zero;
      auto risingBatch = zero;
      auto maxDeltaBatch = zero;
      for (; next + Batch::size <= count; next += Batch::size) {
        const auto value = Batch::load_unaligned(signedValues + next);
        const auto previous = Batch::load_unaligned(signedValues + next - 1);
        orOfValues = orOfValues | value;
        minimumBatch = xsimd::min(minimumBatch, value);
        maximumBatch = xsimd::max(maximumBatch, value);
        const auto delta = value - previous;
        const auto falling = delta < zero;
        sumBatch = sumBatch + xsimd::select(falling, zero - delta, delta);
        risingBatch = risingBatch + xsimd::select(falling, zero, one);
        maxDeltaBatch =
            xsimd::max(maxDeltaBatch, xsimd::select(falling, zero, delta));
      }
      if (xsimd::reduce_min(orOfValues) >= 0) {
        minimum = static_cast<uint64_t>(xsimd::reduce_min(minimumBatch));
        maximum = static_cast<uint64_t>(xsimd::reduce_max(maximumBatch));
        sumAbsDelta = static_cast<uint64_t>(xsimd::reduce_add(sumBatch));
        monotonic = static_cast<uint64_t>(xsimd::reduce_add(risingBatch));
        maxDelta = static_cast<uint64_t>(xsimd::reduce_max(maxDeltaBatch));
      } else {
        next = 1;
      }
    }
    for (size_t i = next; i < count; ++i) {
      const uint64_t previous = values[i - 1];
      const uint64_t value = values[i];
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      const bool rising = value >= previous;
      const uint64_t delta = rising ? value - previous : previous - value;
      sumAbsDelta += delta;
      monotonic += static_cast<uint64_t>(rising);
      maxDelta = std::max(maxDelta, rising ? delta : uint64_t{0});
    }
    SectionMetrics out;
    out.min = minimum;
    out.max = maximum;
    out.range = maximum - minimum;
    out.sumAbsDelta = sumAbsDelta;
    out.monotonicCount = monotonic;
    out.maxDelta = maxDelta;
    return out;
  }

  // Cumulative coverage of the top 1, 2, 4 and 8 values, from the eight
  // largest frequencies. Shared by every path so that supplying counts and
  // counting them cannot drift apart in the arithmetic.
  static void fillCoverage(
      const std::array<uint32_t, 8>& largest,
      size_t count,
      std::array<double, 4>& coverage) noexcept {
    constexpr size_t kTopKs[4] = {1, 2, 4, 8};
    const double total = static_cast<double>(count);
    uint64_t cumulative = 0;
    size_t taken = 0;
    for (size_t ki = 0; ki < 4; ++ki) {
      for (; taken < kTopKs[ki]; ++taken) {
        cumulative += largest[taken];
      }
      coverage[ki] = static_cast<double>(cumulative) / total;
    }
  }

  // The four scan metrics, for the caller that wants all of them and supplies
  // the frequencies itself. Written without the general path's five
  // loop-invariant flag tests, which stand between the body and the
  // vectorisation this hot loop needs. The accumulations reassociate safely
  // since every one is an integer operation.
  static SectionMetrics scanAll(const std::vector<uint64_t>& values) {
    const size_t count = values.size();
    const uint64_t first = values[0];

    uint64_t minimum = first;
    uint64_t maximum = first;
    // Transitions between adjacent values. The first run is added back at the
    // end, which is the same count the general path reaches by starting at one.
    uint64_t transitions = 0;
    uint64_t sumAbsDelta = 0;
    uint64_t monotonic = 0;
    uint64_t maxDelta = 0;

    // Read as values[i - 1] rather than carried in a variable, so that the
    // dependency between adjacent elements is an array access the compiler can
    // serve with a shifted load rather than a loop-carried register.
    for (size_t i = 1; i < count; ++i) {
      const uint64_t previous = values[i - 1];
      const uint64_t value = values[i];

      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      transitions += static_cast<uint64_t>(value != previous);

      const bool rising = value >= previous;
      const uint64_t delta = rising ? value - previous : previous - value;
      sumAbsDelta += delta;
      monotonic += static_cast<uint64_t>(rising);
      // Only a rising pair is a delta the encoding would pack, and taking the
      // maximum against zero on a falling one leaves it alone.
      maxDelta = std::max(maxDelta, rising ? delta : uint64_t{0});
    }

    SectionMetrics out;
    out.min = minimum;
    out.max = maximum;
    out.range = maximum - minimum;
    out.runCount = transitions + 1;
    out.avgRunLength =
        static_cast<double>(count) / static_cast<double>(out.runCount);
    out.sumAbsDelta = sumAbsDelta;
    out.monotonicCount = monotonic;
    out.maxDelta = maxDelta;

    // The histogram gets its own scalar pass rather than being fused into the
    // loop above: its indexed increment is a scatter that does not vectorise,
    // and folding it in would spill the loop's accumulators to the stack.
    for (size_t i = 0; i < count; ++i) {
      ++out.bitWidthBuckets[bitWidthBucket(values[i])];
    }
    return out;
  }

  SectionMetrics computeImpl(
      const std::vector<uint64_t>& values,
      MetricFlags flags,
      const FrequencyCounts* supplied) {
    const bool doMin = hasFlag(flags, MetricFlag::MinMax);
    const bool doRun = hasFlag(flags, MetricFlag::RunStats);
    const bool doDominant = hasFlag(flags, MetricFlag::DominantValue);
    const bool doFreqTiers = hasFlag(flags, MetricFlag::FrequencyTiers);
    // FrequencyTiers requires frequency counts, which subsumes UniqueCount.
    const bool doUniq = hasFlag(flags, MetricFlag::UniqueCount) || doFreqTiers;
    // Unique count and dominant value share a single frequency map pass, and
    // a caller supplying them spares us the pass entirely.
    const bool doFreq = (doUniq || doDominant) && supplied == nullptr;
    const bool doHist = hasFlag(flags, MetricFlag::BitWidthHistogram);
    const bool doDelta = hasFlag(flags, MetricFlag::DeltaStats);

    SectionMetrics out;
    const size_t n = values.size();
    if (n == 0) {
      return out;
    }

    // Everything the loop below would compute, with none of the per-element
    // flag tests. This is every call the selector makes, since it wants all
    // four scan metrics and supplies the frequencies from its partition.
    if (supplied != nullptr && doMin && doRun && doHist && doDelta) {
      SectionMetrics scanned = scanAll(values);
      if (doUniq) {
        scanned.uniqueCount = supplied->uniqueCount;
        // The stream cardinality estimate's only input. Dropping these reads
        // as "no singletons", which pins the estimate to the sample's count.
        scanned.singletonCount = supplied->singletonCount;
        scanned.doubletonCount = supplied->doubletonCount;
        scanned.countedRows = n;
      }
      if (doDominant) {
        scanned.dominantCount = supplied->dominantCount;
      }
      if (doFreqTiers) {
        fillCoverage(supplied->largest, n, scanned.topKCoverage);
      }
      return scanned;
    }

    // An OR across the segment bounds every value below
    // 1 << bit_width(orOfValues), tighter than the segment's nominal width,
    // so a wide bit range whose sampled values happen to be small still gets
    // counted directly instead of falling back to the hash map.
    bool useDirectHistogram = false;
    if (doFreq) {
      uint64_t orOfValues = 0;
      for (size_t i = 0; i < n; ++i) {
        orOfValues |= values[i];
      }
      // The n bound guards correctness, not memory: past kUniqueCountCap the
      // map path freezes its counts, which a direct histogram cannot
      // reproduce, but a segment that small can never exceed the cap anyway.
      useDirectHistogram = std::bit_width(orOfValues) <= kDirectHistogramBits &&
          n <= kUniqueCountCap;
      if (useDirectHistogram && counts_.empty()) {
        counts_.assign(size_t{1} << kDirectHistogramBits, 0u);
      }
    }

    const uint64_t v0 = values[0];
    if (doMin) {
      out.min = v0;
      out.max = v0;
    }
    if (doRun) {
      out.runCount = 1;
    }
    if (doHist) {
      ++out.bitWidthBuckets[bitWidthBucket(v0)];
    }

    bool capped = false;
    uint32_t maxCount = 0;
    if (doFreq) {
      if (useDirectHistogram) {
        touched_.clear();
        counts_[v0] = 1;
        touched_.push_back(static_cast<uint32_t>(v0));
      } else {
        freqMap_.clear();
        freqMap_.reserve(std::min(n, kUniqueCountCap));
        freqMap_.emplace(v0, 1u);
      }
      maxCount = 1;
    }

    uint64_t prev = v0;
    for (size_t i = 1; i < n; ++i) {
      const uint64_t v = values[i];
      if (doMin) {
        if (v < out.min) {
          out.min = v;
        }
        if (v > out.max) {
          out.max = v;
        }
      }
      if (doRun && v != prev) {
        ++out.runCount;
      }
      if (doFreq) {
        if (useDirectHistogram) {
          const uint32_t count = ++counts_[v];
          if (count == 1) {
            touched_.push_back(static_cast<uint32_t>(v));
          }
          if (count > maxCount) {
            maxCount = count;
          }
        } else if (!capped) {
          auto [it, inserted] = freqMap_.try_emplace(v, 0u);
          const uint32_t count = ++it->second;
          if (count > maxCount) {
            maxCount = count;
          }
          if (inserted && freqMap_.size() > kUniqueCountCap) {
            capped = true;
          }
        }
      }
      if (doHist) {
        ++out.bitWidthBuckets[bitWidthBucket(v)];
      }
      if (doDelta) {
        out.sumAbsDelta += (v >= prev) ? (v - prev) : (prev - v);
        if (v >= prev) {
          ++out.monotonicCount;
          out.maxDelta = std::max(out.maxDelta, v - prev);
        }
      }
      prev = v;
    }

    if (doMin) {
      out.range = out.max - out.min;
    }
    if (doRun) {
      out.avgRunLength =
          static_cast<double>(n) / static_cast<double>(out.runCount);
    }
    if (doUniq) {
      out.uniqueCount = supplied != nullptr ? supplied->uniqueCount
          : useDirectHistogram
          ? touched_.size()
          : (capped ? (kUniqueCountCap + 1) : freqMap_.size());
      out.uniqueCountCapped = capped;

      // Frequencies of one and two feed the stream cardinality estimate, and
      // are taken even when capped: the map still describes a valid prefix
      // of the segment, which carries the repetition signal the estimate
      // needs.
      if (supplied != nullptr) {
        out.singletonCount = supplied->singletonCount;
        out.doubletonCount = supplied->doubletonCount;
        out.countedRows = n;
      } else if (useDirectHistogram) {
        for (const uint32_t value : touched_) {
          const uint32_t count = counts_[value];
          out.singletonCount += (count == 1) ? 1 : 0;
          out.doubletonCount += (count == 2) ? 1 : 0;
        }
        out.countedRows = n;
      } else {
        size_t counted = 0;
        for (const auto& [value, count] : freqMap_) {
          (void)value;
          counted += count;
          out.singletonCount += (count == 1) ? 1 : 0;
          out.doubletonCount += (count == 2) ? 1 : 0;
        }
        out.countedRows = counted;
      }
    }
    if (doDominant) {
      out.dominantCount =
          supplied != nullptr ? supplied->dominantCount : maxCount;
      out.dominantCountCapped = capped;
    }
    if (doFreqTiers && !capped) {
      if (supplied != nullptr) {
        fillCoverage(supplied->largest, n, out.topKCoverage);
      } else {
        LargestFrequencies largest;
        if (useDirectHistogram) {
          for (const uint32_t value : touched_) {
            largest.offer(counts_[value]);
          }
        } else {
          for (const auto& [val, cnt] : freqMap_) {
            (void)val;
            largest.offer(cnt);
          }
        }
        fillCoverage(largest.values(), n, out.topKCoverage);
      }
    }

    // Cleared by walking what was touched, so the cost of reuse is the
    // segment's cardinality rather than the table's size.
    if (useDirectHistogram) {
      for (const uint32_t value : touched_) {
        counts_[value] = 0;
      }
    }

    return out;
  }

  // Frequency map for unique/dominant counting, used for segments whose values
  // are too wide to index directly.
  absl::flat_hash_map<uint64_t, uint32_t> freqMap_;

  // Direct-indexed counts, allocated on first use, and the values a segment
  // touched so that only those are cleared again.
  std::vector<uint32_t> counts_;
  std::vector<uint32_t> touched_;
};

} // namespace facebook::nimble::subintsplit
