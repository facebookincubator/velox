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
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"

// DP-based bit-range split selector for SubIntSplitEncoding.
// Evaluates a grid of bit ranges [l..r] on a sample of uint64_t values,
// runs dynamic programming over bit positions 0..kBits to find the minimum-cost
// partition, and returns a list of SegmentPlan entries.

namespace facebook::nimble::detail::subintsplit {

struct SegmentPlan {
  int bitStart{0};
  int bitEnd{0};
  EncodingType encoding{EncodingType::Trivial};
  double cost{0.0}; // estimated total bits for the full stream
};

struct SelectorConfig {
  int minSegmentWidth{1};
  double splitPenalty{10.0}; // extra bits charged per additional split boundary
};

inline SelectorConfig defaultSelectorConfig() noexcept {
  return SelectorConfig{.minSegmentWidth = 1, .splitPenalty = 10.0};
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

struct SelectorResult {
  std::vector<SegmentPlan> segments;
  double totalCost{0.0};
};

// A constant bit-plane run, stored as a single Constant section (costs
// ~nothing to encode or decode).
inline SegmentPlan makeConstantSegment(int bitStart, int bitEnd) {
  return {
      .bitStart = bitStart,
      .bitEnd = bitEnd,
      .encoding = EncodingType::Constant,
      .cost = 0.0};
}

// The contiguous range of bit positions that actually vary across the sample.
// `allConstant()` means every sampled value is identical.
struct ActiveBitRange {
  int lo{0};
  int hi{-1};

  bool allConstant() const noexcept {
    return hi < lo;
  }
};

// Bit-plane pre-pass: find the lowest and highest bit that is not identical
// across every sample. Bits outside [lo, hi] -- a constant high prefix and/or
// low suffix, the common case for narrow, low-cardinality and bit-structured
// data -- carry no information, so they become free Constant sections and the
// O(width^2) cost grid and DP only run over the active range.
inline ActiveBitRange findActiveBitRange(
    const std::vector<uint64_t>& samples,
    int kBits) {
  uint64_t orAll = 0;
  uint64_t andAll = ~uint64_t{0};
  for (const uint64_t s : samples) {
    orAll |= s;
    andAll &= s;
  }
  const uint64_t bitsMask =
      (kBits >= 64) ? ~uint64_t{0} : ((uint64_t{1} << kBits) - 1);
  const uint64_t varying = (orAll & ~andAll) & bitsMask;
  if (varying == 0) {
    return {}; // allConstant()
  }
  return {
      .lo = std::countr_zero(varying), .hi = 63 - std::countl_zero(varying)};
}

// Run the DP split selector on `samples` (uint64_t values drawn from a
// physical-type stream of `kBits` width).
//
// `fullCount` is the total element count of the *full* stream; cost model
// scores are scaled from the sample size to the full stream so the DP
// produces estimates in the right units.
//
// `costFn` scores a single segment: given (metrics, numValues, bitWidth,
// outBestEncoding), return the per-sample cost in bits and write the best
// encoding into the output parameter.  Defaults to `bestCostBits`.
template <typename CostFn>
inline SelectorResult selectSplitsImpl(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg,
    CostFn&& costFn) {
  if (samples.empty() || kBits <= 0) {
    return {};
  }
  kBits = std::min(kBits, 64);

  const int sz = kBits;

  // Trim constant bit planes before doing any work. They cannot affect the
  // chosen split -- a plane that never varies is free either way -- so the
  // grid and the DP below only run over the active range, which shrinks the
  // O(width^2) grid quadratically on narrow and low-cardinality data.
  const ActiveBitRange active = findActiveBitRange(samples, kBits);
  if (active.allConstant()) {
    SelectorResult constResult;
    constResult.segments.push_back(makeConstantSegment(0, sz - 1));
    constResult.totalCost = 0.0;
    return constResult;
  }
  const int lo = active.lo;
  const int hi = active.hi;

  const MetricFlags requiredFlags = allCostModelRequiredFlags();
  MetricCollector collector;

  struct SegmentChoice {
    double cost{std::numeric_limits<double>::infinity()};
    EncodingType encoding{EncodingType::Trivial};
  };

  std::vector<SegmentChoice> bestCost(sz * sz);

  BitRangeExtractor extractor(samples);
  const size_t numSamples = samples.size();

  for (int l = lo; l <= hi; ++l) {
    extractor.reset(l);
    for (int r = l; r <= hi; ++r) {
      extractor.extend(r);
      const std::vector<uint64_t>& segValues = extractor.values();
      const SegmentMetrics metrics =
          collector.compute(segValues, requiredFlags);
      const int bitWidth = r - l + 1;

      EncodingType bestEnc = EncodingType::Trivial;
      const double perSampleCost =
          costFn(metrics, numSamples, bitWidth, bestEnc);

      const double fullCost = perSampleCost * static_cast<double>(fullCount) /
          static_cast<double>(numSamples);

      bestCost[l * sz + r] = {fullCost, bestEnc};
    }
  }

  // dp is indexed in absolute bit positions but only spans the active range:
  // dp[i] = minimum cost to cover bits [lo, i).
  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  std::vector<EncodingType> chosen(sz + 1, EncodingType::Trivial);
  dp[lo] = 0.0;

  for (int i = lo + 1; i <= hi + 1; ++i) {
    for (int j = lo; j < i; ++j) {
      const int width = i - j;
      if (width < cfg.minSegmentWidth) {
        continue;
      }
      const auto& choice = bestCost[j * sz + (i - 1)];
      if (!std::isfinite(choice.cost)) {
        continue;
      }
      const double splitCost = (j == lo) ? 0.0 : cfg.splitPenalty;
      const double candidate = dp[j] + choice.cost + splitCost;
      if (candidate < dp[i]) {
        dp[i] = candidate;
        prev[i] = j;
        chosen[i] = choice.encoding;
      }
    }
  }

  SelectorResult result;
  result.totalCost = dp[hi + 1];

  if (!std::isfinite(result.totalCost)) {
    SegmentPlan fallback;
    fallback.bitStart = lo;
    fallback.bitEnd = hi;
    fallback.encoding = EncodingType::Trivial;
    fallback.cost = bestCost[lo * sz + hi].cost;
    result.segments.push_back(fallback);
    result.totalCost = fallback.cost;
  } else {
    int idx = hi + 1;
    while (idx > lo) {
      const int start = prev[idx];
      if (start < 0) {
        break;
      }
      SegmentPlan plan;
      plan.bitStart = start;
      plan.bitEnd = idx - 1;
      plan.encoding = chosen[idx];
      plan.cost = bestCost[start * sz + (idx - 1)].cost;
      result.segments.push_back(plan);
      idx = start;
    }
    std::reverse(result.segments.begin(), result.segments.end());
  }

  // Re-attach the trimmed planes so the returned plan still covers all kBits.
  if (lo > 0) {
    result.segments.insert(
        result.segments.begin(), makeConstantSegment(0, lo - 1));
  }
  if (hi < sz - 1) {
    result.segments.push_back(makeConstantSegment(hi + 1, sz - 1));
  }
  return result;
}

inline SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  return selectSplitsImpl(samples, kBits, fullCount, cfg, bestCostBits);
}

} // namespace facebook::nimble::detail::subintsplit
