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

// Run the DP split selector on `samples` (uint64_t values drawn from a
// physical-type stream of `kBits` width).
//
// `fullCount` is the total element count of the *full* stream; cost model
// scores are scaled from the sample size to the full stream so the DP
// produces estimates in the right units.
inline SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int kBits, // number of bits in the physical type (32 or 64)
    size_t fullCount,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  if (samples.empty() || kBits <= 0) {
    return {};
  }
  kBits = std::min(kBits, 64);

  const MetricFlags requiredFlags = allCostModelRequiredFlags();
  MetricCollector collector;

  // bestCost[l][r] = {min cost in bits for full stream, best EncodingType}
  // Only lower-triangular (r >= l) entries are valid.
  struct SegmentChoice {
    double cost{std::numeric_limits<double>::infinity()};
    EncodingType encoding{EncodingType::Trivial};
  };

  // Use flat vector for cache friendliness (kBits can be 32 or 64)
  const int sz = kBits;
  std::vector<SegmentChoice> bestCost(sz * sz);

  BitRangeExtractor extractor(samples);
  const size_t numSamples = samples.size();

  for (int l = 0; l < sz; ++l) {
    extractor.reset(l);
    for (int r = l; r < sz; ++r) {
      extractor.extend(r);
      const std::vector<uint64_t>& segValues = extractor.values();
      const SegmentMetrics metrics =
          collector.compute(segValues, requiredFlags);
      const int bitWidth = r - l + 1;

      EncodingType bestEnc = EncodingType::Trivial;
      const double perSampleCost =
          bestCostBits(metrics, numSamples, bitWidth, bestEnc);

      // Scale to full stream
      const double fullCost = perSampleCost * static_cast<double>(fullCount) /
          static_cast<double>(numSamples);

      bestCost[l * sz + r] = {fullCost, bestEnc};
    }
  }

  // DP over bit positions [0..kBits].
  // dp[i] = minimum cost to cover bits [0..i).
  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  std::vector<EncodingType> chosen(sz + 1, EncodingType::Trivial);
  dp[0] = 0.0;

  for (int i = 1; i <= sz; ++i) {
    for (int j = 0; j < i; ++j) {
      const int width = i - j;
      if (width < cfg.minSegmentWidth) {
        continue;
      }
      const auto& choice = bestCost[j * sz + (i - 1)];
      if (!std::isfinite(choice.cost)) {
        continue;
      }
      const double splitCost = (j == 0) ? 0.0 : cfg.splitPenalty;
      const double candidate = dp[j] + choice.cost + splitCost;
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
    // Fallback: single segment covering all bits, Trivial encoding
    SegmentPlan fallback;
    fallback.bitStart = 0;
    fallback.bitEnd = sz - 1;
    fallback.encoding = EncodingType::Trivial;
    fallback.cost = bestCost[0 * sz + (sz - 1)].cost;
    result.segments.push_back(fallback);
    result.totalCost = fallback.cost;
    return result;
  }

  // Backtrack to reconstruct segment plan
  int idx = sz;
  while (idx > 0) {
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
  return result;
}

} // namespace facebook::nimble::detail::subintsplit
