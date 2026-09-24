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
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace facebook::nimble::subintsplit {

SectionPlan makeConstantSection(int bitStart, int bitEnd) {
  return {
      .bitStart = bitStart,
      .bitEnd = bitEnd,
      .encoding = EncodingType::Constant,
      .cost = 0.0};
}

ActiveBitRange findActiveBitRange(
    const std::vector<uint64_t>& samples,
    int numBits) {
  uint64_t orAll = 0;
  uint64_t andAll = ~uint64_t{0};
  for (const uint64_t sample : samples) {
    orAll |= sample;
    andAll &= sample;
  }

  const uint64_t bitsMask =
      (numBits >= 64) ? ~uint64_t{0} : ((uint64_t{1} << numBits) - 1);
  const uint64_t varying = (orAll & ~andAll) & bitsMask;
  if (varying == 0) {
    return {};
  }
  return {
      .lo = std::countr_zero(varying), .hi = 63 - std::countl_zero(varying)};
}

std::vector<int> candidateBoundaries(
    const std::vector<uint64_t>& samples,
    int lo,
    int hi,
    double threshold,
    size_t maxCount) {
  std::vector<int> boundaries;
  if (threshold <= 0.0 && maxCount == 0) {
    for (int bit = lo; bit <= hi + 1; ++bit) {
      boundaries.push_back(bit);
    }
    return boundaries;
  }

  std::vector<uint32_t> setCount(static_cast<size_t>(hi - lo + 1), 0);
  for (const uint64_t sample : samples) {
    uint64_t bits = sample >> lo;
    for (int bit = 0; bit <= hi - lo; ++bit) {
      setCount[static_cast<size_t>(bit)] += static_cast<uint32_t>(bits & 1ULL);
      bits >>= 1;
    }
  }

  const auto setRate = [&](int bit) {
    return static_cast<double>(setCount[static_cast<size_t>(bit - lo)]) /
        samples.size();
  };

  // Interior positions that clear the threshold, with the size of the jump so
  // they can be ranked if there are too many.
  std::vector<std::pair<double, int>> interior;
  for (int bit = lo + 1; bit <= hi; ++bit) {
    const double change = std::fabs(setRate(bit) - setRate(bit - 1));
    if (change >= threshold) {
      interior.emplace_back(change, bit);
    }
  }

  // Keep the sharpest jumps: a bigger set-rate step is a more likely field
  // edge, so dropping the smallest first loses the least real structure.
  if (maxCount > 0 && interior.size() > maxCount) {
    std::partial_sort(
        interior.begin(),
        interior.begin() + maxCount,
        interior.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
    interior.resize(maxCount);
  }
  std::sort(
      interior.begin(), interior.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
      });

  // lo and hi+1 are the stream's own edges and are always available.
  boundaries.push_back(lo);
  for (const auto& [change, bit] : interior) {
    boundaries.push_back(bit);
  }
  boundaries.push_back(hi + 1);
  return boundaries;
}

GridLayout makeGridLayout(
    const std::vector<uint64_t>& samples,
    int sz,
    const SelectorConfig& cfg,
    bool constantAllowed) {
  GridLayout layout;
  const bool trim = cfg.trimConstantPlanes && constantAllowed;
  if (!trim && cfg.boundaryPruneThreshold <= 0.0 &&
      cfg.maxCandidateBoundaries == 0 && cfg.maxSectionWidth <= 0 &&
      cfg.frequencyMetricsMaxWidth <= 0) {
    return layout;
  }
  layout.unrestricted = false;
  layout.lo = 0;
  layout.hi = sz - 1;
  if (trim && !samples.empty()) {
    const ActiveBitRange active = findActiveBitRange(samples, sz);
    if (active.allConstant()) {
      layout.allConstant = true;
      return layout;
    }
    // An edge narrower than the minimum section width cannot be a section of
    // its own, so it stays in the varying range rather than being trimmed.
    if (active.lo >= cfg.minSectionWidth) {
      layout.lo = active.lo;
    }
    if (sz - 1 - active.hi >= cfg.minSectionWidth) {
      layout.hi = active.hi;
    }
  }
  layout.isBoundary.assign(static_cast<size_t>(sz) + 1, 0);
  layout.isBoundary[0] = 1;
  layout.isBoundary[static_cast<size_t>(sz)] = 1;
  for (const int boundary : candidateBoundaries(
           samples,
           layout.lo,
           layout.hi,
           cfg.boundaryPruneThreshold,
           cfg.maxCandidateBoundaries)) {
    layout.isBoundary[static_cast<size_t>(boundary)] = 1;
  }
  return layout;
}

} // namespace facebook::nimble::subintsplit
