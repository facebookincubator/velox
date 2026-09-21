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
#include <limits>

#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"

namespace facebook::nimble::subintsplit {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Incremental bit-range value extractor.
//
// Builds values[i] = bits [bitStart..bitEnd] of samples[i], extending one bit
// at a time so the inner loop of the cost grid reuses the work of the previous
// column instead of re-extracting from scratch.
class BitRangeExtractor {
 public:
  explicit BitRangeExtractor(const std::vector<uint64_t>& samples)
      : samples_(samples), values_(samples.size(), uint64_t{0}) {}

  void reset(int bitStart) {
    bitStart_ = bitStart;
    bitEnd_ = bitStart;
    for (size_t i = 0; i < samples_.size(); ++i) {
      values_[i] = (samples_[i] >> bitStart_) & uint64_t{1};
    }
  }

  void extend(int bitEnd) {
    for (int bit = bitEnd_ + 1; bit <= bitEnd; ++bit) {
      const uint64_t placeValue = uint64_t{1} << (bit - bitStart_);
      for (size_t i = 0; i < samples_.size(); ++i) {
        values_[i] |= ((samples_[i] >> bit) & uint64_t{1}) * placeValue;
      }
    }
    bitEnd_ = std::max(bitEnd_, bitEnd);
  }

  const std::vector<uint64_t>& values() const noexcept {
    return values_;
  }

 private:
  const std::vector<uint64_t>& samples_;
  std::vector<uint64_t> values_;
  int bitStart_{-1};
  int bitEnd_{-1};
};

// Cost of every candidate section [bitStart, bitEnd], indexed by both
// endpoints. Cells whose endpoints are not candidate boundaries stay infinite
// and the DP skips them.
class CostGrid {
 public:
  struct Cell {
    double cost{kInfinity};
    EncodingType encoding{EncodingType::Trivial};
  };

  CostGrid(int numBits, const std::vector<int>& boundaries)
      : numBits_(numBits),
        cells_(static_cast<size_t>(numBits) * numBits),
        isBoundary_(static_cast<size_t>(numBits) + 1, 0) {
    for (const int boundary : boundaries) {
      isBoundary_[static_cast<size_t>(boundary)] = 1;
    }
  }

  bool isBoundary(int bit) const noexcept {
    return isBoundary_[static_cast<size_t>(bit)] != 0;
  }

  const Cell& at(int bitStart, int bitEnd) const noexcept {
    return cells_[static_cast<size_t>(bitStart) * numBits_ + bitEnd];
  }

  // Scores every candidate section within [lo, hi]. Costs are scaled from the
  // sample up to the full stream so the DP's split penalty is in the same
  // units as the section costs it is compared against.
  void score(
      const std::vector<uint64_t>& samples,
      int lo,
      int hi,
      size_t fullCount,
      SectionCostFn costFn) {
    const MetricFlags requiredFlags = allCostModelRequiredFlags();
    const double scale =
        static_cast<double>(fullCount) / static_cast<double>(samples.size());

    MetricCollector collector;
    BitRangeExtractor extractor(samples);

    for (int bitStart = lo; bitStart <= hi; ++bitStart) {
      if (!isBoundary(bitStart)) {
        continue;
      }
      extractor.reset(bitStart);
      for (int bitEnd = bitStart; bitEnd <= hi; ++bitEnd) {
        extractor.extend(bitEnd);
        if (!isBoundary(bitEnd + 1)) {
          continue;
        }
        const int bitWidth = bitEnd - bitStart + 1;
        const SectionMetrics metrics =
            collector.compute(extractor.values(), requiredFlags, bitWidth);

        EncodingType bestEncoding = EncodingType::Trivial;
        const double sampleCost =
            costFn(metrics, samples.size(), bitWidth, bestEncoding);

        cells_[static_cast<size_t>(bitStart) * numBits_ + bitEnd] = {
            sampleCost * scale, bestEncoding};
      }
    }
  }

 private:
  int numBits_;
  std::vector<Cell> cells_;
  std::vector<uint8_t> isBoundary_;
};

// Minimum-cost partition of [lo, hi] into candidate sections.
struct DpSolution {
  // predecessor[i] is the start of the section ending just below bit i, or -1
  // when bit i was never reached.
  std::vector<int> predecessor;
  std::vector<EncodingType> encoding;
  double totalCost{kInfinity};
};

DpSolution solveDp(
    const CostGrid& grid,
    const std::vector<int>& boundaries,
    int numBits,
    int lo,
    int hi,
    const SelectorConfig& config) {
  // cost[i] is the minimum cost to cover bits [lo, i).
  std::vector<double> cost(numBits + 1, kInfinity);
  DpSolution solution{
      .predecessor = std::vector<int>(numBits + 1, -1),
      .encoding = std::vector<EncodingType>(numBits + 1, EncodingType::Trivial),
  };
  cost[lo] = 0.0;

  for (const int end : boundaries) {
    if (end <= lo) {
      continue;
    }
    for (const int start : boundaries) {
      if (start >= end) {
        break;
      }
      if (end - start < config.minSectionWidth) {
        continue;
      }
      const auto& cell = grid.at(start, end - 1);
      if (!std::isfinite(cell.cost)) {
        continue;
      }
      // The first section is free; every later one pays for its header.
      const double splitCost = (start == lo) ? 0.0 : config.splitPenalty;
      const double candidate = cost[start] + cell.cost + splitCost;
      if (candidate < cost[end]) {
        cost[end] = candidate;
        solution.predecessor[end] = start;
        solution.encoding[end] = cell.encoding;
      }
    }
  }

  solution.totalCost = cost[hi + 1];
  return solution;
}

std::vector<SectionPlan> reconstructSections(
    const CostGrid& grid,
    const DpSolution& solution,
    int lo,
    int hi) {
  std::vector<SectionPlan> sections;
  int bitEnd = hi + 1;
  while (bitEnd > lo) {
    const int bitStart = solution.predecessor[bitEnd];
    if (bitStart < 0) {
      break;
    }
    sections.push_back(
        {.bitStart = bitStart,
         .bitEnd = bitEnd - 1,
         .encoding = solution.encoding[bitEnd],
         .cost = grid.at(bitStart, bitEnd - 1).cost});
    bitEnd = bitStart;
  }
  std::reverse(sections.begin(), sections.end());
  return sections;
}

// Re-attaches the bit planes trimmed by findActiveBitRange so the returned plan
// still covers all `numBits` bits.
void padConstantEdges(
    std::vector<SectionPlan>& sections,
    int lo,
    int hi,
    int numBits) {
  if (lo > 0) {
    sections.insert(sections.begin(), makeConstantSection(0, lo - 1));
  }
  if (hi < numBits - 1) {
    sections.push_back(makeConstantSection(hi + 1, numBits - 1));
  }
}

} // namespace

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
    double threshold) {
  std::vector<int> boundaries;
  if (threshold <= 0.0) {
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

  // lo and hi+1 are the stream's own edges and are always available.
  boundaries.push_back(lo);
  for (int bit = lo + 1; bit <= hi; ++bit) {
    if (std::fabs(setRate(bit) - setRate(bit - 1)) >= threshold) {
      boundaries.push_back(bit);
    }
  }
  boundaries.push_back(hi + 1);
  return boundaries;
}

SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int numBits,
    size_t fullCount,
    const SelectorConfig& config,
    SectionCostFn costFn) {
  if (samples.empty() || numBits <= 0) {
    return {};
  }
  numBits = std::min(numBits, 64);

  const ActiveBitRange active = findActiveBitRange(samples, numBits);
  if (active.allConstant()) {
    return {
        .sections = {makeConstantSection(0, numBits - 1)}, .totalCost = 0.0};
  }

  const std::vector<int> boundaries = candidateBoundaries(
      samples, active.lo, active.hi, config.boundaryPruneThreshold);

  CostGrid grid{numBits, boundaries};
  grid.score(samples, active.lo, active.hi, fullCount, costFn);

  const DpSolution solution =
      solveDp(grid, boundaries, numBits, active.lo, active.hi, config);

  SelectorResult result;
  if (std::isfinite(solution.totalCost)) {
    result.sections = reconstructSections(grid, solution, active.lo, active.hi);
    result.totalCost = solution.totalCost;
  } else {
    // No partition of candidate boundaries was scorable; fall back to one
    // section spanning the whole active range.
    const auto& cell = grid.at(active.lo, active.hi);
    result.sections.push_back(
        {.bitStart = active.lo,
         .bitEnd = active.hi,
         .encoding = EncodingType::Trivial,
         .cost = cell.cost});
    result.totalCost = cell.cost;
  }

  padConstantEdges(result.sections, active.lo, active.hi, numBits);
  return result;
}

SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int numBits,
    size_t fullCount,
    const SelectorConfig& config) {
  return selectSplits(samples, numBits, fullCount, config, bestCostBits);
}

} // namespace facebook::nimble::subintsplit
