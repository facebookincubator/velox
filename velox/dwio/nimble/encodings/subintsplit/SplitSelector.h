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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "folly/Function.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"

// DP-based bit-range split selector for SubIntSplitEncoding.
//
// Scores a grid of bit ranges on a sample of uint64_t values, runs dynamic
// programming over bit positions to find the minimum-cost partition, and
// returns the sections that partition implies.

namespace facebook::nimble::subintsplit {

/// Relative set-rate change below which a bit position is not worth
/// considering as a split boundary.
///
/// Chosen empirically: a sweep over production corpora put every threshold at
/// or above 0.02 into a size regression on some stream (id_list_features paid
/// +1.3% for 1.09x planning), while 0.001 left the encoded output
/// byte-identical on every corpus measured and still cut planning time. It only
/// discards positions whose two adjacent bit planes are set at within 0.1% the
/// same rate, which no real field edge is.
constexpr double kBoundaryPruneThreshold = 0.001;

struct SelectorConfig {
  int minSectionWidth{1};

  /// Extra bits charged per additional split boundary, so the DP does not
  /// shred a stream into sections whose headers cost more than they save.
  double splitPenalty{10.0};

  /// Relative change in a bit plane's set-rate required for the position to be
  /// considered as a split boundary. 0.0 considers every position.
  double boundaryPruneThreshold{kBoundaryPruneThreshold};
};

inline SelectorConfig defaultSelectorConfig() noexcept {
  return SelectorConfig{};
}

struct SelectorResult {
  /// Sections covering all `numBits` bits, in LSB-first order.
  std::vector<SectionPlan> sections;

  /// Estimated total bits for the chosen partition across the full stream.
  double totalCost{0.0};
};

/// Scores one candidate section: given its metrics, the value count and its bit
/// width, returns the cost in bits and writes the best encoding to the output
/// parameter.
using SectionCostFn = folly::FunctionRef<
    double(const SectionMetrics&, size_t, int, EncodingType&)>;

/// The contiguous range of bit positions that actually vary across the sample.
struct ActiveBitRange {
  int lo{0};
  int hi{-1};

  bool allConstant() const noexcept {
    return hi < lo;
  }
};

/// Finds the lowest and highest bit that is not identical across every sample.
///
/// Bits outside the result -- a constant high prefix and/or low suffix, the
/// common case for narrow, low-cardinality and bit-structured data -- carry no
/// information, so they become free Constant sections and the O(width^2) cost
/// grid and DP only run over the active range.
ActiveBitRange findActiveBitRange(
    const std::vector<uint64_t>& samples,
    int numBits);

/// Bit positions where a split is worth considering.
///
/// The DP is O(width^2) in candidate boundaries and pays a metrics pass for
/// each cell, so the boundary set -- not the bit width -- is what actually
/// drives planning cost. Adjacent bit planes belonging to the same packed field
/// have near-identical set-rates across the sample; a field edge is where that
/// rate jumps. Keeping only the jumps leaves the boundaries a real layout has,
/// for one O(numSamples * width) popcount pass against O(width^2) metrics
/// passes saved.
std::vector<int> candidateBoundaries(
    const std::vector<uint64_t>& samples,
    int lo,
    int hi,
    double threshold);

/// A constant bit-plane run, stored as a single Constant section (costs
/// ~nothing to encode or decode).
SectionPlan makeConstantSection(int bitStart, int bitEnd);

/// Runs the DP split selector on `samples`, bit patterns drawn from a
/// physical-type stream `numBits` wide.
///
/// `fullCount` is the element count of the *full* stream; cost model scores are
/// scaled from the sample size up to it so the DP produces estimates in the
/// right units.
SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int numBits,
    size_t fullCount,
    const SelectorConfig& config,
    SectionCostFn costFn);

/// Runs the selector with the standard cost models.
SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int numBits,
    size_t fullCount,
    const SelectorConfig& config = defaultSelectorConfig());

} // namespace facebook::nimble::subintsplit
