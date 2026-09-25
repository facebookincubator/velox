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

#include <span>
#include <vector>

#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"

namespace facebook::nimble::subintsplit {

/// Number of cheapest split-DP plans the hybrid planner re-prices.
inline constexpr uint32_t kHybridShortlist{8};

/// Sample size used to re-price shortlisted ranges; only those ranges are
/// priced at this size, keeping a larger sample affordable.
inline constexpr uint32_t kHybridRescoreSamples{16'384};

/// A split plan chosen by the hybrid planner, with the re-priced totals it
/// was chosen on, both in bits for the full stream.
struct RefinedPlan {
  std::vector<SectionPlan> sections;
  /// Size plus split penalties plus decode cost at the requested weight.
  double weightedBits{0.0};
  /// Estimated size alone.
  double sizeBits{0.0};
};

/// The hybrid split planner's second stage: re-prices shortlisted plans with
/// the estimators section selection uses, on a larger sample than the split
/// DP costs with, then refines the cheapest by moving, merging and splitting
/// boundaries under the same pricing. Pricing only the ranges a shortlist and
/// its refinement touch keeps the accurate estimators affordable.
///
/// Defined in PlanRefiner.cpp for uint32_t and uint64_t only, since the
/// estimators it calls live behind SubIntSplitEncoding.h and are unreachable
/// from any header on SubIntSplit's own include path.
class SubIntSplitPlanRefiner {
 public:
  /// Returns the cheapest plan found, or an empty plan if none could be
  /// priced. `cuts`, when non-empty, restricts refinement's split points to
  /// those positions. Decode weighting and
  /// Options::subIntSplitMaxSizeRegression apply as in the DP: if the
  /// weighted plan's size exceeds the size-only plan's by more than the cap,
  /// the size-only plan is returned instead.
  template <typename PhysicalType>
  static RefinedPlan refine(
      std::span<const PhysicalType> values,
      int kBits,
      const std::vector<std::vector<SectionPlan>>& shortlist,
      const std::vector<bool>& cuts,
      const SelectorConfig& selectorConfig,
      const Encoding::Options& options);
};

} // namespace facebook::nimble::subintsplit
