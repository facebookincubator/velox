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
#include <cmath>
#include <vector>

#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

// Standalone, cheap top-level policies for predicting whether a stream is
// likely to benefit from SubIntSplit, built on Statistics<T>::bitFlipProfile()
// (see BitFlipProfile.h). These are evidence-gathering components only: they
// are not wired into EncodingSizeEstimation.h or any production selection
// path. See SubIntSplitEstimator.h for how they're used to gate a real cost
// estimate, and benchmarks/ml_id_compression/MlIdSelectionPolicyBenchmark.cpp
// for how their predictions are compared against ground truth.

namespace facebook::nimble::detail::subintsplit {

struct TopLevelPolicyConfig {
  // The variance gate predicts "worth costing SubIntSplit" when
  // BitFlipProfile::variance exceeds this threshold. A stream with uniform
  // flip probability across all bit positions (e.g. uniform-random, or a
  // single homogeneous distribution) has variance close to 0; concatenated
  // bit-fields with different statistical behavior push it up.
  double varianceGateThreshold{0.01};

  // Gradient boundaries are bit positions where the discrete derivative of
  // the flip-probability curve spikes above (mean + multiplier * stddev) of
  // the gradient array itself -- an adaptive threshold, since the absolute
  // scale of the gradient varies a lot by dataset.
  double gradientStdDevMultiplier{2.0};

  // The gradient gate predicts "worth costing SubIntSplit" when at least
  // this many interior boundaries (excluding the implicit 0 and numBits
  // edges) are found.
  int minGradientBoundaries{1};

  // ... and the largest gradient value in the profile reaches at least this
  // absolute magnitude. The adaptive threshold above is relative to each
  // column's own gradient noise floor, so a nearly flat profile (uniform or
  // constant-like) can still produce a handful of "boundaries" that exceed
  // its own tiny mean + multiplier * stddev without any of them being a
  // meaningful spike. This floor rejects that case.
  double minGradientMagnitude{0.005};
};

/// Predicts whether `profile` indicates a stream heterogeneous enough to be
/// worth costing SubIntSplit against its rivals.
inline bool bitFlipVarianceGate(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  return profile.variance > config.varianceGateThreshold;
}

/// Returns candidate split points derived from spikes in `profile`'s
/// gradient: a split point `s` means "a segment may start or end at bit
/// index `s`" (so a segment's bit range is [boundaries[i], boundaries[i+1] -
/// 1]). Sorted, deduped, always including 0 and `profile.numBits` (the
/// implicit outer edges of the full bit range) -- empty `profile.numBits`
/// yields just those two edges. Currently consumed only by
/// bitFlipGradientGate() below; using it to constrain
/// SubIntSplitSelector.h's DP grid to a smaller candidate set is a possible
/// follow-up, not implemented here.
inline std::vector<int> bitFlipGradientBoundaries(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  const int kBits = profile.numBits;
  std::vector<int> boundaries;
  if (kBits <= 0) {
    return boundaries;
  }

  double sum = 0.0;
  for (int b = 1; b < kBits; ++b) {
    sum += profile.gradient[b];
  }
  const int gradientCount = kBits - 1;
  const double mean = gradientCount > 0 ? sum / gradientCount : 0.0;

  double sqDiffSum = 0.0;
  for (int b = 1; b < kBits; ++b) {
    const double diff = profile.gradient[b] - mean;
    sqDiffSum += diff * diff;
  }
  const double stddev =
      gradientCount > 0 ? std::sqrt(sqDiffSum / gradientCount) : 0.0;
  const double threshold = mean + config.gradientStdDevMultiplier * stddev;

  boundaries.push_back(0);
  for (int b = 1; b < kBits; ++b) {
    if (profile.gradient[b] > threshold) {
      boundaries.push_back(b);
    }
  }
  boundaries.push_back(kBits);

  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(
      std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
  return boundaries;
}

/// Predicts whether `profile` indicates a stream worth costing SubIntSplit
/// against its rivals, using the gradient-boundary signal instead of
/// variance: requires at least `config.minGradientBoundaries` interior
/// boundaries (see bitFlipGradientBoundaries()) whose largest gradient value
/// also reaches `config.minGradientMagnitude`. An alternative to
/// bitFlipVarianceGate() built on the same profile, not a second stage
/// applied after it.
inline bool bitFlipGradientGate(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  const auto boundaries = bitFlipGradientBoundaries(profile, config);
  const int interior =
      boundaries.size() >= 2 ? static_cast<int>(boundaries.size()) - 2 : 0;
  if (interior < config.minGradientBoundaries || profile.numBits <= 0) {
    return false;
  }
  const double maxGradient = *std::max_element(
      profile.gradient.begin(), profile.gradient.begin() + profile.numBits);
  return maxGradient >= config.minGradientMagnitude;
}

} // namespace facebook::nimble::detail::subintsplit
