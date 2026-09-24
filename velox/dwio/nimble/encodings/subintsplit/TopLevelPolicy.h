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
#include <cstdint>
#include <span>
#include <vector>

#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

// Standalone, cheap top-level policies for predicting whether a stream is
// likely to benefit from SubIntSplit, built on Statistics<T>::bitFlipProfile()
// (see BitFlipProfile.h). The gradient gate, optionally with the active-bit
// entropy guard, is what ManualEncodingSelectionPolicy::select() admits
// SubIntSplit by when Encoding::Options::subIntSplitAdmission asks for it;
// the default admission is still SubIntSplitEncoding::estimateSize.

namespace facebook::nimble::subintsplit {

struct TopLevelPolicyConfig {
  // The variance gate predicts "worth costing SubIntSplit" when
  // BitFlipProfile::variance exceeds this threshold: a stream with uniform
  // flip probability has variance close to 0, while concatenated bit-fields
  // with different statistical behavior push it up.
  double varianceGateThreshold{0.01};

  // Gradient boundaries are bit positions where the discrete derivative of
  // the flip-probability curve spikes above an adaptive threshold (mean +
  // multiplier * stddev of the gradient array itself), since the gradient's
  // absolute scale varies a lot by column.
  double gradientStdDevMultiplier{2.0};

  // The gradient gate predicts "worth costing SubIntSplit" when at least
  // this many interior boundaries (excluding the implicit 0 and numBits
  // edges) are found.
  int minGradientBoundaries{1};

  // The gradient gate also requires the largest gradient value in the
  // profile to reach at least this absolute magnitude, since the adaptive
  // threshold above is relative to each column's own noise floor and a
  // nearly flat profile can otherwise still produce spurious "boundaries".
  double minGradientMagnitude{0.005};

  // The entropy guard rejects a stream whose non-constant bits flip, on
  // average, nearly as unpredictably as random bits: mean binary entropy of
  // flipProbability over the bits that ever flip above this. Such a stream
  // has nothing left for a split to exploit once its constant bits are
  // dropped, which FixedBitWidth already does. Varying bits come from the
  // whole stream (BitFlipProfile::varyingBits), so a sampled profile does not
  // drop slow fields and inflate the mean. Since admission only decides
  // candidacy rather than the encoding, a false positive here only costs a
  // size estimate, so the threshold is set loose; this guard exists mainly
  // for the ablation that justifies preferring kBitFlip over kBitFlipEntropy.
  double maxActiveFlipEntropy{0.99};
};

/// How top-level selection decides whether SubIntSplit is tried. Under the
/// bit-flip modes an admitted stream is offered SubIntSplit as a candidate and
/// still has to win the ordinary size comparison; only
/// Encoding::Options::subIntSplitAdmissionForces lets the gate decide alone.
enum class SubIntSplitAdmission : uint8_t {
  /// SubIntSplitEncoding::estimateSize competes with the other candidates on
  /// its read-factor-weighted size.
  kEstimate = 0,
  /// bitFlipGradientGate() alone decides whether SubIntSplit is a candidate.
  kBitFlip = 1,
  /// bitFlipGradientGate() and the active-bit entropy guard must both admit
  /// for SubIntSplit to be a candidate.
  kBitFlipEntropy = 2,
};

/// Returns the mean binary entropy, in bits, of the flip probabilities of
/// the bit positions in `profile.varyingBits`; 0 when there are none.
inline double activeBitFlipEntropy(const BitFlipProfile& profile) {
  double entropySum{0.0};
  int numActiveBits{0};
  for (int b = 0; b < profile.numBits; ++b) {
    if (((profile.varyingBits >> b) & 1) == 0) {
      continue;
    }
    ++numActiveBits;
    const double probability = profile.flipProbability[b];
    if (probability > 0.0 && probability < 1.0) {
      entropySum -= probability * std::log2(probability) +
          (1.0 - probability) * std::log2(1.0 - probability);
    }
  }
  return numActiveBits == 0 ? 0.0 : entropySum / numActiveBits;
}

// Predicts whether `profile` indicates a stream heterogeneous enough to be
// worth costing SubIntSplit against its rivals.
inline bool bitFlipVarianceGate(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  return profile.variance > config.varianceGateThreshold;
}

// Returns candidate split points derived from spikes in `profile`'s
// gradient, sorted and deduped, always including 0 and `profile.numBits` as
// the outer edges of the full bit range.
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

// Predicts whether `profile` is worth costing SubIntSplit against its
// rivals, using the gradient-boundary signal instead of variance.
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

/// Returns which bit-flip work `admission` needs done over the whole stream.
/// Only the entropy guard reads BitFlipProfile::varyingBits, and filling it
/// reads every value; the gradient gate reads the sampled pairs alone, so
/// admitting by it costs a fixed number of pairs whatever the column's length.
inline BitFlipVaryingBits bitFlipVaryingBitsFor(
    SubIntSplitAdmission admission) {
  return admission == SubIntSplitAdmission::kBitFlipEntropy
      ? BitFlipVaryingBits::kWholeStream
      : BitFlipVaryingBits::kSkip;
}

/// Returns the profile `admission` decides from, taken over at most `maxPairs`
/// adjacent pairs of `values` (every pair when `maxPairs` is 0).
template <typename T>
BitFlipProfile bitFlipAdmissionProfile(
    std::span<const T> values,
    SubIntSplitAdmission admission,
    size_t maxPairs) {
  return computeBitFlipProfile<T>(
      values, maxPairs, bitFlipVaryingBitsFor(admission));
}

/// Returns whether `admission` admits SubIntSplit for a stream with
/// `profile`. kEstimate is not a profile decision and always returns true.
inline bool bitFlipAdmits(
    const BitFlipProfile& profile,
    SubIntSplitAdmission admission,
    const TopLevelPolicyConfig& config) {
  switch (admission) {
    case SubIntSplitAdmission::kEstimate:
      return true;
    case SubIntSplitAdmission::kBitFlip:
      return bitFlipGradientGate(profile, config);
    case SubIntSplitAdmission::kBitFlipEntropy:
      return bitFlipGradientGate(profile, config) &&
          activeBitFlipEntropy(profile) <= config.maxActiveFlipEntropy;
  }
  return false;
}

} // namespace facebook::nimble::subintsplit
