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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

// Bit-flip-probability statistics for integral value streams. Shared between
// Statistics<T> (outer, per-column selection) and SubIntSplit's standalone
// evidence-gathering estimator, so the per-bit XOR-and-popcount pass is
// implemented exactly once.

namespace facebook::nimble {

/// Per-bit-position flip-probability profile of an integral value stream.
/// `flipProbability[i]` is P(bit i differs between two consecutive sampled
/// values), estimated by XOR-count-and-divide. `variance` is the variance of
/// `flipProbability` across `numBits` positions; `gradient[i]` is
/// |flipProbability[i] - flipProbability[i - 1]| (gradient[0] == 0). Only
/// the first `numBits` entries of each array are meaningful; the remainder
/// are zero-filled.
struct BitFlipProfile {
  std::array<double, 64> flipProbability{};
  double variance{0.0};
  std::array<double, 64> gradient{};
  int numBits{0};
};

/// Computes the bit-flip profile of `values` by XORing each consecutive pair
/// and accumulating a per-bit set-count via shift-and-mask, so the whole pass
/// is O(n * numBits), not O(n * numBits^2). Returns a zero profile when
/// `values` has fewer than two elements.
template <typename T>
BitFlipProfile computeBitFlipProfile(std::span<const T> values) {
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr int kBits = std::numeric_limits<UnsignedT>::digits;
  static_assert(kBits <= 64);

  BitFlipProfile profile;
  profile.numBits = kBits;
  if (values.size() < 2) {
    return profile;
  }

  std::array<uint64_t, 64> flipCounts{};
  const size_t pairCount = values.size() - 1;
  for (size_t i = 0; i + 1 < values.size(); ++i) {
    const UnsignedT xorVal =
        static_cast<UnsignedT>(values[i]) ^ static_cast<UnsignedT>(values[i + 1]);
    for (int b = 0; b < kBits; ++b) {
      flipCounts[b] += (xorVal >> b) & UnsignedT{1};
    }
  }

  double sum = 0.0;
  for (int b = 0; b < kBits; ++b) {
    profile.flipProbability[b] =
        static_cast<double>(flipCounts[b]) / static_cast<double>(pairCount);
    sum += profile.flipProbability[b];
  }

  const double mean = sum / static_cast<double>(kBits);
  double sqDiffSum = 0.0;
  for (int b = 0; b < kBits; ++b) {
    const double diff = profile.flipProbability[b] - mean;
    sqDiffSum += diff * diff;
  }
  profile.variance = sqDiffSum / static_cast<double>(kBits);

  for (int b = 1; b < kBits; ++b) {
    profile.gradient[b] =
        std::abs(profile.flipProbability[b] - profile.flipProbability[b - 1]);
  }

  return profile;
}

} // namespace facebook::nimble
