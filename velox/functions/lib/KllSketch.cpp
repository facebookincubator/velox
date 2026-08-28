/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <cmath>
#include <limits>

#include "velox/functions/lib/KllSketch.h"

namespace facebook::velox::functions::kll {

uint32_t kFromEpsilon(double epsilon) {
  VELOX_USER_CHECK(
      std::isfinite(epsilon) && epsilon > 0,
      "Accuracy must be positive and finite: {}",
      epsilon);
  const auto k = std::ceil(std::exp(1.0285 * std::log(2.296 / epsilon)));
  VELOX_USER_CHECK(
      std::isfinite(k) && k <= kMaxK,
      "Accuracy is too small: {} produces K {}, but maximum K is {}",
      epsilon,
      k,
      kMaxK);
  return static_cast<uint32_t>(k);
}

namespace detail {

namespace {

constexpr uint8_t kMinBufferWidth = 8;

double powerOfTwoThirds(int n) {
  static const auto kMemo = [] {
    std::array<double, kMaxLevel> memo;
    for (int i = 0; i < kMaxLevel; ++i) {
      memo[i] = pow(2.0 / 3.0, i);
    }
    return memo;
  }();
  return kMemo[n];
}

} // namespace

uint32_t computeTotalCapacity(uint32_t k, uint8_t numLevels) {
  uint64_t total = 0;
  for (uint8_t h = 0; h < numLevels; ++h) {
    total += levelCapacity(k, numLevels, h);
    VELOX_CHECK_LE(
        total,
        std::numeric_limits<uint32_t>::max(),
        "KLL total capacity exceeds uint32_t: k={}, numLevels={}, total={}",
        k,
        numLevels,
        total);
  }
  return static_cast<uint32_t>(total);
}

uint32_t levelCapacity(uint32_t k, uint8_t numLevels, uint8_t height) {
  VELOX_DCHECK_LT(height, numLevels);
  VELOX_DCHECK_LE(numLevels, kMaxLevel);
  return std::max<uint32_t>(
      kMinBufferWidth, k * powerOfTwoThirds(numLevels - height - 1));
}

uint8_t floorLog2(uint64_t p, uint64_t q) {
  for (uint8_t ans = 0;; ++ans) {
    q <<= 1;
    if (p < q) {
      return ans;
    }
  }
}

uint64_t sumSampleWeights(uint8_t numLevels, const uint32_t* levels) {
  uint64_t total = 0;
  uint64_t weight = 1;
  for (uint8_t lvl = 0; lvl < numLevels; lvl++) {
    total += weight * (levels[lvl + 1] - levels[lvl]);
    weight *= 2;
  }
  return total;
}

} // namespace detail
} // namespace facebook::velox::functions::kll
