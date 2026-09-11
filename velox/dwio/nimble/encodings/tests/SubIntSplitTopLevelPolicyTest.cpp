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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/SubIntSplitEstimator.h"
#include "velox/dwio/nimble/encodings/SubIntSplitTopLevelPolicy.h"

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::detail::subintsplit;

namespace {

// Fixed so the synthetic streams are reproducible across runs.
constexpr uint64_t kSeed = 20260901;

std::vector<uint64_t> makeUniformRandomStream(size_t n) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(n);
  for (auto& v : values) {
    v = rng();
  }
  return values;
}

// Bits [0,9]: constant 0. Bits [10,29]: fully random per row. Bits [30,63]:
// constant 0. Two sharp bit-flip-probability boundaries, at bit 10 and bit
// 30, with a clean 0 -> ~0.5 -> 0 shape -- the cleanest possible synthetic
// analogue of Snowflake's concatenated timestamp|machine|sequence fields.
std::vector<uint64_t> makeConcatenatedFieldsStream(size_t n) {
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint64_t> midField(0, (uint64_t{1} << 20) - 1);
  std::vector<uint64_t> values(n);
  for (auto& v : values) {
    v = midField(rng) << 10;
  }
  return values;
}

} // namespace

TEST(SubIntSplitTopLevelPolicyTest, varianceGateRejectsUniformRandom) {
  const auto values = makeUniformRandomStream(10'000);
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));
  EXPECT_EQ(profile.numBits, 64);
  EXPECT_LT(profile.variance, 0.005);

  const TopLevelPolicyConfig config;
  EXPECT_FALSE(bitFlipVarianceGate(profile, config));
}

TEST(SubIntSplitTopLevelPolicyTest, varianceGateAcceptsConcatenatedFields) {
  const auto values = makeConcatenatedFieldsStream(10'000);
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));
  EXPECT_GT(profile.variance, 0.03);

  const TopLevelPolicyConfig config;
  EXPECT_TRUE(bitFlipVarianceGate(profile, config));
}

TEST(SubIntSplitTopLevelPolicyTest, varianceGateRejectsConstantStream) {
  const std::vector<uint64_t> values(1'000, 0x1234'5678'9abc'def0ULL);
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));
  EXPECT_EQ(profile.variance, 0.0);

  const TopLevelPolicyConfig config;
  EXPECT_FALSE(bitFlipVarianceGate(profile, config));
}

TEST(
    SubIntSplitTopLevelPolicyTest,
    gradientBoundariesFindConcatenatedFieldEdges) {
  const auto values = makeConcatenatedFieldsStream(10'000);
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));

  const TopLevelPolicyConfig config;
  const auto boundaries = bitFlipGradientBoundaries(profile, config);

  ASSERT_GE(boundaries.size(), 2u);
  EXPECT_EQ(boundaries.front(), 0);
  EXPECT_EQ(boundaries.back(), 64);

  auto hasBoundaryNear = [&](int target, int tolerance) {
    return std::any_of(boundaries.begin(), boundaries.end(), [&](int b) {
      return std::abs(b - target) <= tolerance;
    });
  };
  EXPECT_TRUE(hasBoundaryNear(10, 2));
  EXPECT_TRUE(hasBoundaryNear(30, 2));
}

TEST(SubIntSplitTopLevelPolicyTest, gradientGateAcceptsConcatenatedFields) {
  const auto values = makeConcatenatedFieldsStream(10'000);
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));

  const TopLevelPolicyConfig config;
  EXPECT_TRUE(bitFlipGradientGate(profile, config));
}

TEST(SubIntSplitTopLevelPolicyTest, gradientGateRejectsUniformRandom) {
  // Needs more samples than the other gate tests: at 10,000 rows, sampling
  // noise in the gradient exceeds minGradientMagnitude and the gate fires
  // on it, not on a real signal.
  const auto values = makeUniformRandomStream(200'000);
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));

  const TopLevelPolicyConfig config;
  EXPECT_FALSE(bitFlipGradientGate(profile, config));
}

TEST(SubIntSplitTopLevelPolicyTest, gradientGateRejectsFlatProfileNoiseSpikes) {
  // A flat profile whose tiny floating-point noise still clears the
  // adaptive boundary threshold; minGradientMagnitude exists to reject
  // exactly this case.
  BitFlipProfile profile;
  profile.numBits = 64;
  for (int i = 0; i < 64; ++i) {
    profile.flipProbability[i] = 0.0;
    profile.gradient[i] = 0.0;
  }
  profile.gradient[5] = 1e-6;
  profile.gradient[40] = 1e-6;
  profile.variance = 0.0;

  const TopLevelPolicyConfig config;
  EXPECT_FALSE(bitFlipGradientGate(profile, config));
}

TEST(SubIntSplitTopLevelPolicyTest, estimatorGateSkipsUniformRandom) {
  const auto values = makeUniformRandomStream(10'000);
  const TopLevelPolicyConfig config;
  const auto result = estimateSubIntSplitSize<uint64_t>(
      std::span<const uint64_t>(values), config);

  EXPECT_FALSE(result.gatePassed);
  EXPECT_FALSE(result.estimatedBytes.has_value());
}

TEST(SubIntSplitTopLevelPolicyTest, estimatorGateAcceptsConcatenatedFields) {
  const auto values = makeConcatenatedFieldsStream(10'000);
  const TopLevelPolicyConfig config;
  const auto result = estimateSubIntSplitSize<uint64_t>(
      std::span<const uint64_t>(values), config);

  EXPECT_TRUE(result.gatePassed);
  ASSERT_TRUE(result.estimatedBytes.has_value());
  EXPECT_GT(result.estimatedBytes.value(), 0u);
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
