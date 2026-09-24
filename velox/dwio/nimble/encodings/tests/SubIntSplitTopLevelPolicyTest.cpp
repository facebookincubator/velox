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
#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/Estimator.h"
#include "velox/dwio/nimble/encodings/subintsplit/TopLevelPolicy.h"

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

// The default read factors with SubIntSplit among them. The defaults need not
// list it, and these tests are about how selection treats it once it is a
// candidate.
std::vector<std::pair<EncodingType, float>> readFactorsWithSubIntSplit() {
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  if (std::none_of(
          readFactors.begin(), readFactors.end(), [](const auto& factor) {
            return factor.first == EncodingType::SubIntSplit;
          })) {
    readFactors.emplace_back(EncodingType::SubIntSplit, 0.85f);
  }
  return readFactors;
}

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

// Snowflake-shaped packed fields: a slowly increasing timestamp in bits
// [30, 63], a machine id in bits [20, 21] that changes every 500 rows, and a
// random 12-bit sequence in bits [0, 11].
std::vector<uint64_t> makePackedFieldsStream(size_t n) {
  std::mt19937_64 rng(kSeed);
  constexpr uint64_t kMachineIds[] = {1, 3, 0, 2};
  std::vector<uint64_t> values(n);
  for (size_t i = 0; i < n; ++i) {
    values[i] = ((uint64_t{1'700'000} + i / 64) << 30) |
        (kMachineIds[(i / 500) % 4] << 20) | (rng() & 0xFFF);
  }
  return values;
}

// The per-bit shift-and-mask count the lane-counted profile has to match.
std::vector<uint64_t> countFlipsBitByBit(
    const std::vector<uint64_t>& values,
    size_t stride) {
  std::vector<uint64_t> counts(64, 0);
  for (size_t i = 0; i + 1 < values.size(); i += stride) {
    const uint64_t flipped = values[i] ^ values[i + 1];
    for (int b = 0; b < 64; ++b) {
      counts[b] += (flipped >> b) & 1;
    }
  }
  return counts;
}

// A sharp bit-flip boundary at bit 10, from a field that steps through 16
// values in runs of eight; bits [0, 9] and [14, 63] never flip. Both gates
// admit it -- the gradient spike is 0.125 and the varying bits carry 0.30 bits
// of flip entropy -- and a run-length or dictionary encoding still stores it
// in a fraction of what a split of its 14-bit range could.
std::vector<uint64_t> makeLowCardinalityFieldsStream(size_t n) {
  std::vector<uint64_t> values(n);
  for (size_t i = 0; i < n; ++i) {
    values[i] = ((i / 8) % 16) << 10;
  }
  return values;
}

// What top-level selection picks for `values` under `mode`, with the
// candidates and read factors the writer uses by default.
EncodingType selectedUnder(
    const std::vector<uint64_t>& values,
    SubIntSplitAdmission mode,
    bool forces,
    uint32_t profilePairs) {
  const std::span<const uint64_t> span(values);
  Encoding::Options options;
  options.subIntSplitAdmission = static_cast<uint8_t>(mode);
  options.subIntSplitAdmissionForces = forces;
  options.subIntSplitAdmissionProfilePairs = profilePairs;
  ManualEncodingSelectionPolicy<uint64_t> policy{
      readFactorsWithSubIntSplit(),
      CompressionOptions{},
      std::nullopt,
  };
  return policy.select(span, Statistics<uint64_t>::create(span), options)
      .encodingType;
}

bool admits(const std::vector<uint64_t>& values, SubIntSplitAdmission mode) {
  return bitFlipAdmits(
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values)),
      mode,
      TopLevelPolicyConfig{});
}

} // namespace

TEST(SubIntSplitTopLevelPolicyTest, profileMatchesBitByBitCount) {
  // 1'000 values span several 255-pair count drains and a partial one.
  auto values = makePackedFieldsStream(1'000);
  values[500] = ~uint64_t{0};
  const auto profile =
      computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(values));
  const auto counts = countFlipsBitByBit(values, 1);
  uint64_t varyingBits{0};
  for (int b = 0; b < 64; ++b) {
    EXPECT_EQ(profile.flipProbability[b], counts[b] / 999.0) << "bit: " << b;
    varyingBits |= uint64_t{counts[b] > 0} << b;
  }
  EXPECT_EQ(profile.varyingBits, varyingBits);
}

TEST(SubIntSplitTopLevelPolicyTest, sampledProfileUsesStridedPairs) {
  const auto values = makePackedFieldsStream(10'001);
  const std::span<const uint64_t> span(values);
  // A cap at or above the pair count is the full profile.
  const auto full = computeBitFlipProfile<uint64_t>(span);
  const auto capped = computeBitFlipProfile<uint64_t>(span, 10'000);
  EXPECT_EQ(full.flipProbability, capped.flipProbability);

  // 10'000 pairs capped at 1'024 take every 10th pair, 1'000 of them.
  const auto sampled = computeBitFlipProfile<uint64_t>(span, 1'024);
  const auto counts = countFlipsBitByBit(values, 10);
  for (int b = 0; b < 64; ++b) {
    EXPECT_EQ(sampled.flipProbability[b], counts[b] / 1'000.0) << "bit: " << b;
  }
  // Varying bits still come from every value.
  EXPECT_EQ(sampled.varyingBits, full.varyingBits);
}

TEST(SubIntSplitTopLevelPolicyTest, flipCountsMatchScalarReference) {
  // Word patterns the bit-sliced counter has to carry through: all bits set
  // (every counter overflows on the same word), one bit set, none set, and
  // random. 1'000 words drive several sixteen-word carries and leave a partial
  // vector at the end.
  auto words = makeUniformRandomStream(1'000);
  words[0] = ~uint64_t{0};
  words[1] = 0;
  words[2] = uint64_t{1} << 63;
  for (size_t i = 100; i < 140; ++i) {
    words[i] = ~uint64_t{0};
  }
  for (size_t size :
       {size_t{0},
        size_t{1},
        size_t{3},
        size_t{17},
        size_t{999},
        size_t{1'000}}) {
    const std::span<const uint64_t> span(words.data(), size);
    std::array<uint64_t, kMaxBitWidth> vectorised{};
    std::array<uint64_t, kMaxBitWidth> reference{};
    ::facebook::nimble::detail::accumulateFlipCounts(span, vectorised);
    ::facebook::nimble::detail::accumulateFlipCountsScalar(span, reference);
    EXPECT_EQ(vectorised, reference) << "size: " << size;
  }
}

TEST(SubIntSplitTopLevelPolicyTest, gradientGateProfileSkipsVaryingBits) {
  const auto values = makePackedFieldsStream(10'001);
  const std::span<const uint64_t> span(values);
  const auto withVaryingBits = computeBitFlipProfile<uint64_t>(span, 1'024);
  const auto gateProfile =
      bitFlipAdmissionProfile(span, SubIntSplitAdmission::kBitFlip, 1'024);
  // The gradient gate reads the probabilities, not the varying bits, so
  // skipping the whole-stream pass leaves its decision alone.
  EXPECT_EQ(gateProfile.flipProbability, withVaryingBits.flipProbability);
  EXPECT_EQ(gateProfile.gradient, withVaryingBits.gradient);
  EXPECT_EQ(gateProfile.varyingBits, 0);
  EXPECT_EQ(
      bitFlipGradientGate(gateProfile, TopLevelPolicyConfig{}),
      bitFlipGradientGate(withVaryingBits, TopLevelPolicyConfig{}));

  // The entropy guard does read them, so its profile keeps the pass.
  const auto entropyProfile = bitFlipAdmissionProfile(
      span, SubIntSplitAdmission::kBitFlipEntropy, 1'024);
  EXPECT_EQ(entropyProfile.varyingBits, withVaryingBits.varyingBits);
}

TEST(SubIntSplitTopLevelPolicyTest, activeBitFlipEntropy) {
  BitFlipProfile profile;
  profile.numBits = 64;
  EXPECT_EQ(activeBitFlipEntropy(profile), 0.0);

  // Bits that always flip carry no entropy but still count as varying.
  profile.flipProbability[0] = 1.0;
  profile.flipProbability[1] = 0.5;
  profile.varyingBits = 0b11;
  EXPECT_DOUBLE_EQ(activeBitFlipEntropy(profile), 0.5);

  profile.flipProbability[2] = 0.25;
  profile.varyingBits = 0b111;
  EXPECT_NEAR(activeBitFlipEntropy(profile), (1.0 + 0.811'278) / 3, 1e-6);

  // A varying bit a sample never saw flip counts, at zero entropy.
  profile.varyingBits = 0b1111;
  EXPECT_NEAR(activeBitFlipEntropy(profile), (1.0 + 0.811'278) / 4, 1e-6);
}

TEST(SubIntSplitTopLevelPolicyTest, admissionRejectsConstantStream) {
  const std::vector<uint64_t> values(10'000, 42);
  EXPECT_FALSE(admits(values, SubIntSplitAdmission::kBitFlip));
  EXPECT_FALSE(admits(values, SubIntSplitAdmission::kBitFlipEntropy));
  EXPECT_FALSE(admits({42}, SubIntSplitAdmission::kBitFlip));
}

TEST(SubIntSplitTopLevelPolicyTest, entropyGuardRejectsRandomStreams) {
  // At 10'000 rows sampling noise is enough for the gradient gate to fire on
  // a random stream; the entropy guard sees every bit flipping at 0.5.
  const auto random = makeUniformRandomStream(10'000);
  EXPECT_GT(
      activeBitFlipEntropy(
          computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(random))),
      0.99);
  EXPECT_FALSE(admits(random, SubIntSplitAdmission::kBitFlipEntropy));
  // One random field with constant bits around it: splitting has nothing to
  // gain over dropping the constant bits.
  const auto field = makeConcatenatedFieldsStream(10'000);
  EXPECT_TRUE(admits(field, SubIntSplitAdmission::kBitFlip));
  EXPECT_FALSE(admits(field, SubIntSplitAdmission::kBitFlipEntropy));
}

TEST(SubIntSplitTopLevelPolicyTest, admissionAcceptsPackedFields) {
  const auto values = makePackedFieldsStream(10'000);
  EXPECT_TRUE(admits(values, SubIntSplitAdmission::kBitFlip));
  EXPECT_TRUE(admits(values, SubIntSplitAdmission::kBitFlipEntropy));
}

TEST(SubIntSplitTopLevelPolicyTest, admissionAcceptsMonotoneCounter) {
  // A dense counter halves its flip rate per bit, a sharp gradient at bit 1
  // and low active entropy (about 0.22), so both modes admit it. Neither
  // guard can tell it from a split-friendly column; this pins that.
  std::vector<uint64_t> values(10'000);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = i;
  }
  EXPECT_TRUE(admits(values, SubIntSplitAdmission::kBitFlip));
  EXPECT_TRUE(admits(values, SubIntSplitAdmission::kBitFlipEntropy));
}

TEST(SubIntSplitTopLevelPolicyTest, selectionFollowsAdmissionMode) {
  const auto packed = makePackedFieldsStream(10'000);
  const auto random = makeUniformRandomStream(10'000);
  auto select = [](const std::vector<uint64_t>& values, uint8_t mode) {
    const std::span<const uint64_t> span(values);
    const auto statistics = Statistics<uint64_t>::create(span);
    ManualEncodingSelectionPolicy<uint64_t> policy{
        readFactorsWithSubIntSplit(), CompressionOptions{}, std::nullopt};
    Encoding::Options options;
    options.subIntSplitAdmission = mode;
    return policy.select(span, statistics, options).encodingType;
  };
  EXPECT_EQ(select(packed, 2), EncodingType::SubIntSplit);
  EXPECT_NE(select(random, 2), EncodingType::SubIntSplit);
  // A sampled profile admits it too. At 1'024 pairs (every 10th pair) the
  // timestamp and machine bits rarely flip in a sampled pair; they still
  // count as varying, so the random sequence bits do not dominate the mean.
  const std::span<const uint64_t> span(packed);
  const auto statistics = Statistics<uint64_t>::create(span);
  ManualEncodingSelectionPolicy<uint64_t> policy{
      readFactorsWithSubIntSplit(), CompressionOptions{}, std::nullopt};
  Encoding::Options options;
  options.subIntSplitAdmission = 2;
  options.subIntSplitAdmissionProfilePairs = 1'024;
  EXPECT_EQ(
      policy.select(span, statistics, options).encodingType,
      EncodingType::SubIntSplit);
}

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

TEST(SubIntSplitTopLevelPolicyTest, admittedStreamStillHasToWinOnSize) {
  const auto values = makeLowCardinalityFieldsStream(50'000);
  // The premise: both gates admit, so whatever selection does next is the
  // size comparison and not the admission.
  ASSERT_TRUE(admits(values, SubIntSplitAdmission::kBitFlip));
  ASSERT_TRUE(admits(values, SubIntSplitAdmission::kBitFlipEntropy));

  EXPECT_NE(
      selectedUnder(
          values,
          SubIntSplitAdmission::kBitFlip,
          /*forces=*/false,
          /*profilePairs=*/0),
      EncodingType::SubIntSplit);
  EXPECT_NE(
      selectedUnder(
          values,
          SubIntSplitAdmission::kBitFlipEntropy,
          /*forces=*/false,
          /*profilePairs=*/0),
      EncodingType::SubIntSplit);
}

TEST(SubIntSplitTopLevelPolicyTest, forcingSelectsAnAdmittedStreamOutright) {
  // The ablation switch: the same admitted stream as above, which the size
  // comparison turns down, is SubIntSplit once the admission decides alone.
  const auto values = makeLowCardinalityFieldsStream(50'000);

  EXPECT_EQ(
      selectedUnder(
          values,
          SubIntSplitAdmission::kBitFlip,
          /*forces=*/true,
          /*profilePairs=*/0),
      EncodingType::SubIntSplit);
}

TEST(SubIntSplitTopLevelPolicyTest, rejectedStreamIsNeverSubIntSplit) {
  const auto values = makeUniformRandomStream(200'000);
  ASSERT_FALSE(admits(values, SubIntSplitAdmission::kBitFlip));
  ASSERT_FALSE(admits(values, SubIntSplitAdmission::kBitFlipEntropy));

  for (const auto mode :
       {SubIntSplitAdmission::kBitFlip,
        SubIntSplitAdmission::kBitFlipEntropy}) {
    // Forcing cannot resurrect a rejected stream either: a rejected stream
    // loses the candidate before the two paths part.
    EXPECT_NE(
        selectedUnder(values, mode, /*forces=*/false, /*profilePairs=*/0),
        EncodingType::SubIntSplit);
    EXPECT_NE(
        selectedUnder(values, mode, /*forces=*/true, /*profilePairs=*/0),
        EncodingType::SubIntSplit);
  }
}

TEST(SubIntSplitTopLevelPolicyTest, sampledGateAdmitsWhatWholeStreamRejects) {
  // What sampling the profile costs, stated rather than hidden. A flip
  // probability taken from 1'024 pairs has a standard error of 0.016, so the
  // largest of 63 adjacent differences is around 0.06 on noise alone, well
  // above the gate's 0.005 floor. Raising the floor to that noise was measured
  // on 60 columns and rejected: it removed no false positive the gate makes on
  // real data -- those have gradients above 0.12, and are streams with real
  // field structure a split still does not pay for -- and cost four true
  // positives whose own boundaries are below it.
  const auto values = makeUniformRandomStream(200'000);
  const std::span<const uint64_t> span(values);
  EXPECT_FALSE(bitFlipGradientGate(
      computeBitFlipProfile<uint64_t>(span), TopLevelPolicyConfig{}));
  EXPECT_TRUE(bitFlipGradientGate(
      bitFlipAdmissionProfile(span, SubIntSplitAdmission::kBitFlip, 1'024),
      TopLevelPolicyConfig{}));

  // Which is why the gate is a candidacy screen and not the decision: with
  // the estimate still pricing the candidate the sampled gate's admission
  // changes nothing, and only letting the gate decide alone turns it into a
  // split nobody wanted.
  EXPECT_NE(
      selectedUnder(
          values,
          SubIntSplitAdmission::kBitFlip,
          /*forces=*/false,
          /*profilePairs=*/1'024),
      EncodingType::SubIntSplit);
  EXPECT_EQ(
      selectedUnder(
          values,
          SubIntSplitAdmission::kBitFlip,
          /*forces=*/true,
          /*profilePairs=*/1'024),
      EncodingType::SubIntSplit);
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
