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

#include "velox/functions/prestosql/aggregates/sketch/SfmSketch.h"
#include "gtest/gtest.h"
#include "velox/common/encode/Base64.h"
#include "velox/common/memory/Memory.h"
#include "velox/functions/prestosql/aggregates/sketch/MersenneTwisterRandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {

class SfmSketchTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  double mergeRandomizedResponseProbabilities(double p1, double p2) {
    return (p1 + p2 - 3 * p1 * p2) / (1 - 2 * p1 * p2);
  }

  double calculateRandomizedResponseProbability(double epsilon) {
    if (epsilon == std::numeric_limits<double>::infinity()) {
      return 0.0;
    }
    return 1.0 / (1.0 + exp(epsilon));
  }

 protected:
  int32_t numberOfBuckets_ = 4096;
  int32_t precision_ = 24;
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(SfmSketchTest, computeIndex) {
  std::vector<int32_t> indexBitLength = {6, 8, 10, 12};
  for (auto& length : indexBitLength) {
    int32_t index = 5;
    auto hash = static_cast<uint64_t>(index) << (64 - length);
    ASSERT_EQ(SfmSketch::computeIndex(hash, length), index);
  }
}

TEST_F(SfmSketchTest, numberOfTrailingZeros) {
  std::vector<int32_t> indexBitLength = {6, 8, 10, 12};
  for (auto& length : indexBitLength) {
    for (int32_t zeros = 0; zeros < 63; zeros++) {
      uint64_t hash = 1ULL << zeros;
      int32_t kBitWidth = sizeof(uint64_t) * 8;
      uint64_t value = hash | (1ULL << (kBitWidth - length));

      auto trailingZeros = static_cast<int32_t>(__builtin_ctzll(value));
      ASSERT_EQ(trailingZeros, std::min(zeros, 64 - length));
    }
  }
}

TEST_F(SfmSketchTest, numberOfBuckets) {
  std::vector<int32_t> indexBitLength = {6, 8, 10, 12};

  for (auto& length : indexBitLength) {
    ASSERT_EQ(SfmSketch::numBuckets(length), 1U << length);
  }
}

TEST_F(SfmSketchTest, privacyEnabled) {
  SfmSketch sketch(&allocator_);
  sketch.initialize(numberOfBuckets_, precision_);
  ASSERT_FALSE(sketch.privacyEnabled());
  sketch.enablePrivacy(std::numeric_limits<double>::infinity());
  ASSERT_FALSE(sketch.privacyEnabled());
  sketch.enablePrivacy(1.0);
  ASSERT_TRUE(sketch.privacyEnabled());
}

TEST_F(SfmSketchTest, mergeNonPrivacy) {
  auto sketchA = SfmSketch(&allocator_);
  sketchA.initialize(numberOfBuckets_, precision_);
  auto sketchB = SfmSketch(&allocator_);
  sketchB.initialize(numberOfBuckets_, precision_);

  // Add random values to the sketches
  for (int i = 0; i < 1000; i++) {
    sketchA.add(i);
    sketchB.add(i + 1000);
  }

  auto refBitMapRange = sketchA.bits();
  // Create a mutable copy for the test
  std::vector<int8_t> refBitMap(refBitMapRange.begin(), refBitMapRange.end());
  // Merge the sketches
  MersenneTwisterRandomizationStrategy strategy(3);
  sketchA.mergeWith(sketchB, strategy);
  // Size of bitmap after merge should be the same
  ASSERT_EQ(sketchA.numberOfBits(), numberOfBuckets_ * precision_);

  velox::bits::orBits(
      reinterpret_cast<uint64_t*>(refBitMap.data()),
      reinterpret_cast<const uint64_t*>(sketchB.bits().data()),
      0,
      static_cast<int32_t>(sketchA.numberOfBits()));
  // refBitMap should be the same as sketchA
  ASSERT_EQ(sketchA.bits().size(), refBitMap.size());

  for (int i = 0; i < sketchA.bits().size(); i++) {
    ASSERT_EQ(sketchA.bits()[i], refBitMap[i]);
  }

  ASSERT_FALSE(sketchA.privacyEnabled());
}

TEST_F(SfmSketchTest, mergePrivacy) {
  auto sketchA = SfmSketch(&allocator_);
  sketchA.initialize(numberOfBuckets_, precision_);
  auto sketchB = SfmSketch(&allocator_);
  sketchB.initialize(numberOfBuckets_, precision_);

  // Add random values to the sketches
  for (int i = 0; i < 100000; i++) {
    sketchA.add(i);
    sketchB.add(-i - 1);
  }
  auto nonPrivateBitMapARange = sketchA.bits();
  auto nonPrivateBitMapBRange = sketchB.bits();
  // Create mutable copies for the test
  std::vector<int8_t> nonPrivateBitMapA(
      nonPrivateBitMapARange.begin(), nonPrivateBitMapARange.end());
  std::vector<int8_t> nonPrivateBitMapB(
      nonPrivateBitMapBRange.begin(), nonPrivateBitMapBRange.end());
  MersenneTwisterRandomizationStrategy privacyStrategy(3);
  sketchA.enablePrivacy(3.0, privacyStrategy);
  sketchB.enablePrivacy(4.0, privacyStrategy);
  auto p1 = sketchA.randomizedResponseProbability();
  auto p2 = sketchB.randomizedResponseProbability();

  auto refBitMapRange = sketchA.bits();
  std::vector<int8_t> refBitMap(refBitMapRange.begin(), refBitMapRange.end());
  velox::bits::orBits(
      reinterpret_cast<uint64_t*>(refBitMap.data()),
      reinterpret_cast<const uint64_t*>(sketchB.bits().data()),
      0,
      static_cast<int32_t>(sketchA.numberOfBits()));

  // Merge the sketches
  MersenneTwisterRandomizationStrategy mergeStrategy(3);
  sketchA.mergeWith(sketchB, mergeStrategy);

  ASSERT_TRUE(sketchA.privacyEnabled());
  ASSERT_EQ(
      sketchA.randomizedResponseProbability(),
      mergeRandomizedResponseProbabilities(p1, p2));

  // Private merge result and non-private bitwiseOr result should be different
  bool same = true;
  auto vector1 = sketchA.bits();
  auto vector2 = refBitMap;
  if (vector1.size() != vector2.size()) {
    same = false;
  }
  for (auto i = 0; i < vector1.size(); i++) {
    if (vector1[i] != vector2[i]) {
      same = false;
      break;
    }
  }
  ASSERT_FALSE(same);

  // But the number of 1s should be approximately the same as a bitmap that
  // is flipped with the merged randomizedResponseProbability.
  auto hypotheticalBitMap = nonPrivateBitMapA;
  velox::bits::orBits(
      reinterpret_cast<uint64_t*>(hypotheticalBitMap.data()),
      reinterpret_cast<const uint64_t*>(nonPrivateBitMapB.data()),
      0,
      static_cast<int32_t>(nonPrivateBitMapB.size() * 8));

  MersenneTwisterRandomizationStrategy randomizationStrategy(3);
  for (auto i = 0; i < hypotheticalBitMap.size() * 8; i++) {
    if (randomizationStrategy.nextBoolean(
            sketchA.randomizedResponseProbability())) {
      velox::bits::negateBit(hypotheticalBitMap.data(), i);
    }
  }

  // numberOfBits = 4096 * 24 = 98304
  // With epsilon 3 and 4, p1: 0.047 p2: 0.018
  // merged p: 0.064
  // standard deviation = sqrt(98304 * 0.064 * (1 - 0.064)) ~= 76
  // Increased tolerance to account for variance in randomized merge algorithm
  // Better to have deterministic seed for testing
  ASSERT_NEAR(
      bits::countBits(
          reinterpret_cast<const uint64_t*>(sketchA.bits().data()),
          0,
          sketchA.numberOfBits()),
      bits::countBits(
          reinterpret_cast<const uint64_t*>(hypotheticalBitMap.data()),
          0,
          hypotheticalBitMap.size() * 8),
      150);
}

TEST_F(SfmSketchTest, mergeMixed) {
  SfmSketch sketchA = SfmSketch(&allocator_);
  sketchA.initialize(numberOfBuckets_, precision_);
  SfmSketch sketchB = SfmSketch(&allocator_);
  sketchB.initialize(numberOfBuckets_, precision_);

  // Add random values to the sketches
  for (int i = 0; i < 100000; i++) {
    sketchA.add(i);
    sketchB.add(-i - 1);
  }

  // sketchA is non-private, sketchB is private
  MersenneTwisterRandomizationStrategy privacyStrategy(3);
  sketchB.enablePrivacy(3.0, privacyStrategy);
  auto nonPrivateBitMapARange = sketchA.bits();
  // Create a copy of the non-private bitmap for comparison
  std::vector<int8_t> nonPrivateBitMapA(
      nonPrivateBitMapARange.begin(), nonPrivateBitMapARange.end());

  // Store sketchB bits before merge for comparison
  auto sketchBBitsRange = sketchB.bits();
  std::vector<int8_t> sketchBBits(
      sketchBBitsRange.begin(), sketchBBitsRange.end());

  // merge sketchA with sketchB
  MersenneTwisterRandomizationStrategy mergeStrategy(3);
  sketchA.mergeWith(sketchB, mergeStrategy);

  // sketchA should be private now
  ASSERT_TRUE(sketchA.privacyEnabled());

  // A mixed-privacy merge is mathematically similar to a normal private
  // merge, but it turns out that some bits are deterministic. In particular,
  // the bits of the merged sketch corresponding to 0s in the non-private
  // sketch should exactly match the private sketch. Mathematically, this is
  // because when the non-private sketch has a 0, it does not contribute to
  // the probability used to flip the bit in the merged sketch. If the private
  // sketch is 1, then the merged sketch is 1 with probability 1, and if the
  // private sketch is 0, then the merged sketch is 1 with probability 0. So
  // the merged sketch is exactly the private sketch for 0s in the non-private
  // sketch.
  auto mergedBitsRange = sketchA.bits();
  for (auto i = 0; i < nonPrivateBitMapA.size() * 8; i++) {
    if (!bits::isBitSet(nonPrivateBitMapA.data(), i)) {
      ASSERT_EQ(
          bits::isBitSet(mergedBitsRange.data(), i),
          bits::isBitSet(sketchBBits.data(), i));
    }
  }
}

TEST_F(SfmSketchTest, mergedProbability) {
  // Symmetric test.
  ASSERT_EQ(
      mergeRandomizedResponseProbabilities(0.1, 0.4),
      mergeRandomizedResponseProbabilities(0.4, 0.1));

  // private + nonprivate = private
  ASSERT_EQ(mergeRandomizedResponseProbabilities(0, 0.1), 0.1);
  ASSERT_EQ(mergeRandomizedResponseProbabilities(0.15, 0), 0.15);

  // nonprivate + nonprivate = nonprivate
  ASSERT_EQ(mergeRandomizedResponseProbabilities(0.0, 0.0), 0.0);

  // private + private = private, and the merged noise is higher
  // In particular, according to https://arxiv.org/pdf/2302.02056.pdf,
  // Theorem 4.8, two sketches with epsilon1 and epsilon2 should have a merged
  // epsilonStar of: -log(e^-epsilon1 + e^-epsilon2 - e^-(epsilon1 +
  // epsilon2))
  double epsilon1 = 1.2;
  double epsilon2 = 3.4;
  double p1 = calculateRandomizedResponseProbability(epsilon1);
  double p2 = calculateRandomizedResponseProbability(epsilon2);
  double epsilonStar = -std::log(
      std::exp(-epsilon1) + std::exp(-epsilon2) -
      std::exp(-(epsilon1 + epsilon2)));
  double pStar = calculateRandomizedResponseProbability(epsilonStar);
  ASSERT_NEAR(mergeRandomizedResponseProbabilities(p1, p2), pStar, 1E-6);
  // note: the merged sketch is noisier (higher probability of flipped bits)
  ASSERT_TRUE(pStar > std::max(p1, p2));
}

TEST_F(SfmSketchTest, emptySketchCardinality) {
  SfmSketch sketch(&allocator_);
  sketch.initialize(numberOfBuckets_, precision_);
  // Non-private sketch should have 0 cardinality
  ASSERT_EQ(sketch.cardinality(), 0);

  // Private sketch should have approximately 0 cardinality, but not exactly 0
  MersenneTwisterRandomizationStrategy strategy(3);
  sketch.enablePrivacy(3.0, strategy);
  ASSERT_NEAR(sketch.cardinality(), 0, 200);
}

TEST_F(SfmSketchTest, smallCardinality) {
  std::vector<int32_t> values = {1, 10, 50, 100, 500, 1000};
  for (auto& value : values) {
    auto nonPrivateSketch = SfmSketch(&allocator_);
    nonPrivateSketch.initialize(numberOfBuckets_, precision_);
    auto privateSketch = SfmSketch(&allocator_);
    privateSketch.initialize(numberOfBuckets_, precision_);
    for (int i = 0; i < value; i++) {
      nonPrivateSketch.add(i);
      privateSketch.add(i);
    }

    // Non-private sketch should have pretty good cardinality estimate.
    ASSERT_NEAR(
        nonPrivateSketch.cardinality(),
        value,
        std::max<double>(10, 0.1 * value));

    // Private sketch not as good.
    MersenneTwisterRandomizationStrategy strategy(3);
    privateSketch.enablePrivacy(3.0, strategy);
    ASSERT_NEAR(privateSketch.cardinality(), value, 200);
  }
}

TEST_F(SfmSketchTest, actualCardinality) {
  std::vector<int32_t> magnitudes = {4, 5, 6};
  std::vector<double> epsilons = {
      2.0, 4.0, std::numeric_limits<double>::infinity()};
  for (auto& magnitude : magnitudes) {
    auto n = std::pow(10L, magnitude);
    for (auto& epsilon : epsilons) {
      auto privateSketch = SfmSketch(&allocator_);
      privateSketch.initialize(numberOfBuckets_, precision_);

      for (auto i = 0; i < n; i++) {
        privateSketch.add(i);
      }
      MersenneTwisterRandomizationStrategy strategy(3);
      privateSketch.enablePrivacy(epsilon, strategy);
      // Typically, the error is about 5% of the cardinality.
      ASSERT_NEAR(privateSketch.cardinality(), n, n * 0.05);
    }
  }
}

TEST_F(SfmSketchTest, mergedCardinality) {
  std::vector<double> epsilons = {
      3, 4, std::numeric_limits<double>::infinity()};

  // Test each pair of epsilons
  for (auto& eps1 : epsilons) {
    for (auto& eps2 : epsilons) {
      // Create two sketches and add disjoint sets of values
      SfmSketch sketchA = SfmSketch(&allocator_);
      sketchA.initialize(numberOfBuckets_, precision_);
      SfmSketch sketchB = SfmSketch(&allocator_);
      sketchB.initialize(numberOfBuckets_, precision_);

      // Add 300,000 positive integers to sketchA and 200,000 negative
      // integers to sketchB This ensures disjoint sets with zero overlap
      for (int i = 1; i <= 300000; i++) {
        sketchA.add(i); // positive integers 1 to 300,000
      }
      for (int i = 0; i < 200000; i++) {
        sketchB.add(-i); // negative integers 0 to -199,999
      }

      MersenneTwisterRandomizationStrategy privacyStrategy(3);
      sketchA.enablePrivacy(eps1, privacyStrategy);
      sketchB.enablePrivacy(eps2, privacyStrategy);

      // Merge the sketches
      MersenneTwisterRandomizationStrategy mergeStrategy(1);
      sketchA.mergeWith(sketchB, mergeStrategy);

      // The merged sketch should have cardinality 500,000 (300,000 + 200,000
      // disjoint values)
      ASSERT_NEAR(sketchA.cardinality(), 500000, 50000);
    }
  }
}

TEST_F(SfmSketchTest, enablePrivacy) {
  SfmSketch sketch = SfmSketch(&allocator_);
  sketch.initialize(numberOfBuckets_, precision_);
  double epsilon = 4;

  for (int32_t i = 0; i < 100000; i++) {
    sketch.add(i);
  }

  int64_t cardinalityBefore = sketch.cardinality();
  MersenneTwisterRandomizationStrategy strategy(3);
  sketch.enablePrivacy(epsilon, strategy);
  int64_t cardinalityAfter = sketch.cardinality();

  // Randomized response probability should reflect the new (private) epsilon
  ASSERT_EQ(
      sketch.randomizedResponseProbability(),
      calculateRandomizedResponseProbability(epsilon));
  ASSERT_TRUE(sketch.privacyEnabled());

  // Cardinality should remain approximately the same
  ASSERT_NEAR(cardinalityAfter, cardinalityBefore, cardinalityBefore * 0.1);
}

TEST_F(SfmSketchTest, serializationRoundTrip) {
  SfmSketch sketch = SfmSketch(&allocator_);
  sketch.initialize(numberOfBuckets_, precision_);
  for (int32_t i = 0; i < 100000; i++) {
    sketch.add(i);
  }

  sketch.enablePrivacy(4.0);
  auto originalBitMapRange = sketch.bits();
  // Create a copy of the original bitmap for comparison
  std::vector<int8_t> originalBitMap(
      originalBitMapRange.begin(), originalBitMapRange.end());
  auto originalIndexBitLength = sketch.indexBitLength();
  auto originalPrecision = sketch.precision();
  auto originalRandomizedResponseProbability =
      sketch.randomizedResponseProbability();

  // Allocate buffer for serialization
  size_t serializedSize = sketch.serializedSize();
  std::vector<char> buffer(serializedSize);
  char* out = buffer.data();

  sketch.serialize(out);
  auto deserialized = SfmSketch::deserialize(out, &allocator_);

  // Test that the deserialized sketch is the same as the original
  ASSERT_EQ(deserialized.indexBitLength(), originalIndexBitLength);
  ASSERT_EQ(deserialized.precision(), originalPrecision);
  ASSERT_EQ(
      deserialized.randomizedResponseProbability(),
      originalRandomizedResponseProbability);

  // Test that the bitmaps are the same
  ASSERT_EQ(deserialized.numberOfBits(), originalBitMap.size() * 8);
  auto deserializedBitsRange = deserialized.bits();
  for (int i = 0; i < deserialized.numberOfBits(); i++) {
    ASSERT_EQ(
        bits::isBitSet(deserializedBitsRange.data(), i),
        bits::isBitSet(originalBitMap.data(), i));
  }
}

TEST_F(SfmSketchTest, javaSerializationCompatibility) {
  // This test is to ensure that the C++ serialization format is compatible with
  // the Java serialization format.

  // The string is from a Java sketch serialized with the Java SfmSketch
  std::string base64_data =
      "BwgAAAAQAAAAw8JDHvpqkj//AQAA77//////////////v///////////3////7//////////////3/////////////////////////////////v//+/+ff/////////f///v//+//////////////////////7//////3////v//////3////v/f/////////3////v///////v/f///////////////////////////////f///////v/////////f///////7////+////9///////////////v//////////////////////////////////////f////////////3//////////////7////+////////////////////3///////////////////////9///33///v///3////f//f//9//+//////r/3//////9//Dfz/e/++vvfV3W//qtq33z//a7///+/m3/fz7/1///1v5hmo/ayTcv3l3VvXmnMr+t/B72YloTNgmBy7+IKv+mIBmaBwSO4GwxBMCwWChIoYSVFTAAIIhLylIEjmJLamBAAooMYLQgikgBAUACEAQYJUAmxMGAhDAoBJGgRgAkBCCCAABAAkIIAAAAAEQAYEECAQYACgAACEAAQEBAACCAgAAAAQgAAAAAKgAoERABAMAAQbABAAgADEAAQIABQ==";

  // Decode base64 to binary data
  std::string decoded = encoding::Base64::decode(base64_data);

  auto sketch = SfmSketch::deserialize(decoded.c_str(), &allocator_);
  // Test that the deserialized sketch is the same as the original
  ASSERT_EQ(sketch.cardinality(), 927499);
}

} // namespace facebook::velox::functions::aggregate
