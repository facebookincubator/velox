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

#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/SfmSketch.h"
#include "gtest/gtest.h"
#include "velox/common/memory/Memory.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/RandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {
using facebook::velox::functions::aggregate::SfmSketch;

class TestingSeededRandomizationStrategy : public RandomizationStrategy {
 public:
  explicit TestingSeededRandomizationStrategy(int64_t seed)
      : rng_(std::mt19937_64()) {
    rng_.seed(seed);
  }

  bool nextBoolean(double probability) override {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(rng_) < probability;
  }

 private:
  std::mt19937_64 rng_;
};

class SfmSketchTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

 protected:
  uint32_t numberOfBuckets_ = 4096;
  uint32_t precision_ = 24;
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};

  static SfmSketch createSketchWithTargetCardinality(
      uint32_t numberOfBuckets,
      uint32_t precision,
      double epsilon,
      uint64_t cardinality,
      HashStringAllocator* allocator) {
    // Building a sketch by adding items is really slow (O(n)) if you want to
    // test billions/trillions/quadrillions/etc. Simulating the sketch is much
    // faster (O(buckets * precision)).
    TestingSeededRandomizationStrategy randomizationStrategy(1);
    SfmSketch sketch = SfmSketch(allocator);
    sketch.setSketchSize(numberOfBuckets, precision);
    double c1 = sketch.getOnProbability();
    double c2 =
        sketch.getOnProbability() - sketch.getRandomizedResponseProbability();

    for (uint32_t l = 0; l < precision; l++) {
      double observationProb = std::pow(2.0, -(static_cast<double>(l) + 1.0)) /
          static_cast<double>(numberOfBuckets);
      double p = c1 -
          c2 *
              std::pow(1.0 - observationProb, static_cast<double>(cardinality));

      for (uint32_t b = 0; b < numberOfBuckets; b++) {
        uint32_t bitPosition = l * numberOfBuckets + b;
        bits::setBit(
            sketch.getBitSet().data(),
            bitPosition,
            randomizationStrategy.nextBoolean(p));
      }
    }

    sketch.enablePrivacy(epsilon, randomizationStrategy);
    return sketch;
  }

  std::vector<int8_t, facebook::velox::StlAllocator<int8_t>>& getBitSet(
      SfmSketch& sketch) {
    return sketch.getBitSet();
  }
};

TEST_F(SfmSketchTest, computeIndexTest) {
  std::vector<uint32_t> indexBitLength = {6, 8, 10, 12};
  for (auto& length : indexBitLength) {
    uint32_t index = 5;
    auto hash = static_cast<uint64_t>(index) << (64 - length);
    ASSERT_EQ(SfmSketch::computeIndex(hash, length), index);
  }
}

TEST_F(SfmSketchTest, numOfZerosTest) {
  std::vector<uint32_t> indexBitLength = {6, 8, 10, 12};
  for (auto& length : indexBitLength) {
    for (uint32_t zeros = 0; zeros < 63; zeros++) {
      uint64_t hash = 1ULL << zeros;
      ASSERT_EQ(
          SfmSketch::numberOfTrailingZeros(hash, length),
          std::min(zeros, 64 - length));
    }
  }
}

TEST_F(SfmSketchTest, numOfbucketsTest) {
  std::vector<uint32_t> indexBitLength = {6, 8, 10, 12};

  for (auto& length : indexBitLength) {
    ASSERT_EQ(SfmSketch::numberOfBuckets(length), 1U << length);
  }
}

TEST_F(SfmSketchTest, bitMapSizeTest) {
  std::vector<uint32_t> numOfBuckets = {32, 64, 512, 1024, 4096, 32768};
  std::vector<uint32_t> precisions = {1, 2, 3, 8, 24, 32};

  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());

  for (auto& numOfBucket : numOfBuckets) {
    for (auto& precision : precisions) {
      auto sketch = SfmSketch(&allocator);
      sketch.setSketchSize(numOfBucket, precision);
      ASSERT_EQ(sketch.getNumberOfBits(), numOfBucket * precision);
    }
  }
}

TEST_F(SfmSketchTest, privacyEnabledTest) {
  auto sketch = SfmSketch(&allocator_);
  sketch.setSketchSize(numberOfBuckets_, precision_);
  ASSERT_FALSE(sketch.privacyEnabled());
  sketch.enablePrivacy(std::numeric_limits<double>::infinity());
  ASSERT_FALSE(sketch.privacyEnabled());
  sketch.enablePrivacy(1.0);
  ASSERT_TRUE(sketch.privacyEnabled());
}

TEST_F(SfmSketchTest, mergeNonPrivacyTest) {
  auto sketch1 = SfmSketch(&allocator_);
  sketch1.setSketchSize(numberOfBuckets_, precision_);
  auto sketch2 = SfmSketch(&allocator_);
  sketch2.setSketchSize(numberOfBuckets_, precision_);

  // Add random values to the sketches
  for (int i = 0; i < 1000; i++) {
    sketch1.add(i);
    sketch2.add(i + 1000);
  }

  auto refBitMap = getBitSet(sketch1);
  // Merge the sketches
  sketch1.mergeWith(sketch2);
  // Size of bitmap after merge should be the same
  ASSERT_EQ(sketch1.getNumberOfBits(), numberOfBuckets_ * precision_);

  velox::bits::orBits(
      reinterpret_cast<uint64_t*>(refBitMap.data()),
      reinterpret_cast<const uint64_t*>(getBitSet(sketch2).data()),
      0,
      static_cast<int32_t>(sketch1.getNumberOfBits()));
  // refBitMap should be the same as sketch1
  ASSERT_EQ(getBitSet(sketch1).size(), refBitMap.size());
  // Convert the bitmaps to vectors and compare them
  for (int i = 0; i < getBitSet(sketch1).size(); i++) {
    ASSERT_EQ(getBitSet(sketch1)[i], refBitMap[i]);
  }

  ASSERT_FALSE(sketch1.privacyEnabled());
}

TEST_F(SfmSketchTest, mergePrivacyTest) {
  auto sketch1 = SfmSketch(&allocator_);
  sketch1.setSketchSize(numberOfBuckets_, precision_);
  auto sketch2 = SfmSketch(&allocator_);
  sketch2.setSketchSize(numberOfBuckets_, precision_);

  // Add random values to the sketches
  for (int i = 0; i < 100000; i++) {
    sketch1.add(i);
    sketch2.add(-i - 1);
  }
  auto nonPrivateBitMap1 = getBitSet(sketch1);
  auto nonPrivateBitMap2 = getBitSet(sketch2);

  sketch1.enablePrivacy(3.0, TestingSeededRandomizationStrategy(1));
  sketch2.enablePrivacy(4.0, TestingSeededRandomizationStrategy(2));
  auto p1 = sketch1.getRandomizedResponseProbability();
  auto p2 = sketch2.getRandomizedResponseProbability();

  auto refBitMap = getBitSet(sketch1);
  velox::bits::orBits(
      reinterpret_cast<uint64_t*>(refBitMap.data()),
      reinterpret_cast<const uint64_t*>(getBitSet(sketch2).data()),
      0,
      static_cast<int32_t>(sketch1.getNumberOfBits()));

  // Merge the sketches
  sketch1.mergeWith(sketch2, TestingSeededRandomizationStrategy(3));

  ASSERT_TRUE(sketch1.privacyEnabled());
  ASSERT_EQ(
      sketch1.getRandomizedResponseProbability(),
      SfmSketch::mergeRandomizedResponseProbabilities(p1, p2));

  // Private merge result and non-private bitwiseOr result should be different
  bool same = true;
  auto vector1 = getBitSet(sketch1);
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
  auto hypotheticalBitMap = nonPrivateBitMap1;
  velox::bits::orBits(
      reinterpret_cast<uint64_t*>(hypotheticalBitMap.data()),
      reinterpret_cast<const uint64_t*>(nonPrivateBitMap2.data()),
      0,
      static_cast<int32_t>(nonPrivateBitMap2.size() * 8));

  TestingSeededRandomizationStrategy randomizationStrategy(1);
  for (auto i = 0; i < hypotheticalBitMap.size() * 8; i++) {
    if (randomizationStrategy.nextBoolean(
            sketch1.getRandomizedResponseProbability())) {
      velox::bits::negateBit(hypotheticalBitMap.data(), i);
    }
  }

  // numberOfBits = 4096 * 24 = 98304
  // With epsilon 3 and 4, p1: 0.047 p2: 0.018
  // merged p: 0.064
  // standard deviation = sqrt(98304 * 0.064 * (1 - 0.064)) ~= 76
  ASSERT_NEAR(
      sketch1.countBits(),
      bits::countBits(
          reinterpret_cast<const uint64_t*>(hypotheticalBitMap.data()),
          0,
          hypotheticalBitMap.size() * 8),
      100);
}

TEST_F(SfmSketchTest, mergeMixedTest) {
  SfmSketch sketch1 = SfmSketch(&allocator_);
  sketch1.setSketchSize(numberOfBuckets_, precision_);
  SfmSketch sketch2 = SfmSketch(&allocator_);
  sketch2.setSketchSize(numberOfBuckets_, precision_);

  // Add random values to the sketches
  for (int i = 0; i < 100000; i++) {
    sketch1.add(i);
    sketch2.add(-i - 1);
  }

  // sketch1 is non-private, sketch2 is private
  sketch2.enablePrivacy(3, TestingSeededRandomizationStrategy(1));
  auto nonPrivateBitMap1 = getBitSet(sketch1);

  // merge sketch1 with sketch2
  sketch1.mergeWith(sketch2, TestingSeededRandomizationStrategy(2));

  // sketch1 should be private now
  ASSERT_TRUE(sketch1.privacyEnabled());

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
  for (auto i = 0; i < nonPrivateBitMap1.size() * 8; i++) {
    if (!bits::isBitSet(nonPrivateBitMap1.data(), i)) {
      ASSERT_EQ(
          bits::isBitSet(getBitSet(sketch1).data(), i),
          bits::isBitSet(getBitSet(sketch2).data(), i));
    }
  }
}

TEST_F(SfmSketchTest, mergedProbabilityTest) {
  // Symmetric test.
  ASSERT_EQ(
      SfmSketch::mergeRandomizedResponseProbabilities(0.1, 0.4),
      SfmSketch::mergeRandomizedResponseProbabilities(0.4, 0.1));

  // private + nonprivate = private
  ASSERT_EQ(SfmSketch::mergeRandomizedResponseProbabilities(0, 0.1), 0.1);
  ASSERT_EQ(SfmSketch::mergeRandomizedResponseProbabilities(0.15, 0), 0.15);

  // nonprivate + nonprivate = nonprivate
  ASSERT_EQ(SfmSketch::mergeRandomizedResponseProbabilities(0.0, 0.0), 0.0);

  // private + private = private, and the merged noise is higher
  // In particular, according to https://arxiv.org/pdf/2302.02056.pdf,
  // Theorem 4.8, two sketches with epsilon1 and epsilon2 should have a merged
  // epsilonStar of: -log(e^-epsilon1 + e^-epsilon2 - e^-(epsilon1 +
  // epsilon2))
  double epsilon1 = 1.2;
  double epsilon2 = 3.4;
  double p1 = SfmSketch::calculateRandomizedResponseProbability(epsilon1);
  double p2 = SfmSketch::calculateRandomizedResponseProbability(epsilon2);
  double epsilonStar = -std::log(
      std::exp(-epsilon1) + std::exp(-epsilon2) -
      std::exp(-(epsilon1 + epsilon2)));
  double pStar = SfmSketch::calculateRandomizedResponseProbability(epsilonStar);
  ASSERT_NEAR(
      SfmSketch::mergeRandomizedResponseProbabilities(p1, p2), pStar, 1E-6);
  // note: the merged sketch is noisier (higher probability of flipped bits)
  ASSERT_TRUE(pStar > std::max(p1, p2));
}

TEST_F(SfmSketchTest, emptySketchCardinalityTest) {
  auto sketch = SfmSketch(&allocator_);
  sketch.setSketchSize(numberOfBuckets_, precision_);
  // Non-private sketch should have 0 cardinality
  ASSERT_EQ(sketch.cardinality(), 0);

  // Private sketch should have approximately 0 cardinality, but not exactly 0
  sketch.enablePrivacy(3.0, TestingSeededRandomizationStrategy(1));
  ASSERT_NEAR(sketch.cardinality(), 0, 200);
}

TEST_F(SfmSketchTest, smallCardinalityTest) {
  std::vector<int32_t> values = {1, 10, 50, 100, 500, 1000};
  for (auto& value : values) {
    auto nonPrivateSketch = SfmSketch(&allocator_);
    nonPrivateSketch.setSketchSize(numberOfBuckets_, precision_);
    auto privateSketch = SfmSketch(&allocator_);
    privateSketch.setSketchSize(numberOfBuckets_, precision_);
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
    ASSERT_NEAR(privateSketch.cardinality(), value, 200);
  }
}

TEST_F(SfmSketchTest, actualCardinalityTest) {
  std::vector<int32_t> magnitudes = {4, 5, 6};
  std::vector<double> epsilons = {2.0, 4.0, SfmSketch::kNonPrivateEpsilon};
  for (auto& magnitude : magnitudes) {
    auto n = std::pow(10L, magnitude);
    for (auto& epsilon : epsilons) {
      auto privateSketch = SfmSketch(&allocator_);
      privateSketch.setSketchSize(numberOfBuckets_, precision_);

      for (auto i = 0; i < n; i++) {
        privateSketch.add(i);
      }
      privateSketch.enablePrivacy(
          epsilon, TestingSeededRandomizationStrategy(1));
      // Typically, the error is about 5% of the cardinality.
      ASSERT_NEAR(privateSketch.cardinality(), n, n * 0.05);
    }
  }
}

TEST_F(SfmSketchTest, simulatedCardinalityTest) {
  // Instead of creating sketches by adding items, we simulate them for fast
  // testing of huge cardinalities. The goal here is to test general
  // functionality and numerical stability.
  std::vector<int32_t> magnitudes = {7, 8, 9};
  std::vector<double> epsilons = {4, SfmSketch::kNonPrivateEpsilon};
  for (auto& mag : magnitudes) {
    uint64_t n = static_cast<uint64_t>(std::pow(10ULL, mag));
    for (double eps : epsilons) {
      SfmSketch sketch = createSketchWithTargetCardinality(
          numberOfBuckets_, precision_, eps, n, &allocator_);

      ASSERT_NEAR(sketch.cardinality(), n, n * 0.1);
    }
  }
}

TEST_F(SfmSketchTest, mergedCardinalityTest) {
  std::vector<double> epsilons = {3, 4, SfmSketch::kNonPrivateEpsilon};

  // Test each pair of epsilons
  for (auto& eps1 : epsilons) {
    for (auto& eps2 : epsilons) {
      // Create two sketches and add disjoint sets of values (like Java
      // version)
      SfmSketch sketch1 = SfmSketch(&allocator_);
      sketch1.setSketchSize(numberOfBuckets_, precision_);
      SfmSketch sketch2 = SfmSketch(&allocator_);
      sketch2.setSketchSize(numberOfBuckets_, precision_);

      // Add 300,000 positive integers to sketch1 and 200,000 negative
      // integers to sketch2 This ensures disjoint sets with zero overlap
      for (int i = 1; i <= 300000; i++) {
        sketch1.add(i); // positive integers 1 to 300,000
      }
      for (int i = 0; i < 200000; i++) {
        sketch2.add(-i); // negative integers 0 to -199,999
      }

      sketch1.enablePrivacy(eps1, TestingSeededRandomizationStrategy(1));
      sketch2.enablePrivacy(eps2, TestingSeededRandomizationStrategy(2));

      // Merge the sketches
      sketch1.mergeWith(sketch2, TestingSeededRandomizationStrategy(3));

      // The merged sketch should have cardinality 500,000 (300,000 + 200,000
      // disjoint values)
      ASSERT_NEAR(sketch1.cardinality(), 500000, 50000);
    }
  }
}

TEST_F(SfmSketchTest, enablePrivacyTest) {
  SfmSketch sketch = SfmSketch(&allocator_);
  sketch.setSketchSize(numberOfBuckets_, precision_);
  double epsilon = 4;

  for (int32_t i = 0; i < 100000; i++) {
    sketch.add(i);
  }

  uint64_t cardinalityBefore = sketch.cardinality();
  sketch.enablePrivacy(epsilon, TestingSeededRandomizationStrategy(1));
  uint64_t cardinalityAfter = sketch.cardinality();

  // Randomized response probability should reflect the new (private) epsilon
  ASSERT_EQ(
      sketch.getRandomizedResponseProbability(),
      SfmSketch::calculateRandomizedResponseProbability(epsilon));
  ASSERT_TRUE(sketch.privacyEnabled());

  // Cardinality should remain approximately the same
  ASSERT_NEAR(cardinalityAfter, cardinalityBefore, cardinalityBefore * 0.1);
}

} // namespace facebook::velox::functions::aggregate
