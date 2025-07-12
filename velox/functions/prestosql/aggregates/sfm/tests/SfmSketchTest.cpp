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

#include "velox/functions/prestosql/aggregates/sfm/SfmSketch.h"
#include <folly/Benchmark.h>
#include <xxhash.h>
#include "gtest/gtest.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/encode/Base64.h"
#include "velox/common/memory/Memory.h"

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
  const int32_t numBuckets_ = 4096;
  const int32_t precision_ = 24;
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(SfmSketchTest, privacyEnabled) {
  SfmSketch sketch(&allocator_);
  sketch.initialize(numBuckets_, precision_);
  ASSERT_FALSE(sketch.privacyEnabled());

  sketch.enablePrivacy(std::numeric_limits<double>::infinity());
  ASSERT_FALSE(sketch.privacyEnabled());

  sketch.enablePrivacy(1.0);
  ASSERT_TRUE(sketch.privacyEnabled());

  VELOX_ASSERT_THROW(sketch.enablePrivacy(1.0), "privacy is already enabled.");
}

TEST_F(SfmSketchTest, mergeNonPrivacy) {
  auto a = SfmSketch(&allocator_, 1);
  a.initialize(numBuckets_, precision_);
  auto b = SfmSketch(&allocator_, 2);
  b.initialize(numBuckets_, precision_);

  // Add values to the sketches.
  for (int i = 0; i < 1000; i++) {
    a.add(i);
    b.add(i + 1000);
  }

  a.mergeWith(b);

  // Merging two non-private sketches results in a non-private sketch.
  ASSERT_FALSE(a.privacyEnabled());
}

TEST_F(SfmSketchTest, mergeMixed) {
  SfmSketch a = SfmSketch(&allocator_, 1);
  a.initialize(numBuckets_, precision_);
  SfmSketch b = SfmSketch(&allocator_, 2);
  b.initialize(numBuckets_, precision_);

  // Add values to the sketches.
  for (int i = 0; i < 100000; i++) {
    a.add(i);
    b.add(-i - 1);
  }

  // A is non-private, b is private.
  b.enablePrivacy(3.0);
  a.mergeWith(b);

  // A should be private now.
  ASSERT_TRUE(a.privacyEnabled());
}

TEST_F(SfmSketchTest, mergedProbability) {
  // Symmetric test.
  ASSERT_EQ(
      mergeRandomizedResponseProbabilities(0.1, 0.4),
      mergeRandomizedResponseProbabilities(0.4, 0.1));

  // Private + nonprivate = private.
  ASSERT_EQ(mergeRandomizedResponseProbabilities(0, 0.1), 0.1);
  ASSERT_EQ(mergeRandomizedResponseProbabilities(0.15, 0), 0.15);

  // Nonprivate + nonprivate = nonprivate.
  ASSERT_EQ(mergeRandomizedResponseProbabilities(0.0, 0.0), 0.0);

  // Private + private = private, and the merged noise is higher.
  // In particular, according to https://arxiv.org/pdf/2302.02056.pdf,
  // Theorem 4.8, two sketches with epsilon1 and epsilon2 should have a merged
  // epsilonStar of: -log(e ^ -epsilon1 + e ^ -epsilon2 - e ^ -(epsilon1 +
  // epsilon2)).
  const double epsilon1 = 1.2;
  const double epsilon2 = 3.4;
  const double p1 = calculateRandomizedResponseProbability(epsilon1);
  const double p2 = calculateRandomizedResponseProbability(epsilon2);
  const double epsilonStar = -std::log(
      std::exp(-epsilon1) + std::exp(-epsilon2) -
      std::exp(-(epsilon1 + epsilon2)));
  const double pStar = calculateRandomizedResponseProbability(epsilonStar);
  ASSERT_NEAR(mergeRandomizedResponseProbabilities(p1, p2), pStar, 1E-6);
  // Note: the merged sketch is noisier (higher probability of flipped bits).
  ASSERT_TRUE(pStar > std::max(p1, p2));
}

TEST_F(SfmSketchTest, emptySketchCardinality) {
  SfmSketch sketch(&allocator_, 3);
  sketch.initialize(numBuckets_, precision_);

  // Non-private sketch should have 0 cardinality.
  ASSERT_EQ(sketch.cardinality(), 0);

  // Private sketch should have approximately 0 cardinality, but not exactly 0.
  sketch.enablePrivacy(2.0);
  LOG(INFO) << "Cardinality: " << sketch.cardinality();
  ASSERT_NEAR(sketch.cardinality(), 0, 200);
}

TEST_F(SfmSketchTest, smallCardinality) {
  std::vector<int32_t> values = {1, 10, 50, 100, 500, 1000};
  for (auto& value : values) {
    auto nonPrivateSketch = SfmSketch(&allocator_, 1);
    nonPrivateSketch.initialize(numBuckets_, precision_);
    auto privateSketch = SfmSketch(&allocator_, 2);
    privateSketch.initialize(numBuckets_, precision_);

    for (int i = 0; i < value; i++) {
      nonPrivateSketch.add(i);
      privateSketch.add(i);
    }

    // Non-private sketch should have pretty good cardinality estimate.
    ASSERT_NEAR(nonPrivateSketch.cardinality(), value, value * 0.05);

    // Private sketch not as good.
    privateSketch.enablePrivacy(3.0);
    ASSERT_NEAR(
        privateSketch.cardinality(), value, std::max<double>(value * 0.1, 30));
  }
}

TEST_F(SfmSketchTest, actualCardinality) {
  std::vector<int32_t> magnitudes = {4, 5, 6};
  std::vector<double> epsilons = {
      2.0, 4.0, std::numeric_limits<double>::infinity()};
  for (auto& magnitude : magnitudes) {
    const auto n = std::pow(10L, magnitude);
    for (auto& epsilon : epsilons) {
      auto privateSketch = SfmSketch(&allocator_, 1);
      privateSketch.initialize(numBuckets_, precision_);

      for (auto i = 0; i < n; i++) {
        privateSketch.add(i);
      }
      privateSketch.enablePrivacy(epsilon);
      // Typically, the error is about 5% of the cardinality.
      ASSERT_NEAR(privateSketch.cardinality(), n, n * 0.05);
    }
  }
}

TEST_F(SfmSketchTest, mergedCardinality) {
  std::vector<double> epsilons = {
      3, 4, std::numeric_limits<double>::infinity()};

  // Test each pair of epsilons.
  for (auto& eps1 : epsilons) {
    for (auto& eps2 : epsilons) {
      // Create two sketches and add disjoint sets of values
      SfmSketch a = SfmSketch(&allocator_, 1);
      a.initialize(numBuckets_, precision_);
      SfmSketch b = SfmSketch(&allocator_, 2);
      b.initialize(numBuckets_, precision_);

      // Add 300,000 positive integers to a and 200,000 negative
      // integers to b. This ensures disjoint sets with zero overlap.
      for (int i = 1; i <= 300000; i++) {
        a.add(i); // Positive integers 1 to 300,000.
      }
      for (int i = 0; i < 200000; i++) {
        b.add(-i); // Negative integers 0 to -199,999.
      }

      a.enablePrivacy(eps1);
      b.enablePrivacy(eps2);

      // Merge the sketches.
      a.mergeWith(b);

      // The merged sketch should have true cardinality equals 500,000,
      // we allow 5% error.
      ASSERT_NEAR(a.cardinality(), 500000, 500000 * 0.05);
    }
  }
}

TEST_F(SfmSketchTest, serializationRoundTrip) {
  SfmSketch sketch = SfmSketch(&allocator_);
  sketch.initialize(numBuckets_, precision_);

  int32_t numElements = folly::Random::rand32(1000, 10000);
  for (int32_t i = 0; i < numElements; i++) {
    sketch.add(i);
  }

  sketch.enablePrivacy(4.0);
  auto originalCardinality = sketch.cardinality();

  // Allocate buffer for serialization.
  const auto serializedSize = sketch.serializedSize();
  std::vector<char> buffer(serializedSize);
  char* out = buffer.data();

  sketch.serialize(out);
  auto deserialized = SfmSketch::deserialize(out, &allocator_);

  // Test that the deserialized sketch is the same as the original.
  ASSERT_EQ(deserialized.cardinality(), originalCardinality);
}

TEST_F(SfmSketchTest, javaSerializationCompatibility) {
  // This test is to ensure that the C++ serialization format is compatible with
  // the Java serialization format.

  // The string is from a Java sketch serialized with the Java SfmSketch.
  std::string javaSketchBase64 =
      "BwgAAAAQAAAAw8JDHvpqkj//AQAA77//////////////v///////////3////7//"
      "////////////3/////////////////////////////////v//+/+ff/////////"
      "f///v//+//////////////////////7//////3////v//////3////v/f//////"
      "///3////v///////v/f///////////////////////////////f///////v/////"
      "////f///////7////+////9///////////////v//////////////////////////"
      "////////////f////////////3//////////////7////+////////////////////"
      "3///////////////////////9///33///v///3////f//f//9//+//////r/3///"
      "///9//Dfz/e/++vvfV3W//qtq33z//a7///+/m3/fz7/1///1v5hmo/ayTcv3l3Vv"
      "XmnMr+t/B72YloTNgmBy7+IKv+mIBmaBwSO4GwxBMCwWChIoYSVFTAAIIhLylIEj"
      "mJLamBAAooMYLQgikgBAUACEAQYJUAmxMGAhDAoBJGgRgAkBCCCAABAAkIIAAAAA"
      "EQAYEECAQYACgAACEAAQEBAACCAgAAAAQgAAAAAKgAoERABAMAAQbABAAgADEAAQ"
      "IABQ==";

  // Decode base64 to binary data.
  std::string decoded = encoding::Base64::decode(javaSketchBase64);

  auto sketch = SfmSketch::deserialize(decoded.c_str(), &allocator_);
  // Test that the deserialized sketch is the same as the original.
  ASSERT_EQ(
      sketch.cardinality(),
      927499); // 927499 is the expected cardinality of the Java sketch.
}

TEST_F(SfmSketchTest, fullWorkflow) {
  // This test is to test the full workflow of the sketch, including
  // initialize -> add -> enablePrivacy/merge -> cardinality.
  // addIndexAndZeros is the same as add except that it precalculates the
  // index and zeros with the same hash algorithm. We skipped
  // testing it.

  SfmSketch base = SfmSketch(&allocator_, 1);
  base.initialize(numBuckets_, precision_);
  for (int32_t i = 0; i < 1000; i++) {
    base.add(i);
  }
  base.enablePrivacy(8.0);
  // Cardinality should be approximately 1000, typically we would allow
  // 25% variance for real sketch, since we are using
  // deterministic randomization, we hardcode the error to be 20.
  const auto baseCardinality = base.cardinality();
  ASSERT_NEAR(baseCardinality, 1000, 20);

  // Since we can't add elements to the sketch after enabling privacy,
  // we create another sketch and add [0, 1000) to the sketch for several
  // rounds, and then call cardinality. Cardinality should be the same as
  // baseCardinality.
  SfmSketch a = SfmSketch(&allocator_, 1);
  a.initialize(numBuckets_, precision_);
  for (int j = 0; j < 10; j++) {
    for (int32_t i = 0; i < 1000; i++) {
      a.add(i);
    }
  }
  a.enablePrivacy(8.0);
  ASSERT_EQ(baseCardinality, a.cardinality());

  // Now we create a sketch and add [0, 1000) to the sketch in random order,
  // and then call cardinality. Cardinality should be the same as
  // baseCardinality.
  SfmSketch b = SfmSketch(&allocator_, 1);
  b.initialize(numBuckets_, precision_);
  // We make sure that 0 - 999 are all present in the sketch, but in random
  // order.
  std::vector<int32_t> values;
  values.reserve(1000);
  for (int32_t i = 0; i < 1000; i++) {
    values.emplace_back(i);
  }
  std::shuffle(values.begin(), values.end(), folly::ThreadLocalPRNG());
  for (int32_t i = 0; i < 1000; i++) {
    b.add(values[i]);
  }

  // Then we add 100 random duplicate values to the sketch.
  for (int32_t i = 0; i < 100; i++) {
    b.add(folly::Random::rand32(1000));
  }

  b.enablePrivacy(8.0);
  ASSERT_EQ(baseCardinality, b.cardinality());

  // Now we create three sketches and separate [0, 1000) to three groups and
  // add them to the three sketches respectively. Then we merge the three
  // sketches and call cardinality. Cardinality should be the same as
  // baseCardinality.
  SfmSketch c = SfmSketch(&allocator_, 1);
  c.initialize(numBuckets_, precision_);
  SfmSketch d = SfmSketch(&allocator_, 1);
  d.initialize(numBuckets_, precision_);
  SfmSketch e = SfmSketch(&allocator_, 1);
  e.initialize(numBuckets_, precision_);
  for (int32_t i = 0; i < 1000; i++) {
    if (i % 3 == 0) {
      c.add(i);
    } else if (i % 3 == 1) {
      d.add(i);
    } else {
      e.add(i);
    }
  }
  // Merge before enabling privacy.
  c.mergeWith(d);
  c.mergeWith(e);
  c.enablePrivacy(8.0);
  ASSERT_EQ(baseCardinality, c.cardinality());

  // Serialize and deserialize sketch c, and test that the deserialized sketch
  // has the same cardinality as the baseCardinality.
  const auto serializedSize = c.serializedSize();
  std::vector<char> buffer(serializedSize);
  char* out = buffer.data();
  c.serialize(out);
  auto deserialized = SfmSketch::deserialize(out, &allocator_);
  ASSERT_EQ(baseCardinality, deserialized.cardinality());

  // Now enable privacy for d and e and merge them into c.
  // There should be noise added to the merged sketch.
  d.enablePrivacy(8.0);
  e.enablePrivacy(8.0);
  c.mergeWith(d);
  c.mergeWith(e);
  ASSERT_NE(baseCardinality, c.cardinality());
  ASSERT_NEAR(baseCardinality, c.cardinality(), baseCardinality * 0.05);
}
} // namespace facebook::velox::functions::aggregate
