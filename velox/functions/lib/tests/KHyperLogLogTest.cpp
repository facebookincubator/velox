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
#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/common/memory/Memory.h"

#include <folly/Random.h>
#include <folly/String.h>
#include <gtest/gtest.h>
#include <set>
#include <unordered_set>
#include "velox/type/Timestamp.h"

using namespace facebook::velox::common::hll;
using namespace facebook::velox::memory;
using namespace facebook::velox;

namespace {
const int32_t kDefaultNumBuckets = 256;
// Theoretical relative standard error formula from the HyperLogLog paper
// (Flajolet et al.): 1.04 / sqrt(num buckets)
const double kDefaultStandardError = 1.04 / std::sqrt(kDefaultNumBuckets);
} // namespace

class KHyperLogLogTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    facebook::velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    hsa_ = std::make_unique<HashStringAllocator>(pool_.get());
    allocator_ = hsa_.get();
  }

  void TearDown() override {
    // Clean up allocator before pool.
    hsa_.reset();
    pool_.reset();
  }

 protected:
  std::shared_ptr<MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> hsa_;
  HashStringAllocator* allocator_{};

  // Helper function to generate random values with quadratic distribution.
  // Squaring a uniform random [0.0, 1.0] heavily skews results toward zero
  // (e.g., ~70% of values fall in the lower half). This simulates realistic
  // data where most values have low cardinality and few have high cardinality,
  // which is critical for testing KHLL's privacy features under conditions
  // where re-identification risk is highest.
  int64_t randomLong(int64_t range) {
    double random = folly::Random::randDouble01();
    return static_cast<int64_t>(std::pow(random, 2.0) * range);
  }

  // Creates a KHyperLogLog with specific values for testing.
  std::unique_ptr<KHyperLogLog<int64_t, HashStringAllocator>> createKHLL(
      const std::vector<int64_t>& values,
      const std::vector<int64_t>& uiis) {
    EXPECT_EQ(values.size(), uiis.size());
    auto khll = std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(
        allocator_);

    for (size_t i = 0; i < values.size(); ++i) {
      khll->add(values[i], uiis[i]);
    }

    return khll;
  }
};

TEST_F(KHyperLogLogTest, basicCardinality) {
  auto khll =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);

  // Empty KHLL should have cardinality 0.
  EXPECT_EQ(0, khll->cardinality());
  EXPECT_TRUE(khll->isExact());

  // Add some values.
  for (int64_t i = 0; i < 100; ++i) {
    khll->add(i, randomLong(100));
  }

  // Should be exact since it is under the default max size.
  EXPECT_TRUE(khll->isExact());
  EXPECT_EQ(100, khll->cardinality());
}

TEST_F(KHyperLogLogTest, cardinalityAccuracy) {
  // Test representative precision levels (low and high) to ensure
  // accuracy across different bucket configurations: 4 (16 buckets) and
  // 12 (4096 buckets).
  const int trials = 30;

  for (int indexBits : {4, 12}) {
    const int numberOfBuckets = 1 << indexBits;
    const int maxCardinality = numberOfBuckets * 2;

    std::vector<double> errors;

    for (int trial = 0; trial < trials; ++trial) {
      auto khll = std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(
          KHyperLogLog<int64_t, HashStringAllocator>::kDefaultMaxSize,
          numberOfBuckets,
          allocator_);

      for (int cardinality = 1; cardinality <= maxCardinality; cardinality++) {
        khll->add(folly::Random::rand64(), 0L);

        // Sample every 20% of bucket count (vs Java's 10%) to reduce test
        // time
        if (cardinality % (numberOfBuckets / 5) == 0) {
          double error =
              (static_cast<double>(khll->cardinality()) - cardinality) /
              cardinality;
          errors.push_back(std::abs(error));
        }
      }
    }

    // Calculate standard deviation
    double mean = 0.0;
    for (double error : errors) {
      mean += error;
    }
    mean /= errors.size();

    double variance = 0.0;
    for (double error : errors) {
      variance += (error - mean) * (error - mean);
    }
    variance /= errors.size();
    double stdDev = std::sqrt(variance);

    EXPECT_LE(stdDev, kDefaultStandardError)
        << "Cardinality mismatch at indexBits " << indexBits << ", bucket "
        << numberOfBuckets;
  }
}

TEST_F(KHyperLogLogTest, mergeWith) {
  auto khll1 = createKHLL(
      std::vector<int64_t>{0, 1, 2, 3, 4, 5},
      std::vector<int64_t>{10, 11, 12, 13, 14, 15});

  auto khll2 = createKHLL(
      std::vector<int64_t>{3, 4, 5, 6, 7, 8},
      std::vector<int64_t>{13, 14, 15, 16, 17, 18});

  auto expected = createKHLL(
      std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8},
      std::vector<int64_t>{10, 11, 12, 13, 14, 15, 16, 17, 18});

  khll1->mergeWith(*khll2);

  EXPECT_EQ(expected->cardinality(), khll1->cardinality());
  EXPECT_EQ(
      expected->reidentificationPotential(10),
      khll1->reidentificationPotential(10));
}

TEST_F(KHyperLogLogTest, merge) {
  // Helpers to create a KHLL with given maxSize and data
  auto createSmaller = [&]() {
    const auto smallerSize = 5;
    auto khll = std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(
        smallerSize, kDefaultNumBuckets, allocator_);
    for (size_t i = 0; i < smallerSize; ++i) {
      khll->add(i, i);
    }
    return khll;
  };

  auto createLarger = [&]() {
    const auto largerSize = 10;
    auto khll = std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(
        largerSize, kDefaultNumBuckets, allocator_);
    for (size_t i = 0; i < largerSize; ++i) {
      khll->add(i, i);
    }
    return khll;
  };

  // Test merge(left, right)
  auto smallerFirst = KHyperLogLog<int64_t, HashStringAllocator>::merge(
      createSmaller(), createLarger());

  // Test merge(right, left) - should produce same result
  auto largerFist = KHyperLogLog<int64_t, HashStringAllocator>::merge(
      createLarger(), createSmaller());

  EXPECT_EQ(smallerFirst->cardinality(), largerFist->cardinality());

  // Explicitly merging smaller into larger should produce different results.
  auto larger = createLarger();
  larger->mergeWith(*createSmaller());
  EXPECT_NE(larger->cardinality(), largerFist->cardinality());
}
namespace {
template <typename TUii>
std::string serialized(KHyperLogLog<TUii, HashStringAllocator>& khll) {
  std::string buffer(khll.estimatedSerializedSize(), '\0');
  khll.serialize(buffer.data());
  return buffer;
}
} // namespace

// The format widens an integral uii to a long, so the declared column width
// must not change the sketch.
TEST_F(KHyperLogLogTest, uiiHashIgnoresIntegerWidth) {
  KHyperLogLog<int64_t, HashStringAllocator> asBigint(allocator_);
  KHyperLogLog<int32_t, HashStringAllocator> asInteger(allocator_);
  KHyperLogLog<int16_t, HashStringAllocator> asSmallint(allocator_);

  for (int32_t i = 0; i < 100; ++i) {
    asBigint.add(int64_t{1}, static_cast<int64_t>(i));
    asInteger.add(int64_t{1}, static_cast<int32_t>(i));
    asSmallint.add(int64_t{1}, static_cast<int16_t>(i));
  }

  EXPECT_EQ(serialized(asBigint), serialized(asInteger));
  EXPECT_EQ(serialized(asBigint), serialized(asSmallint));
}

// The format pre-hashes a varchar uii with XxHash64, so a varchar uii must
// behave like a BIGINT uii holding that hash. Expected values come from:
//
//     SELECT from_big_endian_64(xxhash64(to_utf8(<text>)))
TEST_F(KHyperLogLogTest, uiiVarcharIsPreHashedWithXxHash64) {
  const std::vector<std::pair<std::string, int64_t>> uiis = {
      {"abc", 4'952'883'123'889'572'249LL},
      {"caf\xc3\xa9", -7'331'673'579'364'787'606LL},
      {"aaaaaa\xc3\xa9", 8'051'246'313'758'655'368LL},
  };

  KHyperLogLog<StringView, HashStringAllocator> asVarchar(allocator_);
  KHyperLogLog<int64_t, HashStringAllocator> asBigint(allocator_);

  for (const auto& [text, xxHash] : uiis) {
    asVarchar.add(int64_t{1}, StringView(text));
    asBigint.add(int64_t{1}, xxHash);
  }

  EXPECT_EQ(serialized(asVarchar), serialized(asBigint));
}

namespace {
// A KHyperLogLog produced by a conforming writer of the serialization format.
// Ten join keys, each carrying the same three uii strings rebuilt below.
const char* const kReferenceSketchHex =
    "0100100000000100000A000000A00000001000000010000000100000001000000010"
    "0000001000000010000000100000001000000010000000C1B66A4810D219B4A8C076"
    "6CA02008DE4AC405FBB703440053262CD578ECC9061B776C9BF6C5D40F721E622C39"
    "342F12E6D7EADF801D341BB28F7F259329E13079DFF0068CD287390C8B48B239583D"
    "5B02080300C1A3074DC1AB008500B025CB02080300018AC70A03B2A43603682FCF02"
    "08030042B20A008077C14CC213FD830208030080507B29C029F1AD00D232D1020803"
    "008066C33F8050BF4C0176739A0208030086584C5742285D6F4101CB9D0208030081"
    "BA8552002385A0C386A9A802080300C4A8D542408A4999C06D8FF00208030001A0E0"
    "0201430DA0407273C902080300C037210C40C1E4694185A773";
} // namespace

// The uii hash is part of the serialization format. Two producers that disagree
// on it put the same element in different buckets, and a merge double counts.
TEST_F(KHyperLogLogTest, mergeWithReferenceSketchDoesNotDoubleCount) {
  constexpr int64_t kKeys = 10;
  constexpr int64_t kUiisPerKey = 3;

  std::vector<std::string> uiis;
  for (int64_t key = 1; key <= kKeys; ++key) {
    for (int64_t i = 1; i <= kUiisPerKey; ++i) {
      uiis.push_back(fmt::format("u{}_{}", key, i));
    }
  }

  KHyperLogLog<StringView, HashStringAllocator> sketch(allocator_);
  size_t next = 0;
  for (int64_t key = 1; key <= kKeys; ++key) {
    for (int64_t i = 0; i < kUiisPerKey; ++i) {
      sketch.add(key, StringView(uiis[next++]));
    }
  }

  const auto reference = folly::unhexlify<std::string>(kReferenceSketchHex);
  sketch.mergeWith(StringView(reference), allocator_);

  // The reference covers the same ten keys, so the merge adds none.
  EXPECT_EQ(sketch.cardinality(), kKeys);

  // Every key must still report three distinct uii.
  const auto histogram = sketch.uniquenessDistribution(kKeys);
  EXPECT_NEAR(histogram.at(kUiisPerKey), 1.0, 1e-9);
  EXPECT_NEAR(histogram.at(2 * kUiisPerKey), 0.0, 1e-9);
}

TEST_F(KHyperLogLogTest, serde) {
  // Test small serialization
  std::vector<int64_t> values;
  std::vector<int64_t> uiis;

  for (int64_t i = 0; i < 1000; ++i) {
    values.push_back(i);
    uiis.push_back(randomLong(100));
  }

  auto khll = createKHLL(values, uiis);

  size_t totalSize = khll->estimatedSerializedSize();
  std::string outputBuffer(totalSize, '\0');
  khll->serialize(outputBuffer.data());
  auto deserializedResult =
      KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
          outputBuffer.data(), outputBuffer.size(), allocator_);
  ASSERT_TRUE(deserializedResult.hasValue());
  auto& deserialized = deserializedResult.value();

  EXPECT_EQ(khll->cardinality(), deserialized->cardinality());
  EXPECT_EQ(
      khll->reidentificationPotential(10),
      deserialized->reidentificationPotential(10));

  // Test round-trip
  std::string reserializeBuffer(totalSize, '\0');
  deserialized->serialize(reserializeBuffer.data());
  EXPECT_EQ(outputBuffer, reserializeBuffer);

  // Test empty KHLL round-trip
  auto emptyKhll = createKHLL({}, {});
  size_t emptySize = emptyKhll->estimatedSerializedSize();
  std::string emptyOutputBuffer(emptySize, '\0');
  emptyKhll->serialize(emptyOutputBuffer.data());
  auto deserializedEmptyResult =
      KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
          emptyOutputBuffer.data(), emptyOutputBuffer.size(), allocator_);
  ASSERT_TRUE(deserializedEmptyResult.hasValue());
  auto& deserializedEmpty = deserializedEmptyResult.value();
  EXPECT_EQ(deserializedEmpty->cardinality(), 0);
  std::string reserializedEmptyOutputBuffer(emptySize, '\0');
  deserializedEmpty->serialize(reserializedEmptyOutputBuffer.data());
  EXPECT_EQ(emptyOutputBuffer, reserializedEmptyOutputBuffer);
}

TEST_F(KHyperLogLogTest, uniquenessDistribution) {
  const int histogramSize = 256;
  const int count = 1000;

  auto khll =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);
  std::map<int64_t, std::set<int64_t>> valueToUiis;

  for (int i = 0; i < count; ++i) {
    int64_t uii = randomLong(histogramSize);
    int64_t value = randomLong(count);
    khll->add(value, uii);
    valueToUiis[value].insert(uii);
  }
  auto khllHistogram = khll->uniquenessDistribution(histogramSize);

  // Build the actual histogram
  std::map<int64_t, double> actualHistogram;
  int size = valueToUiis.size();

  for (const auto& [value, uiiSet] : valueToUiis) {
    int64_t bucket = std::min(
        static_cast<int64_t>(uiiSet.size()),
        static_cast<int64_t>(histogramSize));
    actualHistogram[bucket] += 1.0 / size;
  }

  // Verify histogram accuracy (with some tolerance for approximation)
  for (int64_t i = 1; i < histogramSize; ++i) {
    double expected = actualHistogram.count(i) ? actualHistogram[i] : 0.0;
    double khllEstimated = khllHistogram.count(i) ? khllHistogram[i] : 0.0;

    // Use 10% tolerance since the values of uniqueness distribution are a sum
    // of 1 / size of minHash, and not the cardinality estimates.
    // Allow larger tolerance for buckets with very few values.
    double tolerance = std::max(2.0 / size, 0.1 * expected);
    EXPECT_NEAR(khllEstimated, expected, tolerance)
        << "Histogram mismatch at bucket " << i;
  }
}

TEST_F(KHyperLogLogTest, reidentificationPotential) {
  auto khll =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);
  std::map<int64_t, std::set<int64_t>> valueToUiis;

  const int count = 1000;
  for (int i = 0; i < count; ++i) {
    int64_t uii = randomLong(100);
    int64_t value = randomLong(count);
    khll->add(value, uii);
    valueToUiis[value].insert(uii);
  }

  // Test different thresholds
  for (int threshold = 1; threshold < 10; ++threshold) {
    double khllEstimated = khll->reidentificationPotential(threshold);

    // Calculate the actual reidentification potential
    int highlyUniqueCount = 0;
    for (const auto& [value, uiiSet] : valueToUiis) {
      if (static_cast<int>(uiiSet.size()) <= threshold) {
        highlyUniqueCount++;
      }
    }

    double expected =
        static_cast<double>(highlyUniqueCount) / valueToUiis.size();

    if (expected > 0) {
      EXPECT_NEAR(khllEstimated, expected, expected * kDefaultStandardError)
          << "Reidentification potential mismatch for threshold " << threshold;
    }
  }
}

TEST_F(KHyperLogLogTest, exactIntersectionCardinality) {
  auto khll1 = createKHLL(
      std::vector<int64_t>{1, 2, 3, 4, 5},
      std::vector<int64_t>{10, 20, 30, 40, 50});

  auto khll2 = createKHLL(
      std::vector<int64_t>{3, 4, 5, 6, 7},
      std::vector<int64_t>{30, 40, 50, 60, 70});

  EXPECT_TRUE(khll1->isExact());
  EXPECT_TRUE(khll2->isExact());

  int64_t intersection =
      KHyperLogLog<int64_t, HashStringAllocator>::exactIntersectionCardinality(
          *khll1, *khll2);
  // Values 3, 4, 5 are in both
  EXPECT_EQ(3, intersection);
}

TEST_F(KHyperLogLogTest, jaccardIndex) {
  // Test with larger datasets.
  // Create two KHLLs where one is a subset of the other
  const int64_t set1Size = 100000;
  const int64_t set2Size = 150000;

  auto khll1 =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);
  auto khll2 =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);

  // Add values 0 to 99,999 to khll1
  for (int64_t i = 0; i < set1Size; ++i) {
    khll1->add(i, randomLong(100));
  }

  // Add values 0 to 149,999 to khll2 (includes all of khll1)
  for (int64_t i = 0; i < set2Size; ++i) {
    khll2->add(i, randomLong(100));
  }

  double jaccard =
      KHyperLogLog<int64_t, HashStringAllocator>::jaccardIndex(*khll1, *khll2);

  // Expected Jaccard = |intersection| / |union|
  // Intersection: 100,000 (all of set1)
  // Union: 150,000 (all of set2)
  // Jaccard = 100,000 / 150,000 = 2/3 ≈ 0.6667
  double expectedJaccard = static_cast<double>(set1Size) / set2Size;

  EXPECT_NEAR(
      jaccard, expectedJaccard, expectedJaccard * kDefaultStandardError);
}

TEST_F(KHyperLogLogTest, largeDataset) {
  auto khll =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);

  const int count = 200000;
  std::unordered_set<int64_t> uniqueValues;

  for (int i = 0; i < count; ++i) {
    int64_t value = folly::Random::rand64();
    int64_t uii = randomLong(100);
    khll->add(value, uii);
    uniqueValues.insert(value);
  }

  EXPECT_FALSE(khll->isExact());

  int64_t expected = static_cast<int64_t>(uniqueValues.size());
  int64_t khllEstimated = khll->cardinality();

  EXPECT_NEAR(expected, khllEstimated, khllEstimated * kDefaultStandardError);
}

TEST_F(KHyperLogLogTest, minhashSize) {
  auto khll =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);

  EXPECT_EQ(0, khll->minhashSize());

  khll->add(1, 10);
  khll->add(2, 20);
  khll->add(3, 30);

  EXPECT_EQ(3, khll->minhashSize());
}

TEST_F(KHyperLogLogTest, estimatedSizes) {
  auto khll =
      std::make_unique<KHyperLogLog<int64_t, HashStringAllocator>>(allocator_);

  size_t initialEstimatedSerSize = khll->estimatedSerializedSize();

  for (int64_t i = 0; i < 100; ++i) {
    khll->add(i, randomLong(100));
  }

  size_t finalEstimatedSerSize = khll->estimatedSerializedSize();

  // Sizes should increase after adding data.
  EXPECT_GT(finalEstimatedSerSize, initialEstimatedSerSize);

  // Verify the estimated size is accurate.
  std::string serializedBuffer(finalEstimatedSerSize, '\0');
  khll->serialize(serializedBuffer.data());
  size_t actualSerSize = serializedBuffer.size();

  EXPECT_LE(actualSerSize, finalEstimatedSerSize)
      << "Actual serialized size exceeds estimate - potential buffer overflow";

  EXPECT_NEAR(
      actualSerSize,
      finalEstimatedSerSize,
      finalEstimatedSerSize * kDefaultStandardError);
}

TEST_F(KHyperLogLogTest, differentJoinKeyUIITypes) {
  // Test different TJoinKey, TUii combinations:
  // int32_t TJoinKey, int32_t TUii
  {
    auto khll = std::make_unique<KHyperLogLog<int32_t, HashStringAllocator>>(
        allocator_);
    for (int32_t i = 0; i < 100; ++i) {
      khll->add(i, i);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserializedResult =
        KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
            outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserializedResult.hasValue());
    auto& deserialized = deserializedResult.value();
    std::string reserializeBuffer(totalSize, '\0');
    deserialized->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // uint32_t TJoinKey, uint32_t TUii
  {
    auto khll = std::make_unique<KHyperLogLog<uint32_t, HashStringAllocator>>(
        allocator_);
    for (uint32_t i = 0; i < 100; ++i) {
      khll->add(i % 10, i);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(10, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserializedResult =
        KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
            outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserializedResult.hasValue());
    auto& deserialized = deserializedResult.value();
    std::string reserializeBuffer(totalSize, '\0');
    deserialized->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // int16_t TJoinKey, int16_t TUii
  {
    auto khll = std::make_unique<KHyperLogLog<int16_t, HashStringAllocator>>(
        allocator_);
    for (int16_t i = 0; i < 10000; ++i) {
      khll->add(i % 100, i);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // uint16_t TJoinKey, uint16_t TUii
  {
    auto khll = std::make_unique<KHyperLogLog<uint16_t, HashStringAllocator>>(
        allocator_);
    for (uint16_t i = 0; i < 100; ++i) {
      khll->add(i, i);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // int8_t TJoinKey, int8_t TUii
  {
    auto khll =
        std::make_unique<KHyperLogLog<int8_t, HashStringAllocator>>(allocator_);
    for (int8_t i = -100; i < 100; ++i) {
      khll->add(i, i);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(200, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // uint8_t TJoinKey, uint8_t TUii
  {
    auto khll = std::make_unique<KHyperLogLog<uint8_t, HashStringAllocator>>(
        allocator_);
    for (uint8_t i = 0; i < 100; ++i) {
      khll->add(i, i);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // float TJoinKey, float TUii
  {
    auto khll =
        std::make_unique<KHyperLogLog<float, HashStringAllocator>>(allocator_);
    for (int i = 0; i < 10; ++i) {
      khll->add(static_cast<float>(i), static_cast<float>(i));
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(10, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // double TJoinKey, double TUii
  {
    auto khll =
        std::make_unique<KHyperLogLog<double, HashStringAllocator>>(allocator_);
    for (int i = 0; i < 100; ++i) {
      khll->add(static_cast<double>(i), static_cast<double>(i));
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // StringView TJoinKey, StringView TUii
  {
    auto khll = std::make_unique<KHyperLogLog<StringView, HashStringAllocator>>(
        allocator_);
    std::vector<std::string> strings;
    strings.reserve(100);
    for (int i = 0; i < 100; ++i) {
      strings.push_back("key_" + std::to_string(i));
    }
    for (int i = 0; i < 100; ++i) {
      khll->add(StringView(strings[i]), StringView(strings[i]));
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // Timestamp TJoinKey, Timestamp TUii
  {
    auto khll = std::make_unique<KHyperLogLog<Timestamp, HashStringAllocator>>(
        allocator_);
    for (int i = 0; i < 100; ++i) {
      Timestamp ts(i * 1000, 0);
      khll->add(ts, ts);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }

  // int128_t TJoinKey, int128_t TUii
  {
    auto khll = std::make_unique<KHyperLogLog<int128_t, HashStringAllocator>>(
        allocator_);
    for (int i = 0; i < 100; ++i) {
      // Create 128-bit values with different upper and lower 64 bits
      int128_t value = (static_cast<int128_t>(i) << 64) | (i + 1000);
      khll->add(value, value);
    }
    EXPECT_TRUE(khll->isExact());
    EXPECT_EQ(100, khll->cardinality());

    // Test round trip
    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    auto deserialized = KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
        outputBuffer.data(), outputBuffer.size(), allocator_);
    ASSERT_TRUE(deserialized.hasValue());
    std::string reserializeBuffer(totalSize, '\0');
    deserialized.value()->serialize(reserializeBuffer.data());
    EXPECT_EQ(outputBuffer, reserializeBuffer);
  }
}
