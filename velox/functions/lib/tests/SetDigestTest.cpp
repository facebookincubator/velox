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

#include "velox/functions/lib/SetDigest.h"
#include "velox/common/memory/Memory.h"

#include <folly/base64.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>

using namespace facebook::velox::functions;

class SetDigestTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    facebook::velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    allocator_ =
        std::make_unique<facebook::velox::HashStringAllocator>(pool_.get());
  }

  // Helper for Java-equivalent intersection cardinality tests
  void testIntersectionCardinalityHelper(
      int32_t maxHashes1,
      int32_t numBuckets1,
      int32_t maxHashes2,
      int32_t numBuckets2);

  std::string decodeBase64(const std::string& encoded) {
    return folly::base64Decode(encoded);
  }

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
  std::unique_ptr<facebook::velox::HashStringAllocator> allocator_;
};

TEST_F(SetDigestTest, roundTripSerialization) {
  SetDigest digest1(allocator_.get());
  digest1.add(1L);
  digest1.add(1L);
  digest1.add(1L);
  digest1.add(2L);
  digest1.add(2L);

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigest digest2(allocator_.get());
  digest2.deserialize(buffer1.data(), size1);

  int32_t size2 = digest2.estimatedSerializedSize();
  ASSERT_EQ(size1, size2) << "Serialized sizes should match after round-trip";

  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  ASSERT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()))
      << "Serialized data should match after round-trip";

  ASSERT_EQ(digest1.isExact(), digest2.isExact());
  ASSERT_EQ(digest1.cardinality(), digest2.cardinality());
}

TEST_F(SetDigestTest, roundTripWithStrings) {
  SetDigest digest1(allocator_.get());

  digest1.add(facebook::velox::StringView("apple"));
  digest1.add(facebook::velox::StringView("banana"));
  digest1.add(facebook::velox::StringView("apple"));

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigest digest2(allocator_.get());
  digest2.deserialize(buffer1.data(), size1);

  int32_t size2 = digest2.estimatedSerializedSize();
  ASSERT_EQ(size1, size2);

  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  ASSERT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()));
}

TEST_F(SetDigestTest, mergeWithRoundTrip) {
  SetDigest digest1(allocator_.get());
  digest1.add(1L);
  digest1.add(2L);
  digest1.add(3L);

  SetDigest digest2(allocator_.get());
  digest2.add(3L);
  digest2.add(4L);
  digest2.add(5L);

  digest1.mergeWith(digest2);

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigest digest3(allocator_.get());
  digest3.deserialize(buffer1.data(), size1);

  int32_t size2 = digest3.estimatedSerializedSize();
  ASSERT_EQ(size1, size2);

  std::vector<char> buffer2(size2);
  digest3.serialize(buffer2.data());

  ASSERT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()));

  ASSERT_TRUE(digest1.isExact());
  ASSERT_EQ(digest1.cardinality(), 5);
  ASSERT_EQ(digest3.cardinality(), 5);
}

TEST_F(SetDigestTest, mergeWithDuplicates) {
  SetDigest digest1(allocator_.get());
  digest1.add(1L);
  digest1.add(1L);
  digest1.add(1L);

  SetDigest digest2(allocator_.get());
  digest2.add(1L);
  digest2.add(2L);

  digest1.mergeWith(digest2);

  ASSERT_TRUE(digest1.isExact());
  ASSERT_EQ(digest1.cardinality(), 2);

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigest digest3(allocator_.get());
  digest3.deserialize(buffer1.data(), size1);

  int32_t size2 = digest3.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest3.serialize(buffer2.data());

  ASSERT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()));
}

TEST_F(SetDigestTest, javaCompatibility) {
  // Query used: SELECT to_base64(cast(make_set_digest(value) as varbinary))
  // FROM (VALUES 1, 1, 1, 2, 2) T(value);
  SetDigest digest(allocator_.get());

  digest.add(1L);
  digest.add(1L);
  digest.add(1L);
  digest.add(2L);
  digest.add(2L);

  int32_t size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  std::string base64Output =
      folly::base64Encode(folly::StringPiece(buffer.data(), size));

  ASSERT_TRUE(digest.isExact());
  ASSERT_EQ(digest.cardinality(), 2);
  ASSERT_GT(size, 0);
}

TEST_F(SetDigestTest, javaCompatibilityEmptyDigest) {
  // Query: SELECT to_base64(cast(make_set_digest(value) as varbinary))
  // FROM (SELECT NULL as value WHERE 1=0) T(value);
  SetDigest digest(allocator_.get());

  int32_t size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  std::string base64Output =
      folly::base64Encode(folly::StringPiece(buffer.data(), size));

  ASSERT_TRUE(digest.isExact());
  ASSERT_EQ(digest.cardinality(), 0);
}

TEST_F(SetDigestTest, cardinality) {
  SetDigest digest(allocator_.get());

  ASSERT_EQ(digest.cardinality(), 0);

  digest.add(1L);
  ASSERT_EQ(digest.cardinality(), 1);

  digest.add(1L);
  ASSERT_EQ(digest.cardinality(), 1);

  digest.add(2L);
  ASSERT_EQ(digest.cardinality(), 2);
}

TEST_F(SetDigestTest, exactIntersectionCardinality) {
  SetDigest digest1(allocator_.get());
  digest1.add(1L);
  digest1.add(2L);
  digest1.add(3L);

  SetDigest digest2(allocator_.get());
  digest2.add(2L);
  digest2.add(3L);
  digest2.add(4L);

  int64_t intersection =
      SetDigest::exactIntersectionCardinality(digest1, digest2);
  ASSERT_EQ(intersection, 2);

  SetDigest digest3(allocator_.get());
  digest3.add(5L);
  digest3.add(6L);

  intersection = SetDigest::exactIntersectionCardinality(digest1, digest3);
  ASSERT_EQ(intersection, 0);
}

TEST_F(SetDigestTest, jaccardIndex) {
  SetDigest digest1(allocator_.get());
  digest1.add(1L);
  digest1.add(2L);
  digest1.add(3L);

  SetDigest digest2(allocator_.get());
  digest2.add(2L);
  digest2.add(3L);
  digest2.add(4L);

  double jaccard = SetDigest::jaccardIndex(digest1, digest2);
  ASSERT_GE(jaccard, 0.0);
  ASSERT_LE(jaccard, 1.0);

  // Test identical sets - should always be 1.0
  SetDigest digest3(allocator_.get());
  digest3.add(1L);
  digest3.add(2L);
  digest3.add(3L);

  jaccard = SetDigest::jaccardIndex(digest1, digest3);
  ASSERT_NEAR(jaccard, 1.0, 0.001);

  // Test disjoint sets - should be 0.0
  SetDigest digest4(allocator_.get());
  digest4.add(10L);
  digest4.add(20L);
  digest4.add(30L);

  jaccard = SetDigest::jaccardIndex(digest1, digest4);
  ASSERT_NEAR(jaccard, 0.0, 0.001);
}

TEST_F(SetDigestTest, getHashCounts) {
  SetDigest digest(allocator_.get());

  digest.add(1L);
  digest.add(1L);
  digest.add(1L);
  digest.add(2L);
  digest.add(2L);

  auto hashCounts = digest.getHashCounts();
  ASSERT_EQ(hashCounts.size(), 2) << "Should have 2 unique hash values";
  int totalCount = 0;
  for (const auto& entry : hashCounts) {
    totalCount += entry.second;
  }
  ASSERT_EQ(totalCount, 5) << "Total count should be 5";
}

TEST_F(SetDigestTest, hashCountsJavaCompatibility) {
  // This test matches the Java TestSetDigest.testHashCounts() test
  SetDigest digest1(allocator_.get());
  digest1.add(0L);
  digest1.add(0L);
  digest1.add(1L);

  auto hashCounts1 = digest1.getHashCounts();
  ASSERT_EQ(hashCounts1.size(), 2) << "Should have 2 unique hash values";

  std::set<int16_t> counts1;
  for (const auto& entry : hashCounts1) {
    counts1.insert(entry.second);
  }
  std::set<int16_t> expected1 = {1, 2};
  ASSERT_EQ(counts1, expected1)
      << "Java expects hash counts {1, 2} after adding 0, 0, 1";

  SetDigest digest2(allocator_.get());
  digest2.add(0L);
  digest2.add(0L);
  digest2.add(2L);
  digest2.add(2L);

  auto hashCounts2 = digest2.getHashCounts();
  ASSERT_EQ(hashCounts2.size(), 2) << "Should have 2 unique hash values";

  std::set<int16_t> counts2;
  for (const auto& entry : hashCounts2) {
    counts2.insert(entry.second);
  }
  std::set<int16_t> expected2 = {2, 2};
  ASSERT_EQ(counts2, expected2)
      << "Java expects hash counts {2, 2} after adding 0, 0, 2, 2";

  // Merge digest2 into digest1
  digest1.mergeWith(digest2);

  auto hashCountsMerged = digest1.getHashCounts();
  ASSERT_EQ(hashCountsMerged.size(), 3)
      << "After merge should have 3 unique hash values";

  std::set<int16_t> countsMerged;
  for (const auto& entry : hashCountsMerged) {
    countsMerged.insert(entry.second);
  }
  std::set<int16_t> expectedMerged = {1, 2, 4};
  // hash(0): 2 + 2 = 4
  // hash(1): 1 + 0 = 1
  // hash(2): 0 + 2 = 2
  ASSERT_EQ(countsMerged, expectedMerged)
      << "Java expects hash counts {1, 2, 4} after merging";
}

TEST_F(SetDigestTest, javaSerializationCompatibilityIntegerValues) {
  SCOPED_TRACE(
      "SELECT to_base64(cast(make_set_digest(col) as varbinary)) "
      "FROM (VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10)) AS t(col)");

  // Base64 encoded SetDigest generated by Java Presto for values 1-10
  auto data = decodeBase64(
      "ASwAAAACCwoAgANEAEDsyQbAxdQPADQvEoYdNBuBKeEwgtKHOQBYPVsB0hm0gCAI3gAgAAAKAAAAwbZqSBDSGbSowHZsoCAI3krEBfu3A0QAUyYs1XjsyQYbd2yb9sXUD3IeYiw5NC8S5tfq34AdNBuyj38lkynhMHnf8AaM0oc5DItIsjlYPVsBAAEAAQABAAEAAQABAAEAAQABAA==");

  SetDigest digest(allocator_.get());
  digest.deserialize(data.data(), data.size());

  ASSERT_TRUE(digest.isExact()) << "Digest should be exact for 10 values";
  ASSERT_EQ(digest.cardinality(), 10) << "Cardinality should be 10";

  SetDigest expectedDigest(allocator_.get());
  for (int64_t i = 1; i <= 10; i++) {
    expectedDigest.add(i);
  }
  ASSERT_EQ(expectedDigest.cardinality(), 10);
  ASSERT_TRUE(expectedDigest.isExact());

  int64_t intersection =
      SetDigest::exactIntersectionCardinality(digest, expectedDigest);
  ASSERT_EQ(intersection, 10)
      << "Deserialized digest should contain exactly values 1-10";

  auto hashCounts = digest.getHashCounts();
  auto expectedHashCounts = expectedDigest.getHashCounts();
  ASSERT_EQ(hashCounts.size(), expectedHashCounts.size())
      << "Hash count sizes should match";

  for (const auto& entry : hashCounts) {
    ASSERT_EQ(entry.second, 1)
        << "Each value should appear exactly once (count=1)";
  }

  int32_t size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  ASSERT_EQ(size, data.size())
      << "Serialized size should match original Java data size";
  ASSERT_EQ(std::string(buffer.begin(), buffer.end()), data)
      << "Round-trip serialization should produce identical bytes";

  SetDigest digest2(allocator_.get());
  digest2.deserialize(buffer.data(), size);
  ASSERT_EQ(digest2.cardinality(), 10);
  ASSERT_TRUE(digest2.isExact());

  int64_t intersection2 =
      SetDigest::exactIntersectionCardinality(digest2, expectedDigest);
  ASSERT_EQ(intersection2, 10)
      << "Re-deserialized digest should still contain exactly values 1-10";

  int32_t size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  ASSERT_EQ(std::string(buffer2.begin(), buffer2.end()), data)
      << "Multiple round-trips should produce identical bytes";
}

TEST_F(SetDigestTest, javaSerializationCompatibilityStringValues) {
  SCOPED_TRACE(
      "SELECT to_base64(cast(make_set_digest(col) as varbinary)) "
      "FROM (VALUES ('apple'), ('banana'), ('cherry')) AS t(col)");

  SetDigest digest1(allocator_.get());
  digest1.add(facebook::velox::StringView("apple"));
  digest1.add(facebook::velox::StringView("banana"));
  digest1.add(facebook::velox::StringView("cherry"));

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  SetDigest digest2(allocator_.get());
  digest2.deserialize(buffer1.data(), size1);

  ASSERT_TRUE(digest2.isExact()) << "Digest should be exact for 3 values";
  ASSERT_EQ(digest2.cardinality(), 3) << "Cardinality should be 3";

  int32_t size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  ASSERT_EQ(size1, size2) << "Serialized sizes should match";
  ASSERT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()))
      << "Round-trip serialization should produce identical bytes";

  SetDigest digest3(allocator_.get());
  digest3.deserialize(buffer2.data(), size2);
  int32_t size3 = digest3.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest3.serialize(buffer3.data());

  ASSERT_EQ(
      std::string(buffer3.begin(), buffer3.end()),
      std::string(buffer1.begin(), buffer1.end()))
      << "Multiple round-trips should produce identical bytes";
}

TEST_F(SetDigestTest, sparseHllToDenseConversion) {
  SetDigest digest(allocator_.get());

  // digest is still exact (5000 < 8192 maxHashes)
  constexpr int32_t kNumValues = 5000;
  for (int32_t i = 0; i < kNumValues; i++) {
    digest.add(static_cast<int64_t>(i));
  }

  ASSERT_TRUE(digest.isExact())
      << "Digest should be exact for " << kNumValues << " values";
  ASSERT_EQ(digest.cardinality(), kNumValues)
      << "Cardinality should be exact count";

  int32_t size1 = digest.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest.serialize(buffer1.data());

  SetDigest digest2(allocator_.get());
  digest2.deserialize(buffer1.data(), size1);

  ASSERT_TRUE(digest2.isExact()) << "Deserialized digest should still be exact";
  ASSERT_EQ(digest2.cardinality(), kNumValues)
      << "Cardinality should be preserved after deserialization";

  int32_t size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());

  ASSERT_EQ(size1, size2) << "Serialized sizes should match";
  ASSERT_EQ(
      std::string(buffer1.begin(), buffer1.end()),
      std::string(buffer2.begin(), buffer2.end()))
      << "Serialization should be stable after sparse-to-dense conversion";

  digest2.add(static_cast<int64_t>(kNumValues));
  ASSERT_EQ(digest2.cardinality(), kNumValues + 1)
      << "Should be able to add values after deserialization";

  // add enough values to exceed maxHashes (8192) and become approximate
  constexpr int32_t kNumAdditionalValues = 5000;
  for (int32_t i = kNumValues; i < kNumValues + kNumAdditionalValues; i++) {
    digest.add(static_cast<int64_t>(i));
  }

  ASSERT_FALSE(digest.isExact())
      << "Digest should become approximate after exceeding maxHashes";

  int64_t actualCardinality = kNumValues + kNumAdditionalValues;
  int64_t estimatedCardinality = digest.cardinality();
  double errorRate =
      std::abs(static_cast<double>(estimatedCardinality - actualCardinality)) /
      static_cast<double>(actualCardinality);
  ASSERT_LT(errorRate, 0.05)
      << "HLL cardinality estimate should be within 5% of actual. "
      << "Actual: " << actualCardinality
      << ", Estimated: " << estimatedCardinality
      << ", Error: " << (errorRate * 100) << "%";

  int32_t size3 = digest.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest.serialize(buffer3.data());

  SetDigest digest3(allocator_.get());
  digest3.deserialize(buffer3.data(), size3);

  ASSERT_FALSE(digest3.isExact())
      << "Deserialized digest should be approximate";
  ASSERT_EQ(digest3.cardinality(), digest.cardinality())
      << "Cardinality should be preserved for approximate digest";

  int32_t size4 = digest3.estimatedSerializedSize();
  std::vector<char> buffer4(size4);
  digest3.serialize(buffer4.data());

  ASSERT_EQ(size3, size4) << "Approximate digest serialized sizes should match";
  ASSERT_EQ(
      std::string(buffer3.begin(), buffer3.end()),
      std::string(buffer4.begin(), buffer4.end()))
      << "Approximate digest serialization should be stable";
}

// Java-equivalent tests matching TestSetDigest.java

TEST_F(SetDigestTest, testIntersectionCardinality) {
  // Equivalent to Java's testIntersectionCardinality() with default parameters
  testIntersectionCardinalityHelper(
      SetDigest::kDefaultMaxHashes,
      SetDigest::kNumberOfBuckets,
      SetDigest::kDefaultMaxHashes,
      SetDigest::kNumberOfBuckets);
}

TEST_F(SetDigestTest, testUnevenIntersectionCardinality) {
  // Equivalent to Java's testUnevenIntersectionCardinality()
  testIntersectionCardinalityHelper(
      SetDigest::kDefaultMaxHashes / 4,
      SetDigest::kNumberOfBuckets,
      SetDigest::kDefaultMaxHashes,
      SetDigest::kNumberOfBuckets);
}

void SetDigestTest::testIntersectionCardinalityHelper(
    int32_t maxHashes1,
    int32_t numBuckets1,
    int32_t maxHashes2,
    int32_t numBuckets2) {
  std::vector<int32_t> sizes;

  std::mt19937 rand(0); // Same seed as Java for reproducibility
  // Generate random size from each power of ten in [10, 100,000,000]
  for (int32_t i = 10; i < 100000000; i *= 10) {
    std::uniform_int_distribution<int32_t> dist(10, i + 9);
    sizes.push_back(dist(rand));
  }

  for (int32_t size : sizes) {
    int32_t expectedCardinality = 0;
    SetDigest digest1(
        allocator_.get(),
        static_cast<int8_t>(std::log2(numBuckets1)),
        maxHashes1);
    SetDigest digest2(
        allocator_.get(),
        static_cast<int8_t>(std::log2(numBuckets2)),
        maxHashes2);

    std::uniform_int_distribution<int64_t> valueDist;
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    for (int32_t j = 0; j < size; j++) {
      int32_t added = 0;
      int64_t value = valueDist(rand);

      if (probDist(rand) < 0.5) {
        digest1.add(value);
        added++;
      }
      if (probDist(rand) < 0.5) {
        digest2.add(value);
        added++;
      }
      if (added == 2) {
        expectedCardinality++;
      }
    }

    // Skip test if expectedCardinality is too small
    if (expectedCardinality < 10) {
      continue;
    }

    // Calculate intersection using C++ implementation
    int64_t estimatedCardinality;
    if (digest1.isExact() && digest2.isExact()) {
      estimatedCardinality =
          SetDigest::exactIntersectionCardinality(digest1, digest2);
    } else {
      // Use Jaccard index for approximate intersection
      int64_t cardinality1 = digest1.cardinality();
      int64_t cardinality2 = digest2.cardinality();
      double jaccard = SetDigest::jaccardIndex(digest1, digest2);

      // Merge to get union cardinality
      SetDigest tempDigest(
          allocator_.get(),
          static_cast<int8_t>(std::log2(numBuckets1)),
          maxHashes1);
      tempDigest.mergeWith(digest1);
      tempDigest.mergeWith(digest2);
      int64_t unionCardinality = tempDigest.cardinality();

      estimatedCardinality =
          static_cast<int64_t>(std::round(jaccard * unionCardinality));
      estimatedCardinality =
          std::min(estimatedCardinality, std::min(cardinality1, cardinality2));
    }

    double errorRate = std::abs(expectedCardinality - estimatedCardinality) /
        static_cast<double>(expectedCardinality);
    EXPECT_LT(errorRate, 0.10)
        << "Expected intersection cardinality " << expectedCardinality
        << " +/- 10%, got " << estimatedCardinality << ", for set of size "
        << size << " (maxHashes1=" << maxHashes1
        << ", numBuckets1=" << numBuckets1 << ", maxHashes2=" << maxHashes2
        << ", numBuckets2=" << numBuckets2 << ")";
  }
}

TEST_F(SetDigestTest, testSmallLargeIntersections) {
  // Equivalent to Java's testSmallLargeIntersections()
  std::vector<int32_t> sizes;

  std::mt19937 rand(0); // Same seed as Java
  for (int32_t i = 1000; i < 1000000; i *= 10) {
    std::uniform_int_distribution<int32_t> dist(10, i + 9);
    sizes.push_back(dist(rand));
  }

  for (size_t size1_idx = 0; size1_idx < sizes.size(); ++size1_idx) {
    int32_t size1 = sizes[size1_idx];
    SetDigest digest1(allocator_.get());
    std::vector<std::pair<std::unique_ptr<SetDigest>, int32_t>> smallerSets;

    std::uniform_int_distribution<int64_t> valueDist;
    std::uniform_real_distribution<double> probDist(0.0, 1.0);

    for (size_t size2_idx = 0; size2_idx < size1_idx; ++size2_idx) {
      int32_t size2 = sizes[size2_idx];

      for (int32_t overlap = 2; overlap <= 10; overlap += 2) {
        int32_t expectedCardinality = 0;
        auto digest2 = std::make_unique<SetDigest>(allocator_.get());

        // Reset rand for this iteration to generate same values for digest1
        std::mt19937 innerRand(rand());

        for (int32_t j = 0; j < size1; j++) {
          int64_t value = valueDist(innerRand);
          digest1.add(value);

          if (probDist(innerRand) < size2 / static_cast<double>(size1)) {
            if (probDist(innerRand) * 10 < overlap) {
              digest2->add(value);
              expectedCardinality++;
            } else {
              digest2->add(valueDist(innerRand));
            }
          }
        }

        smallerSets.emplace_back(std::move(digest2), expectedCardinality);
      }
    }

    for (const auto& [digest2Ptr, expectedCardinality] : smallerSets) {
      const SetDigest& digest2 = *digest2Ptr;

      // Calculate estimated intersection
      int64_t estIntersectionCardinality;
      if (digest1.isExact() && digest2.isExact()) {
        estIntersectionCardinality =
            SetDigest::exactIntersectionCardinality(digest1, digest2);
      } else {
        int64_t cardinality1 = digest1.cardinality();
        int64_t cardinality2 = digest2.cardinality();
        double jaccard = SetDigest::jaccardIndex(digest1, digest2);

        SetDigest tempDigest(allocator_.get());
        tempDigest.mergeWith(digest1);
        tempDigest.mergeWith(digest2);
        int64_t unionCardinality = tempDigest.cardinality();

        estIntersectionCardinality =
            static_cast<int64_t>(std::round(jaccard * unionCardinality));
        estIntersectionCardinality = std::min(
            estIntersectionCardinality, std::min(cardinality1, cardinality2));
      }

      int64_t size2 = digest2.cardinality();
      EXPECT_LE(estIntersectionCardinality, size2)
          << "Intersection cannot exceed smaller set size";

      double errorRate =
          std::abs(expectedCardinality - estIntersectionCardinality) /
          static_cast<double>(size1);
      EXPECT_LT(errorRate, 0.05)
          << "Expected intersection cardinality " << expectedCardinality
          << ", got " << estIntersectionCardinality << " for size1=" << size1
          << ", size2=" << size2 << ", error rate " << (errorRate * 100) << "%";
    }
  }
}
