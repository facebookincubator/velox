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

#include <folly/base64.h>
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/SetDigest.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/SetDigestRegistration.h"
#include "velox/functions/prestosql/types/SetDigestType.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;

class SetDigestFunctionsTest : public FunctionBaseTest {
 protected:
  void SetUp() override {
    FunctionBaseTest::SetUp();
    registerSetDigestType();
  }
};

TEST_F(SetDigestFunctionsTest, testCardinality) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", SETDIGEST(), input);
  };

  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());
  facebook::velox::functions::SetDigest<int64_t> digest(&allocator);

  std::vector<int64_t> values;
  for (int64_t i = 1; i <= 100; i++) {
    digest.add(i);
    values.push_back(i);
  }

  int64_t expectedCardinality = digest.cardinality();
  ASSERT_EQ(100, expectedCardinality);

  int32_t serializedSize = digest.estimatedSerializedSize();
  std::vector<char> buffer(serializedSize);
  digest.serialize(buffer.data());
  std::string serializedDigest(buffer.begin(), buffer.end());

  auto result = cardinality(serializedDigest);
  ASSERT_EQ(expectedCardinality, result.value());

  // Test with duplicates - should not increase cardinality
  facebook::velox::functions::SetDigest<int64_t> digestWithDuplicates(
      &allocator);
  digestWithDuplicates.add(1L);
  digestWithDuplicates.add(1L);
  digestWithDuplicates.add(1L);
  digestWithDuplicates.add(2L);
  digestWithDuplicates.add(2L);

  expectedCardinality = digestWithDuplicates.cardinality();
  ASSERT_EQ(2, expectedCardinality);

  serializedSize = digestWithDuplicates.estimatedSerializedSize();
  buffer.resize(serializedSize);
  digestWithDuplicates.serialize(buffer.data());
  serializedDigest = std::string(buffer.begin(), buffer.end());

  result = cardinality(serializedDigest);
  ASSERT_EQ(expectedCardinality, result.value());
}

TEST_F(SetDigestFunctionsTest, testIntersectionCardinality) {
  const auto intersectionCardinality =
      [&](const std::optional<std::string>& input1,
          const std::optional<std::string>& input2) {
        return evaluateOnce<int64_t>(
            "intersection_cardinality(c0, c1)",
            {SETDIGEST(), SETDIGEST()},
            input1,
            input2);
      };

  // Create two SetDigest objects in C++ with known overlap
  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());

  facebook::velox::functions::SetDigest<int64_t> digest1(&allocator);
  for (int64_t i = 1; i <= 10; i++) {
    digest1.add(i);
  }

  facebook::velox::functions::SetDigest<int64_t> digest2(&allocator);
  for (int64_t i = 5; i <= 15; i++) {
    digest2.add(i);
  }

  int64_t expectedIntersection = facebook::velox::functions::SetDigest<
      int64_t>::exactIntersectionCardinality(digest1, digest2);
  ASSERT_EQ(6, expectedIntersection);

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());
  std::string serialized1(buffer1.begin(), buffer1.end());

  int32_t size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());
  std::string serialized2(buffer2.begin(), buffer2.end());

  auto result = intersectionCardinality(serialized1, serialized2);
  ASSERT_EQ(expectedIntersection, result.value());

  // Test with no overlap
  facebook::velox::functions::SetDigest<int64_t> digest3(&allocator);
  for (int64_t i = 100; i <= 110; i++) {
    digest3.add(i);
  }

  expectedIntersection = facebook::velox::functions::SetDigest<
      int64_t>::exactIntersectionCardinality(digest1, digest3);
  ASSERT_EQ(0, expectedIntersection);

  int32_t size3 = digest3.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest3.serialize(buffer3.data());
  std::string serialized3(buffer3.begin(), buffer3.end());

  result = intersectionCardinality(serialized1, serialized3);
  ASSERT_EQ(expectedIntersection, result.value());

  // Test with full overlap (same digest)
  expectedIntersection = facebook::velox::functions::SetDigest<
      int64_t>::exactIntersectionCardinality(digest1, digest1);
  result = intersectionCardinality(serialized1, serialized1);
  ASSERT_EQ(expectedIntersection, result.value());

  // Test with null inputs.
  result = intersectionCardinality(std::nullopt, serialized1);
  ASSERT_FALSE(result.has_value());

  result = intersectionCardinality(serialized1, std::nullopt);
  ASSERT_FALSE(result.has_value());

  result = intersectionCardinality(std::nullopt, std::nullopt);
  ASSERT_FALSE(result.has_value());
}

TEST_F(SetDigestFunctionsTest, testHashCounts) {
  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());
  facebook::velox::functions::SetDigest<int64_t> digest(&allocator);

  digest.add(1L);
  digest.add(1L);
  digest.add(1L);
  digest.add(2L);
  digest.add(2L);
  digest.add(3L);

  auto expectedHashCounts = digest.getHashCounts();
  ASSERT_EQ(3, expectedHashCounts.size());

  int32_t size = digest.estimatedSerializedSize();
  std::vector<char> buffer(size);
  digest.serialize(buffer.data());

  auto inputVector = BaseVector::create(SETDIGEST(), 1, execCtx_.pool());
  auto flatVector = inputVector->as<FlatVector<StringView>>();
  flatVector->set(0, StringView(buffer.data(), size));

  auto result = evaluate("hash_counts(c0)", makeRowVector({inputVector}));

  ASSERT_EQ(result->size(), 1);
  auto mapVector = result->as<MapVector>();
  ASSERT_NE(mapVector, nullptr);

  ASSERT_EQ(mapVector->sizeAt(0), expectedHashCounts.size());

  auto keys = mapVector->mapKeys()->as<SimpleVector<int64_t>>();
  auto values = mapVector->mapValues()->as<SimpleVector<int16_t>>();

  auto offset = mapVector->offsetAt(0);
  auto mapSize = mapVector->sizeAt(0);

  for (auto i = 0; i < mapSize; i++) {
    auto key = keys->valueAt(offset + i);
    auto value = values->valueAt(offset + i);

    ASSERT_TRUE(expectedHashCounts.count(key) > 0);
    ASSERT_EQ(expectedHashCounts.at(key), value);
  }
}

TEST_F(SetDigestFunctionsTest, testHashCountsWithMerge) {
  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());
  facebook::velox::functions::SetDigest<int64_t> digest1(&allocator);
  digest1.add(0L);
  digest1.add(0L);
  digest1.add(1L);

  facebook::velox::functions::SetDigest<int64_t> digest2(&allocator);
  digest2.add(0L);
  digest2.add(0L);
  digest2.add(2L);
  digest2.add(2L);

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());

  auto inputVector1 = BaseVector::create(SETDIGEST(), 1, execCtx_.pool());
  auto flatVector1 = inputVector1->as<FlatVector<StringView>>();
  flatVector1->set(0, StringView(buffer1.data(), size1));

  auto result1 = evaluate("hash_counts(c0)", makeRowVector({inputVector1}));
  auto mapVector1 = result1->as<MapVector>();
  ASSERT_NE(mapVector1, nullptr);

  // Extract count values from digest1 (should be {1, 2})
  auto values1 = mapVector1->mapValues()->as<SimpleVector<int16_t>>();
  auto offset1 = mapVector1->offsetAt(0);
  auto mapSize1 = mapVector1->sizeAt(0);

  std::set<int16_t> countValues1;
  for (auto i = 0; i < mapSize1; i++) {
    countValues1.insert(values1->valueAt(offset1 + i));
  }

  std::set<int16_t> expected1 = {1, 2}; // hash(0)→2, hash(1)→1
  ASSERT_EQ(countValues1, expected1);

  digest1.mergeWith(digest2);

  // Test hash_counts after merge
  int32_t sizeMerged = digest1.estimatedSerializedSize();
  std::vector<char> bufferMerged(sizeMerged);
  digest1.serialize(bufferMerged.data());

  auto inputVectorMerged = BaseVector::create(SETDIGEST(), 1, execCtx_.pool());
  auto flatVectorMerged = inputVectorMerged->as<FlatVector<StringView>>();
  flatVectorMerged->set(0, StringView(bufferMerged.data(), sizeMerged));

  auto resultMerged =
      evaluate("hash_counts(c0)", makeRowVector({inputVectorMerged}));
  auto mapVectorMerged = resultMerged->as<MapVector>();
  ASSERT_NE(mapVectorMerged, nullptr);

  // Extract count values from merged digest (should be {1, 2, 4})
  auto valuesMerged = mapVectorMerged->mapValues()->as<SimpleVector<int16_t>>();
  auto offsetMerged = mapVectorMerged->offsetAt(0);
  auto mapSizeMerged = mapVectorMerged->sizeAt(0);

  std::set<int16_t> countValuesMerged;
  for (auto i = 0; i < mapSizeMerged; i++) {
    countValuesMerged.insert(valuesMerged->valueAt(offsetMerged + i));
  }

  std::set<int16_t> expectedMerged = {
      1, 2, 4}; // hash(0)→4, hash(1)→1, hash(2)→2
  ASSERT_EQ(countValuesMerged, expectedMerged);
}

TEST_F(SetDigestFunctionsTest, testIntersectionCardinalityApproximate) {
  const auto intersectionCardinality =
      [&](const std::optional<std::string>& input1,
          const std::optional<std::string>& input2) {
        return evaluateOnce<int64_t>(
            "intersection_cardinality(c0, c1)",
            {SETDIGEST(), SETDIGEST()},
            input1,
            input2);
      };

  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());

  // Test 1: Both digests approximate
  // Approximate digest 1: values 0-9999 (10000 elements)
  facebook::velox::functions::SetDigest<int64_t> approximateDigest1(&allocator);
  for (int64_t i = 0; i < 10000; i++) {
    approximateDigest1.add(i);
  }
  ASSERT_FALSE(approximateDigest1.isExact());

  // Approximate digest 2: values 5000-14999 (10000 elements)
  // Intersection: 5000-9999 (5000 elements)
  // Union: 0-14999 (15000 elements)
  // Expected Jaccard: 5000/15000 = 0.333
  facebook::velox::functions::SetDigest<int64_t> approximateDigest2(&allocator);
  for (int64_t i = 5000; i < 15000; i++) {
    approximateDigest2.add(i);
  }
  ASSERT_FALSE(approximateDigest2.isExact());

  int32_t approx1Size = approximateDigest1.estimatedSerializedSize();
  std::vector<char> approx1Buffer(approx1Size);
  approximateDigest1.serialize(approx1Buffer.data());
  std::string serializedApprox1(approx1Buffer.begin(), approx1Buffer.end());

  int32_t approx2Size = approximateDigest2.estimatedSerializedSize();
  std::vector<char> approx2Buffer(approx2Size);
  approximateDigest2.serialize(approx2Buffer.data());
  std::string serializedApprox2(approx2Buffer.begin(), approx2Buffer.end());

  // Test approximate mode with both digests being approximate
  // approximateDigest1: 0-9999, approximateDigest2: 5000-14999
  // Expected intersection: 5000 values (5000-9999)
  auto result = intersectionCardinality(serializedApprox1, serializedApprox2);
  ASSERT_TRUE(result.has_value());
  // Allow reasonable margin of error for HyperLogLog and MinHash estimation
  EXPECT_GE(result.value(), 3500);
  EXPECT_LE(result.value(), 6500);

  // Test approximate mode with same digest (self-intersection)
  // Should return cardinality of the digest
  result = intersectionCardinality(serializedApprox1, serializedApprox1);
  ASSERT_TRUE(result.has_value());
  int64_t cardinality = approximateDigest1.cardinality();
  EXPECT_EQ(result.value(), cardinality);

  // Test 2: Mixed mode - Exact digest (1000 elements) with Approximate digest
  // This tests that MinHash can detect overlap when exact set is reasonably
  // sized. Exact digest: values 0-999 (1000 elements, stays exact since < 8192)
  facebook::velox::functions::SetDigest<int64_t> exactDigest(&allocator);
  for (int64_t i = 0; i < 1000; i++) {
    exactDigest.add(i);
  }
  ASSERT_TRUE(exactDigest.isExact());

  // Approximate digest 3: values 0-9999 (10000 elements)
  // Contains all elements from exactDigest
  // Intersection: 0-999 (1000 elements)
  // Union: 0-9999 (10000 elements)
  // Expected Jaccard: 1000/10000 = 0.1
  facebook::velox::functions::SetDigest<int64_t> approximateDigest3(&allocator);
  for (int64_t i = 0; i < 10000; i++) {
    approximateDigest3.add(i);
  }
  ASSERT_FALSE(approximateDigest3.isExact());

  int32_t exactSize = exactDigest.estimatedSerializedSize();
  std::vector<char> exactBuffer(exactSize);
  exactDigest.serialize(exactBuffer.data());
  std::string serializedExact(exactBuffer.begin(), exactBuffer.end());

  int32_t approx3Size = approximateDigest3.estimatedSerializedSize();
  std::vector<char> approx3Buffer(approx3Size);
  approximateDigest3.serialize(approx3Buffer.data());
  std::string serializedApprox3(approx3Buffer.begin(), approx3Buffer.end());

  // Test mixed mode: exact (1000) ∩ approximate (10000 containing all exact)
  result = intersectionCardinality(serializedExact, serializedApprox3);
  ASSERT_TRUE(result.has_value());
  // With 1000 elements, MinHash should be able to estimate the overlap
  // Expected ~1000, but allow wider margin due to MinHash estimation variance
  EXPECT_GE(result.value(), 600);
  EXPECT_LE(result.value(), 1400);
}

TEST_F(SetDigestFunctionsTest, testJaccardIndex) {
  const auto jaccardIndex = [&](const std::optional<std::string>& input1,
                                const std::optional<std::string>& input2) {
    return evaluateOnce<double>(
        "jaccard_index(c0, c1)", {SETDIGEST(), SETDIGEST()}, input1, input2);
  };

  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());

  // Test 1: Identical sets - Jaccard index should be 1.0
  facebook::velox::functions::SetDigest<int64_t> digest1(&allocator);
  for (int64_t i = 1; i <= 100; i++) {
    digest1.add(i);
  }

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());
  std::string serialized1(buffer1.begin(), buffer1.end());

  auto result = jaccardIndex(serialized1, serialized1);
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result.value(), 1.0);

  // Test 2: Disjoint sets - Jaccard index should be 0.0
  facebook::velox::functions::SetDigest<int64_t> digest2(&allocator);
  for (int64_t i = 101; i <= 200; i++) {
    digest2.add(i);
  }

  int32_t size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());
  std::string serialized2(buffer2.begin(), buffer2.end());

  result = jaccardIndex(serialized1, serialized2);
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result.value(), 0.0);

  // Test 3: Partially overlapping sets
  // digest1: 1-100, digest3: 50-150
  // Intersection: 50-100 (51 elements)
  // Union: 1-150 (150 elements)
  // Expected Jaccard: 51/150 ≈ 0.34
  facebook::velox::functions::SetDigest<int64_t> digest3(&allocator);
  for (int64_t i = 50; i <= 150; i++) {
    digest3.add(i);
  }

  int32_t size3 = digest3.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest3.serialize(buffer3.data());
  std::string serialized3(buffer3.begin(), buffer3.end());

  result = jaccardIndex(serialized1, serialized3);
  ASSERT_TRUE(result.has_value());
  // Exact expected value: 51/150 = 0.34
  double expectedJaccard = 51.0 / 150.0;
  EXPECT_NEAR(result.value(), expectedJaccard, 0.05);

  // Test 4: One set is a subset of the other
  // digest1: 1-100, digest4: 25-75 (subset of digest1)
  // Intersection: 25-75 (51 elements)
  // Union: 1-100 (100 elements)
  // Expected Jaccard: 51/100 = 0.51
  facebook::velox::functions::SetDigest<int64_t> digest4(&allocator);
  for (int64_t i = 25; i <= 75; i++) {
    digest4.add(i);
  }

  int32_t size4 = digest4.estimatedSerializedSize();
  std::vector<char> buffer4(size4);
  digest4.serialize(buffer4.data());
  std::string serialized4(buffer4.begin(), buffer4.end());

  result = jaccardIndex(serialized1, serialized4);
  ASSERT_TRUE(result.has_value());
  expectedJaccard = 51.0 / 100.0;
  EXPECT_NEAR(result.value(), expectedJaccard, 0.05);

  // Test 5: Small overlap
  // digest1: 1-100, digest5: 95-200
  // Intersection: 95-100 (6 elements)
  // Union: 1-200 (200 elements)
  // Expected Jaccard: 6/200 = 0.03
  facebook::velox::functions::SetDigest<int64_t> digest5(&allocator);
  for (int64_t i = 95; i <= 200; i++) {
    digest5.add(i);
  }

  int32_t size5 = digest5.estimatedSerializedSize();
  std::vector<char> buffer5(size5);
  digest5.serialize(buffer5.data());
  std::string serialized5(buffer5.begin(), buffer5.end());

  result = jaccardIndex(serialized1, serialized5);
  ASSERT_TRUE(result.has_value());
  expectedJaccard = 6.0 / 200.0;
  EXPECT_NEAR(result.value(), expectedJaccard, 0.05);

  // Test with null inputs.
  result = jaccardIndex(std::nullopt, serialized1);
  ASSERT_FALSE(result.has_value());

  result = jaccardIndex(serialized1, std::nullopt);
  ASSERT_FALSE(result.has_value());

  result = jaccardIndex(std::nullopt, std::nullopt);
  ASSERT_FALSE(result.has_value());
}

TEST_F(SetDigestFunctionsTest, testJaccardIndexApproximate) {
  const auto jaccardIndex = [&](const std::optional<std::string>& input1,
                                const std::optional<std::string>& input2) {
    return evaluateOnce<double>(
        "jaccard_index(c0, c1)", {SETDIGEST(), SETDIGEST()}, input1, input2);
  };

  auto pool = memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator(pool.get());

  // Create large approximate digests
  // digest1: 0-14999 (15000 elements)
  facebook::velox::functions::SetDigest<int64_t> digest1(&allocator);
  for (int64_t i = 0; i < 15000; i++) {
    digest1.add(i);
  }
  ASSERT_FALSE(digest1.isExact());

  // digest2: 5000-19999 (15000 elements)
  // Intersection: 5000-14999 (10000 elements)
  // Union: 0-19999 (20000 elements)
  // Expected Jaccard: 10000/20000 = 0.5
  facebook::velox::functions::SetDigest<int64_t> digest2(&allocator);
  for (int64_t i = 5000; i < 20000; i++) {
    digest2.add(i);
  }
  ASSERT_FALSE(digest2.isExact());

  int32_t size1 = digest1.estimatedSerializedSize();
  std::vector<char> buffer1(size1);
  digest1.serialize(buffer1.data());
  std::string serialized1(buffer1.begin(), buffer1.end());

  int32_t size2 = digest2.estimatedSerializedSize();
  std::vector<char> buffer2(size2);
  digest2.serialize(buffer2.data());
  std::string serialized2(buffer2.begin(), buffer2.end());

  auto result = jaccardIndex(serialized1, serialized2);
  ASSERT_TRUE(result.has_value());
  // Expected value is 0.5, allow reasonable margin for approximation
  EXPECT_GE(result.value(), 0.0);
  EXPECT_LE(result.value(), 1.0);
  EXPECT_NEAR(result.value(), 0.5, 0.15);

  // Test with same approximate digest (self-similarity)
  result = jaccardIndex(serialized1, serialized1);
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result.value(), 1.0);

  // Test with disjoint approximate sets
  // digest3: 30000-44999 (15000 elements, no overlap with digest1)
  facebook::velox::functions::SetDigest<int64_t> digest3(&allocator);
  for (int64_t i = 30000; i < 45000; i++) {
    digest3.add(i);
  }
  ASSERT_FALSE(digest3.isExact());

  int32_t size3 = digest3.estimatedSerializedSize();
  std::vector<char> buffer3(size3);
  digest3.serialize(buffer3.data());
  std::string serialized3(buffer3.begin(), buffer3.end());

  result = jaccardIndex(serialized1, serialized3);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result.value(), 0.0, 0.1);
}
