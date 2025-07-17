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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/SfmSketchType.h"

namespace facebook::velox::functions::test {
using SfmSketch = facebook::velox::functions::aggregate::SfmSketch;

class SfmSketchFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(SfmSketchFunctionsTest, cardinalitySignatures) {
  auto signatures = getSignatureStrings("cardinality");
  ASSERT_GE(signatures.size(), 1);
  ASSERT_EQ(1, signatures.count("(sfmsketch) -> bigint"));
}

TEST_F(SfmSketchFunctionsTest, noisyEmptyApproxSetSfmSignatures) {
  auto signatures = getSignatureStrings("noisy_empty_approx_set_sfm");
  ASSERT_EQ(3, signatures.size());
  ASSERT_EQ(1, signatures.count("(constant double) -> sfmsketch"));
  ASSERT_EQ(
      1, signatures.count("(constant double,constant bigint) -> sfmsketch"));
  ASSERT_EQ(
      1,
      signatures.count(
          "(constant double,constant bigint,constant bigint) -> sfmsketch"));
}

TEST_F(SfmSketchFunctionsTest, mergeSfmSketchSignatures) {
  auto signatures = getSignatureStrings("merge_sfm");
  ASSERT_EQ(1, signatures.size());
  ASSERT_EQ(1, signatures.count("(array(sfmsketch)) -> sfmsketch"));
}

TEST_F(SfmSketchFunctionsTest, cardinalityTest) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", SFMSKETCH(), input);
  };

  // Test empty sketch
  SfmSketch sketch{&allocator_};
  sketch.initialize(4096, 24);

  auto serializedSize = sketch.serializedSize();

  std::vector<char> buffer(serializedSize);
  sketch.serialize(buffer.data());
  std::string serializedEmpty(buffer.begin(), buffer.end());
  EXPECT_EQ(*cardinality(serializedEmpty), 0);

  // Add distinct elements
  int64_t numElements = 100000;
  for (int64_t i = 0; i < numElements; i++) {
    sketch.add(i);
  }
  // Add noise to the sketch.
  sketch.enablePrivacy(8.0);

  std::vector<char> buffer2(sketch.serializedSize());
  sketch.serialize(buffer2.data());
  std::string serialized2(buffer2.begin(), buffer2.end());
  auto result = cardinality(serialized2);

  // SfmSketch is an approximate algorithm, so we allow 25% error.
  EXPECT_NEAR(*result, numElements, numElements * 0.25);
}

TEST_F(SfmSketchFunctionsTest, cardinalityWithDuplicates) {
  const auto cardinality = [&](const std::optional<std::string>& input) {
    return evaluateOnce<int64_t>("cardinality(c0)", SFMSKETCH(), input);
  };

  SfmSketch sketch{&allocator_};
  sketch.initialize(4096, 24);

  for (int i = 0; i < 100000; i++) {
    sketch.add(i % 100);
  }

  std::vector<char> buffer(sketch.serializedSize());
  sketch.serialize(buffer.data());
  std::string serialized(buffer.begin(), buffer.end());
  auto result = cardinality(serialized);

  // It turns out that XXH64 has a collision for numbers 0-99,
  // The cardinality using XXH64 is 99, while murmurHash is 100,
  EXPECT_EQ(*result, 99);
}

TEST_F(SfmSketchFunctionsTest, noisyEmptyApproxSetSfm) {
  // Test with epsilon only
  auto cardinality1 = evaluateOnce<int64_t>(
      "cardinality(noisy_empty_approx_set_sfm(INFINITY()))",
      makeRowVector(ROW({}), 1));
  ASSERT_EQ(*cardinality1, 0);

  // Test with epsilon and buckets
  auto cardinality2 = evaluateOnce<int64_t>(
      "cardinality(noisy_empty_approx_set_sfm(INFINITY(), 1024))",
      makeRowVector(ROW({}), 1));
  ASSERT_EQ(*cardinality2, 0);

  // Test with epsilon, buckets, and precision
  auto cardinality3 = evaluateOnce<int64_t>(
      "cardinality(noisy_empty_approx_set_sfm(INFINITY(), 1024, 20))",
      makeRowVector(ROW({}), 1));
  ASSERT_EQ(*cardinality3, 0);
}

TEST_F(SfmSketchFunctionsTest, mergeSfmSketchArray) {
  // Create two sketches with different elements
  SfmSketch sketch1{&allocator_};
  sketch1.initialize(4096, 24);
  for (int i = 0; i < 50; i++) {
    sketch1.add(i);
  }

  SfmSketch sketch2{&allocator_};
  sketch2.initialize(4096, 24);
  for (int i = 25; i < 75; i++) {
    sketch2.add(i);
  }

  std::vector<char> buffer1(sketch1.serializedSize());
  sketch1.serialize(buffer1.data());
  std::string serialized1(buffer1.begin(), buffer1.end());
  std::vector<char> buffer2(sketch2.serializedSize());
  sketch2.serialize(buffer2.data());
  std::string serialized2(buffer2.begin(), buffer2.end());

  // Test merging array of sketches
  auto mergedResult = evaluateOnce<std::string>(
      "merge_sfm(c0)",
      makeRowVector({makeArrayVector<std::string>(
          {{serialized1, serialized2}}, SFMSKETCH())}));

  // Check cardinality of merged sketch
  auto mergedCardinality =
      evaluateOnce<int64_t>("cardinality(c0)", SFMSKETCH(), mergedResult);

  // There are 75 distinct elements in the sketch.
  ASSERT_EQ(mergedCardinality, 75);
}

TEST_F(SfmSketchFunctionsTest, mergeSfmSketchEmpty) {
  // Test merging empty array should return null
  auto result = evaluate(
      "merge_sfm(c0)",
      makeRowVector({makeArrayVector<std::string>({}, SFMSKETCH())}));

  EXPECT_TRUE(result->isNullAt(0));
}

TEST_F(SfmSketchFunctionsTest, cardinalityNull) {
  // Test cardinality with null input
  auto result = evaluate(
      "cardinality(c0)",
      makeRowVector(
          {makeNullableFlatVector<std::string>({std::nullopt}, SFMSKETCH())}));

  EXPECT_TRUE(result->isNullAt(0));
}

} // namespace facebook::velox::functions::test
