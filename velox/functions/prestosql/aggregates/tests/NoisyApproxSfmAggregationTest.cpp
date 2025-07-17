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

#include <gtest/gtest.h>
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/prestosql/aggregates/sfm/SfmSketch.h"

namespace facebook::velox::aggregate::test {

using SfmSketch = functions::aggregate::SfmSketch;
using namespace facebook::velox::exec::test;

class NoisyApproxSfmAggregationTest
    : public functions::aggregate::test::AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(NoisyApproxSfmAggregationTest, distinctNonPrivacy) {
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
       makeConstant(std::numeric_limits<double>::infinity(), 10)});

  auto expectedResult = makeRowVector({makeConstant<int64_t>(10, 1)});
  testAggregations(
      {vectors}, {}, {"noisy_approx_distinct_sfm(c0, c1)"}, {expectedResult});
}

TEST_F(NoisyApproxSfmAggregationTest, setNonPrivacy) {
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
       makeConstant(std::numeric_limits<double>::infinity(), 10)});

  auto returnedSketch =
      AssertQueryBuilder(
          PlanBuilder()
              .values({vectors})
              .singleAggregation({}, {"noisy_approx_set_sfm(c0, c1)"}, {})
              .planNode())
          .copyResults(pool());

  auto serializedView =
      returnedSketch->childAt(0)->asFlatVector<StringView>()->valueAt(0);
  auto deserialized =
      SfmSketch::deserialize(serializedView.data(), &allocator_);
  ASSERT_EQ(deserialized.cardinality(), 10);
}

TEST_F(NoisyApproxSfmAggregationTest, distinctPrivacy) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>(
           numElements, [](vector_size_t row) { return row + 1; }),
       makeConstant(8.0, numElements)});

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({vectors})
              .singleAggregation({}, {"noisy_approx_distinct_sfm(c0, c1)"}, {})
              .planNode())
          .copyResults(pool());

  ASSERT_NEAR(
      result->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
      numElements,
      numElements * 0.25); // 25% tolerance for 100k elements, 8.0 epsilon.
}

TEST_F(NoisyApproxSfmAggregationTest, setPrivacy) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>(
           numElements, [](vector_size_t row) { return row + 1; }),
       makeConstant(8.0, numElements)});

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({vectors})
              .singleAggregation({}, {"noisy_approx_set_sfm(c0, c1)"}, {})
              .planNode())
          .copyResults(pool());

  auto serializedView =
      result->childAt(0)->asFlatVector<StringView>()->valueAt(0);
  auto deserialized =
      SfmSketch::deserialize(serializedView.data(), &allocator_);
  ASSERT_NEAR(deserialized.cardinality(), numElements, numElements * 0.25);
}

TEST_F(NoisyApproxSfmAggregationTest, setBuckets) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>(
           numElements, [](vector_size_t row) { return row + 1; }),
       makeConstant<double>(8.0, numElements),
       makeConstant<int64_t>(8192, numElements)}); // specify buckets.

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({vectors})
              .singleAggregation({}, {"noisy_approx_set_sfm(c0, c1, c2)"}, {})
              .planNode())
          .copyResults(pool());

  auto serializedView =
      result->childAt(0)->asFlatVector<StringView>()->valueAt(0);
  auto deserialized =
      SfmSketch::deserialize(serializedView.data(), &allocator_);
  ASSERT_NEAR(deserialized.cardinality(), numElements, numElements * 0.25);
}

TEST_F(NoisyApproxSfmAggregationTest, distinctBuckets) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>(
           numElements, [](vector_size_t row) { return row + 1; }),
       makeConstant<double>(8.0, numElements),
       makeConstant<int64_t>(8192, numElements)});

  auto result = AssertQueryBuilder(
                    PlanBuilder()
                        .values({vectors})
                        .singleAggregation(
                            {}, {"noisy_approx_distinct_sfm(c0, c1, c2)"}, {})
                        .planNode())
                    .copyResults(pool());

  ASSERT_NEAR(
      result->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
      numElements,
      numElements * 0.25);
}

TEST_F(NoisyApproxSfmAggregationTest, setBucketsAndPrecison) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>(
           numElements, [](vector_size_t row) { return row + 1; }),
       makeConstant<double>(8.0, numElements),
       makeConstant<int64_t>(8192, numElements), // specify buckets.
       makeConstant<int64_t>(30, numElements)}); // specify precision.

  auto result = AssertQueryBuilder(
                    PlanBuilder()
                        .values({vectors})
                        .singleAggregation(
                            {}, {"noisy_approx_set_sfm(c0, c1, c2, c3)"}, {})
                        .planNode())
                    .copyResults(pool());

  auto serializedView =
      result->childAt(0)->asFlatVector<StringView>()->valueAt(0);
  auto deserialized =
      SfmSketch::deserialize(serializedView.data(), &allocator_);
  ASSERT_NEAR(deserialized.cardinality(), numElements, numElements * 0.25);
}

TEST_F(NoisyApproxSfmAggregationTest, distinctBucketsAndPrecison) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<int64_t>(
           numElements, [](vector_size_t row) { return row + 1; }),
       makeConstant<double>(8.0, numElements),
       makeConstant<int64_t>(8192, numElements),
       makeConstant<int64_t>(30, numElements)});

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({vectors})
              .singleAggregation(
                  {}, {"noisy_approx_distinct_sfm(c0, c1, c2, c3)"}, {})
              .planNode())
          .copyResults(pool());

  ASSERT_NEAR(
      result->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
      numElements,
      numElements * 0.25);
}

TEST_F(NoisyApproxSfmAggregationTest, emptyInput) {
  auto vectors =
      makeRowVector({makeFlatVector<int64_t>({}), makeConstant(3.0, 0)});

  auto expectedResult = makeRowVector({makeNullConstant(TypeKind::BIGINT, 1)});
  testAggregations(
      {vectors}, {}, {"noisy_approx_distinct_sfm(c0, c1)"}, {expectedResult});
}

TEST_F(NoisyApproxSfmAggregationTest, doubleInputType) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<double>(
           numElements, [](vector_size_t row) { return row * 1.0; }),
       makeConstant<double>(8.0, numElements)});

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({vectors})
              .singleAggregation({}, {"noisy_approx_distinct_sfm(c0, c1)"}, {})
              .planNode())
          .copyResults(pool());

  ASSERT_NEAR(
      result->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
      numElements,
      numElements * 0.25);
}

TEST_F(NoisyApproxSfmAggregationTest, stringInputType) {
  vector_size_t numElements = 100'000;
  auto vectors = makeRowVector(
      {makeFlatVector<std::string>(
           numElements, [](vector_size_t row) { return std::to_string(row); }),
       makeConstant<double>(8.0, numElements),
       makeConstant<int64_t>(8192, numElements)});

  auto result = AssertQueryBuilder(
                    PlanBuilder()
                        .values({vectors})
                        .singleAggregation(
                            {}, {"noisy_approx_distinct_sfm(c0, c1, c2)"}, {})
                        .planNode())
                    .copyResults(pool());

  ASSERT_NEAR(
      result->childAt(0)->asFlatVector<int64_t>()->valueAt(0),
      numElements,
      numElements * 0.25);
}

} // namespace facebook::velox::aggregate::test
