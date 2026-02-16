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

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::prestosql {

namespace {

class ReservoirSampleTest : public AggregationTestBase {
 protected:
  RowVectorPtr makeResultDouble(
      int64_t seenCount,
      const std::vector<std::optional<double>>& sample) {
    std::vector<std::optional<std::vector<std::optional<double>>>> data = {
        sample};
    return makeRowVector({makeRowVector({
        makeFlatVector<int64_t>({seenCount}),
        makeNullableArrayVector<double>(data),
    })});
  }

  VectorPtr makeNullArray(int count, const TypePtr& elementType) {
    auto arrayType = ARRAY(elementType);
    auto builder = BaseVector::create(arrayType, count, pool());
    for (int i = 0; i < count; i++) {
      builder->setNull(i, true);
    }
    return builder;
  }

  template <typename T>
  VectorPtr makeFlatConstantVector(T value, int count) {
    return makeFlatVector<T>(std::vector<T>(count, value));
  }
};

TEST_F(ReservoirSampleTest, testNoInitialSample) {
  auto data = makeRowVector({
      makeNullArray(5, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  auto expected = makeResultDouble(5L, {1.0, 1.0});

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, testLarge) {
  const int sampleSize = 5000;
  const int inputSize = 15000;

  std::vector<double> inputValues(inputSize, 1.0);
  std::vector<std::optional<double>> expectedSample(sampleSize, 1.0);

  auto data = makeRowVector({
      makeNullArray(inputSize, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, inputSize),
      makeFlatVector<double>(inputValues),
      makeFlatConstantVector<int32_t>(sampleSize, inputSize),
  });

  auto expected =
      makeResultDouble(static_cast<int64_t>(inputSize), expectedSample);

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, testInvalidSampleSizeZero) {
  auto data = makeRowVector({
      makeNullArray(5, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(0, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testInvalidSampleSizeNegative) {
  auto data = makeRowVector({
      makeNullArray(5, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(-1, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testInitialSampleSameSize) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0}}),
      makeFlatConstantVector<int64_t>(10, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  auto expected = makeResultDouble(15L, {1.0, 1.0});

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(ReservoirSampleTest, testInitialSampleWrongSize) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0},
           std::vector<double>{1.0, 1.0, 2.0}}),
      makeFlatConstantVector<int64_t>(10, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testInitialSampleSmallerThanMaxSize) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0},
           std::vector<double>{1.0},
           std::vector<double>{1.0},
           std::vector<double>{1.0},
           std::vector<double>{1.0}}),
      makeFlatConstantVector<int64_t>(1, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  auto expected = makeResultDouble(6L, {1.0, 1.0});

  testAggregations(
      {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, {expected});
}

TEST_F(
    ReservoirSampleTest,
    testInitialSampleSeenCountSmallerThanInitialSample) {
  auto data = makeRowVector({
      makeArrayVector<double>(
          {std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0},
           std::vector<double>{1.0, 1.0}}),
      makeFlatConstantVector<int64_t>(1, 5),
      makeFlatVector<double>({1.0, 1.0, 1.0, 1.0, 1.0}),
      makeFlatConstantVector<int32_t>(2, 5),
  });

  EXPECT_THROW(
      testAggregations(
          {data}, {}, {"reservoir_sample(c0, c1, c2, c3)"}, "Throws exception"),
      VeloxException);
}

TEST_F(ReservoirSampleTest, testValidResults) {
  auto data = makeRowVector({
      makeNullArray(10, DOUBLE()),
      makeFlatConstantVector<int64_t>(0, 10),
      makeFlatVector<double>(
          {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}),
      makeFlatConstantVector<int32_t>(4, 10),
  });

  auto result =
      AssertQueryBuilder(
          PlanBuilder()
              .values({data})
              .singleAggregation({}, {"reservoir_sample(c0, c1, c2, c3)"})
              .planNode())
          .copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto rowResult = result->childAt(0)->as<RowVector>();
  ASSERT_EQ(rowResult->size(), 1);

  auto processedCount =
      rowResult->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  EXPECT_EQ(processedCount, 10L);

  auto sampleArray = rowResult->childAt(1)->as<ArrayVector>();
  ASSERT_EQ(sampleArray->size(), 1);

  auto sampleSize = sampleArray->sizeAt(0);
  EXPECT_EQ(sampleSize, 4);

  std::set<double> allValues = {
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  auto sampleElements = sampleArray->elements()->as<FlatVector<double>>();
  auto offset = sampleArray->offsetAt(0);

  for (int i = 0; i < sampleSize; i++) {
    double value = sampleElements->valueAt(offset + i);
    EXPECT_TRUE(allValues.count(value) > 0)
        << "Sample contains value " << value << " which is not in input set";
  }
}

} // namespace
} // namespace facebook::velox::aggregate::prestosql
