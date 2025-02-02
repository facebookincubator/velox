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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

class AverageAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("spark_");
  }
};

TEST_F(AverageAggregationTest, avgAllNulls) {
  vector_size_t size = 1'000;
  // Have two row vectors a least as it triggers different code paths.
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeAllNullFlatVector<int64_t>(size),
      }),
      makeRowVector({
          makeAllNullFlatVector<int64_t>(size),
      }),
  };
  testAggregations(vectors, {}, {"spark_avg(c0)"}, "SELECT NULL");

  auto plan = PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"spark_avg(c0)"})
                  .planNode();
  assertQuery(plan, "SELECT row(0, 0)");

  // Average with grouping key.
  // Have at least two row vectors as it triggers different code paths.
  vectors = {
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt,
               std::nullopt,
               2,
               std::nullopt,
               10,
               9,
               std::nullopt,
               25,
               12,
               std::nullopt}),
      }),
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt,
               10,
               20,
               std::nullopt,
               std::nullopt,
               25,
               std::nullopt,
               16,
               21,
               std::nullopt}),
      }),
  };
  createDuckDbTable(vectors);
  testAggregations(
      vectors,
      {"c0"},
      {"spark_avg(c1)"},
      "SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  plan = PlanBuilder()
             .values(vectors)
             .partialAggregation({"c0"}, {"spark_avg(c1)"})
             .planNode();
  auto expected = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({0, 1, 2}),
          makeRowVector(
              {"sum", "count"},
              {
                  makeFlatVector<double>({0, 61, 89}),
                  makeFlatVector<int64_t>({0, 4, 6}),
              }),
      });
  assertQuery(plan, expected);
}

TEST_F(AverageAggregationTest, avgDecimal) {
  int64_t kRescale = DecimalUtil::kPowersOfTen[4];
  // Short decimal aggregation
  auto shortDecimal = makeNullableFlatVector<int64_t>(
      {1'000, 2'000, 3'000, 4'000, 5'000, std::nullopt}, DECIMAL(10, 1));
  testAggregations(
      {makeRowVector({shortDecimal})},
      {},
      {"spark_avg(c0)"},
      {},
      {makeRowVector({makeNullableFlatVector<int64_t>(
          {3'000 * kRescale}, DECIMAL(14, 5))})});

  // Long decimal aggregation
  testAggregations(
      {makeRowVector({makeNullableFlatVector<int128_t>(
          {HugeInt::build(10, 100),
           HugeInt::build(10, 200),
           HugeInt::build(10, 300),
           HugeInt::build(10, 400),
           HugeInt::build(10, 500),
           std::nullopt},
          DECIMAL(23, 4))})},
      {},
      {"spark_avg(c0)"},
      {},
      {makeRowVector({makeFlatVector(
          std::vector<int128_t>{HugeInt::build(10, 300) * kRescale},
          DECIMAL(27, 8))})});

  // The total sum overflows the max int128_t limit.
  std::vector<int128_t> rawVector;
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(DecimalUtil::kLongDecimalMax);
  }
  testAggregations(
      {makeRowVector({makeFlatVector<int128_t>(rawVector, DECIMAL(38, 0))})},
      {},
      {"spark_avg(c0)"},
      {},
      {makeRowVector({makeNullableFlatVector(
          std::vector<std::optional<int128_t>>{std::nullopt},
          DECIMAL(38, 4))})});

  // The total sum underflows the min int128_t limit.
  rawVector.clear();
  auto underFlowTestResult = makeNullableFlatVector(
      std::vector<std::optional<int128_t>>{std::nullopt}, DECIMAL(38, 4));
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(DecimalUtil::kLongDecimalMin);
  }
  testAggregations(
      {makeRowVector({makeFlatVector<int128_t>(rawVector, DECIMAL(38, 0))})},
      {},
      {"spark_avg(c0)"},
      {},
      {makeRowVector({underFlowTestResult})});

  // Test constant vector.
  testAggregations(
      {makeRowVector({makeConstant<int64_t>(100, 10, DECIMAL(10, 2))})},
      {},
      {"spark_avg(c0)"},
      {},
      {makeRowVector({makeFlatVector(
          std::vector<int64_t>{100 * kRescale}, DECIMAL(14, 6))})});

  auto newSize = shortDecimal->size() * 2;
  auto indices = makeIndices(newSize, [&](int row) { return row / 2; });
  auto dictVector =
      VectorTestBase::wrapInDictionary(indices, newSize, shortDecimal);

  testAggregations(
      {makeRowVector({dictVector})},
      {},
      {"spark_avg(c0)"},
      {},
      {makeRowVector({makeFlatVector(
          std::vector<int64_t>{3'000 * kRescale}, DECIMAL(14, 5))})});

  // Decimal average aggregation with multiple groups.
  auto inputRows = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 1}),
           makeFlatVector<int64_t>({37220, 53450}, DECIMAL(15, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 2}),
           makeFlatVector<int64_t>({10410, 9250}, DECIMAL(15, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3, 3}),
           makeFlatVector<int64_t>({-12783, 0}, DECIMAL(15, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 2}),
           makeFlatVector<int64_t>({23178, 41093}, DECIMAL(15, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 3}),
           makeFlatVector<int64_t>({-10023, 5290}, DECIMAL(15, 2))}),
  };

  auto expectedResult = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1}),
           makeFlatVector(std::vector<int128_t>{379493333}, DECIMAL(19, 6))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2}),
           makeFlatVector(std::vector<int128_t>{126825000}, DECIMAL(19, 6))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3}),
           makeFlatVector(std::vector<int128_t>{-24976667}, DECIMAL(19, 6))})};

  testAggregations(inputRows, {"c0"}, {"spark_avg(c1)"}, expectedResult);
}

TEST_F(AverageAggregationTest, avgDecimalWithMultipleRowVectors) {
  int64_t kRescale = DecimalUtil::kPowersOfTen[4];
  auto inputRows = {
      makeRowVector({makeFlatVector<int64_t>({100, 200}, DECIMAL(15, 2))}),
      makeRowVector({makeFlatVector<int64_t>({300, 400}, DECIMAL(15, 2))}),
      makeRowVector({makeFlatVector<int64_t>({500, 600}, DECIMAL(15, 2))}),
  };

  auto expectedResult = {makeRowVector(
      {makeFlatVector(std::vector<int128_t>{350 * kRescale}, DECIMAL(19, 6))})};

  testAggregations(inputRows, {}, {"spark_avg(c0)"}, expectedResult);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
