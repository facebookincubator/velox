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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/DecimalAggregate.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

namespace {

class AverageAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }

  core::PlanNodePtr createAvgAggPlanNode(
      std::vector<VectorPtr> input,
      bool isSingle) {
    PlanBuilder builder;
    builder.values({makeRowVector(input)});
    if (isSingle) {
      builder.singleAggregation({}, {"avg(c0)"});
    } else {
      builder.partialAggregation({}, {"avg(c0)"});
    }
    return builder.planNode();
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
          {BIGINT(), SMALLINT(), INTEGER(), BIGINT(), REAL(), DOUBLE()})};
};

TEST_F(AverageAggregationTest, avgConst) {
  // Have two row vectors a lest as it triggers different code paths.
  auto vectors = {
      makeRowVector({
          makeFlatVector<int64_t>(
              10, [](vector_size_t row) { return row / 3; }),
          makeConstant(5, 10),
          makeConstant(6.0, 10),
      }),
      makeRowVector({
          makeFlatVector<int64_t>(
              10, [](vector_size_t row) { return row / 3; }),
          makeConstant(5, 10),
          makeConstant(6.0, 10),
      }),
  };

  createDuckDbTable(vectors);

  testAggregations(
      vectors, {}, {"avg(c1)", "avg(c2)"}, "SELECT avg(c1), avg(c2) FROM tmp");

  testAggregations(
      vectors,
      {"c0"},
      {"avg(c1)", "avg(c2)"},
      "SELECT c0, avg(c1), avg(c2) FROM tmp group by c0");

  testAggregations(vectors, {}, {"avg(c0)"}, "SELECT avg(c0) FROM tmp");

  testAggregations(
      [&](auto& builder) {
        builder.values(vectors).project({"c0 % 2 AS c0_mod_2", "c0"});
      },
      {"c0_mod_2"},
      {"avg(c0)"},
      "SELECT c0 % 2, avg(c0) FROM tmp group by 1");
}

TEST_F(AverageAggregationTest, avgConstNull) {
  // Have at least two row vectors as it triggers different code paths.
  auto vectors = {
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0}),
          makeNullConstant(TypeKind::BIGINT, 10),
          makeNullConstant(TypeKind::DOUBLE, 10),
      }),
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0}),
          makeNullConstant(TypeKind::BIGINT, 10),
          makeNullConstant(TypeKind::DOUBLE, 10),
      }),
  };

  createDuckDbTable(vectors);

  testAggregations(
      vectors, {}, {"avg(c1)", "avg(c2)"}, "SELECT avg(c1), avg(c2) FROM tmp");

  testAggregations(
      vectors,
      {"c0"},
      {"avg(c1)", "avg(c2)"},
      "SELECT c0, avg(c1), avg(c2) FROM tmp group by c0");

  testAggregations(vectors, {}, {"avg(c0)"}, "SELECT avg(c0) FROM tmp");
}

TEST_F(AverageAggregationTest, avgNulls) {
  // Have two row vectors a lest as it triggers different code paths.
  auto vectors = {
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, std::nullopt, 2, 0, 1}),
          makeNullableFlatVector<int64_t>({0, 1, std::nullopt, 3, 4}),
          makeNullableFlatVector<double>({0.1, 1.2, 2.3, std::nullopt, 4.4}),
      }),
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, std::nullopt, 2, 0, 1}),
          makeNullableFlatVector<int64_t>({0, 1, std::nullopt, 3, 4}),
          makeNullableFlatVector<double>({0.1, 1.2, 2.3, std::nullopt, 4.4}),
      }),
  };

  createDuckDbTable(vectors);

  testAggregations(
      vectors, {}, {"avg(c1)", "avg(c2)"}, "SELECT avg(c1), avg(c2) FROM tmp");

  testAggregations(
      vectors,
      {"c0"},
      {"avg(c1)", "avg(c2)"},
      "SELECT c0, avg(c1), avg(c2) FROM tmp group by c0");
}

TEST_F(AverageAggregationTest, avg) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // global aggregation
  testAggregations(
      vectors,
      {},
      {"avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"},
      "SELECT avg(c1), avg(c2), avg(c4), avg(c5) FROM tmp");

  // global aggregation; no input
  testAggregations(
      [&](auto& builder) { builder.values(vectors).filter("c0 % 2 = 5"); },
      {},
      {"avg(c0)"},
      "SELECT null");

  // global aggregation over filter
  testAggregations(
      [&](auto& builder) { builder.values(vectors).filter("c0 % 5 = 3"); },
      {},
      {"avg(c1)"},
      "SELECT avg(c1) FROM tmp WHERE c0 % 5 = 3");

  // group by
  testAggregations(
      [&](auto& builder) {
        builder.values(vectors).project(
            {"c0 % 10 AS c0_mod_10", "c1", "c2", "c3", "c4", "c5"});
      },
      {"c0_mod_10"},
      {"avg(c1)", "avg(c2)", "avg(c3)", "avg(c4)", "avg(c5)"},
      "SELECT c0 % 10, avg(c1), avg(c2), avg(c3::DOUBLE), "
      "avg(c4), avg(c5) FROM tmp GROUP BY 1");

  // group by; no input
  testAggregations(
      [&](auto& builder) {
        builder.values(vectors)
            .project({"c0 % 10 AS c0_mod_10", "c1"})
            .filter("c0_mod_10 > 10");
      },
      {"c0_mod_10"},
      {"avg(c1)"},
      "");

  // group by over filter
  testAggregations(
      [&](auto& builder) {
        builder.values(vectors)
            .filter("c2 % 5 = 3")
            .project({"c0 % 10 AS c0_mod_10", "c1"});
      },
      {"c0_mod_10"},
      {"avg(c1)"},
      "SELECT c0 % 10, avg(c1) FROM tmp WHERE c2 % 5 = 3 GROUP BY 1");
}

TEST_F(AverageAggregationTest, partialResults) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; })});

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"avg(c0)"})
                  .planNode();

  assertQuery(plan, "SELECT row(4950, 100)");
}

TEST_F(AverageAggregationTest, decimalAccumulator) {
  LongDecimalWithOverflowState accumulator;
  accumulator.sum = -1000;
  accumulator.count = 10;
  accumulator.overflow = -1;

  char* buffer = new char[accumulator.serializedSize()];
  StringView serialized(buffer, accumulator.serializedSize());
  accumulator.serialize(serialized);
  LongDecimalWithOverflowState mergedAccumulator;
  mergedAccumulator.mergeWith(serialized);

  ASSERT_EQ(mergedAccumulator.sum, accumulator.sum);
  ASSERT_EQ(mergedAccumulator.count, accumulator.count);
  ASSERT_EQ(mergedAccumulator.overflow, accumulator.overflow);

  // Merging again to same accumulator.
  memset(buffer, 0, accumulator.serializedSize());
  mergedAccumulator.serialize(serialized);
  mergedAccumulator.mergeWith(serialized);
  ASSERT_EQ(mergedAccumulator.sum, accumulator.sum * 2);
  ASSERT_EQ(mergedAccumulator.count, accumulator.count * 2);
  ASSERT_EQ(mergedAccumulator.overflow, accumulator.overflow * 2);
  delete[] buffer;
}

TEST_F(AverageAggregationTest, avgDecimal) {
  auto shortDecimal = makeNullableShortDecimalFlatVector(
      {1'000, 2'000, 3'000, 4'000, 5'000, std::nullopt}, DECIMAL(10, 1));
  // Short decimal aggregation
  testAggregations(
      {makeRowVector({shortDecimal})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({makeShortDecimalFlatVector({3'000}, DECIMAL(10, 1))})});

  // Long decimal aggregation
  testAggregations(
      {makeRowVector({makeNullableLongDecimalFlatVector(
          {buildInt128(10, 100),
           buildInt128(10, 200),
           buildInt128(10, 300),
           buildInt128(10, 400),
           buildInt128(10, 500),
           std::nullopt},
          DECIMAL(23, 4))})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({makeLongDecimalFlatVector(
          {buildInt128(10, 300)}, DECIMAL(23, 4))})});
  // Round-up average.
  testAggregations(
      {makeRowVector({makeNullableShortDecimalFlatVector(
          {100, 400, 510}, DECIMAL(3, 2))})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({makeShortDecimalFlatVector({337}, DECIMAL(3, 2))})});

  // The total sum overflows the max int128_t limit.
  std::vector<int128_t> rawVector;
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(UnscaledLongDecimal::max().unscaledValue());
  }
  testAggregations(
      {makeRowVector({makeLongDecimalFlatVector(rawVector, DECIMAL(38, 0))})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({makeLongDecimalFlatVector(
          {UnscaledLongDecimal::max().unscaledValue()}, DECIMAL(38, 0))})});
  // The total sum underflows the min int128_t limit.
  rawVector.clear();
  auto underFlowTestResult = makeLongDecimalFlatVector(
      {UnscaledLongDecimal::min().unscaledValue()}, DECIMAL(38, 0));
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(UnscaledLongDecimal::min().unscaledValue());
  }
  testAggregations(
      {makeRowVector({makeLongDecimalFlatVector(rawVector, DECIMAL(38, 0))})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({underFlowTestResult})});

  // Add more rows to show that average result starts deviating from expected
  // result with varying row count.
  // Making sure the error value is consistent.
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(UnscaledLongDecimal::min().unscaledValue());
  }
  AssertQueryBuilder assertQueryBuilder(createAvgAggPlanNode(
      {makeLongDecimalFlatVector(rawVector, DECIMAL(38, 0))}, true));
  auto result = assertQueryBuilder.copyResults(pool());

  auto actualResult = result->childAt(0)->asFlatVector<UnscaledLongDecimal>();
  ASSERT_NE(actualResult->valueAt(0), underFlowTestResult->valueAt(0));
  ASSERT_EQ(
      underFlowTestResult->valueAt(0) - actualResult->valueAt(0),
      UnscaledLongDecimal(-13));

  // Test constant vector.
  testAggregations(
      {makeRowVector({makeConstant<UnscaledShortDecimal>(
          UnscaledShortDecimal(100), 10, DECIMAL(3, 2))})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({makeShortDecimalFlatVector({100}, DECIMAL(3, 2))})});

  auto newSize = shortDecimal->size() * 2;
  auto indices = makeIndices(newSize, [&](int row) { return row / 2; });
  auto dictVector =
      VectorTestBase::wrapInDictionary(indices, newSize, shortDecimal);

  testAggregations(
      {makeRowVector({dictVector})},
      {},
      {"avg(c0)"},
      {},
      {makeRowVector({makeShortDecimalFlatVector({3'000}, DECIMAL(10, 1))})});

  // Decimal average aggregation with multiple groups.
  auto inputRows = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 1}),
           makeShortDecimalFlatVector({37220, 53450}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 2}),
           makeShortDecimalFlatVector({10410, 9250}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3, 3}),
           makeShortDecimalFlatVector({-12783, 0}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1, 2}),
           makeShortDecimalFlatVector({23178, 41093}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2, 3}),
           makeShortDecimalFlatVector({-10023, 5290}, DECIMAL(5, 2))}),
  };

  auto expectedResult = {
      makeRowVector(
          {makeNullableFlatVector<int32_t>({1}),
           makeShortDecimalFlatVector({37949}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({2}),
           makeShortDecimalFlatVector({12683}, DECIMAL(5, 2))}),
      makeRowVector(
          {makeNullableFlatVector<int32_t>({3}),
           makeShortDecimalFlatVector({-2498}, DECIMAL(5, 2))})};

  testAggregations(inputRows, {"c0"}, {"avg(c1)"}, expectedResult);
}

TEST_F(AverageAggregationTest, avgDecimalWithMultipleRowVectors) {
  auto inputRows = {
      makeRowVector({makeShortDecimalFlatVector({100, 200}, DECIMAL(5, 2))}),
      makeRowVector({makeShortDecimalFlatVector({300, 400}, DECIMAL(5, 2))}),
      makeRowVector({makeShortDecimalFlatVector({500, 600}, DECIMAL(5, 2))}),
  };

  auto expectedResult = {
      makeRowVector({makeShortDecimalFlatVector({350}, DECIMAL(5, 2))})};

  testAggregations(inputRows, {}, {"avg(c0)"}, expectedResult);
}

TEST_F(AverageAggregationTest, constantVectorOverflow) {
  auto rows = makeRowVector({makeConstant<int32_t>(1073741824, 100)});
  auto plan = PlanBuilder()
                  .values({rows})
                  .singleAggregation({}, {"avg(c0)"})
                  .planNode();
  assertQuery(plan, "SELECT 1073741824");
}

} // namespace
} // namespace facebook::velox::aggregate::test
