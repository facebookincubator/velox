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
#include <folly/init/Init.h>

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
    disableTestIncremental();
    registerAggregateFunctions("spark_", true);
  }

  void testGlobalAverage(
      const std::vector<RowVectorPtr>& input,
      const std::vector<RowVectorPtr>& expected,
      const std::vector<TypePtr>& argTypes) {
    testAggregations(
        input,
        {},
        {"spark_avg(c0)"},
        {},
        expected,
        /*config*/ {});
  }

  core::PlanNodePtr createMergeExtractDecimal(
      const RowVectorPtr& intermediateInput,
      const TypePtr& intermediateType,
      const TypePtr& resultType) {
    core::AggregationNode::Aggregate avgAggregate;
    avgAggregate.call = std::make_shared<core::CallTypedExpr>(
        resultType,
        "spark_avg_merge_extract_DECIMAL",
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(
                intermediateType, "c0")});
    avgAggregate.rawInputTypes = {intermediateType};

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto child =
        PlanBuilder(planNodeIdGenerator).values({intermediateInput}).planNode();

    return std::make_shared<core::AggregationNode>(
        planNodeIdGenerator->next(),
        core::AggregationNode::Step::kSingle,
        std::vector<core::FieldAccessTypedExprPtr>{},
        std::vector<core::FieldAccessTypedExprPtr>{},
        std::vector<std::string>{"c0"},
        std::vector{avgAggregate},
        /*ignoreNullKeys=*/false,
        /*noGroupsSpanBatches=*/false,
        std::move(child));
  }
};

TEST_F(AverageAggregationTest, avgAllNulls) {
  const vector_size_t size = 1'000;
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
  const int64_t kRescale = DecimalUtil::kPowersOfTen[4];
  // Short decimal aggregation.
  auto shortDecimal = makeNullableFlatVector<int64_t>(
      {1'000, 2'000, 3'000, 4'000, 5'000, std::nullopt}, DECIMAL(12, 1));
  auto shortDecimalInput = makeRowVector({shortDecimal});
  auto shortDecimalExpected = makeRowVector(
      {makeConstant<int64_t>(3'000 * kRescale, 1, DECIMAL(16, 5))});
  testGlobalAverage(
      {shortDecimalInput}, {shortDecimalExpected}, {DECIMAL(12, 1)});

  // Long decimal aggregation.
  auto longDecimalInput = makeRowVector({makeNullableFlatVector<int128_t>(
      {HugeInt::build(10, 100),
       HugeInt::build(10, 200),
       HugeInt::build(10, 300),
       HugeInt::build(10, 400),
       HugeInt::build(10, 500),
       std::nullopt},
      DECIMAL(23, 4))});
  auto longDecimalExpected = makeRowVector({makeConstant<int128_t>(
      HugeInt::build(10, 300) * kRescale, 1, DECIMAL(27, 8))});
  testGlobalAverage(
      {longDecimalInput}, {longDecimalExpected}, {DECIMAL(23, 4)});

  // The total sum overflows the max int128_t limit.
  std::vector<int128_t> rawVector;
  auto nullExpected = makeRowVector({makeNullableFlatVector(
      std::vector<std::optional<int128_t>>{std::nullopt}, DECIMAL(38, 4))});
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(DecimalUtil::kLongDecimalMax);
  }
  testGlobalAverage(
      {makeRowVector({makeFlatVector<int128_t>(rawVector, DECIMAL(38, 0))})},
      {nullExpected},
      {DECIMAL(38, 4)});

  // The total sum underflows the min int128_t limit.
  rawVector.clear();
  for (int i = 0; i < 10; ++i) {
    rawVector.push_back(DecimalUtil::kLongDecimalMin);
  }
  testGlobalAverage(
      {makeRowVector({makeFlatVector<int128_t>(rawVector, DECIMAL(38, 0))})},
      {nullExpected},
      {DECIMAL(38, 4)});

  // Test constant vector.
  testGlobalAverage(
      {makeRowVector({makeConstant<int64_t>(100, 10, DECIMAL(12, 2))})},
      {makeRowVector(
          {makeConstant<int64_t>(100 * kRescale, 1, DECIMAL(16, 6))})},
      {DECIMAL(10, 2)});
  auto newSize = shortDecimal->size() * 2;
  auto indices = makeIndices(newSize, [&](int row) { return row / 2; });
  auto dictVector = wrapInDictionary(indices, newSize, shortDecimal);
  testGlobalAverage(
      {makeRowVector({dictVector})}, {shortDecimalExpected}, {DECIMAL(12, 1)});

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
  testAggregations(
      inputRows,
      {"c0"},
      {"spark_avg(c1)"},
      expectedResult,
      /*config*/ {});

  // dividePrecision greater than resultScale in computeFinalResult.
  auto valueA = HugeInt::parse("11999999998800000000");
  auto valueB = HugeInt::parse("12000000000000000000");
  auto longDecimalInputRows = {makeRowVector(
      {makeNullableFlatVector<int32_t>({1, 1, 1, 1, 1, 1, 1}),
       makeFlatVector<int128_t>(
           {valueA, valueA, valueA, valueB, valueB, valueB, valueB},
           DECIMAL(38, 18))})};

  auto longDecimalExpectedResult = {makeRowVector(
      {makeNullableFlatVector<int32_t>({1}),
       makeFlatVector(
           std::vector<int128_t>{HugeInt::parse("119999999994857142857143")},
           DECIMAL(38, 22))})};

  testAggregations(
      longDecimalInputRows,
      {"c0"},
      {"spark_avg(c1)"},
      longDecimalExpectedResult);
}

TEST_F(AverageAggregationTest, avgDecimalWithMultipleRowVectors) {
  const int64_t kRescale = DecimalUtil::kPowersOfTen[4];
  auto inputRows = {
      makeRowVector({makeFlatVector<int128_t>({100, 200}, DECIMAL(28, 2))}),
      makeRowVector({makeFlatVector<int128_t>({300, 400}, DECIMAL(28, 2))}),
      makeRowVector({makeFlatVector<int128_t>({500, 600}, DECIMAL(28, 2))}),
  };
  auto expectedResult = {makeRowVector(
      {makeFlatVector(std::vector<int128_t>{350 * kRescale}, DECIMAL(32, 6))})};
  testGlobalAverage(inputRows, expectedResult, {DECIMAL(28, 2)});
}

TEST_F(AverageAggregationTest, avgDecimalEmptyInput) {
  auto inputRows = {
      makeRowVector({makeFlatVector<int128_t>({}, DECIMAL(28, 2))}),
  };
  auto expectedResult = {makeRowVector({makeNullableFlatVector(
      std::vector<std::optional<int128_t>>{std::nullopt}, DECIMAL(32, 6))})};
  testGlobalAverage(inputRows, expectedResult, {DECIMAL(28, 2)});
}

TEST_F(AverageAggregationTest, avgDecimalZeroCount) {
  auto intermediateInput = makeRowVector({makeRowVector(
      {makeFlatVector<int128_t>(std::vector<int128_t>{0, 0, 0}, DECIMAL(22, 1)),
       makeFlatVector<int64_t>(std::vector<int64_t>{0, 0, 0})})});

  auto plan = createMergeExtractDecimal(
      intermediateInput, ROW({DECIMAL(22, 1), BIGINT()}), DECIMAL(16, 5));

  auto expected = makeRowVector({makeNullableFlatVector<int64_t>(
      std::vector<std::optional<int64_t>>{std::nullopt}, DECIMAL(16, 5))});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(AverageAggregationTest, avgDecimalCompanionPartial) {
  std::vector<int64_t> shortDecimalRawVector;
  int128_t sum = 0;
  for (int i = 0; i < 100; ++i) {
    shortDecimalRawVector.emplace_back(i * 1000);
    sum += i * 1000;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(shortDecimalRawVector, DECIMAL(12, 1))});
  auto plan = PlanBuilder()
                  .values({input})
                  .singleAggregation({}, {"spark_avg_partial(c0)"})
                  .planNode();
  const std::vector<int128_t> sumVector = {sum};
  const std::vector<int64_t> count = {100};
  const auto expected = makeRowVector({makeRowVector(
      {makeFlatVector<int128_t>(sumVector, DECIMAL(22, 1)),
       makeFlatVector<int64_t>(count)})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(AverageAggregationTest, avgDecimalCompanionMerge) {
  auto intermediateInput = makeRowVector({makeRowVector(
      {makeFlatVector<int128_t>(
           std::vector<int128_t>{1000, 2000, 3000}, DECIMAL(22, 1)),
       makeFlatVector<int64_t>(std::vector<int64_t>{10, 10, 10})})});

  auto plan = PlanBuilder()
                  .values({intermediateInput})
                  .singleAggregation({}, {"spark_avg_merge(c0)"})
                  .planNode();
  auto expected = makeRowVector({makeRowVector(
      {makeFlatVector<int128_t>(std::vector<int128_t>{6000}, DECIMAL(22, 1)),
       makeFlatVector<int64_t>(std::vector<int64_t>{30})})});
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_F(AverageAggregationTest, avgDecimalCompanionMergeExtract) {
  auto intermediateInput = makeRowVector({makeRowVector(
      {makeFlatVector<int128_t>(
           std::vector<int128_t>{1000, 2000, 3000}, DECIMAL(22, 1)),
       makeFlatVector<int64_t>(std::vector<int64_t>{10, 10, 10})})});

  // Intermediate sum type is DECIMAL(22, 1), so the input type is
  // DECIMAL(22-10,1)=DECIMAL(12,1), and the result type is
  // DECIMAL(12+4,1+4)=DECIMAL(16,5)
  auto plan = createMergeExtractDecimal(
      intermediateInput, ROW({DECIMAL(22, 1), BIGINT()}), DECIMAL(16, 5));

  auto expected = makeRowVector(
      {makeFlatVector<int64_t>(std::vector<int64_t>{2000000}, DECIMAL(16, 5))});
  AssertQueryBuilder(plan).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
