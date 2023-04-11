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

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

namespace {

class ArbitraryTest : public AggregationTestBase {};

TEST_F(ArbitraryTest, noNulls) {
  // Create vectors without nulls because DuckDB's "first" aggregate does not
  // ignore them.
  const int32_t size = 10'000;
  auto vectors = {makeRowVector(
      {makeFlatVector<int64_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int8_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int16_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<float>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<double>(size, [](vector_size_t row) { return row; })})};
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "arbitrary(c1)",
      "arbitrary(c2)",
      "arbitrary(c3)",
      "arbitrary(c4)",
      "arbitrary(c5)",
      "arbitrary(c6)"};

  // Global aggregation.
  testAggregations(
      vectors,
      {},
      aggregates,
      "SELECT first(c1), first(c2), first(c3), first(c4), first(c5), first(c6) FROM tmp");

  // Group by aggregation.
  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).project(
            {"c0 % 10", "c1", "c2", "c3", "c4", "c5", "c6"});
      },
      {"p0"},
      aggregates,
      "SELECT c0 % 10, first(c1), first(c2), first(c3), first(c4), first(c5), first(c6) FROM tmp GROUP BY 1");

  // encodings: use filter to wrap aggregation inputs in a dictionary.
  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors)
            .filter("c0 % 2 = 0")
            .project({"c0 % 10", "c1", "c2", "c3", "c4", "c5", "c6"});
      },
      {"p0"},
      aggregates,
      "SELECT c0 % 10, first(c1), first(c2), first(c3), first(c4), first(c5), first(c6) FROM tmp WHERE c0 % 2 = 0 GROUP BY 1");

  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).filter("c0 % 2 = 0");
      },
      {},
      aggregates,
      "SELECT first(c1), first(c2), first(c3), first(c4), first(c5), first(c6) FROM tmp WHERE c0 % 2 = 0");
}

TEST_F(ArbitraryTest, nulls) {
  auto vectors = {
      makeRowVector({
          makeNullableFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt, std::nullopt, std::nullopt, 4, std::nullopt, 5}),
          makeNullableFlatVector<double>({
              std::nullopt,
              0.50,
              std::nullopt,
              std::nullopt,
              0.25,
              std::nullopt,
          }),
      }),
  };

  // Global aggregation.
  testAggregations(
      vectors,
      {},
      {"arbitrary(c1)", "arbitrary(c2)"},
      "SELECT * FROM( VALUES (4, 0.50)) AS t");

  // Group by aggregation.
  testAggregations(
      vectors,
      {"c0"},
      {"arbitrary(c1)", "arbitrary(c2)"},
      "SELECT * FROM(VALUES (1, NULL, 0.50), (2, 4, NULL), (3, 5, 0.25)) AS t");
}

TEST_F(ArbitraryTest, varchar) {
  auto rowType = ROW({"c0", "c1"}, {INTEGER(), VARCHAR()});
  auto vectors = makeVectors(rowType, 1000, 10);
  createDuckDbTable(vectors);

  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).project({"c0 % 11", "c1"});
      },
      {"p0"},
      {"arbitrary(c1)"},
      "SELECT c0 % 11, first(c1) FROM tmp WHERE c1 IS NOT NULL GROUP BY 1");

  testAggregations(
      vectors,
      {},
      {"arbitrary(c1)"},
      "SELECT first(c1) FROM tmp WHERE c1 IS NOT NULL");

  // encodings: use filter to wrap aggregation inputs in a dictionary.
  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).filter("c0 % 2 = 0").project({"c0 % 11", "c1"});
      },
      {"p0"},
      {"arbitrary(c1)"},
      "SELECT c0 % 11, first(c1) FROM tmp WHERE c0 % 2 = 0 AND c1 IS NOT NULL GROUP BY 1");

  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).filter("c0 % 2 = 0");
      },
      {},
      {"arbitrary(c1)"},
      "SELECT first(c1) FROM tmp WHERE c0 % 2 = 0 AND c1 IS NOT NULL");
}

TEST_F(ArbitraryTest, varcharConstAndNulls) {
  auto vectors = {makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row % 7; }),
      makeConstant("apple", 100),
      makeNullConstant(TypeKind::VARCHAR, 100),
  })};

  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {},
      {"arbitrary(c1)", "arbitrary(c2)"},
      "SELECT first(c1), first(c2) FROM tmp");

  testAggregations(
      vectors,
      {"c0"},
      {"arbitrary(c1)", "arbitrary(c2)"},
      "SELECT c0, first(c1), first(c2) FROM tmp group by c0");
}

TEST_F(ArbitraryTest, numericConstAndNulls) {
  auto vectors = {makeRowVector(
      {makeFlatVector<int32_t>(100, [](auto row) { return row % 7; }),
       makeConstant(11, 100),
       makeNullConstant(TypeKind::BIGINT, 100)})};

  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {},
      {"arbitrary(c1)", "arbitrary(c2)"},
      "SELECT first(c1), first(c2) FROM tmp");

  testAggregations(
      vectors,
      {"c0"},
      {"arbitrary(c1)", "arbitrary(c2)"},
      "SELECT c0, first(c1), first(c2) FROM tmp group by c0");
}

TEST_F(ArbitraryTest, boolean) {
  auto data = makeRowVector({
      // Grouping key.
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3, 4, 4}),
      // Input values: 'constant' within groups.
      makeNullableFlatVector<bool>(
          {true,
           true,
           false,
           false,
           std::nullopt,
           std::nullopt,
           std::nullopt,
           false}),
      makeConstant<bool>(std::nullopt, 8),
  });

  auto expectedResult = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeNullableFlatVector<bool>({true, false, std::nullopt, false}),
  });

  testAggregations({data}, {"c0"}, {"arbitrary(c1)"}, {expectedResult});

  // Global aggregation.
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"arbitrary(c1)"})
                  .planNode();

  assertQuery(plan, "SELECT true");

  testAggregations({data}, {}, {"arbitrary(c2)"}, "SELECT null");
}

TEST_F(ArbitraryTest, timestamp) {
  auto data = makeRowVector({
      // Grouping key.
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3, 4, 4}),
      // Input values: constant within groups: 100.1, 100.1, 200.2, 200.2, etc.
      makeNullableFlatVector<Timestamp>(
          {Timestamp(100, 1),
           Timestamp(100, 1),
           Timestamp(200, 2),
           Timestamp(200, 2),
           std::nullopt,
           std::nullopt,
           std::nullopt,
           Timestamp(100, 4)}),
      makeConstant<Timestamp>(std::nullopt, 8),
  });

  auto expectedResult = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeNullableFlatVector<Timestamp>(
          {Timestamp(100, 1),
           Timestamp(200, 2),
           std::nullopt,
           Timestamp(100, 4)}),
  });

  testAggregations({data}, {"c0"}, {"arbitrary(c1)"}, {expectedResult});

  // Global aggregation.
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"arbitrary(c1)"})
                  .planNode();

  auto result = readSingleValue(plan);
  ASSERT_TRUE(!result.isNull());
  ASSERT_EQ(result.kind(), TypeKind::TIMESTAMP);

  auto timestamp = result.value<Timestamp>();
  ASSERT_EQ(timestamp, Timestamp(100, 1));

  testAggregations({data}, {}, {"arbitrary(c2)"}, "SELECT null");
}

TEST_F(ArbitraryTest, date) {
  auto data = makeRowVector({
      // Grouping key.
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3, 4, 4}),
      // Input values: constant within groups.
      makeNullableFlatVector<Date>(
          {Date(125),
           Date(125),
           Date(126),
           Date(126),
           std::nullopt,
           std::nullopt,
           std::nullopt,
           Date(128)}),
      makeConstant<Timestamp>(std::nullopt, 8),
  });

  auto expectedResult = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeNullableFlatVector<Date>(
          {Date(125), Date(126), std::nullopt, Date(128)}),
  });

  testAggregations({data}, {"c0"}, {"arbitrary(c1)"}, {expectedResult});

  // Global aggregation.
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"arbitrary(c1)"})
                  .planNode();

  auto result = readSingleValue(plan);
  ASSERT_TRUE(!result.isNull());
  ASSERT_EQ(result.kind(), TypeKind::DATE);

  auto date = result.value<Date>();
  ASSERT_EQ(date, Date(125));

  testAggregations({data}, {}, {"arbitrary(c2)"}, "SELECT null");
}

TEST_F(ArbitraryTest, interval) {
  auto data = makeRowVector({
      // Grouping key.
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3, 4, 4}),
      // Input values: constant within groups.
      makeNullableFlatVector<int64_t>(
          {125, 125, 126, 126, std::nullopt, std::nullopt, std::nullopt, 128},
          INTERVAL_DAY_TIME()),
      makeConstant<Timestamp>(std::nullopt, 8),
  });

  auto expectedResult = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeNullableFlatVector<int64_t>(
          {125, 126, std::nullopt, 128}, INTERVAL_DAY_TIME()),
  });

  testAggregations({data}, {"c0"}, {"arbitrary(c1)"}, {expectedResult});

  // Global aggregation.
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"arbitrary(c1)"})
                  .planNode();

  auto interval = readSingleValue(plan);
  ASSERT_EQ(interval.value<int64_t>(), 125);

  testAggregations({data}, {}, {"arbitrary(c2)"}, "SELECT null");
}

} // namespace
} // namespace facebook::velox::aggregate::test
