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

class MaxSizeForStatsTest : public AggregationTestBase {
 protected:
};

// Input:
// c0(bigint)|c1(tinyint)|c2(smallint)|c3(integer)|c4(bigint)|c5(real)|c6(double)
// rand()|rand()|rand()|rand()|rand()|rand()|rand()
// Query:
// select aggr(c1),aggr(c2),aggr(c3),aggr(c4),aggr(c5),aggr(c6);
// Output
// 1|2|4|8|4|8
TEST_F(MaxSizeForStatsTest, allScalarTypes) {
  auto vectors = {makeRowVector(
      {makeFlatVector<int64_t>(1),
       makeFlatVector<int8_t>(1),
       makeFlatVector<int16_t>(1),
       makeFlatVector<int32_t>(1),
       makeFlatVector<int64_t>(1),
       makeFlatVector<float>(1),
       makeFlatVector<double>(1)})};
  // Global partial + final aggregation.
  {
    auto agg = PlanBuilder()
                   .values(vectors)
                   .partialAggregation(
                       {},
                       {"\"$internal$max_data_size_for_stats\"(c1)",
                        "\"$internal$max_data_size_for_stats\"(c2)",
                        "\"$internal$max_data_size_for_stats\"(c3)",
                        "\"$internal$max_data_size_for_stats\"(c4)",
                        "\"$internal$max_data_size_for_stats\"(c5)",
                        "\"$internal$max_data_size_for_stats\"(c6)"})
                   .finalAggregation()
                   .planNode();
    assertQuery(
        agg,
        makeRowVector({
            makeFlatVector<int64_t>(std::vector<int64_t>{1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8}),
        }));
  }
}

// Input:
// c0(bigint)|c1(tinyint)
// null|null
// Query:
// select aggr(c0),aggr(c1)
// Output
// null|null
TEST_F(MaxSizeForStatsTest, nullValues) {
  auto vectors = {makeRowVector({
      makeNullableFlatVector(std::vector<std::optional<int64_t>>{std::nullopt}),
      makeNullableFlatVector(std::vector<std::optional<int8_t>>{std::nullopt}),
  })};
  // Global partial + final aggregation.
  {
    auto agg = PlanBuilder()
                   .values(vectors)
                   .partialAggregation(
                       {},
                       {"\"$internal$max_data_size_for_stats\"(c0)",
                        "\"$internal$max_data_size_for_stats\"(c1)"})
                   .finalAggregation()
                   .planNode();
    assertQuery(
        agg,
        makeRowVector({
            makeNullableFlatVector(
                std::vector<std::optional<int64_t>>{std::nullopt}),
            makeNullableFlatVector(
                std::vector<std::optional<int64_t>>{std::nullopt}),
        }));
  }
}

// Input:
// c0(bigint)|c1(tinyint)
// null|null
// 0 | 0
// Query:
// select aggr(c0),aggr(c1)
// Output
// 8|1
TEST_F(MaxSizeForStatsTest, nullAndNonNullValues) {
  auto vectors = {makeRowVector({
      makeNullableFlatVector(
          std::vector<std::optional<int64_t>>{std::nullopt, 0}),
      makeNullableFlatVector(
          std::vector<std::optional<int8_t>>{std::nullopt, 0}),
  })};
  // Global partial + final aggregation.
  {
    auto agg = PlanBuilder()
                   .values(vectors)
                   .partialAggregation(
                       {},
                       {"\"$internal$max_data_size_for_stats\"(c0)",
                        "\"$internal$max_data_size_for_stats\"(c1)"})
                   .finalAggregation()
                   .planNode();
    assertQuery(
        agg,
        makeRowVector({
            makeNullableFlatVector(std::vector<std::optional<int64_t>>{8}),
            makeNullableFlatVector(std::vector<std::optional<int64_t>>{1}),
        }));
  }
}

// Input:
// c0(bigint)|c1(tinyint)|c2(smallint)|c3(integer)|c4(bigint)|c5(real)|c6(double)
// 1|rand()|rand()|rand()|rand()|rand()|rand()
// 2|rand()|rand()|rand()|rand()|rand()|rand()
// 1|rand()|rand()|rand()|rand()|rand()|rand()
// 2|rand()|rand()|rand()|rand()|rand()|rand()
// Query:
// select aggr(c1),aggr(c2),aggr(c3),aggr(c4),aggr(c5),aggr(c6) group by c0;
// Output
// 1|1|2|4|8|4|8
// 2|1|2|4|8|4|8
TEST_F(MaxSizeForStatsTest, allScalarTypesGroupBy) {
  auto vectors = {makeRowVector(
      {makeFlatVector<int64_t>(std::vector<int64_t>{1, 2, 1, 2}),
       makeFlatVector<int8_t>(4),
       makeFlatVector<int16_t>(4),
       makeFlatVector<int32_t>(4),
       makeFlatVector<int64_t>(4),
       makeFlatVector<float>(4),
       makeFlatVector<double>(4),
       makeFlatVector<bool>(4),
       makeFlatVector<Date>(4),
       makeFlatVector<Timestamp>(4)})};
  // Group partial + final aggregation.
  {
    auto agg = PlanBuilder()
                   .values(vectors)
                   .partialAggregation(
                       {"c0"},
                       {"\"$internal$max_data_size_for_stats\"(c1)",
                        "\"$internal$max_data_size_for_stats\"(c2)",
                        "\"$internal$max_data_size_for_stats\"(c3)",
                        "\"$internal$max_data_size_for_stats\"(c4)",
                        "\"$internal$max_data_size_for_stats\"(c5)",
                        "\"$internal$max_data_size_for_stats\"(c6)",
                        "\"$internal$max_data_size_for_stats\"(c7)",
                        "\"$internal$max_data_size_for_stats\"(c8)",
                        "\"$internal$max_data_size_for_stats\"(c9)"})
                   .finalAggregation()
                   .planNode();
    assertQuery(
        agg,
        makeRowVector({
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{2, 2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
        }));
  }
}

// Input:
// c0(bigint)|c1(tinyint)|c2(smallint)|c3(integer)|c4(bigint)|c5(real)|c6(double)
// 1|rand()|rand()|rand()|rand()|rand()|rand()
// 2|rand()|rand()|rand()|rand()|rand()|rand()
// 1|rand()|rand()|rand()|rand()|rand()|rand()
// 2|rand()|rand()|rand()|rand()|rand()|rand()
// Query:
// select aggr(c1),aggr(c2),aggr(c3),aggr(c4),aggr(c5),aggr(c6) group by c0;
// Output
// 1|1|2|4|8|4|8
// 2|1|2|4|8|4|8
TEST_F(MaxSizeForStatsTest, allScalarTypesGroupByWithIntermediate) {
  auto vectors = {makeRowVector(
      {makeFlatVector<int64_t>(std::vector<int64_t>{1, 2, 1, 2}),
       makeFlatVector<int8_t>(4),
       makeFlatVector<int16_t>(4),
       makeFlatVector<int32_t>(4),
       makeFlatVector<int64_t>(4),
       makeFlatVector<float>(4),
       makeFlatVector<double>(4),
       makeFlatVector<bool>(4),
       makeFlatVector<Date>(4),
       makeFlatVector<Timestamp>(4)})};
  // Group partial + inter + final aggregation.
  {
    auto agg = PlanBuilder()
                   .values(vectors)
                   .partialAggregation(
                       {"c0"},
                       {"\"$internal$max_data_size_for_stats\"(c1)",
                        "\"$internal$max_data_size_for_stats\"(c2)",
                        "\"$internal$max_data_size_for_stats\"(c3)",
                        "\"$internal$max_data_size_for_stats\"(c4)",
                        "\"$internal$max_data_size_for_stats\"(c5)",
                        "\"$internal$max_data_size_for_stats\"(c6)",
                        "\"$internal$max_data_size_for_stats\"(c7)",
                        "\"$internal$max_data_size_for_stats\"(c8)",
                        "\"$internal$max_data_size_for_stats\"(c9)"})
                   .intermediateAggregation()
                   .finalAggregation()
                   .planNode();
    assertQuery(
        agg,
        makeRowVector({
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{2, 2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
        }));
  }
}

// Input:
// c0(bigint)|c1(tinyint)|c2(smallint)|c3(integer)|c4(bigint)|c5(real)|c6(double)
// 1|rand()|rand()|rand()|rand()|rand()|rand()
// 2|rand()|rand()|rand()|rand()|rand()|rand()
// 1|rand()|rand()|rand()|rand()|rand()|rand()
// 2|rand()|rand()|rand()|rand()|rand()|rand()
// Query:
// select aggr(c1),aggr(c2),aggr(c3),aggr(c4),aggr(c5),aggr(c6) group by c0;
// Output
// 1|1|2|4|8|4|8
// 2|1|2|4|8|4|8
TEST_F(MaxSizeForStatsTest, allScalarTypesGroupSingleAggregate) {
  auto vectors = {makeRowVector(
      {makeFlatVector<int64_t>(std::vector<int64_t>{1, 2, 1, 2}),
       makeFlatVector<int8_t>(4),
       makeFlatVector<int16_t>(4),
       makeFlatVector<int32_t>(4),
       makeFlatVector<int64_t>(4),
       makeFlatVector<float>(4),
       makeFlatVector<double>(4),
       makeFlatVector<bool>(4),
       makeFlatVector<Date>(4),
       makeFlatVector<Timestamp>(4)})};
  // Group single.
  {
    auto agg = PlanBuilder()
                   .values(vectors)
                   .singleAggregation(
                       {"c0"},
                       {"\"$internal$max_data_size_for_stats\"(c1)",
                        "\"$internal$max_data_size_for_stats\"(c2)",
                        "\"$internal$max_data_size_for_stats\"(c3)",
                        "\"$internal$max_data_size_for_stats\"(c4)",
                        "\"$internal$max_data_size_for_stats\"(c5)",
                        "\"$internal$max_data_size_for_stats\"(c6)",
                        "\"$internal$max_data_size_for_stats\"(c7)",
                        "\"$internal$max_data_size_for_stats\"(c8)",
                        "\"$internal$max_data_size_for_stats\"(c9)"})
                   .planNode();
    assertQuery(
        agg,
        makeRowVector({
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{2, 2}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
            makeFlatVector<int64_t>(std::vector<int64_t>{1, 1}),
            makeFlatVector<int64_t>(std::vector<int64_t>{4, 4}),
            makeFlatVector<int64_t>(std::vector<int64_t>{8, 8}),
        }));
  }
}

// Input:
// c0(array(bigint))
// [1,2,3,4,5]
// []
// [1,2,3]
// Query:
// select aggr(c0)
// Output
// 40 (size of 1st row)
TEST_F(MaxSizeForStatsTest, arrayGlobalAggregate) {
  auto vectors = {makeRowVector({makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5},
      {},
      {1, 2, 3},
  })})};
  auto agg =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({}, {"\"$internal$max_data_size_for_stats\"(c0)"})
          .planNode();
  assertQuery(
      agg, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{40})}));
}

// Input:
// c0(map(tinyint,integer))
// map(array(1,1,1,1,1),array(1,1,1,1,1))
// map()
// map(array(1,1,1), array(1,1,1))
// Query:
// select aggr(c0)
// Output
// 25 (size of 1st row, (1[tiny] + 4[int]) * 5)
TEST_F(MaxSizeForStatsTest, mapGlobalAggregate) {
  auto vectors = {makeRowVector({makeMapVector<int8_t, int32_t>(
      {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
       {},
       {{1, 1}, {1, 1}, {1, 1}}})})};
  auto agg =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({}, {"\"$internal$max_data_size_for_stats\"(c0)"})
          .planNode();
  assertQuery(
      agg, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{25})}));
}

// Input:
// c0( row(array(bigint), map(tinyint,integer))
// row(aray[1,2,3,4,5], map(array(1,1,1,1,1),array(1,1,1,1,1)))
// row(array[], map())
// row(array[1,2,3], map(array(1,1,1), array(1,1,1)))
// Query:
// select aggr(c0)
// Output
// 65 (size of 1st row, 40 + 25, i.e the outputs of previous two tests).
TEST_F(MaxSizeForStatsTest, rowGlobalAggregate) {
  auto vectors = {makeRowVector({makeRowVector(
      {makeArrayVector<int64_t>({
           {1, 2, 3, 4, 5},
           {},
           {1, 2, 3},
       }),
       makeMapVector<int8_t, int32_t>(
           {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
            {},
            {{1, 1}, {1, 1}, {1, 1}}})})})};
  auto agg =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({}, {"\"$internal$max_data_size_for_stats\"(c0)"})
          .planNode();
  assertQuery(
      agg, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{65})}));
}

// Input:
// c0(varbinary)
// "buf"
// ""
// "hello"
// query:
// select aggr(c0)
// output:
// 5 (len of hello)
TEST_F(MaxSizeForStatsTest, varbinaryGlobalAggregate) {
  // TODO: Is there a cleaner way to create a varbinary vector?
  VectorPtr varbinary_vector = BaseVector::create(VARBINARY(), 3, pool());
  auto flat_vector = varbinary_vector->asFlatVector<StringView>();
  StringView sv1{"buf"};
  StringView sv2{""};
  StringView sv3{"henlo"};
  flat_vector->set(0, sv1);
  flat_vector->set(1, sv2);
  flat_vector->set(2, sv3);

  auto vectors = {makeRowVector({varbinary_vector})};

  auto agg =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({}, {"\"$internal$max_data_size_for_stats\"(c0)"})
          .planNode();
  assertQuery(
      agg, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{5})}));
}

// Input:
// c0(varchar)
// "{1, 2, 3, 4, 5}"
// "{}"
// "{1, 2, 3}"
// query:
// select aggr(c0)
// output:
// 15 (i.e. len of string literal "{1, 2, 3, 4, 5}")
TEST_F(MaxSizeForStatsTest, varcharGlobalAggregate) {
  auto vectors = {makeRowVector({makeFlatVector<StringView>({
      "{1, 2, 3, 4, 5}",
      "{}",
      "{1, 2, 3}",
  })})};

  auto agg =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({}, {"\"$internal$max_data_size_for_stats\"(c0)"})
          .planNode();
  assertQuery(
      agg, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{15})}));
}

// Input (3 rows of recursive complex data):
// c0(row(varchar, map(tinyint, bigint)))
// ("{1, 2, 3, 4, 5}", {1:null,})
// ("{}", {2:[4,5,null]})
// ("{1, 2, 3}", {null:[7,8,9]})
// query:
// select aggr(c0)
// output:
// 33 (3rd row, 9("{1, 2, 3}") + 0 (null) + 24([7,8,9]))
TEST_F(MaxSizeForStatsTest, complexRecursiveGlobalAggregate) {
  auto vectors = {makeRowVector({makeRowVector(
      {makeFlatVector<StringView>({
           "{1, 2, 3, 4, 5}",
           "{}",
           "{1, 2, 3}",
       }),
       createMapOfArraysVector<int8_t, int64_t>(
           {{{1, std::nullopt}},
            {{2, {{4, 5, std::nullopt}}}},
            {{std::nullopt, {{7, 8, 9}}}}})})})};

  auto agg =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({}, {"\"$internal$max_data_size_for_stats\"(c0)"})
          .planNode();
  assertQuery(
      agg, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{33})}));
}

} // namespace
} // namespace facebook::velox::aggregate::test
