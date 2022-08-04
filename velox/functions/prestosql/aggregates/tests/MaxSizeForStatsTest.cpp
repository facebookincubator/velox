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
 public:
  void SetUp() override {
    AggregationTestBase::SetUp();
    disableSpill();
  }
};

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

  testAggregations(
      vectors,
      {},
      {"\"$internal$max_data_size_for_stats\"(c0)",
       "\"$internal$max_data_size_for_stats\"(c1)"},
      "SELECT NULL, NULL");
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
      makeNullableFlatVector<int64_t>({std::nullopt, 0}),
      makeNullableFlatVector<int8_t>({std::nullopt, 0}),
  })};

  testAggregations(
      vectors,
      {},
      {"\"$internal$max_data_size_for_stats\"(c0)",
       "\"$internal$max_data_size_for_stats\"(c1)"},
      "SELECT 8, 1");
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
TEST_F(MaxSizeForStatsTest, allScalarTypes) {
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

  // With grouping keys.
  testAggregations(
      vectors,
      {"c0"},
      {"\"$internal$max_data_size_for_stats\"(c1)",
       "\"$internal$max_data_size_for_stats\"(c2)",
       "\"$internal$max_data_size_for_stats\"(c3)",
       "\"$internal$max_data_size_for_stats\"(c4)",
       "\"$internal$max_data_size_for_stats\"(c5)",
       "\"$internal$max_data_size_for_stats\"(c6)",
       "\"$internal$max_data_size_for_stats\"(c7)",
       "\"$internal$max_data_size_for_stats\"(c8)",
       "\"$internal$max_data_size_for_stats\"(c9)"},
      "SELECT * FROM (VALUES (1,1,2,4,8,4,8,1,4,16),(2,1,2,4,8,4,8,1,4,16))");

  // Without grouping keys.
  testAggregations(
      vectors,
      {},
      {"\"$internal$max_data_size_for_stats\"(c1)",
       "\"$internal$max_data_size_for_stats\"(c2)",
       "\"$internal$max_data_size_for_stats\"(c3)",
       "\"$internal$max_data_size_for_stats\"(c4)",
       "\"$internal$max_data_size_for_stats\"(c5)",
       "\"$internal$max_data_size_for_stats\"(c6)",
       "\"$internal$max_data_size_for_stats\"(c7)",
       "\"$internal$max_data_size_for_stats\"(c8)",
       "\"$internal$max_data_size_for_stats\"(c9)"},
      "SELECT * FROM (VALUES (1,2,4,8,4,8,1,4,16))");
}

// Input:
// c0(array(bigint))
// [1,2,3,4,5]
// []
// [1,2,3]
// Query:
// select aggr(c0)
// Output
// 44
TEST_F(MaxSizeForStatsTest, arrayGlobalAggregate) {
  auto vectors = {makeRowVector({makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5},
      {},
      {1, 2, 3},
  })})};
  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c0)"}, "SELECT 44");
}

// Input:
// c0(map(tinyint,integer))
// map(array(1,1,1,1,1),array(1,1,1,1,1))
// map()
// map(array(1,1,1), array(1,1,1))
// Query:
// select aggr(c0)
// Output
// 29
TEST_F(MaxSizeForStatsTest, mapGlobalAggregate) {
  auto vectors = {makeRowVector({makeMapVector<int8_t, int32_t>(
      {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
       {},
       {{1, 1}, {1, 1}, {1, 1}}})})};
  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c0)"}, "SELECT 29");
}

// Input:
// c0( row(array(bigint), map(tinyint,integer))
// row(aray[1,2,3,4,5], map(array(1,1,1,1,1),array(1,1,1,1,1)))
// row(array[], map())
// row(array[1,2,3], map(array(1,1,1), array(1,1,1)))
// Query:
// select aggr(c0)
// Output
// 77
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
  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c0)"}, "SELECT 77");
}

// Input:
// c0(varbinary)
// "buf"
// ""
// "hello"
// query:
// select aggr(c0)
// output:
// 18
TEST_F(MaxSizeForStatsTest, varbinaryGlobalAggregate) {
  VectorPtr varbinaryVector = BaseVector::create(VARBINARY(), 3, pool());
  auto flatVector = varbinaryVector->asFlatVector<StringView>();
  StringView buf_sv{"buf"};
  StringView _sv{""};
  StringView hello_sv{"hello, world !"};
  flatVector->set(0, buf_sv);
  flatVector->set(1, _sv);
  flatVector->set(2, hello_sv);

  auto vectors = {makeRowVector({varbinaryVector})};
  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c0)"}, "SELECT 18");
}

// Input:
// c0(varchar)
// "{1, 2, 3, 4, 5}"
// "{}"
// "{1, 2, 3}"
// query:
// select aggr(c0)
// output:
// 19
TEST_F(MaxSizeForStatsTest, varcharGlobalAggregate) {
  auto vectors = {makeRowVector({makeFlatVector<StringView>({
      "{1, 2, 3, 4, 5}",
      "{}",
      "{1, 2, 3}",
  })})};
  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c0)"}, "SELECT 19");
}

// Input (3 rows of recursive complex data):
// c0(row(varchar, map(tinyint, bigint)))
// ("{1, 2, 3, 4, 5}", {1:null,})
// ("{}", {2:[4,5,null]})
// ("{1, 2, 3}", {null:[7,8,9]})
// query:
// select aggr(c0)
// output:
// 50 (3rd row(4), 13("{1, 2, 3}") + 5 (null) + 28([7,8,9]))
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

  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c0)"}, "SELECT 50");
}

TEST_F(MaxSizeForStatsTest, constantEncodingTest) {
  auto columnOne = makeFlatVector<int64_t>({1, 2, 3});
  auto columnTwo = makeRowVector(
      {makeFlatVector<StringView>({
           "{1, 2, 3, 4, 5}",
           "{}",
           "{1, 2, 3}",
       }),
       createMapOfArraysVector<int8_t, int64_t>(
           {{{1, std::nullopt}},
            {{2, {{4, 5, std::nullopt}}}},
            {{std::nullopt, {{7, 8, 9}}}}})});
  auto columnTwoConstantEncoded = BaseVector::wrapInConstant(3, 1, columnTwo);

  auto vectors = {makeRowVector({columnOne, columnTwoConstantEncoded})};

  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c1)"}, "SELECT 36");

  testAggregations(
      vectors,
      {"c0"},
      {"\"$internal$max_data_size_for_stats\"(c1)"},
      "SELECT * FROM (VALUES (1,36),(2,36),(3,36))");
}

TEST_F(MaxSizeForStatsTest, dictionaryEncodingTest) {
  auto columnOne = makeFlatVector<int64_t>({1, 2, 3});
  auto columnTwo = makeRowVector(
      {makeFlatVector<StringView>({
           "{1, 2, 3, 4, 5}",
           "{}",
           "{1, 2, 3}",
       }),
       createMapOfArraysVector<int8_t, int64_t>(
           {{{1, std::nullopt}},
            {{2, {{4, 5, std::nullopt}}}},
            {{std::nullopt, {{7, 8, 9}}}}})});
  vector_size_t size = 3;
  auto indices = AlignedBuffer::allocate<vector_size_t>(size, pool());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (auto i = 0; i < size; ++i) {
    rawIndices[i] = i;
  }
  auto columnTwoDictionaryEncoded =
      BaseVector::wrapInDictionary(nullptr, indices, size, columnTwo);
  auto vectors = {makeRowVector({columnOne, columnTwoDictionaryEncoded})};

  testAggregations(
      vectors, {}, {"\"$internal$max_data_size_for_stats\"(c1)"}, "SELECT 50");

  testAggregations(
      vectors,
      {"c0"},
      {"\"$internal$max_data_size_for_stats\"(c1)"},
      "SELECT * FROM (VALUES (1,32),(2,36),(3,50))");
}

} // namespace
} // namespace facebook::velox::aggregate::test
