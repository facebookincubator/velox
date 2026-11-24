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

#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

class CollectListAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("spark_");
  }
};

TEST_F(CollectListAggregateTest, groupBy) {
  std::vector<RowVectorPtr> batches;
  // Creating 3 batches of input data.
  // 0: {0, null} {0, 1}    {0, 2}
  // 1: {1, 1}    {1, null} {1, 3}
  // 2: {2, 2}    {2, 3}    {2, null}
  // 3: {3, 3}    {3, 4}    {3, 5}
  // 4: {4, 4}    {4, 5}    {4, 6}
  for (auto i = 0; i < 3; i++) {
    RowVectorPtr data = makeRowVector(
        {makeFlatVector<int32_t>({0, 1, 2, 3, 4}),
         makeFlatVector<int64_t>(
             5,
             [&i](const vector_size_t& row) { return i + row; },
             [&i](const auto& row) { return i == row; })});
    batches.push_back(data);
  }

  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({0, 1, 2, 3, 4}),
       makeArrayVectorFromJson<int64_t>(
           {"[1, 2]", "[1, 3]", "[2, 3]", "[3, 4, 5]", "[4, 5, 6]"})});

  AggregationTester(*this)
      .inputs(batches)
      .groupBy({"c0"})
      .aggregates({"spark_collect_list(c1)"})
      .project({"c0", "array_sort(a0)"})
      .withCompanion()
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, global) {
  auto data = makeRowVector({makeNullableFlatVector<int32_t>(
      {std::nullopt, 1, 2, std::nullopt, 4, 5})});
  auto expected =
      makeRowVector({makeArrayVectorFromJson<int32_t>({"[1, 2, 4, 5]"})});

  AggregationTester(*this)
      .inputs(data)
      .aggregates({"spark_collect_list(c0)"})
      .project({"array_sort(a0)"})
      .withCompanion()
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, ignoreNulls) {
  auto input = makeRowVector({makeNullableFlatVector<int32_t>(
      {1, 2, std::nullopt, 4, std::nullopt, 6})});
  // Spark will ignore all null values in the input.
  auto expected =
      makeRowVector({makeArrayVectorFromJson<int32_t>({"[1, 2, 4, 6]"})});
  AggregationTester(*this)
      .inputs(input)
      .aggregates({"spark_collect_list(c0)"})
      .project({"array_sort(a0)"})
      .withCompanion()
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, allNullsInput) {
  auto input = makeRowVector({makeAllNullFlatVector<int64_t>(100)});
  // If all input data is null, Spark will output an empty array.
  auto expected = makeRowVector({makeArrayVectorFromJson<int64_t>({"[]"})});
  AggregationTester(*this)
      .inputs(input)
      .aggregates({"spark_collect_list(c0)"})
      .withCompanion()
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, explicitIgnoreNulls) {
  // The 2-arg form with true behaves like the 1-arg form.
  auto input = makeRowVector({makeNullableFlatVector<int32_t>(
      {1, 2, std::nullopt, 4, std::nullopt, 6})});
  auto expected =
      makeRowVector({makeArrayVectorFromJson<int32_t>({"[1, 2, 4, 6]"})});
  AggregationTester(*this)
      .inputs(input)
      .aggregates({"spark_collect_list(c0, true)"})
      .project({"array_sort(a0)"})
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, respectNulls) {
  // ignoreNulls = false (RESPECT NULLS) keeps nulls.
  auto input = makeRowVector({makeNullableFlatVector<int32_t>(
      {1, 2, std::nullopt, 4, std::nullopt, 6})});
  auto expected = makeRowVector({makeNullableArrayVector<int32_t>(
      std::vector<std::vector<std::optional<int32_t>>>{
          {1, 2, std::nullopt, 4, std::nullopt, 6}})});
  AggregationTester(*this)
      .inputs(input)
      .aggregates({"spark_collect_list(c0, false)"})
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, respectNullsGroupBy) {
  // Runs partial, intermediate and final steps: nulls kept at the raw step
  // must survive the merge steps, which see no constant input.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1, 1, 1}),
       makeNullableFlatVector<int64_t>({std::nullopt, 1, 2, std::nullopt, 3})});
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({0, 1}),
       makeNullableArrayVector<int64_t>(
           std::vector<std::vector<std::optional<int64_t>>>{
               {std::nullopt, 1}, {2, std::nullopt, 3}})});
  AggregationTester(*this)
      .inputs(data)
      .groupBy({"c0"})
      .aggregates({"spark_collect_list(c1, false)"})
      .project({"c0", "a0"})
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, respectNullsAllNulls) {
  // All-null input with ignoreNulls = false gives an array of nulls, not an
  // empty array.
  auto input = makeRowVector({makeAllNullFlatVector<int32_t>(3)});
  auto expected = makeRowVector({makeNullableArrayVector<int32_t>(
      std::vector<std::vector<std::optional<int32_t>>>{
          {std::nullopt, std::nullopt, std::nullopt}})});
  AggregationTester(*this)
      .inputs(input)
      .aggregates({"spark_collect_list(c0, false)"})
      .assertResults(expected);
}

TEST_F(CollectListAggregateTest, respectNullsWindow) {
  // The window operator passes constant inputs too.
  auto data = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1}),
       makeNullableFlatVector<int64_t>({std::nullopt, 1, 2})});
  auto plan =
      exec::test::PlanBuilder()
          .values({data})
          .window({"spark_collect_list(c1, false) over (partition by c0 "
                   "order by c1 nulls first rows between unbounded "
                   "preceding and unbounded following)"})
          .planNode();
  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({0, 0, 1}),
       makeNullableFlatVector<int64_t>({std::nullopt, 1, 2}),
       makeNullableArrayVector<int64_t>(
           std::vector<std::vector<std::optional<int64_t>>>{
               {std::nullopt, 1}, {std::nullopt, 1}, {2}})});
  exec::test::AssertQueryBuilder(plan).assertResults(expected);
}
} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
