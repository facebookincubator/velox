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
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::aggregate::test {

class CountIfAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2"}, {INTEGER(), BOOLEAN(), BOOLEAN()})};
};

TEST_F(CountIfAggregationTest, countIfConst) {
  auto vectors = {
      makeRowVector({
          makeFlatVector<int64_t>(
              10, [](vector_size_t row) { return row / 3; }),
          makeConstant(true, 10),
          makeConstant(false, 10),
      }),
  };

  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {},
      {"count_if(c1)", "count_if(c2)"},
      "SELECT sum(if(c1, 1, 0)), sum(if(c2, 1, 0)) FROM tmp");

  testAggregations(
      vectors,
      {"c0"},
      {"count_if(c1)", "count_if(c2)"},
      "SELECT c0, sum(if(c1, 1, 0)), sum(if(c2, 1, 0)) FROM tmp group by c0");
}

TEST_F(CountIfAggregationTest, oneAggregateSingleGroup) {
  // Make two batches of rows: one with nulls; another without.
  auto vectors = {
      makeRowVector(
          {makeFlatVector<bool>(1'000, [](auto row) { return row % 5 == 0; })}),
      makeRowVector({makeFlatVector<bool>(
          1'100, [](auto row) { return row % 3 == 0; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  testAggregations(
      vectors, {}, {"count_if(c0)"}, "SELECT sum(if(c0, 1, 0)) FROM tmp");
}

TEST_F(CountIfAggregationTest, oneAggregateMultipleGroups) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {"c0"},
      {"count_if(c1)"},
      "SELECT c0, sum(if(c1, 1, 0)) FROM tmp GROUP BY c0");
}

TEST_F(CountIfAggregationTest, twoAggregatesSingleGroup) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {},
      {"count_if(c1)", "count_if(c2)"},
      "SELECT sum(if(c1, 1, 0)), sum(if(c2, 1, 0)) FROM tmp");
}

TEST_F(CountIfAggregationTest, twoAggregatesMultipleGroups) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  testAggregations(
      vectors,
      {"c0"},
      {"count_if(c1)", "count_if(c2)"},
      "SELECT c0, SUM(if(c1, 1, 0)), SUM(if(c2, 1, 0)) FROM tmp GROUP BY c0");
}

TEST_F(CountIfAggregationTest, twoAggregatesMultipleGroupsWrapped) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  testAggregations(
      [&](auto& builder) {
        builder.values(vectors)
            .filter("c0 % 2 = 0")
            .project({"c0 % 11 AS c0_mod_11", "c1", "c2"});
      },
      {"c0_mod_11"},
      {"count_if(c1)", "count_if(c2)"},
      "SELECT c0 % 11, SUM(if(c1, 1, 0)), SUM(if(c2, 1, 0)) FROM tmp WHERE c0 % 2 = 0 GROUP BY 1");
}

} // namespace facebook::velox::aggregate::test
