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

class CountAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           REAL(),
           DOUBLE(),
           VARCHAR(),
           TINYINT()})};
};

TEST_F(CountAggregationTest, count) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  testAggregations(vectors, {}, {"count()"}, "SELECT count(1) FROM tmp");

  testAggregations(vectors, {}, {"count(1)"}, "SELECT count(1) FROM tmp");

  // count over column with nulls; only non-null rows should be counted
  testAggregations(vectors, {}, {"count(c1)"}, "SELECT count(c1) FROM tmp");

  // count over zero rows; the result should be 0, not null
  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).filter("c0 % 3 > 5");
      },
      {},
      {"count(c1)"},
      "SELECT count(c1) FROM tmp WHERE c0 % 3 > 5");

  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).project({"c0 % 10 AS c0_mod_10", "c1"});
      },
      {"c0_mod_10"},
      {"count(1)"},
      "SELECT c0 % 10, count(1) FROM tmp GROUP BY 1");

  testAggregations(
      [&](PlanBuilder& builder) {
        builder.values(vectors).project({"c0 % 10 AS c0_mod_10", "c7"});
      },
      {"c0_mod_10"},
      {"count(c7)"},
      "SELECT c0 % 10, count(c7) FROM tmp GROUP BY 1");
}

TEST_F(CountAggregationTest, mask) {
  auto data = makeRowVector(
      {"c", "m"},
      {
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
          makeFlatVector<bool>(
              {true,
               false,
               true,
               false,
               true,
               false,
               true,
               false,
               true,
               false}),
      });

  createDuckDbTable({data});

  // count(c)
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"count(c)"}, {"m"})
                  .planNode();
  assertQuery(plan, "SELECT count(c) FILTER (where m) FROM tmp");

  plan = PlanBuilder()
             .values({data})
             .partialAggregation({}, {"count(c)"}, {"m"})
             .finalAggregation()
             .planNode();
  assertQuery(plan, "SELECT count(c) FILTER (where m) FROM tmp");

  // count(1)
  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"count()"}, {"m"})
             .planNode();
  assertQuery(plan, "SELECT count(1) FILTER (where m) FROM tmp");

  plan = PlanBuilder()
             .values({data})
             .partialAggregation({}, {"count()"}, {"m"})
             .finalAggregation()
             .planNode();
  assertQuery(plan, "SELECT count(1) FILTER (where m) FROM tmp");
}

} // namespace
} // namespace facebook::velox::aggregate::test
