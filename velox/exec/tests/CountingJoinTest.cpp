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

#include <gtest/gtest.h>

#include "velox/exec/HashTable.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
namespace {

class CountingJoinTest : public OperatorTestBase {
 protected:
  void assertCountingJoin(
      const std::vector<int32_t>& probe,
      const std::vector<int32_t>& build,
      core::JoinType joinType,
      const std::vector<int32_t>& expected) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values(makeRowVector({makeFlatVector<int32_t>(probe)}))
                    .hashJoin(
                        {"c0"},
                        {"u0"},
                        PlanBuilder(planNodeIdGenerator)
                            .values(makeRowVector(
                                {"u0"}, {makeFlatVector<int32_t>(build)}))
                            .planNode(),
                        "",
                        {"c0"},
                        joinType)
                    .planNode();
    assertEqualResults(
        {makeRowVector({makeFlatVector<int32_t>(expected)})},
        {AssertQueryBuilder(plan).copyResults(pool())});
  }
};

// {A, A, A, B, B} EXCEPT ALL {A, A, C} = {A, B, B}
TEST_F(CountingJoinTest, countingAnti) {
  assertCountingJoin(
      {1, 1, 1, 2, 2}, {1, 1, 3}, core::JoinType::kCountingAnti, {1, 2, 2});
}

// {A, A, A, B, B} INTERSECT ALL {A, A, C} = {A, A}
TEST_F(CountingJoinTest, countingSemi) {
  assertCountingJoin(
      {1, 1, 1, 2, 2},
      {1, 1, 3},
      core::JoinType::kCountingLeftSemiFilter,
      {1, 1});
}

// Empty build side: returns all probe rows.
TEST_F(CountingJoinTest, countingAntiEmptyBuild) {
  assertCountingJoin({1, 2, 3}, {}, core::JoinType::kCountingAnti, {1, 2, 3});
}

// Empty build side: returns nothing.
TEST_F(CountingJoinTest, countingSemiEmptyBuild) {
  assertCountingJoin(
      {1, 2, 3}, {}, core::JoinType::kCountingLeftSemiFilter, {});
}

// Empty probe side: returns nothing.
TEST_F(CountingJoinTest, emptyProbe) {
  assertCountingJoin({}, {1, 2, 3}, core::JoinType::kCountingAnti, {});
  assertCountingJoin(
      {}, {1, 2, 3}, core::JoinType::kCountingLeftSemiFilter, {});
}

// No duplicates on build side: degenerates to regular anti/semi.
TEST_F(CountingJoinTest, noDuplicatesOnBuild) {
  assertCountingJoin(
      {1, 2, 3, 4, 5}, {2, 4}, core::JoinType::kCountingAnti, {1, 3, 5});
  assertCountingJoin(
      {1, 2, 3, 4, 5}, {2, 4}, core::JoinType::kCountingLeftSemiFilter, {2, 4});
}

// All duplicates on build side.
TEST_F(CountingJoinTest, allDuplicatesOnBuild) {
  // EXCEPT ALL: 5 - 3 = 2
  assertCountingJoin(
      {1, 1, 1, 1, 1}, {1, 1, 1}, core::JoinType::kCountingAnti, {1, 1});
  // INTERSECT ALL: min(5, 3) = 3
  assertCountingJoin(
      {1, 1, 1, 1, 1},
      {1, 1, 1},
      core::JoinType::kCountingLeftSemiFilter,
      {1, 1, 1});
}

// Multiple batches on probe side.
TEST_F(CountingJoinTest, multipleProbeBatches) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 1, 2})}),
      makeRowVector({makeFlatVector<int32_t>({1, 2, 3})}),
      makeRowVector({makeFlatVector<int32_t>({2, 3, 3})}),
  };
  auto buildVector =
      makeRowVector({"u0"}, {makeFlatVector<int32_t>({1, 2, 2, 3})});

  auto test = [&](core::JoinType joinType,
                  const std::vector<int32_t>& expected) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan =
        PlanBuilder(planNodeIdGenerator)
            .values(probeVectors)
            .hashJoin(
                {"c0"},
                {"u0"},
                PlanBuilder(planNodeIdGenerator).values(buildVector).planNode(),
                "",
                {"c0"},
                joinType)
            .planNode();
    assertEqualResults(
        {makeRowVector({makeFlatVector<int32_t>(expected)})},
        {AssertQueryBuilder(plan).copyResults(pool())});
  };

  // Probe: {1:3, 2:3, 3:3}, Build: {1:1, 2:2, 3:1}
  // EXCEPT ALL: {1:2, 2:1, 3:2}
  test(core::JoinType::kCountingAnti, {1, 1, 2, 3, 3});
  // INTERSECT ALL: {1:1, 2:2, 3:1}
  test(core::JoinType::kCountingLeftSemiFilter, {1, 2, 2, 3});
}

// Multiple columns as join keys.
TEST_F(CountingJoinTest, multipleKeys) {
  auto probeVector = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 1, 2, 2}),
       makeFlatVector<int32_t>({10, 10, 20, 10, 10})});
  auto buildVector = makeRowVector(
      {"u0", "u1"},
      {makeFlatVector<int32_t>({1, 1, 2}),
       makeFlatVector<int32_t>({10, 10, 10})});

  // Probe: {(1,10):2, (1,20):1, (2,10):2}, Build: {(1,10):2, (2,10):1}
  // EXCEPT ALL: {(1,20):1, (2,10):1}
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVector)
          .hashJoin(
              {"c0", "c1"},
              {"u0", "u1"},
              PlanBuilder(planNodeIdGenerator).values(buildVector).planNode(),
              "",
              {"c0", "c1"},
              core::JoinType::kCountingAnti)
          .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int32_t>({20, 10}),
  });
  assertEqualResults(
      {expected}, {AssertQueryBuilder(plan).copyResults(pool())});
}

// Null join keys never match anything (including other nulls).
TEST_F(CountingJoinTest, nullKeys) {
  // Probe: {1, null, 1, null, 2}, Build: {1, null, 3}
  auto probeVector = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 1, std::nullopt, 2})});
  auto buildVector = makeRowVector(
      {"u0"}, {makeNullableFlatVector<int32_t>({1, std::nullopt, 3})});

  auto test = [&](core::JoinType joinType,
                  const std::vector<std::optional<int32_t>>& expected) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan =
        PlanBuilder(planNodeIdGenerator)
            .values(probeVector)
            .hashJoin(
                {"c0"},
                {"u0"},
                PlanBuilder(planNodeIdGenerator).values(buildVector).planNode(),
                "",
                {"c0"},
                joinType)
            .planNode();
    assertEqualResults(
        {makeRowVector({makeNullableFlatVector<int32_t>(expected)})},
        {AssertQueryBuilder(plan).copyResults(pool())});
  };

  // EXCEPT ALL: non-null key 1 appears twice on probe and once on build, so one
  // copy is removed. Key 2 has no build match. Null keys have no match and pass
  // through.
  test(core::JoinType::kCountingAnti, {1, std::nullopt, std::nullopt, 2});

  // INTERSECT ALL: non-null key 1 appears twice on probe and once on build, so
  // one copy is kept. Null keys have no match and are excluded.
  test(core::JoinType::kCountingLeftSemiFilter, {1});
}

// Verifies that the build side deduplicates rows and stores only distinct keys.
TEST_F(CountingJoinTest, buildSideDedup) {
  // 10 distinct keys repeated 1'000 times each = 10'000 build rows.
  auto buildVector = makeRowVector(
      {"u0"}, {makeFlatVector<int32_t>(10, [](auto row) { return row; })});

  // Probe has values 0..14. Build has values 0..9, each repeated 1'000 times.
  // EXCEPT ALL: values 0..9 each appear once on probe but 1'000 times on build,
  // so all are consumed. Values 10..14 have no match and are emitted.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(makeRowVector({makeFlatVector<int32_t>(
                      15, [](auto row) { return row; })}))
                  .hashJoin(
                      {"c0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVector}, false, 1'000)
                          .planNode(),
                      "",
                      {"c0"},
                      core::JoinType::kCountingAnti)
                  .planNode();

  auto expected =
      makeRowVector({makeFlatVector<int32_t>({10, 11, 12, 13, 14})});
  auto task = AssertQueryBuilder(plan).assertResults(expected);

  // Verify the build received 10'000 rows but the hash table has only 10
  // distinct keys.
  auto planStats = toPlanStats(task->taskStats());
  const auto& joinStats = planStats.at(plan->id());
  EXPECT_EQ(10'000, joinStats.operatorStats.at("HashBuild")->inputRows);
  EXPECT_EQ(
      10,
      joinStats.customStats.at(std::string(BaseHashTable::kNumDistinct)).sum);
}

} // namespace
} // namespace facebook::velox::exec::test
