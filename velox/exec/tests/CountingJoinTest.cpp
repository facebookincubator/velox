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
                        joinType,
                        /*nullAware=*/false,
                        /*nullAsValue=*/true)
                    .planNode();
    AssertQueryBuilder(plan).assertResults(
        makeRowVector({makeFlatVector<int32_t>(expected)}));
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
                joinType,
                /*nullAware=*/false,
                /*nullAsValue=*/true)
            .planNode();
    AssertQueryBuilder(plan).assertResults(
        makeRowVector({makeFlatVector<int32_t>(expected)}));
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
              core::JoinType::kCountingAnti,
              /*nullAware=*/false,
              /*nullAsValue=*/true)
          .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int32_t>({20, 10}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

// Null join keys match each other with nullAsValue (IS NOT DISTINCT FROM
// semantics), as required by SQL set operations (EXCEPT ALL, INTERSECT ALL).
TEST_F(CountingJoinTest, nullKeys) {
  // Probe: {1, null, 1, null, 2}, Build: {1, null, 3}
  auto probeVector = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 1, std::nullopt, 2})});
  auto buildVector = makeRowVector(
      {"u0"}, {makeNullableFlatVector<int32_t>({1, std::nullopt, 3})});

  auto test = [&](core::JoinType joinType,
                  const std::vector<std::optional<int32_t>>& expected) {
    SCOPED_TRACE(core::JoinTypeName::toName(joinType));
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values({probeVector})
                    .hashJoin(
                        {"c0"},
                        {"u0"},
                        PlanBuilder(planNodeIdGenerator)
                            .values({buildVector})
                            .planNode(),
                        "",
                        {"c0"},
                        joinType,
                        /*nullAware=*/false,
                        /*nullAsValue=*/true)
                    .planNode();
    AssertQueryBuilder(plan).assertResults(
        makeRowVector({makeNullableFlatVector<int32_t>(expected)}));
  };

  // EXCEPT ALL: null keys match. Probe has 2 nulls, build has 1 null, so 1
  // null passes through. Key 1 appears twice on probe, once on build, so 1
  // copy passes through. Key 2 has no build match.
  test(core::JoinType::kCountingAnti, {1, std::nullopt, 2});

  // INTERSECT ALL: null keys match. min(2, 1) = 1 null kept.
  // Key 1: min(2, 1) = 1. Key 2: no match.
  test(core::JoinType::kCountingLeftSemiFilter, {1, std::nullopt});
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
                      core::JoinType::kCountingAnti,
                      /*nullAware=*/false,
                      /*nullAsValue=*/true)
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

// Verifies that counts are correctly merged when multiple build drivers process
// the same keys. Each driver builds its own hash table with per-key counts;
// during the merge in prepareJoinTable, duplicate keys must have their counts
// summed rather than silently dropped.
//
// Tests three key configurations to exercise all hash table modes:
//  - Small INT32 keys {1, 2}: array mode (arrayPushRow path).
//  - Two INT32 key columns with ~1500 distinct values each: normalized-key
//    mode (buildFullProbe with normalizedKey comparison).
//  - Complex-type (ARRAY(INT32)) keys: hash mode (buildFullProbe with
//    compareKeys).
TEST_F(CountingJoinTest, multipleBuildDrivers) {
  constexpr int32_t kNumDrivers = 4;

  auto test = [&](const RowVectorPtr& probeVector,
                  const RowVectorPtr& buildVector,
                  core::JoinType joinType,
                  BaseHashTable::HashMode expectedHashMode,
                  const std::vector<VectorPtr>& expectedChildren) {
    auto expected =
        makeRowVector(probeVector->rowType()->names(), expectedChildren);
    auto probeKeys = probeVector->rowType()->names();

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .values({probeVector})
                    .hashJoin(
                        probeKeys,
                        buildVector->rowType()->names(),
                        PlanBuilder(planNodeIdGenerator)
                            .values({buildVector}, /*parallelizable=*/true)
                            .planNode(),
                        "",
                        probeKeys,
                        joinType,
                        /*nullAware=*/false,
                        /*nullAsValue=*/true)
                    .planNode();
    auto task = AssertQueryBuilder(plan)
                    .maxDrivers(kNumDrivers)
                    .assertResults(expected);

    // Verify that all build drivers ran. Each driver processes 1 batch.
    auto planStats = toPlanStats(task->taskStats());
    const auto& joinStats = planStats.at(plan->id());

    const auto& buildStats = *joinStats.operatorStats.at("HashBuild");
    EXPECT_EQ(kNumDrivers, buildStats.numDrivers);
    EXPECT_EQ(kNumDrivers, buildStats.inputVectors);
    EXPECT_EQ(buildVector->size() * kNumDrivers, buildStats.inputRows);

    // Probe side runs single-threaded (Values node is not parallelizable by
    // default). It processes 1 batch.
    const auto& probeStats = *joinStats.operatorStats.at("HashProbe");
    EXPECT_EQ(1, probeStats.numDrivers);
    EXPECT_EQ(1, probeStats.inputVectors);
    EXPECT_EQ(probeVector->size(), probeStats.inputRows);

    EXPECT_EQ(
        static_cast<int64_t>(expectedHashMode),
        buildStats.customStats.at(std::string(BaseHashTable::kHashMode)).sum);
  };

  // Small keys {1, 2}: triggers array-based hash mode.
  // Build: {1, 1, 2}, each of 4 drivers => merged {1:8, 2:4}.
  // Probe: {1:3, 2:5}.
  {
    SCOPED_TRACE("small keys (array mode)");
    auto buildVector =
        makeRowVector({"u0"}, {makeFlatVector<int32_t>({1, 1, 2})});
    auto probeVector =
        makeRowVector({makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2, 2, 2})});

    // EXCEPT ALL: {1: max(3-8,0)=0, 2: max(5-4,0)=1} => {2}.
    test(
        probeVector,
        buildVector,
        core::JoinType::kCountingAnti,
        BaseHashTable::HashMode::kArray,
        {makeFlatVector<int32_t>({2})});
    // INTERSECT ALL: {1: min(3,8)=3, 2: min(5,4)=4}.
    test(
        probeVector,
        buildVector,
        core::JoinType::kCountingLeftSemiFilter,
        BaseHashTable::HashMode::kArray,
        {makeFlatVector<int32_t>({1, 1, 1, 2, 2, 2, 2})});
  }

  // Two INT32 key columns with ~1500 distinct values each. The combined
  // cardinality (1500 * 1500 = 2.25M) exceeds kArrayHashMaxSize (2M),
  // forcing normalized-key mode (buildFullProbe with normalizedKey comparison).
  {
    SCOPED_TRACE("multi-column keys (normalized-key mode)");
    constexpr int32_t kNumKeys = 1'500;

    // Build: 2-column keys (i, i) for i in 0..kNumKeys-1, plus duplicates of
    // (0, 0) and (1, 1). Both columns are identical — we need 2 columns so
    // the combined cardinality (1500 * 1500 = 2.25M) exceeds kArrayHashMaxSize
    // (2M).
    auto buildKeys = makeFlatVector<int32_t>(kNumKeys + 2, [](auto row) {
      return row < kNumKeys ? row : row - kNumKeys;
    });

    // Probe: key (0,0) appears 10 times, key (1,1) appears 5 times.
    auto probeKeys =
        makeFlatVector<int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1});

    // Each of 4 drivers: (0,0) count=2, (1,1) count=2, others count=1.
    // After merge: (0,0) count=8, (1,1) count=8.
    // EXCEPT ALL: (0,0): max(10-8,0)=2, (1,1): max(5-8,0)=0.
    auto expectedAnti = makeFlatVector<int32_t>({0, 0});
    test(
        makeRowVector({probeKeys, probeKeys}),
        makeRowVector({"u0", "u1"}, {buildKeys, buildKeys}),
        core::JoinType::kCountingAnti,
        BaseHashTable::HashMode::kNormalizedKey,
        {expectedAnti, expectedAnti});
    // INTERSECT ALL: (0,0): min(10,8)=8, (1,1): min(5,8)=5.
    auto expectedSemi =
        makeFlatVector<int32_t>({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
    test(
        makeRowVector({probeKeys, probeKeys}),
        makeRowVector({"u0", "u1"}, {buildKeys, buildKeys}),
        core::JoinType::kCountingLeftSemiFilter,
        BaseHashTable::HashMode::kNormalizedKey,
        {expectedSemi, expectedSemi});
  }

  // ARRAY(INT32) type keys do not support value IDs in VectorHasher, forcing
  // kHash mode (buildFullProbe with compareKeys).
  {
    SCOPED_TRACE("complex-type keys (hash mode)");
    auto buildVector =
        makeRowVector({"u0"}, {makeArrayVector<int32_t>({{1}, {1}, {2}})});
    auto probeVector = makeRowVector(
        {makeArrayVector<int32_t>({{1}, {1}, {1}, {2}, {2}, {2}, {2}, {2}})});

    // EXCEPT ALL: {[1]: max(3-8,0)=0, [2]: max(5-4,0)=1} => {[2]}.
    test(
        probeVector,
        buildVector,
        core::JoinType::kCountingAnti,
        BaseHashTable::HashMode::kHash,
        {makeArrayVector<int32_t>({{2}})});
    // INTERSECT ALL: {[1]: min(3,8)=3, [2]: min(5,4)=4}.
    test(
        probeVector,
        buildVector,
        core::JoinType::kCountingLeftSemiFilter,
        BaseHashTable::HashMode::kHash,
        {makeArrayVector<int32_t>({{1}, {1}, {1}, {2}, {2}, {2}, {2}})});
  }
}

} // namespace
} // namespace facebook::velox::exec::test
