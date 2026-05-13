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

#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class CudfNestedLoopJoinTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }

  template <typename T>
  VectorPtr sequence(vector_size_t size, T start = 0) {
    return makeFlatVector<int32_t>(
        size, [start](auto row) { return start + row; });
  }
};

// Test 1: Simple cross join (no filter) - the simplest case
TEST_F(CudfNestedLoopJoinTest, crossJoin) {
  // Probe: 3 rows [0, 1, 2]
  auto probeVectors = {makeRowVector({sequence<int32_t>(3)})};

  // Build: 2 rows [0, 1]
  auto buildVectors = {makeRowVector({sequence<int32_t>(2)})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  // Cross join produces 3 * 2 = 6 rows
  assertQuery(plan, "SELECT * FROM t, u");
}

// Test 2: Cross join with empty build side
TEST_F(CudfNestedLoopJoinTest, crossJoinEmptyBuild) {
  auto probeVectors = {makeRowVector({sequence<int32_t>(5)})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  // Cross join with empty build produces 0 rows
  assertQuery(plan, "SELECT * FROM t, u");
}

// Test 3: Cross join with empty probe side
TEST_F(CudfNestedLoopJoinTest, crossJoinEmptyProbe) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({sequence<int32_t>(5)})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  // Cross join with empty probe produces 0 rows
  assertQuery(plan, "SELECT * FROM t, u");
}

// Both probe and build sides are empty.
TEST_F(CudfNestedLoopJoinTest, bothEmpty) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int64_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  assertQuery(plan, "SELECT * FROM t, u");
}

// Test 4: Cross join with single row build (common pattern: scalar subquery)
TEST_F(CudfNestedLoopJoinTest, crossJoinSingleRowBuild) {
  auto probeVectors = {makeRowVector({sequence<int32_t>(100)})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({42})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  // 100 probe rows * 1 build row = 100 rows
  assertQuery(plan, "SELECT * FROM t, u");
}

// Test 5: Inner join with filter condition
TEST_F(CudfNestedLoopJoinTest, innerJoinWithFilter) {
  auto probeVectors = {makeRowVector({sequence<int32_t>(10)})};
  auto buildVectors = {makeRowVector({sequence<int32_t>(10)})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0", // Filter: probe < build (non-equi join)
                      {"c0", "u_c0"},
                      core::JoinType::kInner)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t INNER JOIN u ON t.c0 < u.c0");
}

// Test 6: Multiple batches (tests streaming behavior)
TEST_F(CudfNestedLoopJoinTest, multipleBatches) {
  std::vector<RowVectorPtr> probeVectors;
  for (int i = 0; i < 5; ++i) {
    probeVectors.push_back(makeRowVector({sequence<int32_t>(100, i * 100)}));
  }

  std::vector<RowVectorPtr> buildVectors;
  for (int i = 0; i < 3; ++i) {
    buildVectors.push_back(makeRowVector({sequence<int32_t>(50, i * 50)}));
  }

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeVectors)
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  // 500 probe rows * 150 build rows = 75,000 rows
  assertQuery(plan, "SELECT * FROM t, u");
}

// Larger cross join (100 x 50 = 5,000 rows) with per-row value verification.
TEST_F(CudfNestedLoopJoinTest, largerCrossJoin) {
  constexpr int32_t kProbeRows = 100;
  constexpr int32_t kBuildRows = 50;

  auto probeVectors = {makeRowVector(
      {makeFlatVector<int32_t>(kProbeRows, [](auto row) { return row; })})};
  auto buildVectors = {makeRowVector({makeFlatVector<int64_t>(
      kBuildRows, [](auto row) { return 1000 + row; })})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  assertQuery(plan, "SELECT * FROM t, u");
}

// Three build batches of varying sizes, verifies build-side concatenation.
TEST_F(CudfNestedLoopJoinTest, mergeBuildVectors) {
  std::vector<RowVectorPtr> buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 2})}),
      makeRowVector({makeFlatVector<int32_t>({3, 4})}),
      makeRowVector({makeFlatVector<int32_t>({5, 6, 7})}),
  };
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  // 2 probe rows * 7 build rows = 14 rows
  assertQuery(plan, "SELECT * FROM t, u");
}

// Combined empty scenarios in a single test (ported from CPU
// NestedLoopJoinTest.emptyBuildOrProbeWithoutFilter).
TEST_F(CudfNestedLoopJoinTest, emptyBuildOrProbeWithoutFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto emptyProbe = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int64_t>({10, 20})})};
  auto emptyBuild = {makeRowVector({makeFlatVector<int64_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto makePlan = [](const auto& probe, const auto& build) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return PlanBuilder(planNodeIdGenerator)
        .values({probe})
        .nestedLoopJoin(
            PlanBuilder(planNodeIdGenerator)
                .values({build})
                .project({"c0 AS u_c0"})
                .planNode(),
            {"c0", "u_c0"})
        .planNode();
  };

  // probe x emptyBuild = 0 rows
  assertQuery(
      makePlan(probeVectors, emptyBuild), "SELECT * FROM t WHERE false");

  // emptyProbe x build = 0 rows
  assertQuery(
      makePlan(emptyProbe, buildVectors), "SELECT * FROM t WHERE false");

  // emptyProbe x emptyBuild = 0 rows
  assertQuery(makePlan(emptyProbe, emptyBuild), "SELECT * FROM t WHERE false");

  // probe x build = 6 rows
  assertQuery(makePlan(probeVectors, buildVectors), "SELECT * FROM t, u");
}

// Test 10: Multiple column types
TEST_F(CudfNestedLoopJoinTest, multipleColumnTypes) {
  auto probeVectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
      makeFlatVector<std::string>({"a", "b", "c"}),
  })};

  auto buildVectors = {makeRowVector({
      makeFlatVector<int64_t>({10, 20}),
      makeFlatVector<float>({1.0f, 2.0f}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0", "c1 AS u_c1"})
                          .planNode(),
                      {"c0", "c1", "c2", "u_c0", "u_c1"})
                  .planNode();

  assertQuery(plan, "SELECT * FROM t, u");
}

// Output column order differs from probe-first/build-second.
// Verifies name-based column resolution handles arbitrary output layouts.
TEST_F(CudfNestedLoopJoinTest, reorderedOutputColumns) {
  auto probeVectors = {makeRowVector(
      {"p0", "p1"},
      {
          makeFlatVector<int32_t>({1, 2}),
          makeFlatVector<double>({1.5, 2.5}),
      })};

  auto buildVectors = {makeRowVector(
      {"b0"},
      {
          makeFlatVector<int64_t>({99}),
      })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  // Request build column first, then probe columns in reverse order
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .planNode(),
                      {"b0", "p1", "p0"})
                  .planNode();

  assertQuery(plan, "SELECT u.b0, t.p1, t.p0 FROM t, u");
}

// Cross join with null-containing columns on both sides.
TEST_F(CudfNestedLoopJoinTest, crossJoinWithNulls) {
  auto probeVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 3}),
      makeNullableFlatVector<StringView>({"a", "b", std::nullopt}),
  })};

  auto buildVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt, 20}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "c1", "u_c0"})
                  .planNode();

  // 3 * 2 = 6 rows, nulls preserved
  assertQuery(plan, "SELECT * FROM t, u");
}

// Filtered join with null-containing columns.
// Verifies null comparison semantics: null < X evaluates to null (non-match).
TEST_F(CudfNestedLoopJoinTest, filteredJoinWithNulls) {
  auto probeVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 5, 3}),
  })};

  auto buildVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({2, std::nullopt, 4}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kInner)
                  .planNode();

  // Null comparisons return null (non-match in inner join)
  assertQuery(plan, "SELECT t.c0, u.c0 FROM t INNER JOIN u ON t.c0 < u.c0");
}

// Filtered join with empty build side.
TEST_F(CudfNestedLoopJoinTest, filteredJoinEmptyBuild) {
  auto probeVectors = {makeRowVector({sequence<int32_t>(5)})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kInner)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t INNER JOIN u ON t.c0 < u.c0");
}

// Filtered join with empty probe side.
TEST_F(CudfNestedLoopJoinTest, filteredJoinEmptyProbe) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({sequence<int32_t>(5)})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kInner)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t INNER JOIN u ON t.c0 < u.c0");
}

// Multi-driver build coordination test.
// Verifies that multiple build operators correctly coordinate via
// allPeersFinished(). Build side uses parallelizable values (4 drivers each
// get all vectors), probe side is non-parallelizable (1 driver reads all).
// Build data is 4x due to driver replication.
TEST_F(CudfNestedLoopJoinTest, multiDriverBuild) {
  auto probeVectors = {
      makeRowVector({sequence<int32_t>(10)}),
      makeRowVector({sequence<int32_t>(10, 10)}),
  };

  // Build: 2 values that will be replicated 4x by 4 build drivers.
  auto buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({100, 200})}),
  };

  // DuckDB reference table has the 1x data
  createDuckDbTable("t", {probeVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  params.maxDrivers = 4;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({probeVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator)
                                .values({buildVectors}, true)
                                .filter("c0 = 100")
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            {"c0", "u_c0"})
                        .planNode();

  // Build side: 4 drivers × 1 matching row = 4 build rows (all value 100)
  // Probe side: 1 driver × 20 rows = 20 probe rows
  // Result: 20 * 4 = 80 rows
  assertQuery(
      params,
      "SELECT * FROM t, "
      "(SELECT * FROM UNNEST(ARRAY[100, 100, 100, 100]) AS t(c0)) u");
}

// Multi-driver with filter.
TEST_F(CudfNestedLoopJoinTest, multiDriverWithFilter) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5, 10})}),
  };

  auto buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
  };

  // DuckDB reference table has the 1x data
  createDuckDbTable("t", {probeVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  params.maxDrivers = 2;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({probeVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator)
                                .values({buildVectors}, true)
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            "c0 < u_c0",
                            {"c0", "u_c0"},
                            core::JoinType::kInner)
                        .planNode();

  // Build side: 2 drivers × {3, 7} = {3, 7, 3, 7}
  // Probe: {1, 5, 10}
  // Matches (c0 < u_c0): 1<3, 1<7, 5<7, 1<3, 1<7, 5<7 = 6 rows
  assertQuery(
      params,
      "SELECT t.c0, u.c0 FROM t "
      "INNER JOIN (SELECT * FROM UNNEST(ARRAY[3, 7, 3, 7]) AS t(c0)) u "
      "ON t.c0 < u.c0");
}

// All-null columns on both sides.
TEST_F(CudfNestedLoopJoinTest, allNullColumns) {
  auto probeVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
  })};

  auto buildVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  // Cross join: all-null columns should produce 6 rows with all nulls
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"})
                  .planNode();

  assertQuery(plan, "SELECT * FROM t, u");

  // Filtered join: null < null is null (non-match), so 0 results
  planNodeIdGenerator->reset();
  auto filteredPlan = PlanBuilder(planNodeIdGenerator)
                          .values({probeVectors})
                          .nestedLoopJoin(
                              PlanBuilder(planNodeIdGenerator)
                                  .values({buildVectors})
                                  .project({"c0 AS u_c0"})
                                  .planNode(),
                              "c0 < u_c0",
                              {"c0", "u_c0"},
                              core::JoinType::kInner)
                          .planNode();

  assertQuery(
      filteredPlan, "SELECT t.c0, u.c0 FROM t INNER JOIN u ON t.c0 < u.c0");
}

// ============================================================================
// Left Join Tests
// ============================================================================

// Left join with filter: unmatched probe rows appear with null build columns.
TEST_F(CudfNestedLoopJoinTest, leftJoinWithFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 5, 10})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({3, 7})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  // Probe 1 matches {3,7}; probe 5 matches {7}; probe 10 has no match → null.
  assertQuery(plan, "SELECT t.c0, u.c0 FROM t LEFT JOIN u ON t.c0 < u.c0");
}

// Left join without filter: cross join (all combinations), no mismatches.
TEST_F(CudfNestedLoopJoinTest, leftJoinWithoutFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  // 3 * 2 = 6 rows, same as cross join.
  assertQuery(plan, "SELECT * FROM t LEFT JOIN u ON true");
}

// Left join with empty build: all probe rows emitted with null build columns.
TEST_F(CudfNestedLoopJoinTest, leftJoinEmptyBuild) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t LEFT JOIN u ON t.c0 < u.c0");
}

// Left join with empty build and no filter.
TEST_F(CudfNestedLoopJoinTest, leftJoinEmptyBuildNoFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  assertQuery(plan, "SELECT * FROM t LEFT JOIN u ON true");
}

// Left join with empty probe: no output.
TEST_F(CudfNestedLoopJoinTest, leftJoinEmptyProbe) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t LEFT JOIN u ON t.c0 < u.c0");
}

// Left join with null values in probe and build columns.
TEST_F(CudfNestedLoopJoinTest, leftJoinWithNulls) {
  auto probeVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 5, 3}),
  })};

  auto buildVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({2, std::nullopt, 4}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  // Null comparisons produce null → unmatched (null probe rows get null build).
  assertQuery(plan, "SELECT t.c0, u.c0 FROM t LEFT JOIN u ON t.c0 < u.c0");
}

// Left join with multiple probe and build batches.
TEST_F(CudfNestedLoopJoinTest, leftJoinMultipleBatches) {
  std::vector<RowVectorPtr> probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5})}),
      makeRowVector({makeFlatVector<int32_t>({10, 20})}),
  };

  std::vector<RowVectorPtr> buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
      makeRowVector({makeFlatVector<int32_t>({15})}),
  };

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeVectors)
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t LEFT JOIN u ON t.c0 < u.c0");
}

// Left join where no probe rows match any build rows.
TEST_F(CudfNestedLoopJoinTest, leftJoinNoMatches) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({100, 200})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  // All probe rows > all build rows, so no matches. All probe rows
  // emitted with null build columns.
  assertQuery(plan, "SELECT t.c0, u.c0 FROM t LEFT JOIN u ON t.c0 < u.c0");
}

// Left join with multiple column types including strings.
TEST_F(CudfNestedLoopJoinTest, leftJoinMultipleColumnTypes) {
  auto probeVectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  })};

  auto buildVectors = {makeRowVector({
      makeFlatVector<int64_t>({10, 20}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c1 < cast(u_c0 as double)",
                      {"c0", "c1", "u_c0"},
                      core::JoinType::kLeft)
                  .planNode();

  assertQuery(
      plan,
      "SELECT t.c0, t.c1, u.c0 FROM t LEFT JOIN u ON t.c1 < CAST(u.c0 AS DOUBLE)");
}

// Multi-driver left join.
TEST_F(CudfNestedLoopJoinTest, leftJoinMultiDriver) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5, 10})}),
  };

  auto buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
  };

  createDuckDbTable("t", {probeVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  params.maxDrivers = 2;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({probeVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator)
                                .values({buildVectors}, true)
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            "c0 < u_c0",
                            {"c0", "u_c0"},
                            core::JoinType::kLeft)
                        .planNode();

  // Build side: 2 drivers × {3, 7} = {3, 7, 3, 7}
  assertQuery(
      params,
      "SELECT t.c0, u.c0 FROM t "
      "LEFT JOIN (SELECT * FROM UNNEST(ARRAY[3, 7, 3, 7]) AS t(c0)) u "
      "ON t.c0 < u.c0");
}

// ============================================================================
// Right Join Tests
// ============================================================================

// Right join with filter: unmatched build rows appear with null probe columns.
TEST_F(CudfNestedLoopJoinTest, rightJoinWithFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 5, 10})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({3, 7, 20})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  // Build row 20 has no matching probe row → emitted with null probe column.
  assertQuery(plan, "SELECT t.c0, u.c0 FROM t RIGHT JOIN u ON t.c0 < u.c0");
}

// Right join without filter: cross join, all build rows present.
TEST_F(CudfNestedLoopJoinTest, rightJoinWithoutFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20, 30})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  // 2 * 3 = 6 rows, same as cross join.
  assertQuery(plan, "SELECT * FROM t RIGHT JOIN u ON true");
}

// Right join with empty probe: all build rows emitted with null probe columns.
TEST_F(CudfNestedLoopJoinTest, rightJoinEmptyProbe) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t RIGHT JOIN u ON t.c0 < u.c0");
}

// Right join with empty build: no output.
TEST_F(CudfNestedLoopJoinTest, rightJoinEmptyBuild) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t RIGHT JOIN u ON t.c0 < u.c0");
}

// Right join where no probe rows match any build rows.
TEST_F(CudfNestedLoopJoinTest, rightJoinNoMatches) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({100, 200})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  // All probe > all build with c0 < u_c0 filter → no matches.
  // All build rows emitted with null probe columns.
  assertQuery(plan, "SELECT t.c0, u.c0 FROM t RIGHT JOIN u ON t.c0 < u.c0");
}

// Right join with multiple probe and build batches.
TEST_F(CudfNestedLoopJoinTest, rightJoinMultipleBatches) {
  std::vector<RowVectorPtr> probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5})}),
      makeRowVector({makeFlatVector<int32_t>({10, 20})}),
  };

  std::vector<RowVectorPtr> buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
      makeRowVector({makeFlatVector<int32_t>({15})}),
  };

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeVectors)
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t RIGHT JOIN u ON t.c0 < u.c0");
}

// Right join with null values.
TEST_F(CudfNestedLoopJoinTest, rightJoinWithNulls) {
  auto probeVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt, 5}),
  })};

  auto buildVectors = {makeRowVector({
      makeNullableFlatVector<int32_t>({2, std::nullopt, 4}),
  })};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kRight)
                  .planNode();

  assertQuery(plan, "SELECT t.c0, u.c0 FROM t RIGHT JOIN u ON t.c0 < u.c0");
}

// Multi-driver right join: build mismatch flags must be merged across peers.
TEST_F(CudfNestedLoopJoinTest, rightJoinMultiDriver) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5, 10})}),
  };

  auto buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
  };

  createDuckDbTable("t", {probeVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  params.maxDrivers = 2;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({probeVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator)
                                .values({buildVectors}, true)
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            "c0 < u_c0",
                            {"c0", "u_c0"},
                            core::JoinType::kRight)
                        .planNode();

  // Build side: 2 drivers × {3, 7} = {3, 7, 3, 7}
  assertQuery(
      params,
      "SELECT t.c0, u.c0 FROM t "
      "RIGHT JOIN (SELECT * FROM UNNEST(ARRAY[3, 7, 3, 7]) AS t(c0)) u "
      "ON t.c0 < u.c0");
}

// --- Full outer join tests ---

// Full join with filter: mismatches on both sides.
TEST_F(CudfNestedLoopJoinTest, fullJoinWithFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 5, 10})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({3, 7})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  // Probe 1 matches {3,7}; probe 5 matches {7}; probe 10 has no match → null.
  // Build 3 matches {5,10} but not 1 via < — actually build 3 is matched by
  // probe rows 1,5 (1<3, 5>3 no). Let DuckDB compute the reference.
  assertQuery(
      plan, "SELECT t.c0, u.c0 FROM t FULL OUTER JOIN u ON t.c0 < u.c0");
}

// Full join without filter: cross join — all rows match, no mismatches.
TEST_F(CudfNestedLoopJoinTest, fullJoinWithoutFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  // 3 * 2 = 6 rows, same as cross join.
  assertQuery(plan, "SELECT * FROM t FULL OUTER JOIN u ON true");
}

// Full join with empty build: all probe rows emitted with null build columns.
TEST_F(CudfNestedLoopJoinTest, fullJoinEmptyBuild) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, u.c0 FROM t FULL OUTER JOIN u ON t.c0 < u.c0");
}

// Full join with empty probe: all build rows emitted with null probe columns.
TEST_F(CudfNestedLoopJoinTest, fullJoinEmptyProbe) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20, 30})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, u.c0 FROM t FULL OUTER JOIN u ON t.c0 < u.c0");
}

// Full join where no rows match: all probe and build rows are mismatches.
TEST_F(CudfNestedLoopJoinTest, fullJoinNoMatches) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({100, 200, 300})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  // probe rows all >= build rows, so c0 < u_c0 never true.
  assertQuery(
      plan, "SELECT t.c0, u.c0 FROM t FULL OUTER JOIN u ON t.c0 < u.c0");
}

// Full join with multiple build batches.
TEST_F(CudfNestedLoopJoinTest, fullJoinMultipleBatches) {
  std::vector<RowVectorPtr> probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5})}),
      makeRowVector({makeFlatVector<int32_t>({10})}),
  };
  std::vector<RowVectorPtr> buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3})}),
      makeRowVector({makeFlatVector<int32_t>({7})}),
  };

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, u.c0 FROM t FULL OUTER JOIN u ON t.c0 < u.c0");
}

// Full join with nulls in data.
TEST_F(CudfNestedLoopJoinTest, fullJoinWithNulls) {
  auto probeVectors = {
      makeRowVector({makeNullableFlatVector<int32_t>({1, std::nullopt, 10})})};
  auto buildVectors = {
      makeRowVector({makeNullableFlatVector<int32_t>({3, std::nullopt})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "u_c0"},
                      core::JoinType::kFull)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, u.c0 FROM t FULL OUTER JOIN u ON t.c0 < u.c0");
}

// Multi-driver full join: build mismatch flags must be merged across peers,
// and probe mismatch tracking works per-driver independently.
TEST_F(CudfNestedLoopJoinTest, fullJoinMultiDriver) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5, 10})}),
  };

  auto buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
  };

  createDuckDbTable("t", {probeVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  params.maxDrivers = 2;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({probeVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator)
                                .values({buildVectors}, true)
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            "c0 < u_c0",
                            {"c0", "u_c0"},
                            core::JoinType::kFull)
                        .planNode();

  // Build side: 2 drivers × {3, 7} = {3, 7, 3, 7}
  assertQuery(
      params,
      "SELECT t.c0, u.c0 FROM t "
      "FULL OUTER JOIN (SELECT * FROM UNNEST(ARRAY[3, 7, 3, 7]) AS t(c0)) u "
      "ON t.c0 < u.c0");
}

// --- LeftSemiProject join tests ---

// LeftSemiProject with filter: probe rows + boolean match column.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectWithFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 5, 10})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({3, 7})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  // 1 < 3 true, 1 < 7 true → match=true; 5 < 7 true → match=true;
  // 10 < nothing → match=false.
  assertQuery(
      plan, "SELECT t.c0, EXISTS (SELECT 1 FROM u WHERE t.c0 < u.c0) FROM t");
}

// LeftSemiProject without filter: non-empty build → all true.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectWithoutFilter) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  // No filter + non-empty build → all rows match (true).
  assertQuery(plan, "SELECT t.c0, EXISTS (SELECT 1 FROM u) FROM t");
}

// LeftSemiProject with empty build: all false.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectEmptyBuild) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2, 3})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  // Empty build → all false.
  assertQuery(
      plan, "SELECT t.c0, EXISTS (SELECT 1 FROM u WHERE t.c0 < u.c0) FROM t");
}

// LeftSemiProject with empty probe: no output.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectEmptyProbe) {
  auto probeVectors = {makeRowVector({makeFlatVector<int32_t>({})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({10, 20})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, EXISTS (SELECT 1 FROM u WHERE t.c0 < u.c0) FROM t");
}

// LeftSemiProject where no rows match: all false.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectNoMatches) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({100, 200, 300})})};
  auto buildVectors = {makeRowVector({makeFlatVector<int32_t>({1, 2})})};

  createDuckDbTable("t", {probeVectors});
  createDuckDbTable("u", {buildVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, EXISTS (SELECT 1 FROM u WHERE t.c0 < u.c0) FROM t");
}

// LeftSemiProject with multiple build batches.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectMultipleBatches) {
  std::vector<RowVectorPtr> probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5})}),
      makeRowVector({makeFlatVector<int32_t>({10})}),
  };
  std::vector<RowVectorPtr> buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3})}),
      makeRowVector({makeFlatVector<int32_t>({7})}),
  };

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  assertQuery(
      plan, "SELECT t.c0, EXISTS (SELECT 1 FROM u WHERE t.c0 < u.c0) FROM t");
}

// LeftSemiProject with nulls in data. Velox semantics: unmatched rows
// (including those with NULL comparisons) get match=false, not null.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectWithNulls) {
  auto probeVectors = {
      makeRowVector({makeNullableFlatVector<int32_t>({1, std::nullopt, 10})})};
  auto buildVectors = {
      makeRowVector({makeNullableFlatVector<int32_t>({3, std::nullopt})})};

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({probeVectors})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({buildVectors})
                          .project({"c0 AS u_c0"})
                          .planNode(),
                      "c0 < u_c0",
                      {"c0", "match"},
                      core::JoinType::kLeftSemiProject)
                  .planNode();

  // 1 < 3 → true; null < anything → false (not null); 10 < nothing → false.
  auto expected = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, std::nullopt, 10}),
       makeFlatVector<bool>({true, false, false})});
  AssertQueryBuilder builder{plan};
  auto result = builder.copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

// Multi-driver LeftSemiProject.
TEST_F(CudfNestedLoopJoinTest, leftSemiProjectMultiDriver) {
  auto probeVectors = {
      makeRowVector({makeFlatVector<int32_t>({1, 5, 10})}),
  };

  auto buildVectors = {
      makeRowVector({makeFlatVector<int32_t>({3, 7})}),
  };

  createDuckDbTable("t", {probeVectors});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  params.maxDrivers = 2;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .values({probeVectors})
                        .nestedLoopJoin(
                            PlanBuilder(planNodeIdGenerator)
                                .values({buildVectors}, true)
                                .project({"c0 AS u_c0"})
                                .planNode(),
                            "c0 < u_c0",
                            {"c0", "match"},
                            core::JoinType::kLeftSemiProject)
                        .planNode();

  // Build side: 2 drivers × {3, 7} = {3, 7, 3, 7}. Existence is the same.
  assertQuery(
      params,
      "SELECT t.c0, EXISTS "
      "(SELECT 1 FROM UNNEST(ARRAY[3, 7, 3, 7]) AS u(c0) "
      "WHERE t.c0 < u.c0) FROM t");
}

// TODO: Zero-column build side is not yet supported. cudf::table with zero
// columns reports num_rows() == 0, causing the operator to treat a non-empty
// build as empty. Fixing this requires the bridge to carry row counts
// separately. See CPU NestedLoopJoinTest::zeroColumnBuild for the expected
// behavior.
