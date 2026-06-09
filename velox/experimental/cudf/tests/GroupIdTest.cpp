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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class CudfGroupIdTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }
};

// Test matching core Velox: AggregationTest.groupingSets
TEST_F(CudfGroupIdTest, groupingSets) {
  vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"k1", "k2", "a", "b"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<std::string>(
              size, [](auto row) { return std::string(row % 12, 'x'); }),
      });

  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .groupId({"k1", "k2"}, {{"k1"}, {"k2"}}, {"a", "b"})
          .singleAggregation(
              {"k1", "k2", "group_id"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY GROUPING SETS ((k1), (k2))");

  // Distinct aggregations.
  plan = PlanBuilder()
             .values({data})
             .groupId({"k1", "k2"}, {{"k1"}, {"k2"}}, {})
             .singleAggregation({"k1", "k2", "group_id"}, {})
             .project({"k1", "k2"})
             .planNode();

  assertQuery(
      plan, "SELECT k1, k2 FROM tmp GROUP BY GROUPING SETS ((k1), (k2))");

  // Distinct aggregations with global grouping sets.
  plan = PlanBuilder()
             .values({data})
             .groupId({"k1", "k2"}, {{"k1"}, {"k2"}, {}}, {})
             .singleAggregation({"k1", "k2", "group_id"}, {})
             .project({"k1", "k2"})
             .planNode();

  assertQuery(
      plan, "SELECT k1, k2 FROM tmp GROUP BY GROUPING SETS ((k1), (k2), ())");

  // Cube.
  plan =
      PlanBuilder()
          .values({data})
          .groupId({"k1", "k2"}, {{"k1", "k2"}, {"k1"}, {"k2"}, {}}, {"a", "b"})
          .singleAggregation(
              {"k1", "k2", "group_id"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY CUBE (k1, k2)");

  // Rollup.
  plan = PlanBuilder()
             .values({data})
             .groupId({"k1", "k2"}, {{"k1", "k2"}, {"k1"}, {}}, {"a", "b"})
             .singleAggregation(
                 {"k1", "k2", "group_id"},
                 {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
             .project({"k1", "k2", "count_1", "sum_a", "max_b"})
             .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY ROLLUP (k1, k2)");
}

// Test matching core Velox: AggregationTest.groupingSetsOutput
TEST_F(CudfGroupIdTest, groupingSetsOutput) {
  vector_size_t size = 1'000;
  auto data = makeRowVector(
      {"k1", "k2", "a", "b"},
      {
          makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
          makeFlatVector<int64_t>(size, [](auto row) { return row; }),
          makeFlatVector<std::string>(
              size, [](auto row) { return std::string(row % 12, 'x'); }),
      });

  createDuckDbTable({data});

  core::PlanNodePtr reversedOrderGroupIdNode;
  core::PlanNodePtr orderGroupIdNode;
  auto reversedOrderPlan =
      PlanBuilder()
          .values({data})
          .groupId({"k2", "k1"}, {{"k2", "k1"}, {}}, {"a", "b"})
          .capturePlanNode(reversedOrderGroupIdNode)
          .singleAggregation(
              {"k2", "k1", "group_id"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  auto orderPlan =
      PlanBuilder()
          .values({data})
          .groupId({"k1", "k2"}, {{"k1", "k2"}, {}}, {"a", "b"})
          .capturePlanNode(orderGroupIdNode)
          .singleAggregation(
              {"k1", "k2", "group_id"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  auto reversedOrderExpectedRowType =
      ROW({"k2", "k1", "a", "b", "group_id"},
          {BIGINT(), BIGINT(), BIGINT(), VARCHAR(), BIGINT()});
  auto orderExpectedRowType =
      ROW({"k1", "k2", "a", "b", "group_id"},
          {BIGINT(), BIGINT(), BIGINT(), VARCHAR(), BIGINT()});
  ASSERT_EQ(
      *reversedOrderGroupIdNode->outputType(), *reversedOrderExpectedRowType);
  ASSERT_EQ(*orderGroupIdNode->outputType(), *orderExpectedRowType);

  CursorParameters orderParams;
  orderParams.planNode = orderPlan;
  auto orderResult = readCursor(orderParams);

  CursorParameters reversedOrderParams;
  reversedOrderParams.planNode = reversedOrderPlan;
  auto reversedOrderResult = readCursor(reversedOrderParams);

  assertEqualResults(orderResult.second, reversedOrderResult.second);
}

// Test matching core Velox: AggregationTest.groupingSetsSameKey
TEST_F(CudfGroupIdTest, groupingSetsSameKey) {
  auto data = makeRowVector(
      {"o_key", "o_status"},
      {makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
       makeFlatVector<std::string>({"", "x", "xx", "xxx", "xxxx"})});

  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .groupId(
                      {"o_key", "o_key as o_key_1"},
                      {{"o_key", "o_key_1"}, {"o_key"}, {"o_key_1"}, {}},
                      {"o_status"})
                  .singleAggregation(
                      {"o_key", "o_key_1", "group_id"},
                      {"max(o_status) as max_o_status"})
                  .project({"o_key", "o_key_1", "max_o_status"})
                  .planNode();

  assertQuery(
      plan,
      "SELECT o_key, o_key_1, max(o_status) as max_o_status FROM ("
      "select o_key, o_key as o_key_1, o_status FROM tmp) GROUP BY GROUPING SETS ((o_key, o_key_1), (o_key), (o_key_1), ())");
}

// Test matching core Velox: AggregationTest.groupingSetsEmptyInput
// Note: This test uses truly empty input (empty vectors) rather than filtered
// input, as global grouping set behavior with empty filtered input depends on
// aggregation operator behavior which may differ between CPU and GPU.
TEST_F(CudfGroupIdTest, groupingSetsEmptyInput) {
  auto data = makeRowVector(
      {"k1", "k2", "a"},
      {
          makeFlatVector<int64_t>({}),
          makeFlatVector<int64_t>({}),
          makeFlatVector<int64_t>({}),
      });

  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .groupId({"k1", "k2"}, {{"k1"}, {"k2"}}, {"a"})
          .singleAggregation({"k1", "k2", "group_id"}, {"sum(a) as sum_a"})
          .project({"k1", "k2", "sum_a"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, sum(a) FROM tmp GROUP BY GROUPING SETS ((k1), (k2))");
}

// Test: Verify NULL values in non-grouping columns and nulls within grouping
// key input columns are preserved correctly.
TEST_F(CudfGroupIdTest, nullHandling) {
  // Input has nulls in grouping key columns: c0 has null at row 1,
  // c1 has null at row 0.
  auto input = makeRowVector({
      makeNullableFlatVector<int32_t>({1, std::nullopt}),
      makeNullableFlatVector<int32_t>({std::nullopt, 20}),
      makeFlatVector<int64_t>({100, 200}),
  });

  // GROUPING SETS ((c0), (c1))
  // For grouping set (c0): c1 should be NULL (synthesized), c0 preserves input
  // nulls For grouping set (c1): c0 should be NULL (synthesized), c1 preserves
  // input nulls
  auto plan = PlanBuilder()
                  .values({input})
                  .groupId({"c0", "c1"}, {{"c0"}, {"c1"}}, {"c2"})
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  // Should have 4 rows (2 rows * 2 grouping sets)
  ASSERT_EQ(4, result->size());

  auto c0 = result->childAt(0)->asFlatVector<int32_t>();
  auto c1 = result->childAt(1)->asFlatVector<int32_t>();
  auto groupId = result->childAt(3)->asFlatVector<int64_t>();

  // First grouping set (c0): c1 is NULL (synthesized)
  // c0 preserves input values: row 0 has value 1, row 1 has null (from input)
  EXPECT_EQ(0, groupId->valueAt(0));
  EXPECT_EQ(0, groupId->valueAt(1));
  EXPECT_FALSE(c0->isNullAt(0));
  EXPECT_EQ(1, c0->valueAt(0));
  EXPECT_TRUE(c0->isNullAt(1)); // Input null preserved
  EXPECT_TRUE(c1->isNullAt(0)); // Synthesized null
  EXPECT_TRUE(c1->isNullAt(1)); // Synthesized null

  // Second grouping set (c1): c0 is NULL (synthesized)
  // c1 preserves input values: row 0 has null (from input), row 1 has value 20
  EXPECT_EQ(1, groupId->valueAt(2));
  EXPECT_EQ(1, groupId->valueAt(3));
  EXPECT_TRUE(c0->isNullAt(2)); // Synthesized null
  EXPECT_TRUE(c0->isNullAt(3)); // Synthesized null
  EXPECT_TRUE(c1->isNullAt(2)); // Input null preserved
  EXPECT_FALSE(c1->isNullAt(3));
  EXPECT_EQ(20, c1->valueAt(3));
}

// Test: Multiple batches
TEST_F(CudfGroupIdTest, multipleBatches) {
  auto batch1 = makeRowVector(
      {"k1", "k2", "a"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({10, 20}),
          makeFlatVector<int64_t>({100, 200}),
      });
  auto batch2 = makeRowVector(
      {"k1", "k2", "a"},
      {
          makeFlatVector<int64_t>({3, 4}),
          makeFlatVector<int64_t>({30, 40}),
          makeFlatVector<int64_t>({300, 400}),
      });

  createDuckDbTable({batch1, batch2});

  auto plan =
      PlanBuilder()
          .values({batch1, batch2})
          .groupId({"k1", "k2"}, {{"k1"}, {"k2"}}, {"a"})
          .singleAggregation({"k1", "k2", "group_id"}, {"sum(a) as sum_a"})
          .project({"k1", "k2", "sum_a"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, sum(a) FROM tmp GROUP BY GROUPING SETS ((k1), (k2))");
}

// Test: String type in grouping keys
TEST_F(CudfGroupIdTest, stringKeys) {
  auto input = makeRowVector(
      {"k1", "k2", "a"},
      {
          makeFlatVector<std::string>({"a", "b", "a", "b"}),
          makeFlatVector<int64_t>({1, 1, 2, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 40}),
      });

  createDuckDbTable({input});

  auto plan =
      PlanBuilder()
          .values({input})
          .groupId({"k1", "k2"}, {{"k1"}, {"k2"}}, {"a"})
          .singleAggregation({"k1", "k2", "group_id"}, {"sum(a) as sum_a"})
          .project({"k1", "k2", "sum_a"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, sum(a) FROM tmp GROUP BY GROUPING SETS ((k1), (k2))");
}

// Test: Global grouping set (empty grouping set)
TEST_F(CudfGroupIdTest, globalGroupingSet) {
  auto input = makeRowVector(
      {"k1", "a"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
      });

  createDuckDbTable({input});

  // GROUPING SETS ((k1), ())
  auto plan = PlanBuilder()
                  .values({input})
                  .groupId({"k1"}, {{"k1"}, {}}, {"a"})
                  .singleAggregation({"k1", "group_id"}, {"sum(a) as sum_a"})
                  .project({"k1", "sum_a"})
                  .planNode();

  assertQuery(
      plan, "SELECT k1, sum(a) FROM tmp GROUP BY GROUPING SETS ((k1), ())");
}

// Test: Column reuse across multiple grouping sets (exercises copy vs move
// logic) When a column appears in multiple grouping sets, it must be copied for
// earlier sets and can only be moved on the last use.
TEST_F(CudfGroupIdTest, columnReuseAcrossGroupingSets) {
  auto input = makeRowVector(
      {"k1", "a"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4}),
          makeFlatVector<int64_t>({10, 20, 30, 40}),
      });

  createDuckDbTable({input});

  // k1 appears in both grouping sets, so it must be copied for the first set
  // and moved for the second set. The aggregation input 'a' also appears in
  // both outputs, requiring similar copy/move handling.
  auto plan = PlanBuilder()
                  .values({input})
                  .groupId({"k1"}, {{"k1"}, {"k1"}}, {"a"})
                  .singleAggregation({"k1", "group_id"}, {"sum(a) as sum_a"})
                  .project({"k1", "sum_a"})
                  .planNode();

  // Both grouping sets include k1, so we get the same grouping twice
  // with different group_id values (0 and 1)
  assertQuery(
      plan,
      "SELECT k1, sum(a) FROM tmp GROUP BY k1 "
      "UNION ALL "
      "SELECT k1, sum(a) FROM tmp GROUP BY k1");
}

// Test: Same input column mapped to multiple output columns within a grouping
// set. This exercises the case where a single input column (o_key) is aliased
// to multiple output columns (o_key and o_key_1), and both appear in the same
// grouping set. The column must be copied, not moved, even within a single
// getOutput() call.
TEST_F(CudfGroupIdTest, sameInputMultipleOutputsInGroupingSet) {
  auto input = makeRowVector(
      {"k1", "a"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
      });

  createDuckDbTable({input});

  // k1 is aliased to both k1 and k1_copy in the output.
  // In grouping set 0, both k1 and k1_copy are included, meaning the same
  // input column must be used twice in the same output row.
  auto plan =
      PlanBuilder()
          .values({input})
          .groupId(
              {"k1", "k1 as k1_copy"},
              {{"k1", "k1_copy"}, {"k1"}, {"k1_copy"}},
              {"a"})
          .singleAggregation({"k1", "k1_copy", "group_id"}, {"sum(a) as sum_a"})
          .project({"k1", "k1_copy", "sum_a"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k1_copy, sum(a) FROM ("
      "  SELECT k1, k1 as k1_copy, a FROM tmp"
      ") GROUP BY GROUPING SETS ((k1, k1_copy), (k1), (k1_copy))");

  // Verify the case where the duplicate-input grouping set is the LAST one.
  // This exercises the move path in doGetOutput: the same input column must be
  // copied (not moved) for the first output position and moved for the second.
  plan =
      PlanBuilder()
          .values({input})
          .groupId({"k1", "k1 as k1_copy"}, {{"k1"}, {"k1", "k1_copy"}}, {"a"})
          .singleAggregation({"k1", "k1_copy", "group_id"}, {"sum(a) as sum_a"})
          .project({"k1", "k1_copy", "sum_a"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k1_copy, sum(a) FROM ("
      "  SELECT k1, k1 as k1_copy, a FROM tmp"
      ") GROUP BY GROUPING SETS ((k1), (k1, k1_copy))");
}

// Test: Same input column used as both grouping key and aggregation input.
// This exercises the case where inputColumnUsageCount > 1 because the same
// input column appears in both groupingKeyMappings and aggregationInputs.
// Note: We use an alias (k1 as k1_key) to avoid duplicate column names in the
// output schema, which would cause the aggregation to reference the wrong
// column.
TEST_F(CudfGroupIdTest, columnUsedAsKeyAndAggregationInput) {
  auto input = makeRowVector(
      {"k1", "k2"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      });

  createDuckDbTable({input});

  // k1 is used as both a grouping key (aliased to k1_key) AND an aggregation
  // input (k1). This means the input column k1 has usage count > 1 and must be
  // copied, not moved, for the first use and moved only on the final use.
  auto plan =
      PlanBuilder()
          .values({input})
          .groupId({"k1 as k1_key"}, {{"k1_key"}, {"k1_key"}}, {"k1", "k2"})
          .singleAggregation(
              {"k1_key", "group_id"},
              {"sum(k1) as sum_k1", "sum(k2) as sum_k2"})
          .project({"k1_key", "sum_k1", "sum_k2"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, sum(k1) as sum_k1, sum(k2) as sum_k2 FROM tmp GROUP BY k1 "
      "UNION ALL "
      "SELECT k1, sum(k1) as sum_k1, sum(k2) as sum_k2 FROM tmp GROUP BY k1");

  // Test with multiple grouping sets where k1 is used as key and agg input.
  // For the global grouping set ({}), k1_key is NULL but k1 (agg input) retains
  // original values.
  plan = PlanBuilder()
             .values({input})
             .groupId({"k1 as k1_key"}, {{"k1_key"}, {}}, {"k1", "k2"})
             .singleAggregation(
                 {"k1_key", "group_id"},
                 {"sum(k1) as sum_k1", "sum(k2) as sum_k2"})
             .project({"k1_key", "sum_k1", "sum_k2"})
             .planNode();

  assertQuery(
      plan,
      "SELECT k1_key, sum(k1) as sum_k1, sum(k2) as sum_k2 FROM ("
      "  SELECT k1 as k1_key, k1, k2 FROM tmp"
      ") GROUP BY GROUPING SETS ((k1_key), ())");
}
