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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class CudfExpandTest : public HiveConnectorTestBase {
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

  static bool hasOperator(
      const std::shared_ptr<Task>& task,
      std::string_view operatorType) {
    for (const auto& pipelineStats : task->taskStats().pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == operatorType) {
          return true;
        }
      }
    }
    return false;
  }

  static void assertExpandStats(
      const std::shared_ptr<Task>& task,
      uint64_t inputRows,
      uint64_t inputBatches,
      uint64_t numProjections) {
    size_t numExpandOperators = 0;
    for (const auto& pipelineStats : task->taskStats().pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == "CudfExpand") {
          ++numExpandOperators;
          EXPECT_EQ(operatorStats.inputPositions, inputRows);
          EXPECT_EQ(operatorStats.inputVectors, inputBatches);
          EXPECT_EQ(operatorStats.outputPositions, inputRows * numProjections);
          EXPECT_EQ(operatorStats.outputVectors, inputBatches * numProjections);
        }
      }
    }
    EXPECT_EQ(numExpandOperators, 1);
  }

  RowVectorPtr makeRowVectorData(vector_size_t size) {
    return makeRowVector(
        {"k1", "k2", "a", "b"},
        {
            makeFlatVector<int64_t>(size, [](auto row) { return row % 11; }),
            makeFlatVector<int64_t>(size, [](auto row) { return row % 17; }),
            makeFlatVector<int64_t>(size, [](auto row) { return row; }),
            makeFlatVector<std::string>(
                size, [](auto row) { return std::string(row % 15, 'x'); }),
        });
  }
};

TEST_F(CudfExpandTest, simpleConstant) {
  auto data = makeRowVectorData(3);
  auto children = data->children();
  children.push_back(makeFlatVector<int64_t>({100, 100, 100}));
  children.push_back(makeNullConstant(TypeKind::INTEGER, 3));
  auto expected = makeRowVector(children);

  auto plan =
      PlanBuilder()
          .values({data})
          .expand({{"k1", "k2", "a", "b", "100 as c", "null::integer as d"}})
          .planNode();

  assertQuery(plan, expected);
}

// Complex-type constants cannot be built as cuDF scalars, so Expand must stay
// on the CPU.
TEST_F(CudfExpandTest, complexConstantFallsBackToCpu) {
  // The fallback setting is read when cuDF is registered.
  cudf_velox::unregisterCudf();
  cudf_velox::CudfConfig::getInstance().allowCpuFallback = true;
  cudf_velox::registerCudf();

  auto data = makeRowVectorData(3);
  auto children = data->children();
  children.push_back(
      makeArrayVector<int64_t>({{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}));
  children.push_back(makeAllNullArrayVector(3, BIGINT()));
  auto expected = makeRowVector(children);

  auto plan = PlanBuilder(pool())
                  .values({data})
                  .expand(
                      {{"k1",
                        "k2",
                        "a",
                        "b",
                        "ARRAY[1, 2, 3] as c",
                        "null::bigint[] as d"}})
                  .planNode();

  auto task = assertQuery(plan, expected);
  EXPECT_TRUE(hasOperator(task, OperatorType::kExpand));
  EXPECT_FALSE(hasOperator(task, "CudfExpand"));
}

TEST_F(CudfExpandTest, groupingSets) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .expand(
              {{"k1",
                "null::bigint as k2",
                "a",
                "b",
                "0 as group_id_0",
                "0 as group_id_1"},
               {"k1", "null", "a", "b", "0", "1"},
               {"null", "k2", "a", "b", "1", "2"}})
          .singleAggregation(
              {"k1", "k2", "group_id_0", "group_id_1"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY GROUPING SETS ((k1), (k1), (k2))");
}

TEST_F(CudfExpandTest, cube) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  // Cube.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand({
              {"k1", "k2", "a", "b", "0 as gid"},
              {"k1", "null", "a", "b", "1"},
              {"null", "k2", "a", "b", "2"},
              {"null", "null", "a", "b", "3"},
          })
          .singleAggregation(
              {"k1", "k2", "gid"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"k1", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY CUBE (k1, k2)");
}

TEST_F(CudfExpandTest, rollup) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  // Rollup.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand(
              {{"k1 as foo", "k2", "a", "b", "0 as gid"},
               {"k1", "null", "a", "b", "1"},
               {"null", "null", "a", "b", "2"}})
          .singleAggregation(
              {"foo", "k2", "gid"},
              {"count(1) as count_1", "sum(a) as sum_a", "max(b) as max_b"})
          .project({"foo", "k2", "count_1", "sum_a", "max_b"})
          .planNode();

  assertQuery(
      plan,
      "SELECT k1, k2, count(1), sum(a), max(b) FROM tmp GROUP BY ROLLUP (k1, k2)");
}

TEST_F(CudfExpandTest, countDistinct) {
  auto data = makeRowVectorData(1'000);

  createDuckDbTable({data});

  // count distinct.
  auto plan =
      PlanBuilder()
          .values({data})
          .expand({{"a", "null::varchar as b", "1 as gid"}, {"null", "b", "2"}})
          .singleAggregation({"a", "b", "gid"}, {})
          .singleAggregation({}, {"count(a) as count_a", "count(b) as count_b"})
          .planNode();

  assertQuery(plan, "SELECT count(distinct a), count(distinct b) FROM tmp");
}

TEST_F(CudfExpandTest, duplicateColumnProjection) {
  // Test case where the same input column is projected to multiple output
  // columns
  auto data = makeRowVectorData(100);

  createDuckDbTable({data});

  // Project k1 to both output columns c1 and c2
  auto plan = PlanBuilder()
                  .values({data})
                  .expand({{"k1 as c1", "k1 as c2", "a", "b", "0 as gid"}})
                  .planNode();

  assertQuery(plan, "SELECT k1 as c1, k1 as c2, a, b, 0 as gid FROM tmp");
}

// Exercises the per-batch projection cycle, including moving columns out of
// each batch on its last projection while a column is also projected twice.
TEST_F(CudfExpandTest, multipleBatches) {
  std::vector<RowVectorPtr> data{
      makeRowVectorData(100), makeRowVectorData(200), makeRowVectorData(300)};

  createDuckDbTable(data);

  auto plan = PlanBuilder()
                  .values(data)
                  .expand(
                      {{"k1 as c1", "k1 as c2", "a", "0 as gid"},
                       {"k2", "null", "a", "1"},
                       {"null", "k1", "a", "2"}})
                  .planNode();

  // Values batches are otherwise coalesced into one GPU batch by
  // CudfFromVelox's default 100,000-row target.
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "1")
                  .assertResults(
                      "SELECT k1, k1, a, 0 FROM tmp "
                      "UNION ALL SELECT k2, null, a, 1 FROM tmp "
                      "UNION ALL SELECT null, k1, a, 2 FROM tmp");
  assertExpandStats(task, 600, 3, 3);
}

TEST_F(CudfExpandTest, nullableDuplicateColumnsOnLastProjection) {
  auto data = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt}),
       makeNullableFlatVector<std::string>(
           {"alpha",
            std::nullopt,
            "",
            "a string longer than inline storage"})});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .expand(
                      {{"a as a1",
                        "null::bigint as a2",
                        "b as b1",
                        "null::varchar as b2",
                        "0 as gid"},
                       {"a", "a", "b", "b", "1"}})
                  .planNode();

  // The final projection must copy the first use and move the second use of
  // each nullable column, preserving both the null mask and string storage.
  auto task = assertQuery(
      plan,
      "SELECT a, null::bigint, b, null::varchar, 0 FROM tmp "
      "UNION ALL SELECT a, a, b, b, 1 FROM tmp");
  assertExpandStats(task, 4, 1, 2);
}

TEST_F(CudfExpandTest, supportedConstantTypes) {
  auto constants = makeRowVector(
      {"boolean",
       "tinyint",
       "smallint",
       "integer",
       "bigint",
       "real",
       "double",
       "varchar",
       "date",
       "timestamp",
       "decimal64",
       "decimal128"},
      {makeFlatVector<bool>({true}),
       makeFlatVector<int8_t>({-7}),
       makeFlatVector<int16_t>({-1'234}),
       makeFlatVector<int32_t>({-123'456}),
       makeFlatVector<int64_t>({-1'234'567'890'123LL}),
       makeFlatVector<float>({1.25f}),
       makeFlatVector<double>({-1'234.5}),
       makeFlatVector<std::string>(
           {"a constant longer than inline StringView storage"}),
       makeFlatVector<int32_t>({DATE()->toDays("2026-10-08")}, DATE()),
       makeFlatVector<Timestamp>({Timestamp(1'234'567'890, 123'456'789)}),
       makeFlatVector<int64_t>({12'345}, DECIMAL(10, 2)),
       makeFlatVector<int128_t>(
           {static_cast<int128_t>(123'456'789'012'345'678LL) * 1'000 + 901},
           DECIMAL(21, 3))});
  auto input = makeRowVector({"k"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto source = PlanBuilder().values({input}).planNode();
  std::vector<std::vector<core::TypedExprPtr>> projections(2);
  for (auto& projection : projections) {
    projection.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "k"));
  }
  auto names = std::vector<std::string>{"k"};
  for (size_t i = 0; i < constants->childrenSize(); ++i) {
    const auto& value = constants->childAt(i);
    names.push_back(constants->type()->asRow().nameOf(i));
    projections[0].push_back(std::make_shared<core::ConstantTypedExpr>(value));
    projections[1].push_back(
        std::make_shared<core::ConstantTypedExpr>(
            value->type(), Variant::null(value->type()->kind())));
  }
  auto plan = std::make_shared<core::ExpandNode>(
      "expand", std::move(projections), std::move(names), source);

  // Compare against CPU Expand to preserve exact decimal scales and
  // nanosecond timestamp values without DuckDB result conversions.
  cudf_velox::unregisterCudf();
  auto expected = AssertQueryBuilder(plan).copyResults(pool());
  cudf_velox::registerCudf();
  auto task = AssertQueryBuilder(plan).assertResults(expected);
  assertExpandStats(task, 3, 1, 2);
}
