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
#include "velox/experimental/cudf/exec/CudfPlanNodeChecker.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

#include "velox/core/QueryCtx.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::exec::test;

namespace {

// Exercises the standalone plan-node checker that decides whether a single plan
// node can run on GPU. Both entry points are covered: the no-context overload
// models the Presto coordinator, which validates plans before it has an
// execution context, and the QueryCtx/MemoryPool overload models the Velox
// operator adapters, which check at runtime. The per-operator selection tests
// (e.g. AggregationSelectionTest, ExpressionEvaluatorSelectionTest) cover the
// eligibility rules in depth; these tests verify the checker dispatches to them
// and honours both overloads.
class CudfPlanNodeCheckerTest : public ::testing::Test,
                                public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    queryCtx_ = core::QueryCtx::create();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    window::prestosql::registerAllWindowFunctions();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);
    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    queryCtx_.reset();
    pool_.reset();
  }

  // Builds a three-column numeric input shared by most checks. c0 and c1 are
  // grouping/join keys, c2 supplies a double aggregate input.
  RowVectorPtr makeInput() {
    return makeRowVector(
        {"c0", "c1", "c2"},
        {
            makeFlatVector<int64_t>({1, 2, 3}),
            makeFlatVector<int64_t>({10, 20, 30}),
            makeFlatVector<double>({1.1, 2.2, 3.3}),
        });
  }

  // Builds a single-column bigint build side for the join checks. Its key
  // column is named to avoid colliding with the probe columns from makeInput().
  RowVectorPtr makeJoinBuild() {
    return makeRowVector({"u0"}, {makeFlatVector<int64_t>({1, 2, 3})});
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
};

TEST_F(CudfPlanNodeCheckerTest, filterNode) {
  auto plan = PlanBuilder().values({makeInput()}).filter("c0 > 1").planNode();
  auto filterNode = std::dynamic_pointer_cast<const core::FilterNode>(plan);
  ASSERT_NE(filterNode, nullptr);
  EXPECT_TRUE(isFilterNodeSupported(filterNode.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, filterNodeUnsupported) {
  // to_big_endian_64 has no cuDF evaluator, so a predicate that references it
  // must fall back to CPU.
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .filter("to_big_endian_64(c0) = to_big_endian_64(c1)")
                  .planNode();
  auto filterNode = std::dynamic_pointer_cast<const core::FilterNode>(plan);
  ASSERT_NE(filterNode, nullptr);
  auto support = isFilterNodeSupported(filterNode.get());
  EXPECT_FALSE(support.supported);
  EXPECT_FALSE(support.reason.empty());
}

TEST_F(CudfPlanNodeCheckerTest, projectNode) {
  auto plan =
      PlanBuilder().values({makeInput()}).project({"c0 + c1"}).planNode();
  auto projectNode = std::dynamic_pointer_cast<const core::ProjectNode>(plan);
  ASSERT_NE(projectNode, nullptr);
  EXPECT_TRUE(isProjectNodeSupported(projectNode.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, projectNodeUnsupported) {
  // A projection over the CPU-only to_big_endian_64 function is rejected.
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .project({"to_big_endian_64(c0)"})
                  .planNode();
  auto projectNode = std::dynamic_pointer_cast<const core::ProjectNode>(plan);
  ASSERT_NE(projectNode, nullptr);
  EXPECT_FALSE(isProjectNodeSupported(projectNode.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, projectNodeTimezoneDependsOnQueryCtx) {
  // date_trunc('hour', ...) on a timestamp truncates to a boundary that depends
  // on the session timezone, so cuDF can evaluate it only while
  // adjust_timestamp_to_session_timezone is off. The checker reads that setting
  // from the QueryCtx, so the verdict for one node flips based purely on the
  // session config -- the eligibility decision the optional context exists to
  // make.
  auto input = makeRowVector(
      {"event_ts"},
      {makeFlatVector<Timestamp>(
          {Timestamp(1767314700, 0), Timestamp(1767318300, 0)}, TIMESTAMP())});
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"date_trunc('hour', event_ts)"})
                  .planNode();
  auto projectNode = std::dynamic_pointer_cast<const core::ProjectNode>(plan);
  ASSERT_NE(projectNode, nullptr);

  // Coordinator path and a default session both leave the timezone adjustment
  // off, so the node is GPU-eligible.
  EXPECT_TRUE(isProjectNodeSupported(projectNode.get()).supported);
  EXPECT_TRUE(
      isProjectNodeSupported(projectNode.get(), queryCtx_.get(), pool_.get())
          .supported);

  // Same node, same overload: enabling adjust_timestamp_to_session_timezone is
  // the only change, and it forces a CPU fallback.
  auto timezoneQueryCtx = core::QueryCtx::create(
      nullptr,
      core::QueryConfig{{
          {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
      }});
  EXPECT_FALSE(isProjectNodeSupported(
                   projectNode.get(), timezoneQueryCtx.get(), pool_.get())
                   .supported);
}

TEST_F(CudfPlanNodeCheckerTest, aggregationNodeSupported) {
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .singleAggregation({"c0"}, {"sum(c1)"})
                  .planNode();
  auto aggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  ASSERT_NE(aggregationNode, nullptr);
  EXPECT_TRUE(isAggregationNodeSupported(aggregationNode.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, aggregationNodeUnsupported) {
  // variance has no cuDF aggregator (unlike stddev, which aliases the
  // supported stddev_samp), so the node must fall back to CPU.
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .singleAggregation({"c0"}, {"variance(c2)"})
                  .planNode();
  auto aggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  ASSERT_NE(aggregationNode, nullptr);
  auto support = isAggregationNodeSupported(aggregationNode.get());
  EXPECT_FALSE(support.supported);
  EXPECT_FALSE(support.reason.empty());
}

TEST_F(CudfPlanNodeCheckerTest, hashJoinNodeSupported) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({makeInput()})
                  .hashJoin(
                      {"c0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .values({makeJoinBuild()})
                          .planNode(),
                      "",
                      {"c0", "c1"},
                      core::JoinType::kInner)
                  .planNode();
  auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(plan);
  ASSERT_NE(joinNode, nullptr);
  EXPECT_TRUE(isHashJoinNodeSupported(joinNode.get()).supported);
}

TEST_F(
    CudfPlanNodeCheckerTest,
    hashJoinNodeNullAwareAntiWithFilterUnsupported) {
  // The anti join type is supported, but a null-aware anti join that also
  // carries a filter is disabled until that combination is implemented.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({makeInput()})
                  .hashJoin(
                      {"c0"},
                      {"u0"},
                      PlanBuilder(planNodeIdGenerator)
                          .values({makeJoinBuild()})
                          .planNode(),
                      "c1 > u0",
                      {"c0", "c1"},
                      core::JoinType::kAnti,
                      /*nullAware=*/true)
                  .planNode();
  auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(plan);
  ASSERT_NE(joinNode, nullptr);
  auto support = isHashJoinNodeSupported(joinNode.get());
  EXPECT_FALSE(support.supported);
  EXPECT_THAT(
      support.reason, testing::HasSubstr("null-aware anti join with filter"));
}

TEST_F(CudfPlanNodeCheckerTest, nestedLoopJoinNodeSupported) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({makeInput()})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({makeJoinBuild()})
                          .planNode(),
                      {"c0", "u0"})
                  .planNode();
  auto joinNode =
      std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(plan);
  ASSERT_NE(joinNode, nullptr);
  EXPECT_TRUE(isNestedLoopJoinNodeSupported(joinNode.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, nestedLoopJoinNodeUnsupportedCondition) {
  // The join condition calls to_big_endian_64, which cuDF cannot evaluate.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values({makeInput()})
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values({makeJoinBuild()})
                          .planNode(),
                      "to_big_endian_64(c0) = to_big_endian_64(u0)",
                      {"c0", "u0"},
                      core::JoinType::kInner)
                  .planNode();
  auto joinNode =
      std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(plan);
  ASSERT_NE(joinNode, nullptr);
  auto support = isNestedLoopJoinNodeSupported(joinNode.get());
  EXPECT_FALSE(support.supported);
  EXPECT_THAT(
      support.reason,
      testing::HasSubstr("join condition cannot be evaluated by cuDF"));
}

TEST_F(CudfPlanNodeCheckerTest, windowNodeSupported) {
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .window({"row_number() over (partition by c0 order by c1)"})
                  .planNode();
  auto windowNode = std::dynamic_pointer_cast<const core::WindowNode>(plan);
  ASSERT_NE(windowNode, nullptr);
  EXPECT_TRUE(isWindowNodeSupported(windowNode.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, windowNodeUnsupported) {
  // cuDF supports only single-column ORDER BY, so a multi-column ordering
  // forces a CPU fallback. The reason is threaded through from
  // CudfWindow::canRunOnGPU.
  auto plan =
      PlanBuilder()
          .values({makeInput()})
          .window({"row_number() over (partition by c0 order by c1, c2)"})
          .planNode();
  auto windowNode = std::dynamic_pointer_cast<const core::WindowNode>(plan);
  ASSERT_NE(windowNode, nullptr);
  auto support = isWindowNodeSupported(windowNode.get());
  EXPECT_FALSE(support.supported);
  EXPECT_FALSE(support.reason.empty());
}

TEST_F(CudfPlanNodeCheckerTest, topNRowNumberNodeSupported) {
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .topNRowNumber({"c0"}, {"c1"}, 5, true)
                  .planNode();
  auto node = std::dynamic_pointer_cast<const core::TopNRowNumberNode>(plan);
  ASSERT_NE(node, nullptr);
  EXPECT_TRUE(isTopNRowNumberNodeSupported(node.get()).supported);
}

TEST_F(CudfPlanNodeCheckerTest, topNRowNumberNodeUnsupported) {
  // row_number is the only ranking function cuDF supports; rank must fall back.
  auto plan = PlanBuilder()
                  .values({makeInput()})
                  .topNRank("rank", {"c0"}, {"c1"}, 5, true)
                  .planNode();
  auto node = std::dynamic_pointer_cast<const core::TopNRowNumberNode>(plan);
  ASSERT_NE(node, nullptr);
  auto support = isTopNRowNumberNodeSupported(node.get());
  EXPECT_FALSE(support.supported);
  EXPECT_THAT(support.reason, testing::HasSubstr("row_number"));
}

TEST_F(CudfPlanNodeCheckerTest, localPartitionNodeSupported) {
  auto plan =
      PlanBuilder().values({makeInput()}).localPartition({"c0"}).planNode();
  auto node = std::dynamic_pointer_cast<const core::LocalPartitionNode>(plan);
  ASSERT_NE(node, nullptr);
  EXPECT_TRUE(isLocalPartitionNodeSupported(node.get()).supported);
}

} // namespace
