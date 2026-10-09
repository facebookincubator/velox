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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConnector.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfPlanTestUtils.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

#include <folly/ScopeGuard.h>

#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::cudf_velox::test {
namespace {

using ::facebook::velox::exec::test::AssertQueryBuilder;
using ::facebook::velox::exec::test::PlanBuilder;

bool containsNode(const core::PlanNodePtr& node, std::string_view name) {
  if (node->name() == name) {
    return true;
  }
  return std::any_of(
      node->sources().begin(), node->sources().end(), [&](const auto& source) {
        return containsNode(source, name);
      });
}

class PlanRewriterTest
    : public ::facebook::velox::exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    ::facebook::velox::window::prestosql::registerAllWindowFunctions();
    auto& config = CudfConfig::getInstance();
    savedDriverAdapter_ = config.enableDriverAdapter;
    savedLocalPartitionAdapter_ = config.enableLocalPartitionAdapter;
    savedFallback_ = config.allowCpuFallback;
    savedConcat_ = config.concatOptimizationEnabled;
    config.enableDriverAdapter = false;
    config.enableLocalPartitionAdapter = true;
    config.allowCpuFallback = true;
    config.concatOptimizationEnabled = false;
    registerCudf();
    input_ = makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>({1, 2, 1, 3}),
         makeFlatVector<int64_t>({10, 20, 30, 40})});
  }

  void TearDown() override {
    unregisterCudf();
    auto& config = CudfConfig::getInstance();
    config.enableDriverAdapter = savedDriverAdapter_;
    config.enableLocalPartitionAdapter = savedLocalPartitionAdapter_;
    config.allowCpuFallback = savedFallback_;
    config.concatOptimizationEnabled = savedConcat_;
    input_.reset();
    OperatorTestBase::TearDown();
  }

  struct Sample {
    core::PlanNodePtr plan;
    std::string physicalNode;
    std::string gpuOperator;
  };

  std::unordered_map<std::string, Sample> samples() {
    auto values = [&] { return PlanBuilder().values({input_}); };
    auto build = makeRowVector(
        {"u0", "u1"},
        {makeFlatVector<int64_t>({1, 2, 3}),
         makeFlatVector<int64_t>({5, 6, 7})});
    auto hashGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto nestedLoopGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto hashJoin =
        PlanBuilder(hashGenerator)
            .values({input_})
            .hashJoin(
                {"c0"},
                {"u0"},
                PlanBuilder(hashGenerator).values({build}).planNode(),
                "",
                {"c0", "c1", "u1"})
            .planNode();
    auto nestedLoopJoin =
        PlanBuilder(nestedLoopGenerator)
            .values({input_})
            .nestedLoopJoin(
                PlanBuilder(nestedLoopGenerator).values({build}).planNode(),
                {"c0", "c1", "u0", "u1"})
            .planNode();
    auto localMerge =
        values().orderBy({"c1"}, true).localMerge({"c1"}).planNode();
    auto localPartition = values()
                              .project({"c0", "c1"})
                              .localPartitionRoundRobin()
                              .singleAggregation({"c0"}, {"sum(c1)"})
                              .planNode();
    auto single = makeRowVector({"c0"}, {makeFlatVector<int64_t>({42})});
    return {
        {"FilterProject",
         {values().filter("c0 > 1").project({"c0", "c1 + 1 AS x"}).planNode(),
          "CudfFilterProject",
          "CudfFilterProject"}},
        {"Aggregation",
         {values().singleAggregation({"c0"}, {"sum(c1)"}).planNode(),
          "CudfAggregation",
          "CudfGroupbySINGLE"}},
        {"HashJoinBuild", {hashJoin, "CudfHashJoin", "CudfHashJoinBuild"}},
        {"HashJoinProbe", {hashJoin, "CudfHashJoin", "CudfHashJoinProbe"}},
        {"NestedLoopJoinBuild",
         {nestedLoopJoin, "CudfNestedLoopJoin", "CudfNestedLoopJoinBuild"}},
        {"NestedLoopJoinProbe",
         {nestedLoopJoin, "CudfNestedLoopJoin", "CudfNestedLoopJoinProbe"}},
        {"OrderBy",
         {values().orderBy({"c1"}, false).planNode(),
          "CudfOrderBy",
          "CudfOrderBy"}},
        {"TopN",
         {values().topN({"c1"}, 2, false).planNode(), "CudfTopN", "CudfTopN"}},
        {"TopNRowNumber",
         {values().topNRowNumber({"c0"}, {"c1"}, 1, true).planNode(),
          "CudfTopNRowNumber",
          "CudfTopNRowNumber"}},
        {"Limit",
         {values().limit(0, 2, false).planNode(), "CudfLimit", "CudfLimit"}},
        {"LocalPartition",
         {localPartition, "LocalPartition", "CudfLocalPartition"}},
        {"LocalMerge", {localMerge, "CudfLocalMerge", "CudfLocalMerge"}},
        {"AssignUniqueId",
         {values().assignUniqueId().planNode(),
          "CudfAssignUniqueId",
          "CudfAssignUniqueId"}},
        {"MarkDistinct",
         {values().markDistinct("m", {"c0"}).planNode(),
          "CudfMarkDistinct",
          "CudfMarkDistinct"}},
        {"EnforceSingleRow",
         {PlanBuilder().values({single}).enforceSingleRow().planNode(),
          "CudfEnforceSingleRow",
          "CudfEnforceSingleRow"}},
        {"GroupId",
         {values().groupId({"c0"}, {{"c0"}, {}}, {"c1"}).planNode(),
          "CudfGroupId",
          "CudfGroupId"}},
        {"Window",
         {values()
              .window({"row_number() over (partition by c0 order by c1) AS rn"})
              .planNode(),
          "CudfWindow",
          "CudfWindow"}},
    };
  }

  struct Execution {
    RowVectorPtr results;
    std::unordered_set<std::string> operators;
  };

  Execution execute(const core::PlanNodePtr& plan, bool useDriverAdapter) {
    unregisterCudf();
    CudfConfig::getInstance().enableDriverAdapter = useDriverAdapter;
    registerCudf();
    std::shared_ptr<::facebook::velox::exec::Task> task;
    auto results =
        AssertQueryBuilder(useDriverAdapter ? plan : rewriteToCudfPlan(plan))
            .maxDrivers(1)
            .copyResults(pool(), task);
    std::unordered_set<std::string> operators;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        operators.insert(op.operatorType);
      }
    }
    return {results, std::move(operators)};
  }

  RowVectorPtr input_;

 private:
  bool savedDriverAdapter_;
  bool savedLocalPartitionAdapter_;
  bool savedFallback_;
  bool savedConcat_;
};

TEST_F(PlanRewriterTest, everyAdapterHasPhysicalPlanCoverage) {
  auto cases = samples();
  // Kept operators have no replacement node: their data boundaries are
  // exercised by scan classification, the merge/partition samples, ordinary
  // Values inputs, and the partitioned-output test below.
  const std::unordered_set<std::string> keptOperators{
      "TableScan",
      "Values",
      "CallbackSink",
      "LocalExchange",
      "PartitionedOutput"};
  registerAllOperatorAdapters();
  for (const auto& adapter :
       OperatorAdapterRegistry::getInstance().getAdapters()) {
    SCOPED_TRACE(adapter->name());
    if (adapter->keepOperator()) {
      EXPECT_TRUE(keptOperators.contains(adapter->name()))
          << "Add explicit plan-boundary coverage for this kept adapter";
      continue;
    }
    auto it = cases.find(adapter->name());
    ASSERT_NE(it, cases.end())
        << "Add a physical-plan coverage sample for this adapter";
    auto rewritten = rewriteToCudfPlan(it->second.plan);
    EXPECT_TRUE(containsNode(rewritten, it->second.physicalNode));
  }
}

TEST_F(PlanRewriterTest, allSupportedLocalPartitionKindsStayOnGpu) {
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(kind);
    auto builder = PlanBuilder().values({input_}).project({"c0", "c1"});
    if (kind == 0) {
      builder.localGather();
    } else if (kind == 1) {
      builder.localPartition({"c0"});
    } else if (kind == 2) {
      builder.localPartitionRoundRobin();
    } else {
      builder.localPartitionRoundRobinRow();
    }
    auto plan = builder.singleAggregation({"c0"}, {"sum(c1)"}).planNode();
    auto physical = rewriteToCudfPlan(plan);
    auto aggregation = physical->sources()[0];
    ASSERT_TRUE(aggregation->is<CudfAggregationNode>());
    ASSERT_TRUE(aggregation->sources()[0]->is<core::LocalPartitionNode>());
    EXPECT_TRUE(
        aggregation->sources()[0]->sources()[0]->is<CudfFilterProjectNode>());
    auto legacy = execute(plan, true);
    auto rewritten = execute(plan, false);
    EXPECT_TRUE(legacy.operators.contains("CudfLocalPartition"));
    EXPECT_TRUE(rewritten.operators.contains("CudfLocalPartition"));
    EXPECT_TRUE(
        ::facebook::velox::exec::test::assertEqualResults(
            {legacy.results}, {rewritten.results}));
  }
}

TEST_F(PlanRewriterTest, physicalExecutionMatchesDriverAdapter) {
  for (const auto& [name, sample] : samples()) {
    SCOPED_TRACE(name);
    auto legacy = execute(sample.plan, true);
    ASSERT_TRUE(legacy.operators.contains(sample.gpuOperator));
    auto physical = execute(sample.plan, false);
    ASSERT_TRUE(physical.operators.contains(sample.gpuOperator));
    EXPECT_TRUE(
        ::facebook::velox::exec::test::assertEqualResults(
            {legacy.results}, {physical.results}));
  }
}

TEST_F(PlanRewriterTest, aggregationVariants) {
  for (const auto& sample : std::vector<Sample>{
           {PlanBuilder()
                .values({input_})
                .singleAggregation({}, {"sum(c1)"})
                .planNode(),
            "CudfAggregation",
            "CudfReduceSINGLE"},
           {PlanBuilder()
                .values({input_})
                .singleAggregation({"c0"}, {})
                .planNode(),
            "CudfAggregation",
            "CudfDistinctSINGLE"}}) {
    SCOPED_TRACE(sample.gpuOperator);
    auto legacy = execute(sample.plan, true);
    auto physical = execute(sample.plan, false);
    ASSERT_TRUE(legacy.operators.contains(sample.gpuOperator));
    ASSERT_TRUE(physical.operators.contains(sample.gpuOperator));
    EXPECT_TRUE(
        ::facebook::velox::exec::test::assertEqualResults(
            {legacy.results}, {physical.results}));
  }
}

TEST_F(PlanRewriterTest, partitionedOutputPreservesRewrittenChildren) {
  auto plan = PlanBuilder()
                  .values({input_})
                  .project({"c0", "c1 + 1 AS x"})
                  .partitionedOutput({}, 1)
                  .planNode();
  auto rewritten = rewriteToCudfPlan(plan);
  ASSERT_TRUE(rewritten->is<core::PartitionedOutputNode>());
  ASSERT_TRUE(rewritten->sources()[0]->is<CudfToVeloxNode>());
  EXPECT_TRUE(containsNode(rewritten, "CudfFilterProject"));
}

TEST_F(PlanRewriterTest, cpuFallbackPreservesGpuRegions) {
  auto rowNumber = PlanBuilder()
                       .values({input_})
                       .project({"c0", "c1 + 1 AS c1"})
                       .rowNumber({"c0"})
                       .project({"c0", "c1", "row_number"})
                       .planNode();
  auto arrays = makeRowVector(
      {"c0", "a"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeArrayVector<int64_t>({{10, 20}, {}, {30}})});
  auto unnest = PlanBuilder()
                    .values({arrays})
                    .project({"c0", "a"})
                    .unnest({"c0"}, {"a"})
                    .project({"c0", "a_e + 1 AS x"})
                    .planNode();
  auto expand = PlanBuilder()
                    .values({input_})
                    .project({"c0", "c1 + 1 AS c1"})
                    .expand({{"c0", "c1", "0 AS gid"}, {"c0", "NULL", "1"}})
                    .project({"c0", "c1", "gid"})
                    .planNode();
  for (const auto& plan : {rowNumber, unnest, expand}) {
    auto physical = rewriteToCudfPlan(plan);
    ASSERT_TRUE(physical->is<CudfToVeloxNode>());
    auto project = physical->sources()[0];
    ASSERT_TRUE(project->is<CudfFilterProjectNode>());
    auto upload = project->sources()[0];
    ASSERT_TRUE(upload->is<CudfFromVeloxNode>());
    auto cpuNode = upload->sources()[0];
    ASSERT_TRUE(
        cpuNode->is<core::RowNumberNode>() || cpuNode->is<core::UnnestNode>() ||
        cpuNode->is<core::ExpandNode>());
    ASSERT_TRUE(cpuNode->sources()[0]->is<CudfToVeloxNode>());
    ASSERT_TRUE(
        cpuNode->sources()[0]->sources()[0]->is<CudfFilterProjectNode>());
    auto legacy = execute(plan, true);
    auto rewritten = execute(plan, false);
    EXPECT_TRUE(
        ::facebook::velox::exec::test::assertEqualResults(
            {legacy.results}, {rewritten.results}));
  }
}

TEST_F(PlanRewriterTest, bothGpuConnectorsNeedOutputConversion) {
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  const std::string hiveId = "plan-rewriter-hive";
  const std::string icebergId = "plan-rewriter-iceberg";
  auto& registry = ::facebook::velox::connector::ConnectorRegistry::global();
  registry.insert(
      hiveId,
      std::make_shared<connector::hive::CudfHiveConnector>(
          hiveId, properties, executor_.get()));
  registry.insert(
      icebergId,
      std::make_shared<connector::hive::iceberg::CudfIcebergConnector>(
          icebergId, properties, executor_.get()));
  SCOPE_EXIT {
    registry.erase(hiveId);
    registry.erase(icebergId);
  };
  for (const auto& id : {hiveId, icebergId}) {
    SCOPED_TRACE(id);
    auto plan = PlanBuilder()
                    .startTableScan()
                    .connectorId(id)
                    .outputType(input_->rowType())
                    .endTableScan()
                    .planNode();
    auto rewritten = rewriteToCudfPlan(plan);
    ASSERT_TRUE(rewritten->is<CudfToVeloxNode>());
    EXPECT_EQ(rewritten->sources()[0], plan);
  }
}

TEST_F(PlanRewriterTest, localMergeSplitsEverySourceWithoutAdapter) {
  CudfConfig::getInstance().enableLocalPartitionAdapter = false;
  unregisterCudf();
  registerCudf();
  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  auto left = makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 3})});
  auto right = makeRowVector({"c0"}, {makeFlatVector<int64_t>({2, 4})});
  auto leftSource = std::make_shared<CudfFromVeloxNode>(
      "upload-left", PlanBuilder(generator).values({left}).planNode());
  auto rightSource = std::make_shared<CudfFromVeloxNode>(
      "upload-right", PlanBuilder(generator).values({right}).planNode());
  auto merge = std::make_shared<CudfLocalMergeNode>(
      "merge",
      std::vector<core::PlanNodePtr>{leftSource, rightSource},
      left->rowType(),
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      std::vector<core::SortOrder>{core::SortOrder(true, false)});
  auto plan = std::make_shared<CudfToVeloxNode>("download", merge);
  auto task = AssertQueryBuilder(plan).maxDrivers(4).assertResults(
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3, 4})}));
  EXPECT_EQ(task->taskStats().pipelineStats.size(), 3);
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      EXPECT_NE(op.operatorType, "LocalMerge");
      if (op.operatorType == "CudfLocalMerge") {
        EXPECT_EQ(op.numDrivers, 1);
      }
    }
  }
}

TEST_F(PlanRewriterTest, constructTopNRowNumberWithoutCpuCounterpart) {
  auto source = std::make_shared<CudfFromVeloxNode>(
      "upload", PlanBuilder().values({input_}).planNode());
  auto topN = std::make_shared<CudfTopNRowNumberNode>(
      "top-n",
      source,
      input_->rowType(),
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c1")},
      std::vector<core::SortOrder>{core::SortOrder(true, false)},
      1,
      false);
  AssertQueryBuilder(std::make_shared<CudfToVeloxNode>("download", topN))
      .assertResults(makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeFlatVector<int64_t>({10, 20, 40})}));
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
