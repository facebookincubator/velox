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
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/tests/utils/CudfPlanTestUtils.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/core/FixedPointPlanNodes.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"

#include <folly/ScopeGuard.h>

#include <set>
#include <typeinfo>
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
    savedFallback_ = config.allowCpuFallback;
    savedConcat_ = config.concatOptimizationEnabled;
    config.enableDriverAdapter = false;
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
         {localPartition, "CudfLocalPartition", "CudfLocalPartition"}},
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

  /// Plans where the cuDF driver adapter keeps at least one node on the CPU,
  /// one for each placement rule in GpuPlacement.h, plus unsupported types.
  struct FallbackSample {
    core::PlanNodePtr plan;
    /// A CPU operator that both paths must keep.
    std::string cpuOperator;
  };

  std::unordered_map<std::string, FallbackSample> fallbackSamples() {
    auto values = [&] { return PlanBuilder().values({input_}); };
    // date_add with a variable unit has no cuDF implementation.
    auto dates = makeRowVector(
        {"c0", "unit", "d"},
        {makeFlatVector<int64_t>({1, 2, 3}),
         makeFlatVector<std::string>({"day", "week", "month"}),
         makeFlatVector<int32_t>({18'000, 18'100, 18'200}, DATE())});
    const std::string unsupported = "date_add(unit, c0, d)";
    auto build = makeRowVector(
        {"u0", "u1"},
        {makeFlatVector<int64_t>({1, 2, 3}),
         makeFlatVector<int64_t>({5, 25, 7})});
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();

    const auto customType =
        facebook::velox::test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON();
    auto customProbe = makeRowVector(
        {"p_key"}, {makeFlatVector<int64_t>({1, 257, 2}, customType)});
    auto customBuild =
        makeRowVector({"b_key"}, {makeFlatVector<int64_t>({1}, customType)});
    auto zeroColumns = std::make_shared<RowVector>(
        pool(), ROW({}, {}), nullptr, 4, std::vector<VectorPtr>{});
    auto maps = makeRowVector(
        {"c0", "m"},
        {makeFlatVector<int64_t>({3, 1, 2}),
         makeMapVector<int64_t, int64_t>({{{1, 10}}, {{2, 20}}, {{3, 30}}})});

    return {
        {"NestedLoopJoinCustomComparison",
         {PlanBuilder(ids)
              .values({customProbe})
              .nestedLoopJoin(
                  PlanBuilder(ids).values({customBuild}).planNode(),
                  "p_key = b_key",
                  {"p_key"},
                  core::JoinType::kInner)
              .planNode(),
          "NestedLoopJoinProbe"}},
        {"ZeroColumnFilterProject",
         {PlanBuilder()
              .values({zeroColumns})
              .filter("1 < 2")
              .project({})
              .singleAggregation({}, {"count(1) AS cnt"})
              .planNode(),
          "FilterProject"}},
        {"UnsupportedFilterUnderProject",
         {PlanBuilder()
              .values({dates})
              .filter(unsupported + " > d")
              .project({"c0 + 1 AS x"})
              .planNode(),
          "FilterProject"}},
        {"UnsupportedProjectOverFilter",
         {PlanBuilder()
              .values({dates})
              .filter("c0 > 1")
              .project({unsupported + " AS e"})
              .planNode(),
          "FilterProject"}},
        {"UnsupportedAggregate",
         {values().singleAggregation({"c0"}, {"variance(c1)"}).planNode(),
          "Aggregation"}},
        {"NullAwareAntiJoinWithFilter",
         {values()
              .hashJoin(
                  {"c0"},
                  {"u0"},
                  PlanBuilder(ids).values({build}).planNode(),
                  "c1 > u1",
                  {"c0", "c1"},
                  core::JoinType::kAnti,
                  true)
              .planNode(),
          "HashProbe"}},
        {"UnsupportedJoinFilter",
         {PlanBuilder(ids)
              .values({dates})
              .hashJoin(
                  {"c0"},
                  {"u0"},
                  PlanBuilder(ids).values({build}).planNode(),
                  unsupported + " > d",
                  {"c0", "u1"})
              .planNode(),
          "HashProbe"}},
        {"TopNRank",
         {values().topNRank("rank", {"c0"}, {"c1"}, 1, true).planNode(),
          "TopNRowNumber"}},
        {"UnsupportedLocalPartition",
         {values()
              .localPartition(4, {0}, {})
              .singleAggregation({"c0"}, {"sum(c1)"})
              .planNode(),
          "LocalPartition"}},
        {"UnsupportedWindowFunction",
         {values()
              .window({"lag(c1, 1 IGNORE NULLS) over "
                       "(partition by c0 order by c1) AS l"})
              .planNode(),
          "Window"}},
        {"UnsupportedType",
         {PlanBuilder().values({maps}).orderBy({"c0"}, false).planNode(),
          "OrderBy"}},
    };
  }

  struct Execution {
    RowVectorPtr results;
    std::unordered_set<std::string> operators;
    /// Plan node ID and type of each operator other than the vector
    /// conversions, whose IDs differ between the two paths.
    std::set<std::pair<std::string, std::string>> placement;
    int32_t conversions{0};
  };

  Execution execute(const core::PlanNodePtr& plan, bool useDriverAdapter) {
    unregisterCudf();
    CudfConfig::getInstance().enableDriverAdapter = useDriverAdapter;
    registerCudf();
    std::shared_ptr<::facebook::velox::exec::Task> task;
    auto queryCtx = core::QueryCtx::create(executor_.get());
    auto results =
        AssertQueryBuilder(
            useDriverAdapter ? plan : rewriteToCudfPlan(plan, queryCtx))
            .queryCtx(queryCtx)
            .maxDrivers(1)
            .copyResults(pool(), task);
    Execution execution{results};
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        execution.operators.insert(op.operatorType);
        if (op.operatorType == "CudfFromVelox" ||
            op.operatorType == "CudfToVelox") {
          ++execution.conversions;
        } else {
          execution.placement.emplace(op.planNodeId, op.operatorType);
        }
      }
    }
    return execution;
  }

  RowVectorPtr input_;

 private:
  bool savedDriverAdapter_;
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
      "PartitionedOutput",
      "UcxExchange",
      "UcxPartitionedOutput"};
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

TEST_F(PlanRewriterTest, cpuFallbackMatchesDriverAdapter) {
  for (const auto& [name, sample] : fallbackSamples()) {
    SCOPED_TRACE(name);
    auto legacy = execute(sample.plan, true);
    auto physical = execute(sample.plan, false);
    EXPECT_TRUE(legacy.operators.contains(sample.cpuOperator));
    EXPECT_EQ(legacy.placement, physical.placement);
    EXPECT_EQ(legacy.conversions, physical.conversions);
    EXPECT_TRUE(
        ::facebook::velox::exec::test::assertEqualResults(
            {legacy.results}, {physical.results}));
  }
}

TEST_F(PlanRewriterTest, cpuNodesAboveGpuRegionsAreRebuilt) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto gpuRegion = [&] {
    return PlanBuilder(ids).values({input_}).project({"c0", "c1 + 1 AS c1"});
  };
  auto build = makeRowVector(
      {"u0", "u1"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({5, 50})});
  // These run on the CPU, so the driver adapter path is the reference.
  const std::vector<core::PlanNodePtr> executable{
      gpuRegion()
          .parallelProject({{"c0 * 2 AS a"}, {"c1 * 2 AS b"}})
          .planNode(),
      gpuRegion().enforceDistinct({"c0", "c1"}, "Duplicate row").planNode(),
      gpuRegion()
          .markSorted("sorted", {"c1"}, {core::kAscNullsLast})
          .planNode(),
      core::MixedUnionNode::Builder()
          .id("union")
          .sources({gpuRegion().planNode(), gpuRegion().planNode()})
          .build(),
      gpuRegion()
          .hashJoin(
              {"c0"},
              {"u0"},
              PlanBuilder(ids)
                  .values({build})
                  .project({"u0", "u1 + 1 AS u1"})
                  .planNode(),
              "c1 > u1",
              {"c0", "c1"},
              core::JoinType::kAnti,
              true)
          .planNode(),
  };
  // These need an RPC service or fixed-point state to run.
  const std::vector<core::PlanNodePtr> planOnly{
      std::make_shared<core::RPCNode>(
          "rpc",
          gpuRegion().planNode(),
          std::make_shared<core::CallTypedExpr>(
              BIGINT(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
              "test_rpc"),
          "r",
          ROW({"c0", "c1", "r"}, {BIGINT(), BIGINT(), BIGINT()})),
      std::make_shared<core::StateHashJoinNode>(
          "state_join",
          "state",
          std::vector<std::string>{"c0"},
          input_->rowType(),
          gpuRegion().planNode()),
  };

  auto checkRewrite = [&](const core::PlanNodePtr& plan) {
    SCOPED_TRACE(plan->name());
    auto physical = rewriteToCudfPlan(plan);
    EXPECT_EQ(physical->id(), plan->id());
    EXPECT_TRUE(typeid(*physical) == typeid(*plan));
    for (const auto& source : physical->sources()) {
      EXPECT_TRUE(source->is<CudfToVeloxNode>());
    }
  };
  for (const auto& plan : executable) {
    checkRewrite(plan);
    auto legacy = execute(plan, true);
    auto physical = execute(plan, false);
    EXPECT_EQ(legacy.placement, physical.placement);
    EXPECT_TRUE(
        ::facebook::velox::exec::test::assertEqualResults(
            {legacy.results}, {physical.results}));
  }
  for (const auto& plan : planOnly) {
    checkRewrite(plan);
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

class PlanRewriterHiveTest
    : public cudf_velox::exec::test::CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    savedDriverAdapter_ = CudfConfig::getInstance().enableDriverAdapter;
    CudfConfig::getInstance().enableDriverAdapter = false;
    CudfHiveConnectorTestBase::SetUp();
  }

  void TearDown() override {
    CudfHiveConnectorTestBase::TearDown();
    CudfConfig::getInstance().enableDriverAdapter = savedDriverAdapter_;
  }

 private:
  bool savedDriverAdapter_;
};

TEST_F(PlanRewriterHiveTest, gpuScanFeedsGpuNodesDirectly) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto vectors = makeVectors(rowType, 4, 1'000);
  auto file = common::testutil::TempFilePath::create();
  writeToFile(file->getPath(), vectors);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(cudf_velox::exec::test::kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(
                      ::facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .filter("c0 > 0")
                  .singleAggregation({}, {"count(c0)", "min(c1)", "max(c1)"})
                  .planNode();
  auto physical = rewriteToCudfPlan(plan);
  ASSERT_TRUE(physical->is<CudfToVeloxNode>());
  auto aggregation = physical->sources()[0];
  ASSERT_TRUE(aggregation->is<CudfAggregationNode>());
  auto filter = aggregation->sources()[0];
  ASSERT_TRUE(filter->is<CudfFilterProjectNode>());
  EXPECT_TRUE(filter->sources()[0]->is<core::TableScanNode>());

  auto task = ::facebook::velox::exec::test::AssertQueryBuilder(
                  physical, duckDbQueryRunner_)
                  .splits(makeCudfHiveConnectorSplits({file}))
                  .assertResults(
                      "SELECT count(c0), min(c1), max(c1) FROM tmp "
                      "WHERE c0 > 0");
  std::unordered_set<std::string> operators;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      operators.insert(op.operatorType);
    }
  }
  EXPECT_FALSE(operators.contains("CudfFromVelox"));
  EXPECT_TRUE(operators.contains("CudfFilterProject"));
  EXPECT_TRUE(operators.contains("CudfReduceSINGLE"));
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
