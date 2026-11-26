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
#include "velox/experimental/cudf/exec/CudfPlanNodeDefinitions.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfPlanBuilder.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/core/FixedPointPlanNodes.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
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
    queryCtx_ = core::QueryCtx::create(executor_.get());
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
    queryCtx_.reset();
    OperatorTestBase::TearDown();
  }

  struct Sample {
    core::PlanNodePtr plan;
    std::string physicalNode;
    std::string gpuOperator;
    // The driver count the task asks for. Above 1 only for plans that return
    // the same rows with any number of drivers.
    int32_t maxDrivers{1};
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
    // Every driver of the source pipeline sorts one copy of the input, so the
    // merge has as many sorted streams as there are drivers.
    auto localMerge = PlanBuilder()
                          .values({input_}, /*parallelizable=*/true)
                          .orderBy({"c1"}, true)
                          .localMerge({"c1"})
                          .planNode();
    // Several batches, so that round robin sends rows to more than one
    // consumer. The partial and final steps give the same sums however the
    // rows are spread.
    auto localPartition = PlanBuilder()
                              .values({input_, input_, input_})
                              .project({"c0", "c1"})
                              .localPartitionRoundRobin()
                              .partialAggregation({"c0"}, {"sum(c1)"})
                              .localGather()
                              .finalAggregation()
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
        {"TopNRank",
         {values().topNRank("rank", {"c0"}, {"c1"}, 1, true).planNode(),
          "CudfTopNRowNumber",
          "CudfTopNRowNumber"}},
        {"Limit",
         {values().limit(0, 2, false).planNode(), "CudfLimit", "CudfLimit"}},
        {"LocalPartition",
         {localPartition, "CudfLocalPartition", "CudfLocalPartition", 4}},
        {"LocalMerge", {localMerge, "CudfLocalMerge", "CudfLocalMerge", 4}},
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

  // Plans where the cuDF driver adapter keeps at least one node on the CPU,
  // one for each node definition that has a placement rule, plus unsupported
  // types.
  struct FallbackSample {
    core::PlanNodePtr plan;
    // A CPU operator that both paths must keep.
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
    auto arrays = makeRowVector(
        {"c0", "a"},
        {makeFlatVector<int64_t>({1, 1, 1, 1}),
         makeArrayVector<int64_t>({{2}, {1}, {2}, {3}})});

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
        {"TopNRankByArray",
         {PlanBuilder()
              .values({arrays})
              .topNRank("rank", {"c0"}, {"a"}, 2, true)
              .planNode(),
          "TopNRowNumber"}},
        {"UnsupportedLocalPartition",
         {values()
              .localPartition(4, {0}, {})
              .singleAggregation({"c0"}, {"sum(c1)"})
              .planNode(),
          "LocalPartition"}},
        {"ScaleWriterLocalPartition",
         {values()
              .project({"c0", "c1"})
              .scaleWriterlocalPartitionRoundRobin()
              .singleAggregation({"c0"}, {"sum(c1)"})
              .planNode(),
          "LocalPartition"}},
        {"MaskedMarkDistinct",
         {values()
              .project({"c0", "c1 > 15 AS mask"})
              .markDistinct({"m0", "m1"}, {"c0"}, {"mask"})
              .planNode(),
          "MarkDistinct"}},
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
    // Plan node ID and type of each operator other than the vector
    // conversions, whose IDs differ between the two paths.
    std::set<std::pair<std::string, std::string>> placement;
    int32_t conversions{0};
    // The largest driver count of any operator.
    int32_t numDrivers{0};
  };

  Execution execute(
      const core::PlanNodePtr& plan,
      bool useDriverAdapter,
      int32_t maxDrivers = 1) {
    unregisterCudf();
    CudfConfig::getInstance().enableDriverAdapter = useDriverAdapter;
    registerCudf();
    std::shared_ptr<::facebook::velox::exec::Task> task;
    auto queryCtx = core::QueryCtx::create(executor_.get());
    auto results =
        AssertQueryBuilder(
            useDriverAdapter ? plan : rewriteToCudfPlan(plan, queryCtx))
            .queryCtx(queryCtx)
            .maxDrivers(maxDrivers)
            .copyResults(pool(), task);
    Execution execution{results};
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        execution.operators.insert(op.operatorType);
        execution.numDrivers = std::max(execution.numDrivers, op.numDrivers);
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

  // Runs 'plan' with cuDF unregistered and returns its rows.
  RowVectorPtr executeOnCpu(const core::PlanNodePtr& plan, int32_t maxDrivers) {
    unregisterCudf();
    std::shared_ptr<::facebook::velox::exec::Task> task;
    auto results = AssertQueryBuilder(plan)
                       .queryCtx(core::QueryCtx::create(executor_.get()))
                       .maxDrivers(maxDrivers)
                       .copyResults(pool(), task);
    registerCudf();
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        EXPECT_FALSE(op.operatorType.starts_with("Cudf")) << op.operatorType;
      }
    }
    return results;
  }

  // Checks that the driver adapter run and the rewriter run of 'plan' both
  // returned the rows that the plan returns without cuDF. The two cuDF paths
  // build their operators the same way, so comparing them with each other
  // would miss a mistake they share. 'ordered' also compares the row order.
  void expectCpuResults(
      const core::PlanNodePtr& plan,
      const Execution& legacy,
      const Execution& physical,
      int32_t maxDrivers = 1,
      bool ordered = false) {
    const auto reference = executeOnCpu(plan, maxDrivers);
    for (const auto& [path, results] :
         {std::pair{"driver adapter", legacy.results},
          std::pair{"rewriter", physical.results}}) {
      SCOPED_TRACE(path);
      if (ordered) {
        ::facebook::velox::test::assertEqualVectors(reference, results);
      } else {
        EXPECT_TRUE(
            ::facebook::velox::exec::test::assertEqualResults(
                {reference}, {results}));
      }
    }
  }

  RowVectorPtr input_;
  // The session that plans are rewritten for when a test does not run them.
  std::shared_ptr<core::QueryCtx> queryCtx_;

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
    auto rewritten = rewriteToCudfPlan(it->second.plan, queryCtx_);
    EXPECT_TRUE(containsNode(rewritten, it->second.physicalNode));
  }
}

TEST_F(PlanRewriterTest, loweringBuildsOverTheGivenSources) {
  std::vector<core::PlanNodePtr> nodes;
  for (const auto& [name, sample] : samples()) {
    nodes.push_back(sample.plan);
  }
  // The samples have neither of these two as their root.
  nodes.push_back(PlanBuilder().values({input_}).filter("c0 > 1").planNode());
  nodes.push_back(
      PlanBuilder().values({input_}).localPartitionRoundRobin().planNode());

  for (const auto& node : nodes) {
    SCOPED_TRACE(node->toString());
    const auto* lowering = detail::findCudfPlanNodeLowering(node);
    ASSERT_NE(lowering, nullptr);

    // Stand-ins for the rewritten sources.
    const auto originalSources = node->sources();
    std::vector<core::PlanNodePtr> sources;
    for (const auto& source : originalSources) {
      sources.push_back(
          std::make_shared<CudfFromVeloxNode>(source->id() + "_gpu", source));
    }

    const auto lowered = lowering->lower(node, sources);
    EXPECT_TRUE(lowered->name().starts_with("Cudf")) << lowered->name();
    EXPECT_EQ(lowered->id(), node->id());
    EXPECT_EQ(*lowered->outputType(), *node->outputType());
    EXPECT_EQ(lowered->sources(), sources);
    EXPECT_EQ(node->sources(), originalSources);
  }

  // A core plan node that cuDF does not run, and a cuDF plan node that has no
  // core plan node.
  const auto values = PlanBuilder().values({input_}).planNode();
  EXPECT_EQ(detail::findCudfPlanNodeLowering(values), nullptr);
  EXPECT_EQ(
      detail::findCudfPlanNodeLowering(
          std::make_shared<CudfFromVeloxNode>("from-velox", values)),
      nullptr);

  // MarkDistinct lowering rejects masks even if called directly.
  const auto maskedMarkDistinct =
      PlanBuilder()
          .values({input_})
          .project({"c0", "c1 > 15 AS mask"})
          .markDistinct({"m0", "m1"}, {"c0"}, {"mask"})
          .planNode();
  const auto* markDistinctLowering =
      detail::findCudfPlanNodeLowering(maskedMarkDistinct);
  ASSERT_NE(markDistinctLowering, nullptr);
  VELOX_ASSERT_THROW(
      markDistinctLowering->lower(
          maskedMarkDistinct,
          {std::make_shared<CudfFromVeloxNode>(
              "masked_gpu", maskedMarkDistinct->sources()[0])}),
      "MarkDistinct masks are not supported by cuDF");
}

TEST_F(PlanRewriterTest, batchConcatGoesBelowAggregationAndJoinProbe) {
  // TearDown() restores the setting.
  CudfConfig::getInstance().concatOptimizationEnabled = true;
  const auto cases = samples();

  const auto aggregation =
      rewriteToCudfPlan(cases.at("Aggregation").plan, queryCtx_)->sources()[0];
  ASSERT_TRUE(aggregation->is<CudfAggregationNode>());
  EXPECT_TRUE(aggregation->sources()[0]->is<CudfBatchConcatNode>());

  const auto join = rewriteToCudfPlan(cases.at("HashJoinProbe").plan, queryCtx_)
                        ->sources()[0];
  ASSERT_TRUE(join->is<CudfHashJoinNode>());
  EXPECT_TRUE(join->sources()[0]->is<CudfBatchConcatNode>());
  EXPECT_FALSE(join->sources()[1]->is<CudfBatchConcatNode>());

  const auto limit =
      rewriteToCudfPlan(cases.at("Limit").plan, queryCtx_)->sources()[0];
  ASSERT_TRUE(limit->is<CudfLimitNode>());
  EXPECT_FALSE(limit->sources()[0]->is<CudfBatchConcatNode>());

  // CudfBatchConcatNode has no core plan node, only an operator.
  for (const auto& name : {"Aggregation", "HashJoinProbe"}) {
    SCOPED_TRACE(name);
    const auto& plan = cases.at(name).plan;
    const auto legacy = execute(plan, true);
    const auto rewritten = execute(plan, false);
    EXPECT_TRUE(rewritten.operators.contains("CudfBatchConcat"));
    expectCpuResults(plan, legacy, rewritten);
  }
}

TEST_F(PlanRewriterTest, allSupportedLocalPartitionKindsStayOnGpu) {
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(kind);
    // Several batches and four drivers, so that the rows go to more than one
    // consumer. The partial and final steps give the same sums however the
    // rows are spread.
    constexpr int32_t kNumDrivers{4};
    auto builder =
        PlanBuilder().values({input_, input_, input_}).project({"c0", "c1"});
    if (kind == 0) {
      builder.localGather();
    } else if (kind == 1) {
      builder.localPartition({"c0"});
    } else if (kind == 2) {
      builder.localPartitionRoundRobin();
    } else {
      builder.localPartitionRoundRobinRow();
    }
    auto plan = builder.partialAggregation({"c0"}, {"sum(c1)"})
                    .localGather()
                    .finalAggregation()
                    .planNode();
    auto physical = rewriteToCudfPlan(plan, queryCtx_);
    auto finalAggregation = physical->sources()[0];
    ASSERT_TRUE(finalAggregation->is<CudfAggregationNode>());
    auto gather = finalAggregation->sources()[0];
    ASSERT_TRUE(gather->is<CudfLocalPartitionNode>());
    auto partialAggregation = gather->sources()[0];
    ASSERT_TRUE(partialAggregation->is<CudfAggregationNode>());
    auto partition = partialAggregation->sources()[0];
    ASSERT_TRUE(partition->is<CudfLocalPartitionNode>());
    EXPECT_TRUE(partition->sources()[0]->is<CudfFilterProjectNode>());
    auto legacy = execute(plan, true, kNumDrivers);
    auto rewritten = execute(plan, false, kNumDrivers);
    EXPECT_TRUE(legacy.operators.contains("CudfLocalPartition"));
    EXPECT_TRUE(rewritten.operators.contains("CudfLocalPartition"));
    // Only a gather leaves its consumer with one driver.
    const auto numConsumerDrivers = kind == 0 ? 1 : kNumDrivers;
    EXPECT_EQ(legacy.numDrivers, numConsumerDrivers);
    EXPECT_EQ(rewritten.numDrivers, numConsumerDrivers);
    expectCpuResults(plan, legacy, rewritten, kNumDrivers);
  }
}

TEST_F(PlanRewriterTest, physicalExecutionMatchesDriverAdapter) {
  // The samples whose plan returns its rows in order.
  const std::unordered_set<std::string> sorted{"OrderBy", "TopN", "LocalMerge"};
  for (const auto& [name, sample] : samples()) {
    SCOPED_TRACE(name);
    auto legacy = execute(sample.plan, true, sample.maxDrivers);
    ASSERT_TRUE(legacy.operators.contains(sample.gpuOperator));
    auto physical = execute(sample.plan, false, sample.maxDrivers);
    ASSERT_TRUE(physical.operators.contains(sample.gpuOperator));
    EXPECT_EQ(legacy.numDrivers, sample.maxDrivers);
    EXPECT_EQ(physical.numDrivers, sample.maxDrivers);
    expectCpuResults(
        sample.plan,
        legacy,
        physical,
        sample.maxDrivers,
        sorted.contains(name));
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
    expectCpuResults(sample.plan, legacy, physical);
  }
}

class HostOnlyFilterNode final : public core::FilterNode {
 public:
  using core::FilterNode::FilterNode;

  std::string_view name() const override {
    return "HostOnlyFilter";
  }
};

class HostFilterDescriptor final : public CudfPlanNodeDescriptor {
 public:
  bool canHandle(const core::PlanNode& node) const override {
    return typeid(node) == typeid(HostOnlyFilterNode);
  }

  CudfPlanNodeTraits traits(const core::PlanNode&, const core::QueryCtx&)
      const override {
    return {{InputRepresentation::kVelox}, VectorRepresentation::kVelox, {}};
  }

  core::PlanNodePtr withSources(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources) const override {
    VELOX_CHECK_EQ(sources.size(), 1);
    return std::make_shared<HostOnlyFilterNode>(
        node->id(), node->as<core::FilterNode>()->filter(), sources[0]);
  }
};
TEST_F(PlanRewriterTest, filterProjectPreservesCustomFilterContract) {
  registerCudfPlanNodeDescriptor(
      "host-filter-test", std::make_shared<HostFilterDescriptor>());
  SCOPE_EXIT {
    unregisterCudfPlanNodeDescriptor("host-filter-test");
  };

  for (bool gpuSource : {false, true}) {
    SCOPED_TRACE(gpuSource);
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    auto builder = PlanBuilder(ids).values({input_});
    if (gpuSource) {
      builder.project({"c0", "c1 + 1 AS c1"});
    }
    auto filter = builder.filter("c0 > 1").planNode();
    auto custom = std::make_shared<HostOnlyFilterNode>(
        filter->id(),
        filter->as<core::FilterNode>()->filter(),
        filter->sources()[0]);
    auto plan =
        PlanBuilder(custom, ids).project({"c0", "c1 + 1 AS x"}).planNode();
    auto queryCtx = core::QueryCtx::create(executor_.get());
    auto physical = rewriteToCudfPlan(plan, queryCtx);
    ASSERT_TRUE(physical->is<core::ProjectNode>());
    auto retained = physical->sources()[0];
    ASSERT_TRUE(retained->is<HostOnlyFilterNode>());
    EXPECT_EQ(retained->id(), custom->id());
    if (gpuSource) {
      ASSERT_TRUE(retained->sources()[0]->is<CudfToVeloxNode>());
      EXPECT_TRUE(
          retained->sources()[0]->sources()[0]->is<CudfFilterProjectNode>());
    } else {
      EXPECT_EQ(retained, custom);
    }

    auto task =
        AssertQueryBuilder(physical)
            .queryCtx(queryCtx)
            .maxDrivers(1)
            .assertResults(makeRowVector(
                {"c0", "x"},
                {makeFlatVector<int64_t>({2, 3}),
                 makeFlatVector<int64_t>({21 + gpuSource, 41 + gpuSource})}));
    bool foundCpuFilterProject = false;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.planNodeId == plan->id()) {
          EXPECT_EQ(op.operatorType, "FilterProject");
          foundCpuFilterProject = true;
        }
      }
    }
    EXPECT_TRUE(foundCpuFilterProject);

    // With CPU fallback disabled the Project runs on the GPU by itself, above
    // the Filter of the application.
    auto strictQueryCtx = core::QueryCtx::create(
        executor_.get(),
        core::QueryConfig({{CudfConfig::kCudfAllowCpuFallback, "false"}}));
    auto strict = rewriteToCudfPlan(plan, strictQueryCtx);
    ASSERT_TRUE(strict->is<CudfToVeloxNode>());
    auto gpuProject = strict->sources()[0];
    ASSERT_TRUE(gpuProject->is<CudfFilterProjectNode>());
    EXPECT_EQ(gpuProject->id(), plan->id());
    ASSERT_TRUE(gpuProject->sources()[0]->is<CudfFromVeloxNode>());
    auto strictFilter = gpuProject->sources()[0]->sources()[0];
    ASSERT_TRUE(strictFilter->is<HostOnlyFilterNode>());
    EXPECT_EQ(strictFilter->id(), custom->id());
    AssertQueryBuilder(strict)
        .queryCtx(strictQueryCtx)
        .maxDrivers(1)
        .assertResults(makeRowVector(
            {"c0", "x"},
            {makeFlatVector<int64_t>({2, 3}),
             makeFlatVector<int64_t>({21 + gpuSource, 41 + gpuSource})}));
  }
}

// A subclass of a core plan node may add behaviour, so it does not run as the
// cuDF plan node of its base class.
TEST_F(PlanRewriterTest, subclassOfCoreNodeStaysOnCpu) {
  class CustomLimitNode final : public core::LimitNode {
   public:
    using core::LimitNode::LimitNode;
  };
  class CustomFilterNode final : public core::FilterNode {
   public:
    using core::FilterNode::FilterNode;
  };
  class CustomLocalMergeNode final : public core::LocalMergeNode {
   public:
    using core::LocalMergeNode::LocalMergeNode;
  };
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto values = PlanBuilder(ids).values({input_}).planNode();

  auto limit = std::make_shared<CustomLimitNode>("limit", 0, 2, false, values);
  EXPECT_EQ(rewriteToCudfPlan(limit, queryCtx_), limit);

  // The driver adapter leaves its operator on the CPU too.
  {
    auto adapted = execute(limit, /*useDriverAdapter=*/true);
    EXPECT_TRUE(adapted.operators.contains("Limit"));
    EXPECT_FALSE(adapted.operators.contains("CudfLimit"));
    EXPECT_EQ(adapted.results->size(), 2);
  }

  // A local merge is two operators: the CallbackSink that ends the pipeline
  // of each source, and the LocalMerge. The driver adapter leaves both on the
  // CPU, so the CudfOrderBy below hands Velox vectors to the sink.
  {
    auto merge = PlanBuilder()
                     .values({input_})
                     .orderBy({"c1 DESC"}, true)
                     .localMerge({"c1 DESC"})
                     .planNode();
    const auto* coreMerge = merge->as<core::LocalMergeNode>();
    auto customMerge = std::make_shared<CustomLocalMergeNode>(
        "merge",
        coreMerge->sortingKeys(),
        coreMerge->sortingOrders(),
        merge->sources());
    auto adapted = execute(customMerge, /*useDriverAdapter=*/true);
    EXPECT_TRUE(adapted.operators.contains("CudfOrderBy"));
    EXPECT_TRUE(adapted.operators.contains("LocalMerge"));
    EXPECT_FALSE(adapted.operators.contains("CudfLocalMerge"));
    ::facebook::velox::test::assertEqualVectors(
        makeRowVector(
            {"c0", "c1"},
            {makeFlatVector<int64_t>({3, 1, 2, 1}),
             makeFlatVector<int64_t>({40, 30, 20, 10})}),
        adapted.results);
  }

  // The Project still runs on the GPU, but not fused with the Filter.
  auto filter = PlanBuilder().values({input_}).filter("c0 > 1").planNode();
  core::PlanNodePtr customFilter = std::make_shared<CustomFilterNode>(
      "filter", filter->as<core::FilterNode>()->filter(), values);
  auto plan =
      PlanBuilder(customFilter, ids).project({"c0 + 1 AS x"}).planNode();
  auto physical = rewriteToCudfPlan(plan, queryCtx_);
  ASSERT_TRUE(physical->is<CudfToVeloxNode>());
  auto project = physical->sources()[0]->as<CudfFilterProjectNode>();
  ASSERT_NE(project, nullptr);
  ASSERT_TRUE(project->sources()[0]->is<CudfFromVeloxNode>());
  EXPECT_EQ(project->sources()[0]->sources()[0], customFilter);

  // The driver adapter does not fuse the custom Filter into a cuDF operator.
  {
    auto adapted = execute(plan, /*useDriverAdapter=*/true);
    EXPECT_TRUE(adapted.operators.contains("FilterProject"));
    EXPECT_FALSE(adapted.operators.contains("CudfFilterProject"));
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
  // These need an RPC service, vector index format, or fixed-point state to
  // run.
  auto queries = makeRowVector(
      {"t0", "t1"},
      {makeFlatVector<int64_t>({1}), makeArrayVector<float>({{1.0f, 0.0f}})});
  auto indexes = makeRowVector(
      {"u0", "u1"},
      {makeFlatVector<int64_t>({1}),
       makeFlatVector<std::string>({"index"}, VARBINARY())});
  const auto rpcPlan = std::make_shared<core::RPCNode>(
      "rpc",
      gpuRegion().planNode(),
      std::make_shared<core::CallTypedExpr>(
          BIGINT(),
          std::vector<core::TypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
          "test_rpc"),
      "r",
      ROW({"c0", "c1", "r"}, BIGINT()),
      rpc::RPCStreamingMode::kBatch,
      /*dispatchBatchSize=*/16,
      /*executionPhase=*/"SUBMIT");
  const auto vectorSearchPlan =
      PlanBuilder(ids)
          .values({queries})
          .limit(0, 10, false)
          .vectorSearchJoin(
              {"t0"},
              {"u0"},
              PlanBuilder(ids).values({indexes}).limit(0, 10, false).planNode(),
              "t1",
              "u1",
              /*k=*/5,
              {"t0", "id", "score"},
              "id",
              "score",
              core::JoinType::kLeft,
              {{"ivf.nprobe", "4"}})
          .planNode();
  const std::vector<core::PlanNodePtr> planOnly{
      rpcPlan,
      vectorSearchPlan,
      std::make_shared<core::StateHashJoinNode>(
          "state_join",
          "state",
          std::vector<std::string>{"c0"},
          input_->rowType(),
          gpuRegion().planNode()),
  };

  auto checkRewrite = [&](const core::PlanNodePtr& plan) {
    SCOPED_TRACE(plan->name());
    auto physical = rewriteToCudfPlan(plan, queryCtx_);
    EXPECT_EQ(physical->id(), plan->id());
    EXPECT_TRUE(typeid(*physical) == typeid(*plan));
    for (const auto& source : physical->sources()) {
      EXPECT_TRUE(source->is<CudfToVeloxNode>());
    }
    return physical;
  };
  for (const auto& plan : executable) {
    checkRewrite(plan);
    auto legacy = execute(plan, true);
    auto physical = execute(plan, false);
    EXPECT_EQ(legacy.placement, physical.placement);
    expectCpuResults(plan, legacy, physical);
  }
  for (const auto& plan : planOnly) {
    auto physical = checkRewrite(plan);
    if (const auto* rpc = dynamic_cast<const core::RPCNode*>(physical.get())) {
      EXPECT_EQ(rpc->executionPhase(), "SUBMIT");
    }
  }

  // LazyDereferenceNode derives from ProjectNode without its own accept()
  // overload; verify it is rebuilt as LazyDereferenceNode when kept on CPU.
  {
    const auto fieldType = ROW("a", BIGINT());
    const auto structType = ROW("s", fieldType);
    auto ucxMerge = std::make_shared<core::MergeExchangeNode>(
        "ucx_merge",
        structType,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(fieldType, "s")},
        std::vector<core::SortOrder>{core::kAscNullsLast},
        "Presto",
        std::string{core::TransportKind::kUcx});
    auto lazy = std::make_shared<core::LazyDereferenceNode>(
        "lazy",
        std::vector<std::string>{"s"},
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(fieldType, "s")},
        ucxMerge);
    auto cpuQueryCtx = core::QueryCtx::create(
        executor_.get(),
        core::QueryConfig({{CudfConfig::kCudfEnabled, "false"}}));
    auto physical = rewriteToCudfPlan(lazy, cpuQueryCtx);
    EXPECT_TRUE(typeid(*physical) == typeid(core::LazyDereferenceNode));
    ASSERT_EQ(physical->sources().size(), 1);
    EXPECT_TRUE(physical->sources()[0]->is<CudfToVeloxNode>());
  }

  // An unregistered subclass of a core node is rejected rather than sliced to
  // its base class.
  {
    class UnregisteredRowNumberNode final : public core::RowNumberNode {
     public:
      using core::RowNumberNode::RowNumberNode;
    };
    auto custom = std::make_shared<UnregisteredRowNumberNode>(
        "custom_rn",
        std::vector<core::FieldAccessTypedExprPtr>{},
        std::nullopt,
        /*generateRowNumber=*/true,
        gpuRegion().planNode());
    VELOX_ASSERT_THROW(
        rewriteToCudfPlan(custom, queryCtx_),
        "No rebuilding support for RowNumber (custom_rn)");
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
    expectCpuResults(sample.plan, legacy, physical);
  }
}

TEST_F(PlanRewriterTest, partitionedOutputPreservesRewrittenChildren) {
  auto plan = PlanBuilder()
                  .values({input_})
                  .project({"c0", "c1 + 1 AS x"})
                  .partitionedOutput({}, 1)
                  .planNode();
  auto rewritten = rewriteToCudfPlan(plan, queryCtx_);
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
    auto physical = rewriteToCudfPlan(plan, queryCtx_);
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
    expectCpuResults(plan, legacy, rewritten);
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
    auto rewritten = rewriteToCudfPlan(plan, queryCtx_);
    ASSERT_TRUE(rewritten->is<CudfToVeloxNode>());
    EXPECT_EQ(rewritten->sources()[0], plan);
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
