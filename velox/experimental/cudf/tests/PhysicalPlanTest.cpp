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
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gmock/gmock.h>

#include <future>

namespace facebook::velox::cudf_velox::test {
namespace {
using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;
using Kind = VectorRepresentation;
using Need = InputRepresentation;

/// Represents an application-owned node, with no cuDF base class.
class AppNode : public core::PlanNode {
 public:
  AppNode(
      core::PlanNodeId id,
      std::vector<core::PlanNodePtr> sources,
      RowTypePtr outputType,
      CudfPlanNodeTraits traits)
      : PlanNode(std::move(id)),
        sources_(std::move(sources)),
        outputType_(std::move(outputType)),
        traits_(std::move(traits)) {}
  const RowTypePtr& outputType() const override {
    return outputType_;
  }
  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }
  std::string_view name() const override {
    return "AppForward";
  }
  const CudfPlanNodeTraits& traits() const {
    return traits_;
  }

 private:
  void addDetails(std::stringstream&) const override {}
  const std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
  const CudfPlanNodeTraits traits_;
};

class AppDescriptor : public CudfPlanNodeDescriptor {
 public:
  explicit AppDescriptor(bool staleSources = false, bool changeSchema = false)
      : staleSources_(staleSources), changeSchema_(changeSchema) {}
  bool canHandle(const core::PlanNode& node) const override {
    return typeid(node) == typeid(AppNode);
  }
  CudfPlanNodeTraits traits(const core::PlanNode& node, const core::QueryCtx&)
      const override {
    return static_cast<const AppNode&>(node).traits();
  }
  core::PlanNodePtr withSources(
      const core::PlanNodePtr& node,
      std::vector<core::PlanNodePtr> sources) const override {
    if (staleSources_) {
      return node;
    }
    return std::make_shared<AppNode>(
        node->id(),
        std::move(sources),
        changeSchema_ ? ROW({"changed"}, {BIGINT()}) : node->outputType(),
        static_cast<const AppNode&>(*node).traits());
  }

 private:
  const bool staleSources_;
  const bool changeSchema_;
};

class NativeGpuNode : public CudfUnaryPlanNode {
 public:
  NativeGpuNode(core::PlanNodeId id, core::PlanNodePtr source)
      : CudfUnaryPlanNode(id, source, source->outputType()) {}
  std::string_view name() const override {
    return "NativeGpuForward";
  }
  core::PlanNodePtr withSources(
      std::vector<core::PlanNodePtr> sources) const override {
    return copyWithSources<NativeGpuNode>(std::move(sources));
  }

 private:
  void addDetails(std::stringstream&) const override {}
};

class ForwardOperator : public exec::Operator {
 public:
  ForwardOperator(
      int32_t id,
      exec::DriverCtx* ctx,
      const core::PlanNodePtr& node,
      Need input)
      : Operator(ctx, node->outputType(), id, node->id(), "AppForward"),
        expectedInput_(input) {}
  bool needsInput() const override {
    return !done_ && !pending_;
  }
  void addInput(RowVectorPtr input) override {
    const bool device = dynamic_cast<CudfVector*>(input.get()) != nullptr;
    VELOX_CHECK(
        expectedInput_ == Need::kEither ||
        (expectedInput_ == Need::kCudf) == device);
    pending_ = std::move(input);
  }
  RowVectorPtr getOutput() override {
    return std::exchange(pending_, nullptr);
  }
  void noMoreInput() override {
    Operator::noMoreInput();
    done_ = true;
  }
  bool isFinished() override {
    return done_ && !pending_;
  }
  exec::BlockingReason isBlocked(ContinueFuture*) override {
    return exec::BlockingReason::kNotBlocked;
  }
  void close() override {
    pending_.reset();
    Operator::close();
  }

 private:
  const Need expectedInput_;
  RowVectorPtr pending_;
  bool done_{false};
};

class AppTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto* app = dynamic_cast<const AppNode*>(node.get())) {
      return std::make_unique<ForwardOperator>(
          id, ctx, node, app->traits().inputs.at(0));
    }
    if (dynamic_cast<const NativeGpuNode*>(node.get())) {
      return std::make_unique<ForwardOperator>(id, ctx, node, Need::kCudf);
    }
    return nullptr;
  }
};

class PhysicalPlanTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = CudfConfig::getInstance();
    saved_.emplace(config);
    config.enabled = true;
    config.enableDriverAdapter = false;
    config.allowCpuFallback = true;
    config.concatOptimizationEnabled = false;
    registerCudf();
    registerCudfPlanNodeDescriptor(
        "app-test", std::make_shared<AppDescriptor>());
    exec::Operator::registerOperator(std::make_unique<AppTranslator>());
    queryCtx_ = core::QueryCtx::create(executor_.get());
    input_ = makeRowVector({"k"}, {makeFlatVector<int64_t>({3, 1, 2})});
  }
  void TearDown() override {
    unregisterCudfPlanNodeDescriptor("app-test");
    input_.reset();
    queryCtx_.reset();
    unregisterCudf();
    CudfConfig::getInstance() = *saved_;
    OperatorTestBase::TearDown();
  }
  core::PlanNodePtr rewrite(
      const core::PlanNodePtr& plan,
      Kind output = Kind::kVelox) {
    CudfPlanRewriter::Config config;
    config.queryCtx = queryCtx_;
    config.taskOutput = output;
    return CudfPlanRewriter::rewrite(plan, config);
  }
  core::PlanNodePtr values() {
    return PlanBuilder().values({input_}).planNode();
  }
  core::PlanNodePtr app(
      core::PlanNodePtr source,
      Need input,
      std::optional<Kind> output,
      bool preserve = false) {
    return std::make_shared<AppNode>(
        "app",
        std::vector<core::PlanNodePtr>{source},
        source->outputType(),
        CudfPlanNodeTraits{
            {input},
            output,
            preserve ? std::optional<size_t>{0} : std::nullopt});
  }
  size_t count(const core::PlanNodePtr& node, std::string_view name) {
    size_t result = node->name() == name;
    for (const auto& source : node->sources()) {
      result += count(source, name);
    }
    return result;
  }
  void check(const core::PlanNodePtr& plan, const RowVectorPtr& expected) {
    AssertQueryBuilder(plan).queryCtx(queryCtx_).maxDrivers(4).assertResults(
        expected);
  }
  std::optional<CudfConfig> saved_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  RowVectorPtr input_;
};

TEST_F(PhysicalPlanTest, validationChecksRepresentations) {
  // A descriptor-backed node between explicit conversions validates and runs.
  auto upload = std::make_shared<CudfFromVeloxNode>("upload", values());
  auto plan = std::make_shared<CudfToVeloxNode>(
      "download", app(upload, Need::kCudf, Kind::kCudf));
  validateCudfPlan(plan, *queryCtx_);
  check(plan, input_);

  VELOX_ASSERT_THROW(
      validateCudfPlan(
          std::make_shared<NativeGpuNode>("native", values()),
          *queryCtx_,
          Kind::kCudf),
      "Vector representation mismatch");
  VELOX_ASSERT_THROW(
      validateCudfPlan(upload, *queryCtx_), "requested task representation");
  auto merge = std::make_shared<core::MergeExchangeNode>(
      "merge",
      input_->rowType(),
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "k")},
      std::vector<core::SortOrder>{core::SortOrder(true, false)},
      "Presto",
      std::string{core::TransportKind::kUcx});
  VELOX_ASSERT_THROW(
      validateCudfPlan(merge, *queryCtx_, Kind::kCudf),
      "requires an explicit Exchange and CudfOrderBy");
}

TEST_F(PhysicalPlanTest, conversionsRejectUnsupportedTypes) {
  auto data = makeRowVector(
      {"k", "m"},
      {makeFlatVector<int64_t>({1}),
       makeMapVector<int64_t, int64_t>({{{1, 10}}})});
  auto values = PlanBuilder().values({data}).planNode();

  VELOX_ASSERT_USER_THROW(
      validateCudfPlan(
          std::make_shared<CudfToVeloxNode>(
              "download",
              std::make_shared<CudfFromVeloxNode>("upload", values)),
          *queryCtx_),
      "Unsupported cuDF conversion type at upload");

  // The node of the application produces cuDF vectors, and the task needs
  // Velox vectors.
  VELOX_ASSERT_USER_THROW(
      rewrite(app(values, Need::kVelox, Kind::kCudf)),
      "(app) across a cuDF boundary");
}

TEST_F(PhysicalPlanTest, driverAdapterRejectsCudfPlanNodes) {
  unregisterCudf();
  CudfConfig::getInstance().enableDriverAdapter = true;
  registerCudf();
  auto plan = std::make_shared<CudfToVeloxNode>(
      "download", std::make_shared<CudfFromVeloxNode>("upload", values()));
  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(plan).queryCtx(queryCtx_).copyResults(pool()),
      "Disable cudf.driver_adapter_enabled before executing physical cuDF plans");
}

TEST_F(PhysicalPlanTest, detailsShowKeysAndExpressions) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  auto details = [&](const core::PlanNodePtr& plan, std::string_view name) {
    auto physical = rewrite(plan);
    const auto* node = core::PlanNode::findFirstNode(
        physical.get(),
        [&](const core::PlanNode* node) { return node->name() == name; });
    VELOX_CHECK_NOT_NULL(node, "{}", physical->toString(true, true));
    return node->toString(true, false);
  };
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto values = [&] { return PlanBuilder(ids).values({data}); };

  EXPECT_THAT(
      details(
          values().filter("c0 > 1").project({"c1 + 1 AS x"}).planNode(),
          "CudfFilterProject"),
      testing::AllOf(
          testing::HasSubstr("filter: gt("),
          testing::HasSubstr("expressions: (x:BIGINT, plus(")));
  EXPECT_THAT(
      details(
          values().singleAggregation({"c0"}, {"sum(c1) AS s"}).planNode(),
          "CudfAggregation"),
      testing::HasSubstr("SINGLE [c0] s := sum("));
  EXPECT_THAT(
      details(values().orderBy({"c1 DESC"}, false).planNode(), "CudfOrderBy"),
      testing::HasSubstr("FINAL c1 DESC NULLS LAST"));
  EXPECT_THAT(
      details(values().topN({"c1"}, 5, false).planNode(), "CudfTopN"),
      testing::HasSubstr("FINAL 5 c1 ASC NULLS LAST"));
  EXPECT_THAT(
      details(
          values()
              .hashJoin(
                  {"c0"},
                  {"u0"},
                  PlanBuilder(ids)
                      .values({data})
                      .project({"c0 AS u0", "c1 AS u1"})
                      .planNode(),
                  "c1 < u1",
                  {"c0", "u1"})
              .planNode(),
          "CudfHashJoin"),
      testing::AllOf(
          testing::HasSubstr("INNER c0=u0"),
          testing::HasSubstr("filter: lt(")));
}

TEST_F(PhysicalPlanTest, aggregationConstructsWithoutCpuPlan) {
  core::AggregationNode::Aggregate sum;
  sum.call = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "k")},
      "sum");
  sum.rawInputTypes = {BIGINT()};
  auto source = std::make_shared<CudfFromVeloxNode>("upload", values());
  auto aggregation = std::make_shared<CudfAggregationNode>(
      "aggregation",
      source,
      ROW({"s"}, {BIGINT()}),
      core::AggregationNode::Step::kSingle,
      std::vector<core::FieldAccessTypedExprPtr>{},
      std::vector<core::AggregationNode::Aggregate>{sum},
      std::vector<TypePtr>{BIGINT()},
      /*ignoreNullKeys=*/false);
  auto plan = std::make_shared<CudfToVeloxNode>("download", aggregation);
  validateCudfPlan(plan, *queryCtx_);
  check(plan, makeRowVector({"s"}, {makeFlatVector<int64_t>({6})}));
}

TEST_F(PhysicalPlanTest, localMergeSplitsEverySourceWithBoundaryAdapter) {
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

TEST_F(PhysicalPlanTest, constructorsRejectUnsupportedArguments) {
  using Keys = std::vector<core::FieldAccessTypedExprPtr>;
  using Orders = std::vector<core::SortOrder>;
  auto source = std::make_shared<CudfFromVeloxNode>("upload", values());
  auto key = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "k");
  auto orderBy = [&](Keys keys, Orders orders) {
    return std::make_shared<CudfOrderByNode>(
        "sort",
        source,
        input_->rowType(),
        std::move(keys),
        std::move(orders),
        /*isPartial=*/false);
  };
  auto topN = [&](Keys keys, Orders orders, int32_t count) {
    return std::make_shared<CudfTopNNode>(
        "topn",
        source,
        input_->rowType(),
        std::move(keys),
        std::move(orders),
        count,
        /*isPartial=*/false);
  };

  VELOX_ASSERT_USER_THROW(
      orderBy({}, {}), "CudfOrderBy must specify sorting keys");
  VELOX_ASSERT_USER_THROW(
      orderBy({key}, {}),
      "Number of sorting keys and sorting orders in CudfOrderBy must be the same");
  VELOX_ASSERT_USER_THROW(
      orderBy({nullptr}, {core::kAscNullsLast}),
      "CudfOrderBy sorting keys must not be null");
  VELOX_ASSERT_USER_THROW(
      orderBy({key, key}, {core::kAscNullsLast, core::kAscNullsLast}),
      "CudfOrderBy must specify unique sorting keys. Found duplicate key: k");

  VELOX_ASSERT_USER_THROW(
      topN({}, {}, 1), "CudfTopN must specify sorting keys");
  VELOX_ASSERT_USER_THROW(
      topN({key}, {}, 1),
      "Number of sorting keys and sorting orders in CudfTopN must be the same");
  VELOX_ASSERT_USER_THROW(
      topN({key, key}, {core::kAscNullsLast, core::kAscNullsLast}, 1),
      "CudfTopN must specify unique sorting keys. Found duplicate key: k");
  VELOX_ASSERT_USER_THROW(
      topN({key}, {core::kAscNullsLast}, 0),
      "CudfTopN must specify greater than zero number of rows to keep");

  // The placement rules of a hash join that need no QueryCtx.
  auto hashJoin = [&](core::JoinType joinType, bool nullAware, bool filter) {
    return std::make_shared<CudfHashJoinNode>(
        "join",
        source,
        std::make_shared<CudfFromVeloxNode>("build-upload", values()),
        input_->rowType(),
        joinType,
        nullAware,
        Keys{key},
        Keys{key},
        filter ? std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true)
               : core::TypedExprPtr{});
  };
  VELOX_ASSERT_USER_THROW(
      hashJoin(core::JoinType::kAnti, true, true),
      "HashJoin null-aware anti join with filter not implemented");
  EXPECT_NE(hashJoin(core::JoinType::kAnti, true, false), nullptr);
  EXPECT_NE(hashJoin(core::JoinType::kAnti, false, true), nullptr);

  VELOX_ASSERT_THROW(source->serialize(), "is not serializable");
  VELOX_ASSERT_THROW(
      orderBy({key}, {core::kAscNullsLast})->serialize(),
      "CudfOrderBy plan node is not serializable");
}

TEST_F(PhysicalPlanTest, constructTopNRowNumberWithoutCpuCounterpart) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2, 1, 3}),
       makeFlatVector<int64_t>({10, 20, 30, 40})});
  auto source = std::make_shared<CudfFromVeloxNode>(
      "upload", PlanBuilder().values({data}).planNode());
  auto topN = std::make_shared<CudfTopNRowNumberNode>(
      "top-n",
      source,
      data->rowType(),
      core::TopNRowNumberNode::RankFunction::kRowNumber,
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c1")},
      std::vector<core::SortOrder>{core::SortOrder(true, false)},
      /*limit=*/1,
      /*generateRowNumber=*/false);
  AssertQueryBuilder(std::make_shared<CudfToVeloxNode>("download", topN))
      .assertResults(makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeFlatVector<int64_t>({10, 20, 40})}));
}

TEST_F(PhysicalPlanTest, cudfMaxDriversLimitsGpuPipelines) {
  // Returns the driver count of the pipeline that runs 'operatorType'.
  auto numDrivers = [](const exec::Task& task, std::string_view operatorType) {
    for (const auto& pipeline : task.taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType == operatorType) {
          return op.numDrivers;
        }
      }
    }
    VELOX_FAIL("No {} operator", operatorType);
  };
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  // A pipeline with cuDF nodes feeds a CPU-only pipeline.
  auto upload = std::make_shared<CudfFromVeloxNode>(
      "upload", PlanBuilder(ids).values({input_}, true).planNode());
  auto download = std::make_shared<CudfToVeloxNode>("download", upload);
  auto plan = PlanBuilder(download, ids)
                  .localPartitionRoundRobin()
                  .project({"k + 1 AS k"})
                  .planNode();
  // A final sort must stay single-threaded under any limit.
  auto sort = std::make_shared<CudfToVeloxNode>(
      "sort-download",
      std::make_shared<CudfOrderByNode>(
          "sort",
          std::make_shared<CudfFromVeloxNode>(
              "sort-upload", PlanBuilder().values({input_}, true).planNode()),
          input_->rowType(),
          std::vector<core::FieldAccessTypedExprPtr>{
              std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "k")},
          std::vector<core::SortOrder>{core::kAscNullsLast},
          /*isPartial=*/false));

  for (const auto maxDrivers :
       std::vector<std::optional<int32_t>>{std::nullopt, 2}) {
    SCOPED_TRACE(maxDrivers.value_or(0));
    CudfConfig::getInstance().maxDrivers = maxDrivers;
    std::shared_ptr<exec::Task> task;
    AssertQueryBuilder(plan).queryCtx(queryCtx_).maxDrivers(8).copyResults(
        pool(), task);
    EXPECT_EQ(numDrivers(*task, "CudfFromVelox"), maxDrivers.value_or(8));
    EXPECT_EQ(numDrivers(*task, "FilterProject"), 8);

    task = AssertQueryBuilder(sort)
               .queryCtx(queryCtx_)
               .maxDrivers(8)
               .assertResults(
                   makeRowVector({"k"}, {makeFlatVector<int64_t>({1, 2, 3})}));
    EXPECT_EQ(numDrivers(*task, "CudfOrderBy"), 1);
  }
}

TEST_F(PhysicalPlanTest, externalDeviceNodeGetsConversionsAndExecutes) {
  auto original = app(values(), Need::kCudf, Kind::kCudf);
  auto physical = rewrite(original);
  EXPECT_EQ(count(physical, "CudfFromVelox"), 1);
  EXPECT_EQ(count(physical, "CudfToVelox"), 1);
  EXPECT_EQ(original->sources()[0]->name(), "Values");
  check(physical, input_);
  EXPECT_EQ(rewrite(physical), physical);
}

TEST_F(PhysicalPlanTest, intrinsicDeviceNodeNeedsNoDescriptor) {
  auto node = std::make_shared<NativeGpuNode>("native", values());
  auto physical = rewrite(node);
  check(physical, input_);
  EXPECT_EQ(count(physical, "CudfFromVelox"), 1);
  EXPECT_EQ(count(physical, "CudfToVelox"), 1);
}

TEST_F(PhysicalPlanTest, externalHostNodePreservesGpuRegions) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto source =
      PlanBuilder(ids).values({input_}).project({"k + 1 AS k"}).planNode();
  auto middle = app(source, Need::kVelox, Kind::kVelox);
  auto plan = PlanBuilder(middle, ids).orderBy({"k"}, false).planNode();
  auto physical = rewrite(plan);
  EXPECT_EQ(count(physical, "CudfFromVelox"), 2);
  EXPECT_EQ(count(physical, "CudfToVelox"), 2);
  EXPECT_EQ(count(physical, "CudfFilterProject"), 1);
  EXPECT_EQ(count(physical, "CudfOrderBy"), 1);
  check(physical, makeRowVector({"k"}, {makeFlatVector<int64_t>({2, 3, 4})}));
}

TEST_F(PhysicalPlanTest, representationPreservingNodeDoesNotForceConversion) {
  for (bool gpu : {false, true}) {
    auto source = gpu ? PlanBuilder().values({input_}).project({"k"}).planNode()
                      : values();
    auto physical = rewrite(app(source, Need::kEither, {}, true));
    EXPECT_EQ(count(physical, "CudfFromVelox"), gpu ? 1 : 0);
    EXPECT_EQ(count(physical, "CudfToVelox"), gpu ? 1 : 0);
    check(physical, input_);
  }
}

TEST_F(PhysicalPlanTest, perSourceContractsAndSelectedOutputSource) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto left = PlanBuilder(ids).values({input_}).project({"k"}).planNode();
  auto right = PlanBuilder(ids).values({input_}).planNode();
  auto node = std::make_shared<AppNode>(
      "multi",
      std::vector<core::PlanNodePtr>{left, right},
      input_->rowType(),
      CudfPlanNodeTraits{{Need::kEither, Need::kVelox}, {}, 0});
  auto physical = rewrite(node, Kind::kCudf);
  EXPECT_EQ(physical->sources()[0]->name(), "CudfFilterProject");
  EXPECT_EQ(physical->sources()[1], right);
  validateCudfPlan(physical, *queryCtx_, Kind::kCudf);
}

TEST_F(PhysicalPlanTest, sourceAndSinkContracts) {
  auto source = std::make_shared<AppNode>(
      "gpu-source",
      std::vector<core::PlanNodePtr>{},
      input_->rowType(),
      CudfPlanNodeTraits{{}, Kind::kCudf, {}});
  EXPECT_EQ(rewrite(source)->name(), "CudfToVelox");
  EXPECT_EQ(rewrite(source, Kind::kCudf), source);
  auto sink = std::make_shared<AppNode>(
      "sink",
      std::vector<core::PlanNodePtr>{values()},
      input_->rowType(),
      CudfPlanNodeTraits{{Need::kCudf}, {}, {}});
  auto physical = rewrite(sink);
  EXPECT_EQ(physical->name(), "AppForward");
  EXPECT_EQ(physical->sources()[0]->name(), "CudfFromVelox");
  EXPECT_EQ(count(physical, "CudfToVelox"), 0);
}

TEST_F(PhysicalPlanTest, badRebuilderCannotDiscardRewrittenSources) {
  unregisterCudfPlanNodeDescriptor("app-test");
  registerCudfPlanNodeDescriptor(
      "app-test", std::make_shared<AppDescriptor>(true));
  VELOX_ASSERT_THROW(
      rewrite(app(values(), Need::kCudf, Kind::kCudf)),
      "did not install the rewritten sources");
}

TEST_F(PhysicalPlanTest, validationAndContextAreExplicit) {
  auto invalid = std::make_shared<NativeGpuNode>("native", values());
  CudfPlanRewriter::Config missingContext;
  VELOX_ASSERT_THROW(
      CudfPlanRewriter::rewrite(values(), missingContext),
      "requires the task's QueryCtx");
  auto valid = rewrite(invalid, Kind::kCudf);
  VELOX_ASSERT_THROW(
      validateCudfPlan(valid, *queryCtx_), "requested task representation");
}

TEST_F(PhysicalPlanTest, badRebuilderCannotChangeOutputSchema) {
  unregisterCudfPlanNodeDescriptor("app-test");
  registerCudfPlanNodeDescriptor(
      "app-test", std::make_shared<AppDescriptor>(false, true));
  VELOX_ASSERT_THROW(
      rewrite(app(values(), Need::kCudf, Kind::kCudf)),
      "must preserve its output schema");
}

TEST_F(PhysicalPlanTest, unsupportedColumnsCanBeRemovedBeforeGpuExecution) {
  auto maps =
      makeMapVector<int64_t, int64_t>({{{1, 10}}, {{2, 20}}, {{3, 30}}});
  auto data =
      makeRowVector({"k", "m"}, {makeFlatVector<int64_t>({3, 1, 2}), maps});
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"k"})
                  .orderBy({"k"}, false)
                  .planNode();
  auto physical = rewrite(plan);
  auto sort = core::PlanNode::findFirstNode<CudfOrderByNode>(physical.get());
  ASSERT_NE(sort, nullptr);
  ASSERT_TRUE(sort->sources()[0]->is<CudfFromVeloxNode>());
  EXPECT_TRUE(sort->sources()[0]->sources()[0]->is<core::ProjectNode>());
  EXPECT_EQ(count(physical, "CudfFromVelox"), 1);
  check(physical, input_);
}

TEST_F(PhysicalPlanTest, realQueryContextControlsLoweringAndFallback) {
  queryCtx_ = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig({{CudfConfig::kCudfEnabled, "false"}}));
  auto plan = PlanBuilder().values({input_}).project({"k + 1 AS k"}).planNode();
  EXPECT_EQ(rewrite(plan), plan);
  check(plan, makeRowVector({"k"}, {makeFlatVector<int64_t>({4, 2, 3})}));

  queryCtx_ = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(
          {{CudfConfig::kCudfEnabled, "true"},
           {CudfConfig::kCudfAllowCpuFallback, "false"}}));
  auto unsupported = PlanBuilder().values({input_}).rowNumber({}).planNode();
  VELOX_ASSERT_THROW(rewrite(unsupported), "CPU fallback is disabled");
  auto partition = PlanBuilder().values({input_}).localPartition({}).planNode();
  auto scaled = core::LocalPartitionNode::Builder(
                    *partition->as<core::LocalPartitionNode>())
                    .scaleWriter(true)
                    .build();
  VELOX_ASSERT_THROW(rewrite(scaled), "CPU fallback is disabled");
}

TEST_F(PhysicalPlanTest, httpAndUcxOutputContracts) {
  for (auto transport :
       {core::TransportKind::kInMemory, core::TransportKind::kUcx}) {
    auto plan = PlanBuilder()
                    .values({input_})
                    .partialAggregation({}, {"sum(k)"})
                    .partitionedOutput(
                        {}, 1, false, {}, "Presto", std::string{transport})
                    .planNode();
    auto physical = rewrite(plan);
    EXPECT_TRUE(physical->is<core::PartitionedOutputNode>());
    EXPECT_EQ(
        physical->sources()[0]->name(),
        transport == core::TransportKind::kUcx ? "CudfAggregation"
                                               : "CudfToVelox");
    auto aggregation =
        core::PlanNode::findFirstNode<CudfAggregationNode>(physical.get());
    ASSERT_NE(aggregation, nullptr);
    ASSERT_EQ(aggregation->intermediateTypes().size(), 1);
    EXPECT_TRUE(*aggregation->intermediateTypes()[0] == *BIGINT());
  }
}

TEST_F(PhysicalPlanTest, concurrentFragmentsShareQueryContext) {
  auto plan = PlanBuilder().values({input_}).project({"k + 1 AS k"}).planNode();
  std::vector<std::future<core::PlanNodePtr>> fragments;
  for (size_t i = 0; i < 8; ++i) {
    fragments.push_back(std::async(std::launch::async, [&, plan] {
      core::PlanNodePtr result;
      for (size_t repeat = 0; repeat < 20; ++repeat) {
        result = rewrite(plan);
      }
      return result;
    }));
  }
  for (auto& fragment : fragments) {
    EXPECT_EQ(count(fragment.get(), "CudfFilterProject"), 1);
  }
}

TEST_F(PhysicalPlanTest, httpAndUcxInputContracts) {
  const auto inputType = ROW({"s"}, {BIGINT()});
  for (auto transport :
       {core::TransportKind::kInMemory, core::TransportKind::kUcx}) {
    auto plan = PlanBuilder()
                    .exchange(inputType, "Presto", std::string{transport})
                    .finalAggregation({}, {"sum(s)"}, {{BIGINT()}})
                    .planNode();
    auto physical = rewrite(plan);
    auto aggregation =
        core::PlanNode::findFirstNode<CudfAggregationNode>(physical.get());
    ASSERT_NE(aggregation, nullptr);
    EXPECT_EQ(
        aggregation->sources()[0]->name(),
        transport == core::TransportKind::kUcx ? "Exchange" : "CudfFromVelox");
  }
}

TEST_F(PhysicalPlanTest, ucxMergeRequiresExplicitSort) {
  auto type = ROW({"s"}, {BIGINT()});
  auto merge = std::make_shared<core::MergeExchangeNode>(
      "merge",
      type,
      std::vector<core::FieldAccessTypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "s")},
      std::vector<core::SortOrder>{core::SortOrder(true, false)},
      "Presto",
      std::string{core::TransportKind::kUcx});
  auto physical = rewrite(merge, Kind::kCudf);
  EXPECT_EQ(count(physical, "CudfOrderBy"), 1);
  EXPECT_EQ(physical->sources()[0]->id(), merge->id());
}
} // namespace
} // namespace facebook::velox::cudf_velox::test
