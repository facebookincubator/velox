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

#include <string>

#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/OperatorTraceReader.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/trace/TraceUtil.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/tool/trace/MergeJoinReplayer.h"
#include "velox/tool/trace/TraceReplayRunner.h"

// using namespace facebook::velox;
// using namespace facebook::velox::core;
// using namespace facebook::velox::common;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
// using namespace facebook::velox::connector;
// using namespace facebook::velox::connector::hive;
// using namespace facebook::velox::dwio::common;
// using namespace facebook::velox::common::testutil;
// using namespace facebook::velox::common::hll;

namespace facebook::velox::tool::trace::test {
class MergeJoinReplayerTest : public HiveConnectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    HiveConnectorTestBase::SetUpTestCase();
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    Type::registerSerDe();
    common::Filter::registerSerDe();
    connector::hive::HiveConnector::registerSerDe();
    core::PlanNode::registerSerDe();
    velox::exec::trace::registerDummySourceSerDe();
    core::ITypedExpr::registerSerDe();
    registerPartitionFunctionSerDe();
  }

  void TearDown() override {
    probeInput_.clear();
    buildInput_.clear();
    HiveConnectorTestBase::TearDown();
  }

  struct PlanWithSplits {
    core::PlanNodePtr plan;
    core::PlanNodeId probeScanId;
    core::PlanNodeId buildScanId;
    std::unordered_map<core::PlanNodeId, std::vector<exec::Split>> splits;

    explicit PlanWithSplits(
        const core::PlanNodePtr& _plan,
        const core::PlanNodeId& _probeScanId = "",
        const core::PlanNodeId& _buildScanId = "",
        const std::unordered_map<
            core::PlanNodeId,
            std::vector<velox::exec::Split>>& _splits = {})
        : plan(_plan),
          probeScanId(_probeScanId),
          buildScanId(_buildScanId),
          splits(_splits) {}
  };

  RowTypePtr concat(const RowTypePtr& a, const RowTypePtr& b) {
    std::vector<std::string> names = a->names();
    std::vector<TypePtr> types = a->children();

    for (auto i = 0; i < b->size(); ++i) {
      names.push_back(b->nameOf(i));
      types.push_back(b->childAt(i));
    }

    return ROW(std::move(names), std::move(types));
  }

  std::vector<RowVectorPtr>
  makeVectors(int32_t count, int32_t rowsPerVector, const RowTypePtr& rowType) {
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(count);

    for (int32_t i = 0; i < count; ++i) {
      const int32_t startRow = i * rowsPerVector;
      std::vector<VectorPtr> children;

      for (int32_t col = 0; col < rowType->size(); ++col) {
        const auto& type = rowType->childAt(col);

        // First column is the join key and must be sorted
        if (col == 0) {
          if (type->kind() == TypeKind::BIGINT) {
            children.push_back(
                makeFlatVector<int64_t>(rowsPerVector, [startRow](auto row) {
                  return startRow + row;
                }));
          } else {
            VELOX_FAIL("Unsupported join key type: {}", type->toString());
          }
        } else {
          // Other columns can have random data using BatchMaker
          children.push_back(VELOX_DYNAMIC_TYPE_DISPATCH(
              velox::test::BatchMaker::createVector,
              type->kind(),
              type,
              rowsPerVector,
              *pool()));
        }
      }

      vectors.push_back(makeRowVector(rowType->names(), children));
    }

    return vectors;
  }

  std::vector<Split> makeSplits(
      const std::vector<RowVectorPtr>& inputs,
      const std::string& path,
      memory::MemoryPool* writerPool) {
    std::vector<Split> splits;
    for (auto i = 0; i < 4; ++i) {
      const std::string filePath = fmt::format("{}/{}", path, i);
      writeToFile(filePath, inputs);
      splits.emplace_back(makeHiveConnectorSplit(filePath));
    }

    return splits;
  }

  PlanWithSplits createPlan(
      const std::string& tableDir,
      core::JoinType joinType,
      const std::vector<std::string>& probeKeys,
      const std::vector<std::string>& buildKeys,
      const std::vector<RowVectorPtr>& probeInput,
      const std::vector<RowVectorPtr>& buildInput) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const std::vector<Split> probeSplits =
        makeSplits(probeInput, fmt::format("{}/probe", tableDir), pool());
    const std::vector<Split> buildSplits =
        makeSplits(buildInput, fmt::format("{}/build", tableDir), pool());
    core::PlanNodeId probeScanId;
    core::PlanNodeId buildScanId;
    const auto outputColumns = concat(
                                   asRowType(buildInput_[0]->type()),
                                   asRowType(probeInput_[0]->type()))
                                   ->names();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .tableScan(probeType_)
                    .capturePlanNodeId(probeScanId)
                    .mergeJoin(
                        probeKeys,
                        buildKeys,
                        PlanBuilder(planNodeIdGenerator)
                            .tableScan(buildType_)
                            .capturePlanNodeId(buildScanId)
                            .planNode(),
                        /*filter=*/"",
                        outputColumns,
                        joinType)
                    .capturePlanNodeId(traceNodeId_)
                    .planNode();
    return PlanWithSplits{
        plan,
        probeScanId,
        buildScanId,
        {{probeScanId, probeSplits}, {buildScanId, buildSplits}}};
  }

  core::PlanNodeId traceNodeId_;
  RowTypePtr probeType_{
      ROW({"t0", "t1", "t2", "t3"}, {BIGINT(), VARCHAR(), SMALLINT(), REAL()})};

  RowTypePtr buildType_{
      ROW({"u0", "u1", "u2", "u3"},
          {BIGINT(), INTEGER(), SMALLINT(), VARCHAR()})};
  std::vector<RowVectorPtr> probeInput_ = makeVectors(5, 100, probeType_);
  std::vector<RowVectorPtr> buildInput_ = makeVectors(3, 100, buildType_);

  const std::vector<std::string> probeKeys_{"t0"};
  const std::vector<std::string> buildKeys_{"u0"};
  const std::shared_ptr<TempDirectoryPath> testDir_ =
      TempDirectoryPath::create();
  const std::string tableDir_ =
      fmt::format("{}/{}", testDir_->getPath(), "table");
};

TEST_F(MergeJoinReplayerTest, basic) {
  const auto planWithSplits = createPlan(
      tableDir_,
      core::JoinType::kInner,
      probeKeys_,
      buildKeys_,
      probeInput_,
      buildInput_);
  AssertQueryBuilder builder(planWithSplits.plan);
  for (const auto& [planNodeId, nodeSplits] : planWithSplits.splits) {
    builder.splits(planNodeId, nodeSplits);
  }
  const auto result = builder.copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "basic");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(
      tableDir_,
      core::JoinType::kInner,
      probeKeys_,
      buildKeys_,
      probeInput_,
      buildInput_);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, traceNodeId_);
  for (const auto& [planNodeId, nodeSplits] : tracePlanWithSplits.splits) {
    traceBuilder.splits(planNodeId, nodeSplits);
  }
  auto traceResult = traceBuilder.copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto taskId = task->taskId();
  const auto replayingResult = MergeJoinReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   traceNodeId_,
                                   "MergeJoin",
                                   /*spillBaseDir=*/"",
                                   /*driverIds=*/"",
                                   /*queryCapacity=*/0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(MergeJoinReplayerTest, runner) {
  const auto testDir = TempDirectoryPath::create();
  const auto traceRoot = fmt::format("{}/{}", testDir->getPath(), "traceRoot");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(
      tableDir_,
      core::JoinType::kInner,
      probeKeys_,
      buildKeys_,
      probeInput_,
      buildInput_);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.config(core::QueryConfig::kEnableOperatorBatchSizeStats, true)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, traceNodeId_);
  for (const auto& [planNodeId, nodeSplits] : tracePlanWithSplits.splits) {
    traceBuilder.splits(planNodeId, nodeSplits);
  }
  auto traceResult = traceBuilder.copyResults(pool(), task);

  const auto taskTraceDir = exec::trace::getTaskTraceDirectory(
      traceRoot, task->queryCtx()->queryId(), task->taskId());
  const auto leftOperatorTraceDir = exec::trace::getOpTraceDirectory(
      taskTraceDir,
      traceNodeId_,
      /*pipelineId=*/0,
      /*driverId=*/0);
  const auto leftSummary =
      exec::trace::OperatorTraceSummaryReader(leftOperatorTraceDir, pool())
          .read();
  ASSERT_EQ(leftSummary.opType, "MergeJoin");
  ASSERT_GT(leftSummary.peakMemory, 0);
  ASSERT_GT(leftSummary.inputRows, 0);
  // NOTE: the input bytes is 0 because of the lazy materialization.
  ASSERT_EQ(leftSummary.inputBytes, 0);
  ASSERT_EQ(leftSummary.rawInputRows, 0);
  ASSERT_EQ(leftSummary.rawInputBytes, 0);

  const auto rightOperatorTraceDir = exec::trace::getOpTraceDirectory(
      taskTraceDir,
      traceNodeId_,
      /*pipelineId=*/1,
      /*driverId=*/0);
  const auto rightSummary =
      exec::trace::OperatorTraceSummaryReader(rightOperatorTraceDir, pool())
          .read();
  ASSERT_EQ(rightSummary.opType, "CallbackSink");
  // NOTE: CallbackSink does not have its own memory pool.
  ASSERT_EQ(rightSummary.peakMemory, 0);
  ASSERT_GT(rightSummary.inputRows, 0);
  // NOTE: the input bytes is 0 because of the lazy materialization.
  ASSERT_EQ(rightSummary.inputBytes, 0);
  ASSERT_EQ(rightSummary.rawInputRows, 0);
  ASSERT_EQ(rightSummary.rawInputBytes, 0);

  FLAGS_root_dir = traceRoot;
  FLAGS_query_id = task->queryCtx()->queryId();
  FLAGS_task_id = task->taskId();
  FLAGS_node_id = traceNodeId_;
  FLAGS_summary = true;
  {
    TraceReplayRunner runner;
    runner.init();
    runner.run();
  }

  FLAGS_task_id = task->taskId();
  FLAGS_driver_ids = "";
  FLAGS_summary = false;
  {
    TraceReplayRunner runner;
    runner.init();
    runner.run();
  }
}
} // namespace facebook::velox::tool::trace::test
