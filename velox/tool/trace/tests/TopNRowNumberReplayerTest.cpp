/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/trace/TraceUtil.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/tool/trace/TopNRowNumberReplayer.h"

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace facebook::velox::tool::trace::test {
class TopNRowNumberReplayerTest : public HiveConnectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    HiveConnectorTestBase::SetUpTestCase();
    filesystems::registerLocalFileSystem();
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
    input_.clear();
    HiveConnectorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(int32_t count, int32_t rowsPerVector, const RowTypePtr& rowType) {
    return HiveConnectorTestBase::makeVectors(rowType, count, rowsPerVector);
  }

  std::vector<Split> makeSplits(
      const std::vector<RowVectorPtr>& inputs,
      const std::string& path) {
    std::vector<Split> splits;
    for (auto i = 0; i < 4; ++i) {
      const std::string filePath = fmt::format("{}/{}", path, i);
      writeToFile(filePath, inputs);
      splits.emplace_back(makeHiveConnectorSplit(filePath));
    }

    return splits;
  }

  struct PlanWithSplits {
    core::PlanNodePtr plan;
    core::PlanNodeId scanId;
    std::vector<exec::Split> splits;

    explicit PlanWithSplits(
        const core::PlanNodePtr& _plan,
        const core::PlanNodeId& _scanId = "",
        const std::vector<velox::exec::Split>& _splits = {})
        : plan(_plan), scanId(_scanId), splits(_splits) {}
  };

  PlanWithSplits createPlan(
      int32_t limit,
      bool generateRowNumber = true,
      const std::vector<std::string>& partitionKeys = {"c0"},
      const std::vector<std::string>& sortingKeys = {"c1"}) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId scanId;

    auto builder =
        PlanBuilder(planNodeIdGenerator)
            .tableScan(inputType_)
            .capturePlanNodeId(scanId)
            .topNRowNumber(
                partitionKeys, sortingKeys, limit, generateRowNumber);

    auto plan = builder.capturePlanNodeId(topNRowNumberId_).planNode();

    const std::vector<Split> splits =
        makeSplits(input_, fmt::format("{}/splits", testDir_->getPath()));
    return PlanWithSplits{plan, scanId, splits};
  }

  PlanWithSplits createRankPlan(
      std::string_view function,
      int32_t limit,
      bool generateRowNumber = true,
      const std::vector<std::string>& partitionKeys = {"c0"},
      const std::vector<std::string>& sortingKeys = {"c1"}) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId scanId;

    auto builder =
        PlanBuilder(planNodeIdGenerator)
            .tableScan(inputType_)
            .capturePlanNodeId(scanId)
            .topNRank(
                function, partitionKeys, sortingKeys, limit, generateRowNumber);

    auto plan = builder.capturePlanNodeId(topNRowNumberId_).planNode();

    const std::vector<Split> splits =
        makeSplits(input_, fmt::format("{}/splits", testDir_->getPath()));
    return PlanWithSplits{plan, scanId, splits};
  }

  core::PlanNodeId topNRowNumberId_;
  RowTypePtr inputType_{ROW({"c0", "c1"}, {BIGINT(), VARCHAR()})};

  std::vector<RowVectorPtr> input_ = makeVectors(5, 100, inputType_);

  const std::shared_ptr<TempDirectoryPath> testDir_ =
      TempDirectoryPath::create();
};

TEST_F(TopNRowNumberReplayerTest, basic) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row % 10; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("str_{}", 99 - row); })})};

  const auto planWithSplits = createPlan(5, true);
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result =
      builder.maxDrivers(1).splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "basic");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(5, true);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, topNRowNumberId_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = TopNRowNumberReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   topNRowNumberId_,
                                   "TopNRowNumber",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(TopNRowNumberReplayerTest, withoutRowNumber) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row % 10; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("data_{}", row); })})};

  const auto planWithSplits = createPlan(10, false);
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result =
      builder.maxDrivers(1).splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "withoutRowNumber");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(10, false);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, topNRowNumberId_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = TopNRowNumberReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   topNRowNumberId_,
                                   "TopNRowNumber",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(TopNRowNumberReplayerTest, multiplePartitions) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row % 5; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("value_{}", 99 - row); })})};

  const auto planWithSplits = createPlan(3, true);
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result =
      builder.maxDrivers(1).splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot = fmt::format(
      "{}/{}/traceRoot/", testDir_->getPath(), "multiplePartitions");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(3, true);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, topNRowNumberId_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = TopNRowNumberReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   topNRowNumberId_,
                                   "TopNRowNumber",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(TopNRowNumberReplayerTest, noPartitionKeys) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("str_{}", 99 - row); })})};

  const auto planWithSplits = createPlan(5, true, {}, {"c1"});
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result =
      builder.maxDrivers(1).splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "noPartitionKeys");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(5, true, {}, {"c1"});
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, topNRowNumberId_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = TopNRowNumberReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   topNRowNumberId_,
                                   "TopNRowNumber",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(TopNRowNumberReplayerTest, multipleSortingKeys) {
  inputType_ = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), VARCHAR()});
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row % 5; }),
       makeFlatVector<int64_t>(100, [](auto row) { return row / 10; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("str_{}", 99 - row); })})};

  const auto planWithSplits = createPlan(3, true, {"c0"}, {"c1 DESC", "c2"});
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result =
      builder.maxDrivers(1).splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot = fmt::format(
      "{}/{}/traceRoot/", testDir_->getPath(), "multipleSortingKeys");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(3, true, {"c0"}, {"c1 DESC", "c2"});
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, topNRowNumberId_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = TopNRowNumberReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   topNRowNumberId_,
                                   "TopNRowNumber",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(TopNRowNumberReplayerTest, limitOne) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row % 10; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("str_{}", 99 - row); })})};

  const auto planWithSplits = createPlan(1, true);
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result =
      builder.maxDrivers(1).splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "limitOne");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(1, true);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(1)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, topNRowNumberId_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = TopNRowNumberReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   topNRowNumberId_,
                                   "TopNRowNumber",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

} // namespace facebook::velox::tool::trace::test
