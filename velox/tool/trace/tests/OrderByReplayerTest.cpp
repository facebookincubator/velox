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

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/trace/TraceUtil.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/tool/trace/OrderByReplayer.h"

using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace facebook::velox::tool::trace::test {
class OrderByReplayerTest : public HiveConnectorTestBase {
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

  PlanWithSplits createPlan(bool isPartial = false) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId scanId;

    auto builder = PlanBuilder(planNodeIdGenerator)
                       .tableScan(inputType_)
                       .capturePlanNodeId(scanId)
                       .orderBy({"c0"}, isPartial);

    auto plan = builder.capturePlanNodeId(orderById_).planNode();

    const std::vector<Split> splits =
        makeSplits(input_, fmt::format("{}/splits", testDir_->getPath()));
    return PlanWithSplits{plan, scanId, splits};
  }

  core::PlanNodeId orderById_;
  RowTypePtr inputType_{ROW({"c0", "c1"}, {BIGINT(), VARCHAR()})};

  std::vector<RowVectorPtr> input_ = makeVectors(5, 100, inputType_);

  const std::shared_ptr<TempDirectoryPath> testDir_ =
      TempDirectoryPath::create();
};

TEST_F(OrderByReplayerTest, basic) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return 99 - row; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("str_{}", row); })})};

  const auto planWithSplits = createPlan();
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result = builder.splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "basic");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan();
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(4)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, orderById_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = OrderByReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   orderById_,
                                   "OrderBy",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

TEST_F(OrderByReplayerTest, partial) {
  input_ = {makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row % 10; }),
       makeFlatVector<std::string>(
           100, [](auto row) { return fmt::format("data_{}", row); })})};

  const auto planWithSplits = createPlan(true);
  AssertQueryBuilder builder(planWithSplits.plan);
  const auto result = builder.splits(planWithSplits.splits).copyResults(pool());

  const auto traceRoot =
      fmt::format("{}/{}/traceRoot/", testDir_->getPath(), "partial");
  std::shared_ptr<Task> task;
  auto tracePlanWithSplits = createPlan(true);
  AssertQueryBuilder traceBuilder(tracePlanWithSplits.plan);
  traceBuilder.maxDrivers(4)
      .config(core::QueryConfig::kQueryTraceEnabled, true)
      .config(core::QueryConfig::kQueryTraceDir, traceRoot)
      .config(core::QueryConfig::kQueryTraceMaxBytes, 100UL << 30)
      .config(core::QueryConfig::kQueryTraceTaskRegExp, ".*")
      .config(core::QueryConfig::kQueryTraceNodeId, orderById_);
  auto traceResult =
      traceBuilder.splits(tracePlanWithSplits.splits).copyResults(pool(), task);

  assertEqualResults({result}, {traceResult});

  const auto replayingResult = OrderByReplayer(
                                   traceRoot,
                                   task->queryCtx()->queryId(),
                                   task->taskId(),
                                   orderById_,
                                   "OrderBy",
                                   "",
                                   "",
                                   0,
                                   executor_.get())
                                   .run();
  assertEqualResults({result}, {replayingResult});
}

} // namespace facebook::velox::tool::trace::test
