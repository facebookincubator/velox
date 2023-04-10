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
#include <velox/type/Timestamp.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Type.h"

namespace facebook::velox::exec::test {

class GroupedExecutionTest : public virtual HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
  }

  static void SetUpTestCase() {
    HiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return HiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  exec::Split makeHiveSplitWithGroup(std::string path, int32_t group) {
    return exec::Split(makeHiveConnectorSplit(std::move(path)), group);
  }

  exec::Split makeHiveSplit(std::string path) {
    return exec::Split(makeHiveConnectorSplit(std::move(path)));
  }

  static core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    return PlanBuilder().tableScan(outputType).planNode();
  }

  static std::unordered_set<int32_t> getCompletedSplitGroups(
      const std::shared_ptr<exec::Task>& task) {
    return task->taskStats().completedSplitGroups;
  }

  static void waitForFinishedDrivers(
      const std::shared_ptr<exec::Task>& task,
      uint32_t n) {
    // Limit wait to 10 seconds.
    size_t iteration{0};
    while (task->numFinishedDrivers() < n and iteration < 100) {
      /* sleep override */
      usleep(100'000); // 0.1 second.
      ++iteration;
    }
    ASSERT_EQ(n, task->numFinishedDrivers());
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           REAL(),
           DOUBLE(),
           VARCHAR(),
           TINYINT()})};
};

// Here we test the grouped execution sanity checks.
TEST_F(GroupedExecutionTest, groupedExecutionErrors) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId tableScanNodeId;
  core::PlanNodeId projectNodeId;
  core::PlanNodeId localPartitionNodeId;
  core::PlanNodeId tableScanNodeId2;
  auto planFragment =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin(
              {PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType_)
                   .capturePlanNodeId(tableScanNodeId)
                   .project({"c0", "c1", "c2", "c3", "c4", "c5"})
                   .capturePlanNodeId(projectNodeId)
                   .planNode(),
               PlanBuilder(planNodeIdGenerator)
                   .tableScan(rowType_)
                   .capturePlanNodeId(tableScanNodeId2)
                   .project({"c0", "c1", "c2", "c3", "c4", "c5"})
                   .planNode()})
          .capturePlanNodeId(localPartitionNodeId)
          .partitionedOutput({}, 1, {"c0", "c1", "c2", "c3", "c4", "c5"})
          .planFragment();

  std::shared_ptr<core::QueryCtx> queryCtx;
  std::shared_ptr<exec::Task> task;
  planFragment.numSplitGroups = 10;

  // Check ungrouped execution with supplied leaf node ids.
  planFragment.executionStrategy = core::ExecutionStrategy::kUngrouped;
  planFragment.groupedExecutionLeafNodeIds.clear();
  planFragment.groupedExecutionLeafNodeIds.emplace(tableScanNodeId);
  queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
  task =
      std::make_shared<exec::Task>("0", planFragment, 0, std::move(queryCtx));
  VELOX_ASSERT_THROW(
      task->start(task, 3, 1),
      "groupedExecutionLeafNodeIds must be empty in ungrouped execution mode");

  // Check grouped execution without supplied leaf node ids.
  planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  planFragment.groupedExecutionLeafNodeIds.clear();
  queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
  task =
      std::make_shared<exec::Task>("0", planFragment, 0, std::move(queryCtx));
  VELOX_ASSERT_THROW(
      task->start(task, 3, 1),
      "groupedExecutionLeafNodeIds must not be empty in "
      "grouped execution mode");

  // Check grouped execution with supplied non-leaf node id.
  planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  planFragment.groupedExecutionLeafNodeIds.clear();
  planFragment.groupedExecutionLeafNodeIds.emplace(projectNodeId);
  queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
  task =
      std::make_shared<exec::Task>("0", planFragment, 0, std::move(queryCtx));
  VELOX_ASSERT_THROW(
      task->start(task, 3, 1),
      fmt::format(
          "Grouped execution leaf node {} is not a leaf node in any pipeline",
          projectNodeId));

  // Check grouped execution with supplied leaf and non-leaf node ids.
  planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  planFragment.groupedExecutionLeafNodeIds.clear();
  planFragment.groupedExecutionLeafNodeIds.emplace(tableScanNodeId);
  planFragment.groupedExecutionLeafNodeIds.emplace(projectNodeId);
  queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
  task =
      std::make_shared<exec::Task>("0", planFragment, 0, std::move(queryCtx));
  VELOX_ASSERT_THROW(
      task->start(task, 3, 1),
      fmt::format(
          "Grouped execution leaf node {} is not a leaf node in any pipeline",
          projectNodeId));

  // Check grouped execution with supplied leaf node id for a non-source
  // pipeline.
  planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  planFragment.groupedExecutionLeafNodeIds.clear();
  planFragment.groupedExecutionLeafNodeIds.emplace(localPartitionNodeId);
  queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
  task =
      std::make_shared<exec::Task>("0", planFragment, 0, std::move(queryCtx));
  VELOX_ASSERT_THROW(
      task->start(task, 3, 1),
      fmt::format(
          "Grouped execution leaf node {} not found or it is not a leaf node",
          localPartitionNodeId));
}

// Here we test various aspects of grouped/bucketed execution involving
// output buffer and 3 pipelines.
TEST_F(GroupedExecutionTest, groupedExecutionWithOutputBuffer) {
  // Create source file - we will read from it in 6 splits.
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->path, vectors);

  // A chain of three pipelines separated by local exchange with the leaf one
  // having scan running grouped execution - this will make all three pipelines
  // running grouped execution.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId tableScanNodeId;
  auto pipe0Node =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(rowType_)
          .capturePlanNodeId(tableScanNodeId)
          .project({"c3 as x", "c2 as y", "c1 as z", "c0 as w", "c4", "c5"})
          .planNode();
  auto pipe1Node =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin({pipe0Node})
          .project({"w as c0", "z as c1", "y as c2", "x as c3", "c4", "c5"})
          .planNode();
  auto planFragment =
      PlanBuilder(planNodeIdGenerator)
          .localPartitionRoundRobin({pipe1Node})
          .partitionedOutput({}, 1, {"c0", "c1", "c2", "c3", "c4", "c5"})
          .planFragment();
  planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  planFragment.groupedExecutionLeafNodeIds.emplace(tableScanNodeId);
  planFragment.numSplitGroups = 10;
  auto queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
  auto task = std::make_shared<exec::Task>(
      "0", std::move(planFragment), 0, std::move(queryCtx));
  // 3 drivers max and 1 concurrent split group.
  task->start(task, 3, 1);

  // All pipelines run grouped execution, so no drivers should be running.
  EXPECT_EQ(0, task->numRunningDrivers());

  // Add one split for group (8).
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 8));

  // Only one split group should be in the processing mode, so 9 drivers (3 per
  // pipeline).
  EXPECT_EQ(9, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

  // Add the rest of splits
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 1));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 5));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 8));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 5));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 8));

  // One split group should be in the processing mode, so 9 drivers.
  EXPECT_EQ(9, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

  // Finalize one split group (8) and wait until 3 drivers are finished.
  task->noMoreSplitsForGroup("0", 8);
  waitForFinishedDrivers(task, 9);
  // As one split group is finished, another one should kick in, so 3 drivers.
  EXPECT_EQ(9, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>({8}), getCompletedSplitGroups(task));

  // Finalize the second split group (1) and wait until 18 drivers are finished.
  task->noMoreSplitsForGroup("0", 1);
  waitForFinishedDrivers(task, 18);

  // As one split group is finished, another one should kick in, so 3 drivers.
  EXPECT_EQ(9, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>({1, 8}), getCompletedSplitGroups(task));

  // Finalize the third split group (5) and wait until 27 drivers are finished.
  task->noMoreSplitsForGroup("0", 5);
  waitForFinishedDrivers(task, 27);

  // No split groups should be processed at the moment, so 0 drivers.
  EXPECT_EQ(0, task->numRunningDrivers());
  EXPECT_EQ(
      std::unordered_set<int32_t>({1, 5, 8}), getCompletedSplitGroups(task));

  // Flag that we would have no more split groups.
  task->noMoreSplits("0");

  // 'Delete results' from output buffer triggers 'set all output consumed',
  // which should finish the task.
  auto outputBufferManager =
      exec::PartitionedOutputBufferManager::getInstance().lock();
  outputBufferManager->deleteResults(task->taskId(), 0);

  // Task must be finished at this stage.
  EXPECT_EQ(exec::TaskState::kFinished, task->state());
  EXPECT_EQ(
      std::unordered_set<int32_t>({1, 5, 8}), getCompletedSplitGroups(task));
}

// Here we test various aspects of grouped/bucketed execution involving
// output buffer and 3 pipelines.
TEST_F(GroupedExecutionTest, groupedExecutionWithHashAndCrossJoin) {
  // Create source file - we will read from it in 6 splits.
  auto vectors = makeVectors(4, 20);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->path, vectors);

  // Run the test twice - for Hash and Cross Join.
  for (size_t i = 0; i < 2; ++i) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId probeScanNodeId;
    core::PlanNodeId buildScanNodeId;
    core::PlanNodePtr pipe0Node;
    core::PlanNodePtr pipe1Node;

    // Hash or Cross join.
    if (i == 0) {
      pipe0Node =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(rowType_)
              .capturePlanNodeId(probeScanNodeId)
              .project({"c3 as x", "c2 as y", "c1 as z", "c0 as w", "c4", "c5"})
              .hashJoin(
                  {"w"},
                  {"r"},
                  PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType_, {"c0 > 0"})
                      .capturePlanNodeId(buildScanNodeId)
                      .project({"c0 as r"})
                      .planNode(),
                  "",
                  {"x", "y", "z", "w", "c4", "c5"})
              .planNode();
      pipe1Node =
          PlanBuilder(planNodeIdGenerator)
              .localPartitionRoundRobin({pipe0Node})
              .project({"w as c0", "z as c1", "y as c2", "x as c3", "c4", "c5"})
              .planNode();
    } else {
      pipe0Node =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(rowType_)
              .capturePlanNodeId(probeScanNodeId)
              .project({"c3 as x", "c2 as y", "c1 as z", "c0 as w", "c4", "c5"})
              .crossJoin(
                  PlanBuilder(planNodeIdGenerator)
                      .tableScan(rowType_, {"c0 > 0"})
                      .capturePlanNodeId(buildScanNodeId)
                      .project({"c0 as r"})
                      .planNode(),
                  {"x", "y", "z", "r", "c4", "c5"})
              .planNode();
      pipe1Node =
          PlanBuilder(planNodeIdGenerator)
              .localPartitionRoundRobin({pipe0Node})
              .project({"r as c0", "z as c1", "y as c2", "x as c3", "c4", "c5"})
              .planNode();
    }
    auto planFragment =
        PlanBuilder(planNodeIdGenerator)
            .localPartitionRoundRobin({pipe1Node})
            .partitionedOutput({}, 1, {"c0", "c1", "c2", "c3", "c4", "c5"})
            .planFragment();

    planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
    planFragment.groupedExecutionLeafNodeIds.emplace(probeScanNodeId);
    planFragment.numSplitGroups = 10;
    auto queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
    auto task = std::make_shared<exec::Task>(
        "0", std::move(planFragment), 0, std::move(queryCtx));
    // 3 drivers max and 1 concurrent split group.
    task->start(task, 3, 1);

    // Build pipeline runs ungrouped execution, so it should have drivers
    // running.
    EXPECT_EQ(3, task->numRunningDrivers());

    // Add single split to the build scan.
    task->addSplit(buildScanNodeId, makeHiveSplit(filePath->path));

    // Add one split for group (8).
    task->addSplit(probeScanNodeId, makeHiveSplitWithGroup(filePath->path, 8));

    // Only one split group should be in the processing mode, so 9 drivers (3
    // per pipeline) grouped + 3 ungrouped.
    EXPECT_EQ(3 + 9, task->numRunningDrivers());
    EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

    // Add the rest of splits
    task->addSplit(probeScanNodeId, makeHiveSplitWithGroup(filePath->path, 1));
    task->addSplit(probeScanNodeId, makeHiveSplitWithGroup(filePath->path, 5));
    task->addSplit(probeScanNodeId, makeHiveSplitWithGroup(filePath->path, 8));
    task->addSplit(probeScanNodeId, makeHiveSplitWithGroup(filePath->path, 5));
    task->addSplit(probeScanNodeId, makeHiveSplitWithGroup(filePath->path, 8));

    // One split group should be in the processing mode, so 9 drivers (3 per
    // pipeline) grouped + 3 ungrouped.
    EXPECT_EQ(3 + 9, task->numRunningDrivers());
    EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

    // Finalize the build splits.
    task->noMoreSplits(buildScanNodeId);
    // Wait till the build is finished and check drivers and splits again.
    waitForFinishedDrivers(task, 3);
    // One split group should be in the processing mode, so 9 drivers.
    EXPECT_EQ(9, task->numRunningDrivers());
    EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

    // Finalize one split group (8) and wait until 3 drivers are finished.
    task->noMoreSplitsForGroup(probeScanNodeId, 8);
    waitForFinishedDrivers(task, 3 + 9);
    // As one split group is finished, another one should kick in, so 9 drivers.
    EXPECT_EQ(9, task->numRunningDrivers());
    EXPECT_EQ(std::unordered_set<int32_t>({8}), getCompletedSplitGroups(task));

    // Finalize the second split group (1) and wait until 18 drivers are
    // finished.
    task->noMoreSplitsForGroup(probeScanNodeId, 1);
    waitForFinishedDrivers(task, 3 + 18);

    // As one split group is finished, another one should kick in, so 9 drivers.
    EXPECT_EQ(9, task->numRunningDrivers());
    EXPECT_EQ(
        std::unordered_set<int32_t>({1, 8}), getCompletedSplitGroups(task));

    // Finalize the third split group (5) and wait until 27 drivers are
    // finished.
    task->noMoreSplitsForGroup(probeScanNodeId, 5);
    waitForFinishedDrivers(task, 3 + 27);

    // No split groups should be processed at the moment, so 0 drivers.
    EXPECT_EQ(0, task->numRunningDrivers());
    EXPECT_EQ(
        std::unordered_set<int32_t>({1, 5, 8}), getCompletedSplitGroups(task));

    // Flag that we would have no more split groups.
    task->noMoreSplits(probeScanNodeId);

    // 'Delete results' from output buffer triggers 'set all output consumed',
    // which should finish the task.
    auto outputBufferManager =
        exec::PartitionedOutputBufferManager::getInstance().lock();
    outputBufferManager->deleteResults(task->taskId(), 0);

    // Task must be finished at this stage.
    EXPECT_EQ(exec::TaskState::kFinished, task->state());
    EXPECT_EQ(
        std::unordered_set<int32_t>({1, 5, 8}), getCompletedSplitGroups(task));
  }
}

// Here we test various aspects of grouped/bucketed execution.
TEST_F(GroupedExecutionTest, groupedExecution) {
  // Create source file - we will read from it in 6 splits.
  const size_t numSplits{6};
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->path, vectors);

  CursorParameters params;
  params.planNode = tableScanNode(ROW({}, {}));
  params.maxDrivers = 2;
  // We will have 10 split groups 'in total', but our task will only handle
  // three of them: 1, 5 and 8.
  // Split 0 is from split group 1.
  // Splits 1 and 2 are from split group 5.
  // Splits 3, 4 and 5 are from split group 8.
  params.executionStrategy = core::ExecutionStrategy::kGrouped;
  params.groupedExecutionLeafNodeIds.emplace(params.planNode->id());
  params.numSplitGroups = 3;
  params.numConcurrentSplitGroups = 2;

  // Create the cursor with the task underneath. It is not started yet.
  auto cursor = std::make_unique<TaskCursor>(params);
  auto task = cursor->task();

  // Add one splits before start to ensure we can handle such cases.
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 8));

  // Start task now.
  cursor->start();

  // Only one split group should be in the processing mode, so 2 drivers.
  EXPECT_EQ(2, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

  // Add the rest of splits
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 1));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 5));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 8));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 5));
  task->addSplit("0", makeHiveSplitWithGroup(filePath->path, 8));

  // Only two split groups should be in the processing mode, so 4 drivers.
  EXPECT_EQ(4, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>{}, getCompletedSplitGroups(task));

  // Finalize one split group (8) and wait until 2 drivers are finished.
  task->noMoreSplitsForGroup("0", 8);
  waitForFinishedDrivers(task, 2);

  // As one split group is finished, another one should kick in, so 4 drivers.
  EXPECT_EQ(4, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>({8}), getCompletedSplitGroups(task));

  // Finalize the second split group (5) and wait until 4 drivers are finished.
  task->noMoreSplitsForGroup("0", 5);
  waitForFinishedDrivers(task, 4);

  // As the second split group is finished, only one is left, so 2 drivers.
  EXPECT_EQ(2, task->numRunningDrivers());
  EXPECT_EQ(std::unordered_set<int32_t>({5, 8}), getCompletedSplitGroups(task));

  // Finalize the third split group (1) and wait until 6 drivers are finished.
  task->noMoreSplitsForGroup("0", 1);
  waitForFinishedDrivers(task, 6);

  // No split groups should be processed at the moment, so 0 drivers.
  EXPECT_EQ(0, task->numRunningDrivers());
  EXPECT_EQ(
      std::unordered_set<int32_t>({1, 5, 8}), getCompletedSplitGroups(task));

  // Make sure split groups with no splits are reported as complete.
  task->noMoreSplitsForGroup("0", 3);
  EXPECT_EQ(
      std::unordered_set<int32_t>({1, 3, 5, 8}), getCompletedSplitGroups(task));

  // Flag that we would have no more split groups.
  task->noMoreSplits("0");

  // Make sure we've got the right number of rows.
  int32_t numRead = 0;
  while (cursor->moveNext()) {
    auto vector = cursor->current();
    EXPECT_EQ(vector->childrenSize(), 0);
    numRead += vector->size();
  }

  // Task must be finished at this stage.
  EXPECT_EQ(exec::TaskState::kFinished, task->state());
  EXPECT_EQ(
      std::unordered_set<int32_t>({1, 3, 5, 8}), getCompletedSplitGroups(task));
  EXPECT_EQ(numRead, numSplits * 10'000);
}

} // namespace facebook::velox::exec::test
