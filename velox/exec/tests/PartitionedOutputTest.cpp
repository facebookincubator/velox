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
#include "velox/exec/PartitionedOutput.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

class PartitionedOutputTest
    : public OperatorTestBase,
      public testing::WithParamInterface<VectorSerde::Kind> {
 public:
  static std::vector<VectorSerde::Kind> getTestParams() {
    const std::vector<VectorSerde::Kind> kinds(
        {VectorSerde::Kind::kPresto,
         VectorSerde::Kind::kCompactRow,
         VectorSerde::Kind::kUnsafeRow});
    return kinds;
  }

 protected:
  std::shared_ptr<core::QueryCtx> createQueryContext(
      std::unordered_map<std::string, std::string> config) {
    return core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(config)));
  }

  std::vector<std::unique_ptr<folly::IOBuf>>
  getData(const std::string& taskId, int destination, int64_t sequence) {
    auto [promise, semiFuture] = folly::makePromiseContract<
        std::vector<std::unique_ptr<folly::IOBuf>>>();
    VELOX_CHECK(bufferManager_->getData(
        taskId,
        destination,
        PartitionedOutput::kMinDestinationSize,
        sequence,
        [result = std::make_shared<
             folly::Promise<std::vector<std::unique_ptr<folly::IOBuf>>>>(
             std::move(promise))](
            std::vector<std::unique_ptr<folly::IOBuf>> pages,
            int64_t /*inSequence*/,
            std::vector<int64_t> /*remainingBytes*/) {
          result->setValue(std::move(pages));
        }));
    auto future = std::move(semiFuture).via(executor_.get());
    future.wait(std::chrono::seconds{10});
    VELOX_CHECK(future.isReady());
    return std::move(future).value();
  }

  std::vector<std::unique_ptr<folly::IOBuf>> getAllData(
      const std::string& taskId,
      int destination) {
    std::vector<std::unique_ptr<folly::IOBuf>> result;
    int attempts = 0;
    bool done = false;
    while (!done) {
      attempts++;
      VELOX_CHECK_LT(attempts, 100);
      std::vector<std::unique_ptr<folly::IOBuf>> pages =
          getData(taskId, destination, result.size());
      for (auto& page : pages) {
        if (page) {
          result.push_back(std::move(page));
        } else {
          bufferManager_->deleteResults(taskId, destination);
          done = true;
          break;
        }
      }
    }
    return result;
  }

 private:
  const std::shared_ptr<OutputBufferManager> bufferManager_{
      OutputBufferManager::getInstanceRef()};
};

TEST_P(PartitionedOutputTest, flush) {
  // This test verifies
  //  - Flush thresholds are respected (flush doesn't happen neither too early
  //  nor too late)
  //  - Flush is done independently for each output partition (flush for one
  //  partition doesn't trigger flush for another one)
  auto input = makeRowVector(
      {"p1", "v1"},
      {makeFlatVector<int32_t>({0, 1}),
       makeFlatVector<std::string>({
           // twice as large to make sure it is always flushed (even if
           // PartitionedOutput#setTargetSizePct rolls 120%)
           std::string(PartitionedOutput::kMinDestinationSize * 2, '0'),
           // 10 times smaller, so the data from 13 pages is always flushed as
           // 2
           // pages
           // 130% > 120% (when PartitionedOutput#setTargetSizePct rolls 120%)
           // 130% < 140% (when PartitionedOutput#setTargetSizePct rolls 70%
           // two
           // times in a row)
           std::string(PartitionedOutput::kMinDestinationSize / 10, '1'),
       })});

  core::PlanNodeId partitionNodeId;
  auto plan = PlanBuilder()
                  // produce 13 pages
                  .values({input}, false, 13)
                  .partitionedOutput(
                      {"p1"}, 2, std::vector<std::string>{"v1"}, GetParam())
                  .capturePlanNodeId(partitionNodeId)
                  .planNode();

  auto taskId = "local://test-partitioned-output-flush-0";
  auto task = Task::create(
      taskId,
      core::PlanFragment{plan},
      0,
      createQueryContext(
          {{core::QueryConfig::kMaxPartitionedOutputBufferSize,
            std::to_string(PartitionedOutput::kMinDestinationSize * 2)}}),
      Task::ExecutionMode::kParallel);
  task->start(1);

  const auto partition0 = getAllData(taskId, 0);
  const auto partition1 = getAllData(taskId, 1);

  const auto taskWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::seconds{10})
                              .count();
  auto future = task->taskCompletionFuture()
                    .within(std::chrono::microseconds(taskWaitUs))
                    .via(executor_.get());
  future.wait();

  ASSERT_TRUE(waitForTaskDriversToFinish(task.get(), taskWaitUs));

  // Since each row for partition 0 is over the flush threshold as
  // many pages as there are input pages are expected
  ASSERT_EQ(partition0.size(), 13);
  // Data for the second partition is much smaller and expected to be buffered
  // up to a defined threshold
  ASSERT_EQ(partition1.size(), 2);

  auto planStats = toPlanStats(task->taskStats());
  const auto serdeKindRuntimsStats =
      planStats.at(partitionNodeId)
          .customStats.at(std::string(Operator::kShuffleSerdeKind));
  ASSERT_EQ(serdeKindRuntimsStats.count, 1);
  ASSERT_EQ(serdeKindRuntimsStats.min, static_cast<int64_t>(GetParam()));
  ASSERT_EQ(serdeKindRuntimsStats.max, static_cast<int64_t>(GetParam()));
}

TEST_P(PartitionedOutputTest, keyChannelNotAtBeginningWithNulls) {
  // This test verifies that PartitionedOutput can handle the case where a key
  // channel is not at the beginning of the input type when nulls are present
  // in the key channel.  This triggers collectNullRows() to run which has
  // special handling logic for the key channels.

  auto input = makeRowVector(
      // The key column p1 is the second column.
      {"v1", "p1"},
      {makeFlatVector<std::string>({"0", "1", "2", "3"}),
       // Add nulls to the key column.
       makeNullableFlatVector<int32_t>(std::vector<std::optional<int32_t>>{
           0, std::nullopt, 1, std::nullopt})});

  auto plan =
      PlanBuilder()
          .values({input}, false, 13)
          // Set replicateNullsAndAny to true so we trigger the null path.
          .partitionedOutput(
              {"p1"}, 2, true, std::vector<std::string>{"v1"}, GetParam())
          .planNode();

  auto taskId = "local://test-partitioned-output-0";
  auto task = Task::create(
      taskId,
      core::PlanFragment{plan},
      0,
      createQueryContext({}),
      Task::ExecutionMode::kParallel);
  task->start(1);

  const auto partition0 = getAllData(taskId, 0);
  const auto partition1 = getAllData(taskId, 1);

  ASSERT_TRUE(waitForTaskCompletion(
      task.get(),
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::seconds(10))
          .count()));
}

// This test verifies that the Destination properly handles multiple
// flush-then-append cycles. After flush(), the VectorStreamGroup must be
// properly reset so that subsequent advance() calls create a fresh serializer
// with proper initialization via createStreamTree(). This test exercises
// the fix for T254261397 where crashes occurred due to improper state after
// flush when current_->clear() was called instead of current_.reset().
TEST_P(PartitionedOutputTest, multipleFlushCycles) {
  // Create input data where each row is large enough to trigger a flush
  // (exceeds kMinDestinationSize), but we have many batches to ensure
  // multiple flush-then-advance cycles occur for the same destination.
  const auto largeString =
      std::string(PartitionedOutput::kMinDestinationSize * 2, 'x');

  auto input = makeRowVector(
      {"p1", "v1"},
      {// All rows go to partition 0 to ensure multiple flushes on same dest.
       makeFlatVector<int32_t>({0, 0, 0, 0}),
       makeFlatVector<std::string>(
           {largeString, largeString, largeString, largeString})});

  core::PlanNodeId partitionNodeId;
  // Use 20 batches to ensure many flush cycles (each row triggers a flush).
  auto plan = PlanBuilder()
                  .values({input}, false, 20)
                  .partitionedOutput(
                      {"p1"}, 2, std::vector<std::string>{"v1"}, GetParam())
                  .capturePlanNodeId(partitionNodeId)
                  .planNode();

  auto taskId = "local://test-partitioned-output-multiple-flush-cycles-0";
  auto task = Task::create(
      taskId,
      core::PlanFragment{plan},
      0,
      createQueryContext(
          {{core::QueryConfig::kMaxPartitionedOutputBufferSize,
            std::to_string(PartitionedOutput::kMinDestinationSize * 2)}}),
      Task::ExecutionMode::kParallel);
  task->start(1);

  const auto partition0 = getAllData(taskId, 0);
  const auto partition1 = getAllData(taskId, 1);

  const auto taskWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::seconds{10})
                              .count();
  auto future = task->taskCompletionFuture()
                    .within(std::chrono::microseconds(taskWaitUs))
                    .via(executor_.get());
  future.wait();

  ASSERT_TRUE(waitForTaskDriversToFinish(task.get(), taskWaitUs));

  // With 20 batches * 4 rows per batch = 80 rows going to partition 0.
  // Each row exceeds the flush threshold, so we expect many pages (~80).
  // The exact count may vary due to targetSizePct randomization, but we
  // should have at least 40 pages (assuming some batching).
  ASSERT_GE(partition0.size(), 40);

  // Partition 1 should have no data (or just the final flush marker).
  ASSERT_LE(partition1.size(), 1);

  auto planStats = toPlanStats(task->taskStats());
  const auto serdeKindRuntimsStats =
      planStats.at(partitionNodeId)
          .customStats.at(std::string(Operator::kShuffleSerdeKind));
  ASSERT_EQ(serdeKindRuntimsStats.count, 1);
  ASSERT_EQ(serdeKindRuntimsStats.min, static_cast<int64_t>(GetParam()));
  ASSERT_EQ(serdeKindRuntimsStats.max, static_cast<int64_t>(GetParam()));
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    PartitionedOutputTest,
    PartitionedOutputTest,
    testing::ValuesIn(PartitionedOutputTest::getTestParams()));
} // namespace facebook::velox::exec::test
