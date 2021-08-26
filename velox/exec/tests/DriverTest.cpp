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
#include <folly/init/Init.h>
#include "velox/dwio/dwrf/test/utils/BatchMaker.h"
#include "velox/exec/tests/Cursor.h"
#include "velox/exec/tests/OperatorTestBase.h"
#include "velox/exec/tests/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

using facebook::velox::test::BatchMaker;

// A PlanNode that passes its input to its output and makes variable
// memory reservations.
class TestingConsumerNode : public core::PlanNode {
 public:
  explicit TestingConsumerNode(std::shared_ptr<const core::PlanNode> input)
      : PlanNode("consumer"), sources_{input} {}

  const std::shared_ptr<const RowType>& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "consumer";
  }

 private:
  std::vector<std::shared_ptr<const core::PlanNode>> sources_;
};

// A PlanNode that passes its input to its output and periodically
// pauses and resumes other Tasks.
class TestingPauserNode : public core::PlanNode {
 public:
  explicit TestingPauserNode(std::shared_ptr<const core::PlanNode> input)
      : PlanNode("Pauser"), sources_{input} {}

  const std::shared_ptr<const RowType>& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "Pauser";
  }

 private:
  std::vector<std::shared_ptr<const core::PlanNode>> sources_;
};

class DriverTest : public OperatorTestBase {
 protected:
  enum class ResultOperation {
    kRead,
    kReadSlow,
    kDrop,
    kCancel,
    kTerminate,
    kPause,
    kYield
  };

  void SetUp() override {
    OperatorTestBase::SetUp();
    rowType_ =
        ROW({"key", "m1", "m2", "m3", "m4", "m5", "m6", "m7"},
            {BIGINT(),
             BIGINT(),
             BIGINT(),
             BIGINT(),
             BIGINT(),
             BIGINT(),
             BIGINT(),
             BIGINT()});
  }

  void TearDown() override {
    if (wakeupInitialized_) {
      wakeupCancelled_ = true;
      wakeupThread_.join();
    }
    OperatorTestBase::TearDown();
  }

  std::shared_ptr<const core::PlanNode> makeValuesFilterProject(
      const std::shared_ptr<const RowType>& rowType,
      const std::string& filter,
      const std::string& project,
      int32_t numBatches,
      int32_t rowsInBatch,
      // applies to second column
      std::function<bool(int64_t)> filterFunc = nullptr,
      int32_t* filterHits = nullptr,
      bool addTestingPauser = false,
      bool addTestingConsumer = false) {
    std::vector<RowVectorPtr> batches;
    for (int32_t i = 0; i < numBatches; ++i) {
      batches.push_back(std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(rowType, rowsInBatch, *pool_)));
    }
    if (filterFunc) {
      int32_t hits = 0;
      for (auto& batch : batches) {
        auto child = batch->childAt(1)->as<FlatVector<int64_t>>();
        for (vector_size_t i = 0; i < child->size(); ++i) {
          if (!child->isNullAt(i) && filterFunc(child->valueAt(i))) {
            hits++;
          }
        }
      }
      *filterHits = hits;
    }

    PlanBuilder planBuilder;
    planBuilder.values(batches, true).planNode();

    if (!filter.empty()) {
      planBuilder.filter(filter);
    }

    if (!project.empty()) {
      auto projectNames = rowType->names();
      auto expressions = projectNames;
      projectNames.push_back("expr");
      expressions.push_back(project);

      planBuilder.project(expressions, projectNames);
    }
    if (addTestingConsumer) {
      planBuilder.addNode([](std::shared_ptr<const core::PlanNode> input) {
        return std::make_shared<TestingConsumerNode>(input);
      });
    }
    if (addTestingPauser) {
      planBuilder.addNode([](std::shared_ptr<const core::PlanNode> input) {
        return std::make_shared<TestingPauserNode>(input);
      });
    }

    return planBuilder.planNode();
  }

  // Opens a cursor and reads data. Takes action 'operation' every 'numRows'
  // rows of data. Increments the 'counter' for each successfully read row.
  void readResults(
      CursorParameters& params,
      ResultOperation operation,
      int32_t numRows,
      int32_t* counter,
      int32_t threadId = 0) {
    auto cursor = std::make_unique<RowCursor>(params);
    {
      std::lock_guard<std::mutex> l(mutex_);
      tasks_.push_back(cursor->task());
      // To be realized either after 1s wall time or when the corresponding Task
      // is no longer running.
      auto& executor = folly::QueuedImmediateExecutor::instance();
      auto future = tasks_.back()->stateChangeFuture(1'000'000).via(&executor);
      stateFutures_.emplace(threadId, std::move(future));

      EXPECT_FALSE(stateFutures_.at(threadId).isReady());
    }
    bool paused = false;
    for (;;) {
      if (operation == ResultOperation::kPause && paused) {
        if (!cursor->hasNext()) {
          paused = false;
          cursor->cancelPool()->requestPause(false);
          Task::resume(cursor->task());
        }
      }
      if (!cursor->next()) {
        break;
      }
      ++*counter;
      if (*counter % numRows == 0) {
        if (operation == ResultOperation::kDrop) {
          return;
        }
        if (operation == ResultOperation::kReadSlow) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          // If this is an EXPECT this is flaky when running on a
          // noisy test cloud.
          LOG(INFO) << "Task::toString() while probably blocked: "
                    << tasks_[0]->toString();
        } else if (operation == ResultOperation::kCancel) {
          cursor->cancelPool()->requestTerminate();
        } else if (operation == ResultOperation::kTerminate) {
          cursor->task()->terminate(kAborted);
        } else if (operation == ResultOperation::kYield) {
          cursor->cancelPool()->requestYield();
        } else if (operation == ResultOperation::kPause) {
          cursor->cancelPool()->requestPause(true);
          auto& executor = folly::QueuedImmediateExecutor::instance();
          auto future = cursor->cancelPool()->finishFuture().via(&executor);
          future.wait();
          paused = true;
        }
      }
    }
  }

 public:
  // Sets 'future' to a future that will be realized within a random
  // delay of a few ms.
  void registerForWakeup(ContinueFuture* future) {
    std::lock_guard<std::mutex> l(wakeupMutex_);
    if (!wakeupInitialized_) {
      wakeupInitialized_ = true;
      wakeupThread_ = std::thread([&]() {
        int32_t counter = 0;
        for (;;) {
          if (wakeupCancelled_) {
            return;
          }
          // Wait a small interval and realize a small number of queued
          // promises, if any.
          auto units = 1 + (++counter % 5);
          std::this_thread::sleep_for(std::chrono::milliseconds(units));
          auto count = 1 + (++counter % 4);
          for (auto i = 0; i < count; ++i) {
            if (wakeupPromises_.empty()) {
              break;
            }
            wakeupPromises_.front().setValue(true);
            wakeupPromises_.pop_front();
          }
        }
      });
    }
    auto [promise, semiFuture] = makeVeloxPromiseContract<bool>("wakeup");
    *future = std::move(semiFuture);
    wakeupPromises_.push_back(std::move(promise));
  }

  // Registers a Task for use in randomTask().
  void registerTask(std::shared_ptr<Task> task) {
    std::lock_guard<std::mutex> l(taskMutex_);
    if (std::find(allTasks_.begin(), allTasks_.end(), task) !=
        allTasks_.end()) {
      return;
    }
    allTasks_.push_back(task);
  }

  void unregisterTask(std::shared_ptr<Task> task) {
    std::lock_guard<std::mutex> l(taskMutex_);
    auto it = std::find(allTasks_.begin(), allTasks_.end(), task);
    if (it == allTasks_.end()) {
      return;
    }
    allTasks_.erase(it);
  }

  std::shared_ptr<Task> randomTask() {
    std::lock_guard<std::mutex> l(taskMutex_);
    if (allTasks_.empty()) {
      return nullptr;
    }
    return allTasks_[folly::Random::rand32() % allTasks_.size()];
  }

 protected:
  // State for registerForWakeup().
  std::mutex wakeupMutex_;
  std::thread wakeupThread_;
  std::deque<folly::Promise<bool>> wakeupPromises_;
  bool wakeupInitialized_{false};
  // Set to true when it is time to exit 'wakeupThread_'.
  bool wakeupCancelled_{false};

  std::shared_ptr<const RowType> rowType_;
  std::mutex mutex_;
  std::vector<std::shared_ptr<Task>> tasks_;
  std::unordered_map<int32_t, folly::Future<bool>> stateFutures_;

  // Mutex for randomTask()
  std::mutex taskMutex_;
  // Tasks registered for randomTask()
  std::vector<std::shared_ptr<Task>> allTasks_;

  folly::Random::DefaultGenerator rng_;
};

TEST_F(DriverTest, error) {
  Driver::testingJoinAndReinitializeExecutor(10);
  CursorParameters params;
  params.planNode = makeValuesFilterProject(rowType_, "m1 % 0", "", 100, 10);
  params.numThreads = 20;
  int32_t numRead = 0;
  try {
    readResults(params, ResultOperation::kRead, 1'000'000, &numRead);
    EXPECT_TRUE(false) << "Expected exception";
  } catch (const VeloxException& e) {
    EXPECT_NE(e.message().find("Cannot divide by 0"), std::string::npos);
  }
  EXPECT_EQ(numRead, 0);
  EXPECT_TRUE(stateFutures_.at(0).isReady());
  // Realized immediately since task not running.
  EXPECT_TRUE(tasks_[0]->stateChangeFuture(1'000'000).isReady());
  EXPECT_EQ(tasks_[0]->state(), kFailed);
}

TEST_F(DriverTest, cancel) {
  CursorParameters params;
  params.queryCtx = core::QueryCtx::create();

  params.planNode = makeValuesFilterProject(
      rowType_,
      "m1 % 10 > 0",
      "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
      100,
      100'000);
  params.numThreads = 10;
  int32_t numRead = 0;
  try {
    readResults(params, ResultOperation::kCancel, 1'000'000, &numRead);
    EXPECT_TRUE(false) << "Expected exception";
  } catch (const VeloxRuntimeError& e) {
    EXPECT_EQ("Cancelled", e.message());
  }
  EXPECT_GE(numRead, 1'000'000);
  auto& executor = folly::QueuedImmediateExecutor::instance();
  auto future = tasks_[0]->cancelPool()->finishFuture().via(&executor);
  future.wait();
  EXPECT_TRUE(stateFutures_.at(0).isReady());
  EXPECT_EQ(tasks_[0]->numDrivers(), 0);
}

TEST_F(DriverTest, terminate) {
  CursorParameters params;
  params.queryCtx = core::QueryCtx::create();

  params.planNode = makeValuesFilterProject(
      rowType_,
      "m1 % 10 > 0",
      "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
      100,
      100'000);
  params.numThreads = 10;
  int32_t numRead = 0;
  try {
    readResults(params, ResultOperation::kTerminate, 1'000'000, &numRead);
    // Not necessarily an exception.
  } catch (const std::exception& e) {
    // If this is an exception, it will be a cancellation.
    EXPECT_EQ("Cancelled", std::string(e.what()));
  }
  EXPECT_GE(numRead, 1'000'000);
  EXPECT_TRUE(stateFutures_.at(0).isReady());
  EXPECT_EQ(tasks_[0]->state(), kAborted);
}

TEST_F(DriverTest, slow) {
  CursorParameters params;
  params.planNode = makeValuesFilterProject(
      rowType_,
      "m1 % 10 > 0",
      "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
      300,
      1'000);
  params.numThreads = 10;
  int32_t numRead = 0;
  readResults(params, ResultOperation::kReadSlow, 50'000, &numRead);
  EXPECT_GE(numRead, 50'000);
  // Sync before checking end state. The cursor is at end as soon as
  // CallbackSink::finish is called. The thread count and task state
  // are updated some tens of instructions after this. Determinism
  // requires a barrier.
  auto& executor = folly::QueuedImmediateExecutor::instance();
  auto future = tasks_[0]->cancelPool()->finishFuture().via(&executor);
  future.wait();
  EXPECT_EQ(tasks_[0]->numDrivers(), 0);
  const auto stats = tasks_[0]->taskStats().pipelineStats;
  ASSERT_TRUE(!stats.empty() && !stats[0].operatorStats.empty());
  // Check that the blocking of the CallbackSink at the end of the pipeline is
  // recorded.
  EXPECT_GT(stats[0].operatorStats.back().blockedWallNanos, 0);
  EXPECT_TRUE(stateFutures_.at(0).isReady());
  // The future was realized by timeout.
  EXPECT_TRUE(stateFutures_.at(0).hasException());
}

TEST_F(DriverTest, pause) {
  CursorParameters params;
  int32_t hits;
  params.planNode = makeValuesFilterProject(
      rowType_,
      "m1 % 10 > 0",
      "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
      100,
      10'000,
      [](int64_t num) { return num % 10 > 0; },
      &hits);
  params.numThreads = 10;
  int32_t numRead = 0;
  readResults(params, ResultOperation::kPause, 370'000'000, &numRead);
  // Each thread will fully read the 1M rows in values.
  EXPECT_EQ(numRead, 10 * hits);
  EXPECT_TRUE(stateFutures_.at(0).isReady());
  EXPECT_EQ(tasks_[0]->state(), kFinished);
  EXPECT_EQ(tasks_[0]->numDrivers(), 0);
  const auto taskStats = tasks_[0]->taskStats();
  ASSERT_EQ(taskStats.pipelineStats.size(), 1);
  const auto& operators = taskStats.pipelineStats[0].operatorStats;
  EXPECT_GT(operators[1].getOutputTiming.wallNanos, 0);
  EXPECT_EQ(operators[0].outputPositions, 10000000);
  EXPECT_EQ(operators[1].inputPositions, 10000000);
  EXPECT_EQ(operators[1].outputPositions, 10 * hits);
}

TEST_F(DriverTest, yield) {
  constexpr int32_t kNumTasks = 20;
  constexpr int32_t kThreadsPerTask = 5;
  std::vector<int32_t> counters;
  counters.reserve(kNumTasks);
  std::vector<CursorParameters> params;
  params.resize(kNumTasks);
  int32_t hits;
  for (int32_t i = 0; i < kNumTasks; ++i) {
    params[i].planNode = makeValuesFilterProject(
        rowType_,
        "m1 % 10 > 0",
        "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
        200,
        2'000,
        [](int64_t num) { return num % 10 > 0; },
        &hits);
    params[i].numThreads = kThreadsPerTask;
  }
  std::vector<std::thread> threads;
  threads.reserve(kNumTasks);
  for (int32_t i = 0; i < kNumTasks; ++i) {
    counters.push_back(0);
    threads.push_back(std::thread([this, &params, &counters, i]() {
      readResults(params[i], ResultOperation::kYield, 10'000, &counters[i], i);
    }));
  }
  for (int32_t i = 0; i < kNumTasks; ++i) {
    threads[i].join();
    EXPECT_EQ(counters[i], kThreadsPerTask * hits);
    EXPECT_TRUE(stateFutures_.at(i).isReady());
  }
}

// A testing Operator that periodically does one of the following:
//
// 1. Blocks and registers a resume that continues the Driver after a timed
// pause. This simulates blocking to wait for exchange or consumer.
//
// 2. Enters a cancel-free section where the Driver is on thread but is not
// counted as running and is therefore instantaneously cancellable and pausable.
// Comes back on thread after a timed pause. This simulates an RPC to an out of
// process service.
//
// 3.  Enters a cancel-free section where this pauses and resumes random Tasks,
// including its own Task. This simulates making Tasks release memory under
// memory contention, checkpointing Tasks for migration or fault tolerance and
// other process-wide coordination activities.
//
// These situations will occur with arbitrary concurrency and sequence and must
// therefore be in one test to check against deadlocks.
class TestingPauser : public Operator {
 public:
  TestingPauser(
      DriverCtx* ctx,
      int32_t id,
      std::shared_ptr<const TestingPauserNode> node,
      DriverTest* test,
      int32_t sequence)
      : Operator(ctx, node->outputType(), id, node->id(), "Pauser"),
        test_(test),
        sequence_(sequence),
        counter_(sequence),
        future_(false) {
    test_->registerTask(operatorCtx_->task());
  }

  bool needsInput() const override {
    return !isFinishing_ && !input_;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    if (!input_) {
      return nullptr;
    }
    ++counter_;
    auto label = operatorCtx_->driver()->label();
    // Block for a time quantum evern 10th time.
    if (counter_ % 10 == 0) {
      test_->registerForWakeup(&future_);
      hasFuture_ = true;
      return nullptr;
    }
    {
      CancelFreeSection noCancel(operatorCtx_->driver());
      sleep(1);
      if (counter_ % 7 == 0) {
        // Every 7th time, stop and resume other Tasks. This operation is
        // globally serilized.
        std::lock_guard<std::mutex> l(pauseMutex_);

        for (auto i = 0; i <= counter_ % 3; ++i) {
          auto task = test_->randomTask();
          if (!task) {
            continue;
          }
          auto cancelPool = task->cancelPool();
          cancelPool->requestPause(true);
          auto& executor = folly::QueuedImmediateExecutor::instance();
          auto future = cancelPool->finishFuture().via(&executor);
          future.wait();
          sleep(2);
          cancelPool->requestPause(false);
          Task::resume(task);
        }
      }
    }

    return std::move(input_);
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    VELOX_CHECK(!operatorCtx_->driver()->state().isCancelFree);
    if (hasFuture_) {
      hasFuture_ = false;
      *future = std::move(future_);
      return BlockingReason::kWaitForConsumer;
    }
    return BlockingReason::kNotBlocked;
  }

  void finish() override {
    test_->unregisterTask(operatorCtx_->task());
    Operator::finish();
  }

 private:
  void sleep(int32_t units) {
    std::this_thread::sleep_for(std::chrono::milliseconds(units));
  }
  // The DriverTest under which this is running. Used for global context.
  DriverTest* test_;
  // Mutex to serialize the pause/restart exercise so that only one instance
  // does this at a time.
  static std::mutex pauseMutex_;

  // Sequence number of 'this' within the test run.
  const int32_t sequence_;
  // Counter for actions. Initialized from 'sequence_'. Decides what
  // the next action in getOutput() will be.
  int32_t counter_;
  bool hasFuture_{false};
  ContinueFuture future_;
};

std::mutex TestingPauser ::pauseMutex_;

TEST_F(DriverTest, pauserNode) {
  constexpr int32_t kNumTasks = 20;
  constexpr int32_t kThreadsPerTask = 5;
  static int32_t sequence = 0;
  Operator::registerOperator(
      [&](DriverCtx* ctx,
          int32_t id,
          std::shared_ptr<const core::PlanNode>& node)
          -> std::unique_ptr<TestingPauser> {
        if (auto pauser =
                std::dynamic_pointer_cast<const TestingPauserNode>(node)) {
          return std::make_unique<TestingPauser>(
              ctx, id, pauser, this, ++sequence);
        }
        return nullptr;
      });

  std::vector<int32_t> counters;
  counters.reserve(kNumTasks);
  std::vector<CursorParameters> params;
  params.resize(kNumTasks);
  int32_t hits;
  for (int32_t i = 0; i < kNumTasks; ++i) {
    params[i].queryCtx = core::QueryCtx::create();
    params[i].planNode = makeValuesFilterProject(
        rowType_,
        "m1 % 10 > 0",
        "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
        200,
        2'000,
        [](int64_t num) { return num % 10 > 0; },
        &hits,
        true);
    params[i].numThreads = kThreadsPerTask;
  }
  std::vector<std::thread> threads;
  threads.reserve(kNumTasks);
  for (int32_t i = 0; i < kNumTasks; ++i) {
    counters.push_back(0);
    threads.push_back(std::thread([this, &params, &counters, i]() {
      try {
        readResults(params[i], ResultOperation::kRead, 10'000, &counters[i], i);
      } catch (const std::exception& e) {
        LOG(INFO) << "Pauser task errored out " << e.what();
      }
    }));
  }
  for (int32_t i = 0; i < kNumTasks; ++i) {
    threads[i].join();
    EXPECT_EQ(counters[i], kThreadsPerTask * hits);
    EXPECT_TRUE(stateFutures_.at(i).isReady());
  }
}

// An operator that passes through its input but maintains a varying
// memory allocation. For example, a distinct with spilling would have a similar
// behavior.
class TestingConsumer : public Operator {
 public:
  TestingConsumer(
      DriverCtx* ctx,
      int32_t id,
      std::shared_ptr<const TestingConsumerNode> node)
      : Operator(ctx, node->outputType(), id, node->id(), "consumer"),
        recoverableTracker_(memory::MemoryUsageTracker::create(
            operatorCtx_->pool()->getMemoryUsageTracker(),
            memory::UsageType::kRecoverableMem,
            memory::MemoryUsageConfigBuilder().build())) {}

  bool needsInput() const override {
    return !isFinishing_ && !input_;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    if (!input_) {
      return nullptr;
    }
    int64_t size = 50 << 20;
    if (!reserveAndRun(
            recoverableTracker_,
            size,
            [this](int64_t size) {
              LOG(INFO) << "Spiller called: " << size;
              return spill(size);
            },
            [&]() {
              // Use the reserved memory.
              recoverableTracker_->update(
                  recoverableTracker_->getAvailableReservation());
            })) {
      VELOX_FAIL("Out of memory");
    }
    return std::move(input_);
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    return BlockingReason::kNotBlocked;
  }

  int64_t spill(int64_t size) override {
    int64_t freed =
        std::min(size, recoverableTracker_->getCurrentRecoverableBytes());
    recoverableTracker_->update(-freed);
    recoverableTracker_->release();
    return freed;
  }

 private:
  std::shared_ptr<memory::MemoryUsageTracker> recoverableTracker_;
};

TEST_F(DriverTest, memoryReservation) {
  return;
  constexpr int32_t kNumTasks = 20;
  constexpr int32_t kThreadsPerTask = 5;
  constexpr int64_t kProcessBytes = 1 << 30;
  constexpr int64_t kTotalMemory = 100 << 20; // 100MB.
  memory::MemoryManagerStrategy::registerFactory(
      [&]() { return std::make_unique<TaskMemoryStrategy>(kProcessBytes); });

  Operator::registerOperator(
      [](DriverCtx* ctx,
         int32_t id,
         std::shared_ptr<const core::PlanNode>& node)
          -> std::unique_ptr<TestingConsumer> {
        if (auto consumer =
                std::dynamic_pointer_cast<const TestingConsumerNode>(node)) {
          return std::make_unique<TestingConsumer>(ctx, id, consumer);
        }
        return nullptr;
      });

  auto& manager = memory::getProcessDefaultMemoryManager();
  std::vector<int32_t> counters;
  counters.reserve(kNumTasks);
  std::vector<CursorParameters> params;
  params.resize(kNumTasks);
  int32_t hits;
  for (int32_t i = 0; i < kNumTasks; ++i) {
    params[i].queryCtx = core::QueryCtx::create();
    params[i].planNode = makeValuesFilterProject(
        rowType_,
        "m1 % 10 > 0",
        "m1 % 3 + m2 % 5 + m3 % 7 + m4 % 11 + m5 % 13 + m6 % 17 + m7 % 19",
        200,
        2'000,
        [](int64_t num) { return num % 10 > 0; },
        &hits,
        false,
        true);
    params[i].numThreads = kThreadsPerTask;
  }
  std::vector<std::thread> threads;
  threads.reserve(kNumTasks);
  for (int32_t i = 0; i < kNumTasks; ++i) {
    counters.push_back(0);
    threads.push_back(std::thread([this, &params, &counters, i]() {
      try {
        readResults(params[i], ResultOperation::kRead, 10'000, &counters[i], i);
      } catch (const std::exception& e) {
        LOG(INFO) << "Reservation task errored out " << e.what();
      }
    }));
  }
  for (int32_t i = 0; i < kNumTasks; ++i) {
    threads[i].join();
    EXPECT_EQ(counters[i], kThreadsPerTask * hits);
    EXPECT_TRUE(stateFutures_.at(i).isReady());
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
