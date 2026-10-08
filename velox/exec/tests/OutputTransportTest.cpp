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

#include <functional>

#include <gmock/gmock.h>

#include "folly/Synchronized.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/DefaultOutputBufferManager.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {
namespace {

// A CPU test transport double: the default output buffer manager registered
// under a distinct transport id (kTransport), so operator selection can be
// verified without a real second transport (e.g. UCX).
class TestTransportBufferManager : public DefaultOutputBufferManager {
 public:
  static constexpr std::string_view kTransport{"test-transport"};

  TestTransportBufferManager() : DefaultOutputBufferManager(Options{}) {}
};

// Records task initialization and removal. Hooks run immediately before and
// after registration inside initializeTask().
class LifecycleRecordingBufferManager : public TestTransportBufferManager {
 public:
  void initializeTask(
      std::shared_ptr<Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers,
      const std::string& transportOptions) override {
    record("initializeTask");
    if (beforeRegister != nullptr) {
      beforeRegister(*task);
    }
    TestTransportBufferManager::initializeTask(
        task, kind, numDestinations, numDrivers, transportOptions);
    if (afterRegister != nullptr) {
      afterRegister(*task);
    }
  }

  void removeTask(const std::string& taskId) override {
    record("removeTask");
    TestTransportBufferManager::removeTask(taskId);
  }

  std::vector<std::string> calls() const {
    return calls_.copy();
  }

  std::function<void(Task&)> beforeRegister{};
  std::function<void(Task&)> afterRegister{};

 private:
  void record(std::string_view call) {
    calls_.wlock()->emplace_back(call);
  }

  folly::Synchronized<std::vector<std::string>> calls_;
};

using RecordingOutputOperatorBuilder = std::function<std::unique_ptr<Operator>(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& node,
    bool eagerFlush,
    const std::shared_ptr<LifecycleRecordingBufferManager>& manager)>;

std::unique_ptr<Operator> makePartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& node,
    bool eagerFlush,
    const std::shared_ptr<LifecycleRecordingBufferManager>& manager) {
  return std::make_unique<PartitionedOutput>(
      operatorId, ctx, node, eagerFlush, manager);
}

class OutputTransportTest : public HiveConnectorTestBase {
 protected:
  // Creates a broadcast task with a query-scoped 'manager'.
  std::shared_ptr<Task> makeTaskWithOutputManager(
      const std::string& taskId,
      const std::shared_ptr<LifecycleRecordingBufferManager>& manager,
      RecordingOutputOperatorBuilder makeOutputOperator) {
    const std::string transportType{TestTransportBufferManager::kTransport};
    auto queryRegistry = OutputTransportRegistry::create();
    queryRegistry->insert(
        transportType,
        OutputTransportEntry::make<LifecycleRecordingBufferManager>(
            manager, std::move(makeOutputOperator)));
    auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
    queryCtx->setRegistry(OutputTransportRegistry::kRegistryKey, queryRegistry);
    return Task::create(
        taskId,
        PlanBuilder()
            .tableScan(ROW("c0", BIGINT()))
            .partitionedOutputBroadcast({}, "Presto", transportType)
            .planFragment(),
        0,
        queryCtx,
        Task::ExecutionMode::kParallel,
        exec::Consumer{});
  }
};

TEST_F(OutputTransportTest, usesDefaultAfterRegistryClear) {
  // unregisterAll() resets the registry to its baseline -- the built-in
  // in-memory default -- so a default task still runs after a clear.
  OutputTransportRegistry::unregisterAll();

  auto plan = PlanBuilder()
                  .tableScan(ROW({"c0"}, {BIGINT()}))
                  .project({"c0 % 10"})
                  .partitionedOutputBroadcast({})
                  .planFragment();
  auto task = Task::create(
      "task-default-output-manager-after-registry-clear",
      plan,
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel,
      exec::Consumer{});

  task->start(1, 1);

  auto defaultEntry = OutputTransportRegistry::tryGet(
      std::string(core::TransportKind::kInMemory));
  EXPECT_NE(defaultEntry, nullptr);
  EXPECT_TRUE(task->updateOutputBuffers(10, true /*noMoreBuffers*/));

  task->requestCancel();
  waitForTaskCompletion(task.get());
}

TEST_F(OutputTransportTest, errorsOnUnregisteredTransport) {
  // A node naming a transport with no registered manager is a misconfiguration:
  // resolution fails fast rather than silently running over another transport.
  const std::string transportType{"ucx-unregistered"};
  auto plan = PlanBuilder()
                  .tableScan(ROW({"c0"}, {BIGINT()}))
                  .project({"c0 % 10"})
                  .partitionedOutputBroadcast({}, "Presto", transportType)
                  .planFragment();

  auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
  EXPECT_EQ(OutputTransportRegistry::tryGet(transportType), nullptr);

  auto task = Task::create(
      "task-transport-unregistered",
      std::move(plan),
      0,
      queryCtx,
      Task::ExecutionMode::kParallel,
      exec::Consumer{});
  VELOX_ASSERT_USER_THROW(
      task->start(1, 1), "No output buffer manager registered for transport");
}

TEST_F(OutputTransportTest, selectsOperatorByTransportKind) {
  OutputTransportRegistry::unregisterAll();

  const std::string transportType{TestTransportBufferManager::kTransport};
  // One entry pairing a test manager with an operator factory that records it
  // ran, then delegates to a real PartitionedOutput. Proves createDriver builds
  // the operator from the resolved manager's entry, without needing UCX.
  auto builderInvocations = std::make_shared<std::atomic<int>>(0);
  auto manager = std::make_shared<TestTransportBufferManager>();
  OutputTransportRegistry::global().insert(
      transportType,
      OutputTransportEntry::make<TestTransportBufferManager>(
          manager,
          [builderInvocations](
              int32_t operatorId,
              DriverCtx* ctx,
              const std::shared_ptr<const core::PartitionedOutputNode>& node,
              bool eagerFlush,
              const std::shared_ptr<TestTransportBufferManager>& manager)
              -> std::unique_ptr<Operator> {
            ++*builderInvocations;
            return std::make_unique<PartitionedOutput>(
                operatorId, ctx, node, eagerFlush, manager);
          }));

  auto plan = PlanBuilder()
                  .tableScan(ROW({"c0"}, {BIGINT()}))
                  .project({"c0 % 10"})
                  .partitionedOutputBroadcast({}, "Presto", transportType)
                  .planFragment();

  auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
  auto task = Task::create(
      "task-transport-selection",
      std::move(plan),
      0,
      queryCtx,
      Task::ExecutionMode::kParallel,
      exec::Consumer{});
  task->start(1, 1);

  // createDriver used the transport entry's factory, not the default path.
  EXPECT_EQ(builderInvocations->load(), 1);
  EXPECT_TRUE(task->updateOutputBuffers(10, true /*noMoreBuffers*/));

  task->requestCancel();
  waitForTaskCompletion(task.get());
  OutputTransportRegistry::unregisterAll();
}

TEST_F(OutputTransportTest, removesTaskOnceWhenStartFails) {
  // A startup failure after buffer creation must remove the task exactly once.
  auto manager = std::make_shared<LifecycleRecordingBufferManager>();
  auto task = makeTaskWithOutputManager(
      "task-output-start-failure",
      manager,
      [](int32_t,
         DriverCtx*,
         const std::shared_ptr<const core::PartitionedOutputNode>&,
         bool,
         const std::shared_ptr<LifecycleRecordingBufferManager>&)
          -> std::unique_ptr<Operator> {
        VELOX_FAIL("Output operator construction failed");
      });

  VELOX_ASSERT_THROW(task->start(1, 1), "Output operator construction failed");
  EXPECT_THAT(
      manager->calls(), ::testing::ElementsAre("initializeTask", "removeTask"));
}

// If abort overlaps initializeTask(), start() owns removal after it returns.
TEST_F(OutputTransportTest, removesTaskOnceWhenAbortedBeforeRegistration) {
  auto manager = std::make_shared<LifecycleRecordingBufferManager>();
  manager->beforeRegister = [](Task& task) { task.requestAbort().wait(); };
  auto task = makeTaskWithOutputManager(
      "task-output-abort-before-registration", manager, makePartitionedOutput);

  task->start(1, 1);
  EXPECT_EQ(task->state(), TaskState::kAborted);
  EXPECT_THAT(
      manager->calls(), ::testing::ElementsAre("initializeTask", "removeTask"));
}

TEST_F(OutputTransportTest, removesTaskOnceWhenAbortedAfterRegistration) {
  auto manager = std::make_shared<LifecycleRecordingBufferManager>();
  manager->afterRegister = [](Task& task) { task.requestAbort().wait(); };
  auto task = makeTaskWithOutputManager(
      "task-output-abort-after-registration", manager, makePartitionedOutput);

  task->start(1, 1);
  EXPECT_EQ(task->state(), TaskState::kAborted);
  EXPECT_THAT(
      manager->calls(), ::testing::ElementsAre("initializeTask", "removeTask"));
}

// removeTask() cleans up a partial registration when initializeTask() throws.
TEST_F(OutputTransportTest, removesTaskOnceWhenInitializeTaskThrows) {
  auto manager = std::make_shared<LifecycleRecordingBufferManager>();
  manager->afterRegister = [](Task&) {
    VELOX_FAIL("Output buffer initialization failed");
  };
  auto task = makeTaskWithOutputManager(
      "task-output-initialize-failure", manager, makePartitionedOutput);

  VELOX_ASSERT_THROW(task->start(1, 1), "Output buffer initialization failed");
  EXPECT_EQ(task->state(), TaskState::kFailed);
  EXPECT_THAT(
      manager->calls(), ::testing::ElementsAre("initializeTask", "removeTask"));
}

TEST_F(OutputTransportTest, removesTaskOnceWhenAbortedAndInitializeTaskThrows) {
  auto manager = std::make_shared<LifecycleRecordingBufferManager>();
  manager->afterRegister = [](Task& task) {
    task.requestAbort().wait();
    VELOX_FAIL("Output buffer initialization failed");
  };
  auto task = makeTaskWithOutputManager(
      "task-output-abort-and-initialize-failure",
      manager,
      makePartitionedOutput);

  VELOX_ASSERT_THROW(task->start(1, 1), "Output buffer initialization failed");
  EXPECT_EQ(task->state(), TaskState::kAborted);
  EXPECT_THAT(
      manager->calls(), ::testing::ElementsAre("initializeTask", "removeTask"));
}

// After initialization completes, terminate() owns removal.
DEBUG_ONLY_TEST_F(
    OutputTransportTest,
    removesTaskOnceWhenAbortedBeforeDriversStart) {
  auto manager = std::make_shared<LifecycleRecordingBufferManager>();
  auto task = makeTaskWithOutputManager(
      "task-output-abort-before-drivers", manager, makePartitionedOutput);
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Task::createAndStartDrivers",
      std::function<void(Task*)>(
          [](Task* task) { task->requestAbort().wait(); }));

  task->start(1, 1);
  EXPECT_EQ(task->state(), TaskState::kAborted);
  EXPECT_THAT(
      manager->calls(), ::testing::ElementsAre("initializeTask", "removeTask"));
}

} // namespace
} // namespace facebook::velox::exec::test
