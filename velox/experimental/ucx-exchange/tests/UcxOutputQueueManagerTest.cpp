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
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <folly/Executor.h>
#include <folly/synchronization/EventCount.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeRegistration.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

using namespace facebook::velox::ucx_exchange;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

class UcxOutputQueueManagerTest : public testing::Test {
 protected:
  UcxOutputQueueManagerTest() {}

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    common::testutil::TestValue::enable();
  }

  void SetUp() override {
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    queueManager_ = UcxOutputQueueManager::getInstanceRef();
  }

  std::shared_ptr<Task> initializeTask(
      const std::string& taskId,
      int numDestinations,
      int numDrivers,
      bool cleanup = true,
      core::PartitionedOutputNode::Kind kind =
          core::PartitionedOutputNode::Kind::kPartitioned,
      uint64_t maxOutputBufferSize = FOUR_GBYTES) {
    if (cleanup) {
      queueManager_->removeTask(taskId);
    }

    auto task = createSourceTask(
        taskId, pool_, UcxTestData::kTestRowType, maxOutputBufferSize);

    queueManager_->initializeTask(task, kind, numDestinations, numDrivers);
    return task;
  }

  std::unique_ptr<cudf::packed_columns> makePackedColumns(std::size_t numRows) {
    cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
    auto table = facebook::velox::ucx_exchange::makeTable(
        numRows, UcxTestData::kTestRowType, stream);
    auto cols = std::make_unique<cudf::packed_columns>(
        cudf::pack(table->view(), stream));
    stream.sync();
    return cols;
  }

  void enqueue(std::string_view taskId, vector_size_t size) {
    enqueue(taskId, 0, size);
  }

  void enqueue(std::string_view taskId, int destination, vector_size_t size) {
    auto data = makePackedColumns(size);
    queueManager_->enqueue(taskId, destination, std::move(data), size);
  }

  void noMoreData(std::string_view taskId) {
    queueManager_->noMoreData(taskId);
  }

  void fetch(
      std::string_view taskId,
      int destination,
      bool expectedEndMarker = false) {
    bool receivedData = false;
    queueManager_->getData(
        taskId,
        destination,
        [destination, expectedEndMarker, &receivedData](
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> remainingBytes,
            std::optional<std::string> /*producerError*/) {
          ASSERT_EQ(expectedEndMarker, data == nullptr)
              << "for destination " << destination;
          receivedData = true;
        });
    ASSERT_TRUE(receivedData) << "for destination " << destination;
  }

  UcxDataAvailableCallback receiveEndMarker(
      int destination,
      bool& receivedEndMarker) {
    return [destination, &receivedEndMarker](
               std::shared_ptr<cudf::packed_columns> data,
               vector_size_t /*numRows*/,
               std::vector<int64_t> remainingBytes,
               std::optional<std::string> /*producerError*/) {
      EXPECT_FALSE(receivedEndMarker) << "for destination " << destination;
      EXPECT_TRUE(data == nullptr) << "for destination " << destination;
      EXPECT_TRUE(remainingBytes.empty());
      receivedEndMarker = true;
    };
  }

  void fetchEndMarker(std::string_view taskId, int destination) {
    bool receivedData = false;
    queueManager_->getData(
        taskId, destination, receiveEndMarker(destination, receivedData));
    EXPECT_TRUE(receivedData) << "for destination " << destination;
    queueManager_->deleteResults(taskId, destination);
  }

  void deleteResults(std::string_view taskId, int destination) {
    queueManager_->deleteResults(taskId, destination);
  }

  // What a server passes to IntraNodeTransferRegistry::publish() for a page it
  // took with getIntraNodeData().
  std::function<void()> countIntraNodePageRead(
      std::string_view taskId,
      int destination) {
    return [this, taskId = std::string{taskId}, destination]() {
      queueManager_->intraNodePageRetrieved(taskId, destination);
    };
  }

  void registerForEndMarker(
      std::string_view taskId,
      int destination,
      bool& receivedEndMarker) {
    receivedEndMarker = false;
    queueManager_->getData(
        taskId, destination, receiveEndMarker(destination, receivedEndMarker));
    EXPECT_FALSE(receivedEndMarker) << "for destination " << destination;
  }

  UcxDataAvailableCallback receiveData(int destination, bool& receivedData) {
    receivedData = false;
    return [destination, &receivedData](
               std::shared_ptr<cudf::packed_columns> data,
               vector_size_t /*numRows*/,
               std::vector<int64_t> /*remainingBytes*/,
               std::optional<std::string> /*producerError*/) {
      EXPECT_FALSE(receivedData) << "for destination " << destination;
      EXPECT_TRUE(data != nullptr) << "for destination " << destination;
      receivedData = true;
    };
  }

  void registerForData(
      std::string_view taskId,
      int destination,
      bool& receivedData) {
    receivedData = false;
    queueManager_->getData(
        taskId, destination, receiveData(destination, receivedData));
    EXPECT_FALSE(receivedData) << "for destination " << destination;
  }

  void dataFetcher(
      std::string_view taskId,
      int destination,
      int64_t& fetchedPackedColumns,
      bool earlyTermination) {
    int64_t received{0};
    while (true) {
      if (earlyTermination && folly::Random().oneIn(200)) {
        queueManager_->deleteResults(taskId, destination);
        return;
      }
      bool atEnd{false};
      folly::EventCount dataWait;
      auto dataWaitKey = dataWait.prepareWait();
      queueManager_->getData(
          taskId,
          destination,
          [&](std::shared_ptr<cudf::packed_columns> data,
              vector_size_t /*numRows*/,
              std::vector<int64_t> /*remainingBytes*/,
              std::optional<std::string> /*producerError*/) {
            if (data == nullptr) {
              atEnd = true;
            } else {
              received++;
            }
            dataWait.notify();
          });
      dataWait.wait(dataWaitKey);
      if (atEnd) {
        break;
      }
    }
    queueManager_->deleteResults(taskId, destination);
    // out of order requests are allowed (fetch after delete)
    {
      struct Response {
        std::shared_ptr<cudf::packed_columns> data;
        std::vector<int64_t> remainingBytes;
      };
      folly::Promise<Response> promise;
      auto future = promise.getSemiFuture();
      queueManager_->getData(
          taskId,
          destination,
          [&promise](
              std::shared_ptr<cudf::packed_columns> data,
              vector_size_t /*numRows*/,
              std::vector<int64_t> remainingBytes,
              std::optional<std::string> /*producerError*/) {
            promise.setValue(
                Response{std::move(data), std::move(remainingBytes)});
          });
      future.wait();
      ASSERT_TRUE(future.isReady());
      auto& response = future.value();
      ASSERT_EQ(response.remainingBytes.size(), 0);
      ASSERT_EQ(response.data, nullptr);
    }

    fetchedPackedColumns = received;
  }

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
  std::shared_ptr<UcxOutputQueueManager> queueManager_;
};

// Drives all eight exec::OutputBufferManager virtuals through a base-class
// pointer, which is the only surface exec::Task and the Presto stats layer see.
// The utilization pair is pinned to the queue's own byte accounting rather than
// to a loose bound, so a constant-returning implementation cannot satisfy it.
TEST_F(UcxOutputQueueManagerTest, implementsOutputBufferManager) {
  static_assert(
      std::is_base_of_v<exec::OutputBufferManager, UcxOutputQueueManager>);

  std::shared_ptr<exec::OutputBufferManager> mgr = queueManager_;
  ASSERT_NE(mgr, nullptr);

  // An unknown task is tolerated by every method, and the two observability
  // methods report "no answer" rather than inventing one.
  const std::string unknownTaskId = "outputBufferManagerIface.unknown";
  EXPECT_NO_THROW(mgr->toString(unknownTaskId));
  EXPECT_FALSE(mgr->updateNumDrivers(unknownTaskId, 1));
  EXPECT_FALSE(mgr->updateOutputBuffers(unknownTaskId, 1, true));
  EXPECT_EQ(mgr->stats(unknownTaskId), std::nullopt);
  EXPECT_EQ(mgr->getUtilization(unknownTaskId), std::nullopt);
  EXPECT_EQ(mgr->isOverutilized(unknownTaskId), std::nullopt);

  // Use an id unique to this case: the manager is a process-wide singleton
  // shared with every other case in the binary.
  const std::string taskId = "outputBufferManagerIface";
  const int numDestinations = 2;
  // A capacity small enough that a handful of pages crosses the
  // over-utilization threshold, so isOverutilized() is seen in both states.
  const uint64_t maxOutputBufferSize{4'096};
  auto task = initializeTask(
      taskId,
      numDestinations,
      1 /*numDrivers*/,
      true /*cleanup*/,
      core::PartitionedOutputNode::Kind::kPartitioned,
      maxOutputBufferSize);

  // Empty but bounded: a real ratio of exactly zero, which is a different
  // answer from the unbounded placeholder queue's nullopt (see
  // lateTaskCreation).
  auto stats = mgr->stats(taskId);
  ASSERT_TRUE(stats.has_value());
  ASSERT_EQ(stats->bufferedBytes, 0);
  EXPECT_EQ(mgr->getUtilization(taskId), 0.0);
  EXPECT_EQ(mgr->isOverutilized(taskId), false);

  // One page: utilization is queued bytes over the configured capacity.
  // Producers are only blocked *after* adding data (UcxOutputQueue::
  // checkBlocked), so queuedBytes_ may exceed maxSize_ and the ratio is not
  // bounded by 1.0.
  enqueue(taskId, 0, /*size=*/10);
  stats = mgr->stats(taskId);
  ASSERT_TRUE(stats.has_value());
  const int64_t bufferedBytes = stats->bufferedBytes;
  ASSERT_GT(bufferedBytes, 0);

  auto utilization = mgr->getUtilization(taskId);
  ASSERT_TRUE(utilization.has_value());
  EXPECT_GT(*utilization, 0.0);
  EXPECT_DOUBLE_EQ(
      *utilization, bufferedBytes / static_cast<double>(maxOutputBufferSize));

  const auto halfCapacity = static_cast<int64_t>(maxOutputBufferSize / 2);
  auto overutilized = mgr->isOverutilized(taskId);
  ASSERT_TRUE(overutilized.has_value());
  EXPECT_EQ(*overutilized, bufferedBytes > halfCapacity);

  // Fill past half the capacity: the flag must follow the queued bytes. The
  // iteration bound only stops a runaway loop if enqueue ever stops
  // accumulating; the assertion after it is what the case actually checks.
  for (int i = 0; i < 100 && mgr->stats(taskId)->bufferedBytes <= halfCapacity;
       ++i) {
    enqueue(taskId, 0, /*size=*/10);
  }
  ASSERT_GT(mgr->stats(taskId)->bufferedBytes, halfCapacity);
  EXPECT_EQ(mgr->isOverutilized(taskId), true);
  ASSERT_TRUE(mgr->getUtilization(taskId).has_value());
  EXPECT_GT(*mgr->getUtilization(taskId), 0.5);

  EXPECT_TRUE(mgr->updateNumDrivers(taskId, 2));
  EXPECT_TRUE(mgr->updateOutputBuffers(taskId, numDestinations, true));
  EXPECT_FALSE(mgr->toString(taskId).empty());

  mgr->removeTask(taskId);
  EXPECT_EQ(mgr->stats(taskId), std::nullopt);
}

// The output half of the kUcx registration: the entry the manager is published
// under, and that a second registration replaces rather than rejects.
TEST_F(UcxOutputQueueManagerTest, registersUcxOutputTransport) {
  // Start from a clean baseline so this case does not depend on registration
  // state left behind by other suites in the binary. unregisterAll() re-seeds
  // the built-in in-memory default, so only kUcx is actually cleared.
  exec::OutputTransportRegistry::unregisterAll();
  ASSERT_EQ(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);

  registerUcxTransports();
  auto entry = exec::OutputTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->manager, UcxOutputQueueManager::getInstanceRef());
  EXPECT_TRUE(static_cast<bool>(entry->makeOutputOperator));

  // Idempotent: a second call replaces the entry rather than throwing on the
  // duplicate key.
  registerUcxTransports();
  auto entryAfterSecondCall = exec::OutputTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(entryAfterSecondCall, nullptr);
  EXPECT_EQ(
      entryAfterSecondCall->manager, UcxOutputQueueManager::getInstanceRef());

  // unregisterUcxTransports() erases kUcx from both registries and leaves the
  // in-memory default alone.
  unregisterUcxTransports();
  EXPECT_EQ(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);
  EXPECT_EQ(
      exec::ExchangeTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);
  EXPECT_NE(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kInMemory}),
      nullptr);
}

TEST_F(UcxOutputQueueManagerTest, basicPartitioned) {
  vector_size_t size = 100;
  std::string taskId = "t0";
  auto task = initializeTask(taskId, 5 /* numDestinations*/, 1 /*numDrivers*/);

  ASSERT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 5; ++destination) {
    enqueue(taskId, destination, size);
  }
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 5; destination++) {
    fetch(taskId, destination);
  }

  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 3; destination++) {
    enqueue(taskId, destination, size);
  }

  for (int destination = 0; destination < 3; destination++) {
    fetch(taskId, destination);
  }

  // destinations 0-2 received 2 packed_columns each
  // destinations 3-4 received 1 packed_columns each

  bool receivedEndMarker3;
  registerForEndMarker(taskId, 3, receivedEndMarker3);

  bool receivedData4;
  registerForData(taskId, 4, receivedData4);

  enqueue(taskId, 4, size);
  EXPECT_TRUE(receivedData4);

  noMoreData(taskId);
  EXPECT_TRUE(receivedEndMarker3);
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 3; destination++) {
    fetchEndMarker(taskId, destination);
  }

  EXPECT_TRUE(task->isRunning());

  deleteResults(taskId, 3);
  fetchEndMarker(taskId, 4);
  EXPECT_TRUE(queueManager_->isFinished(taskId));

  queueManager_->removeTask(taskId);
  EXPECT_TRUE(task->isFinished());
}

TEST_F(UcxOutputQueueManagerTest, invalidReaderDestinations) {
  const std::string taskId = "invalidReaderDestinations";
  initializeTask(taskId, 1, 1);
  for (int destination : {-1, 1, std::numeric_limits<int>::max()}) {
    bool receivedEnd{false};
    queueManager_->getData(
        taskId, destination, receiveEndMarker(destination, receivedEnd));
    EXPECT_TRUE(receivedEnd);
    queueManager_->deleteResults(taskId, destination);
  }
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, basicAsyncFetch) {
  const vector_size_t size = 10;
  const std::string taskId = "t0";
  int numPartitions = 1;
  bool earlyTermination = false;

  initializeTask(taskId, numPartitions, 1);

  // asynchronously fetch data.
  int64_t fetchedPackedColumns = 0;
  std::thread thread([&]() {
    dataFetcher(taskId, 0, fetchedPackedColumns, earlyTermination);
  });
  const int totalPackedColumns = 100;
  int64_t producedPackedColumns = 0;

  for (int i = 0; i < totalPackedColumns; ++i) {
    const int partition = 0;
    try {
      enqueue(taskId, partition, size);
    } catch (...) {
      // Early termination might cause the task to fail early.
      ASSERT_TRUE(earlyTermination);
      break;
    }
    ++producedPackedColumns;
    if (folly::Random().oneIn(4)) {
      std::this_thread::sleep_for(std::chrono::microseconds(5)); // NOLINT
    }
  }
  noMoreData(taskId);

  thread.join();

  if (!earlyTermination) {
    ASSERT_EQ(fetchedPackedColumns, producedPackedColumns);
  }
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, lateTaskCreation) {
  const vector_size_t size = 10;
  const std::string taskId = "t0";
  int numPartitions = 1;
  int destination = 0;

  // A manager with no state from prior tests, so that getData() below creates
  // a placeholder for a task it has not seen.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  // Fetch data from a non-existing task.
  struct Response {
    std::shared_ptr<cudf::packed_columns> data;
    std::vector<int64_t> remainingBytes;
  };
  folly::Promise<Response> promise;
  auto future = promise.getSemiFuture();
  queueManager_->getData(
      taskId,
      destination,
      [&promise](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes,
          std::optional<std::string> /*producerError*/) {
        promise.setValue(Response{std::move(data), std::move(remainingBytes)});
      });

  // A placeholder has no configured capacity until initializeTask().
  std::shared_ptr<exec::OutputBufferManager> mgr = queueManager_;
  EXPECT_EQ(mgr->getUtilization(taskId), std::nullopt);
  EXPECT_EQ(mgr->isOverutilized(taskId), std::nullopt);

  // initialize task.
  auto task = initializeTask(taskId, numPartitions, 1, false);
  // enqueue some data.
  enqueue(taskId, destination, size);
  noMoreData(taskId);
  uint32_t i = 0;
  while (true) {
    ++i;
    future.wait();
    ASSERT_TRUE(future.isReady());
    Response& response = future.value();
    if (!response.data) {
      // nullptr -> end of transmission.
      break;
    }
    folly::Promise<Response> promise;
    future = promise.getSemiFuture();
    queueManager_->getData(
        taskId,
        destination,
        [&promise](
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> remainingBytes,
            std::optional<std::string> /*producerError*/) {
          promise.setValue(
              Response{std::move(data), std::move(remainingBytes)});
        });
  }
  ASSERT_EQ(i, 2);

  queueManager_->deleteResults(taskId, destination);
  queueManager_->removeTask(taskId);
  EXPECT_TRUE(task->isFinished());
}

TEST_F(UcxOutputQueueManagerTest, pendingReadersBeforeInitialization) {
  for (auto kind :
       {core::PartitionedOutputNode::Kind::kPartitioned,
        core::PartitionedOutputNode::Kind::kBroadcast}) {
    SCOPED_TRACE(kind);
    const std::string taskId = "pendingBeforeInit";
    // A manager that has not seen the task, so getData() below leaves
    // placeholders.
    queueManager_ = std::make_shared<UcxOutputQueueManager>();
    bool receivedEnd{false};
    queueManager_->getData(taskId, 3, receiveEndMarker(3, receivedEnd));
    EXPECT_FALSE(receivedEnd);
    bool receivedData{false};
    queueManager_->getData(taskId, 0, receiveData(0, receivedData));
    int endCount{0};
    queueManager_->getData(
        taskId,
        std::numeric_limits<int>::max(),
        [&](std::shared_ptr<cudf::packed_columns> data,
            vector_size_t,
            std::vector<int64_t>,
            std::optional<std::string>) {
          EXPECT_EQ(data, nullptr);
          EXPECT_TRUE(queueManager_->stats(taskId).has_value());
          ++endCount;
        });
    EXPECT_FALSE(queueManager_->isFinished(taskId));
    initializeTask(taskId, 1, 1, false, kind);
    EXPECT_EQ(
        endCount,
        kind == core::PartitionedOutputNode::Kind::kPartitioned ? 1 : 0);
    enqueue(taskId, 0, 10);
    EXPECT_TRUE(receivedData);
    EXPECT_NO_THROW(queueManager_->updateOutputBuffers(taskId, 1, true));
    EXPECT_TRUE(receivedEnd);
    EXPECT_EQ(endCount, 1);
    noMoreData(taskId);
    fetchEndMarker(taskId, 0);
    EXPECT_TRUE(queueManager_->isFinished(taskId));
    queueManager_->removeTask(taskId);
  }
}

TEST_F(UcxOutputQueueManagerTest, multiFetchers) {
  const std::vector<bool> earlyTerminations = {false, true};
  for (const auto earlyTermination : earlyTerminations) {
    SCOPED_TRACE(fmt::format("earlyTermination {}", earlyTermination));

    const vector_size_t size = 10;
    const std::string taskId = "t0";
    int numPartitions = 10;
    initializeTask(taskId, numPartitions, 1);

    // create one thread per partition to asynchronously fetch data.
    std::vector<std::thread> threads;
    std::vector<int64_t> fetchedPackedColumns(numPartitions, 0);

    for (size_t i = 0; i < numPartitions; ++i) {
      threads.emplace_back([&, i]() {
        dataFetcher(taskId, i, fetchedPackedColumns.at(i), earlyTermination);
      });
    }

    // enqueue packed columns.
    folly::Random::DefaultGenerator rng;
    rng.seed(1234);
    const int totalPackedColumns = 1000;
    std::vector<int64_t> producedPackedColumns(numPartitions, 0);
    for (int i = 0; i < totalPackedColumns; ++i) {
      // randomly chose a partition.
      const int partition = folly::Random().rand32(rng) % numPartitions;
      try {
        enqueue(taskId, partition, size);
      } catch (...) {
        // Early termination might cause the task to fail early.
        ASSERT_TRUE(earlyTermination);
        break;
      }
      ++producedPackedColumns[partition];
      if (folly::Random().oneIn(4)) {
        std::this_thread::sleep_for(std::chrono::microseconds(5)); // NOLINT
      }
    }
    noMoreData(taskId);

    for (int i = 0; i < threads.size(); ++i) {
      threads[i].join();
    }

    if (!earlyTermination) {
      for (int i = 0; i < numPartitions; ++i) {
        ASSERT_EQ(fetchedPackedColumns[i], producedPackedColumns[i]);
      }
    }
    queueManager_->removeTask(taskId);
  }
}

// A producer that never initialized its output and is then removed must still
// unblock its consumer, and the consumer must learn that the producer ended
// early: a getData() before initializeTask() leaves a placeholder queue whose
// callback removeTask() fires.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateBeforeInit) {
  const std::string taskId = "orphanTest";
  // A manager that has not seen the task, so getData() below leaves a
  // placeholder.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  bool callbackFired = false;
  bool receivedNullptr = false;
  std::optional<std::string> producerError;
  queueManager_->getData(
      taskId,
      0, // destination
      [&callbackFired, &receivedNullptr, &producerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes,
          std::optional<std::string> error) {
        callbackFired = true;
        receivedNullptr = (data == nullptr);
        producerError = std::move(error);
      });

  // Callback should NOT have fired yet (queue is empty, no data).
  EXPECT_FALSE(callbackFired);

  // Now simulate the producer failing: removeTask fires terminate().
  // The stub queue has task_ == nullptr so the isRunning() check is skipped.
  queueManager_->removeTask(taskId);

  // Callback must have been fired with nullptr.
  EXPECT_TRUE(callbackFired);
  EXPECT_TRUE(receivedNullptr);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before initializing its output")));
}

// A task whose output initialization failed is removed with no queue
// registered. A consumer that arrives while the removal runs leaves a
// placeholder queue, which the removal then takes. That consumer must learn
// that the producer never initialized its output, rather than get a clean end
// of stream.
DEBUG_ONLY_TEST_F(
    UcxOutputQueueManagerTest,
    placeholderCreatedDuringRemoveTaskReportsProducerError) {
  const std::string taskId = "placeholderDuringRemoval";
  // A manager that has not seen the task, as after an initializeTask() that
  // threw before it registered a queue.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  bool fired = false;
  std::optional<std::string> producerError;
  bool consumerArrived = false;
  SCOPED_TESTVALUE_SET(
      "facebook::velox::ucx_exchange::UcxOutputQueueManager::removeTask::afterProducerError",
      std::function<void(UcxOutputQueueManager*)>(
          [&](UcxOutputQueueManager* manager) {
            if (consumerArrived) {
              return;
            }
            consumerArrived = true;
            manager->getData(
                taskId,
                0,
                [&fired, &producerError](
                    std::shared_ptr<cudf::packed_columns> data,
                    vector_size_t /*numRows*/,
                    std::vector<int64_t> /*remainingBytes*/,
                    std::optional<std::string> error) {
                  EXPECT_EQ(data, nullptr);
                  fired = true;
                  producerError = std::move(error);
                });
            EXPECT_FALSE(fired);
          }));

  queueManager_->removeTask(taskId);

  ASSERT_TRUE(consumerArrived);
  EXPECT_TRUE(fired);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before initializing its output")));
}

// A task whose output initialization failed is removed with no queue
// registered. A consumer that arrives only after the removal finds no queue
// either. It must learn that the producer never initialized its output, rather
// than wait on a queue that no task will initialize or remove.
TEST_F(
    UcxOutputQueueManagerTest,
    consumerAfterRemoveTaskWithoutQueueReportsProducerError) {
  const std::string taskId = "consumerAfterRemovalWithoutQueue";
  // A manager that has not seen the task, as after an initializeTask() that
  // threw before it registered a queue.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  queueManager_->removeTask(taskId);

  bool fired = false;
  std::optional<std::string> producerError;
  queueManager_->getData(
      taskId,
      0,
      [&fired, &producerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        fired = true;
        producerError = std::move(error);
      });

  EXPECT_TRUE(fired);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before initializing its output")));
}

// A producer that initialized its output but is removed before every driver
// called noMoreData() must end each destination with its reason, not with a
// clean end of stream.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateAfterInit) {
  const std::string taskId = "crashTest";
  constexpr int kNumDestinations = 2;
  auto task = initializeTask(taskId, kNumDestinations, 1);

  std::array<bool, kNumDestinations> receivedEndMarkers{};
  for (int destination = 0; destination < kNumDestinations; ++destination) {
    registerForEndMarker(taskId, destination, receivedEndMarkers[destination]);
  }

  // Register callbacks on both destinations.
  bool callback0Fired = false;
  bool callback0Nullptr = false;
  std::optional<std::string> producerError0;
  bool callback1Fired = false;
  bool callback1Nullptr = false;
  std::optional<std::string> producerError1;
  queueManager_->getData(
      taskId,
      0,
      [&callback0Fired, &callback0Nullptr, &producerError0](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes,
          std::optional<std::string> error) {
        callback0Fired = true;
        callback0Nullptr = (data == nullptr);
        producerError0 = std::move(error);
      });
  queueManager_->getData(
      taskId,
      1,
      [&callback1Fired, &callback1Nullptr, &producerError1](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> remainingBytes,
          std::optional<std::string> error) {
        callback1Fired = true;
        callback1Nullptr = (data == nullptr);
        producerError1 = std::move(error);
      });

  // Neither callback should fire yet.
  EXPECT_FALSE(callback0Fired);
  EXPECT_FALSE(callback1Fired);

  // Simulate task failure: abort the task, as Task::terminate() does before it
  // removes the task from the queue manager.
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  const auto abortedEarly = testing::Optional(
      testing::AllOf(
          testing::HasSubstr(
              "Producer task ended before sending all its output"),
          testing::HasSubstr("Aborted")));
  EXPECT_TRUE(callback0Fired);
  EXPECT_TRUE(callback0Nullptr);
  EXPECT_THAT(producerError0, abortedEarly);
  EXPECT_TRUE(callback1Fired);
  EXPECT_TRUE(callback1Nullptr);
  EXPECT_THAT(producerError1, abortedEarly);
}

// The coordinator also aborts producers that already sent all their output,
// for instance after a downstream LIMIT was satisfied. Once their servers took
// every page, their consumers have everything, so both the end marker queued
// by noMoreData() and a getData() after removal must stay a clean end of
// stream.
TEST_F(UcxOutputQueueManagerTest, removeTaskAfterAllOutputKeepsCleanEndMarker) {
  const std::string taskId = "finishedThenAborted";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);

  bool endMarkerFired = false;
  std::optional<std::string> endMarkerError{"not set"};
  queueManager_->getData(
      taskId,
      0,
      [&endMarkerFired, &endMarkerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        endMarkerFired = true;
        endMarkerError = std::move(error);
      });
  noMoreData(taskId);
  EXPECT_TRUE(endMarkerFired);
  EXPECT_EQ(endMarkerError, std::nullopt);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  bool lateFired = false;
  std::optional<std::string> lateError{"not set"};
  queueManager_->getData(
      taskId,
      0,
      [&lateFired, &lateError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        lateFired = true;
        lateError = std::move(error);
      });
  EXPECT_TRUE(lateFired);
  EXPECT_EQ(lateError, std::nullopt);
}

// Removing a task drops whatever its queues still hold. A producer that queued
// all its output but is aborted before its consumers fetched it has therefore
// not delivered it, and its consumers must fail rather than end cleanly.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskWithUndeliveredOutputReportsProducerError) {
  const std::string taskId = "queuedThenAborted";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  bool fired = false;
  bool receivedNullptr = false;
  std::optional<std::string> producerError;
  queueManager_->getData(
      taskId,
      0,
      [&fired, &receivedNullptr, &producerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        fired = true;
        receivedNullptr = (data == nullptr);
        producerError = std::move(error);
      });
  EXPECT_TRUE(fired);
  EXPECT_TRUE(receivedNullptr);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before its consumers received all its output")));
}

// A producer that discards its own queued output before it finished, as
// UcxPartitionedOutput does when it runs out of GPU memory, must end every
// waiting and later server's stream with an error. A clean end would book the
// producer as finished while its rows are gone.
TEST_F(
    UcxOutputQueueManagerTest,
    deleteResultsBeforeAllOutputReportsProducerError) {
  const std::string taskId = "discardedBeforeEnd";
  auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
  enqueue(taskId, /*destination=*/1, /*size=*/100);

  // A server waits on destination 0, which is empty.
  bool waitingFired = false;
  std::optional<std::string> waitingError;
  queueManager_->getData(
      taskId,
      0,
      [&waitingFired, &waitingError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        waitingFired = true;
        waitingError = std::move(error);
      });
  EXPECT_FALSE(waitingFired);

  deleteResults(taskId, 0);
  deleteResults(taskId, 1);

  EXPECT_TRUE(waitingFired);
  EXPECT_THAT(
      waitingError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task discarded its output before sending all of it")));

  // A server that arrives after the discard gets the same error.
  bool lateFired = false;
  std::optional<std::string> lateError;
  queueManager_->getData(
      taskId,
      1,
      [&lateFired, &lateError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        lateFired = true;
        lateError = std::move(error);
      });
  EXPECT_TRUE(lateFired);
  EXPECT_THAT(
      lateError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task discarded its output before sending all of it")));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// An intra-node server takes a page from the producer's queue and later
// publishes it to IntraNodeTransferRegistry, which drops the page when the
// producer is removed. The page must count as undelivered from the moment it
// leaves the queue: removing the producer before the server published it must
// fail its consumers, even though its queues are empty and every driver
// finished. Covers a page that was queued before the server asked, which is
// handed over at once, and one that arrives later, which enqueue() hands over.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskDuringIntraNodeHandoffReportsProducerError) {
  for (const bool queuedBeforeRequest : {true, false}) {
    SCOPED_TRACE(queuedBeforeRequest ? "immediate" : "deferred");
    const std::string taskId = queuedBeforeRequest ? "intraNodeHandoffImmediate"
                                                   : "intraNodeHandoffDeferred";
    auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
    if (queuedBeforeRequest) {
      enqueue(taskId, /*size=*/100);
      noMoreData(taskId);
    }

    bool taken = false;
    queueManager_->getIntraNodeData(
        taskId,
        0,
        [&taken](
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> /*remainingBytes*/,
            std::optional<std::string> /*error*/) {
          EXPECT_NE(data, nullptr);
          taken = true;
        });
    if (!queuedBeforeRequest) {
      enqueue(taskId, /*size=*/100);
      noMoreData(taskId);
    }
    ASSERT_TRUE(taken);

    // The server has the page and has not published it yet.
    task->requestAbort().wait();
    queueManager_->removeTask(taskId);

    bool fired = false;
    std::optional<std::string> producerError;
    queueManager_->getData(
        taskId,
        0,
        [&fired, &producerError](
            std::shared_ptr<cudf::packed_columns> data,
            vector_size_t /*numRows*/,
            std::vector<int64_t> /*remainingBytes*/,
            std::optional<std::string> error) {
          EXPECT_EQ(data, nullptr);
          fired = true;
          producerError = std::move(error);
        });
    EXPECT_TRUE(fired);
    EXPECT_THAT(
        producerError,
        testing::Optional(
            testing::HasSubstr(
                "Producer task ended before its consumers received all its output")));
  }
}

// Once the consumer read the page, the producer's output is delivered, and
// removing it ends the stream cleanly.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskAfterIntraNodePageReadKeepsCleanEndMarker) {
  const std::string taskId = "intraNodePageRead";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);
  queueManager_->getIntraNodeData(
      taskId,
      0,
      [](std::shared_ptr<cudf::packed_columns> data,
         vector_size_t /*numRows*/,
         std::vector<int64_t> /*remainingBytes*/,
         std::optional<std::string> /*error*/) { EXPECT_NE(data, nullptr); });
  queueManager_->intraNodePageRetrieved(taskId, /*destination=*/0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  bool fired = false;
  std::optional<std::string> producerError{"not set"};
  queueManager_->getData(
      taskId,
      0,
      [&fired, &producerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        fired = true;
        producerError = std::move(error);
      });
  EXPECT_TRUE(fired);
  EXPECT_EQ(producerError, std::nullopt);
}

// The consumer owns a page once it polled it from IntraNodeTransferRegistry.
// Its server learns of the read only on a later Communicator turn, so the
// producer can be removed in between. That removal must still end the stream
// cleanly: the consumer has all the producer's output.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskBetweenIntraNodePollAndAcknowledgeKeepsCleanEndMarker) {
  const std::string taskId = "intraNodePolledNotAcknowledged";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);

  std::shared_ptr<cudf::packed_columns> page;
  queueManager_->getIntraNodeData(
      taskId,
      0,
      [&page](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> /*error*/) { page = std::move(data); });
  ASSERT_NE(page, nullptr);

  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey pageKey{taskId, /*destination=*/0, 0};
  auto retrieved = registry->publish(
      pageKey,
      std::move(page),
      /*numRows=*/100,
      /*atEnd=*/false,
      /*producerError=*/std::nullopt,
      countIntraNodePageRead(taskId, /*destination=*/0));
  const auto pageRead = registry->poll(pageKey);
  ASSERT_TRUE(pageRead.has_value());
  ASSERT_NE(pageRead->data, nullptr);
  ASSERT_EQ(
      retrieved.wait_for(std::chrono::seconds(0)), std::future_status::ready);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  const auto endMarker = registry->poll({taskId, /*destination=*/0, 1});
  ASSERT_TRUE(endMarker.has_value());
  EXPECT_TRUE(endMarker->atEnd);
  EXPECT_EQ(endMarker->producerError, std::nullopt);
  registry->clearCancelledTask(taskId);
}

// A page its server published but its consumer has not read yet is dropped
// with the producer, so that consumer must fail rather than end cleanly.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskBeforeIntraNodePollReportsProducerError) {
  const std::string taskId = "intraNodePublishedNotPolled";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);

  std::shared_ptr<cudf::packed_columns> page;
  queueManager_->getIntraNodeData(
      taskId,
      0,
      [&page](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> /*error*/) { page = std::move(data); });
  ASSERT_NE(page, nullptr);

  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey pageKey{taskId, /*destination=*/0, 0};
  auto retrieved = registry->publish(
      pageKey,
      std::move(page),
      /*numRows=*/100,
      /*atEnd=*/false,
      /*producerError=*/std::nullopt,
      countIntraNodePageRead(taskId, /*destination=*/0));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  // The server does not wait for a read that will not come.
  EXPECT_EQ(
      retrieved.wait_for(std::chrono::seconds(0)), std::future_status::ready);
  const auto result = registry->poll(pageKey);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_THAT(
      result->producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before its consumers received all its output")));
  registry->clearCancelledTask(taskId);
}

// A source can poll while the removal of its producer decides whether the
// consumers read all its output. Once that decision counted a page as unread,
// the source must not get the page and then an error: it gets only the
// producer's error. The poll here runs from a server callback on destination 1,
// which removeTask() fires after the decision and before it cancels the task
// in IntraNodeTransferRegistry.
TEST_F(
    UcxOutputQueueManagerTest,
    intraNodePollDuringRemoveTaskReportsProducerError) {
  const std::string taskId = "intraNodePolledDuringRemoval";
  auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
  enqueue(taskId, /*destination=*/0, /*size=*/100);
  noMoreData(taskId);

  std::shared_ptr<cudf::packed_columns> page;
  queueManager_->getIntraNodeData(
      taskId,
      0,
      [&page](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> /*error*/) { page = std::move(data); });
  ASSERT_NE(page, nullptr);

  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey pageKey{taskId, /*destination=*/0, 0};
  auto retrieved = registry->publish(
      pageKey,
      std::move(page),
      /*numRows=*/100,
      /*atEnd=*/false,
      /*producerError=*/std::nullopt,
      countIntraNodePageRead(taskId, /*destination=*/0));

  // Destination 1 is empty after its end marker, so this request waits until
  // removeTask() ends the stream.
  fetch(taskId, /*destination=*/1, /*expectedEndMarker=*/true);
  bool polledDuringRemoval = false;
  std::optional<IntraNodeTransferResult> readDuringRemoval;
  queueManager_->getData(
      taskId,
      1,
      [&](std::shared_ptr<cudf::packed_columns> /*data*/,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> /*error*/) {
        polledDuringRemoval = true;
        readDuringRemoval = registry->poll(pageKey);
      });

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  ASSERT_TRUE(polledDuringRemoval);
  EXPECT_FALSE(readDuringRemoval.has_value())
      << "The source read a page that the removal counted as lost";
  EXPECT_EQ(
      retrieved.wait_for(std::chrono::seconds(0)), std::future_status::ready);
  const auto result = registry->poll(pageKey);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_THAT(
      result->producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before its consumers received all its output")));
  registry->clearCancelledTask(taskId);
}

// A broadcast producer keeps its pages for destinations that have not arrived
// yet. Removing it before the last destination arrived drops those pages, so
// a destination that arrives later must not see a clean, empty stream.
TEST_F(
    UcxOutputQueueManagerTest,
    removeBroadcastBeforeAllDestinationsReportsProducerError) {
  const std::string taskId = "broadcastWithBacklog";
  auto task = initializeTask(
      taskId,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      /*cleanup=*/true,
      core::PartitionedOutputNode::Kind::kBroadcast);
  enqueue(taskId, 0, 50);
  noMoreData(taskId);
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  bool fired = false;
  std::optional<std::string> producerError;
  queueManager_->getData(
      taskId,
      /*destination=*/1,
      [&fired, &producerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        fired = true;
        producerError = std::move(error);
      });
  EXPECT_TRUE(fired);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before its consumers received all its output")));
}

// A consumer that connects only after its failed producer was removed must
// learn the reason too, rather than read an empty, complete stream.
TEST_F(
    UcxOutputQueueManagerTest,
    lateGetDataAfterFailedRemovalReportsProducerError) {
  const std::string taskId = "abortedThenFetched";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  bool fired = false;
  bool receivedNullptr = false;
  std::optional<std::string> producerError;
  queueManager_->getData(
      taskId,
      0,
      [&fired, &receivedNullptr, &producerError](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t /*numRows*/,
          std::vector<int64_t> /*remainingBytes*/,
          std::optional<std::string> error) {
        fired = true;
        receivedNullptr = (data == nullptr);
        producerError = std::move(error);
      });
  EXPECT_TRUE(fired);
  EXPECT_TRUE(receivedNullptr);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task ended before sending all its output")));
}

TEST_F(UcxOutputQueueManagerTest, terminateUnpublishedReader) {
  const std::string taskId = "terminatedPendingReader";
  auto task = initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  auto queue = std::make_shared<UcxOutputQueue>(
      task, 1, 1, core::PartitionedOutputNode::Kind::kBroadcast);
  int endCount{0};
  queue->getData(
      5,
      [&](std::shared_ptr<cudf::packed_columns> data,
          vector_size_t,
          std::vector<int64_t>,
          std::optional<std::string>) {
        EXPECT_EQ(data, nullptr);
        // A racing reader must not register another pending callback.
        bool receivedEnd{false};
        queue->getData(10, receiveEndMarker(10, receivedEnd));
        EXPECT_TRUE(receivedEnd);
        ++endCount;
      });
  EXPECT_EQ(endCount, 0);
  queue->terminate(std::nullopt);
  queue->terminate(std::nullopt);
  EXPECT_EQ(endCount, 1);
  queue->updateOutputBuffers(3, true);
  EXPECT_FALSE(queue->stats().noMoreBuffers);
  bool receivedEnd{false};
  queue->getData(0, receiveEndMarker(0, receivedEnd));
  EXPECT_TRUE(receivedEnd);
  queueManager_->removeTask(taskId);
}

// A reader that waits for a destination the producer has not published yet
// must learn why the producer ended early, as a reader of a published
// destination does.
TEST_F(
    UcxOutputQueueManagerTest,
    terminateReportsProducerErrorToUnpublishedReader) {
  const std::string taskId = "terminatedPendingReaderWithError";
  auto task = initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  auto queue = std::make_shared<UcxOutputQueue>(
      task, 1, 1, core::PartitionedOutputNode::Kind::kBroadcast);
  const std::string reason{"Injected producer failure"};

  int endCount{0};
  std::optional<std::string> producerError;
  queue->getData(
      5,
      [&](std::shared_ptr<cudf::packed_columns> data,
          vector_size_t,
          std::vector<int64_t>,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        producerError = std::move(error);
        ++endCount;
      });
  EXPECT_EQ(endCount, 0);

  queue->terminate(reason);
  EXPECT_EQ(endCount, 1);
  EXPECT_EQ(producerError, reason);
  queueManager_->removeTask(taskId);
}

// A reader that arrives after the producer was removed gets the end marker the
// producer was removed with: its error if it ended early, a clean end
// otherwise.
TEST_F(UcxOutputQueueManagerTest, getDataAfterTerminateReportsProducerError) {
  for (const auto& reason :
       {std::optional<std::string>{"Injected producer failure"},
        std::optional<std::string>{}}) {
    SCOPED_TRACE(reason.value_or("clean end"));
    const std::string taskId = "getDataAfterTerminate";
    auto task = initializeTask(taskId, 1, 1);
    auto queue = std::make_shared<UcxOutputQueue>(
        task, 1, 1, core::PartitionedOutputNode::Kind::kPartitioned);
    queue->terminate(reason);

    bool fired{false};
    std::optional<std::string> producerError{"not set"};
    queue->getData(
        0,
        [&](std::shared_ptr<cudf::packed_columns> data,
            vector_size_t,
            std::vector<int64_t>,
            std::optional<std::string> error) {
          EXPECT_EQ(data, nullptr);
          producerError = std::move(error);
          fired = true;
        });
    EXPECT_TRUE(fired);
    EXPECT_EQ(producerError, reason);
    queueManager_->removeTask(taskId);
  }
}

// --- Broadcast tests ---

// Basic broadcast: enqueue data, all destinations receive the same data.
TEST_F(UcxOutputQueueManagerTest, broadcastBasic) {
  const vector_size_t size = 100;
  const std::string taskId = "broadcast0";
  const int numDestinations = 3;

  auto task = initializeTask(
      taskId,
      numDestinations,
      1 /* numDrivers */,
      true /* cleanup */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  // Broadcast must not be finished before noMoreQueues.
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  // Tell the queue manager the final number of destinations.
  queueManager_->updateOutputBuffers(taskId, numDestinations, true);

  // Enqueue one batch (broadcast always uses destination 0).
  enqueue(taskId, 0, size);

  // All destinations should receive the data.
  for (int dest = 0; dest < numDestinations; ++dest) {
    fetch(taskId, dest);
  }

  // Signal no more data.
  noMoreData(taskId);

  // Fetch end markers from all destinations.
  for (int dest = 0; dest < numDestinations; ++dest) {
    fetchEndMarker(taskId, dest);
  }

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: late destination receives backfilled data.
TEST_F(UcxOutputQueueManagerTest, broadcastLateDestination) {
  const vector_size_t size = 50;
  const std::string taskId = "broadcast1";

  // Start with 2 destinations.
  auto task = initializeTask(
      taskId,
      2 /* numDestinations */,
      1 /* numDrivers */,
      true /* cleanup */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  // Enqueue 2 batches before the 3rd destination arrives.
  enqueue(taskId, 0, size);
  enqueue(taskId, 0, size);

  // Fetch from existing destinations to verify data is there.
  fetch(taskId, 0);
  fetch(taskId, 1);

  // Now add a 3rd destination (late arrival). It should be backfilled.
  queueManager_->updateOutputBuffers(taskId, 3, false);

  // The new destination should have both batches.
  fetch(taskId, 2);
  fetch(taskId, 2);

  // Finalize destinations and data.
  queueManager_->updateOutputBuffers(taskId, 3, true);
  noMoreData(taskId);

  // Fetch remaining data + end markers.
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);
  fetch(taskId, 1);
  fetchEndMarker(taskId, 1);
  fetchEndMarker(taskId, 2);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, broadcastReaderArrivesBeforeBufferUpdate) {
  for (bool hasData : {false, true}) {
    for (bool atEnd : {false, true}) {
      SCOPED_TRACE(fmt::format("hasData={} atEnd={}", hasData, atEnd));
      const std::string taskId = "broadcastPendingReader";
      initializeTask(
          taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
      if (hasData) {
        enqueue(taskId, 0, 10);
      }
      if (atEnd) {
        noMoreData(taskId);
      }
      int dataCount{0};
      int endCount{0};
      auto receive = [&](std::shared_ptr<cudf::packed_columns> data,
                         vector_size_t rows,
                         std::vector<int64_t>,
                         std::optional<std::string>) {
        // Re-enter the manager and queue to detect notifications under locks.
        EXPECT_TRUE(queueManager_->stats(taskId).has_value());
        if (data) {
          ++dataCount;
          EXPECT_EQ(rows, 10);
        } else {
          ++endCount;
          EXPECT_EQ(rows, 0);
        }
      };
      const auto bytes = queueManager_->stats(taskId)->bufferedBytes;
      queueManager_->getData(taskId, 1, receive);
      EXPECT_EQ(dataCount, 0);
      EXPECT_EQ(endCount, 0);
      EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, bytes);

      queueManager_->updateOutputBuffers(taskId, 2, false);
      EXPECT_EQ(dataCount, hasData ? 1 : 0);
      EXPECT_EQ(endCount, !hasData && atEnd ? 1 : 0);
      if (hasData) {
        queueManager_->getData(taskId, 1, receive);
      }
      if (!atEnd) {
        EXPECT_EQ(endCount, 0);
        noMoreData(taskId);
      }
      EXPECT_EQ(endCount, 1);
      queueManager_->updateOutputBuffers(taskId, 2, true);
      queueManager_->deleteResults(taskId, 0);
      queueManager_->deleteResults(taskId, 1);
      EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
      EXPECT_TRUE(queueManager_->isFinished(taskId));
      queueManager_->removeTask(taskId);
    }
  }
}

TEST_F(UcxOutputQueueManagerTest, broadcastLargeDestinationBothArrivalOrders) {
  for (bool readerFirst : {false, true}) {
    SCOPED_TRACE(readerFirst);
    const std::string taskId = "broadcastLargeDestination";
    constexpr int kDestination = 4096;
    initializeTask(
        taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
    enqueue(taskId, 0, 10);
    noMoreData(taskId);
    bool receivedData{false};
    if (readerFirst) {
      queueManager_->getData(
          taskId, kDestination, receiveData(kDestination, receivedData));
      EXPECT_FALSE(receivedData);
      queueManager_->updateOutputBuffers(taskId, 2, false);
      EXPECT_FALSE(receivedData);
    }
    queueManager_->updateOutputBuffers(taskId, kDestination + 1, true);
    if (!readerFirst) {
      queueManager_->getData(
          taskId, kDestination, receiveData(kDestination, receivedData));
    }
    EXPECT_TRUE(receivedData);
    fetchEndMarker(taskId, kDestination);
    for (int destination = 0; destination < kDestination; ++destination) {
      queueManager_->deleteResults(taskId, destination);
    }
    EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
    EXPECT_TRUE(queueManager_->isFinished(taskId));
    queueManager_->removeTask(taskId);
  }
}

TEST_F(UcxOutputQueueManagerTest, deleteUnpublishedReader) {
  const std::string taskId = "deletedPendingReader";
  initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  enqueue(taskId, 0, 10);
  bool receivedEnd{false};
  queueManager_->getData(taskId, 1, receiveEndMarker(1, receivedEnd));
  EXPECT_FALSE(receivedEnd);
  queueManager_->deleteResults(taskId, 1);
  EXPECT_TRUE(receivedEnd);
  const auto bytes = queueManager_->stats(taskId)->bufferedBytes;
  queueManager_->updateOutputBuffers(taskId, 2, true);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, bytes);
  bool retryEnd{false};
  queueManager_->getData(taskId, 1, receiveEndMarker(1, retryEnd));
  EXPECT_TRUE(retryEnd);
  noMoreData(taskId);
  queueManager_->deleteResults(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Results of a destination that is not published yet are deleted before every
// producing driver finished only when the producer discards its own output, as
// for a published destination. The reader waiting for that destination must
// fail rather than end cleanly.
TEST_F(
    UcxOutputQueueManagerTest,
    deleteUnpublishedReaderBeforeEndReportsProducerError) {
  const std::string taskId = "deletedPendingReaderBeforeEnd";
  initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  enqueue(taskId, 0, 10);

  bool fired{false};
  std::optional<std::string> producerError;
  queueManager_->getData(
      taskId,
      1,
      [&](std::shared_ptr<cudf::packed_columns> data,
          vector_size_t,
          std::vector<int64_t>,
          std::optional<std::string> error) {
        EXPECT_EQ(data, nullptr);
        producerError = std::move(error);
        fired = true;
      });
  EXPECT_FALSE(fired);

  queueManager_->deleteResults(taskId, 1);
  EXPECT_TRUE(fired);
  EXPECT_THAT(
      producerError,
      testing::Optional(
          testing::HasSubstr(
              "Producer task discarded its output before sending all of it")));

  queueManager_->updateOutputBuffers(taskId, 2, true);
  noMoreData(taskId);
  queueManager_->deleteResults(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, deleteUnpublishedDestinationBeforeGetData) {
  const std::string taskId = "deletedUnpublishedDestination";
  initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  enqueue(taskId, 0, 10);
  const auto pageBytes = queueManager_->stats(taskId)->bufferedBytes;
  queueManager_->deleteResults(taskId, -1);
  queueManager_->deleteResults(taskId, 2);
  bool receivedEnd{false};
  queueManager_->getData(taskId, 2, receiveEndMarker(2, receivedEnd));
  EXPECT_TRUE(receivedEnd);
  queueManager_->updateOutputBuffers(taskId, 3, true);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 2 * pageBytes);
  noMoreData(taskId);
  for (int destination : {0, 1}) {
    fetch(taskId, destination);
    fetchEndMarker(taskId, destination);
  }
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// Broadcast: isFinished requires noMoreQueues.
TEST_F(UcxOutputQueueManagerTest, broadcastNotFinishedWithoutNoMoreQueues) {
  const std::string taskId = "broadcast2";

  auto task = initializeTask(
      taskId,
      2 /* numDestinations */,
      1 /* numDrivers */,
      true /* cleanup */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  // Signal no more data.
  noMoreData(taskId);

  // Fetch end markers and delete results from all destinations.
  fetchEndMarker(taskId, 0);
  fetchEndMarker(taskId, 1);

  // Still not finished because noMoreQueues hasn't been signaled.
  EXPECT_FALSE(queueManager_->isFinished(taskId));

  // Signal no more queues.
  queueManager_->updateOutputBuffers(taskId, 2, true);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, broadcastEndsReaderAfterFinalBuffers) {
  const std::string taskId = "broadcastReaderAfterFinalBuffers";
  initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  queueManager_->updateOutputBuffers(taskId, 1, true);
  bool receivedEnd{false};
  queueManager_->getData(taskId, 1, receiveEndMarker(1, receivedEnd));
  EXPECT_TRUE(receivedEnd);
  EXPECT_TRUE(queueManager_->updateOutputBuffers(taskId, 1, true));
  VELOX_ASSERT_THROW(
      queueManager_->updateOutputBuffers(taskId, 2, true),
      "Cannot add broadcast destinations after no more output buffers");
  queueManager_->deleteResults(taskId, 1);
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, unpublishedReadersDoNotAmplifyBackpressure) {
  const std::string taskId = "broadcastSparseReaders";
  auto page = makePackedColumns(10);
  const auto pageBytes = page->gpu_data->size();
  ASSERT_GT(pageBytes, 0);
  // One queued page fits; a phantom destination would block the producer.
  auto task = initializeTask(
      taskId,
      1,
      1,
      true,
      core::PartitionedOutputNode::Kind::kBroadcast,
      pageBytes * 2);
  int endCount{0};
  for (int destination : {5, 4095, std::numeric_limits<int>::max()}) {
    queueManager_->getData(
        taskId,
        destination,
        [&](std::shared_ptr<cudf::packed_columns> data,
            vector_size_t rows,
            std::vector<int64_t> remaining,
            std::optional<std::string>) {
          EXPECT_EQ(data, nullptr);
          EXPECT_EQ(rows, 0);
          EXPECT_TRUE(remaining.empty());
          EXPECT_TRUE(queueManager_->stats(taskId).has_value());
          ++endCount;
        });
  }
  EXPECT_EQ(endCount, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
  queueManager_->updateOutputBuffers(taskId, 1, false);
  EXPECT_EQ(endCount, 0);
  queueManager_->enqueue(taskId, 0, std::move(page), 10);
  EXPECT_EQ(endCount, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, pageBytes);
  EXPECT_DOUBLE_EQ(*queueManager_->getUtilization(taskId), 0.5);
  ContinueFuture future;
  EXPECT_FALSE(queueManager_->checkBlocked(taskId, &future));
  noMoreData(taskId);
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);
  EXPECT_EQ(endCount, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
  EXPECT_FALSE(task->isFinished());
  queueManager_->updateOutputBuffers(taskId, 1, true);
  EXPECT_EQ(endCount, 3);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  EXPECT_TRUE(task->isFinished());
  queueManager_->removeTask(taskId);
  EXPECT_EQ(endCount, 3);
}

// Broadcast: end marker propagated to late-arriving destinations.
TEST_F(UcxOutputQueueManagerTest, broadcastEndMarkerToLateDestination) {
  const std::string taskId = "broadcast3";

  auto task = initializeTask(
      taskId,
      1 /* numDestinations */,
      1 /* numDrivers */,
      true /* cleanup */,
      core::PartitionedOutputNode::Kind::kBroadcast);

  enqueue(taskId, 0, 10);
  noMoreData(taskId);

  // Add a late destination after end marker was set.
  queueManager_->updateOutputBuffers(taskId, 2, true);

  // Destination 0: data + end marker.
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);

  // Destination 1 (late): backfilled data + end marker.
  fetch(taskId, 1);
  fetchEndMarker(taskId, 1);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}
