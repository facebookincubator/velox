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
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <array>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryPool.h"
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
        [destination, expectedEndMarker, &receivedData](ReaderStep step) {
          ASSERT_EQ(expectedEndMarker, isCleanEnd(step))
              << "for destination " << destination;
          receivedData = true;
        });
    ASSERT_TRUE(receivedData) << "for destination " << destination;
  }

  static bool isCleanEnd(const ReaderStep& step) {
    return step.kind == ReaderStep::Kind::kEnd && step.outcome.isClean();
  }

  ReaderCallback receiveEndMarker(int destination, bool& receivedEndMarker) {
    return [destination, &receivedEndMarker](ReaderStep step) {
      EXPECT_FALSE(receivedEndMarker) << "for destination " << destination;
      EXPECT_TRUE(isCleanEnd(step)) << "for destination " << destination;
      EXPECT_TRUE(step.remainingBytes.empty());
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

  void registerForEndMarker(
      std::string_view taskId,
      int destination,
      bool& receivedEndMarker) {
    receivedEndMarker = false;
    queueManager_->getData(
        taskId, destination, receiveEndMarker(destination, receivedEndMarker));
    EXPECT_FALSE(receivedEndMarker) << "for destination " << destination;
  }

  ReaderCallback receiveData(int destination, bool& receivedData) {
    receivedData = false;
    return [destination, &receivedData](ReaderStep step) {
      EXPECT_FALSE(receivedData) << "for destination " << destination;
      EXPECT_EQ(step.kind, ReaderStep::Kind::kChunk)
          << "for destination " << destination;
      EXPECT_NE(step.data, nullptr) << "for destination " << destination;
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
      queueManager_->getData(taskId, destination, [&](ReaderStep step) {
        if (step.kind == ReaderStep::Kind::kEnd) {
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
      folly::Promise<ReaderStep> promise;
      auto future = promise.getSemiFuture();
      queueManager_->getData(taskId, destination, [&promise](ReaderStep step) {
        promise.setValue(std::move(step));
      });
      future.wait();
      ASSERT_TRUE(future.isReady());
      auto& response = future.value();
      ASSERT_EQ(response.remainingBytes.size(), 0);
      ASSERT_TRUE(isCleanEnd(response));
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

TEST_F(UcxOutputQueueManagerTest, endStepIsClean) {
  const std::string taskId = "endStep";
  initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, 0, 10);
  noMoreData(taskId);

  std::vector<ReaderStep> steps;
  // The queue holds a chunk and the end marker: stop after three reads, so that
  // a regression fails instead of hanging.
  while (steps.empty() || steps.back().kind == ReaderStep::Kind::kChunk) {
    ASSERT_LT(steps.size(), 3);
    queueManager_->getData(taskId, 0, [&steps](ReaderStep step) {
      steps.push_back(std::move(step));
    });
  }
  ASSERT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0].kind, ReaderStep::Kind::kChunk);
  EXPECT_NE(steps[0].data, nullptr);
  EXPECT_EQ(steps[0].numRows, 10);
  EXPECT_EQ(steps[1].kind, ReaderStep::Kind::kEnd);
  EXPECT_EQ(steps[1].outcome, ChannelOutcome::clean());
  EXPECT_EQ(steps[1].data, nullptr);
  EXPECT_EQ(steps[1].numRows, 0);
  EXPECT_TRUE(steps[1].remainingBytes.empty());
  queueManager_->deleteResults(taskId, 0);
  queueManager_->removeTask(taskId);
}

TEST_F(UcxOutputQueueManagerTest, lateTaskCreation) {
  const vector_size_t size = 10;
  const std::string taskId = "t0";
  int numPartitions = 1;
  int destination = 0;

  // A manager with no state from prior tests, so that getData() below creates
  // a Pending entry for a task it has not seen.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  // Fetch data from a non-existing task.
  folly::Promise<ReaderStep> promise;
  auto future = promise.getSemiFuture();
  queueManager_->getData(taskId, destination, [&promise](ReaderStep step) {
    promise.setValue(std::move(step));
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
    ReaderStep& response = future.value();
    if (response.kind == ReaderStep::Kind::kEnd) {
      // End of transmission.
      EXPECT_TRUE(response.outcome.isClean());
      break;
    }
    folly::Promise<ReaderStep> promise;
    future = promise.getSemiFuture();
    queueManager_->getData(taskId, destination, [&promise](ReaderStep step) {
      promise.setValue(std::move(step));
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
    // A manager that has not seen the task, so getData() below leaves Pending
    // readers.
    queueManager_ = std::make_shared<UcxOutputQueueManager>();
    bool receivedEnd{false};
    queueManager_->getData(taskId, 3, receiveEndMarker(3, receivedEnd));
    EXPECT_FALSE(receivedEnd);
    bool receivedData{false};
    queueManager_->getData(taskId, 0, receiveData(0, receivedData));
    int endCount{0};
    queueManager_->getData(
        taskId, std::numeric_limits<int>::max(), [&](ReaderStep step) {
          EXPECT_TRUE(isCleanEnd(step));
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

// A server that closes while it waits on a Pending entry leaves nothing that
// would ever erase the entry if its producer never arrives. Releasing the last
// reader erases it, and the task can still be initialized later.
TEST_F(UcxOutputQueueManagerTest, releasingLastReaderDropsPendingEntry) {
  const std::string taskId = "pendingEntryWithClosedReaders";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  bool fired{false};
  auto receive = [&fired](ReaderStep /*step*/) { fired = true; };
  auto reader0 = queueManager_->bindReader(taskId, 0);
  reader0->getData(0, receive);
  auto reader1 = queueManager_->bindReader(taskId, 1);
  reader1->getData(1, receive);
  ASSERT_EQ(reader0, reader1);
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());

  queueManager_->releaseReader(reader0, 0);
  EXPECT_TRUE(queueManager_->stats(taskId).has_value());
  queueManager_->releaseReader(reader1, 1);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  EXPECT_FALSE(fired);

  auto task = initializeTask(
      taskId, /*numDestinations=*/1, /*numDrivers=*/1, /*cleanup=*/false);
  enqueue(taskId, /*size=*/10);
  noMoreData(taskId);
  fetch(taskId, 0);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  EXPECT_FALSE(fired);
  queueManager_->removeTask(taskId);
}

// Releasing one reader of a Pending entry keeps the entry for the readers that
// are still bound, so that the producer finds them when it initializes.
TEST_F(UcxOutputQueueManagerTest, boundReaderKeepsPendingEntry) {
  const std::string taskId = "pendingEntryWithBoundReader";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  bool receivedData{false};
  auto reader0 = queueManager_->bindReader(taskId, 0);
  reader0->getData(0, receiveData(0, receivedData));
  auto reader1 = queueManager_->bindReader(taskId, 1);
  queueManager_->releaseReader(reader1, 1);
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  EXPECT_EQ(reader0->state(), TaskEntryState::kPending);

  auto task = initializeTask(
      taskId, /*numDestinations=*/1, /*numDrivers=*/1, /*cleanup=*/false);
  EXPECT_EQ(reader0->state(), TaskEntryState::kActive);
  enqueue(taskId, /*size=*/10);
  EXPECT_TRUE(receivedData);
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->releaseReader(reader0, 0);
  queueManager_->removeTask(taskId);
}

// A caller that holds a Pending entry for another reason must not keep it in
// the map once its last reader is released: nothing would erase it later.
TEST_F(UcxOutputQueueManagerTest, pendingEntryHeldElsewhereIsStillErased) {
  const std::string taskId = "pendingEntryHeldElsewhere";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto held = queueManager_->bindReader(taskId, 0);
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  queueManager_->releaseReader(held, 0);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  EXPECT_EQ(held->state(), TaskEntryState::kPending);

  // The next reader gets a new entry, not the one still held.
  auto next = queueManager_->bindReader(taskId, 0);
  EXPECT_NE(next, held);
  queueManager_->releaseReader(next, 0);
}

// A reader that arrives after a removal is bound to the Removed entry, which
// ends it at once, rather than to a new Pending entry that no producer fills.
TEST_F(UcxOutputQueueManagerTest, readerBoundDuringRemovalFindsRemovedEntry) {
  const std::string taskId = "readerAfterRemoval";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  queueManager_->removeTask(taskId);
  auto entry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(entry->state(), TaskEntryState::kRemoved);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);

  bool receivedEnd{false};
  entry->getData(0, receiveEndMarker(0, receivedEnd));
  EXPECT_TRUE(receivedEnd);
  queueManager_->releaseReader(entry, 0);
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
}

// The methods Velox calls see a Removed task as absent.
TEST_F(UcxOutputQueueManagerTest, removedTaskIsInvisibleToVelox) {
  const std::string taskId = "removedInvisibleToVelox";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  EXPECT_EQ(queueManager_->getUtilization(taskId), std::nullopt);
  EXPECT_EQ(queueManager_->isOverutilized(taskId), std::nullopt);
  EXPECT_FALSE(queueManager_->updateOutputBuffers(taskId, 1, true));
  EXPECT_FALSE(queueManager_->updateNumDrivers(taskId, 1));
  EXPECT_FALSE(queueManager_->canUseIntraNode(taskId));
  VELOX_ASSERT_THROW(
      queueManager_->isFinished(taskId),
      "Output cudf queue for task not found");
  VELOX_ASSERT_THROW(
      queueManager_->noMoreData(taskId),
      "Output cudf queue for task not found");
}

// A server keeps the entry it was bound to. When the task ID is initialized
// again, the old server's calls must not reach the new task.
TEST_F(UcxOutputQueueManagerTest, queueHandleDoesNotFollowReusedTaskId) {
  const std::string taskId = "reusedTaskId";
  auto oldTask = initializeTask(taskId, 1, 1);
  auto oldEntry = queueManager_->bindReader(taskId, /*destination=*/0);

  oldTask->requestAbort().wait();
  queueManager_->removeTask(taskId);
  auto newTask = initializeTask(taskId, 1, 1);

  oldEntry->deleteResults(0);

  EXPECT_EQ(oldTask->state(), TaskState::kAborted);
  EXPECT_EQ(newTask->state(), TaskState::kRunning);
  EXPECT_FALSE(queueManager_->isFinished(taskId));
  EXPECT_EQ(oldEntry->state(), TaskEntryState::kRemoved);
  queueManager_->releaseReader(oldEntry, 0);
  newTask->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// A Removed entry lives as long as its Task, so that a consumer that connects
// late still finds the removal. The next removal after the Task is destroyed
// erases it.
TEST_F(UcxOutputQueueManagerTest, removedTaskRecordLivesAsLongAsItsTask) {
  const std::string taskId = "recordedWhileTaskLives";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  // Other removals keep the entry while the Task is alive.
  queueManager_->removeTask("otherRemovedTask");
  auto entry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(entry->state(), TaskEntryState::kRemoved);
  queueManager_->releaseReader(entry, 0);

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterTaskDestroyed");

  // A consumer that arrives now is treated like one of a task not created yet.
  auto lateEntry = queueManager_->bindReader(taskId, 0);
  EXPECT_NE(lateEntry, entry);
  EXPECT_EQ(lateEntry->state(), TaskEntryState::kPending);
  queueManager_->releaseReader(lateEntry, 0);
  for (const auto& removedTaskId :
       {taskId,
        std::string{"otherRemovedTask"},
        std::string{"removalAfterTaskDestroyed"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
  }
}

// A server bound to a Removed entry keeps it, even after its Task is destroyed.
TEST_F(UcxOutputQueueManagerTest, boundReaderKeepsRemovedEntry) {
  const std::string taskId = "removedEntryWithBoundReader";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  queueManager_->removeTask("otherRemovedTask");
  auto entry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(entry->state(), TaskEntryState::kRemoved);

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterTaskDestroyed");

  auto sameEntry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(sameEntry, entry);
  EXPECT_EQ(sameEntry->state(), TaskEntryState::kRemoved);
  queueManager_->releaseReader(sameEntry, 0);
  queueManager_->releaseReader(entry, 0);
  for (const auto& removedTaskId :
       {taskId,
        std::string{"otherRemovedTask"},
        std::string{"removalAfterTaskDestroyed"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
  }
}

// A Removed entry outlives its removal while a server is bound to it or its
// Task lives, but its chunks are unreachable from then on. The removal frees
// them rather than keeping them for the entry's lifetime.
TEST_F(UcxOutputQueueManagerTest, removedEntryReleasesQueuedData) {
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  {
    SCOPED_TRACE("partitioned");
    const std::string taskId = "removedReleasesPartitionedData";
    auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
    auto entry = queueManager_->bindReader(taskId, 0);
    enqueue(taskId, 0, /*size=*/10);
    enqueue(taskId, 1, /*size=*/10);
    const auto before = queueManager_->stats(taskId);
    ASSERT_TRUE(before.has_value());
    ASSERT_GT(before->bufferedBytes, 0);
    ASSERT_EQ(before->bufferedPages, 2);

    task->requestAbort().wait();
    queueManager_->removeTask(taskId);
    const auto after = entry->stats();
    EXPECT_EQ(after.bufferedBytes, 0);
    EXPECT_EQ(after.bufferedPages, 0);
    EXPECT_EQ(after.totalBytesSent, before->totalBytesSent);
    EXPECT_EQ(after.totalPagesSent, 2);

    bool receivedEnd{false};
    entry->getData(0, receiveEndMarker(0, receivedEnd));
    EXPECT_TRUE(receivedEnd);
    queueManager_->releaseReader(entry, 0);
  }
  {
    SCOPED_TRACE("broadcast");
    // Destinations share each broadcast chunk, and the queue keeps it for
    // destinations that are not added yet, so a chunk that one destination
    // fetched lives until the others free it.
    const std::string taskId = "removedReleasesBroadcastData";
    auto task = initializeTask(
        taskId,
        /*numDestinations=*/2,
        /*numDrivers=*/1,
        /*cleanup=*/true,
        core::PartitionedOutputNode::Kind::kBroadcast);
    auto entry = queueManager_->bindReader(taskId, 0);
    enqueue(taskId, 0, /*size=*/10);
    std::weak_ptr<cudf::packed_columns> chunk;
    entry->getData(0, [&chunk](ReaderStep step) { chunk = step.data; });
    ASSERT_FALSE(chunk.expired());

    task->requestAbort().wait();
    queueManager_->removeTask(taskId);
    EXPECT_EQ(entry->state(), TaskEntryState::kRemoved);
    EXPECT_TRUE(chunk.expired());
    EXPECT_EQ(entry->stats().bufferedBytes, 0);
    queueManager_->releaseReader(entry, 0);
  }
}

// A task whose initializeTask() threw is removed with its entry still Pending,
// so only the failed initialization knew its Task. The Removed entry must still
// live as long as the Task rather than until the next removal.
TEST_F(
    UcxOutputQueueManagerTest,
    recordOfFailedInitializationLivesAsLongAsItsTask) {
  const std::string taskId = "failedInitialization";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto task = createSourceTask(taskId, pool_, UcxTestData::kTestRowType);
  VELOX_ASSERT_THROW(
      queueManager_->initializeTask(
          task,
          core::PartitionedOutputNode::Kind::kPartitioned,
          /*numDestinations=*/-1,
          /*numDrivers=*/1),
      "Number of destinations must be non-negative");
  // The failed initialization left the entry Pending and untouched: it has no
  // capacity yet and cannot use the intra-node path.
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  EXPECT_EQ(queueManager_->getUtilization(taskId), std::nullopt);
  EXPECT_FALSE(queueManager_->canUseIntraNode(taskId));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  queueManager_->removeTask("removalAfterFailedInitialization");
  auto entry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(entry->state(), TaskEntryState::kRemoved);
  queueManager_->releaseReader(entry, 0);

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterTaskDestroyed");
  auto lateEntry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(lateEntry->state(), TaskEntryState::kPending);
  queueManager_->releaseReader(lateEntry, 0);
  for (const auto& removedTaskId :
       {taskId,
        std::string{"removalAfterFailedInitialization"},
        std::string{"removalAfterTaskDestroyed"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
  }
}

// The entry keeps a failed initializeTask() even when its last server closes
// before the removal, so that the Removed entry lives as long as the Task.
TEST_F(UcxOutputQueueManagerTest, releasedReaderKeepsFailedInitialization) {
  const std::string taskId = "failedInitializationWithReader";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto entry = queueManager_->bindReader(taskId, 0);
  auto task = createSourceTask(taskId, pool_, UcxTestData::kTestRowType);
  VELOX_ASSERT_THROW(
      queueManager_->initializeTask(
          task,
          core::PartitionedOutputNode::Kind::kPartitioned,
          /*numDestinations=*/-1,
          /*numDrivers=*/1),
      "Number of destinations must be non-negative");
  queueManager_->releaseReader(entry, 0);

  auto pendingEntry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(pendingEntry, entry);
  EXPECT_EQ(pendingEntry->state(), TaskEntryState::kPending);
  queueManager_->releaseReader(pendingEntry, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  queueManager_->removeTask("removalAfterFailedInitialization");
  auto removedEntry = queueManager_->bindReader(taskId, 0);
  EXPECT_EQ(removedEntry, entry);
  EXPECT_EQ(removedEntry->state(), TaskEntryState::kRemoved);
  queueManager_->releaseReader(removedEntry, 0);
  for (const auto& removedTaskId :
       {taskId, std::string{"removalAfterFailedInitialization"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
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

TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateBeforeInit) {
  const std::string taskId = "orphanTest";
  // A manager that has not seen the task, so the reader below waits on a
  // Pending entry. A removal leaves a Removed entry that ends readers at once.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  bool receivedEndMarker{false};
  registerForEndMarker(taskId, 0, receivedEndMarker);

  queueManager_->removeTask(taskId);
  EXPECT_TRUE(receivedEndMarker);
}

TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateAfterInit) {
  const std::string taskId = "crashTest";
  constexpr int kNumDestinations = 2;
  auto task = initializeTask(taskId, kNumDestinations, 1);

  std::array<bool, kNumDestinations> receivedEndMarkers{};
  for (int destination = 0; destination < kNumDestinations; ++destination) {
    registerForEndMarker(taskId, destination, receivedEndMarkers[destination]);
  }

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  for (int destination = 0; destination < kNumDestinations; ++destination) {
    EXPECT_TRUE(receivedEndMarkers[destination])
        << "for destination " << destination;
  }
}

TEST_F(UcxOutputQueueManagerTest, terminateUnpublishedReader) {
  const std::string taskId = "terminatedPendingReader";
  auto task = initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  auto queue = std::make_shared<UcxOutputQueue>(taskId);
  std::vector<UcxDataAvailable> notifications;
  ASSERT_TRUE(queue->initialize(
      task,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      core::PartitionedOutputNode::Kind::kBroadcast,
      notifications));
  EXPECT_TRUE(notifications.empty());
  int endCount{0};
  queue->getData(5, [&](ReaderStep step) {
    EXPECT_TRUE(isCleanEnd(step));
    // A racing reader must not register another pending callback.
    bool receivedEnd{false};
    queue->getData(10, receiveEndMarker(10, receivedEnd));
    EXPECT_TRUE(receivedEnd);
    ++endCount;
  });
  EXPECT_EQ(endCount, 0);
  queue->terminate();
  queue->terminate();
  EXPECT_EQ(endCount, 1);
  queue->updateOutputBuffers(3, true);
  EXPECT_FALSE(queue->stats().noMoreBuffers);
  bool receivedEnd{false};
  queue->getData(0, receiveEndMarker(0, receivedEnd));
  EXPECT_TRUE(receivedEnd);
  queueManager_->removeTask(taskId);
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
      auto receive = [&](ReaderStep step) {
        // Re-enter the manager and queue to detect notifications under locks.
        EXPECT_TRUE(queueManager_->stats(taskId).has_value());
        if (step.kind == ReaderStep::Kind::kChunk) {
          ++dataCount;
          EXPECT_EQ(step.numRows, 10);
        } else {
          EXPECT_TRUE(isCleanEnd(step));
          ++endCount;
          EXPECT_EQ(step.numRows, 0);
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
    queueManager_->getData(taskId, destination, [&](ReaderStep step) {
      EXPECT_TRUE(isCleanEnd(step));
      EXPECT_EQ(step.numRows, 0);
      EXPECT_TRUE(step.remainingBytes.empty());
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
