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
#include <chrono>
#include <future>
#include <limits>
#include <memory>
#include <optional>
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

namespace facebook::velox::ucx_exchange {
// Prints an outcome in assertion failures.
void PrintTo(const ChannelOutcome& outcome, std::ostream* os) {
  *os << fmt::format("{}({})", outcome.kind, outcome.error);
}

// Prints a step in assertion failures.
void PrintTo(const ReaderStep& step, std::ostream* os) {
  switch (step.kind) {
    case ReaderStep::Kind::kChunk:
      *os << "chunk of " << step.numRows << " rows";
      break;
    case ReaderStep::Kind::kEnd:
      *os << "end ";
      PrintTo(step.outcome, os);
      break;
    case ReaderStep::Kind::kStop:
      *os << "stop";
      break;
  }
}
} // namespace facebook::velox::ucx_exchange

namespace {
// Matches a kEnd step whose outcome is a producer error that matches 'error'.
testing::Matcher<const ReaderStep&> isProducerError(
    testing::Matcher<const std::string&> error) {
  return testing::AllOf(
      testing::Field("kind", &ReaderStep::kind, ReaderStep::Kind::kEnd),
      testing::Field(
          "outcome",
          &ReaderStep::outcome,
          testing::AllOf(
              testing::Field(
                  "kind", &ChannelOutcome::kind, OutcomeKind::kProducerError),
              testing::Field("error", &ChannelOutcome::error, error))));
}

// Returns a reader that stores its one step in 'step'.
ReaderCallback storeStep(std::optional<ReaderStep>& step) {
  return [&step](ReaderStep received) {
    EXPECT_FALSE(step.has_value()) << "A reader got a second step";
    step = std::move(received);
  };
}
} // namespace

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

  // Reads one step of 'destination', which must be ready and of
  // 'expectedKind'. A chunk must carry data.
  void fetch(
      std::string_view taskId,
      int destination,
      ReaderStep::Kind expectedKind) {
    std::optional<ReaderStep> step;
    queueManager_->getData(taskId, destination, storeStep(step));
    ASSERT_TRUE(step.has_value()) << "for destination " << destination;
    ASSERT_EQ(step->kind, expectedKind)
        << "for destination " << destination << ": "
        << testing::PrintToString(*step);
    if (expectedKind == ReaderStep::Kind::kChunk) {
      EXPECT_NE(step->data, nullptr) << "for destination " << destination;
    }
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
      bool stopped{false};
      queueManager_->getData(taskId, destination, [&](ReaderStep step) {
        switch (step.kind) {
          case ReaderStep::Kind::kChunk:
            received++;
            break;
          case ReaderStep::Kind::kEnd:
            atEnd = true;
            break;
          case ReaderStep::Kind::kStop:
            stopped = true;
            break;
        }
        dataWait.notify();
      });
      dataWait.wait(dataWaitKey);
      // Only deleteResults() below Releases the destination, so a stop before
      // the end is wrong.
      ASSERT_FALSE(stopped)
          << "Destination stopped before its end: " << destination;
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
      // The destination is Released after a clean end: the reader stops
      // without a second end.
      ASSERT_EQ(response.kind, ReaderStep::Kind::kStop);
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
    fetch(taskId, destination, ReaderStep::Kind::kChunk);
  }

  EXPECT_FALSE(queueManager_->isFinished(taskId));

  for (int destination = 0; destination < 3; destination++) {
    enqueue(taskId, destination, size);
  }

  for (int destination = 0; destination < 3; destination++) {
    fetch(taskId, destination, ReaderStep::Kind::kChunk);
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
  // a Pending output queue for a task it has not seen.
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

// A server that closes while it waits on a Pending output queue leaves nothing
// that would ever erase the output queue if its producer never arrives.
// Releasing the last reader erases it, and the task can still be initialized
// later.
TEST_F(UcxOutputQueueManagerTest, releasingLastReaderDropsPendingOutputQueue) {
  const std::string taskId = "pendingOutputQueueWithClosedReaders";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  bool fired{false};
  auto receive = [&fired](ReaderStep /*step*/) { fired = true; };
  auto reader0 = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  reader0->getData(0, receive, /*intraNode=*/false);
  auto reader1 = queueManager_->bindReader(taskId, 1, /*onReleased=*/nullptr);
  reader1->getData(1, receive, /*intraNode=*/false);
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
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  EXPECT_FALSE(fired);
  queueManager_->removeTask(taskId);
}

// Releasing one reader of a Pending output queue keeps the output queue for the
// readers that are still bound, so that the producer finds them when it
// initializes.
TEST_F(UcxOutputQueueManagerTest, boundReaderKeepsPendingOutputQueue) {
  const std::string taskId = "pendingOutputQueueWithBoundReader";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  bool receivedData{false};
  auto reader0 = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  reader0->getData(0, receiveData(0, receivedData), /*intraNode=*/false);
  auto reader1 = queueManager_->bindReader(taskId, 1, /*onReleased=*/nullptr);
  queueManager_->releaseReader(reader1, 1);
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  EXPECT_EQ(reader0->state(), OutputQueueState::kPending);

  auto task = initializeTask(
      taskId, /*numDestinations=*/1, /*numDrivers=*/1, /*cleanup=*/false);
  EXPECT_EQ(reader0->state(), OutputQueueState::kActive);
  enqueue(taskId, /*size=*/10);
  EXPECT_TRUE(receivedData);
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->releaseReader(reader0, 0);
  queueManager_->removeTask(taskId);
}

// A caller that holds a Pending output queue for another reason must not keep
// it in the map once its last reader is released: nothing would erase it later.
TEST_F(
    UcxOutputQueueManagerTest,
    pendingOutputQueueHeldElsewhereIsStillErased) {
  const std::string taskId = "pendingOutputQueueHeldElsewhere";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto held = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  queueManager_->releaseReader(held, 0);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  EXPECT_EQ(held->state(), OutputQueueState::kPending);

  // The next reader gets a new output queue, not the one still held.
  auto next = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_NE(next, held);
  queueManager_->releaseReader(next, 0);
}

// A reader that arrives after a removal is bound to the Removed output queue,
// which ends it at once with the removal's outcome, rather than to a new
// Pending output queue that no producer fills.
TEST_F(
    UcxOutputQueueManagerTest,
    readerBoundDuringRemovalFindsRemovedOutputQueue) {
  const std::string taskId = "readerAfterRemoval";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  queueManager_->removeTask(taskId);
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);

  std::optional<ReaderStep> step;
  outputQueue->getData(0, storeStep(step), /*intraNode=*/false);
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::StrEq(std::string{kProducerNotInitializedError}))));
  queueManager_->releaseReader(outputQueue, 0);
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

// A server keeps the output queue it was bound to. When the task ID is
// initialized again, the old server's calls must not reach the new task.
TEST_F(UcxOutputQueueManagerTest, queueHandleDoesNotFollowReusedTaskId) {
  const std::string taskId = "reusedTaskId";
  auto oldTask = initializeTask(taskId, 1, 1);
  auto oldOutputQueue = queueManager_->bindReader(
      taskId, /*destination=*/0, /*onReleased=*/nullptr);

  oldTask->requestAbort().wait();
  queueManager_->removeTask(taskId);
  auto newTask = initializeTask(taskId, 1, 1);

  oldOutputQueue->deleteResults(0);

  EXPECT_EQ(oldTask->state(), TaskState::kAborted);
  EXPECT_EQ(newTask->state(), TaskState::kRunning);
  EXPECT_FALSE(queueManager_->isFinished(taskId));
  EXPECT_EQ(oldOutputQueue->state(), OutputQueueState::kRemoved);
  queueManager_->releaseReader(oldOutputQueue, 0);
  newTask->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// A Removed output queue lives as long as its Task, so that a consumer that
// connects late still finds the removal. The next removal after the Task is
// destroyed erases it.
TEST_F(UcxOutputQueueManagerTest, removedTaskRecordLivesAsLongAsItsTask) {
  const std::string taskId = "recordedWhileTaskLives";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  // Other removals keep the output queue while the Task is alive.
  queueManager_->removeTask("otherRemovedTask");
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);
  queueManager_->releaseReader(outputQueue, 0);

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterTaskDestroyed");

  // A consumer that arrives now is treated like one of a task not created yet.
  auto lateOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_NE(lateOutputQueue, outputQueue);
  EXPECT_EQ(lateOutputQueue->state(), OutputQueueState::kPending);
  queueManager_->releaseReader(lateOutputQueue, 0);
  for (const auto& removedTaskId :
       {taskId,
        std::string{"otherRemovedTask"},
        std::string{"removalAfterTaskDestroyed"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
  }
}

// The cancellation of a removed task in IntraNodeTransferRegistry lives as long
// as its Removed output queue, so that the registry keeps no record per removed
// task for the life of the worker.
TEST_F(UcxOutputQueueManagerTest, removedTaskRecordLeavesIntraNodeRegistry) {
  const std::string taskId = "removedIntraNodeRecord";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey key{
      taskId, /*destination=*/0, /*sequenceNumber=*/0};
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  // Other removals keep the record while the Task is alive.
  queueManager_->removeTask("otherRemovedIntraNodeTask");
  const auto cancelled = registry->poll(key);
  ASSERT_TRUE(cancelled.has_value());
  EXPECT_TRUE(cancelled->atEnd);
  EXPECT_EQ(cancelled->outcome.kind, OutcomeKind::kProducerError);
  EXPECT_THAT(
      cancelled->outcome.error,
      testing::HasSubstr("Producer task ended before sending all its output"));

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterIntraNodeTaskDestroyed");

  EXPECT_FALSE(registry->poll(key).has_value());
  EXPECT_FALSE(registry
                   ->poll(
                       {"otherRemovedIntraNodeTask",
                        /*destination=*/0,
                        /*sequenceNumber=*/0})
                   .has_value());
  registry->clearCancelledTask("removalAfterIntraNodeTaskDestroyed");
}

// A server bound to a Removed output queue keeps it, even after its Task is
// destroyed.
TEST_F(UcxOutputQueueManagerTest, boundReaderKeepsRemovedOutputQueue) {
  const std::string taskId = "removedOutputQueueWithBoundReader";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  queueManager_->removeTask("otherRemovedTask");
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterTaskDestroyed");

  auto sameOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(sameOutputQueue, outputQueue);
  EXPECT_EQ(sameOutputQueue->state(), OutputQueueState::kRemoved);
  queueManager_->releaseReader(sameOutputQueue, 0);
  queueManager_->releaseReader(outputQueue, 0);
  for (const auto& removedTaskId :
       {taskId,
        std::string{"otherRemovedTask"},
        std::string{"removalAfterTaskDestroyed"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
  }
}

// A Removed output queue outlives its removal while a server is bound to it or
// its Task lives, but its chunks are unreachable from then on. The removal
// frees them rather than keeping them for the output queue's lifetime.
TEST_F(UcxOutputQueueManagerTest, removedOutputQueueReleasesQueuedData) {
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  {
    SCOPED_TRACE("partitioned");
    const std::string taskId = "removedReleasesPartitionedData";
    auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
    auto outputQueue =
        queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
    enqueue(taskId, 0, /*size=*/10);
    enqueue(taskId, 1, /*size=*/10);
    const auto before = queueManager_->stats(taskId);
    ASSERT_TRUE(before.has_value());
    ASSERT_GT(before->bufferedBytes, 0);
    ASSERT_EQ(before->bufferedPages, 2);

    task->requestAbort().wait();
    queueManager_->removeTask(taskId);
    const auto after = outputQueue->stats();
    EXPECT_EQ(after.bufferedBytes, 0);
    EXPECT_EQ(after.bufferedPages, 0);
    EXPECT_EQ(after.totalBytesSent, before->totalBytesSent);
    EXPECT_EQ(after.totalPagesSent, 2);

    // The freed chunks were never delivered.
    std::optional<ReaderStep> step;
    outputQueue->getData(0, storeStep(step), /*intraNode=*/false);
    EXPECT_THAT(
        step,
        testing::Optional(isProducerError(
            testing::HasSubstr(
                "Producer task ended before sending all its output"))));
    queueManager_->releaseReader(outputQueue, 0);
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
    auto outputQueue =
        queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
    enqueue(taskId, 0, /*size=*/10);
    std::weak_ptr<cudf::packed_columns> chunk;
    outputQueue->getData(
        0,
        [&chunk](ReaderStep step) { chunk = step.data; },
        /*intraNode=*/false);
    ASSERT_FALSE(chunk.expired());

    task->requestAbort().wait();
    queueManager_->removeTask(taskId);
    EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);
    EXPECT_TRUE(chunk.expired());
    EXPECT_EQ(outputQueue->stats().bufferedBytes, 0);
    queueManager_->releaseReader(outputQueue, 0);
  }
}

// A task whose initializeTask() threw is removed with its output queue still
// Pending, so only the failed initialization knew its Task. The Removed output
// queue must still live as long as the Task rather than until the next removal.
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
  // The failed initialization left the output queue Pending and untouched: it
  // has no capacity yet and cannot use the intra-node path.
  ASSERT_TRUE(queueManager_->stats(taskId).has_value());
  EXPECT_EQ(queueManager_->getUtilization(taskId), std::nullopt);
  EXPECT_FALSE(queueManager_->canUseIntraNode(taskId));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  queueManager_->removeTask("removalAfterFailedInitialization");
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);
  queueManager_->releaseReader(outputQueue, 0);

  std::weak_ptr<Task> weakTask = task;
  task.reset();
  ASSERT_TRUE(weakTask.expired());
  queueManager_->removeTask("removalAfterTaskDestroyed");
  auto lateOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(lateOutputQueue->state(), OutputQueueState::kPending);
  queueManager_->releaseReader(lateOutputQueue, 0);
  for (const auto& removedTaskId :
       {taskId,
        std::string{"removalAfterFailedInitialization"},
        std::string{"removalAfterTaskDestroyed"}}) {
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(removedTaskId);
  }
}

// The output queue keeps a failed initializeTask() even when its last server
// closes before the removal, so that the Removed output queue lives as long as
// the Task.
TEST_F(UcxOutputQueueManagerTest, releasedReaderKeepsFailedInitialization) {
  const std::string taskId = "failedInitializationWithReader";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  auto task = createSourceTask(taskId, pool_, UcxTestData::kTestRowType);
  VELOX_ASSERT_THROW(
      queueManager_->initializeTask(
          task,
          core::PartitionedOutputNode::Kind::kPartitioned,
          /*numDestinations=*/-1,
          /*numDrivers=*/1),
      "Number of destinations must be non-negative");
  queueManager_->releaseReader(outputQueue, 0);

  auto pendingOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(pendingOutputQueue, outputQueue);
  EXPECT_EQ(pendingOutputQueue->state(), OutputQueueState::kPending);
  queueManager_->releaseReader(pendingOutputQueue, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  queueManager_->removeTask("removalAfterFailedInitialization");
  auto removedOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(removedOutputQueue, outputQueue);
  EXPECT_EQ(removedOutputQueue->state(), OutputQueueState::kRemoved);
  queueManager_->releaseReader(removedOutputQueue, 0);
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

// A removal of a task that never initialized its output ends a waiting reader
// with an error: its producer will never send the rows.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateBeforeInit) {
  const std::string taskId = "orphanTest";
  // A manager that has not seen the task, so the reader below waits on a
  // Pending output queue. A removal leaves a Removed output queue that ends
  // readers at once.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 0, storeStep(step));
  EXPECT_FALSE(step.has_value());

  queueManager_->removeTask(taskId);
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::StrEq(
              "Producer task ended before initializing its output"))));
}

// A producer that initialized its output but is removed before every driver
// called noMoreData() ends each destination with its reason, not with a clean
// end.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnTerminateAfterInit) {
  const std::string taskId = "crashTest";
  constexpr int kNumDestinations = 2;
  auto task = initializeTask(taskId, kNumDestinations, 1);

  std::array<std::optional<ReaderStep>, kNumDestinations> steps;
  for (int destination = 0; destination < kNumDestinations; ++destination) {
    queueManager_->getData(taskId, destination, storeStep(steps[destination]));
    EXPECT_FALSE(steps[destination].has_value());
  }

  // Abort the task, as Task::terminate() does before it removes the task.
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  for (int destination = 0; destination < kNumDestinations; ++destination) {
    EXPECT_THAT(
        steps[destination],
        testing::Optional(isProducerError(
            testing::AllOf(
                testing::HasSubstr(
                    "Producer task ended before sending all its output"),
                testing::HasSubstr("Aborted")))))
        << "for destination " << destination;
  }
}

// The coordinator also aborts producers that already sent all their output,
// for instance after a downstream LIMIT was satisfied. Once their servers took
// every chunk, their consumers have everything, so both the end marker queued
// by noMoreData() and a getData() after removal stay a clean end.
TEST_F(UcxOutputQueueManagerTest, removeTaskAfterAllOutputKeepsCleanEndMarker) {
  const std::string taskId = "finishedThenAborted";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);

  std::optional<ReaderStep> endMarker;
  queueManager_->getData(taskId, 0, storeStep(endMarker));
  noMoreData(taskId);
  ASSERT_TRUE(endMarker.has_value());
  EXPECT_TRUE(isCleanEnd(*endMarker));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> late;
  queueManager_->getData(taskId, 0, storeStep(late));
  ASSERT_TRUE(late.has_value());
  EXPECT_TRUE(isCleanEnd(*late));
}

// Removing a task frees whatever its destinations still hold. A producer that
// queued all its output but is aborted before its consumers fetched it has
// therefore not delivered it, and its consumers must fail.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskWithUndeliveredOutputReportsProducerError) {
  const std::string taskId = "queuedThenAborted";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 0, storeStep(step));
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::HasSubstr(
              "Producer task ended before its consumers received all its output"))));
}

// A producer that discards its own queued output before it finished, as
// UcxPartitionedOutput does when it runs out of GPU memory, ends every waiting
// and later reader with an error. A clean end would book the producer as
// finished while its rows are gone.
TEST_F(
    UcxOutputQueueManagerTest,
    deleteResultsBeforeAllOutputReportsProducerError) {
  const std::string taskId = "discardedBeforeEnd";
  auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
  enqueue(taskId, /*destination=*/1, /*size=*/100);

  // A reader waits on destination 0, which is empty.
  std::optional<ReaderStep> waiting;
  queueManager_->getData(taskId, 0, storeStep(waiting));
  EXPECT_FALSE(waiting.has_value());

  deleteResults(taskId, 0);
  deleteResults(taskId, 1);

  const auto discarded = isProducerError(
      testing::HasSubstr(
          "Producer task discarded its output before sending all of it"));
  EXPECT_THAT(waiting, testing::Optional(discarded));
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);

  // A reader that arrives after the discard gets the same error.
  std::optional<ReaderStep> late;
  queueManager_->getData(taskId, 1, storeStep(late));
  EXPECT_THAT(late, testing::Optional(discarded));

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// A broadcast producer keeps its chunks for destinations that have not arrived
// yet. Removing it before the last destination arrived drops those chunks, so
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
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, /*destination=*/1, storeStep(step));
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::AllOf(
              testing::HasSubstr(
                  "Producer task ended before its consumers received all its output"),
              testing::HasSubstr("undelivered packed tables: 1")))));
}

// A consumer that connects only after its failed producer was removed learns
// the reason too, rather than read an empty, complete stream.
TEST_F(
    UcxOutputQueueManagerTest,
    lateGetDataAfterFailedRemovalReportsProducerError) {
  const std::string taskId = "abortedThenFetched";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 0, storeStep(step));
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::HasSubstr(
              "Producer task ended before sending all its output"))));
}

// A task whose output initialization failed is removed with no initialized
// output queue. A consumer that arrives after the removal learns that the
// producer never initialized its output, rather than wait on an output queue
// that no task will initialize or remove.
TEST_F(
    UcxOutputQueueManagerTest,
    consumerAfterRemoveTaskWithoutQueueReportsProducerError) {
  const std::string taskId = "consumerAfterRemovalWithoutQueue";
  // A manager that has not seen the task, as after an initializeTask() that
  // threw before it created an output queue.
  queueManager_ = std::make_shared<UcxOutputQueueManager>();

  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 0, storeStep(step));
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::StrEq(
              "Producer task ended before initializing its output"))));
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
}

// A task error can quote user data in any language. The producer error keeps
// at most 4'096 bytes of it, and the cut must not split a character, which
// would leave invalid UTF-8 in the consumer's error message.
TEST_F(UcxOutputQueueManagerTest, truncatedProducerErrorKeepsWholeCharacters) {
  // Returns whether 'text' consists of whole UTF-8 characters.
  const auto isWholeUtf8 = [](std::string_view text) {
    size_t i = 0;
    while (i < text.size()) {
      const auto lead = static_cast<unsigned char>(text[i]);
      const size_t length = lead < 0x80 ? 1
          : (lead >> 5) == 0x06         ? 2
          : (lead >> 4) == 0x0E         ? 3
          : (lead >> 3) == 0x1E         ? 4
                                        : 0;
      if (length == 0 || i + length > text.size()) {
        return false;
      }
      for (size_t j = 1; j < length; ++j) {
        if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) {
          return false;
        }
      }
      i += length;
    }
    return true;
  };

  // A run of two-byte characters, once at an even and once at an odd offset,
  // so that one case cuts inside a character whatever the text before it.
  for (const std::string lead : {"", "a"}) {
    SCOPED_TRACE(fmt::format("lead of {} bytes", lead.size()));
    const auto taskId = fmt::format("truncatedUtf8Error{}", lead.size());
    auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
    std::string message = lead;
    for (int i = 0; i < 3'000; ++i) {
      message += "\xC3\xA9";
    }
    task->setError(message);
    queueManager_->removeTask(taskId);

    std::optional<ReaderStep> step;
    queueManager_->getData(taskId, 0, storeStep(step));
    ASSERT_THAT(
        step,
        testing::Optional(
            isProducerError(testing::EndsWith("\xC3\xA9...(truncated)"))));
    const auto& error = step->outcome.error;
    EXPECT_LE(error.size(), 4'096);
    EXPECT_TRUE(isWholeUtf8(error))
        << "Ends with: " << error.substr(error.size() - 20);
  }
}

// A Removed output queue keeps its outcome for every reader that arrives later,
// not only for the first.
TEST_F(UcxOutputQueueManagerTest, removedOutputQueueServesStoredOutcome) {
  const std::string taskId = "removedServesOutcome";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  task->setError("Injected producer failure");
  queueManager_->removeTask(taskId);

  std::vector<ReaderStep> steps;
  for (int reader = 0; reader < 2; ++reader) {
    SCOPED_TRACE(fmt::format("reader {}", reader));
    auto outputQueue =
        queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
    EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);
    std::optional<ReaderStep> step;
    outputQueue->getData(0, storeStep(step), /*intraNode=*/false);
    ASSERT_THAT(
        step,
        testing::Optional(isProducerError(
            testing::AllOf(
                testing::HasSubstr(
                    "Producer task ended before sending all its output"),
                testing::HasSubstr("Injected producer failure")))));
    steps.push_back(std::move(*step));
    queueManager_->releaseReader(outputQueue, 0);
  }
  EXPECT_EQ(steps[0].outcome, steps[1].outcome);
}

// An intra-node server takes a chunk from the producer's queue and later
// publishes it to IntraNodeTransferRegistry, which drops the chunk when the
// producer is removed. The chunk counts as undelivered from the moment it
// leaves the queue: removing the producer before the server published it fails
// its consumers, although its queues are empty and every driver finished.
// Covers a chunk that was queued before the server asked, which is handed over
// at once, and one that arrives later, which enqueue() hands over.
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
    auto outputQueue =
        queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
    std::optional<ReaderStep> chunk;
    outputQueue->getData(0, storeStep(chunk), /*intraNode=*/true);
    if (!queuedBeforeRequest) {
      enqueue(taskId, /*size=*/100);
      noMoreData(taskId);
    }
    ASSERT_TRUE(chunk.has_value());
    ASSERT_EQ(chunk->kind, ReaderStep::Kind::kChunk);

    // The server has the chunk and has not published it yet.
    task->requestAbort().wait();
    queueManager_->removeTask(taskId);

    const auto lostChunk = testing::AllOf(
        testing::HasSubstr(
            "Producer task ended before its consumers received all its output"),
        testing::HasSubstr("undelivered packed tables: 1"));
    std::optional<ReaderStep> end;
    outputQueue->getData(0, storeStep(end), /*intraNode=*/true);
    EXPECT_THAT(end, testing::Optional(isProducerError(lostChunk)));
    const auto polled =
        IntraNodeTransferRegistry::getInstance()->poll({taskId, 0, 0});
    ASSERT_TRUE(polled.has_value());
    EXPECT_TRUE(polled->atEnd);
    EXPECT_EQ(polled->outcome.kind, OutcomeKind::kProducerError);
    EXPECT_THAT(polled->outcome.error, lostChunk);
    queueManager_->releaseReader(outputQueue, 0);
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
  }
}

// Once the server reports that the consumer read the chunk, the producer's
// output is delivered, and removing it ends the stream cleanly.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskAfterIntraNodePageReadKeepsCleanEndMarker) {
  const std::string taskId = "intraNodePageRead";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  std::optional<ReaderStep> chunk;
  outputQueue->getData(0, storeStep(chunk), /*intraNode=*/true);
  ASSERT_TRUE(chunk.has_value());
  ASSERT_EQ(chunk->kind, ReaderStep::Kind::kChunk);
  outputQueue->intraNodeChunkRetrieved(0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> end;
  outputQueue->getData(0, storeStep(end), /*intraNode=*/true);
  ASSERT_TRUE(end.has_value());
  EXPECT_TRUE(isCleanEnd(*end));
  const auto polled =
      IntraNodeTransferRegistry::getInstance()->poll({taskId, 0, 1});
  ASSERT_TRUE(polled.has_value());
  EXPECT_TRUE(polled->atEnd);
  EXPECT_EQ(polled->outcome, ChannelOutcome::clean());
  queueManager_->releaseReader(outputQueue, 0);
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
}

// A chunk its server published but its consumer has not read yet is dropped
// with the producer, so that consumer fails rather than end cleanly. The
// server does not wait for a read that will not come.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskBeforeIntraNodePollReportsProducerError) {
  const std::string taskId = "intraNodePublishedNotPolled";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, /*size=*/100);
  noMoreData(taskId);
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  std::optional<ReaderStep> chunk;
  outputQueue->getData(0, storeStep(chunk), /*intraNode=*/true);
  ASSERT_TRUE(chunk.has_value());
  ASSERT_EQ(chunk->kind, ReaderStep::Kind::kChunk);

  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey chunkKey{
      taskId, /*destination=*/0, /*sequenceNumber=*/0};
  auto retrieved = registry->publish(
      chunkKey,
      std::move(chunk->data),
      chunk->numRows,
      /*atEnd=*/false,
      ChannelOutcome::clean());

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  EXPECT_EQ(
      retrieved.wait_for(std::chrono::seconds(0)), std::future_status::ready);
  const auto polled = registry->poll(chunkKey);
  ASSERT_TRUE(polled.has_value());
  EXPECT_TRUE(polled->atEnd);
  EXPECT_EQ(polled->data, nullptr);
  EXPECT_EQ(polled->outcome.kind, OutcomeKind::kProducerError);
  EXPECT_THAT(
      polled->outcome.error,
      testing::HasSubstr(
          "Producer task ended before its consumers received all its output"));
  queueManager_->releaseReader(outputQueue, 0);
  registry->clearCancelledTask(taskId);
}

// A chunk in flight to a consumer that is gone is not lost: nobody waits for
// it. A Released destination therefore does not count its chunks in flight,
// and a removal after every driver finished ends the other destinations
// cleanly.
TEST_F(
    UcxOutputQueueManagerTest,
    removeTaskAfterIntraNodeConsumerGoneEndsCleanly) {
  const std::string taskId = "intraNodeConsumerGone";
  auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
  enqueue(taskId, /*destination=*/0, /*size=*/100);
  noMoreData(taskId);
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  std::optional<ReaderStep> chunk;
  outputQueue->getData(0, storeStep(chunk), /*intraNode=*/true);
  ASSERT_TRUE(chunk.has_value());
  ASSERT_EQ(chunk->kind, ReaderStep::Kind::kChunk);
  queueManager_->consumerGone(taskId, 0);
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kReleased);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);

  std::optional<ReaderStep> end;
  outputQueue->getData(1, storeStep(end), /*intraNode=*/true);
  ASSERT_TRUE(end.has_value());
  EXPECT_TRUE(isCleanEnd(*end));
  const auto polled =
      IntraNodeTransferRegistry::getInstance()->poll({taskId, 1, 0});
  ASSERT_TRUE(polled.has_value());
  EXPECT_EQ(polled->outcome, ChannelOutcome::clean());
  queueManager_->releaseReader(outputQueue, 0);
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
}

// A removal also decides the destinations that are not published yet: their
// waiting reader gets the error once.
TEST_F(UcxOutputQueueManagerTest, removalEndsUnpublishedReaderWithError) {
  const std::string taskId = "removalEndsUnpublishedReader";
  auto task = initializeTask(
      taskId,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      /*cleanup=*/true,
      core::PartitionedOutputNode::Kind::kBroadcast);
  int numSteps{0};
  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 5, [&](ReaderStep received) {
    ++numSteps;
    step = std::move(received);
  });
  EXPECT_EQ(numSteps, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  queueManager_->removeTask(taskId);

  EXPECT_EQ(numSteps, 1);
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::HasSubstr(
              "Producer task ended before sending all its output"))));
}

// A removal frees what the output queue retains: its queued chunks, the
// broadcast chunks kept for later destinations, the blocked producers and the
// Task. A chunk that a reader took stays alive until that reader lets go.
TEST_F(UcxOutputQueueManagerTest, terminateReleasesRetainedQueueResources) {
  using Kind = core::PartitionedOutputNode::Kind;
  for (auto kind : {Kind::kPartitioned, Kind::kBroadcast}) {
    SCOPED_TRACE(kind);
    const std::string taskId = "terminateRetainedQueue";
    auto task = initializeTask(taskId, 2, 1, true, kind, 64);
    auto outputQueue =
        queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
    enqueue(taskId, 0, 100);
    enqueue(taskId, 0, 100);
    ContinueFuture blocked;
    ASSERT_TRUE(outputQueue->checkBlocked(&blocked));
    ASSERT_FALSE(blocked.isReady());
    std::shared_ptr<cudf::packed_columns> inFlightData;
    outputQueue->getData(
        0,
        [&](ReaderStep step) { inFlightData = std::move(step.data); },
        /*intraNode=*/false);
    ASSERT_NE(inFlightData, nullptr);
    std::weak_ptr<cudf::packed_columns> dataReference = inFlightData;

    // The task is not started, so its termination does not remove its output.
    task->requestAbort().wait();
    queueManager_->removeTask(taskId);
    std::weak_ptr<Task> taskReference = task;
    task.reset();
    EXPECT_TRUE(taskReference.expired());
    EXPECT_TRUE(blocked.isReady());
    EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
    EXPECT_EQ(outputQueue->stats().bufferedBytes, 0);
    // The reader may still use the chunk it took. Once it lets go, neither
    // the destinations nor the broadcast backfill may retain the chunk.
    EXPECT_FALSE(dataReference.expired());
    inFlightData.reset();
    EXPECT_TRUE(dataReference.expired());

    // A removal with chunks left is an error, for published destinations and
    // for one never published.
    for (int destination : {0, 1, 2}) {
      std::optional<ReaderStep> step;
      outputQueue->getData(destination, storeStep(step), /*intraNode=*/false);
      EXPECT_THAT(
          step,
          testing::Optional(isProducerError(
              testing::HasSubstr(
                  "Producer task ended before sending all its output"))))
          << "for destination " << destination;
    }
    // Repeated removal must not re-create queues or retain callbacks.
    queueManager_->removeTask(taskId);
    EXPECT_FALSE(outputQueue->checkBlocked(&blocked));
    queueManager_->releaseReader(outputQueue, 0);
  }
}

// A consumer that is gone releases its destination and deletes its results,
// as an HTTP consumer that aborts does. Otherwise the producer would keep
// them, and could not finish, until it is aborted.
TEST_F(UcxOutputQueueManagerTest, releasingGoneConsumerDeletesItsResults) {
  const std::string taskId = "resultsOfGoneConsumer";
  auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
  enqueue(taskId, /*destination=*/0, /*size=*/10);
  enqueue(taskId, /*destination=*/1, /*size=*/10);

  queueManager_->consumerGone(taskId, 0);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 1);

  noMoreData(taskId);
  fetch(taskId, 1, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 1);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// A close for a task this worker has not seen is dropped: it creates no output
// queue that nothing would erase.
TEST_F(UcxOutputQueueManagerTest, consumerGoneForUnknownTaskIsDropped) {
  const std::string taskId = "unknownTaskOfGoneConsumer";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  queueManager_->consumerGone(taskId, 0);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
  // A later reader gets a new Pending output queue, with destination 0
  // Unpublished.
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->state(), OutputQueueState::kPending);
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kUnpublished);
  queueManager_->releaseReader(outputQueue, 0);
  EXPECT_EQ(queueManager_->stats(taskId), std::nullopt);
}

// A consumer that goes away before its destination is published keeps the
// destination Released: publishing it later does not reopen it, and the
// producer finishes without it.
TEST_F(UcxOutputQueueManagerTest, consumerGoneBeforePublicationStaysReleased) {
  const std::string taskId = "consumerGoneBeforePublication";
  auto task = initializeTask(
      taskId,
      /*numDestinations=*/1,
      /*numDrivers=*/1,
      /*cleanup=*/true,
      core::PartitionedOutputNode::Kind::kBroadcast);
  queueManager_->consumerGone(taskId, 1);
  queueManager_->updateOutputBuffers(taskId, 2, true);
  noMoreData(taskId);
  fetchEndMarker(taskId, 0);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  auto outputQueue =
      queueManager_->bindReader(taskId, 1, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->destinationState(1), DestinationState::kReleased);
  queueManager_->releaseReader(outputQueue, 1);
  queueManager_->removeTask(taskId);
}

// A consumer that goes away while its producer has not initialized its output
// leaves its destination Released in the Pending output queue, which is kept
// for the producer. The producer then finishes without waiting for that
// consumer.
TEST_F(
    UcxOutputQueueManagerTest,
    consumerGoneOnPendingOutputQueueFinishesProducerAtInit) {
  const std::string taskId = "consumerGoneBeforeInit";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto reader = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  queueManager_->consumerGone(taskId, 0);
  queueManager_->releaseReader(reader, 0);
  reader.reset();
  EXPECT_TRUE(queueManager_->stats(taskId).has_value());

  auto task = initializeTask(
      taskId, /*numDestinations=*/1, /*numDrivers=*/1, /*cleanup=*/false);
  noMoreData(taskId);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  EXPECT_EQ(task->state(), TaskState::kFinished);
  queueManager_->removeTask(taskId);
}

// The Pending output queue that a gone consumer leaves holds neither a reader
// nor a chunk, only the Released destination, whose next reader stops at once.
TEST_F(UcxOutputQueueManagerTest, releasedPendingOutputQueueHoldsNoReader) {
  const std::string taskId = "releasedPendingOutputQueue";
  queueManager_ = std::make_shared<UcxOutputQueueManager>();
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  std::optional<ReaderStep> waiting;
  outputQueue->getData(0, storeStep(waiting), /*intraNode=*/false);
  EXPECT_FALSE(waiting.has_value());
  queueManager_->consumerGone(taskId, 0);
  ASSERT_TRUE(waiting.has_value());
  EXPECT_EQ(waiting->kind, ReaderStep::Kind::kStop);
  queueManager_->releaseReader(outputQueue, 0);

  // The map and this test hold the output queue; no reader callback does.
  EXPECT_LE(outputQueue.use_count(), 2);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedPages, 0);
  auto again = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(again, outputQueue);
  std::optional<ReaderStep> step;
  again->getData(0, storeStep(step), /*intraNode=*/false);
  ASSERT_TRUE(step.has_value());
  EXPECT_EQ(step->kind, ReaderStep::Kind::kStop);
  queueManager_->releaseReader(again, 0);
  queueManager_->removeTask(taskId);
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
}

// A failed send releases the destination and returns the error for the caller
// to fail the task with. The output queue calls no Task method itself.
TEST_F(UcxOutputQueueManagerTest, sendFailedFailsProducerTask) {
  const std::string taskId = "sendFailed";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  enqueue(taskId, 0, 10);

  auto error = outputQueue->sendFailed(0, "Connection reset");
  ASSERT_TRUE(error.has_value());
  EXPECT_THAT(
      *error,
      testing::AllOf(
          testing::HasSubstr("Failed to send to consumer over UCX"),
          testing::HasSubstr(taskId),
          testing::HasSubstr("Connection reset")));
  EXPECT_EQ(task->state(), TaskState::kRunning);
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kReleased);
  EXPECT_EQ(outputQueue->stats().bufferedPages, 0);
  // The first decision wins.
  EXPECT_EQ(outputQueue->sendFailed(0, "Connection reset"), std::nullopt);

  task->setError(*error);
  EXPECT_EQ(task->state(), TaskState::kFailed);
  queueManager_->releaseReader(outputQueue, 0);
  queueManager_->removeTask(taskId);
}

// The release callback runs outside the manager's and the output queue's locks,
// so that it may call back into them.
TEST_F(UcxOutputQueueManagerTest, onReleasedFiresOutsideLock) {
  const std::string taskId = "onReleasedOutsideLock";
  auto task = initializeTask(taskId, /*numDestinations=*/2, /*numDrivers=*/1);
  int numReleased{0};
  auto outputQueue = queueManager_->bindReader(taskId, 0, [&]() {
    EXPECT_TRUE(queueManager_->stats(taskId).has_value());
    ++numReleased;
  });
  queueManager_->consumerGone(taskId, 0);
  EXPECT_EQ(numReleased, 1);
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kReleased);
  // The callback fires once.
  queueManager_->consumerGone(taskId, 0);
  EXPECT_EQ(numReleased, 1);
  queueManager_->releaseReader(outputQueue, 0);
  queueManager_->removeTask(taskId);
}

// A reader that arrives after the producer discarded a destination's output
// fails, even once that destination is Released, rather than wait for the
// producer's abort.
TEST_F(UcxOutputQueueManagerTest, readerAfterDiscardedOutputFails) {
  const std::string taskId = "readerAfterDiscard";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  enqueue(taskId, 0, 10);
  deleteResults(taskId, 0);

  const auto discarded = isProducerError(
      testing::HasSubstr(
          "Producer task discarded its output before sending all of it"));
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kEnded);
  {
    std::optional<ReaderStep> step;
    outputQueue->getData(0, storeStep(step), /*intraNode=*/false);
    EXPECT_THAT(step, testing::Optional(discarded));
  }
  // The server deletes the results after it sent the end.
  deleteResults(taskId, 0);
  EXPECT_EQ(outputQueue->destinationState(0), DestinationState::kReleased);
  queueManager_->releaseReader(outputQueue, 0);

  auto late = queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  std::optional<ReaderStep> step;
  late->getData(0, storeStep(step), /*intraNode=*/false);
  EXPECT_THAT(step, testing::Optional(discarded));
  queueManager_->releaseReader(late, 0);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// A second removal keeps the Removed output queue and the outcome of the
// first, even though the task's state changed in between. Only initialization
// replaces the output queue.
TEST_F(UcxOutputQueueManagerTest, removeTaskTombstoneIsIdempotent) {
  const std::string taskId = "idempotentTombstone";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);
  queueManager_->removeTask(taskId);
  auto outputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
  auto sameOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_EQ(sameOutputQueue, outputQueue);
  EXPECT_EQ(outputQueue->state(), OutputQueueState::kRemoved);

  std::optional<ReaderStep> stale;
  outputQueue->getData(0, storeStep(stale), /*intraNode=*/false);
  EXPECT_THAT(
      stale,
      testing::Optional(isProducerError(
          testing::AllOf(
              testing::HasSubstr("state: Running"),
              testing::Not(testing::HasSubstr("Aborted"))))));
  queueManager_->releaseReader(sameOutputQueue, 0);
  queueManager_->releaseReader(outputQueue, 0);

  // Initialization replaces the Removed output queue with a new one.
  auto newTask = initializeTask(taskId, 1, 1, /*cleanup=*/false);
  auto liveOutputQueue =
      queueManager_->bindReader(taskId, 0, /*onReleased=*/nullptr);
  EXPECT_NE(liveOutputQueue, outputQueue);
  std::optional<ReaderStep> live;
  liveOutputQueue->getData(0, storeStep(live), /*intraNode=*/false);
  EXPECT_FALSE(live.has_value());

  newTask->requestAbort().wait();
  queueManager_->removeTask(taskId);
  EXPECT_THAT(
      live, testing::Optional(isProducerError(testing::HasSubstr("Aborted"))));
  queueManager_->releaseReader(liveOutputQueue, 0);
}

// A reader waiting on an open destination whose results the producer deletes
// is ended at once, with the "discarded" error.
TEST_F(UcxOutputQueueManagerTest, callbackFiredOnDeleteResults) {
  const std::string taskId = "callbackOnDeleteResults";
  auto task = initializeTask(taskId, /*numDestinations=*/1, /*numDrivers=*/1);

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 0, storeStep(step));
  ASSERT_FALSE(step.has_value());

  queueManager_->deleteResults(taskId, 0);
  ASSERT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::HasSubstr(
              "Producer task discarded its output before sending all of it"))));
  EXPECT_TRUE(step->remainingBytes.empty());
  EXPECT_EQ(step->numRows, 0);

  task->requestAbort().wait();
  queueManager_->removeTask(taskId);
}

// A reader waiting for a destination that is not published yet is ended by
// the removal of its task, with the removal's error. A reader that arrives
// while the first one is notified ends at once.
TEST_F(UcxOutputQueueManagerTest, terminateUnpublishedReader) {
  const std::string taskId = "terminatedPendingReader";
  auto task = initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  auto outputQueue =
      queueManager_->bindReader(taskId, 5, /*onReleased=*/nullptr);
  const auto removedEarly = isProducerError(
      testing::HasSubstr("Producer task ended before sending all its output"));
  int endCount{0};
  outputQueue->getData(
      5,
      [&](ReaderStep step) {
        EXPECT_THAT(step, removedEarly);
        // A racing reader must not register another pending callback.
        std::optional<ReaderStep> racing;
        outputQueue->getData(10, storeStep(racing), /*intraNode=*/false);
        EXPECT_THAT(racing, testing::Optional(removedEarly));
        ++endCount;
      },
      /*intraNode=*/false);
  EXPECT_EQ(endCount, 0);
  queueManager_->removeTask(taskId);
  queueManager_->removeTask(taskId);
  EXPECT_EQ(endCount, 1);
  outputQueue->updateOutputBuffers(3, true);
  EXPECT_FALSE(outputQueue->stats().noMoreBuffers);
  std::optional<ReaderStep> step;
  outputQueue->getData(0, storeStep(step), /*intraNode=*/false);
  EXPECT_THAT(step, testing::Optional(removedEarly));
  queueManager_->releaseReader(outputQueue, 5);
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
    fetch(taskId, dest, ReaderStep::Kind::kChunk);
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
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetch(taskId, 1, ReaderStep::Kind::kChunk);

  // Now add a 3rd destination (late arrival). It should be backfilled.
  queueManager_->updateOutputBuffers(taskId, 3, false);

  // The new destination should have both batches.
  fetch(taskId, 2, ReaderStep::Kind::kChunk);
  fetch(taskId, 2, ReaderStep::Kind::kChunk);

  // Finalize destinations and data.
  queueManager_->updateOutputBuffers(taskId, 3, true);
  noMoreData(taskId);

  // Fetch remaining data + end markers.
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 0);
  fetch(taskId, 1, ReaderStep::Kind::kChunk);
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
      // Nobody read destination 0; its consumer goes away.
      queueManager_->consumerGone(taskId, 0);
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
    // Nobody read the other destinations; their consumers go away.
    for (int destination = 0; destination < kDestination; ++destination) {
      queueManager_->consumerGone(taskId, destination);
    }
    EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 0);
    EXPECT_TRUE(queueManager_->isFinished(taskId));
    queueManager_->removeTask(taskId);
  }
}

// Results of a destination that is not published yet are deleted before every
// producing driver finished only when the producer discards its own output, as
// for a published destination. The reader waiting for that destination, and a
// later one, fail rather than end cleanly.
TEST_F(UcxOutputQueueManagerTest, deleteUnpublishedReader) {
  const std::string taskId = "deletedPendingReader";
  initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  enqueue(taskId, 0, 10);
  const auto discarded = isProducerError(
      testing::HasSubstr(
          "Producer task discarded its output before sending all of it"));
  std::optional<ReaderStep> waiting;
  queueManager_->getData(taskId, 1, storeStep(waiting));
  EXPECT_FALSE(waiting.has_value());
  queueManager_->deleteResults(taskId, 1);
  EXPECT_THAT(waiting, testing::Optional(discarded));
  const auto bytes = queueManager_->stats(taskId)->bufferedBytes;
  queueManager_->updateOutputBuffers(taskId, 2, true);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, bytes);
  std::optional<ReaderStep> retry;
  queueManager_->getData(taskId, 1, storeStep(retry));
  EXPECT_THAT(retry, testing::Optional(discarded));
  noMoreData(taskId);
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 0);
  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}

// The same for a destination whose results are deleted before any reader
// asked for it.
TEST_F(
    UcxOutputQueueManagerTest,
    deleteUnpublishedReaderBeforeEndReportsProducerError) {
  const std::string taskId = "deletedPendingReaderBeforeEnd";
  initializeTask(
      taskId, 1, 1, true, core::PartitionedOutputNode::Kind::kBroadcast);
  enqueue(taskId, 0, 10);

  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 1, storeStep(step));
  EXPECT_FALSE(step.has_value());

  queueManager_->deleteResults(taskId, 1);
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::HasSubstr(
              "Producer task discarded its output before sending all of it"))));

  queueManager_->updateOutputBuffers(taskId, 2, true);
  noMoreData(taskId);
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 0);
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
  std::optional<ReaderStep> step;
  queueManager_->getData(taskId, 2, storeStep(step));
  EXPECT_THAT(
      step,
      testing::Optional(isProducerError(
          testing::HasSubstr(
              "Producer task discarded its output before sending all of it"))));
  queueManager_->updateOutputBuffers(taskId, 3, true);
  EXPECT_EQ(queueManager_->stats(taskId)->bufferedBytes, 2 * pageBytes);
  noMoreData(taskId);
  for (int destination : {0, 1}) {
    fetch(taskId, destination, ReaderStep::Kind::kChunk);
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
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
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
  fetch(taskId, 0, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 0);

  // Destination 1 (late): backfilled data + end marker.
  fetch(taskId, 1, ReaderStep::Kind::kChunk);
  fetchEndMarker(taskId, 1);

  EXPECT_TRUE(queueManager_->isFinished(taskId));
  queueManager_->removeTask(taskId);
}
