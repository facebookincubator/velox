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
#include "velox/experimental/ucx-exchange/UcxExchange.h"
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <folly/Executor.h>
#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/EventCount.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/tests/utils/Ports.h"
#include "velox/exec/tests/utils/TaskMonitor.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxChannelState.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include "velox/experimental/ucx-exchange/tests/FakeUcxPeer.h"
#include "velox/experimental/ucx-exchange/tests/SinkDriverMock.h"
#include "velox/experimental/ucx-exchange/tests/SourceDriverMock.h"
#include "velox/experimental/ucx-exchange/tests/UcxExchangeTestBase.h"
#include "velox/experimental/ucx-exchange/tests/UcxPartitionedOutputMock.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestData.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

namespace facebook::velox::ucx_exchange {

struct ExchangeTestParams {
  int numSrcDrivers;
  int numDstDrivers;
  int numPartitions;
  int numChunks;
  int numRowsPerChunk;
  int numUpstreamTasks;
  TableType tableType = TableType::NARROW; // Default to narrow table

  bool operator==(const ExchangeTestParams&) const = default;
};

// Helper function to generate test parameters with different numUpstreamTasks
static std::vector<ExchangeTestParams> generateTestParams() {
  std::vector<ExchangeTestParams> params;

  // Base configurations
  struct BaseConfig {
    const char* description;
    int numSrcDrivers;
    int numDstDrivers;
    int numPartitions;
    int numChunks;
    int numRowsPerChunk;
    TableType tableType;
  };

  std::vector<BaseConfig> baseConfigs = {
      // Test to check end-2-end connectivity
      {"Simple", 1, 1, 1, 100, 1000 * 1000, TableType::NARROW},
      // Test to check parallelism at source
      {"SourceDrivers", 10, 1, 1, 10, 1000 * 1000, TableType::NARROW},
      // Test to check parallelism at source and sink
      {"SourceSinkDrivers", 10, 10, 1, 10, 1000, TableType::NARROW},
      // Test with multiple partitions (hash partitioning)
      {"MultiPartition", 1, 1, 4, 100, 1000, TableType::NARROW},
      // Test with multiple partitions and multiple drivers
      {"MultiPartitionDrivers", 4, 4, 4, 25, 1000, TableType::NARROW},
      // Wide table tests with all data types including STRUCT
      // Single partition wide table (no hash partitioning)
      {"WideTableSingle", 1, 1, 1, 100, 1000, TableType::WIDE},
      // Multi-partition wide table (uses hash partitioning)
      {"WideTableMulti", 1, 1, 4, 10, 1000 * 10000, TableType::WIDE}};

  // Generate variants with different number of upstream tasks.
  std::vector<int> upstreamTaskCounts = {1, 10};

  for (const auto& base : baseConfigs) {
    for (int numUpstream : upstreamTaskCounts) {
      params.push_back(
          {.numSrcDrivers = base.numSrcDrivers,
           .numDstDrivers = base.numDstDrivers,
           .numPartitions = base.numPartitions,
           .numChunks = base.numChunks,
           .numRowsPerChunk = base.numRowsPerChunk,
           .numUpstreamTasks = numUpstream,
           .tableType = base.tableType});
    }
  }

  return params;
}

// Custom parameter name generator for readable test names
struct ExchangeTestParamsPrinter {
  std::string operator()(
      const ::testing::TestParamInfo<ExchangeTestParams>& info) const {
    const auto& p = info.param;
    std::ostringstream oss;
    oss << "Src" << p.numSrcDrivers << "_Dst" << p.numDstDrivers << "_Part"
        << p.numPartitions << "_Chunks" << p.numChunks << "_RowsPer"
        << p.numRowsPerChunk << "_Upstream" << p.numUpstreamTasks << "_"
        << (p.tableType == TableType::WIDE ? "Wide" : "Narrow");
    return oss.str();
  }
};

class UcxExchangeTest : public UcxExchangeTestBase,
                        public testing::WithParamInterface<ExchangeTestParams> {
 protected:
  // Some tests are not yet compatible with WideTestTable.
  bool shouldSkipWideTable() {
    return GetParam().tableType == TableType::WIDE;
  }
};

namespace {

// UcxExchangeSource connects to the split URL's port + 3.
constexpr uint16_t kSplitUrlPortOffset{3};

// Keeps an endpoint that a test opens registered with the Communicator until
// the test removes it.
class TestEndpointOwner final : public CommElement {
 public:
  explicit TestEndpointOwner(std::shared_ptr<Communicator> communicator)
      : CommElement(std::move(communicator)) {}

  void process() override {}

  void onEndpointClosed(std::string_view /*reason*/) override {}
};

// Returns true if 'buffer' holds an end message with a clean outcome. Runs in a
// UCX callback, so it does not throw.
bool isCleanEnd(const std::vector<uint8_t>& buffer) {
  try {
    const auto message = decodeProducerMessage(buffer.data(), buffer.size());
    const auto* outcome = std::get_if<ChannelOutcome>(&message);
    return outcome != nullptr && outcome->isClean();
  } catch (const std::exception&) {
    return false;
  }
}

} // namespace

// Fixture for focused behavioral tests that should run once per process rather
// than once for every ExchangeTestParams value.
class UcxExchangeFocusedTest : public UcxExchangeTestBase {
 protected:
  // A server for destination 0 of a producer task, sending to this worker's
  // own listener. Nothing receives what it sends until
  // postTerminalMetadataReceive() posts the receive of its end message.
  struct TerminalMetadataSession {
    std::string taskId;
    std::shared_ptr<exec::Task> task;
    std::shared_ptr<TestEndpointOwner> owner;
    std::shared_ptr<EndpointRef> endpoint;
    // The Communicator owns the server until the server closes.
    std::weak_ptr<UcxExchangeServer> server;
    std::shared_ptr<ucxx::Request> peerRequest;
    std::shared_ptr<std::vector<uint8_t>> receiveBuffer;
    // The close message that sendCloseMessage() sent.
    std::shared_ptr<ucxx::Request> closeRequest;
    // -1 until the end message arrives, then 1 for a clean end and 0 for
    // anything else.
    std::atomic<int> receiveResult{-1};
  };

  // Runs 'action' on the Communicator thread, outside every UCX callback, and
  // waits for it. Returns false if it does not run within 10 seconds. The
  // wait is generous because the Communicator loop may be busy with other work
  // items.
  static bool runOnCommunicator(std::function<void()> action) {
    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    Communicator::getInstance()->postToLoop(
        [action = std::move(action), done]() mutable {
          action();
          // Destroys what 'action' captured before the caller resumes, so that
          // a session's task never outlives the test's memory pool.
          action = nullptr;
          done->set_value();
        });
    return future.wait_for(std::chrono::seconds(10)) ==
        std::future_status::ready;
  }

  // Starts a producer task 'taskId' with one destination and one driver, which
  // enqueues one chunk of 'numChunkRows' rows if that is not zero and then
  // calls noMoreData(). Then starts a server for destination 0 and advances it
  // until it posts its first send.
  std::shared_ptr<TerminalMetadataSession> startTerminalMetadataSend(
      const std::string& taskId,
      vector_size_t numChunkRows) {
    auto session = std::make_shared<TerminalMetadataSession>();
    session->taskId = taskId;
    session->task = makeProducerTask(taskId, pool_.get());
    queueManager()->initializeTask(
        session->task,
        core::PartitionedOutputNode::Kind::kPartitioned,
        /*numDestinations=*/1,
        /*numDrivers=*/1);
    if (numChunkRows > 0) {
      cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
      auto table = makeTable(numChunkRows, UcxTestData::kTestRowType, stream);
      auto data = std::make_unique<cudf::packed_columns>(
          cudf::pack(table->view(), stream));
      stream.sync();
      queueManager()->enqueue(taskId, 0, std::move(data), numChunkRows);
    }
    queueManager()->noMoreData(taskId);
    startServer(session);
    EXPECT_FALSE(queueManager()->isFinished(taskId))
        << "An unmatched send must keep its output alive";
    return session;
  }

  // Starts a server for destination 0 of producer task 'taskId', whose output
  // is not initialized, so that the server waits on a Pending entry.
  std::shared_ptr<TerminalMetadataSession> startPendingSession(
      const std::string& taskId) {
    auto session = std::make_shared<TerminalMetadataSession>();
    session->taskId = taskId;
    session->task = makeProducerTask(taskId, pool_.get());
    startServer(session);
    return session;
  }

  // Starts a server for destination 0 of 'session's task, sending to this
  // worker's own listener, and advances it until it posts its first send or,
  // for a Pending entry, waits for data.
  static void startServer(
      const std::shared_ptr<TerminalMetadataSession>& session) {
    EXPECT_TRUE(runOnCommunicator([session] {
      auto communicator = Communicator::getInstance();
      session->owner = std::make_shared<TestEndpointOwner>(communicator);
      session->endpoint = communicator->assocEndpointRef(
          session->owner,
          HostPort{"127.0.0.1", communicator->getListenerPort()});
      if (session->endpoint == nullptr) {
        return;
      }
      auto server = UcxExchangeServer::create(
          communicator,
          session->endpoint,
          PartitionKey{session->taskId, 0},
          /*isIntraNodeTransfer=*/false);
      session->endpoint->addCommElem(server);
      communicator->registerCommElement(server);
      // From Created through ReadyToTransfer and DataReady to the send.
      server->process();
      server->process();
      server->process();
      session->server = server;
    }));
    EXPECT_NE(session->endpoint, nullptr);
  }

  // Sends the close message of destination 0 of 'session's task over
  // 'session's endpoint, as a source that closes does.
  static void sendCloseMessage(
      const std::shared_ptr<TerminalMetadataSession>& session) {
    EXPECT_TRUE(runOnCommunicator([session] {
      auto message = std::make_shared<CloseMsg>(
          makeCloseMsg(PartitionKey{session->taskId, 0}));
      session->closeRequest =
          session->endpoint->endpoint_
              ->tagSendBuilder(
                  message.get(),
                  sizeof(CloseMsg),
                  ucxx::Tag{getCloseTag(
                      fnv1a_32(PartitionKey{session->taskId, 0}.toString()))})
              .callbackData(message)
              .build();
    }));
  }

  // Waits up to 'timeout' for 'condition' to hold, and returns whether it
  // holds.
  static bool waitUntil(
      const std::function<bool()>& condition,
      std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
  }

  // Returns true if every UCX send goes by rendezvous, so that a send completes
  // only once its receive matched it.
  static bool isRendezvousOnly() {
    const auto* threshold = std::getenv("UCX_RNDV_THRESH");
    return threshold != nullptr && std::string_view(threshold) == "0";
  }

  // Returns the next table that 'client' receives for consumer 0, or null if
  // its stream ends or no table arrives within 10 seconds.
  static PackedTableWithStreamPtr nextTable(UcxExchangeClient& client) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      bool atEnd{false};
      ContinueFuture future = ContinueFuture::makeEmpty();
      auto data = client.next(0, &atEnd, &future);
      if (data != nullptr || atEnd) {
        return data;
      }
      std::move(future).wait(std::chrono::seconds(1));
    }
    return nullptr;
  }

  // Returns the URL of destination 0 of 'taskId' on this worker, as a split
  // gives it to UcxExchangeClient::addRemoteTaskId().
  std::string remoteTaskUrl(std::string_view taskId) {
    auto split = remoteSplit(taskId, /*partitionId=*/0);
    return std::dynamic_pointer_cast<exec::RemoteConnectorSplit>(
               split.connectorSplit)
        ->taskId;
  }

  // Posts the receive of the end message that 'session's server sends.
  void postTerminalMetadataReceive(
      const std::shared_ptr<TerminalMetadataSession>& session) {
    EXPECT_TRUE(runOnCommunicator([session] {
      session->receiveBuffer =
          std::make_shared<std::vector<uint8_t>>(kMaxMetaBufSize);
      std::weak_ptr<TerminalMetadataSession> weakSession = session;
      session->peerRequest =
          session->endpoint->endpoint_
              ->tagRecvBuilder(
                  session->receiveBuffer->data(),
                  session->receiveBuffer->size(),
                  ucxx::Tag{getMetadataTag(
                      fnv1a_32(PartitionKey{session->taskId, 0}.toString()),
                      0)},
                  ucxx::TagMaskFull)
              .callbackFunction(
                  [weakSession](
                      ucs_status_t status, std::shared_ptr<void> arg) {
                    if (auto locked = weakSession.lock()) {
                      const auto buffer =
                          std::static_pointer_cast<std::vector<uint8_t>>(arg);
                      locked->receiveResult =
                          status == UCS_OK && isCleanEnd(*buffer) ? 1 : 0;
                    }
                  })
              .callbackData(session->receiveBuffer)
              .build();
    }));
  }

  // Waits up to 3 seconds for the output of 'taskId' to finish.
  bool waitForOutputFinished(std::string_view taskId) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!queueManager()->isFinished(taskId) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return queueManager()->isFinished(taskId);
  }

  // Waits up to 3 seconds for the end message of 'session' to arrive, and
  // returns true if it is a clean end.
  static bool waitForTerminalReceive(
      const std::shared_ptr<TerminalMetadataSession>& session) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (session->receiveResult.load() == -1 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return session->receiveResult.load() == 1;
  }

  // Waits up to 3 seconds for 'server' to be destroyed.
  static bool waitForServerDestroyed(
      const std::weak_ptr<UcxExchangeServer>& server) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!server.expired() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return server.expired();
  }

  // Returns the state of destination 0 of 'taskId'. Binds a reader for the
  // lookup, so call it only once the session's server is done.
  DestinationState destinationState(std::string_view taskId) {
    auto entry = queueManager()->bindReader(taskId, 0, /*onReleased=*/nullptr);
    const auto state = entry->destinationState(0);
    queueManager()->releaseReader(entry, 0);
    return state;
  }

  void cleanUpTerminalMetadataSession(
      const std::shared_ptr<TerminalMetadataSession>& session) {
    EXPECT_TRUE(runOnCommunicator([session] {
      if (auto server = session->server.lock()) {
        server->close();
      }
      if (session->peerRequest != nullptr) {
        if (!session->peerRequest->isCompleted()) {
          session->peerRequest->cancel();
        }
        Communicator::getInstance()->deferRequestCleanup(
            std::move(session->peerRequest));
      }
      // Keeps the close message alive until its send completes.
      Communicator::getInstance()->deferRequestCleanup(
          std::move(session->closeRequest));
      if (session->endpoint != nullptr) {
        session->endpoint->removeCommElem(session->owner);
      }
    }));
    queueManager()->removeTask(session->taskId);
  }
};

INSTANTIATE_TEST_SUITE_P(
    UcxExchangeTest,
    UcxExchangeTest,
    ::testing::ValuesIn(generateTestParams()),
    ExchangeTestParamsPrinter());

namespace {

// Packs a 'numRows'-row table of UcxTestData::kTestRowType, as a producer would
// enqueue it.
std::unique_ptr<cudf::packed_columns> makePackedColumns(vector_size_t numRows) {
  auto stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
  auto table = makeTable(numRows, UcxTestData::kTestRowType, stream);
  auto packed =
      std::make_unique<cudf::packed_columns>(cudf::pack(table->view(), stream));
  stream.sync();
  return packed;
}

// Returns a 'numRows'-row table as a source puts it into the exchange queue.
PackedTableWithStreamPtr makePackedTable(vector_size_t numRows) {
  auto packed = makePackedColumns(numRows);
  const auto view = cudf::unpack(*packed);
  return std::make_unique<PackedTableWithStream>(
      std::make_unique<cudf::packed_table>(
          cudf::packed_table{view, std::move(*packed)}),
      cuda::stream_ref{cudaStream_t{cudaStreamDefault}},
      numRows);
}

UcxExchangeQueue::SourceId addSource(UcxExchangeQueue& queue) {
  std::lock_guard<std::mutex> l(queue.mutex());
  return queue.addSourceLocked();
}

// Puts 'table' from source 'id' into 'queue' and wakes the consumers that
// enqueueTable() returns. Returns false if the queue refused the table, which
// then stays in 'table'.
bool enqueue(
    UcxExchangeQueue& queue,
    UcxExchangeQueue::SourceId id,
    PackedTableWithStreamPtr& table) {
  std::vector<ContinuePromise> promises;
  const bool taken = queue.enqueueTable(id, table, promises);
  for (auto& promise : promises) {
    promise.setValue();
  }
  return taken;
}

// Dequeues one table for consumer 0.
PackedTableWithStreamPtr dequeue(UcxExchangeQueue& queue, bool& atEnd) {
  std::lock_guard<std::mutex> l(queue.mutex());
  ContinueFuture future = ContinueFuture::makeEmpty();
  ContinuePromise stalePromise = ContinuePromise::makeEmpty();
  return queue.dequeueLocked(0, &atEnd, &future, &stalePromise);
}

// Runs 'first' and 'second' on two threads that start together.
void runConcurrently(
    const std::function<void()>& first,
    const std::function<void()>& second) {
  folly::Baton<> startFirst;
  folly::Baton<> startSecond;
  std::thread firstThread([&] {
    startFirst.wait();
    first();
  });
  std::thread secondThread([&] {
    startSecond.wait();
    second();
  });
  startFirst.post();
  startSecond.post();
  firstThread.join();
  secondThread.join();
}

// The number of times each close race runs.
constexpr int32_t kNumRaceIterations{1'000};

} // namespace

// size() and empty() read a counter maintained under the queue's mutex rather
// than the deque, so that the Communicator progress thread can poll the depth
// lock-free. Covers that the counter stays in step with every mutation.
TEST(UcxExchangeQueueTest, sizeTracksQueuedTables) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto first = addSource(queue);
  const auto second = addSource(queue);
  queue.sourceOpened(first);
  queue.sourceOpened(second);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());

  for (int32_t expected = 1; expected <= 3; ++expected) {
    auto table = makePackedTable(100);
    ASSERT_TRUE(enqueue(queue, first, table));
    EXPECT_EQ(queue.size(), expected);
  }
  EXPECT_FALSE(queue.empty());

  bool atEnd{false};
  EXPECT_NE(dequeue(queue, atEnd), nullptr);
  EXPECT_FALSE(atEnd);
  EXPECT_EQ(queue.size(), 2);

  // A clean end leaves the queued tables to be read.
  queue.end(first, ChannelOutcome::clean());
  EXPECT_EQ(queue.size(), 2);

  // An error frees them, because a failed consumer reads nothing more.
  queue.end(second, ChannelOutcome::producerError("Producer failed"));
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.totalBytes(), 0);
}

TEST(UcxExchangeQueueTest, closeResetsSize) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto source = addSource(queue);
  queue.sourceOpened(source);
  for (int32_t i = 0; i < 2; ++i) {
    auto table = makePackedTable(100);
    ASSERT_TRUE(enqueue(queue, source, table));
  }
  EXPECT_EQ(queue.size(), 2);
  EXPECT_GT(queue.totalBytes(), 0);

  // close() discards whatever is still queued.
  queue.close();
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.totalBytes(), 0);
}

TEST(UcxExchangeQueueTest, cleanEndsFinishConsumer) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto first = addSource(queue);
  const auto second = addSource(queue);
  queue.sourceOpened(first);
  queue.sourceOpened(second);
  queue.end(first, ChannelOutcome::clean());
  queue.end(second, ChannelOutcome::clean());

  // More sources may still come.
  bool atEnd{true};
  EXPECT_EQ(dequeue(queue, atEnd), nullptr);
  EXPECT_FALSE(atEnd);

  queue.noMoreSources();
  EXPECT_EQ(dequeue(queue, atEnd), nullptr);
  EXPECT_TRUE(atEnd);
  EXPECT_EQ(queue.sourceState(first), SourceState::kEnded);
  EXPECT_EQ(queue.sourceState(second), SourceState::kEnded);
}

TEST(UcxExchangeQueueTest, noSourcesFinishesAtOnce) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  queue.noMoreSources();
  bool atEnd{false};
  EXPECT_EQ(dequeue(queue, atEnd), nullptr);
  EXPECT_TRUE(atEnd);
}

TEST(UcxExchangeQueueTest, producerErrorFailsConsumer) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto source = addSource(queue);
  queue.noMoreSources();
  queue.sourceOpened(source);
  queue.end(source, ChannelOutcome::producerError("Injected producer error"));
  // A later end of the same source does not replace the first.
  queue.end(source, ChannelOutcome::clean());

  bool atEnd{false};
  VELOX_ASSERT_THROW(dequeue(queue, atEnd), "Injected producer error");
  EXPECT_TRUE(atEnd);
  EXPECT_EQ(queue.sourceState(source), SourceState::kEnded);
}

// A consumer that closed, for instance after a satisfied LIMIT, ignores the
// errors of the producers it no longer reads.
TEST(UcxExchangeQueueTest, errorAfterCloseIsIgnored) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto source = addSource(queue);
  queue.noMoreSources();
  queue.sourceOpened(source);
  queue.close();
  queue.end(source, ChannelOutcome::producerError("Injected producer error"));

  bool atEnd{false};
  EXPECT_EQ(dequeue(queue, atEnd), nullptr);
  EXPECT_TRUE(atEnd);
  EXPECT_EQ(queue.sourceState(source), SourceState::kReleased);
}

TEST(UcxExchangeQueueTest, terminalClientErrorIsNotMaskedByClose) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto source = addSource(queue);
  queue.sourceOpened(source);
  queue.end(
      source,
      ChannelOutcome::transportError(
          "downstream CUDA stream synchronization failed"));
  queue.close();

  bool atEnd{false};
  VELOX_ASSERT_THROW(
      dequeue(queue, atEnd), "downstream CUDA stream synchronization failed");
  EXPECT_TRUE(atEnd);
}

TEST(UcxExchangeQueueTest, closeTellsServersOfSourcesNotEnded) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto registered = addSource(queue);
  const auto open = addSource(queue);
  const auto ended = addSource(queue);
  queue.sourceOpened(open);
  queue.sourceOpened(ended);
  queue.end(ended, ChannelOutcome::clean());

  EXPECT_THAT(queue.close(), testing::ElementsAre(registered, open));
  EXPECT_TRUE(queue.isClosed());
  for (const auto source : {registered, open, ended}) {
    EXPECT_EQ(queue.sourceState(source), SourceState::kReleased);
  }
  // Every entry is Released already.
  EXPECT_THAT(queue.close(), testing::IsEmpty());
}

TEST(UcxExchangeQueueTest, tableAfterCloseIsFreed) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto source = addSource(queue);
  queue.sourceOpened(source);
  queue.close();

  auto table = makePackedTable(100);
  EXPECT_FALSE(enqueue(queue, source, table));
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_EQ(queue.totalBytes(), 0);
}

TEST(UcxExchangeQueueTest, tableAfterOtherSourceFailedIsFreed) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto failed = addSource(queue);
  const auto running = addSource(queue);
  queue.sourceOpened(failed);
  queue.sourceOpened(running);
  queue.end(failed, ChannelOutcome::transportError("Injected receive error"));

  auto table = makePackedTable(100);
  EXPECT_FALSE(enqueue(queue, running, table));
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(queue.size(), 0);
  EXPECT_EQ(queue.totalBytes(), 0);
}

TEST(UcxExchangeQueueTest, tableOfUnopenedSourceIsRefused) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  {
    const auto source = addSource(queue);
    auto table = makePackedTable(100);
    EXPECT_FALSE(enqueue(queue, source, table));
    EXPECT_NE(table, nullptr);
  }
  {
    const auto source = addSource(queue);
    queue.sourceOpened(source);
    queue.end(source, ChannelOutcome::clean());
    auto table = makePackedTable(100);
    EXPECT_FALSE(enqueue(queue, source, table));
    EXPECT_NE(table, nullptr);
  }
  EXPECT_EQ(queue.size(), 0);
  EXPECT_EQ(queue.totalBytes(), 0);
}

// Only one of a clean end and a close is counted, and the consumer finishes
// without an error either way.
TEST(UcxExchangeQueueTest, cleanEndRacesClose) {
  for (int32_t i = 0; i < kNumRaceIterations; ++i) {
    UcxExchangeQueue queue{/*numberOfConsumers=*/1};
    const auto source = addSource(queue);
    queue.sourceOpened(source);
    queue.noMoreSources();

    runConcurrently(
        [&] { queue.end(source, ChannelOutcome::clean()); },
        [&] { queue.close(); });

    // close() releases an Ended entry, too.
    ASSERT_EQ(queue.sourceState(source), SourceState::kReleased);
    bool atEnd{false};
    ASSERT_EQ(dequeue(queue, atEnd), nullptr);
    ASSERT_TRUE(atEnd);
  }
}

// The first decision wins: an error decided before the close fails the
// consumer, and one that comes after it is ignored. Either way the consumer
// ends, and the source is Released.
TEST(UcxExchangeQueueTest, producerErrorRacesClose) {
  // Every dequeue that throws logs the error; keep the loop quiet.
  gflags::FlagSaver flagSaver;
  FLAGS_minloglevel = google::GLOG_FATAL;
  for (int32_t i = 0; i < kNumRaceIterations; ++i) {
    UcxExchangeQueue queue{/*numberOfConsumers=*/1};
    const auto source = addSource(queue);
    queue.sourceOpened(source);
    queue.noMoreSources();

    runConcurrently(
        [&] {
          queue.end(
              source, ChannelOutcome::producerError("Injected producer error"));
        },
        [&] { queue.close(); });

    ASSERT_EQ(queue.sourceState(source), SourceState::kReleased);
    bool atEnd{false};
    try {
      ASSERT_EQ(dequeue(queue, atEnd), nullptr);
    } catch (const VeloxException& e) {
      ASSERT_THAT(e.message(), testing::HasSubstr("Injected producer error"));
    }
    ASSERT_TRUE(atEnd);
  }
}

// A table that arrives as the consumer closes is either refused, and freed by
// its source, or taken and then freed by close().
TEST(UcxExchangeQueueTest, tableInFlightRacesClose) {
  for (int32_t i = 0; i < kNumRaceIterations; ++i) {
    UcxExchangeQueue queue{/*numberOfConsumers=*/1};
    const auto source = addSource(queue);
    queue.sourceOpened(source);
    queue.noMoreSources();

    auto table = makePackedTable(100);
    bool taken{false};
    bool keptByCaller{false};
    runConcurrently(
        [&] {
          taken = enqueue(queue, source, table);
          keptByCaller = table != nullptr;
          table.reset();
        },
        [&] { queue.close(); });

    ASSERT_NE(taken, keptByCaller);
    ASSERT_EQ(queue.size(), 0);
    ASSERT_EQ(queue.totalBytes(), 0);
    bool atEnd{false};
    ASSERT_EQ(dequeue(queue, atEnd), nullptr);
    ASSERT_TRUE(atEnd);
  }
}

// A source added after the close only tells its producer that the consumer is
// gone; the consumer does not wait for it.
TEST(UcxExchangeQueueTest, sourceAddedAfterCloseIsReleased) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  const auto source = addSource(queue);
  queue.sourceOpened(source);
  queue.end(source, ChannelOutcome::clean());
  queue.noMoreSources();
  bool atEnd{false};
  EXPECT_EQ(dequeue(queue, atEnd), nullptr);
  EXPECT_TRUE(atEnd);

  queue.close();
  UcxExchangeQueue::SourceId lateSource;
  {
    std::lock_guard<std::mutex> l(queue.mutex());
    lateSource = queue.addReleasedSourceLocked();
  }
  EXPECT_EQ(queue.sourceState(lateSource), SourceState::kReleased);
  atEnd = false;
  EXPECT_EQ(dequeue(queue, atEnd), nullptr);
  EXPECT_TRUE(atEnd);
}

TEST_P(UcxExchangeTest, basicTest) {
  VLOG(3) << "+ UcxExchangeTest::basicTest";
  ExchangeTestParams p = GetParam();

  // Skip wide table tests - UcxPartitionedOutputMock only supports narrow
  // tables
  if (shouldSkipWideTable()) {
    GTEST_SKIP()
        << "basicTest skipped for WideTable - uses UcxPartitionedOutputMock";
  }

  int numUpstreamTasks = p.numUpstreamTasks;

  // Use unique task prefix to avoid collisions between parametrized tests
  const std::string taskPrefix = getUniqueTaskPrefix();
  std::vector<std::string> srcTaskIds;

  std::vector<std::shared_ptr<UcxPartitionedOutputMock>> sourceMocks;

  // Create n upstream tasks.
  for (int i = 0; i < numUpstreamTasks; i++) {
    const std::string srcTaskId = taskPrefix + "sourceTask" + std::to_string(i);
    srcTaskIds.push_back(srcTaskId);
    auto srcTask =
        createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);

    // tell the queue manager that a new source task exists.
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        p.numPartitions,
        p.numSrcDrivers);

    sourceMocks.emplace_back(
        std::make_shared<UcxPartitionedOutputMock>(
            srcTaskId,
            p.numSrcDrivers,
            p.numPartitions,
            p.numChunks,
            p.numRowsPerChunk));
  }

  // Create one sink task per partition to receive data from each partition
  std::vector<std::shared_ptr<SinkDriverMock>> sinkDrivers;
  for (int partitionId = 0; partitionId < p.numPartitions; ++partitionId) {
    const std::string sinkTaskId =
        taskPrefix + "sinkTask" + std::to_string(partitionId);
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(
        sinkTaskId, UcxTestData::kTestRowType, partitionId, exchangeNodeId);

    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, p.numDstDrivers);

    // Add remote splits for all upstream tasks to this partition
    std::vector<facebook::velox::exec::Split> splits;
    for (int i = 0; i < numUpstreamTasks; i++) {
      splits.emplace_back(remoteSplit(srcTaskIds[i], partitionId));
    }
    sinkDriver->addSplits(splits);

    sinkDrivers.push_back(sinkDriver);
  }

  // Start the mocks.
  VLOG(3) << "Starting source tasks";
  for (int i = 0; i < numUpstreamTasks; i++) {
    sourceMocks[i]->run();
  }
  VLOG(3) << "Starting " << p.numPartitions << " sink tasks";
  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->run();
  }

  for (int i = 0; i < numUpstreamTasks; i++) {
    sourceMocks[i]->joinThreads();
  }
  VLOG(3) << "Source tasks done.";
  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->joinThreads();
  }
  VLOG(3) << "All sink tasks done.";

  // Total rows received across all partitions should equal total rows sent.
  // UcxPartitionedOutputMock sends numChunks * numRowsPerChunk to EACH
  // partition, so total rows = chunks * rowsPerChunk * partitions *
  // upstreamTasks * srcDrivers
  size_t expectedRows = static_cast<size_t>(p.numChunks) * p.numRowsPerChunk *
      p.numPartitions * numUpstreamTasks * p.numSrcDrivers;
  size_t totalReceivedRows = 0;
  for (auto& sinkDriver : sinkDrivers) {
    totalReceivedRows += sinkDriver->numRows();
  }

  GTEST_ASSERT_EQ(expectedRows, totalReceivedRows);

  // Remove the srcTasks from the queue manager, so queue get freed
  for (const auto& srcTaskId : srcTaskIds) {
    queueManager()->removeTask(srcTaskId);
  }

  VLOG(3) << "- UcxExchangeTest::basicTest";
}

TEST_P(UcxExchangeTest, dataIntegrityTest) {
  VLOG(3) << "+ UcxExchangeTest::dataIntegrityTest";
  ExchangeTestParams p = GetParam();

  // Skip wide table tests - UcxPartitionedOutputMock only supports narrow
  // tables
  if (shouldSkipWideTable()) {
    GTEST_SKIP()
        << "dataIntegrityTest skipped for WideTable - uses UcxPartitionedOutputMock";
  }

  int numUpstreamTasks = p.numUpstreamTasks;

  // Use unique task prefix to avoid collisions between parametrized tests
  const std::string taskPrefix = getUniqueTaskPrefix();
  std::vector<std::string> srcTaskIds;

  // Create some reference data to send which we will check against at the
  // receiver
  std::shared_ptr<UcxTestData> dataToSend = std::make_shared<UcxTestData>();
  dataToSend->initialize(p.numRowsPerChunk);

  std::vector<std::shared_ptr<UcxPartitionedOutputMock>> sourceMocks;

  // Create n upstream tasks.
  for (int i = 0; i < numUpstreamTasks; i++) {
    const std::string srcTaskId = taskPrefix + "sourceTask" + std::to_string(i);
    srcTaskIds.push_back(srcTaskId);
    auto srcTask =
        createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);

    // tell the queue manager that a new source task exists.
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        p.numPartitions,
        p.numSrcDrivers);

    // Mock the UcxPartitionedOutput operator, it will produce numChunks of
    // data each containing numRowsPerChunk of data copied from the UcxTestData
    // object data
    sourceMocks.emplace_back(
        std::make_shared<UcxPartitionedOutputMock>(
            srcTaskId,
            p.numSrcDrivers,
            p.numPartitions,
            p.numChunks,
            p.numRowsPerChunk,
            dataToSend));
  }

  // Create one sink task per partition to receive data from each partition
  std::vector<std::shared_ptr<SinkDriverMock>> sinkDrivers;
  for (int partitionId = 0; partitionId < p.numPartitions; ++partitionId) {
    const std::string sinkTaskId =
        taskPrefix + "sinkTask" + std::to_string(partitionId);
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(
        sinkTaskId, UcxTestData::kTestRowType, partitionId, exchangeNodeId);

    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, p.numDstDrivers, dataToSend);

    // Add remote splits for all upstream tasks to this partition
    std::vector<facebook::velox::exec::Split> splits;
    for (int i = 0; i < numUpstreamTasks; i++) {
      splits.emplace_back(remoteSplit(srcTaskIds[i], partitionId));
    }
    sinkDriver->addSplits(splits);

    sinkDrivers.push_back(sinkDriver);
  }

  // Start the mocks.
  VLOG(3) << "Starting source tasks";
  for (int i = 0; i < numUpstreamTasks; i++) {
    sourceMocks[i]->run();
  }

  VLOG(3) << "Starting " << p.numPartitions << " sink tasks";
  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->run();
  }

  for (int i = 0; i < numUpstreamTasks; i++) {
    sourceMocks[i]->joinThreads();
  }
  VLOG(3) << "Source tasks done.";

  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->joinThreads();
  }
  VLOG(3) << "All sink tasks done.";

  // Remove the srcTasks from the queue manager, so queue get freed
  for (const auto& srcTaskId : srcTaskIds) {
    queueManager()->removeTask(srcTaskId);
  }

  // Check data integrity across all partitions
  bool allDataValid = true;
  for (auto& sinkDriver : sinkDrivers) {
    if (!sinkDriver->dataIsValid()) {
      allDataValid = false;
      break;
    }
  }

  VLOG(3) << "- UcxExchangeTest::dataIntegrityTest";
  GTEST_ASSERT_EQ(allDataValid, true);
}

TEST_P(UcxExchangeTest, bandwidthTest) {
  // Test to measure the bandwidth at the Velox level

  // Skip by default, enable with environment variable
  if (!std::getenv("RUN_BANDWIDTH_TEST")) {
    GTEST_SKIP()
        << "Bandwidth test skipped. Set RUN_BANDWIDTH_TEST=1 to enable.";
  }

  // Skip wide table tests - UcxPartitionedOutputMock only supports narrow
  // tables
  if (shouldSkipWideTable()) {
    GTEST_SKIP()
        << "bandwidthTest skipped for WideTable - uses UcxPartitionedOutputMock";
  }

  VLOG(3) << "+ UcxExchangeTest::bandwidthTest";
  ExchangeTestParams p = GetParam();
  int numUpstreamTasks = p.numUpstreamTasks;

  // Use unique task prefix to avoid collisions between parametrized tests
  const std::string taskPrefix = getUniqueTaskPrefix();
  std::vector<std::string> srcTaskIds;

  // Create some reference data to send which we will check against at the
  // receiver
  std::shared_ptr<UcxTestData> dataToSend = std::make_shared<UcxTestData>();
  dataToSend->initialize(p.numRowsPerChunk);

  std::vector<std::shared_ptr<UcxPartitionedOutputMock>> sourceMocks;

  // Create n upstream tasks.
  for (int i = 0; i < numUpstreamTasks; i++) {
    const std::string srcTaskId = taskPrefix + "sourceTask" + std::to_string(i);
    srcTaskIds.push_back(srcTaskId);

    // Create a source task with a large maximum queue size so that we don't
    // block sending
    auto srcTask = createSourceTask(
        srcTaskId, pool_, UcxTestData::kTestRowType, FOUR_GBYTES * 10);
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        p.numPartitions,
        p.numSrcDrivers);

    // Mock the UcxPartitionedOutput operator, it will produce numChunks of
    // data each containing numRowsPerChunk of data copied from the UcxTestData
    // object data
    sourceMocks.emplace_back(
        std::make_shared<UcxPartitionedOutputMock>(
            srcTaskId,
            p.numSrcDrivers,
            p.numPartitions,
            p.numChunks,
            p.numRowsPerChunk,
            dataToSend));
  }

  const std::string sinkTaskId = taskPrefix + "sinkTask";
  int partitionId = 0;
  core::PlanNodeId exchangeNodeId;
  auto sinkTask = createExchangeTask(
      sinkTaskId, UcxTestData::kTestRowType, partitionId, exchangeNodeId);

  SinkDriverMock sinkDriver(
      sinkTask, p.numDstDrivers, nullptr /* Don't check data too slow*/);

  // create n remote splits and add it to the sink driver mock.
  std::vector<facebook::velox::exec::Split> splits;
  for (int i = 0; i < numUpstreamTasks; i++) {
    splits.emplace_back(remoteSplit(srcTaskIds[i], partitionId));
  }
  sinkDriver.addSplits(splits);

  // Start the mocks.
  VLOG(3) << "Starting source tasks";
  for (int i = 0; i < numUpstreamTasks; i++) {
    sourceMocks[i]->run();
  }
  for (int i = 0; i < numUpstreamTasks; i++) {
    sourceMocks[i]->joinThreads();
  }
  VLOG(3) << "Source tasks done.";

  // Only starting receiving when sender is done, note this can be dangeous
  // if the total data send is larger than the queue as the source thread
  // will block and we will never arrive here

  VLOG(3) << "Starting sink task";
  std::chrono::time_point<std::chrono::high_resolution_clock> send_start =
      std::chrono::high_resolution_clock::now();

  sinkDriver.run();
  sinkDriver.joinThreads();
  std::chrono::time_point<std::chrono::high_resolution_clock> send_end =
      std::chrono::high_resolution_clock::now();

  auto rx_bytes = sinkDriver.numBytes();
  auto duration = send_end - send_start;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  auto throughput = (float)rx_bytes / (float)micros;
  VLOG(3)
      << "*** duration: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms ";
  VLOG(3) << "*** MBytes " << (float)rx_bytes / (float)(1024 * 1024);
  VLOG(0) << "*** throughput: " << throughput << " MByte/s";

  VLOG(3) << "Sink task done.";

  // Remove the srcTasks from the queue manager, so queue get freed
  for (const auto& srcTaskId : srcTaskIds) {
    queueManager()->removeTask(srcTaskId);
  }

  VLOG(3) << "- UcxExchangeTest::bandwidth";
  GTEST_ASSERT_EQ(sinkDriver.dataIsValid(), true);
}

// Test using real UcxPartitionedOutput operator via SourceDriverMock
TEST_P(UcxExchangeTest, realPartitionedOutputTest) {
  VLOG(3) << "+ UcxExchangeTest::realPartitionedOutputTest";
  ExchangeTestParams p = GetParam();

  // Wide table multi-partition tests work by using
  // structs_column_view::get_sliced_child() in
  // UcxExchangeServer/UcxExchangeProtocol to get STRUCT children with the
  // parent's offset/size applied after cudf::split.

  // Use unique task prefix to avoid collisions between parametrized tests
  const std::string taskPrefix = getUniqueTaskPrefix();

  // For this test, we use a single upstream task to keep it simple
  const int numUpstreamTasks = 1;
  const std::string srcTaskId = taskPrefix + "sourceTask0";

  // Get the row type based on the table type
  auto rowType = getRowType(p.tableType);

  // Specify partition keys when numPartitions > 1 to enable hash partitioning.
  // Use "c0" for narrow tables (INTEGER column) or "int32_col" for wide tables.
  std::vector<std::string> partitionKeys;
  if (p.numPartitions > 1) {
    partitionKeys = {p.tableType == TableType::WIDE ? "int32_col" : "c0"};
  }

  // Create source task with PartitionedOutput plan node
  auto srcTask = createPartitionedOutputTask(
      srcTaskId, pool_, rowType, p.numPartitions, partitionKeys);

  // Tell the queue manager that a new source task exists
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      p.numPartitions,
      p.numSrcDrivers);

  // Create table generator for wide tables, nullptr for narrow tables
  std::shared_ptr<BaseTableGenerator> tableGenerator;
  if (p.tableType == TableType::WIDE) {
    auto wideTable = std::make_shared<WideTestTable>();
    wideTable->initialize(p.numRowsPerChunk);
    tableGenerator = wideTable;
  }

  // Create SourceDriverMock to drive real UcxPartitionedOutput operators
  auto sourceDriver = std::make_shared<SourceDriverMock>(
      srcTask, p.numSrcDrivers, p.numChunks, p.numRowsPerChunk, tableGenerator);

  // Create one sink task per partition to receive data from each partition
  std::vector<std::shared_ptr<SinkDriverMock>> sinkDrivers;
  for (int partitionId = 0; partitionId < p.numPartitions; ++partitionId) {
    const std::string sinkTaskId =
        taskPrefix + "sinkTask" + std::to_string(partitionId);
    core::PlanNodeId exchangeNodeId;
    auto sinkTask =
        createExchangeTask(sinkTaskId, rowType, partitionId, exchangeNodeId);

    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, p.numDstDrivers);

    // Add remote split for this partition
    std::vector<facebook::velox::exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, partitionId));
    sinkDriver->addSplits(splits);

    sinkDrivers.push_back(sinkDriver);
  }

  // Start the drivers
  VLOG(3) << "Starting source task with real UcxPartitionedOutput";
  sourceDriver->run();

  VLOG(3) << "Starting " << p.numPartitions << " sink tasks";
  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->run();
  }

  sourceDriver->joinThreads();
  VLOG(3) << "Source task done.";

  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->joinThreads();
  }
  VLOG(3) << "All sink tasks done.";

  // Total rows received across all partitions should equal total rows sent
  size_t expectedRows = p.numChunks * p.numRowsPerChunk * p.numSrcDrivers;
  size_t totalReceivedRows = 0;
  for (auto& sinkDriver : sinkDrivers) {
    totalReceivedRows += sinkDriver->numRows();
  }

  VLOG(3) << "Expected rows: " << expectedRows
          << ", received rows: " << totalReceivedRows;
  GTEST_ASSERT_EQ(expectedRows, totalReceivedRows);

  // Cleanup
  queueManager()->removeTask(srcTaskId);

  VLOG(3) << "- UcxExchangeTest::realPartitionedOutputTest";
}

// Test using real UcxPartitionedOutput with data integrity verification.
// This test:
// 1. Creates reference data (UcxTestData or WideTestTable) - same as
// dataIntegrityTest
// 2. For narrow tables with multi-partition: partitions that data using
// cudf::hash_partition
//    (same algorithm as UcxPartitionedOutput) to create per-partition
//    reference data
// 3. Sends data through SourceDriverMock (which uses UcxPartitionedOutput)
// 4. Each SinkDriverMock verifies received data against its partition's
//    expected data using row-by-row comparison
TEST_P(UcxExchangeTest, realPartitionedOutputDataIntegrityTest) {
  VLOG(3) << "+ UcxExchangeTest::realPartitionedOutputDataIntegrityTest";
  ExchangeTestParams p = GetParam();

  // Wide table multi-partition tests work by using
  // structs_column_view::get_sliced_child() in
  // UcxExchangeServer/UcxExchangeProtocol to get STRUCT children with the
  // parent's offset/size applied after cudf::split.

  // Use unique task prefix to avoid collisions between parametrized tests
  const std::string taskPrefix = getUniqueTaskPrefix();

  // For this test, use a single upstream task and single driver for simplicity
  // This allows deterministic data verification
  const int numUpstreamTasks = 1;
  const int numSrcDrivers = 1;
  const std::string srcTaskId = taskPrefix + "sourceTask0";

  // Get the row type based on the table type
  auto rowType = getRowType(p.tableType);

  // Create reference data that will be sent - UcxTestData for narrow,
  // WideTestTable for wide
  std::shared_ptr<BaseTableGenerator> tableGenerator;
  if (p.tableType == TableType::WIDE) {
    auto wideTable = std::make_shared<WideTestTable>();
    wideTable->initialize(p.numRowsPerChunk);
    tableGenerator = wideTable;
  } else {
    auto dataToSend = std::make_shared<UcxTestData>();
    dataToSend->initialize(p.numRowsPerChunk);
    tableGenerator = dataToSend;
  }

  // Specify partition keys when numPartitions > 1 to enable hash partitioning.
  // Use "c0" for narrow tables (column index 0) or "int32_col" for wide tables
  // (column index 2).
  std::vector<std::string> partitionKeys;
  std::vector<cudf::size_type> partitionKeyIndices;
  if (p.numPartitions > 1) {
    if (p.tableType == TableType::WIDE) {
      partitionKeys = {"int32_col"};
      partitionKeyIndices = {2}; // int32_col is column 2 in wide table
    } else {
      partitionKeys = {"c0"};
      partitionKeyIndices = {0}; // c0 is column 0 in narrow table
    }
  }

  // Create per-partition reference data by applying cudf::hash_partition
  // to the source data - same algorithm as UcxPartitionedOutput uses
  auto stream = cuda::stream_ref{cudaStream_t{cudaStreamDefault}};
  std::vector<std::shared_ptr<BaseTableGenerator>> partitionedDataToVerify(
      p.numPartitions);

  // For narrow tables with multi-partition, we can compute per-partition
  // reference data For wide tables or single partition, we use the
  // tableGenerator directly
  bool canVerifyDataIntegrity = true;

  if (p.numPartitions > 1 && !partitionKeyIndices.empty()) {
    // Multi-partition: skip data integrity verification.
    // cudf::hash_partition does not guarantee deterministic row ordering within
    // partitions, so the reference data created here may have different row
    // order than the data sent through UcxPartitionedOutput::hashPartition(),
    // even though both use the same input data and hash function. Row count
    // verification still confirms all data is transferred correctly.
    canVerifyDataIntegrity = false;
    VLOG(3) << "Multi-partition test: skipping data integrity verification "
            << "(hash_partition row order is not deterministic)";
  } else {
    // Single partition: all data goes to partition 0, use tableGenerator
    // directly
    partitionedDataToVerify[0] = tableGenerator;
  }

  // Create source task with PartitionedOutput plan node
  auto srcTask = createPartitionedOutputTask(
      srcTaskId, pool_, rowType, p.numPartitions, partitionKeys);

  // Tell the queue manager that a new source task exists
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      p.numPartitions,
      numSrcDrivers);

  // Create SourceDriverMock with the tableGenerator
  auto sourceDriver = std::make_shared<SourceDriverMock>(
      srcTask, numSrcDrivers, p.numChunks, p.numRowsPerChunk, tableGenerator);

  // Create one SinkDriverMock per partition, each with its partition's
  // expected data for row-by-row verification (if available)
  std::vector<std::shared_ptr<SinkDriverMock>> sinkDrivers;
  for (int partitionId = 0; partitionId < p.numPartitions; ++partitionId) {
    const std::string sinkTaskId =
        taskPrefix + "sinkTask" + std::to_string(partitionId);
    core::PlanNodeId exchangeNodeId;
    auto sinkTask =
        createExchangeTask(sinkTaskId, rowType, partitionId, exchangeNodeId);

    // Pass the partitioned reference data for this partition (may be nullptr
    // for wide multi-partition)
    auto sinkDriver = std::make_shared<SinkDriverMock>(
        sinkTask,
        p.numDstDrivers,
        canVerifyDataIntegrity ? partitionedDataToVerify[partitionId]
                               : nullptr);

    std::vector<facebook::velox::exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, partitionId));
    sinkDriver->addSplits(splits);

    sinkDrivers.push_back(sinkDriver);
  }

  // Start the drivers
  VLOG(3) << "Starting source task with real UcxPartitionedOutput";
  sourceDriver->run();

  VLOG(3) << "Starting " << p.numPartitions << " sink tasks";
  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->run();
  }

  sourceDriver->joinThreads();
  VLOG(3) << "Source task done.";

  for (auto& sinkDriver : sinkDrivers) {
    sinkDriver->joinThreads();
  }
  VLOG(3) << "All sink tasks done.";

  // Verify total row count
  size_t expectedTotalRows = p.numChunks * p.numRowsPerChunk * numSrcDrivers;
  size_t totalReceivedRows = 0;
  for (auto& sinkDriver : sinkDrivers) {
    totalReceivedRows += sinkDriver->numRows();
  }
  GTEST_ASSERT_EQ(expectedTotalRows, totalReceivedRows);

  // Verify data integrity - SinkDriverMock sets dataIsValid() to false
  // if any row doesn't match the reference data
  if (canVerifyDataIntegrity) {
    bool allDataValid = true;
    for (int partId = 0; partId < p.numPartitions; ++partId) {
      if (!sinkDrivers[partId]->dataIsValid()) {
        VLOG(0) << "Partition " << partId << ": data validation failed";
        allDataValid = false;
      } else {
        VLOG(3) << "Partition " << partId << ": data validated successfully";
      }
    }

    GTEST_ASSERT_EQ(allDataValid, true);
  } else {
    VLOG(3)
        << "Data integrity verification skipped for wide table with multi-partition";
  }

  // Cleanup
  queueManager()->removeTask(srcTaskId);

  VLOG(3) << "- UcxExchangeTest::realPartitionedOutputDataIntegrityTest";
}

// A consumer that fails in a case that expects success must fail the case.
// SinkDriverMock drives its exchanges on bare threads and catches what they
// throw, and the cases that check only dataIsValid() would otherwise pass
// with rows missing.
TEST_F(UcxExchangeFocusedTest, sinkDriverMockReportsUnexpectedConsumerError) {
  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "abortedBeforeSink";
  auto srcTask = createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  srcTask->requestAbort().wait();
  queueManager()->removeTask(srcTaskId);

  core::PlanNodeId exchangeNodeId;
  auto sinkTask = createExchangeTask(
      taskPrefix + "sink",
      UcxTestData::kTestRowType,
      /*partitionId=*/0,
      exchangeNodeId);
  auto sinkDriver =
      std::make_shared<SinkDriverMock>(sinkTask, /*numDrivers=*/1);
  std::vector<exec::Split> splits;
  splits.emplace_back(remoteSplit(srcTaskId, /*partitionId=*/0));
  sinkDriver->addSplits(splits);

  EXPECT_NONFATAL_FAILURE(
      {
        sinkDriver->run();
        sinkDriver->joinThreads();
      },
      "Producer failed before sending all its data");
}

// Focused regression test for shared UcxExchangeClient ownership. This
// intentionally creates and seeds the client directly, bypassing the normal
// task-split path, to isolate close behavior after the client is populated.
// Closing one operator must not close the shared client while another operator
// still needs to drain data.
TEST_F(UcxExchangeFocusedTest, sharedClientSurvivesOneExchangeClose) {
  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "sharedClientSrc";
  const std::string sinkTaskId = taskPrefix + "sharedClientSink";
  const int numPartitions = 1;
  const int partitionId = 0;
  const int numSourceDrivers = 1;
  const int numSinkDrivers = 2;
  const int numChunks = 5;
  const int numRowsPerChunk = 1000;

  auto rowType = UcxTestData::kTestRowType;
  auto srcTask = createSourceTask(srcTaskId, pool_, rowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      numPartitions,
      numSourceDrivers);

  auto sourceMock = std::make_shared<UcxPartitionedOutputMock>(
      srcTaskId, numSourceDrivers, numPartitions, numChunks, numRowsPerChunk);
  sourceMock->run();
  sourceMock->joinThreads();

  core::PlanNodeId exchangeNodeId;
  auto sinkTask =
      createExchangeTask(sinkTaskId, rowType, partitionId, exchangeNodeId);
  auto exchangeClient = std::make_shared<UcxExchangeClient>(
      sinkTask->taskId(), sinkTask->destination(), numSinkDrivers);

  auto split = remoteSplit(srcTaskId, partitionId);
  auto remoteConnectorSplit =
      std::dynamic_pointer_cast<exec::RemoteConnectorSplit>(
          split.connectorSplit);
  ASSERT_NE(remoteConnectorSplit, nullptr);
  exchangeClient->addRemoteTaskId(remoteConnectorSplit->taskId);
  exchangeClient->noMoreRemoteTasks();

  const uint32_t pipelineId = 0;
  const uint32_t partition = 0;
  auto planNode = sinkTask->planFragment().planNode;
  auto closingDriverCtx = std::make_shared<DriverCtx>(
      sinkTask, 0, pipelineId, kUngroupedGroupId, partition);
  auto drainingDriverCtx = std::make_shared<DriverCtx>(
      sinkTask, 1, pipelineId, kUngroupedGroupId, partition);

  UcxExchange closingExchange(
      0, closingDriverCtx.get(), planNode, exchangeClient);
  UcxExchange drainingExchange(
      1, drainingDriverCtx.get(), planNode, exchangeClient);

  closingExchange.close();

  uint64_t rowsReceived = 0;
  while (true) {
    ContinueFuture future;
    auto blocked = drainingExchange.isBlocked(&future);
    if (blocked != BlockingReason::kNotBlocked) {
      future.wait();
      continue;
    }

    RowVectorPtr result = drainingExchange.getOutput();
    if (result) {
      auto cudfResult =
          std::dynamic_pointer_cast<cudf_velox::CudfVector>(result);
      ASSERT_NE(cudfResult, nullptr);
      rowsReceived += cudfResult->getTableView().num_rows();
    }

    if (drainingExchange.isFinished()) {
      break;
    }
  }
  drainingExchange.close();

  EXPECT_EQ(rowsReceived, static_cast<uint64_t>(numChunks) * numRowsPerChunk);

  queueManager()->removeTask(srcTaskId);
}

namespace {

// Drives 'exchange' until it finishes or has returned at least 'maxRows' rows,
// and returns the rows it read. Fails if a wait for data takes longer than
// 'maxWait', so that a stream that never ends fails the case instead of
// hanging the binary.
uint64_t drainToEnd(
    UcxExchange& exchange,
    uint64_t maxRows,
    std::chrono::seconds maxWait) {
  uint64_t numRows{0};
  while (numRows < maxRows && !exchange.isFinished()) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    if (exchange.isBlocked(&future) != BlockingReason::kNotBlocked) {
      if (!std::move(future).wait(maxWait)) {
        VELOX_FAIL("Exchange made no progress");
      }
      continue;
    }
    if (auto result = exchange.getOutput()) {
      auto cudfResult =
          std::dynamic_pointer_cast<cudf_velox::CudfVector>(result);
      VELOX_CHECK_NOT_NULL(cudfResult);
      numRows += cudfResult->getTableView().num_rows();
    }
  }
  return numRows;
}

} // namespace

// A producer whose connection closes before it sent all its output must fail
// its consumer. The close reaches the consumer through
// EndpointRef::closeAndDrainCommunicators(), as when the producer's worker
// dies; treating it as a clean end of stream would drop the rest of the rows.
TEST_F(UcxExchangeFocusedTest, endpointClosedMidStreamFailsConsumer) {
  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "endpointClosedSrc";
  const std::string sinkTaskId = taskPrefix + "endpointClosedSink";
  const int partitionId = 0;

  auto rowType = UcxTestData::kTestRowType;
  auto srcTask = createSourceTask(srcTaskId, pool_, rowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  // Removes the producer even when an assertion ends the case early: the
  // process-wide queue manager would otherwise keep its Task until static
  // destruction, after the memory manager it allocates from is gone.
  auto removeProducer = folly::makeGuard([&]() {
    srcTask->requestAbort().wait();
    queueManager()->removeTask(srcTaskId);
  });
  // One page, and no noMoreData(): the producer is still running.
  queueManager()->enqueue(
      srcTaskId, partitionId, makePackedColumns(100), /*numRows=*/100);

  core::PlanNodeId exchangeNodeId;
  auto sinkTask =
      createExchangeTask(sinkTaskId, rowType, partitionId, exchangeNodeId);
  auto exchangeClient = std::make_shared<UcxExchangeClient>(
      sinkTask->taskId(), sinkTask->destination(), /*numberOfConsumers=*/1);
  auto split = remoteSplit(srcTaskId, partitionId);
  exchangeClient->addRemoteTaskId(
      std::dynamic_pointer_cast<exec::RemoteConnectorSplit>(
          split.connectorSplit)
          ->taskId);
  exchangeClient->noMoreRemoteTasks();

  auto driverCtx = std::make_shared<DriverCtx>(
      sinkTask,
      /*driverId=*/0,
      /*pipelineId=*/0,
      kUngroupedGroupId,
      /*partitionId=*/0);
  UcxExchange exchange(
      0, driverCtx.get(), sinkTask->planFragment().planNode, exchangeClient);

  EXPECT_EQ(
      drainToEnd(exchange, /*maxRows=*/100, std::chrono::seconds(30)), 100);

  // Close the consumer's connection to the producer the way a dead peer does.
  std::shared_ptr<EndpointRef> endpoint;
  ASSERT_TRUE(runOnCommunicator([&endpoint] {
    auto communicator = Communicator::getInstance();
    endpoint = communicator->assocEndpointRef(
        nullptr, HostPort{"127.0.0.1", communicator->getListenerPort()});
    if (endpoint != nullptr) {
      communicator->deferEndpointCleanup(endpoint);
    }
  }));
  ASSERT_NE(endpoint, nullptr);

  VELOX_ASSERT_THROW(
      drainToEnd(
          exchange,
          /*maxRows=*/std::numeric_limits<uint64_t>::max(),
          std::chrono::seconds(30)),
      "Connection to producer closed before it sent all its data, host 127.0.0.1:");
  exchange.close();
}

// A producer the consumer cannot connect to must fail the consumer, rather than
// end its stream as if the producer had sent nothing. The host name below is
// rejected by getaddrinfo locally, without a DNS round trip, so endpoint
// creation fails at once.
TEST_F(UcxExchangeFocusedTest, unresolvableProducerHostFailsConsumer) {
  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string sinkTaskId = taskPrefix + "unreachableSink";

  core::PlanNodeId exchangeNodeId;
  auto sinkTask = createExchangeTask(
      sinkTaskId, UcxTestData::kTestRowType, /*partitionId=*/0, exchangeNodeId);
  auto exchangeClient = std::make_shared<UcxExchangeClient>(
      sinkTask->taskId(), sinkTask->destination(), /*numberOfConsumers=*/1);
  const auto url = fmt::format(
      "http://bad_host!:{}/v1/task/{}/results/0",
      UcxExchangeTestEnvironment::splitUrlPort(),
      taskPrefix + "unreachableProducer");
  exchangeClient->addRemoteTaskId(url);
  exchangeClient->noMoreRemoteTasks();

  auto driverCtx = std::make_shared<DriverCtx>(
      sinkTask,
      /*driverId=*/0,
      /*pipelineId=*/0,
      kUngroupedGroupId,
      /*partitionId=*/0);
  UcxExchange exchange(
      0, driverCtx.get(), sinkTask->planFragment().planNode, exchangeClient);

  VELOX_ASSERT_THROW(
      drainToEnd(
          exchange,
          /*maxRows=*/std::numeric_limits<uint64_t>::max(),
          std::chrono::seconds(30)),
      "Failed to connect to producer, host bad_host!:");
  exchange.close();
}

// A source that the client adds after it closed still connects, to send the
// close message. If its producer's host cannot be resolved, the source ends at
// once, without waiting for its close deadline, and the consumer does not
// fail.
TEST_F(UcxExchangeFocusedTest, closedSourceWithUnreachableProducerEndsQuietly) {
  auto client = std::make_shared<UcxExchangeClient>(
      "closedConsumer", /*destination=*/0, /*numberOfConsumers=*/1);
  // The source holds the queue until it is destroyed, so the queue outlives
  // the client exactly as long as the source does.
  std::weak_ptr<UcxExchangeQueue> queue = client->queue();
  client->close();
  client->addRemoteTaskId("http://bad_host!:1234/v1/task/x.0.0.0.0/results/0");
  client->noMoreRemoteTasks();

  bool atEnd{false};
  ContinueFuture future = ContinueFuture::makeEmpty();
  EXPECT_EQ(client->next(0, &atEnd, &future), nullptr);
  EXPECT_TRUE(atEnd);
  EXPECT_EQ(client->queue()->sourceState(0), SourceState::kReleased);

  client.reset();
  EXPECT_TRUE(waitUntil(
      [&]() { return queue.expired(); }, std::chrono::milliseconds(1'000)));
}

// A consumer can connect to a producer task that is not yet created on the
// worker. Its server then waits on a Pending entry. When the consumer closes,
// its close message Releases the destination, and the entry keeps it: the
// producer that initializes later finishes without waiting for a reader.
TEST_F(UcxExchangeFocusedTest, closedExchangeDropsPendingEntry) {
  const std::string srcTaskId =
      getUniqueTaskPrefix() + "closedConsumerOfPendingSrc";
  // Initialized only after the consumer is gone.
  auto srcTask = makeProducerTask(srcTaskId, pool_.get());
  SCOPE_EXIT {
    queueManager()->removeTask(srcTaskId);
  };
  auto client = std::make_shared<UcxExchangeClient>(
      "closedConsumer", /*destination=*/0, /*numberOfConsumers=*/1);
  client->addRemoteTaskId(remoteTaskUrl(srcTaskId));
  client->noMoreRemoteTasks();
  // The server asked for data and left a Pending entry.
  ASSERT_TRUE(waitUntil(
      [&]() { return queueManager()->stats(srcTaskId).has_value(); },
      std::chrono::milliseconds(3'000)));

  client->close();
  EXPECT_TRUE(waitUntil(
      [&]() {
        return destinationState(srcTaskId) == DestinationState::kReleased;
      },
      std::chrono::milliseconds(3'000)));

  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  queueManager()->noMoreData(srcTaskId);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *srcTask, TaskState::kFinished, 3'000'000));
}

// A consumer of a Pending entry whose connection fails Releases its
// destination through its server's closed endpoint. The entry keeps the
// Released destination, and the producer that initializes later finishes.
TEST_F(UcxExchangeFocusedTest, endpointFailureKeepsReleasedPendingEntry) {
  const std::string srcTaskId =
      getUniqueTaskPrefix() + "failedConsumerOfPendingSrc";
  // Initialized only after the consumer is gone.
  auto srcTask = makeProducerTask(srcTaskId, pool_.get());
  SCOPE_EXIT {
    queueManager()->removeTask(srcTaskId);
  };
  auto client = std::make_shared<UcxExchangeClient>(
      "failedConsumer", /*destination=*/0, /*numberOfConsumers=*/1);
  client->addRemoteTaskId(remoteTaskUrl(srcTaskId));
  client->noMoreRemoteTasks();
  ASSERT_TRUE(waitUntil(
      [&]() { return queueManager()->stats(srcTaskId).has_value(); },
      std::chrono::milliseconds(3'000)));

  // Fail the connection the way a dead peer does.
  std::shared_ptr<EndpointRef> endpoint;
  ASSERT_TRUE(runOnCommunicator([&endpoint] {
    auto communicator = Communicator::getInstance();
    endpoint = communicator->assocEndpointRef(
        nullptr, HostPort{"127.0.0.1", communicator->getListenerPort()});
    if (endpoint != nullptr) {
      communicator->deferEndpointCleanup(endpoint);
    }
  }));
  ASSERT_NE(endpoint, nullptr);
  EXPECT_TRUE(waitUntil(
      [&]() {
        return destinationState(srcTaskId) == DestinationState::kReleased;
      },
      std::chrono::milliseconds(3'000)));
  client->close();

  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  queueManager()->noMoreData(srcTaskId);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *srcTask, TaskState::kFinished, 3'000'000));
}

// Test that verifies intra-node exchange does not livelock when a producing
// task is removed while the consumer is polling IntraNodeTransferRegistry.
// Before the fix: test times out (livelock). After the fix: test passes.
TEST_F(UcxExchangeFocusedTest, intraNodeTaskRemovalLivelock) {
  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "srcProducerNeverSends";
  const std::string sinkTaskId = taskPrefix + "sinkConsumer";
  const int numPartitions = 1;
  const int partitionId = 0;

  // 1. Create and initialize source task but never enqueue any data.
  //    This simulates a producer that gets cancelled before producing.
  auto srcTask = createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      numPartitions,
      /*numDrivers=*/1);

  // 2. Create sink task with exchange plan node.
  core::PlanNodeId exchangeNodeId;
  auto sinkTask = createExchangeTask(
      sinkTaskId, UcxTestData::kTestRowType, partitionId, exchangeNodeId);
  auto sinkDriver =
      std::make_shared<SinkDriverMock>(sinkTask, /*numDrivers=*/1);
  // The removal before all drivers finished fails the consumer.
  sinkDriver->expectError();

  // Add split pointing to source task. CudfConfig::intraNodeExchange is off by
  // default, so the handshake takes the remote path.
  std::vector<exec::Split> splits;
  splits.emplace_back(remoteSplit(srcTaskId, partitionId));
  sinkDriver->addSplits(splits);

  // 3. Start sink driver on background threads — it will begin polling
  //    IntraNodeTransferRegistry for data that never arrives.
  sinkDriver->run();

  // 4. Wait for the UcxExchangeSource to complete handshake and start polling.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // 5. Premature task cancellation — abort the source task then remove it.
  //    This mirrors the production flow where the task is aborted before
  //    removal. After the fix, the consumer should detect this and stop
  //    polling.
  srcTask->requestAbort();
  queueManager()->removeTask(srcTaskId);

  // 6. Wait for sink to complete with a timeout.
  auto future =
      std::async(std::launch::async, [&]() { sinkDriver->joinThreads(); });
  auto status = future.wait_for(std::chrono::seconds(10));

  // 7. Verify that the sink completed (no livelock).
  if (status != std::future_status::ready) {
    // Abort the sink task to prevent the test from hanging indefinitely.
    sinkTask->requestAbort();
    future.wait();
    FAIL() << "Sink driver did not complete within 10s after removeTask()"
           << " — intra-node livelock: source stuck polling "
           << "IntraNodeTransferRegistry for cancelled task";
  }
  // The source detected the removed task, which ended before its only driver
  // finished, and fails with the producer's error.
  EXPECT_THAT(
      sinkDriver->error(),
      testing::Optional(
          testing::HasSubstr("Producer failed before sending all its data")));
}

// Regression test for broadcast + intra-node SIGSEGV.
// Before the fix in Acceptor.cpp, broadcast tasks using intra-node transfer
// would crash because the intra-node source destructively moves gpu_data from
// a shared packed_columns object, corrupting it for other servers.
// The fix disables intra-node at handshake time for broadcast tasks, falling
// back to UCXX. This test verifies that broadcast with intra-node enabled
// completes without crash and delivers correct data.
TEST_F(UcxExchangeFocusedTest, broadcastIntraNodeFallback) {
  // Enable intra-node exchange so the Acceptor's broadcast guard is exercised.
  auto& config = cudf_velox::CudfConfig::getInstance();
  const bool origIntraNode = config.intraNodeExchange;
  config.intraNodeExchange = true;

  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "broadcastSrc";
  const int numDestinations = 3;
  const int numDrivers = 1;
  const int numChunks = 5;
  const int numRowsPerChunk = 1000;

  // Create source task with broadcast mode.
  auto srcTask = createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kBroadcast,
      numDestinations,
      numDrivers);
  // Finalize destinations for broadcast.
  queueManager()->updateOutputBuffers(srcTaskId, numDestinations, true);

  // Create one sink per destination. Each connects to its own destination
  // index.
  std::vector<std::shared_ptr<SinkDriverMock>> sinkDrivers;
  for (int destId = 0; destId < numDestinations; ++destId) {
    const std::string sinkTaskId =
        taskPrefix + "broadcastSink" + std::to_string(destId);
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(
        sinkTaskId, UcxTestData::kTestRowType, destId, exchangeNodeId);
    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, /*numDrivers=*/1);

    std::vector<exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, destId));
    sinkDriver->addSplits(splits);

    sinkDrivers.push_back(sinkDriver);
  }

  // Producer sends to 1 partition (destination 0); broadcast replicates to all.
  auto sourceMock = std::make_shared<UcxPartitionedOutputMock>(
      srcTaskId, numDrivers, /*numPartitions=*/1, numChunks, numRowsPerChunk);

  // Start source and sinks.
  sourceMock->run();
  for (auto& sink : sinkDrivers) {
    sink->run();
  }

  // Wait for completion.
  sourceMock->joinThreads();
  for (auto& sink : sinkDrivers) {
    sink->joinThreads();
  }

  // Each sink should receive all chunks: 5 * 1000 = 5000 rows.
  const size_t expectedRowsPerSink =
      static_cast<size_t>(numChunks) * numRowsPerChunk;
  for (int i = 0; i < numDestinations; ++i) {
    EXPECT_EQ(sinkDrivers[i]->numRows(), expectedRowsPerSink)
        << "Sink " << i << " row count mismatch";
  }

  // Cleanup.
  queueManager()->removeTask(srcTaskId);
  config.intraNodeExchange = origIntraNode;
}

// Regression test for broadcast + intra-node placeholder race condition.
// When sinks connect BEFORE initializeTask() is called, the Acceptor creates
// a placeholder UcxOutputQueue. If initializeTask() later upgrades that
// placeholder to broadcast mode, the intra-node flag may be incorrectly set
// because the broadcast guard in Acceptor only runs at handshake time — but
// the placeholder was already created with intra-node enabled.
// Without a fix, this causes a SIGSEGV when the intra-node source
// destructively moves gpu_data from the shared packed_columns object.
TEST_F(UcxExchangeFocusedTest, broadcastIntraNodePlaceholderRace) {
  // Enable intra-node exchange so the race condition can manifest.
  auto& config = cudf_velox::CudfConfig::getInstance();
  const bool origIntraNode = config.intraNodeExchange;
  config.intraNodeExchange = true;

  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "broadcastPlaceholderSrc";
  const int numDestinations = 3;
  const int numDrivers = 1;
  const int numChunks = 5;
  const int numRowsPerChunk = 1000;

  // Step 1: Create sink tasks and start them BEFORE initializeTask().
  // This triggers handshakes that create a placeholder queue in
  // UcxOutputQueueManager with intra-node potentially enabled.
  std::vector<std::shared_ptr<SinkDriverMock>> sinkDrivers;
  for (int destId = 0; destId < numDestinations; ++destId) {
    const std::string sinkTaskId =
        taskPrefix + "broadcastPlaceholderSink" + std::to_string(destId);
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(
        sinkTaskId, UcxTestData::kTestRowType, destId, exchangeNodeId);
    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, /*numDrivers=*/1);

    std::vector<exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, destId));
    sinkDriver->addSplits(splits);

    sinkDrivers.push_back(sinkDriver);
  }

  // Start sinks — they will handshake and create placeholder queues.
  for (auto& sink : sinkDrivers) {
    sink->run();
  }

  // Step 2: Wait for handshakes to be processed.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Step 3: NOW initialize the task with broadcast mode.
  // This upgrades the placeholder queue to broadcast.
  auto srcTask = createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kBroadcast,
      numDestinations,
      numDrivers);

  // Step 4: Finalize destinations for broadcast.
  queueManager()->updateOutputBuffers(srcTaskId, numDestinations, true);

  // Step 5: Create and run the producer.
  auto sourceMock = std::make_shared<UcxPartitionedOutputMock>(
      srcTaskId, numDrivers, /*numPartitions=*/1, numChunks, numRowsPerChunk);
  sourceMock->run();

  // Step 6: Wait for completion — without a fix this crashes (SIGSEGV).
  sourceMock->joinThreads();
  for (auto& sink : sinkDrivers) {
    sink->joinThreads();
  }

  // Step 7: Verify all sinks received correct row counts.
  const size_t expectedRowsPerSink =
      static_cast<size_t>(numChunks) * numRowsPerChunk;
  for (int i = 0; i < numDestinations; ++i) {
    EXPECT_EQ(sinkDrivers[i]->numRows(), expectedRowsPerSink)
        << "Sink " << i << " row count mismatch";
  }

  // Cleanup.
  queueManager()->removeTask(srcTaskId);
  config.intraNodeExchange = origIntraNode;
}

// Test that UcxPartitionedOutput's batch accumulation correctly merges many
// small input chunks into fewer, larger output chunks while preserving all rows
// and data integrity.
TEST_F(UcxExchangeFocusedTest, batchAccumulationTest) {
  const int kTargetRows = UcxPartitionedOutput::kDefaultTargetRowsPerChunk;

  // --- Scenario 1: Small chunks that SHOULD be accumulated ---
  // 500 chunks × 100 rows = 50,000 total rows.
  // With kTargetRowsPerChunk = 10,000 and 100 rows/chunk, we need 100 chunks
  // to reach the threshold → expect 5 flushes (500/100 = 5), 0 remainder.
  {
    const int numChunks = 500;
    const int numRowsPerChunk = 100;
    const int numPartitions = 1;
    const int numDrivers = 1;
    const std::string taskPrefix = getUniqueTaskPrefix();
    const std::string srcTaskId = taskPrefix + "sourceTask0";

    auto rowType = UcxTestData::kTestRowType;

    // Create reference data for integrity verification.
    auto dataToSend = std::make_shared<UcxTestData>();
    dataToSend->initialize(numRowsPerChunk);

    auto srcTask =
        createPartitionedOutputTask(srcTaskId, pool_, rowType, numPartitions);
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        numPartitions,
        numDrivers);

    auto sourceDriver = std::make_shared<SourceDriverMock>(
        srcTask, numDrivers, numChunks, numRowsPerChunk, dataToSend);

    const std::string sinkTaskId = taskPrefix + "sinkTask0";
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(sinkTaskId, rowType, 0, exchangeNodeId);
    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, numDrivers, dataToSend);

    std::vector<exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, 0));
    sinkDriver->addSplits(splits);

    sourceDriver->run();
    sinkDriver->run();
    sourceDriver->joinThreads();
    sinkDriver->joinThreads();

    size_t expectedTotalRows = static_cast<size_t>(numChunks) * numRowsPerChunk;

    // Verify all rows arrived.
    EXPECT_EQ(sinkDriver->numRows(), expectedTotalRows)
        << "Accumulation must not lose rows";

    // Verify data integrity.
    EXPECT_TRUE(sinkDriver->dataIsValid())
        << "Accumulated data must match reference";

    // Verify chunk reduction. Compute expected output chunks:
    // chunksPerFlush = ceil(kTargetRows / numRowsPerChunk)
    // outputChunks = ceil(numChunks / chunksPerFlush)
    size_t chunksPerFlush =
        (kTargetRows + numRowsPerChunk - 1) / numRowsPerChunk;
    size_t expectedOutputChunks =
        (numChunks + chunksPerFlush - 1) / chunksPerFlush;

    VLOG(0) << "batchAccumulationTest scenario 1: sent " << numChunks
            << " chunks of " << numRowsPerChunk << " rows, received "
            << sinkDriver->numChunksReceived() << " chunks (expected "
            << expectedOutputChunks << ")";

    EXPECT_EQ(sinkDriver->numChunksReceived(), expectedOutputChunks)
        << "Small chunks should be accumulated into fewer output chunks";

    // Sanity: output chunks must be strictly fewer than input chunks.
    EXPECT_LT(sinkDriver->numChunksReceived(), static_cast<uint64_t>(numChunks))
        << "Accumulation should reduce chunk count";

    queueManager()->removeTask(srcTaskId);
  }

  // --- Scenario 2: Small chunks with a remainder (not evenly divisible) ---
  // 150 chunks × 100 rows = 15,000 total rows.
  // 100 chunks → first flush (10,000 rows), 50 remaining → partial flush.
  // Expected: 2 output chunks.
  {
    const int numChunks = 150;
    const int numRowsPerChunk = 100;
    const int numPartitions = 1;
    const int numDrivers = 1;
    const std::string taskPrefix = getUniqueTaskPrefix();
    const std::string srcTaskId = taskPrefix + "sourceTask0";

    auto rowType = UcxTestData::kTestRowType;

    auto dataToSend = std::make_shared<UcxTestData>();
    dataToSend->initialize(numRowsPerChunk);

    auto srcTask =
        createPartitionedOutputTask(srcTaskId, pool_, rowType, numPartitions);
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        numPartitions,
        numDrivers);

    auto sourceDriver = std::make_shared<SourceDriverMock>(
        srcTask, numDrivers, numChunks, numRowsPerChunk, dataToSend);

    const std::string sinkTaskId = taskPrefix + "sinkTask0";
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(sinkTaskId, rowType, 0, exchangeNodeId);
    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, numDrivers, dataToSend);

    std::vector<exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, 0));
    sinkDriver->addSplits(splits);

    sourceDriver->run();
    sinkDriver->run();
    sourceDriver->joinThreads();
    sinkDriver->joinThreads();

    size_t expectedTotalRows = static_cast<size_t>(numChunks) * numRowsPerChunk;

    EXPECT_EQ(sinkDriver->numRows(), expectedTotalRows)
        << "Remainder scenario must not lose rows";

    EXPECT_TRUE(sinkDriver->dataIsValid())
        << "Remainder scenario data must match reference";

    size_t chunksPerFlush =
        (kTargetRows + numRowsPerChunk - 1) / numRowsPerChunk;
    size_t expectedOutputChunks =
        (numChunks + chunksPerFlush - 1) / chunksPerFlush;

    VLOG(0) << "batchAccumulationTest scenario 2: sent " << numChunks
            << " chunks of " << numRowsPerChunk << " rows, received "
            << sinkDriver->numChunksReceived() << " chunks (expected "
            << expectedOutputChunks << ")";

    EXPECT_EQ(sinkDriver->numChunksReceived(), expectedOutputChunks)
        << "Remainder chunks should be flushed on noMoreInput";

    queueManager()->removeTask(srcTaskId);
  }

  // --- Scenario 3: Large chunks (>= threshold) should NOT be accumulated ---
  // 5 chunks × 20,000 rows = 100,000 total rows.
  // Each chunk exceeds kTargetRowsPerChunk, so each addInput triggers an
  // immediate flush via the single-input fast path. Expected: 5 output chunks.
  {
    const int numChunks = 5;
    const int numRowsPerChunk = 20000;
    const int numPartitions = 1;
    const int numDrivers = 1;
    const std::string taskPrefix = getUniqueTaskPrefix();
    const std::string srcTaskId = taskPrefix + "sourceTask0";

    auto rowType = UcxTestData::kTestRowType;

    auto dataToSend = std::make_shared<UcxTestData>();
    dataToSend->initialize(numRowsPerChunk);

    auto srcTask =
        createPartitionedOutputTask(srcTaskId, pool_, rowType, numPartitions);
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        numPartitions,
        numDrivers);

    auto sourceDriver = std::make_shared<SourceDriverMock>(
        srcTask, numDrivers, numChunks, numRowsPerChunk, dataToSend);

    const std::string sinkTaskId = taskPrefix + "sinkTask0";
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(sinkTaskId, rowType, 0, exchangeNodeId);
    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, numDrivers, dataToSend);

    std::vector<exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, 0));
    sinkDriver->addSplits(splits);

    sourceDriver->run();
    sinkDriver->run();
    sourceDriver->joinThreads();
    sinkDriver->joinThreads();

    size_t expectedTotalRows = static_cast<size_t>(numChunks) * numRowsPerChunk;

    EXPECT_EQ(sinkDriver->numRows(), expectedTotalRows)
        << "Large-chunk scenario must not lose rows";

    EXPECT_TRUE(sinkDriver->dataIsValid())
        << "Large-chunk scenario data must match reference";

    VLOG(0) << "batchAccumulationTest scenario 3: sent " << numChunks
            << " chunks of " << numRowsPerChunk << " rows, received "
            << sinkDriver->numChunksReceived() << " chunks (expected "
            << numChunks << ")";

    // Large chunks should pass through without accumulation — each addInput
    // immediately flushes because pendingRows >= kTargetRowsPerChunk.
    EXPECT_EQ(sinkDriver->numChunksReceived(), static_cast<uint64_t>(numChunks))
        << "Large chunks should not be accumulated";

    queueManager()->removeTask(srcTaskId);
  }

  // --- Scenario 4: Custom threshold via QueryConfig ---
  // 50 chunks × 100 rows = 5,000 total rows with a custom threshold of 500.
  // chunksPerFlush = ceil(500/100) = 5
  // outputChunks = ceil(50/5) = 10
  {
    const int numChunks = 50;
    const int numRowsPerChunk = 100;
    const int64_t customThreshold = 500;
    const int numPartitions = 1;
    const int numDrivers = 1;
    const std::string taskPrefix = getUniqueTaskPrefix();
    const std::string srcTaskId = taskPrefix + "sourceTask0";

    auto rowType = UcxTestData::kTestRowType;

    auto dataToSend = std::make_shared<UcxTestData>();
    dataToSend->initialize(numRowsPerChunk);

    // Pass custom threshold via QueryConfig.
    std::unordered_map<std::string, std::string> extraConfig{
        {cudf_velox::CudfConfig::kUcxPartitionedOutputBatchRows,
         std::to_string(customThreshold)}};

    auto srcTask = createPartitionedOutputTask(
        srcTaskId, pool_, rowType, numPartitions, {}, FOUR_GBYTES, extraConfig);
    queueManager()->initializeTask(
        srcTask,
        core::PartitionedOutputNode::Kind::kPartitioned,
        numPartitions,
        numDrivers);

    auto sourceDriver = std::make_shared<SourceDriverMock>(
        srcTask, numDrivers, numChunks, numRowsPerChunk, dataToSend);

    const std::string sinkTaskId = taskPrefix + "sinkTask0";
    core::PlanNodeId exchangeNodeId;
    auto sinkTask = createExchangeTask(sinkTaskId, rowType, 0, exchangeNodeId);
    auto sinkDriver =
        std::make_shared<SinkDriverMock>(sinkTask, numDrivers, dataToSend);

    std::vector<exec::Split> splits;
    splits.emplace_back(remoteSplit(srcTaskId, 0));
    sinkDriver->addSplits(splits);

    sourceDriver->run();
    sinkDriver->run();
    sourceDriver->joinThreads();
    sinkDriver->joinThreads();

    size_t expectedTotalRows = static_cast<size_t>(numChunks) * numRowsPerChunk;

    EXPECT_EQ(sinkDriver->numRows(), expectedTotalRows)
        << "Custom threshold scenario must not lose rows";

    EXPECT_TRUE(sinkDriver->dataIsValid())
        << "Custom threshold scenario data must match reference";

    size_t chunksPerFlush =
        (customThreshold + numRowsPerChunk - 1) / numRowsPerChunk;
    size_t expectedOutputChunks =
        (numChunks + chunksPerFlush - 1) / chunksPerFlush;

    VLOG(0) << "batchAccumulationTest scenario 4: sent " << numChunks
            << " chunks of " << numRowsPerChunk
            << " rows with custom threshold=" << customThreshold
            << ", received " << sinkDriver->numChunksReceived()
            << " chunks (expected " << expectedOutputChunks << ")";

    EXPECT_EQ(sinkDriver->numChunksReceived(), expectedOutputChunks)
        << "Custom threshold should control accumulation granularity";

    queueManager()->removeTask(srcTaskId);
  }
}

// Smoke test: aborting a source task after starting a UCX exchange must not
// crash or leave the sink driver hanging.
TEST_F(UcxExchangeFocusedTest, sourceTaskAbortDoesNotCrashOrHang) {
  // Ensure intra-node is disabled so we exercise the UCXX path (tagRecv).
  auto& config = cudf_velox::CudfConfig::getInstance();
  const bool origIntraNode = config.intraNodeExchange;
  SCOPE_EXIT {
    config.intraNodeExchange = origIntraNode;
  };
  config.intraNodeExchange = false;

  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "srcActiveTransfer";
  const std::string sinkTaskId = taskPrefix + "sinkAborted";
  const int numPartitions = 1;
  const int partitionId = 0;
  const int numDrivers = 1;
  // Generate data for the exchange before aborting the source.
  const int numChunks = 50;
  const int numRowsPerChunk = 100000;

  auto rowType = UcxTestData::kTestRowType;

  // 1. Create and initialize source task with data to send.
  auto srcTask = createSourceTask(srcTaskId, pool_, rowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      numPartitions,
      numDrivers);

  auto sourceMock = std::make_shared<UcxPartitionedOutputMock>(
      srcTaskId, numDrivers, numPartitions, numChunks, numRowsPerChunk);

  // 2. Create sink task with exchange plan node.
  core::PlanNodeId exchangeNodeId;
  auto sinkTask =
      createExchangeTask(sinkTaskId, rowType, partitionId, exchangeNodeId);
  auto sinkDriver =
      std::make_shared<SinkDriverMock>(sinkTask, /*numDrivers=*/1);
  sinkDriver->expectError();

  std::vector<exec::Split> splits;
  splits.emplace_back(remoteSplit(srcTaskId, partitionId));
  sinkDriver->addSplits(splits);

  // 3. Start source (enqueues data) and sink (begins receiving via UCXX).
  sourceMock->run();
  sinkDriver->run();

  // 4. Allow the exchange to start before aborting.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // 5. Stop the mock publisher, which is not owned by the task, so it cannot
  //    enqueue into a terminated output queue after abort.
  sourceMock->requestStop();
  sourceMock->joinThreads();

  srcTask->requestAbort();
  queueManager()->removeTask(srcTaskId);

  // 6. Wait for the sink driver to complete (it should detect the abort
  //    and finish without crashing).
  auto future =
      std::async(std::launch::async, [&]() { sinkDriver->joinThreads(); });
  auto status = future.wait_for(std::chrono::seconds(15));

  if (status != std::future_status::ready) {
    sinkTask->requestAbort();
    future.wait();
    FAIL() << "Sink driver did not complete within 15s after source abort"
           << " — possible exchange abort hang";
  }

  // The consumer either received every row before the abort or stops with the
  // producer's error. It never ends cleanly with rows missing.
  const uint64_t expectedRows =
      static_cast<uint64_t>(numChunks) * numRowsPerChunk;
  if (sinkDriver->numRows() != expectedRows) {
    EXPECT_THAT(
        sinkDriver->error(),
        testing::Optional(
            testing::HasSubstr("Producer failed before sending all its data")))
        << "Rows received: " << sinkDriver->numRows() << " of " << expectedRows;
  }

  // 7. Allow the Communicator's event loop to run before teardown.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  VLOG(0) << "sourceTaskAbortDoesNotCrashOrHang: completed without crash";
}

// The end message of a channel is the last send. The producer's output stays
// alive until that send completes, that is, until the consumer receives it.
TEST_F(UcxExchangeFocusedTest, terminalMetadataWaitsForRemoteReceive) {
  if (std::getenv("UCX_RNDV_THRESH") == nullptr ||
      std::string_view(std::getenv("UCX_RNDV_THRESH")) != "0") {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "terminalMetadataSuccess", /*numChunkRows=*/0);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  EXPECT_FALSE(queueManager()->isFinished(session->taskId));
  postTerminalMetadataReceive(session);
  EXPECT_TRUE(waitForTerminalReceive(session));
  EXPECT_TRUE(waitForOutputFinished(session->taskId));
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kFinished, 3'000'000));
}

// An endpoint closed locally, with no close callback to tell its servers,
// cancels the pending end send. The endpoint is no longer alive, so the server
// reports its consumer gone: the destination is Released, and the producer
// finishes rather than fails.
TEST_F(
    UcxExchangeFocusedTest,
    terminalMetadataLocalEndpointCloseReleasesDestination) {
  if (std::getenv("UCX_RNDV_THRESH") == nullptr ||
      std::string_view(std::getenv("UCX_RNDV_THRESH")) != "0") {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "terminalMetadataLocalClose",
      /*numChunkRows=*/0);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  EXPECT_TRUE(runOnCommunicator([session] {
    auto endpoint = session->endpoint->endpoint_;
    endpoint->setCloseCallback(nullptr, nullptr);
    endpoint->closeBlocking();
    Communicator::getInstance()->removeEndpointRef(session->endpoint);
  }));
  EXPECT_TRUE(waitForServerDestroyed(session->server));
  EXPECT_TRUE(waitForOutputFinished(session->taskId));
  EXPECT_EQ(destinationState(session->taskId), DestinationState::kReleased);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kFinished, 3'000'000));
}

// A peer that fails closes the endpoint, which tells its servers. The server's
// destination is Released, so the producer finishes rather than fails.
TEST_F(UcxExchangeFocusedTest, terminalMetadataPeerFailureReleasesDestination) {
  if (std::getenv("UCX_RNDV_THRESH") == nullptr ||
      std::string_view(std::getenv("UCX_RNDV_THRESH")) != "0") {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "terminalMetadataPeerFailure",
      /*numChunkRows=*/0);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  Communicator::getInstance()->deferEndpointCleanup(session->endpoint);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kFinished, 3'000'000));
  EXPECT_TRUE(waitForServerDestroyed(session->server));
  EXPECT_TRUE(queueManager()->isFinished(session->taskId));
  EXPECT_EQ(destinationState(session->taskId), DestinationState::kReleased);
}

// A server that closes without a decision from its entry, as at shutdown,
// leaves the output for result deletion or task cleanup.
TEST_F(UcxExchangeFocusedTest, terminalMetadataCloseLeavesOutputPending) {
  if (std::getenv("UCX_RNDV_THRESH") == nullptr ||
      std::string_view(std::getenv("UCX_RNDV_THRESH")) != "0") {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "terminalMetadataClose", /*numChunkRows=*/0);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  ASSERT_FALSE(session->server.expired());
  ASSERT_TRUE(runOnCommunicator([session] {
    if (auto server = session->server.lock()) {
      server->close();
    }
  }));
  EXPECT_EQ(session->task->state(), TaskState::kRunning);
  EXPECT_FALSE(queueManager()->isFinished(session->taskId));

  queueManager()->deleteResults(session->taskId, 0);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kFinished, 3'000'000));
}

// Aborting a task whose UCX output has not been fetched removes that output
// through the manager the Task looked up in the query's transport registry.
TEST_F(UcxExchangeFocusedTest, terminalMetadataTaskAbortRemovesUcxOutput) {
  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(1);
  auto queryCtx = core::QueryCtx::create(executor.get());
  auto registry = exec::OutputTransportRegistry::create();
  const std::string transport{core::TransportKind::kUcx};
  registry->insert(
      transport,
      exec::OutputTransportEntry::make<UcxOutputQueueManager>(
          queueManager(),
          [](int32_t operatorId,
             exec::DriverCtx* ctx,
             const std::shared_ptr<const core::PartitionedOutputNode>& node,
             bool eagerFlush,
             const std::shared_ptr<UcxOutputQueueManager>& manager)
              -> std::unique_ptr<exec::Operator> {
            return std::make_unique<UcxPartitionedOutput>(
                operatorId, ctx, node, eagerFlush, manager);
          }));
  queryCtx->setRegistry(exec::OutputTransportRegistry::kRegistryKey, registry);
  const auto& rowType = UcxTestData::kTestRowType;
  auto input = std::static_pointer_cast<RowVector>(
      BaseVector::create(rowType, 0, pool_.get()));
  auto output = core::PartitionedOutputNode::single(
      "output",
      rowType,
      VectorSerde::kindName(VectorSerde::Kind::kPresto),
      transport,
      exec::test::PlanBuilder().values({input}).planNode());
  const auto taskId = getUniqueTaskPrefix() + "taskAbortRemovesUcxOutput";
  auto task = exec::Task::create(
      taskId,
      core::PlanFragment{output},
      0,
      queryCtx,
      exec::Task::ExecutionMode::kParallel);
  SCOPE_EXIT {
    task->requestAbort().wait();
  };
  task->start(1);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (task->numFinishedDrivers() != 1 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_EQ(task->numFinishedDrivers(), 1);
  ASSERT_EQ(task->outputBufferManager().lock(), queueManager());
  ASSERT_EQ(task->state(), TaskState::kRunning);
  ASSERT_TRUE(queueManager()->stats(taskId).has_value());
  ASSERT_FALSE(queueManager()->isFinished(taskId));

  // The coordinator's DELETE request reaches the same Task cleanup.
  task->requestAbort().wait();
  EXPECT_EQ(task->state(), TaskState::kAborted);
  EXPECT_EQ(queueManager()->stats(taskId), std::nullopt);
}

// A server is bound to the entry of the task it was created for. Completing
// its end send after that task was removed, and a new task reuses the ID, does
// not finish the new task's output.
TEST_F(UcxExchangeFocusedTest, terminalMetadataCallbackStaysWithOriginalTask) {
  if (std::getenv("UCX_RNDV_THRESH") == nullptr ||
      std::string_view(std::getenv("UCX_RNDV_THRESH")) != "0") {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "terminalMetadataTaskReuse", /*numChunkRows=*/0);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  session->task->requestAbort().wait();
  queueManager()->removeTask(session->taskId);
  auto newTask = makeProducerTask(session->taskId, pool_.get());
  queueManager()->initializeTask(
      newTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  queueManager()->noMoreData(session->taskId);

  postTerminalMetadataReceive(session);
  EXPECT_TRUE(waitForTerminalReceive(session));
  // The server acts on the completed send before it stops.
  EXPECT_TRUE(waitForServerDestroyed(session->server));
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kAborted, 3'000'000));
  EXPECT_EQ(newTask->state(), TaskState::kRunning);
  EXPECT_FALSE(queueManager()->isFinished(session->taskId));
  newTask->requestAbort();
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *newTask, TaskState::kAborted, 3'000'000));
}

// A consumer that goes away while the end send is unmatched Releases the
// destination. The entry tells the server, which cancels the send and stops.
TEST_F(
    UcxExchangeFocusedTest,
    terminalMetadataReleasedDestinationCancelsPendingSend) {
  if (std::getenv("UCX_RNDV_THRESH") == nullptr ||
      std::string_view(std::getenv("UCX_RNDV_THRESH")) != "0") {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "terminalMetadataReleased", /*numChunkRows=*/0);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  ASSERT_FALSE(session->server.expired());
  queueManager()->consumerGone(session->taskId, 0);
  EXPECT_TRUE(waitForServerDestroyed(session->server));
  EXPECT_TRUE(queueManager()->isFinished(session->taskId));
  EXPECT_EQ(destinationState(session->taskId), DestinationState::kReleased);
}

// An endpoint that closes before a source's stream ended fails the consumer
// with the endpoint's close reason, unless the consumer closed first.
TEST_F(
    UcxExchangeFocusedTest,
    terminalMetadataEndpointCloseReportsActiveSourceError) {
  for (bool consumerClosed : {false, true}) {
    SCOPED_TRACE(consumerClosed);
    auto queue = std::make_shared<UcxExchangeQueue>(1);
    UcxExchangeQueue::SourceId sourceId;
    {
      std::lock_guard<std::mutex> lock(queue->mutex());
      sourceId = queue->addSourceLocked();
    }
    queue->noMoreSources();
    ASSERT_TRUE(runOnCommunicator([queue, sourceId, consumerClosed] {
      auto communicator = Communicator::getInstance();
      auto source = UcxExchangeSource::create(
          "consumer",
          "http://127.0.0.1:12345/v1/task/producer/results/0",
          queue,
          sourceId);
      communicator->unregister(source);
      // Exercise endpoint cleanup before a receive is posted. There is no
      // receive callback that could otherwise report the lost connection.
      EndpointRef endpoint{nullptr};
      endpoint.addCommElem(source);
      if (consumerClosed) {
        // As UcxExchangeClient::close() does.
        queue->close();
        source->close();
        communicator->unregister(source);
      }
      endpoint.closeAndDrainCommunicators();
      communicator->unregister(source);
    }));
    std::lock_guard<std::mutex> lock(queue->mutex());
    bool atEnd = false;
    ContinueFuture future = ContinueFuture::makeEmpty();
    ContinuePromise stalePromise = ContinuePromise::makeEmpty();
    if (consumerClosed) {
      EXPECT_EQ(
          queue->dequeueLocked(0, &atEnd, &future, &stalePromise), nullptr);
      EXPECT_TRUE(atEnd);
    } else {
      VELOX_ASSERT_THROW(
          queue->dequeueLocked(0, &atEnd, &future, &stalePromise),
          "Connection to producer closed before it sent all its data, host 127.0.0.1:12348, task producer/0: UCX endpoint closed");
    }
  }
}

// A closed endpoint Releases the destination of every server on it. A chunk
// large enough to go by rendezvous keeps the server's send pending, whatever
// UCX_RNDV_THRESH says.
TEST_F(UcxExchangeFocusedTest, serverEndpointClosedReleasesDestination) {
  auto session = startTerminalMetadataSend(
      getUniqueTaskPrefix() + "serverEndpointClosed",
      /*numChunkRows=*/1'000'000);
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  Communicator::getInstance()->deferEndpointCleanup(session->endpoint);
  EXPECT_TRUE(waitForOutputFinished(session->taskId));
  EXPECT_TRUE(waitForServerDestroyed(session->server));
  EXPECT_EQ(destinationState(session->taskId), DestinationState::kReleased);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kFinished, 3'000'000));
}

// A consumer that closes after reading part of its stream sends the close
// message. The producer's worker receives it and Releases the destination, so
// the producer finishes once its drivers do, rather than wait for a reader that
// is gone.
TEST_F(UcxExchangeFocusedTest, consumerCloseReleasesProducer) {
  const auto taskId = getUniqueTaskPrefix() + "closedConsumerProducer";
  auto task = makeProducerTask(taskId, pool_.get());
  queueManager()->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  SCOPE_EXIT {
    queueManager()->removeTask(taskId);
  };
  // One chunk, and no noMoreData(): the producer is still running.
  queueManager()->enqueue(taskId, 0, makePackedColumns(100), /*numRows=*/100);

  auto client = std::make_shared<UcxExchangeClient>(
      "closedConsumer", /*destination=*/0, /*numberOfConsumers=*/1);
  client->addRemoteTaskId(remoteTaskUrl(taskId));
  client->noMoreRemoteTasks();
  auto table = nextTable(*client);
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->numRows, 100);
  table.reset();
  client->close();

  EXPECT_TRUE(waitUntil(
      [&]() { return destinationState(taskId) == DestinationState::kReleased; },
      std::chrono::milliseconds(3'000)));
  queueManager()->noMoreData(taskId);
  EXPECT_TRUE(waitForOutputFinished(taskId));
  EXPECT_NE(task->state(), TaskState::kFailed);
}

// A source that closes before the handshake response arrives does not know
// whether its server exists. It waits for the response, then sends the close
// message. If no response arrives within kCloseTimeout, it ends and sends
// nothing.
TEST_F(UcxExchangeFocusedTest, closeDuringHandshakeWaitsForResponse) {
  for (const bool reply : {true, false}) {
    SCOPED_TRACE(reply ? "late response" : "no response");
    const auto taskId = getUniqueTaskPrefix() + "closeDuringHandshake";
    const PartitionKey key{taskId, 0};
    const auto port = exec::test::Ports::findFreePort();
    auto peer = FakeUcxPeer::listen(port);
    auto queue = std::make_shared<UcxExchangeQueue>(1);
    const auto sourceId = addSource(*queue);
    queue->noMoreSources();
    std::weak_ptr<UcxExchangeSource> source;
    {
      auto created = UcxExchangeSource::create(
          "closedConsumer",
          fmt::format(
              "http://127.0.0.1:{}/v1/task/{}/results/0",
              port - kSplitUrlPortOffset,
              taskId),
          queue,
          sourceId);
      source = created;
      ASSERT_EQ(
          peer->waitForHandshake(std::chrono::seconds(3)).size(),
          sizeof(HandshakeMsg));
      // As UcxExchangeClient::close() does.
      ASSERT_THAT(queue->close(), testing::ElementsAre(sourceId));
      created->close();
    }

    const auto closeTag = getCloseTag(fnv1a_32(key.toString()));
    if (reply) {
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      const HandshakeResponse response{};
      ASSERT_EQ(
          peer->sendTag(
              getHandshakeResponseTag(fnv1a_32(key.toString())),
              &response,
              sizeof(response),
              std::chrono::seconds(3)),
          UCS_OK);
      const auto [status, bytes] =
          peer->receiveTag(closeTag, sizeof(CloseMsg), std::chrono::seconds(3));
      ASSERT_EQ(status, UCS_OK);
      ASSERT_EQ(bytes.size(), sizeof(CloseMsg));
      CloseMsg message;
      std::memcpy(&message, bytes.data(), sizeof(message));
      EXPECT_EQ(decodeCloseMsg(message), key);
      EXPECT_TRUE(waitUntil(
          [&]() { return source.expired(); },
          std::chrono::milliseconds(3'000)));
    } else {
      EXPECT_TRUE(waitUntil(
          [&]() { return source.expired(); },
          UcxExchangeSource::kCloseTimeout + std::chrono::seconds(1)));
      EXPECT_EQ(
          peer->receiveTag(
                  closeTag, sizeof(CloseMsg), std::chrono::milliseconds(1'500))
              .first,
          UCS_ERR_TIMED_OUT);
    }
    EXPECT_EQ(queue->sourceState(sourceId), SourceState::kReleased);
  }
}

// A source closed while its handshake send is stalled, because the producer's
// worker has not accepted the connection yet, still sends the close message
// once the producer accepts and responds within kCloseTimeout.
TEST_F(
    UcxExchangeFocusedTest,
    terminalMetadataCloseDuringStalledHandshakeSend) {
  for (const bool errorHandling : {false, true}) {
    SCOPED_TRACE(errorHandling);
    auto context =
        ucxx::contextBuilder(UCP_FEATURE_TAG | UCP_FEATURE_AM).build();
    auto worker = context->workerBuilder().build();
    ucp_conn_request_h connectionRequest = nullptr;
    const auto port = exec::test::Ports::findFreePort();
    auto listener = worker
                        ->listenerBuilder(
                            port,
                            [](ucp_conn_request_h request, void* argument) {
                              *static_cast<ucp_conn_request_h*>(argument) =
                                  request;
                            },
                            &connectionRequest)
                        .build();
    const auto taskId = getUniqueTaskPrefix() + "stalledHandshake";
    const PartitionKey key{taskId, 0};
    auto queue = std::make_shared<UcxExchangeQueue>(1);
    const auto sourceId = addSource(*queue);
    queue->noMoreSources();
    const auto url = fmt::format(
        "http://127.0.0.1:{}/v1/task/{}/results/0",
        port - kSplitUrlPortOffset,
        taskId);
    std::weak_ptr<UcxExchangeSource> source;
    // Connects on the Communicator thread, so that the endpoint takes this
    // case's error handling.
    ASSERT_TRUE(
        runOnCommunicator([&source, queue, sourceId, url, errorHandling] {
          auto& config = cudf_velox::CudfConfig::getInstance();
          const bool previous = config.ucxxErrorHandling;
          SCOPE_EXIT {
            config.ucxxErrorHandling = previous;
          };
          config.ucxxErrorHandling = errorHandling;
          auto created =
              UcxExchangeSource::create("closedConsumer", url, queue, sourceId);
          created->process();
          source = created;
        }));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (connectionRequest == nullptr &&
           std::chrono::steady_clock::now() < deadline) {
      worker->progress();
    }
    ASSERT_NE(connectionRequest, nullptr);
    EXPECT_EQ(queue->sourceState(sourceId), SourceState::kRegistered);
    // As UcxExchangeClient::close() does.
    ASSERT_THAT(queue->close(), testing::ElementsAre(sourceId));
    if (auto active = source.lock()) {
      active->close();
    }

    // The handshake stays stalled for a while, but less than kCloseTimeout.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    bool handshakeReceived{false};
    worker->registerAmReceiverCallback(
        ucxx::AmReceiverCallbackInfo{
            Communicator::getInstance()->kAmCallbackOwner,
            Communicator::getInstance()->kAmCallbackId},
        [&](std::shared_ptr<ucxx::Request> /*request*/, ucp_ep_h /*endpoint*/) {
          handshakeReceived = true;
        });
    auto peer = listener->endpointBuilder(connectionRequest)
                    .endpointErrorHandling(errorHandling)
                    .build();
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!handshakeReceived && std::chrono::steady_clock::now() < deadline) {
      worker->progress();
    }
    ASSERT_TRUE(handshakeReceived);

    auto response = std::make_shared<HandshakeResponse>();
    auto responseRequest =
        peer->tagSendBuilder(
                response.get(),
                sizeof(*response),
                ucxx::Tag{getHandshakeResponseTag(fnv1a_32(key.toString()))})
            .callbackData(response)
            .build();
    auto message = std::make_shared<CloseMsg>();
    auto closeReceive =
        peer->tagRecvBuilder(
                message.get(),
                sizeof(*message),
                ucxx::Tag{getCloseTag(fnv1a_32(key.toString()))},
                ucxx::TagMaskFull)
            .callbackData(message)
            .build();
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!closeReceive->isCompleted() &&
           std::chrono::steady_clock::now() < deadline) {
      worker->progress();
    }
    ASSERT_TRUE(closeReceive->isCompleted());
    EXPECT_EQ(closeReceive->getStatus(), UCS_OK);
    EXPECT_EQ(decodeCloseMsg(*message), key);
    EXPECT_TRUE(responseRequest->isCompleted());
    EXPECT_TRUE(waitUntil(
        [&]() { return source.expired(); }, std::chrono::milliseconds(3'000)));
    peer->closeBlocking();
  }
}

// A client that closed before it was given its producer still connects to the
// producer to send the close message, so that the producer finishes.
TEST_F(UcxExchangeFocusedTest, terminalMetadataCloseBeforeRemoteTaskAdded) {
  auto& config = cudf_velox::CudfConfig::getInstance();
  const bool previousIntraNode = config.intraNodeExchange;
  SCOPE_EXIT {
    config.intraNodeExchange = previousIntraNode;
  };
  config.intraNodeExchange = false;

  const auto taskId = getUniqueTaskPrefix() + "closeBeforeRemoteTaskAdded";
  auto task = makeProducerTask(taskId, pool_.get());
  queueManager()->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  SCOPE_EXIT {
    queueManager()->removeTask(taskId);
  };
  queueManager()->noMoreData(taskId);

  auto client = std::make_shared<UcxExchangeClient>(
      "closedConsumer", /*destination=*/0, /*numberOfConsumers=*/1);
  client->close();
  client->addRemoteTaskId(remoteTaskUrl(taskId));
  client->noMoreRemoteTasks();

  EXPECT_TRUE(waitForOutputFinished(taskId));
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *task, TaskState::kFinished, 3'000'000));
}

// A close message that arrives before the producer initializes its output
// Releases the destination of the Pending entry. The server stops, and the
// producer finishes once it initializes and its drivers finish.
TEST_F(UcxExchangeFocusedTest, terminalMetadataCancellationBeforeTaskInit) {
  if (!isRendezvousOnly()) {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  auto session =
      startPendingSession(getUniqueTaskPrefix() + "terminalMetadataEarlyClose");
  SCOPE_EXIT {
    cleanUpTerminalMetadataSession(session);
  };

  ASSERT_FALSE(session->server.expired());
  sendCloseMessage(session);
  EXPECT_TRUE(waitForServerDestroyed(session->server));
  EXPECT_EQ(destinationState(session->taskId), DestinationState::kReleased);

  queueManager()->initializeTask(
      session->task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  queueManager()->noMoreData(session->taskId);
  EXPECT_TRUE(
      exec::test::TaskMonitor::waitForStateChange(
          *session->task, TaskState::kFinished, 3'000'000));
}

// The Communicator receives every close message on its worker, also one for a
// task it has no entry for, as after the task's server ended and its entry was
// erased. It drops the message and creates no entry. With rendezvous, the
// send completes only if the message was received.
TEST_F(UcxExchangeFocusedTest, terminalMetadataCloseAfterServerEndedIsDropped) {
  if (!isRendezvousOnly()) {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  const PartitionKey key{"no.such.task.0.0", 0};
  auto peer = FakeUcxPeer::connect(
      "127.0.0.1", Communicator::getInstance()->getListenerPort());
  const auto message = makeCloseMsg(key);
  EXPECT_EQ(
      peer->sendTag(
          getCloseTag(fnv1a_32(key.toString())),
          &message,
          sizeof(message),
          std::chrono::seconds(3)),
      UCS_OK);
  EXPECT_EQ(queueManager()->stats(key.taskId), std::nullopt);
}

// A close message for a removed task is received, and leaves the task
// invisible to Velox.
TEST_F(
    UcxExchangeFocusedTest,
    terminalMetadataRemovedQueueStillReceivesCancellation) {
  if (!isRendezvousOnly()) {
    GTEST_SKIP() << "Requires UCX_RNDV_THRESH=0";
  }

  const PartitionKey key{getUniqueTaskPrefix() + "removedQueueClose", 0};
  auto task = makeProducerTask(key.taskId, pool_.get());
  queueManager()->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  queueManager()->removeTask(key.taskId);
  ASSERT_EQ(queueManager()->stats(key.taskId), std::nullopt);

  auto peer = FakeUcxPeer::connect(
      "127.0.0.1", Communicator::getInstance()->getListenerPort());
  const auto message = makeCloseMsg(key);
  EXPECT_EQ(
      peer->sendTag(
          getCloseTag(fnv1a_32(key.toString())),
          &message,
          sizeof(message),
          std::chrono::seconds(3)),
      UCS_OK);
  EXPECT_EQ(queueManager()->stats(key.taskId), std::nullopt);
}

// The producer's end and the consumer's close race. The source sends the close
// message only if it has not seen the end; a message that arrives after the
// server ended is dropped. Either way the producer finishes, and the consumer
// ends or is closed, never failed.
TEST_F(UcxExchangeFocusedTest, closeMessageRacesEnd) {
  const auto taskPrefix = getUniqueTaskPrefix();
  for (int32_t i = 0; i < 50; ++i) {
    SCOPED_TRACE(i);
    const auto taskId = taskPrefix + "closeRacesEnd" + std::to_string(i);
    auto task = makeProducerTask(taskId, pool_.get());
    queueManager()->initializeTask(
        task,
        core::PartitionedOutputNode::Kind::kPartitioned,
        /*numDestinations=*/1,
        /*numDrivers=*/1);
    auto removeProducer =
        folly::makeGuard([&]() { queueManager()->removeTask(taskId); });

    auto client = std::make_shared<UcxExchangeClient>(
        "racingConsumer", /*destination=*/0, /*numberOfConsumers=*/1);
    client->addRemoteTaskId(remoteTaskUrl(taskId));
    client->noMoreRemoteTasks();
    ASSERT_TRUE(waitUntil(
        [&]() { return client->queue()->sourceState(0) == SourceState::kOpen; },
        std::chrono::milliseconds(3'000)));

    runConcurrently(
        [&]() { queueManager()->noMoreData(taskId); },
        [&]() { client->close(); });

    EXPECT_TRUE(waitForOutputFinished(taskId));
    EXPECT_TRUE(
        exec::test::TaskMonitor::waitForStateChange(
            *task, TaskState::kFinished, 3'000'000));
    bool atEnd{false};
    ContinueFuture future = ContinueFuture::makeEmpty();
    EXPECT_EQ(client->next(0, &atEnd, &future), nullptr);
    EXPECT_TRUE(atEnd);
  }
}

} // namespace facebook::velox::ucx_exchange
