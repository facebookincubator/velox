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
#include <fmt/format.h>
#include <folly/Executor.h>
#include <folly/ScopeGuard.h>
#include <folly/synchronization/EventCount.h>
#include <gmock/gmock.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryConfig.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
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

// Fixture for focused behavioral tests that should run once per process rather
// than once for every ExchangeTestParams value.
class UcxExchangeFocusedTest : public UcxExchangeTestBase {};

INSTANTIATE_TEST_SUITE_P(
    UcxExchangeTest,
    UcxExchangeTest,
    ::testing::ValuesIn(generateTestParams()),
    ExchangeTestParamsPrinter());

// size() and empty() read a counter maintained under the queue's mutex rather
// than the deque, so that the Communicator progress thread can poll the depth
// lock-free. Covers that the counter stays in step with every mutation.
TEST(UcxExchangeQueueTest, sizeTracksQueuedTables) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(queue.mutex());
    queue.addSourceLocked();
    EXPECT_EQ(queue.size(), 0);
    EXPECT_TRUE(queue.empty());

    for (int32_t expected = 1; expected <= 3; ++expected) {
      queue.enqueueLocked(std::make_unique<PackedTableWithStream>(), promises);
      EXPECT_EQ(queue.size(), expected);
    }
    EXPECT_FALSE(queue.empty());

    bool atEnd{false};
    ContinueFuture future = ContinueFuture::makeEmpty();
    ContinuePromise stalePromise = ContinuePromise::makeEmpty();
    EXPECT_NE(queue.dequeueLocked(0, &atEnd, &future, &stalePromise), nullptr);
    EXPECT_FALSE(atEnd);
    EXPECT_EQ(queue.size(), 2);

    // The end-of-stream marker counts a source as completed instead of being
    // queued, so the depth must not move.
    queue.enqueueLocked(nullptr, promises);
    EXPECT_EQ(queue.size(), 2);
  }

  // setError() discards the queued tables, because an errored queue is never
  // consumed from.
  queue.setError("Producer failed");
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
}

TEST(UcxExchangeQueueTest, closeResetsSize) {
  UcxExchangeQueue queue{/*numberOfConsumers=*/1};
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(queue.mutex());
    queue.addSourceLocked();
    queue.enqueueLocked(std::make_unique<PackedTableWithStream>(), promises);
    queue.enqueueLocked(std::make_unique<PackedTableWithStream>(), promises);
    EXPECT_EQ(queue.size(), 2);
  }

  // close() discards whatever is still queued.
  queue.close();
  EXPECT_EQ(queue.size(), 0);
  EXPECT_TRUE(queue.empty());
}

// An end-of-stream message from a producer that ended early carries the reason,
// so the consumer can fail instead of booking the stream as complete.
TEST(UcxExchangeProtocolTest, metadataRoundTripsProducerError) {
  MetadataMsg message;
  message.dataSizeBytes = 0;
  message.numRows = 0;
  message.atEnd = true;
  message.producerError = "Injected producer failure";

  auto [buffer, size] = message.serialize();
  EXPECT_EQ(size, message.getSerializedSize());

  const auto decoded = MetadataMsg::deserializeMetadataMsg(buffer.get());
  EXPECT_TRUE(decoded.atEnd);
  EXPECT_EQ(
      decoded.producerError,
      std::optional<std::string>("Injected producer failure"));
}

TEST(UcxExchangeProtocolTest, metadataWithoutProducerErrorDecodesNone) {
  MetadataMsg message;
  message.dataSizeBytes = 0;
  message.numRows = 0;
  message.atEnd = true;

  auto [buffer, size] = message.serialize();
  EXPECT_EQ(size, message.getSerializedSize());

  const auto decoded = MetadataMsg::deserializeMetadataMsg(buffer.get());
  EXPECT_TRUE(decoded.atEnd);
  EXPECT_EQ(decoded.producerError, std::nullopt);
}

// A task can fail with an empty message. The consumer must still fail, so an
// empty reason is an error and not the absence of one.
TEST(UcxExchangeProtocolTest, metadataRoundTripsEmptyProducerError) {
  MetadataMsg message;
  message.dataSizeBytes = 0;
  message.numRows = 0;
  message.atEnd = true;
  message.producerError = "";

  auto [buffer, size] = message.serialize();
  EXPECT_EQ(size, message.getSerializedSize());

  const auto decoded = MetadataMsg::deserializeMetadataMsg(buffer.get());
  EXPECT_EQ(decoded.producerError, std::optional<std::string>(""));
}

// The producer error is the last field: a signed length, -1 when there is no
// error, followed by that many bytes of text.
TEST(UcxExchangeProtocolTest, producerErrorIsLengthPrefixedLastField) {
  MetadataMsg message;
  message.dataSizeBytes = 0;
  message.numRows = 0;
  message.atEnd = true;

  auto [cleanBuffer, cleanSize] = message.serialize();
  WireErrorLengthType cleanLength = 0;
  std::memcpy(
      &cleanLength,
      cleanBuffer.get() + cleanSize - sizeof(cleanLength),
      sizeof(cleanLength));
  EXPECT_EQ(cleanLength, -1);

  const std::string reason{"Injected producer failure"};
  message.producerError = reason;
  auto [errorBuffer, errorSize] = message.serialize();
  EXPECT_EQ(errorSize, cleanSize + reason.size());
  EXPECT_EQ(
      std::string(
          reinterpret_cast<const char*>(errorBuffer.get()) + errorSize -
              reason.size(),
          reason.size()),
      reason);
}

// Only the last message of a stream can say why the stream ended.
TEST(UcxExchangeProtocolTest, producerErrorRequiresEndOfStream) {
  MetadataMsg message;
  message.dataSizeBytes = 0;
  message.numRows = 0;
  message.atEnd = false;
  message.producerError = "Injected producer failure";

  VELOX_ASSERT_THROW(message.serialize(), "producerError");
}

// An intra-node consumer of a producer that was removed early learns the
// reason, as a remote consumer does from MetadataMsg.
TEST(IntraNodeTransferRegistryTest, cancelledTaskReportsProducerError) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId{"registryCancelledWithError"};
  registry->cancelTask(taskId, "Injected producer failure");

  const auto result =
      registry->poll({taskId, /*destination=*/0, /*sequenceNumber=*/0});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(
      result->producerError,
      std::optional<std::string>("Injected producer failure"));
  registry->clearCancelledTask(taskId);
}

// A producer removed after it sent all its output ends its stream cleanly.
TEST(IntraNodeTransferRegistryTest, cancelledFinishedTaskEndsCleanly) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId{"registryCancelledFinished"};
  registry->cancelTask(taskId, /*producerError=*/std::nullopt);

  const auto result =
      registry->poll({taskId, /*destination=*/0, /*sequenceNumber=*/0});
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(result->data, nullptr);
  EXPECT_EQ(result->producerError, std::nullopt);
  registry->clearCancelledTask(taskId);
}

TEST(IntraNodeTransferRegistryTest, publishedEndMarkerCarriesProducerError) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const IntraNodeTransferKey key{
      "registryPublishedError", /*destination=*/0, /*sequenceNumber=*/0};
  auto retrieved = registry->publish(
      key,
      nullptr,
      /*numRows=*/0,
      /*atEnd=*/true,
      "Injected producer failure");

  const auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
  EXPECT_EQ(
      result->producerError,
      std::optional<std::string>("Injected producer failure"));
  EXPECT_EQ(
      retrieved.wait_for(std::chrono::seconds(0)), std::future_status::ready);
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

// Drives 'exchange' until it finishes or has returned at least 'maxRows' rows,
// and returns the rows it read. Fails if a wait for data takes longer than
// 'maxWait', so that a stream that never ends fails the case instead of
// hanging the binary.
uint64_t drainToEnd(
    UcxExchange& exchange,
    uint64_t maxRows,
    std::chrono::seconds maxWait) {
  uint64_t numRows = 0;
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
  auto communicator = Communicator::getInstance();
  auto endpoint = communicator->assocEndpointRef(
      nullptr, HostPort{"127.0.0.1", communicator->getListenerPort()});
  ASSERT_NE(endpoint, nullptr);
  communicator->deferEndpointCleanup(endpoint);

  VELOX_ASSERT_THROW(
      drainToEnd(
          exchange,
          /*maxRows=*/std::numeric_limits<uint64_t>::max(),
          std::chrono::seconds(30)),
      "Connection to producer closed before it sent all its data");
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
      "Failed to connect to producer");
  exchange.close();
}

// An intra-node consumer reads a page from IntraNodeTransferRegistry some time
// after its server took the page from the producer's queue. A producer removed
// in between drops that page, so its consumer must get every row or fail. The
// consumer here stops reading: past UcxExchangeSource's backpressure high
// water mark it stops polling the registry, and the server waits with the
// producer's last page published but unread.
TEST_F(UcxExchangeFocusedTest, intraNodeProducerRemovedBeforePageReadFails) {
  auto& config = cudf_velox::CudfConfig::getInstance();
  const bool previousIntraNode = config.intraNodeExchange;
  config.intraNodeExchange = true;
  auto restoreIntraNode =
      folly::makeGuard([&]() { config.intraNodeExchange = previousIntraNode; });

  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "intraNodeSrc";
  const int partitionId = 0;
  // One page more than the consumer queues before it stops polling, so the
  // last one stays in the registry.
  const int numPages = UcxExchangeSource::kBackpressureHighWaterMark + 2;
  const vector_size_t numRowsPerPage = 100;

  auto srcTask = createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);
  queueManager()->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kPartitioned,
      /*numDestinations=*/1,
      /*numDrivers=*/1);
  auto removeProducer = folly::makeGuard([&]() {
    srcTask->requestAbort().wait();
    queueManager()->removeTask(srcTaskId);
  });
  for (int page = 0; page < numPages; ++page) {
    queueManager()->enqueue(
        srcTaskId,
        partitionId,
        makePackedColumns(numRowsPerPage),
        numRowsPerPage);
  }
  queueManager()->noMoreData(srcTaskId);

  core::PlanNodeId exchangeNodeId;
  auto sinkTask = createExchangeTask(
      taskPrefix + "intraNodeSink",
      UcxTestData::kTestRowType,
      partitionId,
      exchangeNodeId);
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

  // Wait until the server has taken every page: the producer's queue is empty
  // while the consumer has not read the last one.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (exchangeClient->queue()->size() <
             UcxExchangeSource::kBackpressureHighWaterMark + 1 ||
         queueManager()->stats(srcTaskId)->bufferedPages > 0) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "Consumer queued " << exchangeClient->queue()->size()
        << " pages; producer still holds "
        << queueManager()->stats(srcTaskId)->bufferedPages;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  srcTask->requestAbort().wait();
  queueManager()->removeTask(srcTaskId);
  removeProducer.dismiss();

  uint64_t numRows = 0;
  std::optional<std::string> error;
  try {
    numRows = drainToEnd(
        exchange,
        /*maxRows=*/std::numeric_limits<uint64_t>::max(),
        std::chrono::seconds(30));
  } catch (const VeloxException& e) {
    error = e.message();
  }
  exchange.close();

  if (numRows != static_cast<uint64_t>(numPages) * numRowsPerPage) {
    EXPECT_THAT(
        error,
        testing::Optional(
            testing::HasSubstr("Producer failed before sending all its data")))
        << "Rows received: " << numRows << " of "
        << static_cast<uint64_t>(numPages) * numRowsPerPage;
  }
}

// A producing task removed while its intra-node consumer polls
// IntraNodeTransferRegistry must stop that consumer with the producer's error,
// and must not leave it polling forever.
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
  sinkDriver->expectError();

  // Add split pointing to source task. Since we use a single Communicator,
  // the handshake will resolve to intra-node (same listener IP:port).
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

// Aborting a source task while UCXX tagRecv requests are in flight must stop
// its consumer with the producer's error, and must not crash.
// UcxExchangeSource::cleanUp() defers outstanding requests to
// Communicator::deferredRequests_, so their GPU buffers stay alive until UCX
// is done with them; destroying them earlier fails with
// cudaErrorIllegalAddress in ucp_mem_type_unpack.
TEST_F(UcxExchangeFocusedTest, deferredRequestCleanupOnTaskAbort) {
  // Ensure intra-node is disabled so we exercise the UCXX path (tagRecv).
  auto& config = cudf_velox::CudfConfig::getInstance();
  const bool origIntraNode = config.intraNodeExchange;
  config.intraNodeExchange = false;

  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "srcActiveTransfer";
  const std::string sinkTaskId = taskPrefix + "sinkAborted";
  const int numPartitions = 1;
  const int partitionId = 0;
  const int numDrivers = 1;
  // Enough data to keep UCXX transfers actively in-flight when we abort.
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

  // 4. Wait for UCXX transfers to be actively in-flight.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // 5. Abort the source task while transfers are in-flight.
  //    This triggers UcxExchangeServer::close() which cancels tagSend,
  //    and eventually UcxExchangeSource::cleanUp() which must defer
  //    the request (with its GPU buffer) to Communicator::deferredRequests_.
  srcTask->requestAbort();
  queueManager()->removeTask(srcTaskId);

  // 6. Wait for the sink driver to complete (it should detect the abort
  //    and finish, not crash with cudaErrorIllegalAddress).
  auto future =
      std::async(std::launch::async, [&]() { sinkDriver->joinThreads(); });
  auto status = future.wait_for(std::chrono::seconds(15));

  if (status != std::future_status::ready) {
    sinkTask->requestAbort();
    future.wait();
    FAIL() << "Sink driver did not complete within 15s after source abort"
           << " — possible hang in UCXX request cleanup";
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

  // 7. Join source mock threads.
  sourceMock->joinThreads();

  // 8. Allow Communicator's event loop to sweep deferred requests.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // If we reach here without crashing, the deferred cleanup is working.
  VLOG(0) << "deferredRequestCleanupOnTaskAbort: completed without crash";

  config.intraNodeExchange = origIntraNode;
}

} // namespace facebook::velox::ucx_exchange
