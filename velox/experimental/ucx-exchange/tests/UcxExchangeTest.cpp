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
#include <cudf/utilities/memory_resource.hpp>
#include <folly/Executor.h>
#include <folly/Synchronized.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/synchronization/EventCount.h>
#include <gmock/gmock.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <algorithm>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/ExchangeTransportRegistry.h"
#include "velox/exec/OutputTransportRegistry.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/PortUtil.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/tests/SinkDriverMock.h"
#include "velox/experimental/ucx-exchange/tests/SourceDriverMock.h"
#include "velox/experimental/ucx-exchange/tests/UcxPartitionedOutputMock.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestData.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/FlatVector.h"

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

class UcxExchangeTest : public testing::TestWithParam<ExchangeTestParams> {
 protected:
  // Chosen per process in SetUpTestCase() rather than hardcoded. The
  // communicator opens a listener on this port without address reuse, so two
  // runs of this binary in quick succession fail the second with
  // "bind(0.0.0.0:21346) failed: Address already in use" while the first port
  // is still in TIME_WAIT.
  static uint16_t communicatorPort_;

  // UcxExchangeSource computes the UCX port as the split URL's port + 3
  // (UcxExchangeSource.cpp), so remoteSplit() advertises this much below the
  // communicator's port for the round trip to land back on it.
  static constexpr int kSplitUrlPortOffset = 3;

  static constexpr auto kUnusedCoordinatorUrl =
      std::string_view("http://localhost:12345/bla");

  static std::shared_ptr<UcxOutputQueueManager> queueManager_;
  static std::shared_ptr<std::thread> communicatorThread_;
  static std::shared_ptr<Communicator> communicator_;
  static std::atomic<uint32_t> testCounter_;

  // Generate a unique task ID prefix for this test run to avoid collisions
  // between parametrized tests
  std::string getUniqueTaskPrefix() {
    return "t" + std::to_string(testCounter_.fetch_add(1)) + "_";
  }

  // Get the row type based on the table type from test params
  facebook::velox::RowTypePtr getRowType(TableType tableType) {
    if (tableType == TableType::WIDE) {
      return WideTestTable::kRowType;
    }
    return UcxTestData::kTestRowType;
  }

  // Check if we should skip this test for wide table configurations
  // Some tests are not yet compatible with WideTestTable
  bool shouldSkipWideTable() {
    ExchangeTestParams p = GetParam();
    return p.tableType == TableType::WIDE;
  }

  static void SetUpTestCase() {
    VLOG(0) << "setup test case, creating queue manager, communicator, etc..";
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

    // UcxExchangeSource derives the UCX port as the split URL's port + 3, and
    // remoteSplit() advertises communicatorPort_ - 3 to match, so the offset
    // must stay below the chosen port.
    const auto freePort = exec::test::getFreePort();
    ASSERT_GT(freePort, kSplitUrlPortOffset);
    ASSERT_LE(freePort, std::numeric_limits<uint16_t>::max());
    communicatorPort_ = static_cast<uint16_t>(freePort);

    queueManager_ = UcxOutputQueueManager::getInstanceRef();
    ContinueFuture future;
    communicator_ = facebook::velox::ucx_exchange::Communicator::initAndGet(
        communicatorPort_, std::string(kUnusedCoordinatorUrl), &future);
    if (communicator_) {
      communicatorThread_ = std::make_shared<std::thread>(
          &facebook::velox::ucx_exchange::Communicator::run,
          communicator_.get());
    } else {
      ADD_FAILURE() << "Communicator initialization failed";
    }
    future.wait();
  }

  static void TearDownTestCase() {
    communicator_->stop();
    communicator_.reset();
    communicatorThread_->join();
    communicatorThread_.reset();
  }

  void SetUp() override {
    VLOG(0) << "creating pool";
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool(
        "UcxTestMemoryPool");
  }

  exec::Split remoteSplit(std::string_view taskId, int partitionId) {
    std::string remoteUrl = fmt::format(
        "http://127.0.0.1:{}/v1/task/{}/results/{}",
        communicatorPort_ - kSplitUrlPortOffset,
        taskId,
        partitionId);
    return exec::Split(
        std::make_shared<facebook::velox::exec::RemoteConnectorSplit>(
            remoteUrl));
  }

  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
};

INSTANTIATE_TEST_SUITE_P(
    UcxExchangeTest,
    UcxExchangeTest,
    ::testing::ValuesIn(generateTestParams()),
    ExchangeTestParamsPrinter());

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
    queueManager_->initializeTask(
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
    queueManager_->removeTask(srcTaskId);
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
    queueManager_->initializeTask(
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
    queueManager_->removeTask(srcTaskId);
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
    queueManager_->initializeTask(
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
    queueManager_->removeTask(srcTaskId);
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
  queueManager_->initializeTask(
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
  queueManager_->removeTask(srcTaskId);

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
  auto stream = rmm::cuda_stream_default;
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
  queueManager_->initializeTask(
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
  queueManager_->removeTask(srcTaskId);

  VLOG(3) << "- UcxExchangeTest::realPartitionedOutputDataIntegrityTest";
}

// Focused regression test for shared UcxExchangeClient ownership. This
// intentionally creates and seeds the client directly, bypassing the normal
// task-split path, to isolate close behavior after the client is populated.
// Closing one operator must not close the shared client while another operator
// still needs to drain data.
TEST_P(UcxExchangeTest, sharedClientSurvivesOneExchangeClose) {
  // This test doesn't use parameters - run only for the first param set.
  if (GetParam() != generateTestParams().front()) {
    GTEST_SKIP() << "sharedClientSurvivesOneExchangeClose: runs only once";
  }

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
  queueManager_->initializeTask(
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

  queueManager_->removeTask(srcTaskId);
}

// Test that verifies intra-node exchange does not livelock when a producing
// task is removed while the consumer is polling IntraNodeTransferRegistry.
// Before the fix: test times out (livelock). After the fix: test passes.
TEST_P(UcxExchangeTest, intraNodeTaskRemovalLivelock) {
  // This test doesn't use parameters — run only for the first param set.
  {
    ExchangeTestParams p = GetParam();
    if (p.numSrcDrivers != 1 || p.numDstDrivers != 1 || p.numPartitions != 1 ||
        p.numChunks != 100 || p.numUpstreamTasks != 1 ||
        p.tableType != TableType::NARROW) {
      GTEST_SKIP() << "intraNodeTaskRemovalLivelock: runs only once";
    }
  }

  const std::string taskPrefix = getUniqueTaskPrefix();
  const std::string srcTaskId = taskPrefix + "srcProducerNeverSends";
  const std::string sinkTaskId = taskPrefix + "sinkConsumer";
  const int numPartitions = 1;
  const int partitionId = 0;

  // 1. Create and initialize source task but never enqueue any data.
  //    This simulates a producer that gets cancelled before producing.
  auto srcTask = createSourceTask(srcTaskId, pool_, UcxTestData::kTestRowType);
  queueManager_->initializeTask(
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
  queueManager_->removeTask(srcTaskId);

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
  // If we get here, the source correctly detected the cancelled task.
}

// Regression test for broadcast + intra-node SIGSEGV.
// Before the fix in Acceptor.cpp, broadcast tasks using intra-node transfer
// would crash because the intra-node source destructively moves gpu_data from
// a shared packed_columns object, corrupting it for other servers.
// The fix disables intra-node at handshake time for broadcast tasks, falling
// back to UCXX. This test verifies that broadcast with intra-node enabled
// completes without crash and delivers correct data.
TEST_P(UcxExchangeTest, broadcastIntraNodeFallback) {
  // This test doesn't use parameters — run only for the first param set.
  {
    ExchangeTestParams p = GetParam();
    if (p.numSrcDrivers != 1 || p.numDstDrivers != 1 || p.numPartitions != 1 ||
        p.numChunks != 100 || p.numUpstreamTasks != 1 ||
        p.tableType != TableType::NARROW) {
      GTEST_SKIP() << "broadcastIntraNodeFallback: runs only once";
    }
  }

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
  queueManager_->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kBroadcast,
      numDestinations,
      numDrivers);
  // Finalize destinations for broadcast.
  queueManager_->updateOutputBuffers(srcTaskId, numDestinations, true);

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
  queueManager_->removeTask(srcTaskId);
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
TEST_P(UcxExchangeTest, broadcastIntraNodePlaceholderRace) {
  // This test doesn't use parameters — run only for the first param set.
  {
    ExchangeTestParams p = GetParam();
    if (p.numSrcDrivers != 1 || p.numDstDrivers != 1 || p.numPartitions != 1 ||
        p.numChunks != 100 || p.numUpstreamTasks != 1 ||
        p.tableType != TableType::NARROW) {
      GTEST_SKIP() << "broadcastIntraNodePlaceholderRace: runs only once";
    }
  }

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
  queueManager_->initializeTask(
      srcTask,
      core::PartitionedOutputNode::Kind::kBroadcast,
      numDestinations,
      numDrivers);

  // Step 4: Finalize destinations for broadcast.
  queueManager_->updateOutputBuffers(srcTaskId, numDestinations, true);

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
  queueManager_->removeTask(srcTaskId);
  config.intraNodeExchange = origIntraNode;
}

// Test that UcxPartitionedOutput's batch accumulation correctly merges many
// small input chunks into fewer, larger output chunks while preserving all rows
// and data integrity.
TEST_P(UcxExchangeTest, batchAccumulationTest) {
  // This test doesn't use parameters — run only for the first param set.
  {
    ExchangeTestParams p = GetParam();
    if (p.numSrcDrivers != 1 || p.numDstDrivers != 1 || p.numPartitions != 1 ||
        p.numChunks != 100 || p.numUpstreamTasks != 1 ||
        p.tableType != TableType::NARROW) {
      GTEST_SKIP() << "batchAccumulationTest: runs only once";
    }
  }

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
    queueManager_->initializeTask(
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

    queueManager_->removeTask(srcTaskId);
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
    queueManager_->initializeTask(
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

    queueManager_->removeTask(srcTaskId);
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
    queueManager_->initializeTask(
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

    queueManager_->removeTask(srcTaskId);
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
    queueManager_->initializeTask(
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

    queueManager_->removeTask(srcTaskId);
  }
}

// Regression test: aborting a source task while UCXX tagRecv requests are
// in-flight must not crash.  Before the deferred-request-cleanup fix,
// UcxExchangeSource::cleanUp() would destroy the request (and its GPU
// buffer) while UCX was still using it, causing cudaErrorIllegalAddress in
// ucp_mem_type_unpack.  The fix moves outstanding requests to
// Communicator::deferredRequests_ so buffers stay alive until UCX finishes.
TEST_P(UcxExchangeTest, deferredRequestCleanupOnTaskAbort) {
  // This test doesn't use parameters — run only for the first param set.
  {
    ExchangeTestParams p = GetParam();
    if (p.numSrcDrivers != 1 || p.numDstDrivers != 1 || p.numPartitions != 1 ||
        p.numChunks != 100 || p.numUpstreamTasks != 1 ||
        p.tableType != TableType::NARROW) {
      GTEST_SKIP() << "deferredRequestCleanupOnTaskAbort: runs only once";
    }
  }

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
  queueManager_->initializeTask(
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
  queueManager_->removeTask(srcTaskId);

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

  // 7. Join source mock threads.
  sourceMock->joinThreads();

  // 8. Allow Communicator's event loop to sweep deferred requests.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // If we reach here without crashing, the deferred cleanup is working.
  VLOG(0) << "deferredRequestCleanupOnTaskAbort: completed without crash";

  config.intraNodeExchange = origIntraNode;
}

namespace {

// Row type for the real-Task cases below. Two fixed-width columns keep the
// host/device round trip cheap while still exercising a multi-column table.
const RowTypePtr& taskShuffleRowType() {
  static const RowTypePtr rowType = ROW({"c0", "c1"}, {INTEGER(), BIGINT()});
  return rowType;
}

// Serde named on both plan nodes of the two-fragment plans. The UCX transport
// ignores it because it ships packed cudf tables, but PlanBuilder requires a
// registered name on the nodes.
const std::string& taskShuffleSerdeKind() {
  static const std::string kind =
      VectorSerde::kindName(VectorSerde::Kind::kPresto);
  return kind;
}

// Registers what a serialized-page transport needs. OperatorTestBase does this
// for the tests derived from it; this suite has its own fixture.
void registerSerdes() {
  if (!isRegisteredVectorSerde()) {
    serializer::presto::PrestoVectorSerde::registerVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(taskShuffleSerdeKind())) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
}

// Drives every Task built below. QueryCtx keeps a raw pointer to the executor,
// so it has to outlive the Tasks. It is deliberately never destroyed: folly
// joins its threads in the destructor, and doing that at static destruction
// races with the communicator shutdown this binary performs at exit.
folly::CPUThreadPoolExecutor* taskShuffleExecutor() {
  static auto* executor = new folly::CPUThreadPoolExecutor(8);
  return executor;
}

std::shared_ptr<exec::Task> makeTaskShuffleTask(
    const std::string& taskId,
    core::PlanFragment planFragment,
    exec::Consumer consumer) {
  auto queryCtx = core::QueryCtx::create(
      taskShuffleExecutor(),
      core::QueryConfig(std::unordered_map<std::string, std::string>{}));
  return exec::Task::create(
      taskId,
      std::move(planFragment),
      /*destination=*/0,
      std::move(queryCtx),
      exec::Task::ExecutionMode::kParallel,
      std::move(consumer));
}

// Builds one deterministic batch. c0 runs 'firstValue', 'firstValue + stride',
// ... so callers can interleave the ranges of several producers, and c1 is
// derived from c0 so a column mix-up cannot go unnoticed.
RowVectorPtr makeTaskShuffleBatch(
    memory::MemoryPool* pool,
    int32_t firstValue,
    int32_t stride,
    vector_size_t numRowsInBatch) {
  auto keys =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), numRowsInBatch, pool);
  auto payload =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), numRowsInBatch, pool);
  for (vector_size_t i = 0; i < numRowsInBatch; ++i) {
    const int32_t value = firstValue + i * stride;
    keys->set(i, value);
    payload->set(i, static_cast<int64_t>(value) * 10);
  }
  return std::make_shared<RowVector>(
      pool,
      taskShuffleRowType(),
      BufferPtr(nullptr),
      numRowsInBatch,
      std::vector<VectorPtr>{keys, payload});
}

// Brings a batch the consumer saw into 'pool'. CallbackSinkAdapter declares it
// does not accept GPU input, so CompileState splices a CudfToVelox in front of
// the sink and the batches arrive on the host; the CudfVector branch is here so
// a change in that wiring surfaces as a failed assertion rather than a crash.
// Either way the result is a fresh vector owned by 'pool': what the consumer
// hands over is allocated from its operator pools, which report a leak and fail
// the arbitrator's reservation check if anything still references them when the
// Task is destroyed.
RowVectorPtr toHost(const RowVectorPtr& batch, memory::MemoryPool* pool) {
  auto cudfVector = std::dynamic_pointer_cast<cudf_velox::CudfVector>(batch);
  if (cudfVector != nullptr) {
    auto stream = cudfVector->stream();
    auto host = cudf_velox::with_arrow::toVeloxColumn(
        cudfVector->getTableView(),
        pool,
        taskShuffleRowType(),
        "",
        stream,
        cudf::get_current_device_resource_ref());
    stream.synchronize();
    return host;
  }
  auto copy =
      BaseVector::create<RowVector>(taskShuffleRowType(), batch->size(), pool);
  copy->copy(batch.get(), 0, 0, batch->size());
  return copy;
}

// Collects a consumer Task's output. The Consumer callback runs on driver
// threads, so the batches need their own lock.
class TaskShuffleResults {
 public:
  exec::Consumer consumer() {
    return
        [this](
            RowVectorPtr batch, bool /*drained*/, ContinueFuture* /*future*/) {
          if (batch != nullptr && batch->size() > 0) {
            batches_.wlock()->push_back(std::move(batch));
          }
          return exec::BlockingReason::kNotBlocked;
        };
  }

  std::vector<RowVectorPtr> batches() const {
    return batches_.copy();
  }

  void clear() {
    batches_.wlock()->clear();
  }

 private:
  folly::Synchronized<std::vector<RowVectorPtr>> batches_;
};

// Pairs registerCudf() with unregisterCudf() so the cuDF driver adapter, and
// the transport registrations and memory resources it installs, do not outlive
// the case even if the body throws.
class CudfRegistration {
 public:
  CudfRegistration() {
    cudf_velox::registerCudf();
  }

  ~CudfRegistration() {
    cudf_velox::unregisterCudf();
  }

  CudfRegistration(const CudfRegistration&) = delete;
  CudfRegistration& operator=(const CudfRegistration&) = delete;
};

// A producer task id paired with the batches its Values node emits.
using TaskShuffleProducer = std::pair<std::string, std::vector<RowVectorPtr>>;

// Sums the row counts of 'batches'.
vector_size_t totalRows(const std::vector<RowVectorPtr>& batches) {
  vector_size_t total = 0;
  for (const auto& batch : batches) {
    total += batch->size();
  }
  return total;
}

// The UCX round trip goes through the communicator's event loop, so allow well
// over waitForTaskCompletion's 1s default.
constexpr uint64_t kTaskShuffleMaxWaitMicros = 60'000'000;

// Starts one 'Values -> PartitionedOutput' Task per entry in 'producers', all
// on 'transportKind'. The Tasks are returned rather than waited for: a producer
// only completes once whatever reads its output buffer has drained it.
std::vector<std::shared_ptr<exec::Task>> startTaskShuffleProducers(
    const std::string& transportKind,
    const std::vector<TaskShuffleProducer>& producers) {
  std::vector<std::shared_ptr<exec::Task>> producerTasks;
  for (const auto& producer : producers) {
    auto plan = exec::test::PlanBuilder()
                    .values(producer.second)
                    .partitionedOutput(
                        /*keys=*/{},
                        /*numPartitions=*/1,
                        /*outputLayout=*/{},
                        taskShuffleSerdeKind(),
                        transportKind)
                    .planFragment();
    auto task = makeTaskShuffleTask(producer.first, std::move(plan), nullptr);
    task->start(1);
    producerTasks.push_back(std::move(task));
  }
  return producerTasks;
}

// The operator types of 'pipeline', in pipeline order. Read from TaskStats
// rather than from the Driver because TaskStats is the vector
// Driver::initializeOperatorStats() indexes by operatorId: a repeated id lands
// here as an overwritten entry plus a nameless leftover.
std::vector<std::string> operatorTypesOf(const exec::PipelineStats& pipeline) {
  std::vector<std::string> types;
  types.reserve(pipeline.operatorStats.size());
  for (const auto& stats : pipeline.operatorStats) {
    types.push_back(stats.operatorType);
  }
  return types;
}

// The operator ids of 'pipeline', in pipeline order.
std::vector<int32_t> operatorIdsOf(const exec::PipelineStats& pipeline) {
  std::vector<int32_t> ids;
  ids.reserve(pipeline.operatorStats.size());
  for (const auto& stats : pipeline.operatorStats) {
    ids.push_back(stats.operatorId);
  }
  return ids;
}

// The c0 column of 'batches' flattened into one sequence, so that a single
// assertion can report the first ordering violation rather than one per row.
std::vector<int32_t> sortKeysOf(const std::vector<RowVectorPtr>& batches) {
  std::vector<int32_t> sortKeys;
  for (const auto& batch : batches) {
    auto* keyColumn = batch->childAt(0)->as<SimpleVector<int32_t>>();
    VELOX_CHECK_NOT_NULL(keyColumn, "Sort key column is not a flat INTEGER");
    for (vector_size_t i = 0; i < batch->size(); ++i) {
      VELOX_CHECK(!keyColumn->isNullAt(i), "Sort key is null at row: {}", i);
      sortKeys.push_back(keyColumn->valueAt(i));
    }
  }
  return sortKeys;
}

// Index of the first key that is not strictly greater than its predecessor, or
// 'sortKeys.size()' when the whole sequence ascends.
int64_t firstUnorderedIndex(const std::vector<int32_t>& sortKeys) {
  return std::distance(
      sortKeys.begin(),
      std::adjacent_find(
          sortKeys.begin(), sortKeys.end(), [](int32_t left, int32_t right) {
            return left >= right;
          }));
}

// Starts one 'Values -> PartitionedOutput' producer Task per entry in
// 'producers', all on 'transportKind', then runs 'consumerPlan' against them,
// feeding it one split per producer built by 'splitFor'. Returns the
// consumer's output in host memory. Copies the consumer's stats into
// 'consumerStats' when it is not null, which is the only way to see the
// consumer's operator pipeline after its Task is gone.
std::vector<RowVectorPtr> runTaskShuffle(
    const std::string& transportKind,
    const std::vector<TaskShuffleProducer>& producers,
    const core::PlanFragment& consumerPlan,
    const core::PlanNodeId& consumerNodeId,
    const std::string& consumerTaskId,
    const std::function<exec::Split(const std::string&)>& splitFor,
    memory::MemoryPool* pool,
    exec::TaskStats* consumerStats) {
  auto producerTasks = startTaskShuffleProducers(transportKind, producers);

  TaskShuffleResults results;
  auto consumerTask =
      makeTaskShuffleTask(consumerTaskId, consumerPlan, results.consumer());
  consumerTask->start(1);
  for (const auto& producer : producers) {
    consumerTask->addSplit(consumerNodeId, splitFor(producer.first));
  }
  consumerTask->noMoreSplits(consumerNodeId);

  VELOX_CHECK(
      exec::test::waitForTaskCompletion(
          consumerTask.get(), kTaskShuffleMaxWaitMicros),
      "Consumer task did not complete for transport: {}",
      transportKind);
  for (const auto& task : producerTasks) {
    VELOX_CHECK(
        exec::test::waitForTaskCompletion(
            task.get(), kTaskShuffleMaxWaitMicros),
        "Producer task did not complete: {}",
        task->taskId());
  }

  if (consumerStats != nullptr) {
    *consumerStats = consumerTask->taskStats();
  }

  std::vector<RowVectorPtr> hostBatches;
  for (const auto& batch : results.batches()) {
    hostBatches.push_back(toHost(batch, pool));
  }
  // Drop the consumer's own vectors before the Tasks go out of scope below.
  // They are allocated from operator pools that check for leaks on destruction,
  // and 'results' outlives 'consumerTask' by declaration order.
  results.clear();
  return hostBatches;
}

// Row shape shared by taskShuffleOverUcx and the merge case below.
constexpr vector_size_t kTaskShuffleRowsPerBatch = 1'000;
constexpr int kTaskShuffleNumBatches = 3;
constexpr int64_t kTaskShuffleTotalRows =
    static_cast<int64_t>(kTaskShuffleNumBatches) * kTaskShuffleRowsPerBatch;

std::vector<RowVectorPtr> makeTaskShuffleBatches(memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> batches;
  batches.reserve(kTaskShuffleNumBatches);
  for (int i = 0; i < kTaskShuffleNumBatches; ++i) {
    batches.push_back(makeTaskShuffleBatch(
        pool,
        /*firstValue=*/i * kTaskShuffleRowsPerBatch,
        /*stride=*/1,
        kTaskShuffleRowsPerBatch));
  }
  return batches;
}

} // namespace

// Runs a two-fragment plan over UCX through real Tasks. The
// realPartitionedOutput cases above drive UcxPartitionedOutput and UcxExchange
// from SourceDriverMock/SinkDriverMock; this one goes through Task and
// LocalPlanner, so the exchange operator is resolved through
// ExchangeTransportRegistry and the output buffer through
// OutputTransportRegistry, both keyed by the transportKind on the plan nodes.
// It is the non-merge control for mergeExchangeOverUcxIsGloballyOrdered: if
// both fail the Task-level UCX path is at fault, if only the merge case fails
// the merge expansion is.
TEST_P(UcxExchangeTest, taskShuffleOverUcx) {
  // This test doesn't use parameters - run only for the first param set.
  if (GetParam() != generateTestParams().front()) {
    GTEST_SKIP() << "taskShuffleOverUcx: runs only once";
  }

  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();

  // registerCudf() is what seeds both transport registries in this tree, so it
  // is a prerequisite of the Task-level path and not just of the merge case.
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  // The batches stay on the host. UcxPartitionedOutput requires CudfVector
  // input, but UcxPartitionedOutputAdapter declares acceptsGpuInput(), so
  // CompileState splices a CudfFromVelox between Values and it. Staging the
  // batches on the device first makes that conversion operator call childAt() on
  // a childless CudfVector and the producer task dies with
  // "Trying to access non-existing child in RowVector".
  const auto expected = makeTaskShuffleBatches(pool_.get());

  core::PlanNodeId exchangeNodeId;
  auto consumerPlan = exec::test::PlanBuilder()
                          .exchange(
                              taskShuffleRowType(),
                              taskShuffleSerdeKind(),
                              std::string{core::TransportKind::kUcx})
                          .capturePlanNodeId(exchangeNodeId)
                          .planFragment();

  auto actual = runTaskShuffle(
      std::string{core::TransportKind::kUcx},
      {{taskPrefix + "ucxProducer", expected}},
      consumerPlan,
      exchangeNodeId,
      taskPrefix + "ucxConsumer",
      [this](const std::string& taskId) { return remoteSplit(taskId, 0); },
      pool_.get(),
      /*consumerStats=*/nullptr);

  EXPECT_EQ(totalRows(actual), kTaskShuffleTotalRows);
  EXPECT_TRUE(exec::test::assertEqualResults(expected, actual));
}

// A fragment whose output layout is empty ships rows that carry no columns --
// the build side of a cross join that projects nothing, for instance. A cuDF
// table derives num_rows() from its columns, so once such a payload is packed it
// can no longer report its own cardinality: the count only survives if it
// travels beside the data, in MetadataMsg on the remote path and in the registry
// entry on the intra-node one.
//
// The row count is the entire observable result here, which is the point. With
// the count derived from the packed table instead, UcxExchange rebuilt a 0-row
// vector and the consumer task died on
// "Operator::getOutput() must return nullptr or a non-empty vector", which a
// downstream global aggregation would then swallow into a wrong scalar rather
// than an error -- SELECT count(*) over such a plan returned 0.
TEST_P(UcxExchangeTest, zeroColumnPayloadKeepsItsRowCount) {
  // This test doesn't use parameters - run only for the first param set.
  if (GetParam() != generateTestParams().front()) {
    GTEST_SKIP() << "zeroColumnPayloadKeepsItsRowCount: runs only once";
  }

  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();

  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  // Zero columns, non-zero rows, staged on the host so that CudfFromVelox takes
  // its zero-column path and hands UcxPartitionedOutput a CudfVector whose
  // size() is the only remaining record of the cardinality.
  const RowTypePtr zeroColumnRowType = ROW({});
  constexpr vector_size_t kRowsPerBatch = 125;
  constexpr int kNumBatches = 2;
  constexpr int64_t kExpectedRows =
      static_cast<int64_t>(kNumBatches) * kRowsPerBatch;

  std::vector<RowVectorPtr> batches;
  batches.reserve(kNumBatches);
  for (int i = 0; i < kNumBatches; ++i) {
    batches.push_back(
        std::make_shared<RowVector>(
            pool_.get(),
            zeroColumnRowType,
            BufferPtr(nullptr),
            kRowsPerBatch,
            std::vector<VectorPtr>{}));
  }

  core::PlanNodeId exchangeNodeId;
  auto consumerPlan =
      exec::test::PlanBuilder()
          .exchange(
              zeroColumnRowType,
              taskShuffleSerdeKind(),
              std::string{core::TransportKind::kUcx})
          .capturePlanNodeId(exchangeNodeId)
          .planFragment();

  const auto producerTaskId = taskPrefix + "zeroColumnProducer";
  auto producerTasks = startTaskShuffleProducers(
      std::string{core::TransportKind::kUcx}, {{producerTaskId, batches}});

  // Not runTaskShuffle(): its toHost() converts through taskShuffleRowType(),
  // and there is nothing to convert here. The counts are read directly instead.
  TaskShuffleResults results;
  auto consumerTask = makeTaskShuffleTask(
      taskPrefix + "zeroColumnConsumer", consumerPlan, results.consumer());
  consumerTask->start(1);
  consumerTask->addSplit(exchangeNodeId, remoteSplit(producerTaskId, 0));
  consumerTask->noMoreSplits(exchangeNodeId);

  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          consumerTask.get(), kTaskShuffleMaxWaitMicros))
      << "Consumer task did not complete";
  for (const auto& task : producerTasks) {
    ASSERT_TRUE(
        exec::test::waitForTaskCompletion(
            task.get(), kTaskShuffleMaxWaitMicros))
        << "Producer task did not complete: " << task->taskId();
  }

  int64_t receivedRows = 0;
  for (const auto& batch : results.batches()) {
    EXPECT_EQ(batch->type()->size(), 0) << "expected a column-less batch";
    receivedRows += batch->size();
  }
  // Drop the consumer's vectors before the Tasks go out of scope, for the reason
  // given in runTaskShuffle().
  results.clear();

  EXPECT_EQ(receivedRows, kExpectedRows);
}

// End-to-end guard on the merge path this tree restored: a kUcx
// MergeExchangeNode runs as UcxExchange followed by CudfOrderBy, because
// UcxExchangeClient multiplexes every source into one queue and so destroys the
// per-source orderings exec::MergeExchange relies on. The transport builds only
// the exchange; the sort is spliced in behind it by the cuDF
// driver-adaptation pass, which is also what renumbers the operator ids.
//
// Two properties are asserted, and only two: global ordering, since nothing
// about merge internals, batch boundaries or per-source runs survives that
// substitution, and the shape of the resulting operator pipeline, since that is
// where the expansion and the renumbering are observable.
TEST_P(UcxExchangeTest, mergeExchangeOverUcxIsGloballyOrdered) {
  // This test doesn't use parameters - run only for the first param set.
  if (GetParam() != generateTestParams().front()) {
    GTEST_SKIP() << "mergeExchangeOverUcxIsGloballyOrdered: runs only once";
  }

  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();

  // Start from an empty pair of registries so the assertions below prove that
  // registerCudf() -- the path a real worker takes with cudf.exchange enabled
  // -- is what seeds the kUcx transport on both sides of the edge.
  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;
  auto exchangeEntry = exec::ExchangeTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(exchangeEntry, nullptr);
  // Without this, Task fails the MergeExchangeNode before any driver is built.
  ASSERT_NE(exchangeEntry->makeMergeExchangeOperator, nullptr);
  ASSERT_NE(
      exec::OutputTransportRegistry::tryGet(
          std::string{core::TransportKind::kUcx}),
      nullptr);

  // Two producers with interleaved key ranges: neither is globally ordered on
  // its own, so an ordered result can only come from ordering across both.
  // CudfOrderBy accumulates every batch on the device before sorting, so the
  // row counts stay small.
  constexpr vector_size_t kNumRowsPerProducer = 512;
  constexpr int kNumProducers = 2;
  std::vector<RowVectorPtr> expected;
  std::vector<TaskShuffleProducer> producers;
  for (int i = 0; i < kNumProducers; ++i) {
    auto batch = makeTaskShuffleBatch(
        pool_.get(),
        /*firstValue=*/i,
        /*stride=*/kNumProducers,
        kNumRowsPerProducer);
    expected.push_back(batch);
    // Host batches: CompileState puts a CudfFromVelox between Values and
    // UcxPartitionedOutput, see taskShuffleOverUcx.
    producers.emplace_back(
        taskPrefix + "mergeProducer" + std::to_string(i),
        std::vector<RowVectorPtr>{batch});
  }

  core::PlanNodeId mergeNodeId;
  auto consumerPlan = exec::test::PlanBuilder()
                          .mergeExchange(
                              taskShuffleRowType(),
                              {"c0"},
                              taskShuffleSerdeKind(),
                              std::string{core::TransportKind::kUcx})
                          .capturePlanNodeId(mergeNodeId)
                          .planFragment();

  exec::TaskStats consumerStats;
  auto actual = runTaskShuffle(
      std::string{core::TransportKind::kUcx},
      producers,
      consumerPlan,
      mergeNodeId,
      taskPrefix + "mergeConsumer",
      [this](const std::string& taskId) { return remoteSplit(taskId, 0); },
      pool_.get(),
      &consumerStats);

  ASSERT_EQ(totalRows(actual), kNumProducers * kNumRowsPerProducer);
  EXPECT_TRUE(exec::test::assertEqualResults(expected, actual));

  // One plan node became four operators: the exchange the transport built, the
  // sort the cuDF pass spliced in behind it, the conversion back to host
  // vectors that the CallbackSink needs, and the sink. The ids have to be
  // consecutive from zero -- Driver::initializeOperatorStats indexes the stats
  // by operatorId, so a repeated id would silently overwrite a slot and leave a
  // nameless one behind.
  ASSERT_EQ(consumerStats.pipelineStats.size(), 1);
  EXPECT_THAT(
      operatorTypesOf(consumerStats.pipelineStats[0]),
      testing::ElementsAre(
          "UcxExchange", "CudfOrderBy", "CudfToVelox", "CallbackSink"));
  EXPECT_THAT(
      operatorIdsOf(consumerStats.pipelineStats[0]),
      testing::ElementsAre(0, 1, 2, 3));

  const auto sortKeys = sortKeysOf(actual);
  const auto firstUnordered = firstUnorderedIndex(sortKeys);
  EXPECT_EQ(firstUnordered, static_cast<int64_t>(sortKeys.size()))
      << "MergeExchange over UCX produced unordered output at index: "
      << firstUnordered;
}

// The same merge expansion as above, but in the driver shape where the operator
// ids are not repaired by anything else: every operator of the pipeline runs on
// the GPU, so the cuDF pass splices in no conversion operator and the merge
// expansion is the only replacement made.
//
// That distinction is the whole point of the case. When the transport built
// both operators itself it numbered them by plan node, giving them the same
// operatorId, and any unrelated splice hid it again --
// DriverFactory::replaceOperators renumbers the entire driver, so a single
// CudfToVelox in front of a CPU sink is enough to make the ids come out
// consecutive. mergeExchangeOverUcxIsGloballyOrdered ends in a CallbackSink and
// therefore cannot see the difference. This pipeline ends in
// UcxPartitionedOutput, which consumes CudfVectors, so nothing is spliced and
// the ids are whatever produced them: [0, 1, 2] when the expansion goes through
// replaceOperators, and a repeated id when it does not -- [0, 0, 2] with the
// registry-side expansion, which numbered the output operator after two
// operators rather than one.
//
// Three Tasks are needed to reach that shape: the two producers feed the merge,
// and a third Task drains the sorted output so the middle Task can finish and
// report its stats.
TEST_P(UcxExchangeTest, mergeExchangeOverUcxNumbersOperatorsConsecutively) {
  // This test doesn't use parameters - run only for the first param set.
  if (GetParam() != generateTestParams().front()) {
    GTEST_SKIP()
        << "mergeExchangeOverUcxNumbersOperatorsConsecutively: runs only once";
  }

  registerSerdes();
  const auto taskPrefix = getUniqueTaskPrefix();
  const auto ucx = std::string{core::TransportKind::kUcx};

  exec::ExchangeTransportRegistry::unregisterAll();
  exec::OutputTransportRegistry::unregisterAll();
  CudfRegistration cudfRegistration;

  // Interleaved key ranges again, so the output cannot be ordered by accident.
  constexpr vector_size_t kNumRowsPerProducer = 512;
  constexpr int kNumProducers = 2;
  std::vector<RowVectorPtr> expected;
  std::vector<TaskShuffleProducer> producers;
  for (int i = 0; i < kNumProducers; ++i) {
    auto batch = makeTaskShuffleBatch(
        pool_.get(),
        /*firstValue=*/i,
        /*stride=*/kNumProducers,
        kNumRowsPerProducer);
    expected.push_back(batch);
    producers.emplace_back(
        taskPrefix + "gpuMergeProducer" + std::to_string(i),
        std::vector<RowVectorPtr>{batch});
  }
  auto producerTasks = startTaskShuffleProducers(ucx, producers);

  // The Task under test: a kUcx merge exchange straight into a kUcx partitioned
  // output, which is the all-GPU driver.
  core::PlanNodeId mergeNodeId;
  auto sorterPlan =
      exec::test::PlanBuilder()
          .mergeExchange(
              taskShuffleRowType(), {"c0"}, taskShuffleSerdeKind(), ucx)
          .capturePlanNodeId(mergeNodeId)
          .partitionedOutput(
              /*keys=*/{},
              /*numPartitions=*/1,
              /*outputLayout=*/{},
              taskShuffleSerdeKind(),
              ucx)
          .planFragment();
  const auto sorterTaskId = taskPrefix + "gpuMergeSorter";
  auto sorterTask = makeTaskShuffleTask(sorterTaskId, sorterPlan, nullptr);
  sorterTask->start(1);
  for (const auto& producer : producers) {
    sorterTask->addSplit(mergeNodeId, remoteSplit(producer.first, 0));
  }
  sorterTask->noMoreSplits(mergeNodeId);

  // A plain exchange to drain the sorted output. Its own pipeline is not under
  // test; it exists so the sorter's output buffer is consumed.
  core::PlanNodeId drainNodeId;
  auto drainPlan =
      exec::test::PlanBuilder()
          .exchange(taskShuffleRowType(), taskShuffleSerdeKind(), ucx)
          .capturePlanNodeId(drainNodeId)
          .planFragment();
  TaskShuffleResults results;
  auto drainTask = makeTaskShuffleTask(
      taskPrefix + "gpuMergeDrain", drainPlan, results.consumer());
  drainTask->start(1);
  drainTask->addSplit(drainNodeId, remoteSplit(sorterTaskId, 0));
  drainTask->noMoreSplits(drainNodeId);

  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          drainTask.get(), kTaskShuffleMaxWaitMicros))
      << "Drain task did not complete";
  ASSERT_TRUE(
      exec::test::waitForTaskCompletion(
          sorterTask.get(), kTaskShuffleMaxWaitMicros))
      << "Sorter task did not complete";
  for (const auto& task : producerTasks) {
    ASSERT_TRUE(
        exec::test::waitForTaskCompletion(
            task.get(), kTaskShuffleMaxWaitMicros))
        << "Producer task did not complete: " << task->taskId();
  }

  const auto sorterStats = sorterTask->taskStats();

  std::vector<RowVectorPtr> actual;
  for (const auto& batch : results.batches()) {
    actual.push_back(toHost(batch, pool_.get()));
  }
  // The consumer's vectors come from its operator pools, which check for leaks
  // when the Task is destroyed; 'results' outlives 'drainTask' by declaration
  // order, so let them go here. See runTaskShuffle.
  results.clear();

  ASSERT_EQ(sorterStats.pipelineStats.size(), 1);
  EXPECT_THAT(
      operatorTypesOf(sorterStats.pipelineStats[0]),
      testing::ElementsAre(
          "UcxExchange", "CudfOrderBy", "cudfPartitionedOutput"));
  EXPECT_THAT(
      operatorIdsOf(sorterStats.pipelineStats[0]),
      testing::ElementsAre(0, 1, 2));

  // The consequence of a repeated id, asserted directly: stats.resize() default
  // constructs OperatorStats(0, 0, "", ""), and every slot has to be claimed by
  // an operator afterwards.
  for (const auto& pipeline : sorterStats.pipelineStats) {
    for (const auto& stats : pipeline.operatorStats) {
      EXPECT_FALSE(stats.operatorType.empty())
          << "Operator stats slot left unwritten at operatorId: "
          << stats.operatorId << ", planNodeId: " << stats.planNodeId;
    }
  }

  // The sort still has to be a sort, or the assertions above would be happy
  // with a pipeline that merely has the right shape.
  ASSERT_EQ(totalRows(actual), kNumProducers * kNumRowsPerProducer);
  EXPECT_TRUE(exec::test::assertEqualResults(expected, actual));
  const auto sortKeys = sortKeysOf(actual);
  const auto firstUnordered = firstUnorderedIndex(sortKeys);
  EXPECT_EQ(firstUnordered, static_cast<int64_t>(sortKeys.size()))
      << "All-GPU merge over UCX produced unordered output at index: "
      << firstUnordered;
}

std::shared_ptr<UcxOutputQueueManager> UcxExchangeTest::queueManager_;
std::shared_ptr<std::thread> UcxExchangeTest::communicatorThread_;
std::shared_ptr<Communicator> UcxExchangeTest::communicator_;
std::atomic<uint32_t> UcxExchangeTest::testCounter_{0};
uint16_t UcxExchangeTest::communicatorPort_{0};

} // namespace facebook::velox::ucx_exchange
