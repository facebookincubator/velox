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
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <folly/Executor.h>
#include <folly/synchronization/EventCount.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <chrono>
#include <future>
#include <memory>
#include <sstream>
#include <vector>
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/tests/SinkDriverMock.h"
#include "velox/experimental/ucx-exchange/tests/SourceDriverMock.h"
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
  static constexpr uint16_t kCommunicatorPort = 21346;
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

    queueManager_ = UcxOutputQueueManager::getInstanceRef();
    ContinueFuture future;
    communicator_ = facebook::velox::ucx_exchange::Communicator::initAndGet(
        kCommunicatorPort, std::string(kUnusedCoordinatorUrl), &future);
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
        kCommunicatorPort - 3,
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
        {core::QueryConfig::kUcxPartitionedOutputBatchRows,
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

std::shared_ptr<UcxOutputQueueManager> UcxExchangeTest::queueManager_;
std::shared_ptr<std::thread> UcxExchangeTest::communicatorThread_;
std::shared_ptr<Communicator> UcxExchangeTest::communicator_;
std::atomic<uint32_t> UcxExchangeTest::testCounter_{0};

} // namespace facebook::velox::ucx_exchange
