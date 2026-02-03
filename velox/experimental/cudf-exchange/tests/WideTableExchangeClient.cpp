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

/// @file WideTableExchangeClient.cpp
/// @brief Standalone client that connects to a WideTableExchangeServer,
/// fetches one partition of data, and reports transfer statistics.
///
/// Usage:
///   export UCX_TCP_CM_REUSEADDR=y
///   export CUDA_VISIBLE_DEVICES=7
///   ./wide_table_exchange_client --server_host=127.0.0.1 --server_port=21346 \
///       --task_id=wideTableTask0 --partition_id=0 --num_sink_drivers=4
///
/// Profiling with Nsight Systems:
///   nsys profile --trace=cuda,nvtx ./wide_table_exchange_client ...

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <nvtx3/nvToolsExt.h>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/Exchange.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestData.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestHelpers.h"
#include "velox/experimental/cudf-exchange/tests/SinkDriverMock.h"

using namespace facebook::velox::cudf_exchange;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

DEFINE_string(server_host, "127.0.0.1", "Server hostname or IP address");
DEFINE_uint32(server_port, 21346, "Server port for UCX communication");
DEFINE_string(task_id, "wideTableTask0", "Remote task ID to fetch data from");
DEFINE_uint32(partition_id, 0, "Partition ID to fetch (0, 1, ...)");
DEFINE_uint32(num_sink_drivers, 4, "Number of sink drivers (parallel receivers)");
DEFINE_uint32(client_port, 21347, "Local port for UCX communication");
DEFINE_uint32(rmm_pool_percent, 50, "Percentage of free GPU memory for RMM pool");
DEFINE_bool(complex_types, false, "If true, expect WideComplexTestTable schema (includes STRING and STRUCT)");

/// @brief Creates a remote split pointing to the server's partition.
/// The URL format follows Velox's exchange protocol.
exec::Split createRemoteSplit(
    const std::string& host,
    uint16_t port,
    const std::string& taskId,
    int partitionId) {
  // Port offset -3 is used per the cudf-exchange test infrastructure
  // to derive the HTTP port from the UCX port
  std::string remoteUrl = "http://" + host + ":" +
      std::to_string(port - 3) + "/v1/task/" + taskId +
      "/results/" + std::to_string(partitionId);
  return exec::Split(
      std::make_shared<exec::RemoteConnectorSplit>(remoteUrl));
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv, false);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Set UCX environment variable to reuse addresses
  setenv("UCX_TCP_CM_REUSEADDR", "y", 1);

  // Force CUDA context creation
  cudaFree(0);

  std::cout << "========================================" << std::endl;
  std::cout << "Wide Table Exchange Client" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Server: " << FLAGS_server_host << ":" << FLAGS_server_port << std::endl;
  std::cout << "  Task ID: " << FLAGS_task_id << std::endl;
  std::cout << "  Partition ID: " << FLAGS_partition_id << std::endl;
  std::cout << "  Sink drivers: " << FLAGS_num_sink_drivers << std::endl;
  std::cout << "  Client port: " << FLAGS_client_port << std::endl;
  std::cout << "  Complex types: " << (FLAGS_complex_types ? "yes (STRING + STRUCT)" : "no (numeric only)") << std::endl;
  std::cout << "========================================" << std::endl;

  // Setup RMM memory resource with a pool
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
      &cuda_mr, rmm::percent_of_free_device_memory(FLAGS_rmm_pool_percent)};
  rmm::mr::set_current_device_resource(&mr);

  // Create a stream pool for better concurrency
  auto stream_pool = std::make_unique<rmm::cuda_stream_pool>(16);

  // Initialize Velox memory manager
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

  // Initialize local Communicator (needed for UCX endpoint management)
  std::cout << "Initializing local Communicator on port " << FLAGS_client_port << "..." << std::endl;
  ContinueFuture readyFuture;
  auto communicator = Communicator::initAndGet(
      FLAGS_client_port, "http://localhost:12345/unused", &readyFuture);

  if (!communicator) {
    std::cerr << "ERROR: Failed to initialize Communicator" << std::endl;
    return 1;
  }

  // Start communicator in background thread
  std::thread communicatorThread([&]() {
    communicator->run();
  });

  // Wait for communicator to be ready
  readyFuture.wait();
  std::cout << "Local Communicator is ready." << std::endl;

  // Create sink task for this partition
  // Use the appropriate row type based on --complex_types flag
  auto rowType = FLAGS_complex_types
      ? WideComplexTestTable::kRowType
      : WideTestTable::kRowType;
  std::string sinkTaskId = "clientSinkTask_p" + std::to_string(FLAGS_partition_id);
  core::PlanNodeId exchangeNodeId;
  auto sinkTask = createExchangeTask(
      sinkTaskId, rowType, FLAGS_partition_id, exchangeNodeId);

  // Create SinkDriverMock with multiple drivers for parallel receiving
  // Pass nullptr for reference data (no data validation in client mode)
  auto sinkDriver = std::make_shared<SinkDriverMock>(
      sinkTask, FLAGS_num_sink_drivers, nullptr);

  // Add remote split pointing to server's partition
  std::vector<exec::Split> splits;
  splits.push_back(createRemoteSplit(
      FLAGS_server_host, FLAGS_server_port, FLAGS_task_id, FLAGS_partition_id));
  sinkDriver->addSplits(splits);

  std::cout << "Connecting to server and fetching partition " << FLAGS_partition_id << "..." << std::endl;

  // ========================================
  // NVTX marker: Start of exchange region
  // This marks the region that can be profiled with Nsight Systems
  // ========================================
  nvtxRangePushA("WideTableExchange_Client");

  auto startTime = std::chrono::high_resolution_clock::now();

  // Run sink drivers to receive data
  sinkDriver->run();
  sinkDriver->joinThreads();

  auto endTime = std::chrono::high_resolution_clock::now();

  // ========================================
  // NVTX marker: End of exchange region
  // ========================================
  nvtxRangePop();

  // Calculate statistics
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  uint64_t numRows = sinkDriver->numRows();
  uint64_t numBytes = sinkDriver->numBytes();

  double durationSec = duration.count() / 1000.0;
  double throughputMBps = 0;
  double rowsPerSec = 0;
  if (durationSec > 0) {
    throughputMBps = (double)numBytes / (1024.0 * 1024.0) / durationSec;
    rowsPerSec = (double)numRows / durationSec;
  }

  // Report results
  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Transfer Complete!" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Results:" << std::endl;
  std::cout << "  Rows received: " << numRows << std::endl;
  std::cout << "  Bytes received: " << numBytes << " ("
            << (numBytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
  std::cout << "  Duration: " << duration.count() << " ms" << std::endl;
  std::cout << "  Throughput: " << throughputMBps << " MB/s" << std::endl;
  std::cout << "  Row rate: " << rowsPerSec << " rows/s" << std::endl;
  std::cout << "========================================" << std::endl;

  // Cleanup - stop communicator
  communicator->stop();
  communicatorThread.join();

  std::cout << "Client shutdown complete." << std::endl;

  return 0;
}
