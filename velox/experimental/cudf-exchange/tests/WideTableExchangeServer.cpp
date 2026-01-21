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

/// @file WideTableExchangeServer.cpp
/// @brief Standalone server that creates a wide table with random values,
/// partitions it into multiple partitions, and serves them to downstream
/// clients via the cudf-exchange protocol.
///
/// Usage:
///   export UCX_TCP_CM_REUSEADDR=y
///   export CUDA_VISIBLE_DEVICES=7
///   ./wide_table_exchange_server --port=21346 --num_rows=1000000 \
///       --num_partitions=2 --num_chunks=100
///
/// Profiling with Nsight Systems:
///   nsys profile --trace=cuda,nvtx ./wide_table_exchange_server ...

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <nvtx3/nvToolsExt.h>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include "velox/common/memory/MemoryPool.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/CudfOutputQueueManager.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestData.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestHelpers.h"
#include "velox/experimental/cudf-exchange/tests/SourceDriverMock.h"

using namespace facebook::velox::cudf_exchange;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

DEFINE_uint32(port, 21346, "Server port for UCX communication");
DEFINE_uint32(num_rows, 1000000, "Total number of rows in the wide table");
DEFINE_uint32(num_partitions, 2, "Number of partitions to split the data into");
DEFINE_uint32(
    num_chunks,
    100,
    "Number of chunks to send (rows_per_chunk = num_rows / num_chunks)");
DEFINE_uint32(
    num_src_drivers,
    1,
    "Number of source drivers (parallel senders)");
DEFINE_string(
    task_id,
    "wideTableTask0",
    "Task ID that clients will use to fetch data");
DEFINE_uint32(
    rmm_pool_percent,
    50,
    "Percentage of free GPU memory for RMM pool");
DEFINE_bool(
    wait_forever,
    false,
    "If true, wait for Ctrl+C instead of auto-terminating when done");
DEFINE_bool(
    complex_types,
    false,
    "If true, use WideComplexTestTable (includes STRING and STRUCT columns)");

// Global communicator pointer for signal handler
static std::shared_ptr<Communicator> g_communicator;
static std::atomic<bool> g_shutdownRequested{false};

void signalHandler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\nReceived signal " << signal << ", shutting down..."
              << std::endl;
    g_shutdownRequested = true;
    if (g_communicator) {
      g_communicator->stop();
    }
  }
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv, false);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Set UCX environment variable to reuse addresses
  setenv("UCX_TCP_CM_REUSEADDR", "y", 1);

  // Force CUDA context creation
  cudaFree(0);

  std::cout << "========================================" << std::endl;
  std::cout << "Wide Table Exchange Server" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Port: " << FLAGS_port << std::endl;
  std::cout << "  Task ID: " << FLAGS_task_id << std::endl;
  std::cout << "  Total rows: " << FLAGS_num_rows << std::endl;
  std::cout << "  Partitions: " << FLAGS_num_partitions << std::endl;
  std::cout << "  Chunks: " << FLAGS_num_chunks << std::endl;
  std::cout << "  Source drivers: " << FLAGS_num_src_drivers << std::endl;
  std::cout << "  Rows per chunk: " << (FLAGS_num_rows / FLAGS_num_chunks)
            << std::endl;
  std::cout << "  Wait forever: " << (FLAGS_wait_forever ? "yes" : "no")
            << std::endl;
  std::cout << "  Complex types: "
            << (FLAGS_complex_types ? "yes (STRING + STRUCT)"
                                    : "no (numeric only)")
            << std::endl;
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
  auto pool = memory::memoryManager()->addLeafPool("WideTableServerPool");

  // Get the CudfOutputQueueManager singleton
  auto queueManager = CudfOutputQueueManager::getInstanceRef();

  // Initialize the Communicator
  std::cout << "Initializing Communicator on port " << FLAGS_port << "..."
            << std::endl;
  ContinueFuture readyFuture;
  g_communicator = Communicator::initAndGet(
      FLAGS_port, "http://localhost:12345/unused", &readyFuture);

  if (!g_communicator) {
    std::cerr << "ERROR: Failed to initialize Communicator" << std::endl;
    return 1;
  }

  // Start communicator in a background thread
  std::thread communicatorThread([&]() { g_communicator->run(); });

  // Wait for communicator to be ready
  readyFuture.wait();
  std::cout << "Communicator is ready and listening." << std::endl;

  // Install signal handlers for graceful shutdown
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // Create table generator with random values based on --complex_types flag
  // Initialize with rows_per_chunk since SourceDriverMock creates this many
  // rows per chunk
  size_t rowsPerChunk = FLAGS_num_rows / FLAGS_num_chunks;
  std::shared_ptr<BaseTableGenerator> tableGenerator;
  if (FLAGS_complex_types) {
    tableGenerator = std::make_shared<WideComplexTestTable>();
    std::cout
        << "Using WideComplexTestTable (includes STRING and STRUCT columns)"
        << std::endl;
  } else {
    tableGenerator = std::make_shared<WideTestTable>();
    std::cout << "Using WideTestTable (numeric columns only)" << std::endl;
  }
  tableGenerator->initialize(rowsPerChunk);

  std::cout << "Created table with " << rowsPerChunk << " rows per chunk"
            << std::endl;
  std::cout << "Table schema: " << tableGenerator->getRowType()->toString()
            << std::endl;

  // Create source task with PartitionedOutput plan node
  // Use int32_col (column index 2) for hash partitioning
  std::vector<std::string> partitionKeys;
  if (FLAGS_num_partitions > 1) {
    partitionKeys = {"int32_col"}; // Column index 2 in WideTestTable
    std::cout << "Using hash partitioning on column: int32_col" << std::endl;
  } else {
    std::cout << "Single partition mode (no hash partitioning)" << std::endl;
  }

  auto srcTask = createPartitionedOutputTask(
      FLAGS_task_id,
      pool,
      tableGenerator->getRowType(),
      FLAGS_num_partitions,
      partitionKeys);

  // Register task with the queue manager
  queueManager->initializeTask(
      srcTask, FLAGS_num_partitions, FLAGS_num_src_drivers);
  std::cout << "Registered task '" << FLAGS_task_id << "' with queue manager"
            << std::endl;

  // Create and run SourceDriverMock to drive the CudfPartitionedOutput
  // operators
  auto sourceDriver = std::make_shared<SourceDriverMock>(
      srcTask,
      FLAGS_num_src_drivers,
      FLAGS_num_chunks,
      rowsPerChunk,
      tableGenerator);

  std::cout << "Starting data production..." << std::endl;

  // ========================================
  // NVTX marker: Start of exchange region
  // This marks the region that can be profiled with Nsight Systems
  // ========================================
  nvtxRangePushA("WideTableExchange_Server");

  // Start the source driver (produces data in background threads)
  sourceDriver->run();

  // Wait for all data to be produced
  sourceDriver->joinThreads();

  std::cout << "========================================" << std::endl;
  std::cout << "Data production complete!" << std::endl;
  std::cout << "Total rows produced: "
            << (FLAGS_num_chunks * rowsPerChunk * FLAGS_num_src_drivers)
            << std::endl;
  std::cout << "Waiting for clients to fetch data..." << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;
  std::cout << "Client connection URL format:" << std::endl;
  for (uint32_t p = 0; p < FLAGS_num_partitions; ++p) {
    std::cout << "  Partition " << p << ": http://<host>:" << (FLAGS_port - 3)
              << "/v1/task/" << FLAGS_task_id << "/results/" << p << std::endl;
  }
  std::cout << std::endl;

  // Monitor for completion: poll until all data has been consumed by clients
  // The queueManager->isFinished() returns true when all partition queues have
  // been deleted (which happens after each partition's data is fully
  // transferred)
  if (!FLAGS_wait_forever) {
    std::cout
        << "Monitoring for completion (use --wait_forever to disable auto-shutdown)..."
        << std::endl;
    while (!g_shutdownRequested) {
      if (queueManager->isFinished(FLAGS_task_id)) {
        std::cout << "All data has been consumed by clients. Shutting down..."
                  << std::endl;
        g_communicator->stop();
        break;
      }
      // Poll every 100ms
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  } else {
    std::cout << "Press Ctrl+C to shutdown" << std::endl;
  }

  // Wait for communicator to stop
  communicatorThread.join();

  // ========================================
  // NVTX marker: End of exchange region
  // ========================================
  nvtxRangePop();

  // Cleanup
  queueManager->removeTask(FLAGS_task_id);
  std::cout << "Server shutdown complete." << std::endl;

  return 0;
}
