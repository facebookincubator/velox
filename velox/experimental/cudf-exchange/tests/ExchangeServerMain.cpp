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

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <csignal>
#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/CudfExchangeProtocol.h"
#include "velox/experimental/cudf-exchange/CudfExchangeServer.h"
#include "velox/experimental/cudf-exchange/CudfOutputQueueManager.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

using namespace facebook::velox::cudf_exchange;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

// FIXME: Move this into VeloxConfig
DEFINE_uint32(port, 24356, "Port number");
DEFINE_uint32(num_dests, 1, "Number of destinations (partitions)");
DEFINE_uint32(num_chunks, 5, "Number of chunks");
DEFINE_uint32(rows, 1'000'000'000, "Number of rows");

std::string kDummyCoordinatorUrl{"localhost:1/nowhere"};

class CudfExchangeServerTest {
 public:
  CudfExchangeServerTest()
      : executor_(
            std::make_shared<folly::CPUThreadPoolExecutor>(
                std::thread::hardware_concurrency())) {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    server_ = Communicator::initAndGet(FLAGS_port, kDummyCoordinatorUrl);
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    queueManager_ = CudfOutputQueueManager::getInstanceRef();
    // rowType_ needed for plan fragment which is needed for the task.
    std::vector<std::string> names = {"c0", "c1"};
    std::vector<TypePtr> types = {INTEGER(), DOUBLE()};
    rowType_ = ROW(std::move(names), std::move(types));
  }

  std::shared_ptr<Task> initializeTask(
      const std::string& taskId,
      int numDestinations,
      int numDrivers) {
    queueManager_->removeTask(taskId);

    const size_t vectorSize = 10;
    auto intVec = BaseVector::create(INTEGER(), vectorSize, pool_.get());
    auto dblVec = BaseVector::create(DOUBLE(), vectorSize, pool_.get());

    // Wrap the vector (column) in a RowVector.
    auto rowVector = std::make_shared<RowVector>(
        pool_.get(), // pool where allocations will be made.
        rowType_, // input row type (defined above).
        BufferPtr(nullptr), // no nulls on this example.
        vectorSize, // length of the vectors.
        std::vector<VectorPtr>{intVec, dblVec}); // the input vector data.

    auto planFragment =
        exec::test::PlanBuilder().values({rowVector}).planFragment();

    std::unordered_map<std::string, std::string> configSettings;
    auto queryCtx = core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(configSettings)));

    auto task = Task::create(
        taskId,
        std::move(planFragment),
        0,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel);

    queueManager_->initializeTask(task, numDestinations, numDrivers);
    return task;
  }

  std::unique_ptr<cudf::packed_columns> makePackedColumns(
      std::size_t numRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    // Create two numeric columns using cudf::make_numeric_column
    auto col1 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        numRows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    auto col2 = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT64},
        numRows,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);

    // Table will contain arbitrary values.
    // fill with some recognizable data.
    // Obtain a mutable view of the column so that we can write into its data
    // buffer.
    std::size_t len = 100;
    auto mutable_view = col1->mutable_view();

    // Cast the underlying data pointer to uint32_t*
    uint32_t* data1 = mutable_view.template data<uint32_t>();

    std::vector<uint32_t> vec1(len);
    std::iota(vec1.begin(), vec1.end(), 1);
    cudaMemcpy(
        data1,
        vec1.data(),
        vec1.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    // now the same for the double column.
    mutable_view = col2->mutable_view();
    double* data2 = mutable_view.template data<double>();

    std::vector<double> vec2(len);
    int i = 0;
    for (double& val : vec2) {
      val = 1.5 + 0.25 * i++;
    }
    cudaMemcpy(
        data2,
        vec2.data(),
        vec2.size() * sizeof(double),
        cudaMemcpyHostToDevice);

    // Build cudf::table
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = std::make_unique<cudf::table>(std::move(columns));

    cudf::packed_columns packed = cudf::pack(table->view());

    return std::unique_ptr<cudf::packed_columns>(new cudf::packed_columns(
        std::move(packed.metadata), std::move(packed.gpu_data)));
  }

  // Returns the enqueued page byte size.
  void enqueue(
      const std::string& taskId,
      int destination,
      vector_size_t size,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto data = makePackedColumns(size, stream, mr);
    queueManager_->enqueue(taskId, destination, std::move(data), size);
  }

  void noMoreData(const std::string& taskId) {
    queueManager_->noMoreData(taskId);
  }

  std::shared_ptr<Communicator> getServerPtr() {
    return server_;
  }

 private:
  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<Communicator> server_;
  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
  std::shared_ptr<CudfOutputQueueManager> queueManager_;
  RowTypePtr rowType_;
};

static std::shared_ptr<Communicator> serverPtr;

void signalHandler(int signal) {
  if (signal == SIGTERM) {
    serverPtr->stop();
  }
}

#define MEM_MGR 1

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv, false);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  setenv("UCX_TCP_CM_REUSEADDR", "y", 1);

  // Force CUDA context creation
  cudaFree(0);

#if MEM_MGR
  // configure a cuda memory manager. This is not strictly necessary for this
  // example.
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
      &cuda_mr, rmm::percent_of_free_device_memory(25)};
  rmm::mr::set_current_device_resource(&mr);
  // Same for the stream pool: Could use the default stream.
  auto stream_pool = std::make_unique<rmm::cuda_stream_pool>(16);
  auto stream = stream_pool->get_stream();
#else
  auto stream = cudf::get_default_stream();
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
#endif

  CudfExchangeServerTest test;
  std::string taskId = "task0";

  auto task = test.initializeTask(taskId, FLAGS_num_dests, 1 /* numDrivers*/);
  uint32_t rowsPerChunk = FLAGS_rows / FLAGS_num_chunks;
  std::cout << "Created task with id: " << taskId << std::endl;
  // enqueue a few packed_columns for all destinations.
  for (uint32_t dest = 0; dest < FLAGS_num_dests; ++dest) {
    for (uint32_t i = 0; i < FLAGS_num_chunks; ++i) {
      test.enqueue(taskId, dest, rowsPerChunk, stream, mr);
    }
  }
  test.noMoreData(taskId);

  // start the server in the current thread.
  serverPtr = test.getServerPtr();
  std::signal(SIGTERM, signalHandler);
  serverPtr->run();
}
