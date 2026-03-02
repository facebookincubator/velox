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
#include <folly/init/Init.h>
#include <csignal>
#include "cuda_runtime.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf-exchange/Communicator.h"
#include "velox/experimental/cudf-exchange/CudfExchange.h"
#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <gflags/gflags.h>

#include "TableWithNames.hpp"

using namespace facebook::velox::cudf_exchange;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::core;

DEFINE_uint32(port, 24356, "Port number to connect to");
DEFINE_uint32(srv_port, 13131, "Dummy listening port that is not used here");
DEFINE_string(hostname, "127.0.0.1", "Host name");
DEFINE_string(taskId, "task0", "task id");
DEFINE_uint32(destination, 0, "destination");

std::string kDummyCoordinatorUrl{"localhost:1/nowhere"};

// Client Side Part of a Simple (Velox Independent) Cudf Exchange
class CudfExchangeClientTest {
 public:
  CudfExchangeClientTest(uint32_t destination)
      : destination_(destination),
        executor_(
            std::make_shared<folly::CPUThreadPoolExecutor>(
                std::thread::hardware_concurrency())) {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    pool_ = facebook::velox::memory::memoryManager()->addLeafPool();
    // rowType_ needed for plan fragment which is needed for the task.
    std::vector<std::string> names = {"c0", "c1"};
    std::vector<TypePtr> types = {INTEGER(), DOUBLE()};
    rowType_ = ROW(std::move(names), std::move(types));
  }

  std::shared_ptr<Task> initializeTask(const std::string& taskId) {
    auto planFragment =
        exec::test::PlanBuilder()
            .exchange(rowType_, VectorSerde::Kind::kPresto /*unused*/)
            .capturePlanNodeId(exchangeNodeId_)
            .planFragment();

    // exchange node is the source node in the little plan fragment.
    exchangeNode_ =
        std::dynamic_pointer_cast<const ExchangeNode>(planFragment.planNode);

    std::unordered_map<std::string, std::string> configSettings;
    auto queryCtx = core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(configSettings)));

    task_ = Task::create(
        taskId,
        std::move(planFragment),
        destination_,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel);

    // create the exchange client.
    client_ = std::make_shared<CudfExchangeClient>(
        taskId, destination_, 1 /* number of consumers */);

    // Create (fake) DriverCtx
    driverCtx_ = std::make_unique<DriverCtx>(
        task_,
        0, /* driverId. Must be 0, otherwise CudfExchange won't process any
              splits. */
        0, /* pipelineId */
        kUngroupedGroupId, /* splitGroupId */
        destination_);

    // create an cudf exchange operator
    exchange_ = std::make_shared<CudfExchange>(
        0, /* operatorId*/
        driverCtx_.get(),
        exchangeNode_,
        client_);

    return task_;
  }

  void addRemoteSplit(const std::string& taskId) {
    auto split = exec::Split(
        std::make_shared<facebook::velox::exec::RemoteConnectorSplit>(taskId));
    task_->addSplit(exchangeNodeId_, std::move(split));
    task_->noMoreSplits(exchangeNodeId_);
  }

  void getAllChunks() {
    // Iterate on the cudf exchange operator to get all chunks.
    bool atEnd = false;
    uint32_t numChunks = 0;
    while (!atEnd) {
      ContinueFuture future = ContinueFuture::makeEmpty();
      auto blocked = exchange_->isBlocked(&future);
      if (blocked != BlockingReason::kNotBlocked) {
        VLOG(3) << "Waiting on future";
        VELOX_CHECK(future.valid());
        std::move(future).via(executor_.get()).wait();
        VLOG(3) << "Future is ready, there should be data in the queue.";
      } else {
        VLOG(3) << "CudfExchange is not blocked";
        if (exchange_->isFinished()) {
          VLOG(3) << "CudfExchange is finished";
          atEnd = true;
        } else {
          auto result = std::dynamic_pointer_cast<
              facebook::velox::cudf_velox::CudfVector>(exchange_->getOutput());
          VELOX_CHECK_NOT_NULL(result);
          auto tblView = result->getTableView();
          VLOG(3) << "Got chunk #" << numChunks++ << " with "
                  << tblView.num_rows() << " rows and " << tblView.num_columns()
                  << " columns";
          std::vector<std::string> names(tblView.num_columns());
          for (int i = 0; i < tblView.num_columns(); i++) {
            names[i] = "col_" + std::to_string(i);
          }
          table_with_names t(result->release(), names);
          t.dump(10, result->stream());
        }
      }
      // simulate some pipeline work.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  core::PlanNodeId exchangeNodeId_;
  uint32_t destination_;
  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<facebook::velox::memory::MemoryPool> pool_;
  RowTypePtr rowType_;
  std::shared_ptr<Task> task_;
  std::shared_ptr<CudfExchangeClient> client_;
  std::shared_ptr<CudfExchange> exchange_;
  std::shared_ptr<const ExchangeNode> exchangeNode_;
  std::unique_ptr<DriverCtx> driverCtx_;
};

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv, false);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  setenv("UCX_TCP_CM_REUSEADDR", "y", 1);

  auto timeOutMs = std::chrono::microseconds(1000); // not used.

  // Force CUDA context creation
  cudaFree(0);

  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
      &cuda_mr, rmm::percent_of_free_device_memory(25)};
  rmm::mr::set_current_device_resource(&mr);

  // initialize the communicator with some listener port (that won't be used).
  auto communicator =
      Communicator::initAndGet(FLAGS_srv_port, kDummyCoordinatorUrl);
  // start the communicator in a thread
  std::thread commThread(
      &Communicator::run, communicator.get()); // Create and start the thread
  commThread.detach();

  CudfExchangeClientTest test(FLAGS_destination);
  std::string taskId = "clientTask0";
  auto task = test.initializeTask(taskId);

  std::string remoteUrl = "http://" + FLAGS_hostname + ":" +
      std::to_string(FLAGS_port - 3) + "/v1/task/" + FLAGS_taskId +
      "/results/" + std::to_string(FLAGS_destination);

  // add one remote split.
  test.addRemoteSplit(remoteUrl);

  test.getAllChunks();
}
