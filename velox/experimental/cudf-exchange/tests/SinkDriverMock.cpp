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
#include "velox/experimental/cudf-exchange/tests/SinkDriverMock.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_exchange {

constexpr int kPipelineId = 0;
constexpr uint32_t kPartitionId = 0; // not used in tests.

SinkDriverMock::SinkDriverMock(
    std::shared_ptr<facebook::velox::exec::Task> task,
    uint32_t numDrivers,
    std::shared_ptr<BaseTableGenerator> referenceData)
    : task_{std::move(task)},
      numDrivers_{numDrivers},
      numRows_{0},
      numBytes_{0},
      referenceData_(referenceData) {
  // create a new exchange client facade. Since this test doesn't use
  // HTTP exchange, the facade will only use a cudf exchange client.
  // create new cudfExchangeClient
  auto cudfClient = std::make_shared<CudfExchangeClient>(
      task_->taskId(), task_->destination(), numDrivers_);
  exchangeClient_ = std::make_shared<ExchangeClientFacade>(
      task_->taskId(),
      kPipelineId,
      std::move(cudfClient),
      nullptr); // no HTTP client.
  uint32_t operatorId = 0;
  auto planNode = task_->planFragment().planNode;
  // create the set of exchange operators.
  for (uint32_t driverId = 0; driverId < numDrivers; ++driverId) {
    driverCtxs_.emplace_back(
        std::make_shared<DriverCtx>(
            task_, driverId, kPipelineId, kUngroupedGroupId, kPartitionId));
    hybridExchanges_.emplace_back(
        std::make_unique<HybridExchange>(
            operatorId, driverCtxs_.back().get(), planNode, exchangeClient_));
  }
}

void SinkDriverMock::updateDataValidity(const cudf::table_view& tab) {
  if (!referenceData_) {
    return; // No reference data to check against
  }

  auto stream = rmm::cuda_stream_default;

  // Use the polymorphic verifyTable method
  // Note: For chunk-based verification, we use startRow=0 since each chunk
  // contains rows that should match the reference data from the beginning.
  // This assumes the test sends the same data pattern in each chunk.
  bool valid = referenceData_->verifyTable(tab, 0, tab.num_rows(), stream);

  if (!valid) {
    dataValidFlag_ = false;
  }
}

void SinkDriverMock::run() {
  threads_.clear();
  for (int32_t driver = 0; driver < numDrivers_; ++driver) {
    threads_.emplace_back(
        &SinkDriverMock::receiveAllData, this, hybridExchanges_[driver].get());
  }
}

void SinkDriverMock::receiveAllData(HybridExchange* hybridExchange) {
  while (true) {
    ContinueFuture future;
    auto blocked = hybridExchange->isBlocked(&future);
    if (blocked != BlockingReason::kNotBlocked) {
      future.wait();
    } else {
      // not blocked.
      RowVectorPtr res = hybridExchange->getOutput();
      if (res) {
        facebook::velox::cudf_velox::CudfVectorPtr cudfRes =
            std::dynamic_pointer_cast<facebook::velox::cudf_velox::CudfVector>(
                res);
        numBytes_.fetch_add(cudfRes->estimateFlatSize());
        numRows_ += cudfRes->getTableView().num_rows();
        // If we have Reference data check the received data is the same
        if (referenceData_)
          updateDataValidity(cudfRes->getTableView());
      }
    }
    if (hybridExchange->isFinished()) {
      break;
    }
  }
  hybridExchange->close();
}

void SinkDriverMock::joinThreads() {
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

void SinkDriverMock::addSplits(
    std::vector<facebook::velox::exec::Split>& splits) {
  auto planNode = task_->planFragment().planNode;
  for (auto& split : splits) {
    VLOG(3) << "Adding split to planNode: " << planNode->id()
            << " to sink driver for task " << task_->taskId();
    task_->addSplit(planNode->id(), std::move(split));
  }
  task_->noMoreSplits(planNode->id());
}

} // namespace facebook::velox::cudf_exchange
