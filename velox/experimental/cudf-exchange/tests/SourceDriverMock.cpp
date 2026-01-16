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
#include "velox/experimental/cudf-exchange/tests/SourceDriverMock.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestHelpers.h"

namespace facebook::velox::cudf_exchange {

constexpr int kPipelineId = 0;
constexpr uint32_t kPartitionId = 0;

SourceDriverMock::SourceDriverMock(
    std::shared_ptr<facebook::velox::exec::Task> task,
    uint32_t numDrivers,
    uint32_t numChunks,
    size_t numRowsPerChunk,
    std::shared_ptr<BaseTableGenerator> tableGenerator)
    : task_{std::move(task)},
      numDrivers_{numDrivers},
      numChunks_{numChunks},
      numRowsPerChunk_{numRowsPerChunk},
      tableGenerator_(tableGenerator) {
  // Get the plan node - should be a PartitionedOutputNode
  auto planNode = task_->planFragment().planNode;
  auto partitionedOutputNode =
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(planNode);
  VELOX_CHECK_NOT_NULL(
      partitionedOutputNode,
      "Plan node must be a PartitionedOutputNode");

  uint32_t operatorId = 0;

  // Create the set of CudfPartitionedOutput operators, one per driver.
  for (uint32_t driverId = 0; driverId < numDrivers; ++driverId) {
    driverCtxs_.emplace_back(std::make_shared<exec::DriverCtx>(
        task_, driverId, kPipelineId, exec::kUngroupedGroupId, kPartitionId));

    partitionedOutputs_.emplace_back(std::make_unique<CudfPartitionedOutput>(
        operatorId,
        driverCtxs_.back().get(),
        partitionedOutputNode,
        false /* eagerFlush */));
  }
}

void SourceDriverMock::run() {
  threads_.clear();
  for (uint32_t driver = 0; driver < numDrivers_; ++driver) {
    threads_.emplace_back(
        &SourceDriverMock::sendAllData, this, partitionedOutputs_[driver].get());
  }
}

void SourceDriverMock::sendAllData(CudfPartitionedOutput* partitionedOutput) {
  auto stream = rmm::cuda_stream_default;
  // Use tableGenerator's rowType if available, otherwise get from task
  auto rowType = tableGenerator_ ? tableGenerator_->getRowType()
                                 : task_->planFragment().planNode->outputType();

  // Get memory pool from the operator
  auto* pool = partitionedOutput->pool();

  for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
    // 1. Create CudfVector with test data using makeCudfVector helper
    auto cudfVector = makeCudfVector(
        pool, numRowsPerChunk_, rowType, tableGenerator_, stream);

    // 2. Check isBlocked() - if blocked, wait on future
    while (true) {
      ContinueFuture future;
      auto blocked = partitionedOutput->isBlocked(&future);
      if (blocked == exec::BlockingReason::kNotBlocked) {
        break;
      }
      // Wait for the operator to become unblocked
      future.wait();
    }

    // 3. Call addInput() now that we're not blocked
    partitionedOutput->addInput(cudfVector);

    // 4. Call getOutput() to advance operator state
    partitionedOutput->getOutput();

    VLOG(3) << "SourceDriverMock: sent chunk " << chunk << " of " << numChunks_;
  }

  // Signal no more input and finalize
  partitionedOutput->noMoreInput();

  // Continue calling getOutput() until finished
  while (!partitionedOutput->isFinished()) {
    ContinueFuture future;
    auto blocked = partitionedOutput->isBlocked(&future);
    if (blocked != exec::BlockingReason::kNotBlocked) {
      future.wait();
    }
    partitionedOutput->getOutput();
  }

  VLOG(3) << "SourceDriverMock: driver finished";
}

void SourceDriverMock::joinThreads() {
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

} // namespace facebook::velox::cudf_exchange
