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
#include "velox/experimental/ucx-exchange/tests/SourceDriverMock.h"
#include <cudf/utilities/default_stream.hpp>
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

namespace facebook::velox::ucx_exchange {

constexpr int kPipelineId = 0;
constexpr uint32_t kPartitionId = 0;

namespace {

void driveUntilNeedsInput(UcxPartitionedOutput* partitionedOutput) {
  while (!partitionedOutput->needsInput() && !partitionedOutput->isFinished()) {
    ContinueFuture future;
    const auto blocked = partitionedOutput->isBlocked(&future);
    if (blocked != exec::BlockingReason::kNotBlocked) {
      future.wait();
      continue;
    }
    // needsInput() can remain false with no queue future when the operator has
    // an internal input suffix or partition batch to drain. Real Driver calls
    // getOutput() in this state; the mock must do the same before offering the
    // next input.
    partitionedOutput->getOutput();
  }
}

} // namespace

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
      partitionedOutputNode, "Plan node must be a PartitionedOutputNode");

  uint32_t operatorId = 0;

  // Create the set of UcxPartitionedOutput operators, one per driver.
  for (uint32_t driverId = 0; driverId < numDrivers; ++driverId) {
    driverCtxs_.emplace_back(
        std::make_shared<exec::DriverCtx>(
            task_,
            driverId,
            kPipelineId,
            exec::kUngroupedGroupId,
            kPartitionId));

    // The operator normally gets its manager from the OutputTransportEntry
    // registered under core::TransportKind::kUcx; these mocks drive the
    // process-wide manager directly.
    partitionedOutputs_.emplace_back(
        std::make_unique<UcxPartitionedOutput>(
            operatorId,
            driverCtxs_.back().get(),
            partitionedOutputNode,
            /*eagerFlush=*/false,
            UcxOutputQueueManager::getInstanceRef()));
  }
}

void SourceDriverMock::run() {
  threads_.clear();
  for (uint32_t driver = 0; driver < numDrivers_; ++driver) {
    threads_.emplace_back(
        &SourceDriverMock::sendAllData,
        this,
        partitionedOutputs_[driver].get());
  }
}

void SourceDriverMock::sendAllData(UcxPartitionedOutput* partitionedOutput) {
  // Empty input allocates no device buffers, so it cannot rely on an
  // allocation to initialize this thread's CUDA context before stream.sync().
  CUDF_CUDA_TRY(cudaSetDevice(0));
  auto stream = cudf::get_default_stream();
  // Use tableGenerator's rowType if available, otherwise get from task
  auto rowType = tableGenerator_ ? tableGenerator_->getRowType()
                                 : task_->planFragment().planNode->outputType();

  // Get memory pool from the operator
  auto* pool = partitionedOutput->pool();

  for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
    // Match Driver's input contract: an unblocked operator may still be
    // draining internal state and therefore not need input yet.
    driveUntilNeedsInput(partitionedOutput);
    VELOX_CHECK(
        partitionedOutput->needsInput(),
        "Source operator finished before all mock inputs were consumed");

    // Create CudfVector only after the operator is ready, so the mock does not
    // retain an extra device batch while waiting for output backpressure.
    auto cudfVector = makeCudfVector(
        pool, numRowsPerChunk_, rowType, tableGenerator_, stream);

    partitionedOutput->addInput(cudfVector);

    VLOG(3) << "SourceDriverMock: sent chunk " << chunk << " of " << numChunks_;
  }

  // Driver signals end-of-input only after the sink asks for another batch.
  // This also exercises multi-chunk inputs whose previous addInput initiated
  // more than one internal output chunk.
  driveUntilNeedsInput(partitionedOutput);
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

} // namespace facebook::velox::ucx_exchange
