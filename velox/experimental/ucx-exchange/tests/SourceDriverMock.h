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
#pragma once

#include "velox/exec/Driver.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestData.h"

namespace facebook::velox::ucx_exchange {

/// @class SourceDriverMock
/// @brief A mock class that drives real UcxPartitionedOutput operators to send
/// data to downstream tasks via the UcxOutputQueueManager. This class properly
/// instantiates and drives UcxPartitionedOutput operators, unlike
/// UcxPartitionedOutputMock which bypasses the operator entirely.
class SourceDriverMock {
 public:
  /// @brief Creates a mock driver to execute a pipeline with
  /// UcxPartitionedOutput operators.
  /// @param task A pointer to the source task with a PartitionedOutput plan
  /// node.
  /// @param numDrivers The number of drivers (parallel operator instances).
  /// @param numChunks The number of data chunks each driver will send.
  /// @param numRowsPerChunk The number of rows per chunk.
  /// @param tableGenerator Optional table generator for test data (for
  /// integrity testing).
  SourceDriverMock(
      std::shared_ptr<facebook::velox::exec::Task> task,
      uint32_t numDrivers,
      uint32_t numChunks,
      size_t numRowsPerChunk,
      std::shared_ptr<BaseTableGenerator> tableGenerator = nullptr);

  /// @brief Starts the driver threads to send data through
  /// UcxPartitionedOutput.
  void run();

  /// @brief Waits until all threads have terminated.
  void joinThreads();

 private:
  /// @brief Driver loop that sends all data chunks through the operator.
  /// Follows the standard Velox operator driver pattern:
  /// 1. Create CudfVector with test data
  /// 2. Check isBlocked() - if blocked, wait on future
  /// 3. When not blocked, call addInput(cudfVector)
  /// 4. Call getOutput() to advance operator state
  /// @param partitionedOutput The UcxPartitionedOutput operator to drive.
  void sendAllData(UcxPartitionedOutput* partitionedOutput);

  std::shared_ptr<facebook::velox::exec::Task> task_;
  std::vector<std::shared_ptr<exec::DriverCtx>> driverCtxs_;
  std::vector<std::unique_ptr<UcxPartitionedOutput>> partitionedOutputs_;

  const uint32_t numDrivers_;
  const uint32_t numChunks_;
  const size_t numRowsPerChunk_;
  const std::shared_ptr<BaseTableGenerator> tableGenerator_;

  std::vector<std::thread> threads_;
};

} // namespace facebook::velox::ucx_exchange
