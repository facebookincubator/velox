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

#include "velox/experimental/cudf-exchange/CudfOutputQueueManager.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestData.h"

namespace facebook::velox::cudf_exchange {

/// @brief Mocks one or more cudf partitioned output operators. Each
/// operator pushes data into a set of partitions.
/// The operators run in separate threads, one thread per driver.
class CudfPartitionedOutputMock {
 public:
  CudfPartitionedOutputMock(
      const std::string& taskId,
      const uint32_t numDrivers,
      const size_t numPartitions,
      const uint32_t numDataChunks,
      const size_t numRowsPerChunk,
      std::shared_ptr<BaseTableGenerator> tableGenerator = nullptr);

  /// @brief Created the driver threads and pushes data into the task's output
  /// queue.
  void run();

  /// @brief Wait for all threads to complete.
  void joinThreads();

 private:
  // creates numPartitions_ x numDataChunks_ data chunks and
  // pushes it into the destination queues identified by the taskId.
  void publishDataChunks();

  const std::shared_ptr<CudfOutputQueueManager> queueManager_;
  const std::string taskId_;
  const uint32_t numDrivers_;
  const size_t numPartitions_;
  const uint32_t numDataChunks_;
  const size_t numRowsPerChunk_;
  const std::shared_ptr<BaseTableGenerator> tableGenerator_;
  std::vector<std::thread> threads_;
};

} // namespace facebook::velox::cudf_exchange
