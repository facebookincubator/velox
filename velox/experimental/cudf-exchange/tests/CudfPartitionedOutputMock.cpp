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
#include "velox/experimental/cudf-exchange/tests/CudfPartitionedOutputMock.h"
#include "velox/experimental/cudf-exchange/CudfExchangeProtocol.h"
#include "velox/experimental/cudf-exchange/CudfQueues.h"
#include "velox/experimental/cudf-exchange/tests/CudfTestHelpers.h"

namespace facebook::velox::cudf_exchange {

CudfPartitionedOutputMock::CudfPartitionedOutputMock(
    const std::string& taskId,
    const uint32_t numDrivers,
    const size_t numPartitions,
    const uint32_t numDataChunks,
    const size_t numRowsPerChunk,
    std::shared_ptr<BaseTableGenerator> tableGenerator)
    : queueManager_(CudfOutputQueueManager::getInstanceRef()),
      taskId_(taskId),
      numDrivers_(numDrivers),
      numPartitions_(numPartitions),
      numDataChunks_(numDataChunks),
      numRowsPerChunk_(numRowsPerChunk),
      tableGenerator_(tableGenerator) {}

void CudfPartitionedOutputMock::run() {
  threads_.clear();
  for (int32_t driver = 0; driver < numDrivers_; ++driver) {
    threads_.emplace_back(&CudfPartitionedOutputMock::publishDataChunks, this);
  }
}

void CudfPartitionedOutputMock::joinThreads() {
  for (auto& thread : threads_) {
    thread.join();
  }
  threads_.clear();
}

void CudfPartitionedOutputMock::publishDataChunks() {
  // create numPartitions_ x numDataChunks_ data chunks and
  // push it into the destination queues identified by the taskId.
  auto stream = rmm::cuda_stream_default;
  for (size_t partition = 0; partition < numPartitions_; ++partition) {
    for (uint32_t dataChunk = 0; dataChunk < numDataChunks_; ++dataChunk) {
      VLOG(3)
          << "In CudfPartitionedOutputMock::publishDataChunks writing to partition "
          << partition << " chunk " << dataChunk;
      // create a data chunk with numRowsPerChunk_ rows
      // and push it into the destination queue identified by the taskId.

      std::unique_ptr<cudf::table> table;
      if (tableGenerator_ == nullptr) {
        table = makeTable(numRowsPerChunk_, CudfTestData::kTestRowType, stream);
      } else {
        table = tableGenerator_->makeTable(stream);
      }
      auto packedCols = std::make_unique<cudf::packed_columns>(cudf::pack(table->view(), stream));
      // Sync the stream since UCXX/UCX is not stream oriented and without
      // syncing, data could get lost. Syncing here is  easy but not the most
      // efficient. A better approach is to create an event and pass it along
      // the data through the queue and synchronize on the event before calling
      // into UCXX.
      stream.synchronize();
      queueManager_->enqueue(
          taskId_,
          partition,
          std::move(packedCols),
          numRowsPerChunk_);
    }
  }
  // tell the queue manager that we are done.
  queueManager_->noMoreData(taskId_);
}

} // namespace facebook::velox::cudf_exchange
