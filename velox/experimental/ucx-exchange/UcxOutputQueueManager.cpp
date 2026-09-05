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
// Assisted by watsonx Code Assistant
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"

namespace facebook::velox::ucx_exchange {

/* static */
std::shared_ptr<UcxOutputQueueManager> UcxOutputQueueManager::getInstanceRef() {
  // In C++11, the static local variable is guaranteed to only be initialized
  // once even in a multi-threaded context.
  static std::shared_ptr<UcxOutputQueueManager> instance =
      std::make_shared<UcxOutputQueueManager>();
  return instance;
}

void UcxOutputQueueManager::initializeTask(
    std::shared_ptr<exec::Task> task,
    core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers) {
  const auto& taskId = task->taskId();
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskId);
    if (it == queues.end()) {
      queues[taskId] = std::make_shared<UcxOutputQueue>(
          std::move(task), numDestinations, numDrivers, kind);
    } else {
      if (!it->second->initialize(task, numDestinations, numDrivers, kind)) {
        VELOX_FAIL(
            "Registering a cudf output queue for pre-existing taskId {}",
            taskId);
      }
    }
  });
  // Clear any stale "removed" state so that getData() calls after this
  // initializeTask() create proper placeholder queues if needed.
  removedTasks_.withLock([&](auto& removed) { removed.erase(taskId); });
  // Clear any stale "cancelled" state in the intra-node registry so
  // that the cancelledTasks_ set doesn't grow unboundedly across queries.
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
}

void UcxOutputQueueManager::updateOutputBuffers(
    std::string_view taskId,
    int numBuffers,
    bool noMoreBuffers) {
  if (auto queue = getQueueIfActive(taskId)) {
    queue->updateOutputBuffers(numBuffers, noMoreBuffers);
  }
}

void UcxOutputQueueManager::enqueue(
    std::string_view taskId,
    int destination,
    std::unique_ptr<cudf::packed_columns> txData,
    int numRows,
    int64_t transferReservationBytes) {
  if (auto queue = getQueueIfActive(taskId)) {
    queue->enqueue(
        destination, std::move(txData), numRows, transferReservationBytes);
  }
}

bool UcxOutputQueueManager::checkBlocked(
    std::string_view taskId,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->checkBlocked(future);
  }
  return false;
}

bool UcxOutputQueueManager::checkTransferCapacity(
    std::string_view taskId,
    int destination,
    int64_t maxBytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->checkTransferCapacity(destination, maxBytes, future);
  }
  return false;
}

bool UcxOutputQueueManager::reserveTransferBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    int64_t maxBytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->reserveTransferBytes(destination, bytes, maxBytes, future);
  }
  return false;
}

bool UcxOutputQueueManager::reserveFullTransferBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->reserveFullTransferBytes(destination, bytes, future);
  }
  return false;
}

bool UcxOutputQueueManager::waitForFullTransferCapacity(
    std::string_view taskId,
    int64_t bytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->waitForFullTransferCapacity(bytes, future);
  }
  return false;
}

void UcxOutputQueueManager::releaseTransferReservation(
    std::string_view taskId,
    int destination,
    int64_t bytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->releaseTransferReservation(destination, bytes);
  }
}

int64_t UcxOutputQueueManager::transferWindowBytes(
    std::string_view taskId,
    int destination,
    int64_t baseBytes,
    int64_t normalBytes,
    int64_t maxBytes) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->transferWindowBytes(
        destination, baseBytes, normalBytes, maxBytes);
  }
  return baseBytes;
}

void UcxOutputQueueManager::recordTransferCongestion(
    std::string_view taskId,
    int destination,
    int64_t baseBytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->recordTransferCongestion(destination, baseBytes);
  }
}

void UcxOutputQueueManager::recordTransferDemand(
    std::string_view taskId,
    int destination,
    int64_t targetBytes,
    int64_t baseBytes,
    int64_t maxBytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->recordTransferDemand(destination, targetBytes, baseBytes, maxBytes);
  }
}

void UcxOutputQueueManager::recordFullTransferCongestion(
    std::string_view taskId) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->recordFullTransferCongestion();
  }
}

UcxDestinationTransferStats UcxOutputQueueManager::transferStats(
    std::string_view taskId,
    int destination) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->transferStats(destination);
  }
  return {};
}

bool UcxOutputQueueManager::reserveOutputBytes(
    std::string_view taskId,
    int64_t bytes,
    ContinueFuture* future) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->reserveOutputBytes(bytes, future);
  }
  return false;
}

void UcxOutputQueueManager::releaseOutputReservation(
    std::string_view taskId,
    int64_t bytes) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->releaseOutputReservation(bytes);
  }
}

void UcxOutputQueueManager::releaseInFlightBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    int64_t numPackedCols) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->releaseInFlightBytes(destination, bytes, numPackedCols);
  }
}

void UcxOutputQueueManager::noMoreData(std::string_view taskId) {
  if (auto queue = getQueueIfActive(taskId)) {
    queue->noMoreData();
  }
}

bool UcxOutputQueueManager::isFinished(std::string_view taskId) {
  if (auto queue = getQueueIfActive(taskId)) {
    return queue->isFinished();
  }
  return true;
}

void UcxOutputQueueManager::deleteResults(
    std::string_view taskId,
    int destination) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->deleteResults(destination);
  }
}

void UcxOutputQueueManager::getData(
    std::string_view taskId,
    int destination,
    UcxDataAvailableCallback notify) {
  std::shared_ptr<UcxOutputQueue> outputQueue;
  bool taskRemoved = false;
  std::string taskIdStr{taskId};
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    if (it == queues.end()) {
      // Check if the task was already removed. If so, don't re-create a
      // placeholder - the task is dead and any server calling getData() is a
      // stale leftover. Re-creating would produce an undersized queue that
      // crashes when deleteResults() is called for other destinations.
      if (removedTasks_.withLock(
              [&](auto& removed) { return removed.count(taskIdStr) > 0; })) {
        taskRemoved = true;
        return;
      }
      // create the queue structures such that the notify callback can be
      // stored. It will be later initialized once the task is being created.
      outputQueue = std::make_shared<UcxOutputQueue>(nullptr, destination, 0);
      queues[taskIdStr] = outputQueue;
    } else {
      // queue exists.
      outputQueue = it->second;
    }
  });
  if (taskRemoved) {
    // Fire callback immediately with nullptr to signal end-of-stream.
    notify(nullptr, {});
    return;
  }
  // outside of lock. Queue must exist.
  // get the data or install the notify callback.
  outputQueue->getData(destination, std::move(notify));
}

bool UcxOutputQueueManager::canUseIntraNode(std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  return queue && queue->isInitialized() &&
      queue->kind() != core::PartitionedOutputNode::Kind::kBroadcast;
}

void UcxOutputQueueManager::removeTask(std::string_view taskId) {
  std::string taskIdStr{taskId};
  auto queue =
      queues_.withLock([&](auto& queues) -> std::shared_ptr<UcxOutputQueue> {
        auto it = queues.find(taskIdStr);
        if (it == queues.end()) {
          // Already removed. Clear any stale "removed" state so the task ID
          // can be reused.
          removedTasks_.withLock(
              [&](auto& removed) { removed.erase(taskIdStr); });
          return nullptr;
        }
        auto taskQueue = it->second;
        queues.erase(it);
        // Insert into removedTasks_ while still holding the queues_ lock
        // to prevent getData() from seeing a gap between erase and insert,
        // which would cause it to create a zombie placeholder queue.
        removedTasks_.withLock(
            [&](auto& removed) { removed.insert(taskIdStr); });
        return taskQueue;
      });
  if (queue != nullptr) {
    queue->terminate();
  }
  // Notify the intra-node registry so that any sources polling for this
  // task get an atEnd result instead of spinning forever.
  IntraNodeTransferRegistry::getInstance()->cancelTask(taskId);
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueueIfExists(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    return it == queues.end() ? nullptr : it->second;
  });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueueIfActive(
    std::string_view taskId) {
  if (auto queue = getQueueIfExists(taskId)) {
    return queue;
  }
  VELOX_CHECK(
      isRemovedTask(taskId), "Output cudf queue for task not found: {}", taskId);
  return nullptr;
}

bool UcxOutputQueueManager::isRemovedTask(std::string_view taskId) {
  std::string taskIdStr{taskId};
  return removedTasks_.withLock(
      [&](auto& removed) { return removed.count(taskIdStr) > 0; });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueue(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    VELOX_CHECK(
        it != queues.end(), "Output cudf queue for task not found: {}", taskId);
    return it->second;
  });
}

std::optional<exec::OutputBuffer::Stats> UcxOutputQueueManager::stats(
    std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->stats();
  }
  return std::nullopt;
}

} // namespace facebook::velox::ucx_exchange
