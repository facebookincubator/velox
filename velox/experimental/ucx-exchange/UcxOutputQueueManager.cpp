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
#include <rmm/exec_policy.hpp>
#include <cuda/stream>
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"

namespace facebook::velox::ucx_exchange {

/* static */
const std::shared_ptr<UcxOutputQueueManager>&
UcxOutputQueueManager::getInstanceRef() {
  // In C++11, the static local variable is guaranteed to only be initialized
  // once even in a multi-threaded context.
  static const std::shared_ptr<UcxOutputQueueManager> instance =
      std::make_shared<UcxOutputQueueManager>();
  return instance;
}

void UcxOutputQueueManager::initializeTask(
    std::shared_ptr<exec::Task> task,
    core::PartitionedOutputNode::Kind kind,
    int numDestinations,
    int numDrivers,
    const std::string& /*transportOptions*/) {
  VELOX_CHECK_NOT_NULL(task);
  VELOX_CHECK_GE(numDestinations, 0);
  VELOX_CHECK_GE(numDrivers, 0);
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
    // Keep the queues_ -> removedTasks_ lock order used by removeTask(). The
    // queue publication and tombstone removal must be one atomic lifecycle
    // transition; otherwise a concurrent remove can be undone here.
    removedTasks_.withLock([&](auto& removed) { removed.erase(taskId); });
    // The intra-node cancellation marker is the same task lifecycle state.
    // Clear it before publishing unlocks queues_; a concurrent remove then
    // necessarily runs later and reinstates both tombstones.
    IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
  });
}

bool UcxOutputQueueManager::updateOutputBuffers(
    const std::string& taskId,
    int numBuffers,
    bool noMoreBuffers) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->updateOutputBuffers(numBuffers, noMoreBuffers);
    return true;
  }
  return false;
}

void UcxOutputQueueManager::enqueue(
    std::string_view taskId,
    int destination,
    std::unique_ptr<cudf::packed_columns> txData,
    vector_size_t numRows,
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

void UcxOutputQueueManager::releaseInFlightBytes(
    std::string_view taskId,
    int destination,
    int64_t bytes,
    int64_t numPackedColumns) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->releaseInFlightBytes(destination, bytes, numPackedColumns);
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

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getData(
    std::string_view taskId,
    int destination,
    UcxDataAvailableCallback notify) {
  return getDataWithQueue(
      taskId,
      destination,
      [notify = std::move(notify)](
          std::shared_ptr<UcxOutputQueue> /*outputQueue*/,
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) mutable {
        notify(std::move(data), numRows, std::move(remainingBytes));
      });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getDataWithQueue(
    std::string_view taskId,
    int destination,
    UcxManagedDataAvailableCallback notify) {
  VELOX_CHECK_GE(destination, 0);
  std::shared_ptr<UcxOutputQueue> outputQueue;
  bool taskRemoved = false;
  std::string taskIdStr{taskId};
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    if (it == queues.end()) {
      // Check if the task was already removed. If so, don't re-create a
      // placeholder — the task is dead and any server calling getData() is a
      // stale leftover. Re-creating would produce an undersized queue that
      // crashes when deleteResults() is called for other destinations.
      if (removedTasks_.withLock(
              [&](auto& removed) { return removed.count(taskIdStr) > 0; })) {
        VLOG(2) << "[QUEUE-MGR] task=" << taskId << " dest=" << destination
                << " getData ignored (task already removed)";
        taskRemoved = true;
        return;
      }
      // create the queue structures such that the notify callback can be
      // stored. It will be later initialized once the task is being created.
      VLOG(2)
          << "[QUEUE-MGR] task=" << taskId << " dest=" << destination
          << " creating placeholder queue (server arrived before task init)";
      outputQueue = std::make_shared<UcxOutputQueue>(nullptr, destination, 0);
      queues[taskIdStr] = outputQueue;
    } else {
      // queue exists.
      outputQueue = it->second;
    }
  });
  if (taskRemoved) {
    // Fire callback immediately with nullptr to signal end-of-stream.
    notify(nullptr, nullptr, /*numRows=*/0, {});
    return nullptr;
  }
  // outside of lock. Queue must exist.
  // get the data or install the notify callback. Capture a weak pointer to
  // avoid a queue -> callback -> queue ownership cycle while still making the
  // exact queue available before invoking the server callback.
  std::weak_ptr<UcxOutputQueue> weakQueue = outputQueue;
  outputQueue->getData(
      destination,
      [weakQueue, notify = std::move(notify)](
          std::shared_ptr<cudf::packed_columns> data,
          vector_size_t numRows,
          std::vector<int64_t> remainingBytes) mutable {
        auto stableQueue = weakQueue.lock();
        VELOX_CHECK_NOT_NULL(
            stableQueue,
            "Output queue disappeared while dispatching dequeued data");
        notify(
            std::move(stableQueue),
            std::move(data),
            numRows,
            std::move(remainingBytes));
      });
  return outputQueue;
}

bool UcxOutputQueueManager::canUseIntraNode(std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  return queue && queue->isInitialized() &&
      queue->kind() != core::PartitionedOutputNode::Kind::kBroadcast;
}

void UcxOutputQueueManager::removeTask(const std::string& taskId) {
  std::string taskIdStr{taskId};
  auto queue =
      queues_.withLock([&](auto& queues) -> std::shared_ptr<UcxOutputQueue> {
        auto it = queues.find(taskIdStr);
        if (it == queues.end()) {
          // Idempotent removal must retain the tombstone. Otherwise a stale
          // server can recreate a zombie placeholder after a second cleanup.
          // initializeTask() is the sole operation that re-enables task IDs.
          removedTasks_.withLock(
              [&](auto& removed) { removed.insert(taskIdStr); });
          IntraNodeTransferRegistry::getInstance()->cancelTask(taskId);
          return nullptr;
        }
        auto taskQueue = it->second;
        queues.erase(it);
        // Insert into removedTasks_ while still holding the queues_ lock
        // to prevent getData() from seeing a gap between erase and insert,
        // which would cause it to create a zombie placeholder queue.
        removedTasks_.withLock(
            [&](auto& removed) { removed.insert(taskIdStr); });
        // Serialize the matching intra-node tombstone with queue removal. This
        // gives initializeTask() and removeTask() one consistent lifecycle
        // order even when they race on a reused ID.
        IntraNodeTransferRegistry::getInstance()->cancelTask(taskId);
        return taskQueue;
      });
  VLOG(2) << "[QUEUE-MGR] removeTask=" << taskId
          << " queueExists=" << (queue != nullptr);
  if (queue != nullptr) {
    queue->terminate();
  }
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
  std::string taskIdStr{taskId};
  return queues_.withLock([&](auto& queues) -> std::shared_ptr<UcxOutputQueue> {
    if (auto it = queues.find(taskIdStr); it != queues.end()) {
      return it->second;
    }
    // Keep absence and its lifecycle classification in one queues_ critical
    // section. Otherwise initializeTask() can publish between two independent
    // lookups and make an active, reused task ID look unknown.
    const bool removed = removedTasks_.withLock(
        [&](auto& removedTasks) { return removedTasks.count(taskIdStr) > 0; });
    VELOX_CHECK(removed, "Output cudf queue for task not found: {}", taskId);
    return nullptr;
  });
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
    const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->stats();
  }
  return std::nullopt;
}

bool UcxOutputQueueManager::updateNumDrivers(
    const std::string& taskId,
    uint32_t newNumDrivers) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->updateNumDrivers(newNumDrivers);
    return true;
  }
  return false;
}

std::optional<double> UcxOutputQueueManager::getUtilization(
    const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue == nullptr) {
    return std::nullopt;
  }
  return queue->getUtilization();
}

std::optional<bool> UcxOutputQueueManager::isOverutilized(
    const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue == nullptr) {
    return std::nullopt;
  }
  return queue->isOverutilized();
}

std::string UcxOutputQueueManager::toString(const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->toString();
  }
  return "UcxOutputQueue[" + taskId + " not found]";
}

} // namespace facebook::velox::ucx_exchange
