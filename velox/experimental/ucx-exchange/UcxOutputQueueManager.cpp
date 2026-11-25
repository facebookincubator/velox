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
  const std::string taskId = task->taskId();
  std::vector<UcxDataAvailable> notifications;
  // Destroyed after the manager's lock is released.
  std::shared_ptr<UcxOutputQueue> replacedEntry;
  entries_.withLock([&](auto& entries) {
    auto& entry = entries[taskId];
    if (entry != nullptr && entry->state() == TaskEntryState::kRemoved) {
      // The ID is reused. Servers bound to the Removed entry keep it.
      replacedEntry = std::move(entry);
    }
    if (entry == nullptr) {
      entry = std::make_shared<UcxOutputQueue>(taskId);
    }
    bool initialized{false};
    try {
      initialized = entry->initialize(
          task, numDestinations, numDrivers, kind, notifications);
    } catch (...) {
      entry->recordFailedInitialization(task);
      throw;
    }
    if (!initialized) {
      VELOX_FAIL(
          "Registering a cudf output queue for pre-existing taskId {}", taskId);
    }
  });
  // Clear any stale "cancelled" state in the intra-node registry so
  // that the cancelledTasks_ set doesn't grow unboundedly across queries.
  IntraNodeTransferRegistry::getInstance()->clearCancelledTask(taskId);
  if (kind == core::PartitionedOutputNode::Kind::kPartitioned &&
      !notifications.empty()) {
    VLOG(1) << "UCX reader requests outside partitioned destinations: task="
            << taskId << " rejected=" << notifications.size();
  }
  for (auto& notification : notifications) {
    notification.notify();
  }
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
    vector_size_t numRows) {
  getQueue(taskId)->enqueue(destination, std::move(txData), numRows);
}

bool UcxOutputQueueManager::checkBlocked(
    std::string_view taskId,
    ContinueFuture* future) {
  return getQueue(taskId)->checkBlocked(future);
}

void UcxOutputQueueManager::noMoreData(std::string_view taskId) {
  getQueue(taskId)->noMoreData();
}

bool UcxOutputQueueManager::isFinished(std::string_view taskId) {
  return getQueue(taskId)->isFinished();
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
    ReaderCallback notify) {
  bindReader(taskId, destination)->getData(destination, std::move(notify));
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::bindReader(
    std::string_view taskId,
    int destination) {
  return entries_.withLock([&](auto& entries) {
    auto& entry = entries[std::string{taskId}];
    if (entry == nullptr) {
      VLOG(2) << "[QUEUE-MGR] task=" << taskId << " dest=" << destination
              << " creating Pending entry (server arrived before task init)";
      entry = std::make_shared<UcxOutputQueue>(std::string{taskId});
    }
    entry->addBoundReader();
    return entry;
  });
}

void UcxOutputQueueManager::releaseReader(
    const std::shared_ptr<UcxOutputQueue>& entry,
    int destination) {
  // Destroyed after the manager's lock is released.
  std::shared_ptr<UcxOutputQueue> erasedEntry;
  entries_.withLock([&](auto& entries) {
    if (!entry->removeBoundReader()) {
      return;
    }
    auto it = entries.find(entry->taskId());
    // The map may hold a newer entry for the same ID.
    if (it != entries.end() && it->second == entry) {
      VLOG(2) << "[QUEUE-MGR] task=" << entry->taskId()
              << " dest=" << destination
              << " erasing Pending entry with no bound reader";
      erasedEntry = std::move(it->second);
      entries.erase(it);
    }
  });
}

bool UcxOutputQueueManager::canUseIntraNode(std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  return queue && queue->isInitialized() &&
      queue->kind() != core::PartitionedOutputNode::Kind::kBroadcast;
}

void UcxOutputQueueManager::failTask(
    std::string_view taskId,
    std::string_view error) {
  auto queue = getQueueIfExists(taskId);
  // Null unless the entry is Active.
  auto task = queue != nullptr ? queue->task() : nullptr;
  if (task == nullptr) {
    LOG(ERROR) << "Cannot fail UCX producer task " << taskId
               << ", which has no initialized output on this worker: " << error;
    return;
  }
  task->setError(std::string(error));
}

void UcxOutputQueueManager::removeTask(const std::string& taskId) {
  // Destroyed after the manager's lock is released.
  std::vector<std::shared_ptr<UcxOutputQueue>> droppedEntries;
  auto entry = entries_.withLock([&](auto& entries) {
    auto& slot = entries[taskId];
    if (slot == nullptr) {
      slot = std::make_shared<UcxOutputQueue>(taskId);
    }
    auto removedEntry = slot;
    for (auto it = entries.begin(); it != entries.end();) {
      if (it->second != removedEntry && it->second->isDroppable()) {
        droppedEntries.push_back(std::move(it->second));
        it = entries.erase(it);
      } else {
        ++it;
      }
    }
    return removedEntry;
  });
  VLOG(2) << "[QUEUE-MGR] removeTask=" << taskId
          << " droppedEntries=" << droppedEntries.size();
  entry->terminate();
  entry->markRemoved();
  // Notify the intra-node registry so that any sources polling for this
  // task get an atEnd result instead of spinning forever.
  IntraNodeTransferRegistry::getInstance()->cancelTask(taskId);
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueueIfExists(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return entries_.withLock(
      [&](auto& entries) -> std::shared_ptr<UcxOutputQueue> {
        auto it = entries.find(taskIdStr);
        if (it == entries.end() ||
            it->second->state() == TaskEntryState::kRemoved) {
          return nullptr;
        }
        return it->second;
      });
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueue(
    std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  VELOX_CHECK_NOT_NULL(
      queue, "Output cudf queue for task not found: {}", taskId);
  return queue;
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
    return "UcxOutputQueue[" + taskId + "]";
  }
  return "UcxOutputQueue[" + taskId + " not found]";
}

} // namespace facebook::velox::ucx_exchange
