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
#include <folly/ScopeGuard.h>
#include <rmm/exec_policy.hpp>
#include "velox/common/testutil/TestValue.h"
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
  const auto& taskId = task->taskId();
  const std::weak_ptr<exec::Task> weakTask = task;
  std::vector<UcxDataAvailable> notifications;
  try {
    queues_.withLock([&](auto& queues) {
      common::testutil::TestValue::adjust(
          "facebook::velox::ucx_exchange::UcxOutputQueueManager::initializeTask",
          task.get());
      auto it = queues.find(taskId);
      if (it == queues.end()) {
        queues[taskId] = std::make_shared<UcxOutputQueue>(
            std::move(task), numDestinations, numDrivers, kind);
      } else {
        if (!it->second->initialize(
                task, numDestinations, numDrivers, kind, notifications)) {
          VELOX_FAIL(
              "Registering a cudf output queue for pre-existing taskId {}",
              taskId);
        }
      }
    });
  } catch (...) {
    // A Task whose initialization throws before its queue is registered is
    // removed with no queue that knows it, so its removal finds it here.
    failedInitializations_.withLock(
        [&](auto& failed) { failed.insert_or_assign(taskId, weakTask); });
    throw;
  }
  // Clear any stale "removed" state so that getData() calls after this
  // initializeTask() create proper placeholder queues if needed.
  removedTasks_.withLock([&](auto& removed) { removed.erase(taskId); });
  failedInitializations_.withLock([&](auto& failed) { failed.erase(taskId); });
  // Clear any stale "cancelled" state that a removal of the same task ID left
  // in the intra-node registry.
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
    UcxDataAvailableCallback notify) {
  getDataImpl(taskId, destination, std::move(notify), /*intraNode=*/false);
}

void UcxOutputQueueManager::getIntraNodeData(
    std::string_view taskId,
    int destination,
    UcxDataAvailableCallback notify) {
  getDataImpl(taskId, destination, std::move(notify), /*intraNode=*/true);
}

void UcxOutputQueueManager::getDataImpl(
    std::string_view taskId,
    int destination,
    UcxDataAvailableCallback notify,
    bool intraNode) {
  std::shared_ptr<UcxOutputQueue> outputQueue;
  bool taskRemoved = false;
  std::optional<std::string> producerError;
  std::string taskIdStr{taskId};
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskIdStr);
    if (it == queues.end()) {
      // Late readers of removed tasks must receive EOS, with the reason the
      // task was removed with, rather than wait on a new placeholder.
      if (removedTasks_.withLock([&](auto& removed) {
            auto removedIt = removed.find(taskIdStr);
            if (removedIt == removed.end()) {
              return false;
            }
            producerError = removedIt->second.producerError;
            return true;
          })) {
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
      outputQueue = std::make_shared<UcxOutputQueue>(nullptr, 0, 0);
      queues[taskIdStr] = outputQueue;
    } else {
      // queue exists.
      outputQueue = it->second;
    }
  });
  if (taskRemoved) {
    // Fire callback immediately with the end marker the task was removed with.
    notify(nullptr, /*numRows=*/0, {}, std::move(producerError));
    return;
  }
  // outside of lock. Queue must exist.
  // get the data or install the notify callback.
  if (intraNode) {
    outputQueue->getIntraNodeData(destination, std::move(notify));
  } else {
    outputQueue->getData(destination, std::move(notify));
  }
}

void UcxOutputQueueManager::intraNodePageRetrieved(
    std::string_view taskId,
    int destination) {
  // After removeTask() the queue is gone and the count no longer matters.
  if (auto queue = getQueueIfExists(taskId)) {
    queue->intraNodePageRetrieved(destination);
  }
}

bool UcxOutputQueueManager::canUseIntraNode(std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  return queue && queue->isInitialized() &&
      queue->kind() != core::PartitionedOutputNode::Kind::kBroadcast;
}

void UcxOutputQueueManager::removeTask(const std::string& taskId) {
  std::string taskIdStr{taskId};
  auto registry = IntraNodeTransferRegistry::getInstance();
  // Intra-node sources stop reading until cancelTask() publishes the outcome:
  // a page read after the decision below would be counted as lost although
  // its consumer has it.
  registry->freezeTask(taskId);
  auto unfreeze = folly::makeGuard(
      [&registry, &taskId]() { registry->unfreezeTask(taskId); });
  // Decided before the queue is unpublished, and with no manager lock held:
  // reading the task's state takes the task's mutex, and a getData() that
  // finds the task removed must already see the reason. The reason must come
  // from the queue that is removed, so it is decided again if the queue
  // changed meanwhile, as when a consumer left a placeholder for a task that
  // registered none. This ends because a task's queue is created at most once
  // and removed at most once.
  std::shared_ptr<UcxOutputQueue> queue;
  std::optional<std::string> producerError;
  auto expectedQueue = getQueueIfExists(taskIdStr);
  // Task::terminate() calls removeTask(), so the producer's Task is alive.
  // Its queue knows it, or a failed initializeTask() if it registered none.
  // Tests also remove task IDs that have no Task.
  auto producerTask = failedInitializations_.withLock([&](auto& failed) {
    std::weak_ptr<exec::Task> task;
    if (auto it = failed.find(taskIdStr); it != failed.end()) {
      task = std::move(it->second);
      failed.erase(it);
    }
    return task;
  });
  if (expectedQueue != nullptr) {
    if (auto task = expectedQueue->task()) {
      producerTask = task;
    }
  }
  // A record is needed only while a consumer can still arrive, which ends
  // when the host destroys the producer's Task.
  std::vector<std::string> droppedTaskIds;
  auto dropRecordsOfDestroyedTasks = [&droppedTaskIds](auto& removed) {
    for (auto it = removed.begin(); it != removed.end();) {
      if (it->second.task.expired()) {
        droppedTaskIds.push_back(it->first);
        it = removed.erase(it);
      } else {
        ++it;
      }
    }
  };
  for (bool decided = false; !decided;) {
    producerError = expectedQueue != nullptr
        ? expectedQueue->incompleteOutputError()
        : std::nullopt;
    common::testutil::TestValue::adjust(
        "facebook::velox::ucx_exchange::UcxOutputQueueManager::removeTask::afterProducerError",
        this);
    decided = queues_.withLock([&](auto& queues) {
      auto it = queues.find(taskIdStr);
      auto currentQueue = it == queues.end() ? nullptr : it->second;
      if (currentQueue != expectedQueue) {
        expectedQueue = std::move(currentQueue);
        return false;
      }
      if (it == queues.end()) {
        // The task never created its queue, as when initializeTask() threw
        // before registering one. A consumer that arrives later must fail
        // rather than wait on a placeholder that no task will remove, and the
        // intra-node registry below records the same reason. An earlier
        // removal keeps its own reason in both; initializeTask() clears the
        // entries if the task ID is reused.
        producerError = std::string{kProducerNotInitializedError};
        removedTasks_.withLock([&](auto& removed) {
          dropRecordsOfDestroyedTasks(removed);
          removed.try_emplace(
              taskIdStr, RemovedTask{producerTask, producerError});
        });
        return true;
      }
      queues.erase(it);
      // Insert into removedTasks_ while still holding the queues_ lock
      // to prevent getData() from seeing a gap between erase and insert,
      // which would cause it to create a zombie placeholder queue.
      removedTasks_.withLock([&](auto& removed) {
        dropRecordsOfDestroyedTasks(removed);
        removed.insert_or_assign(
            taskIdStr, RemovedTask{producerTask, producerError});
      });
      queue = std::move(currentQueue);
      return true;
    });
  }
  VLOG(2) << "[QUEUE-MGR] removeTask=" << taskId
          << " queueExists=" << (queue != nullptr);
  if (queue != nullptr) {
    queue->terminate(producerError);
  }
  // Notify the intra-node registry so that any sources polling for this
  // task get an atEnd result instead of spinning forever.
  registry->cancelTask(taskId, producerError);
  for (const auto& droppedTaskId : droppedTaskIds) {
    registry->clearCancelledTask(droppedTaskId);
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
    return "UcxOutputQueue[" + taskId + "]";
  }
  return "UcxOutputQueue[" + taskId + " not found]";
}

} // namespace facebook::velox::ucx_exchange
