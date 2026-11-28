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
  std::shared_ptr<UcxOutputQueue> replacedOutputQueue;
  outputQueues_.withLock([&](auto& outputQueues) {
    auto& outputQueue = outputQueues[taskId];
    if (outputQueue != nullptr &&
        outputQueue->state() == OutputQueueState::kRemoved) {
      // The ID is reused. Servers bound to the Removed output queue keep it.
      replacedOutputQueue = std::move(outputQueue);
    }
    if (outputQueue == nullptr) {
      outputQueue = std::make_shared<UcxOutputQueue>(taskId);
    }
    bool initialized{false};
    try {
      initialized = outputQueue->initialize(
          task, numDestinations, numDrivers, kind, notifications);
    } catch (...) {
      outputQueue->recordFailedInitialization(task);
      throw;
    }
    if (!initialized) {
      VELOX_FAIL(
          "Registering a cudf output queue for pre-existing taskId {}", taskId);
    }
  });
  // A reused task ID must not inherit the cancellation of its earlier task.
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
  bindReader(taskId, destination, /*onReleased=*/nullptr)
      ->getData(destination, std::move(notify), /*intraNode=*/false);
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::bindReader(
    std::string_view taskId,
    int destination,
    std::function<void()> onReleased) {
  return outputQueues_.withLock([&](auto& outputQueues) {
    auto& outputQueue = outputQueues[std::string{taskId}];
    if (outputQueue == nullptr) {
      VLOG(2) << "[QUEUE-MGR] task=" << taskId << " dest=" << destination
              << " creating Pending output queue (server arrived before task "
                 "init)";
      outputQueue = std::make_shared<UcxOutputQueue>(std::string{taskId});
    }
    outputQueue->addBoundReader();
    if (onReleased != nullptr) {
      outputQueue->setOnReleased(destination, std::move(onReleased));
    }
    return outputQueue;
  });
}

void UcxOutputQueueManager::consumerGone(
    std::string_view taskId,
    int destination) {
  auto outputQueue = outputQueues_.withLock(
      [&](auto& outputQueues) -> std::shared_ptr<UcxOutputQueue> {
        auto it = outputQueues.find(std::string{taskId});
        return it != outputQueues.end() ? it->second : nullptr;
      });
  if (outputQueue == nullptr) {
    VLOG(1) << "Dropping a close for unknown UCX producer task " << taskId
            << ", destination " << destination;
    return;
  }
  // Outside the manager's lock: the output queue calls its release callback.
  outputQueue->consumerGone(destination);
}

void UcxOutputQueueManager::releaseReader(
    const std::shared_ptr<UcxOutputQueue>& outputQueue,
    int destination) {
  // Destroyed after the manager's lock is released.
  std::shared_ptr<UcxOutputQueue> erasedOutputQueue;
  outputQueues_.withLock([&](auto& outputQueues) {
    if (!outputQueue->removeBoundReader(destination)) {
      return;
    }
    auto it = outputQueues.find(outputQueue->taskId());
    // The map may hold a newer output queue for the same ID.
    if (it != outputQueues.end() && it->second == outputQueue) {
      VLOG(2) << "[QUEUE-MGR] task=" << outputQueue->taskId()
              << " dest=" << destination
              << " erasing Pending output queue with no bound reader";
      erasedOutputQueue = std::move(it->second);
      outputQueues.erase(it);
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
  // Null unless the output queue is Active.
  auto task = queue != nullptr ? queue->task() : nullptr;
  if (task == nullptr) {
    LOG(ERROR) << "Cannot fail UCX producer task " << taskId
               << ", which has no initialized output on this worker: " << error;
    return;
  }
  task->setError(std::string(error));
}

namespace {
// The producer task's error without its stack trace and context, which the
// producer's own failure reports.
std::string taskErrorMessage(const exec::Task& task) {
  const auto error = task.error();
  if (error == nullptr) {
    return "";
  }
  try {
    std::rethrow_exception(error);
  } catch (const VeloxException& e) {
    return e.message();
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "<Unknown exception type>";
  }
}
} // namespace

void UcxOutputQueueManager::removeTask(const std::string& taskId) {
  // Destroyed after the manager's lock is released.
  std::vector<std::shared_ptr<UcxOutputQueue>> droppedOutputQueues;
  auto outputQueue = outputQueues_.withLock([&](auto& outputQueues) {
    auto& slot = outputQueues[taskId];
    if (slot == nullptr) {
      slot = std::make_shared<UcxOutputQueue>(taskId);
    }
    auto removedOutputQueue = slot;
    for (auto it = outputQueues.begin(); it != outputQueues.end();) {
      // The lock-free state check first, so that live output queues are never
      // locked here.
      if (it->second != removedOutputQueue &&
          it->second->state() == OutputQueueState::kRemoved &&
          it->second->isDroppable()) {
        droppedOutputQueues.push_back(std::move(it->second));
        it = outputQueues.erase(it);
      } else {
        ++it;
      }
    }
    return removedOutputQueue;
  });
  VLOG(2) << "[QUEUE-MGR] removeTask=" << taskId
          << " droppedOutputQueues=" << droppedOutputQueues.size();
  // Read with no output queue lock held, since no Task method may run under
  // one. Task::terminate() sets the final state before it removes the task, so
  // the facts are final. They could race with an initialize() of a Pending
  // output queue, but Velox never initializes a task's output while it
  // terminates the task.
  RemovalFacts facts;
  facts.taskId = taskId;
  if (auto task = outputQueue->task()) {
    facts.state = task->state();
    facts.error = taskErrorMessage(*task);
  }
  auto outcome = outputQueue->remove(facts);
  // Without the output queue's lock, which is never held together with the
  // registry's. Sources polling for this task then end with the removal's
  // outcome instead of spinning forever.
  auto registry = IntraNodeTransferRegistry::getInstance();
  registry->cancelTask(taskId, std::move(outcome));
  // A dropped output queue had no bound reader and its Task is gone, so no
  // server publishes for it any more. Its cancellation goes with it, so that
  // the registry keeps no record per removed task for the life of the worker. A
  // source that has not yet read the cancellation is closed by its consumer's
  // abort.
  for (const auto& droppedOutputQueue : droppedOutputQueues) {
    registry->clearCancelledTask(droppedOutputQueue->taskId());
  }
}

std::shared_ptr<UcxOutputQueue> UcxOutputQueueManager::getQueueIfExists(
    std::string_view taskId) {
  std::string taskIdStr{taskId};
  return outputQueues_.withLock(
      [&](auto& outputQueues) -> std::shared_ptr<UcxOutputQueue> {
        auto it = outputQueues.find(taskIdStr);
        if (it == outputQueues.end() ||
            it->second->state() == OutputQueueState::kRemoved) {
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
