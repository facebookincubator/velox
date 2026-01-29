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
#include "velox/experimental/cudf-exchange/CudfOutputQueueManager.h"
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace facebook::velox::cudf_exchange {

/* static */
std::shared_ptr<CudfOutputQueueManager>
CudfOutputQueueManager::getInstanceRef() {
  // In C++11, the static local variable is guaranteed to only be initialized
  // once even in a multi-threaded context.
  static std::shared_ptr<CudfOutputQueueManager> instance =
      std::make_shared<CudfOutputQueueManager>();
  return instance;
}

void CudfOutputQueueManager::initializeTask(
    std::shared_ptr<exec::Task> task,
    int numDestinations,
    int numDrivers) {
  const auto& taskId = task->taskId();
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskId);
    if (it == queues.end()) {
      queues[taskId] = std::make_shared<CudfOutputQueue>(
          std::move(task), numDestinations, numDrivers);
    } else {
      if (!it->second->initialize(task, numDestinations, numDrivers)) {
        VELOX_FAIL(
            "Registering a cudf output queue for pre-existing taskId {}",
            taskId);
      }
    }
  });
}

void CudfOutputQueueManager::enqueue(
    const std::string& taskId,
    int destination,
    std::unique_ptr<cudf::packed_columns> txData,
    int numRows) {
  getQueue(taskId)->enqueue(destination, std::move(txData), numRows);
}

bool CudfOutputQueueManager::checkBlocked(
    const std::string& taskId,
    ContinueFuture* future) {
  return getQueue(taskId)->checkBlocked(future);
}

void CudfOutputQueueManager::noMoreData(const std::string& taskId) {
  getQueue(taskId)->noMoreData();
}

bool CudfOutputQueueManager::isFinished(const std::string& taskId) {
  return getQueue(taskId)->isFinished();
}

void CudfOutputQueueManager::deleteResults(
    const std::string& taskId,
    int destination) {
  if (auto queue = getQueueIfExists(taskId)) {
    queue->deleteResults(destination);
  }
}

void CudfOutputQueueManager::getData(
    const std::string& taskId,
    int destination,
    CudfDataAvailableCallback notify) {
  std::shared_ptr<CudfOutputQueue> outputQueue;
  queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskId);
    if (it == queues.end()) {
      // create the queue structures such that the notify callback can be
      // stored. It will be later initialized once the task is being created.
      outputQueue = std::make_shared<CudfOutputQueue>(nullptr, destination, 0);
      queues[taskId] = outputQueue;
    } else {
      // queue exists.
      outputQueue = it->second;
    }
  });
  // outside of lock. Queue must exist.
  // get the data or install the notify callback.
  outputQueue->getData(destination, notify);
}

void CudfOutputQueueManager::removeTask(const std::string& taskId) {
  auto queue =
      queues_.withLock([&](auto& queues) -> std::shared_ptr<CudfOutputQueue> {
        auto it = queues.find(taskId);
        if (it == queues.end()) {
          // Already removed.
          return nullptr;
        }
        auto taskQueue = it->second;
        queues.erase(taskId);
        return taskQueue;
      });
  if (queue != nullptr) {
    queue->terminate();
  }
}

std::shared_ptr<CudfOutputQueue> CudfOutputQueueManager::getQueueIfExists(
    const std::string& taskId) {
  return queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskId);
    return it == queues.end() ? nullptr : it->second;
  });
}

std::shared_ptr<CudfOutputQueue> CudfOutputQueueManager::getQueue(
    const std::string& taskId) {
  return queues_.withLock([&](auto& queues) {
    auto it = queues.find(taskId);
    VELOX_CHECK(
        it != queues.end(), "Output cudf queue for task not found: {}", taskId);
    return it->second;
  });
}

std::optional<exec::OutputBuffer::Stats> CudfOutputQueueManager::stats(
    const std::string& taskId) {
  auto queue = getQueueIfExists(taskId);
  if (queue != nullptr) {
    return queue->stats();
  }
  return std::nullopt;
}

} // namespace facebook::velox::cudf_exchange
