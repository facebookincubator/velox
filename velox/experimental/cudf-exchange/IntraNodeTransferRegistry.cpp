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
#include "velox/experimental/cudf-exchange/IntraNodeTransferRegistry.h"
#include <glog/logging.h>

namespace facebook::velox::cudf_exchange {

/* static */
std::shared_ptr<IntraNodeTransferRegistry>
IntraNodeTransferRegistry::getInstance() {
  // In C++11, the static local variable is guaranteed to only be initialized
  // once even in a multi-threaded context.
  static std::shared_ptr<IntraNodeTransferRegistry> instance =
      std::shared_ptr<IntraNodeTransferRegistry>(
          new IntraNodeTransferRegistry());
  return instance;
}

std::future<void> IntraNodeTransferRegistry::publish(
    const IntraNodeTransferKey& key,
    std::shared_ptr<cudf::packed_columns> data,
    rmm::cuda_stream_view stream,
    bool atEnd) {
  std::shared_ptr<IntraNodeTransferEntry> entry;
  std::future<void> future;
  bool entryExisted;
  size_t registrySize;

  bool cancelled = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // If the task was already cancelled (removeTask was called), don't create
    // a registry entry. Return an already-fulfilled future so the server
    // doesn't block waiting for a source that will never come.
    if (cancelledTasks_.count(key.taskId)) {
      cancelled = true;
    } else {
      // Check if entry already exists (source may have started waiting)
      auto it = registry_.find(key);
      entryExisted = (it != registry_.end());
      if (entryExisted) {
        entry = it->second;
      } else {
        entry = std::make_shared<IntraNodeTransferEntry>();
        registry_[key] = entry;
      }
      registrySize = registry_.size();
    }
  }

  if (cancelled) {
    VLOG(2) << "[INTRA-REG] publish skipped (task cancelled): task="
            << key.taskId << " dest=" << key.destination
            << " seq=" << key.sequenceNumber;
    std::promise<void> p;
    p.set_value();
    return p.get_future();
  }

  // Update the entry with data (under entry's own mutex)
  {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    entry->data = std::move(data);
    entry->stream = stream;
    entry->atEnd = atEnd;
    entry->ready = true;
    // Get the future while holding the lock to avoid race with consumer
    future = entry->retrievedPromise.get_future();
  }

  // Notify waiting source
  entry->dataAvailable.notify_one();

  VLOG(2) << "[INTRA-REG] publish: task=" << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << atEnd << " entryExisted=" << entryExisted
          << " registrySize=" << registrySize;

  return future;
}

std::optional<IntraNodeTransferResult> IntraNodeTransferRegistry::poll(
    const IntraNodeTransferKey& key) {
  std::shared_ptr<IntraNodeTransferEntry> entry;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if this task has been cancelled (producer removed).
    if (cancelledTasks_.count(key.taskId)) {
      VLOG(2) << "[INTRA-REG] poll cancelled: task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber;
      return IntraNodeTransferResult{nullptr, rmm::cuda_stream_default, true};
    }

    auto it = registry_.find(key);
    if (it == registry_.end()) {
      // No entry yet - server hasn't published
      VLOG(3) << "[INTRA-REG] poll miss (no entry): task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber
              << " registrySize=" << registry_.size();
      return std::nullopt;
    }
    entry = it->second;
  }

  // FIX: Hold the entry lock for the entire retrieval operation to prevent
  // race conditions. Previously, the lock was released before accessing
  // entry->data, entry->atEnd, and entry->retrievedPromise.
  IntraNodeTransferResult result;
  {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    if (!entry->ready) {
      // Entry exists but data not ready yet
      VLOG(3) << "[INTRA-REG] poll miss (not ready): task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber;
      return std::nullopt;
    }

    // Data is ready, retrieve it while holding the lock
    result.data = std::move(entry->data);
    result.stream = entry->stream;
    result.atEnd = entry->atEnd;

    // Fulfill the promise to notify the server while still holding entry lock
    entry->retrievedPromise.set_value();
  }

  // Remove entry from registry (after releasing entry lock but before
  // returning)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_.erase(key);
  }

  VLOG(2) << "[INTRA-REG] poll hit: task=" << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << result.atEnd;

  return result;
}

IntraNodeTransferResult IntraNodeTransferRegistry::waitFor(
    const IntraNodeTransferKey& key,
    std::chrono::milliseconds timeout) {
  std::shared_ptr<IntraNodeTransferEntry> entry;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if entry already exists (server may have published)
    auto it = registry_.find(key);
    if (it != registry_.end()) {
      entry = it->second;
    } else {
      // Create entry so server can find it when it publishes
      entry = std::make_shared<IntraNodeTransferEntry>();
      registry_[key] = entry;
    }
  }

  // FIX: Hold the entry lock for the entire wait and retrieval operation
  // to prevent race conditions. Previously, the lock was released before
  // accessing entry->data, entry->atEnd, and entry->retrievedPromise.
  IntraNodeTransferResult result;
  {
    std::unique_lock<std::mutex> entryLock(entry->entryMutex);
    if (!entry->ready) {
      auto status = entry->dataAvailable.wait_for(
          entryLock, timeout, [&entry]() { return entry->ready; });
      if (!status) {
        // Timeout - return empty result
        VLOG(0) << "Timeout waiting for intra-node transfer: " << key.taskId
                << " dest=" << key.destination << " seq=" << key.sequenceNumber;
        return {nullptr, rmm::cuda_stream_default, false};
      }
    }

    // Data is ready, retrieve it while holding the lock
    result.data = std::move(entry->data);
    result.stream = entry->stream;
    result.atEnd = entry->atEnd;

    // Fulfill the promise to notify the server while still holding entry lock
    entry->retrievedPromise.set_value();
  }

  // Remove entry from registry (after releasing entry lock but before
  // returning)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_.erase(key);
  }

  VLOG(3) << "Retrieved intra-node transfer (wait): " << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << result.atEnd;

  return result;
}

std::shared_ptr<cudf::packed_columns> IntraNodeTransferRegistry::retrieve(
    const IntraNodeTransferKey& key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = registry_.find(key);
  if (it == registry_.end()) {
    VLOG(0) << "Intra-node transfer entry not found: " << key.taskId
            << " dest=" << key.destination << " seq=" << key.sequenceNumber;
    return nullptr;
  }

  auto entry = it->second;
  auto data = std::move(entry->data);

  // Fulfill the promise to notify the server that data has been retrieved
  entry->retrievedPromise.set_value();

  // Remove the entry from the registry
  registry_.erase(it);

  VLOG(3) << "Retrieved intra-node transfer: " << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber;

  return data;
}

void IntraNodeTransferRegistry::cancelTask(const std::string& taskId) {
  std::vector<std::shared_ptr<IntraNodeTransferEntry>> entriesToFulfill;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelledTasks_.insert(taskId);

    // Clean up any existing registry entries for this task so servers
    // waiting on the retrieved-promise don't hang.
    for (auto it = registry_.begin(); it != registry_.end();) {
      if (it->first.taskId == taskId) {
        entriesToFulfill.push_back(it->second);
        it = registry_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Fulfill promises outside the lock to avoid potential deadlocks.
  for (auto& entry : entriesToFulfill) {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    if (!entry->ready) {
      entry->ready = true;
      entry->atEnd = true;
    }
    try {
      entry->retrievedPromise.set_value();
    } catch (const std::future_error&) {
      // Promise already satisfied — safe to ignore.
    }
  }

  VLOG(2) << "[INTRA-REG] cancelTask: task=" << taskId
          << " entriesCleaned=" << entriesToFulfill.size();
}

void IntraNodeTransferRegistry::clearCancelledTask(const std::string& taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  cancelledTasks_.erase(taskId);
}

} // namespace facebook::velox::cudf_exchange
