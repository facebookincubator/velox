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
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include <glog/logging.h>

namespace facebook::velox::ucx_exchange {

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
    entry->atEnd = atEnd;
    entry->ready = true;
    // Get the future while holding the lock to avoid race with consumer
    future = entry->retrievedPromise.get_future();
  }

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
      return IntraNodeTransferResult{nullptr, true};
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

void IntraNodeTransferRegistry::cancelTask(std::string_view taskId) {
  std::vector<std::shared_ptr<IntraNodeTransferEntry>> entriesToFulfill;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelledTasks_.insert(std::string{taskId});

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

void IntraNodeTransferRegistry::clearCancelledTask(std::string_view taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  cancelledTasks_.erase(std::string{taskId});
}

} // namespace facebook::velox::ucx_exchange
