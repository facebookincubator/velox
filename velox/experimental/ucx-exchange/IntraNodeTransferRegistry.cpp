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

#include <exception>

#include <glog/logging.h>

#include "velox/common/testutil/TestValue.h"

namespace facebook::velox::ucx_exchange {
namespace {

std::exception_ptr transferCancelledException() {
  static const auto exception =
      std::make_exception_ptr(IntraNodeTransferCancelled{});
  return exception;
}

} // namespace

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
    vector_size_t numRows,
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
    if (cancelledTasks_.count(key.taskId) || cancelledTransfers_.count(key)) {
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
    VLOG(2) << "[INTRA-REG] publish skipped (transfer cancelled): task="
            << key.taskId << " dest=" << key.destination
            << " seq=" << key.sequenceNumber;
    std::promise<void> p;
    auto future = p.get_future();
    p.set_exception(transferCancelledException());
    return future;
  }

  common::testutil::TestValue::adjust(
      "facebook::velox::ucx_exchange::IntraNodeTransferRegistry::publish",
      &entry);

  // Cancellation may have removed and completed this entry after the registry
  // lock was released. Preserve its terminal state for any poll() that already
  // obtained a reference to it.
  {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    // Get the future while holding the lock to avoid race with consumer
    future = entry->retrievedPromise.get_future();
    if (!entry->retrievalSignaled) {
      entry->data = std::move(data);
      entry->numRows = numRows;
      entry->atEnd = atEnd;
      entry->ready = true;
    }
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
    if (cancelledTasks_.count(key.taskId) || cancelledTransfers_.count(key)) {
      VLOG(2) << "[INTRA-REG] poll cancelled: task=" << key.taskId
              << " dest=" << key.destination << " seq=" << key.sequenceNumber;
      return IntraNodeTransferResult{
          .data = nullptr, .numRows = 0, .atEnd = true};
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

  // Hold the entry lock for the entire retrieval operation. Cancellation can
  // retain the entry after removing it from the registry, so the payload and
  // exactly-once promise state must be read and updated atomically.
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
    result.numRows = entry->numRows;
    result.atEnd = entry->atEnd;

    // Cancellation can remove this entry after poll() obtains its shared_ptr.
    // Serialize promise fulfillment with that path.
    if (!entry->retrievalSignaled) {
      entry->retrievalSignaled = true;
      entry->retrievedPromise.set_value();
    }
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

void IntraNodeTransferRegistry::cancelTransfer(
    const IntraNodeTransferKey& key) {
  // Construct the exception before changing registry ownership. If host
  // allocation fails, a later cleanup attempt can retry without having lost
  // the entry or its payload.
  auto cancellation = transferCancelledException();
  std::shared_ptr<IntraNodeTransferEntry> entry;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Install the tombstone before removing a published entry. If allocation
    // fails, the entry remains discoverable and a later cleanup attempt can
    // retry without losing ownership of the producer's promise or payload.
    if (!cancelledTasks_.count(key.taskId)) {
      cancelledTransfers_.insert(key);
    }
    auto it = registry_.find(key);
    if (it != registry_.end()) {
      entry = std::move(it->second);
      registry_.erase(it);
    }
  }

  if (entry) {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    entry->data.reset();
    entry->numRows = 0;
    entry->atEnd = true;
    entry->ready = true;
    if (!entry->retrievalSignaled) {
      entry->retrievedPromise.set_exception(std::move(cancellation));
      entry->retrievalSignaled = true;
    }
  }

  VLOG(2) << "[INTRA-REG] cancelTransfer: task=" << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " entryCleaned=" << (entry != nullptr);
}

void IntraNodeTransferRegistry::cancelTask(std::string_view taskId) {
  std::vector<std::shared_ptr<IntraNodeTransferEntry>> entriesToFulfill;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelledTasks_.insert(std::string{taskId});

    // The task-wide tombstone supersedes individual transfer tombstones. This
    // keeps early consumer cancellations bounded by the lifetime of the task.
    for (auto it = cancelledTransfers_.begin();
         it != cancelledTransfers_.end();) {
      if (it->taskId == taskId) {
        it = cancelledTransfers_.erase(it);
      } else {
        ++it;
      }
    }

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
    if (!entry->retrievalSignaled) {
      entry->retrievalSignaled = true;
      entry->retrievedPromise.set_value();
    }
  }

  VLOG(2) << "[INTRA-REG] cancelTask: task=" << taskId
          << " entriesCleaned=" << entriesToFulfill.size();
}

void IntraNodeTransferRegistry::clearCancelledTask(std::string_view taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  cancelledTasks_.erase(std::string{taskId});
  for (auto it = cancelledTransfers_.begin();
       it != cancelledTransfers_.end();) {
    if (it->taskId == taskId) {
      it = cancelledTransfers_.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace facebook::velox::ucx_exchange
