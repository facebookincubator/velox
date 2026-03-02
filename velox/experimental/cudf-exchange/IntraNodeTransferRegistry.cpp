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
    bool atEnd) {
  std::shared_ptr<IntraNodeTransferEntry> entry;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if entry already exists (source may have started waiting)
    auto it = registry_.find(key);
    if (it != registry_.end()) {
      entry = it->second;
    } else {
      entry = std::make_shared<IntraNodeTransferEntry>();
      registry_[key] = entry;
    }
  }

  // Update the entry with data (under entry's own mutex)
  {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    entry->data = std::move(data);
    entry->atEnd = atEnd;
    entry->ready = true;
  }

  // Notify waiting source
  entry->dataAvailable.notify_one();

  VLOG(3) << "Published intra-node transfer: " << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << atEnd;

  return entry->retrievedPromise.get_future();
}

std::optional<std::pair<std::shared_ptr<cudf::packed_columns>, bool>>
IntraNodeTransferRegistry::poll(const IntraNodeTransferKey& key) {
  std::shared_ptr<IntraNodeTransferEntry> entry;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = registry_.find(key);
    if (it == registry_.end()) {
      // No entry yet - server hasn't published
      return std::nullopt;
    }
    entry = it->second;
  }

  // Check if data is ready (non-blocking)
  {
    std::lock_guard<std::mutex> entryLock(entry->entryMutex);
    if (!entry->ready) {
      // Entry exists but data not ready yet
      return std::nullopt;
    }
  }

  // Data is ready, retrieve it
  auto data = std::move(entry->data);
  bool atEnd = entry->atEnd;

  // Fulfill the promise to notify the server
  entry->retrievedPromise.set_value();

  // Remove entry from registry
  {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_.erase(key);
  }

  VLOG(3) << "Retrieved intra-node transfer (poll): " << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << atEnd;

  return std::make_pair(std::move(data), atEnd);
}

std::pair<std::shared_ptr<cudf::packed_columns>, bool>
IntraNodeTransferRegistry::waitFor(
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

  // Wait for data to be ready
  {
    std::unique_lock<std::mutex> entryLock(entry->entryMutex);
    if (!entry->ready) {
      auto status = entry->dataAvailable.wait_for(
          entryLock, timeout, [&entry]() { return entry->ready; });
      if (!status) {
        // Timeout - return empty result
        VLOG(0) << "Timeout waiting for intra-node transfer: " << key.taskId
                << " dest=" << key.destination << " seq=" << key.sequenceNumber;
        return {nullptr, false};
      }
    }
  }

  // Data is ready, retrieve it
  auto data = std::move(entry->data);
  bool atEnd = entry->atEnd;

  // Fulfill the promise to notify the server
  entry->retrievedPromise.set_value();

  // Remove entry from registry
  {
    std::lock_guard<std::mutex> lock(mutex_);
    registry_.erase(key);
  }

  VLOG(3) << "Retrieved intra-node transfer (wait): " << key.taskId
          << " dest=" << key.destination << " seq=" << key.sequenceNumber
          << " atEnd=" << atEnd;

  return {std::move(data), atEnd};
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

} // namespace facebook::velox::cudf_exchange
