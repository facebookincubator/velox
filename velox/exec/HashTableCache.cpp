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

#include "velox/exec/HashTableCache.h"

#include <fmt/format.h>

namespace facebook::velox::exec {

std::atomic<HashTableCache*> HashTableCache::instance_{nullptr};
std::mutex HashTableCache::instanceMutex_;

HashTableCache* HashTableCache::getInstance() {
  auto* instance = instance_.load(std::memory_order_acquire);
  if (instance == nullptr) {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    instance = instance_.load(std::memory_order_relaxed);
    if (instance == nullptr) {
      instance = new HashTableCache();
      instance_.store(instance, std::memory_order_release);
    }
  }
  return instance;
}

/// Retrieves or creates the memory pool for a cached table.
/// This must be called while holding the tables_ lock.
///
/// @param key The unique identifier for the cached table
/// @param queryPool The query-level memory pool to create child pool under
/// @param entry The cache entry to store the pool in
static void getOrCreateTablePool(
    const std::string& key,
    memory::MemoryPool* queryPool,
    std::shared_ptr<HashTableEntry>& entry) {
  if (entry->tablePool == nullptr) {
    entry->tablePool =
        queryPool->addLeafChild(fmt::format("cached_table_{}", key));
  }
}

std::shared_ptr<HashTableEntry> HashTableCache::getOrAwait(
    const std::string& key,
    memory::MemoryPool* queryPool,
    const std::string& taskId,
    ContinueFuture* future) {
  auto lockedTables = tables_.wlock();

  auto it = lockedTables->find(key);
  if (it == lockedTables->end()) {
    // No entry exists - create a placeholder for this task to build the table.
    auto entry = std::make_shared<HashTableEntry>();
    entry->buildComplete.store(false, std::memory_order_release);
    entry->builderTaskId = taskId; // Set builder task atomically with creation

    // Create the memory pool for this broadcast table
    getOrCreateTablePool(key, queryPool, entry);

    lockedTables->insert({key, entry});
    return entry; // Return entry with pool, table will be filled later
  }

  auto& entry = it->second;

  // Ensure pool is created (in case entry was created without pool somehow)
  getOrCreateTablePool(key, queryPool, entry);

  // Check if build is complete
  if (entry->buildComplete.load(std::memory_order_acquire)) {
    return entry;
  }

  // Table is being built by another task - set up waiting if future provided
  if (future != nullptr) {
    std::lock_guard<std::mutex> promiseLock(entry->promisesMutex);
    // Double-check after acquiring lock
    if (!entry->buildComplete.load(std::memory_order_acquire)) {
      entry->buildPromises.emplace_back(fmt::format("HashTableCache::{}", key));
      *future = entry->buildPromises.back().getSemiFuture();
    }
  }

  return entry;
}

void HashTableCache::put(
    const std::string& key,
    std::shared_ptr<BaseHashTable> table,
    bool hasNullKeys) {
  std::vector<ContinuePromise> promises;

  {
    auto lockedTables = tables_.wlock();

    auto it = lockedTables->find(key);

    if (it == lockedTables->end()) {
      // Shouldn't happen - entry should be created by getOrAwait
      auto entry =
          std::make_shared<HashTableEntry>(std::move(table), hasNullKeys);
      lockedTables->insert({key, entry});
      return;
    }

    auto& entry = it->second;

    // Update the entry with the built table
    entry->table = std::move(table);
    entry->hasNullKeys = hasNullKeys;
    entry->buildComplete.store(true, std::memory_order_release);

    // Collect promises to notify waiters
    {
      std::lock_guard<std::mutex> promiseLock(entry->promisesMutex);
      promises = std::move(entry->buildPromises);
    }
  }

  // Notify all waiting tasks outside the lock
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void HashTableCache::flushQuery(const std::string& cacheKey) {
  auto* cache = getInstance();
  auto lockedTables = cache->tables_.wlock();
  lockedTables->erase(cacheKey);
}

} // namespace facebook::velox::exec
