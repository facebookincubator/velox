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

#include "velox/core/QueryCtx.h"

namespace facebook::velox::exec {

HashTableCache* HashTableCache::instance() {
  static HashTableCache instance;
  return &instance;
}

std::shared_ptr<HashTableCacheEntry> HashTableCache::get(
    const std::string& key,
    const std::string& taskId,
    core::QueryCtx* queryCtx,
    ContinueFuture* future) {
  VELOX_CHECK_NOT_NULL(future, "future parameter must not be null");
  VELOX_CHECK_NOT_NULL(queryCtx, "queryCtx parameter must not be null");

  std::lock_guard<std::mutex> guard(lock_);

  auto it = tables_.find(key);
  if (it == tables_.end()) {
    // No entry exists - create a placeholder for this task to build the table.
    auto* queryPool = queryCtx->pool();
    auto entry = std::make_shared<HashTableCacheEntry>(
        key,
        taskId,
        queryPool->addLeafChild(fmt::format("cached_table_{}", key)));
    tables_.insert({key, entry});

    // Register callback to clean up this cache entry when QueryCtx is
    // destroyed. This ensures tablePool memory is freed before the query
    // pool is destroyed.
    queryCtx->addReleaseCallback(
        [cacheKey = key]() { HashTableCache::instance()->drop(cacheKey); });

    // Return entry with pool, table will be filled later.
    return entry;
  }

  auto& entry = it->second;

  // Check if build is complete
  if (entry->buildComplete) {
    return entry;
  }

  // If this is the builder task, don't wait - all drivers of the builder task
  // should proceed to build (they coordinate via JoinBridge, not here).
  if (entry->builderTaskId == taskId) {
    return entry;
  }

  auto [promise, _future] =
      makeVeloxContinuePromiseContract(fmt::format("HashTableCache::{}", key));
  entry->buildPromises.push_back(std::move(promise));
  *future = std::move(_future);

  return entry;
}

void HashTableCache::put(
    const std::string& key,
    std::shared_ptr<BaseHashTable> table,
    bool hasNullKeys) {
  std::vector<ContinuePromise> promises;

  {
    std::lock_guard<std::mutex> guard(lock_);

    auto it = tables_.find(key);
    VELOX_CHECK(
        it != tables_.end(),
        "Cache entry for key '{}' must be created by get() before put()",
        key);

    auto& entry = it->second;
    VELOX_CHECK(!entry->buildComplete);
    VELOX_CHECK_NULL(entry->table);
    // Update the entry with the built table
    entry->table = std::move(table);
    entry->hasNullKeys = hasNullKeys;
    entry->buildComplete = true;

    // Collect promises to notify waiters
    promises = std::move(entry->buildPromises);
  }

  // Notify all waiting tasks outside the lock
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void HashTableCache::drop(const std::string& key) {
  std::shared_ptr<HashTableCacheEntry> entry;
  {
    std::lock_guard<std::mutex> guard(lock_);
    auto it = tables_.find(key);
    if (it != tables_.end()) {
      entry = std::move(it->second);
      tables_.erase(it);
    }
  }

  // Clear the table outside the lock to free memory before the entry
  // is destroyed. This ensures the tablePool's memory is released
  // before any parent pools are destroyed.
  if (entry) {
    entry->table.reset();
  }
}

} // namespace facebook::velox::exec
