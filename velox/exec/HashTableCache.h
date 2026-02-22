/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "velox/exec/HashTable.h"

namespace facebook::velox::core {
class QueryCtx;
}

namespace facebook::velox::exec {

/// Cached hash table entry with build coordination metadata.
struct HashTableCacheEntry {
  HashTableCacheEntry(
      std::string _cacheKey,
      std::string _builderTaskId,
      std::shared_ptr<memory::MemoryPool> _tablePool)
      : cacheKey(std::move(_cacheKey)),
        builderTaskId(std::move(_builderTaskId)),
        tablePool(std::move(_tablePool)) {}

  const std::string cacheKey;
  const std::string builderTaskId;
  const std::shared_ptr<memory::MemoryPool> tablePool;
  std::shared_ptr<BaseHashTable> table;
  bool hasNullKeys{false};
  tsan_atomic<bool> buildComplete{false};
  std::vector<ContinuePromise> buildPromises;
};

/// Global cache for hash tables shared across tasks within the same query.
/// First task builds the table, subsequent tasks wait and reuse it.
class HashTableCache {
 public:
  static HashTableCache* instance();

  /// Gets or creates a cache entry. First caller becomes the builder.
  /// Subsequent callers from different tasks get a future to wait on.
  /// When a new entry is created, a release callback is registered on queryCtx
  /// to clean up the entry when the query completes.
  /// @param future Must be non-null; set if caller needs to wait.
  std::shared_ptr<HashTableCacheEntry> get(
      const std::string& key,
      const std::string& taskId,
      core::QueryCtx* queryCtx,
      ContinueFuture* future);

  /// Stores a built hash table and notifies waiting tasks.
  void put(
      const std::string& key,
      std::shared_ptr<BaseHashTable> table,
      bool hasNullKeys);

  /// Removes a cache entry.
  void drop(const std::string& key);

 private:
  HashTableCache() = default;

  std::mutex lock_;
  std::unordered_map<std::string, std::shared_ptr<HashTableCacheEntry>> tables_;
};

} // namespace facebook::velox::exec
