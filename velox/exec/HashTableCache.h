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
#pragma once

#include <atomic>
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
  /// Leaf pool the cached table is allocated from. Immutable for the entry's
  /// lifetime: drop() removes the entry rather than emptying it, so readers
  /// need no synchronisation. The pool is destroyed once the cache and every
  /// operator holding this entry have released it.
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

  /// Stores a built hash table in an entry created by get() and notifies
  /// waiting tasks.
  void put(
      const std::string& key,
      std::shared_ptr<BaseHashTable> table,
      bool hasNullKeys);

  /// Removes a cache entry.
  void drop(const std::string& key);

  /// Adds an externally built hash table. This is used when an external system
  /// (e.g., Gluten) pre-builds a hash table and wants Velox tasks to reuse it
  /// via HashBuild's cache path.
  ///
  /// NOTE: Unlike put(), this does not complete an entry reserved by get().
  /// Instead, it creates a complete entry for an externally built table, so the
  /// key must not already exist in the cache.
  std::shared_ptr<HashTableCacheEntry> add(
      const std::string& key,
      std::shared_ptr<BaseHashTable> table,
      bool hasNullKeys,
      std::shared_ptr<memory::MemoryPool> tablePool);

  /// Returns true if a table exists and build is complete for the given key.
  bool exist(const std::string& key);

 private:
  HashTableCache() = default;

  std::mutex lock_;
  std::unordered_map<std::string, std::shared_ptr<HashTableCacheEntry>> tables_;

  // Distinguishes the leaf pool names of successive entries for one key. Every
  // task of a query shares its query pool, so an entry created after a previous
  // one for the same key was dropped would otherwise ask for a leaf child name
  // that the previous pool still holds, and addLeafChild() rejects duplicates.
  std::atomic_uint64_t tablePoolId_{0};
};

} // namespace facebook::velox::exec
