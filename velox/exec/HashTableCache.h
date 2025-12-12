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

#include <memory>
#include <string>
#include <unordered_map>

#include <folly/Synchronized.h>

#include "velox/common/base/SpillConfig.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/Spill.h"

namespace facebook::velox::exec {

/// Represents a cached hash table with associated metadata.
/// Note: Cached hash tables do not support spilling - they must fit in
/// memory.
struct HashTableEntry {
  /// The actual hash table.
  std::shared_ptr<BaseHashTable> table;

  /// Leaf memory pool for the cached table (under query pool).
  /// This allows the table to outlive tasks while being a leaf pool
  /// (required for memory allocations).
  std::shared_ptr<memory::MemoryPool> tablePool;

  /// Flag indicating if the build side has null keys (for anti-join
  /// optimization).
  bool hasNullKeys{false};

  /// Indicates if the table build is complete.
  std::atomic<bool> buildComplete{false};

  /// Task ID of the task building this table.
  /// Set by the first task to arrive, used to distinguish builder from waiters.
  std::optional<std::string> builderTaskId;

  /// Promises waiting for the table build to complete.
  std::vector<ContinuePromise> buildPromises;

  /// Mutex protecting buildPromises and builderTaskId access.
  std::mutex promisesMutex;

  explicit HashTableEntry() = default;

  HashTableEntry(std::shared_ptr<BaseHashTable> _table, bool _hasNullKeys)
      : table(std::move(_table)),
        hasNullKeys(_hasNullKeys),
        buildComplete(true) {}
};

/// Global cache for hash tables that can be shared across
/// tasks within the same query and stage. This cache manages the lifecycle
/// of hash tables, allowing them to be built once by
/// the first task and reused by subsequent tasks, rather than being rebuilt
/// for each task.
///
/// IMPORTANT: Cached hash tables do not support spilling. They must fit
/// entirely in memory. This simplifies the implementation and aligns with
/// the typical use case where cached tables are smaller dimension tables.
///
/// The cache is managed at the query context level and provides:
/// - Thread-safe access to shared hash tables
/// - Explicit cleanup API for stage completion
/// - First-task-wins build coordination
class HashTableCache {
 public:
  /// Returns the singleton instance of the cache.
  static HashTableCache* getInstance();

  /// Retrieves a hash table from the cache. If the table doesn't
  /// exist, creates a new entry with a memory pool. The entry is returned
  /// regardless of whether the table has been built yet - callers should
  /// check the buildComplete flag.
  ///
  /// @param key The unique identifier for the table (queryId:planNodeId)
  /// @param queryPool The query-level memory pool (to create leaf pool if
  /// needed)
  /// @param taskId The task ID of the caller (for builder identification)
  /// @param future Output future to wait on if table is being built by another
  /// task
  /// @return The cache entry (always non-null) with pool and possibly table
  std::shared_ptr<HashTableEntry> getOrAwait(
      const std::string& key,
      memory::MemoryPool* queryPool,
      const std::string& taskId,
      ContinueFuture* future);

  /// Stores a newly built hash table in the cache.
  /// Note: Cached hash tables do not support spilling.
  ///
  /// @param key The unique identifier for the table (queryId:planNodeId)
  /// @param table The hash table to cache
  /// @param hasNullKeys Flag indicating if build has null keys
  void put(
      const std::string& key,
      std::shared_ptr<BaseHashTable> table,
      bool hasNullKeys);

  /// Removes a cached hash table entry.
  /// This is called when a query completes to release cached tables.
  ///
  /// @param cacheKey The cache key to flush (queryId:planNodeId)
  static void flushQuery(const std::string& cacheKey);

 private:
  HashTableCache() = default;

  // Synchronized map of cached tables.
  // Key format: "queryId:planNodeId"
  folly::Synchronized<
      std::unordered_map<std::string, std::shared_ptr<HashTableEntry>>>
      tables_;

  // Singleton instance.
  static std::atomic<HashTableCache*> instance_;
  static std::mutex instanceMutex_;
};

} // namespace facebook::velox::exec
