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
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/caching/CachedFactory.h"
#include "velox/common/caching/SimpleLRUCache.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/type/Type.h"

namespace facebook::nimble {

/// Wraps a shared_ptr<TabletReader> and its deserialized schemas so they can
/// be managed by CachedFactory (which requires unique_ptr ownership of the
/// value type). Both schemas are computed once at creation time and shared
/// across all consumers, avoiding repeated FlatBuffers deserialization and
/// nimble-to-velox type conversion.
struct CachedTabletReader {
  std::shared_ptr<TabletReader> tablet;
  /// Nimble-native schema from the file footer.
  std::shared_ptr<const facebook::nimble::Type> nimbleSchema;
  /// Velox type converted from the nimble schema. This is the base file schema
  /// before any per-consumer column name mapping (which depends on
  /// ReaderOptions and is applied at ReaderBase creation time).
  velox::RowTypePtr veloxSchema;
};

/// Process-wide cache for sharing TabletReader instances across multiple
/// consumers (e.g., NimbleIndexProjectors reading the same file). Uses
/// sharded system memory pools from MemoryManager for memory accounting,
/// independent of any task/query pool.
///
/// Thread-safe. Concurrent lookups for the same filename only create the
/// TabletReader once (deduplication via CachedFactory's pending set).
///
/// The cache assumes that the same filename always uses compatible
/// TabletReader::Options. On cache hit, the stored options from the first
/// creation are used; the caller's options are ignored.
class TabletReaderCache {
 public:
  struct Options {
    /// Number of cache shards. Each shard has its own system memory pool
    /// and mutex. Must be a power of 2.
    uint32_t numShards{32};

    /// Maximum number of cached TabletReaders across all shards.
    size_t maxEntries{4'096};

    /// TTL in milliseconds. 0 means no expiration.
    size_t expireDurationMs{0};

    /// Executor for background IO during TabletReader construction
    /// (e.g., parallel metadata loading).
    std::shared_ptr<folly::Executor> executor;

    std::string toString() const {
      return fmt::format(
          "numShards={}, maxEntries={}, expireDuration={}, executor={}",
          numShards,
          maxEntries,
          velox::succinctMillis(expireDurationMs),
          executor != nullptr ? "set" : "null");
    }
  };

  /// Properties passed to the generator on cache miss.
  struct Properties {
    std::shared_ptr<velox::ReadFile> readFile;
    TabletReader::Options tabletOptions;
  };

  explicit TabletReaderCache(const Options& options);

  /// Returns a cached or newly created CachedTabletReader for the given file.
  /// Uses readFile->getName() as the cache key. On cache miss, creates the
  /// TabletReader and deserializes the schema using a sharded system pool and
  /// the provided readFile/options.
  CachedTabletReader get(
      const std::shared_ptr<velox::ReadFile>& readFile,
      const TabletReader::Options& tabletOptions);

  /// Returns cache statistics (hits, misses, evictions, etc.).
  velox::SimpleLRUCacheStats stats();

  /// Initializes the process-wide TabletReaderCache singleton. Must be called
  /// once before getInstance(). Throws if called more than once.
  static void initialize(const Options& options);

  /// Returns the process-wide TabletReaderCache singleton. Must call
  /// initialize() first.
  static TabletReaderCache& getInstance();

  /// Resets the singleton to uninitialized state. Test-only.
  static void testingReset();

  /// Looks up a cached entry by filename without creating on miss. Test-only.
  std::optional<CachedTabletReader> testingGet(const std::string& filename);

 private:
  class Generator {
   public:
    Generator(
        std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools,
        std::shared_ptr<folly::Executor> executor);

    std::unique_ptr<CachedTabletReader> operator()(
        const std::string& filename,
        const Properties* properties,
        void* stats);

   private:
    const std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools_;
    const std::shared_ptr<folly::Executor> executor_;
    const uint32_t shardMask_;
  };

  struct Sizer {
    int64_t operator()(const CachedTabletReader& /*entry*/) const {
      return 1;
    }
  };

  using LRUCache = velox::SimpleLRUCache<std::string, CachedTabletReader>;

  using Factory = velox::CachedFactory<
      std::string,
      CachedTabletReader,
      Generator,
      Properties,
      void,
      Sizer>;

  static Factory createFactory(const Options& opts);

  Factory factory_;
};

} // namespace facebook::nimble
