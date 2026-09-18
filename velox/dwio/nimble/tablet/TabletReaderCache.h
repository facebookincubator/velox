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

#include <functional>
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

/// One cached tablet: the TabletReader, its deserialized schemas, and the IO
/// statistics the reader writes its own metadata and index reads into.
///
/// Those statistics live here because the cache replaces the caller's
/// ReaderOptions on the miss path, so a caller can otherwise never observe
/// footer, stripe-group or cluster-index IO -- only the stripe data it reads
/// through its own DataInput.
///
/// Non-copyable and handed out by shared_ptr, so exactly one instance exists
/// per cached file however many readers hold it. `onRelease` fires exactly
/// once, when the cache and every holder have let go. It fires for entries
/// that never entered the cache too: when insertion fails, CachedFactory hands
/// back an owning pointer instead, and destruction still runs.
///
/// Holding `tablet()` alone does NOT keep the entry alive, and the tablet has
/// its own reference to the statistics below -- so a consumer outliving the
/// entry would go on doing metadata and index IO that no observer is watching
/// any more. Whatever retains the tablet has to retain the entry with it;
/// ReaderBase and NimbleIndexProjector do that by aliasing their tablet
/// pointer onto the entry.
class CachedTabletReader {
 public:
  CachedTabletReader(
      std::shared_ptr<TabletReader> tablet,
      std::shared_ptr<const facebook::nimble::Type> nimbleSchema,
      velox::RowTypePtr veloxSchema,
      std::shared_ptr<velox::io::IoStatistics> metadataIoStats,
      std::shared_ptr<velox::io::IoStatistics> indexIoStats,
      std::function<void(const CachedTabletReader&)> onRelease);

  ~CachedTabletReader();

  CachedTabletReader(const CachedTabletReader&) = delete;
  CachedTabletReader& operator=(const CachedTabletReader&) = delete;

  const std::shared_ptr<TabletReader>& tablet() const {
    return tablet_;
  }

  /// Nimble-native schema from the file footer.
  const std::shared_ptr<const facebook::nimble::Type>& nimbleSchema() const {
    return nimbleSchema_;
  }

  /// Velox type converted from the nimble schema. This is the base file schema
  /// before any per-consumer column name mapping (which depends on
  /// ReaderOptions and is applied at ReaderBase creation time).
  const velox::RowTypePtr& veloxSchema() const {
    return veloxSchema_;
  }

  const velox::io::IoStatistics& metadataIoStats() const {
    return *metadataIoStats_;
  }

  const velox::io::IoStatistics& indexIoStats() const {
    return *indexIoStats_;
  }

 private:
  const std::shared_ptr<TabletReader> tablet_;
  const std::shared_ptr<const facebook::nimble::Type> nimbleSchema_;
  const velox::RowTypePtr veloxSchema_;
  // The statistics the tablet writes its own footer, stripe-group and
  // cluster-index reads into. They are held here because the generator
  // replaces the caller's ReaderOptions, so the metadata and index statistics
  // a caller supplies are never written -- only its dataIoStats is, and only
  // because the projector reads stripe data through its own DataInput. Keeping
  // the pair on the entry is what makes this IO reachable for export.
  const std::shared_ptr<velox::io::IoStatistics> metadataIoStats_;
  const std::shared_ptr<velox::io::IoStatistics> indexIoStats_;
  const std::function<void(const CachedTabletReader&)> onRelease_;
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
///
/// No entry may outlive the cache. Each TabletReader frees its metadata
/// buffers back to a memory pool the cache owns, so destroying the cache while
/// an entry is still held is a use-after-free. The process-wide instance is
/// never destroyed, which is why this holds in production.
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

    /// Observe the lifetime of each cached tablet, which is the only way to
    /// reach the metadata and index IO it does. onCreate runs once the entry
    /// exists; onRelease once it is gone -- evicted, or dropped by its last
    /// holder if it never made it into the cache. Both default to unset, in
    /// which case that IO stays unobservable as before, and both must be set
    /// together.
    ///
    /// A throw from onCreate propagates to the caller. A throw from onRelease
    /// cannot: it is invoked from a destructor, so it is logged and swallowed.
    std::function<void(const CachedTabletReader&)> onCreate;
    std::function<void(const CachedTabletReader&)> onRelease;

    std::string toString() const {
      return fmt::format(
          "numShards={}, maxEntries={}, expireDuration={}, executor={}, "
          "lifetimeObserver={}",
          numShards,
          maxEntries,
          velox::succinctMillis(expireDurationMs),
          executor != nullptr ? "set" : "null",
          (onCreate != nullptr || onRelease != nullptr) ? "set" : "unset");
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
  std::shared_ptr<CachedTabletReader> get(
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
  std::shared_ptr<CachedTabletReader> testingGet(const std::string& filename);

 private:
  class Generator {
   public:
    Generator(
        std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools,
        std::shared_ptr<folly::Executor> executor,
        std::function<void(const CachedTabletReader&)> onCreate,
        std::function<void(const CachedTabletReader&)> onRelease);

    std::unique_ptr<std::shared_ptr<CachedTabletReader>> operator()(
        const std::string& filename,
        const Properties* properties,
        void* stats);

   private:
    const std::vector<std::shared_ptr<velox::memory::MemoryPool>> pools_;
    const std::shared_ptr<folly::Executor> executor_;
    const std::function<void(const CachedTabletReader&)> onCreate_;
    const std::function<void(const CachedTabletReader&)> onRelease_;
    const uint32_t shardMask_;
  };

  struct Sizer {
    int64_t operator()(
        const std::shared_ptr<CachedTabletReader>& /*entry*/) const {
      return 1;
    }
  };

  // The value is a shared_ptr so eviction only drops the cache's reference:
  // a reader still using the entry keeps it alive, and the entry is destroyed
  // -- firing onRelease -- when the last holder lets go. Holding the
  // CachedPtr instead would pin the entry, and once maxEntries slots are
  // pinned SimpleLRUCache refuses new inserts and tablets stop being shared.
  using LRUCache =
      velox::SimpleLRUCache<std::string, std::shared_ptr<CachedTabletReader>>;

  using Factory = velox::CachedFactory<
      std::string,
      std::shared_ptr<CachedTabletReader>,
      Generator,
      Properties,
      void,
      Sizer>;

  static Factory createFactory(const Options& opts);

  Factory factory_;
};

} // namespace facebook::nimble
