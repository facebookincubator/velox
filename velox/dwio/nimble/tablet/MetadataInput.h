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
#include <span>
#include <vector>

#include <optional>
#include "folly/Range.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/file/File.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::velox {
struct FileHandle;
} // namespace facebook::velox

namespace facebook::nimble {

/// Reads metadata sections from a Nimble file with IO coalescing.
///
/// Stateless, thread-safe API: load() takes a span of sections and returns
/// decompressed MetadataBuffers. All per-call state is local — no member
/// state is mutated during load(), so concurrent calls are safe.
///
/// IO coalescing is controlled by maxCoalesceDistance and maxCoalesceBytes
/// from io::ReaderOptions, shared with data IO settings.
///
/// Two implementations:
///   - DirectMetadataInput: IO coalescing, no caching
///   - CachedMetadataInput: IO coalescing + AsyncDataCache for decompressed
///     metadata with memory/SSD tier support
class MetadataInput {
 public:
  struct Options {
    velox::memory::MemoryPool* pool;
    std::shared_ptr<velox::io::IoStatistics> ioStats;
    int32_t maxCoalesceDistance{
        velox::io::ReaderOptions::kDefaultCoalesceDistance};
    int64_t maxCoalesceBytes{velox::io::ReaderOptions::kDefaultCoalesceBytes};
    folly::Executor* executor{nullptr};
    // When both fileHandle and cache are set, creates CachedMetadataInput.
    const velox::FileHandle* fileHandle{nullptr};
    velox::cache::AsyncDataCache* cache{nullptr};
    /// When set, metadata entries above this size in bytes bypass the cache.
    std::optional<uint32_t> maxCacheEntrySize{};
  };

  static std::unique_ptr<MetadataInput> create(
      velox::ReadFile* file,
      const Options& options);

  virtual ~MetadataInput() = default;

  /// Returns true if this MetadataInput is backed by a cache.
  virtual bool cached() const {
    return false;
  }

  /// Loads metadata sections with coalesced IO. Returns one MetadataBuffer
  /// per input section, in the same order. Thread-safe: all per-call state
  /// is local.
  virtual std::vector<std::shared_ptr<MetadataBuffer>> load(
      std::span<const MetadataSection> sections) = 0;

  /// Tries to find cached metadata at the given offset. Returns a
  /// MetadataBuffer holding a cache pin if found, nullptr otherwise.
  /// DirectMetadataInput always returns nullptr.
  virtual std::unique_ptr<MetadataBuffer> findCachedMetadata(uint64_t offset);

  /// Raw tracked read for bootstrap IO (e.g. speculative footer read)
  /// before section boundaries are known. Records bytes and latency into
  /// the same ioStats used by load().
  void readRaw(uint64_t offset, uint64_t size, void* dest);

  /// Caches data at the given offset from multiple contiguous ranges.
  /// Ranges are written sequentially into a single cache entry.
  virtual void cacheMetadata(
      uint64_t cacheOffset,
      std::span<const std::string_view> ranges);

 protected:
  MetadataInput(velox::ReadFile* file, const Options& options);

  struct LoadedSection {
    explicit LoadedSection(MetadataSection _section) : section{_section} {}

    MetadataSection section;
    std::shared_ptr<MetadataBuffer> buffer;
  };

  // Per-section read state produced by prepareBuffers().
  // buffers[i] owns the allocated memory (null when reading into cache pin).
  // readRanges[i] is the writable destination for IO.
  struct ReadBuffers {
    std::vector<velox::BufferPtr> buffers;
    std::vector<folly::Range<char*>> readRanges;
  };

  /// Sorts by file offset internally, computes coalesced IO groups,
  /// and executes preadv. Does not reorder readRanges — the caller's
  /// buffers[i] ↔ loadIndices[i] correspondence is preserved.
  void loadFromFile(
      const std::vector<LoadedSection>& sections,
      const std::vector<uint32_t>& loadIndices,
      const std::vector<folly::Range<char*>>& readRanges);

  /// Overload for loading all sections (sequential indices).
  void loadFromFile(
      const std::vector<LoadedSection>& sections,
      const std::vector<folly::Range<char*>>& readRanges);

  /// Decompresses the buffer and calls store(). For uncompressed data,
  /// passes the buffer directly (zero copy).
  std::shared_ptr<MetadataBuffer> decompressAndStore(
      const MetadataSection& section,
      velox::BufferPtr&& buffer);

  /// Subclass creates the MetadataBuffer from a decompressed buffer.
  virtual std::shared_ptr<MetadataBuffer> store(
      const MetadataSection& section,
      velox::BufferPtr&& decompressed) = 0;

  /// Extracts results from loaded sections into a vector of MetadataBuffers.
  static std::vector<std::shared_ptr<MetadataBuffer>> extractResults(
      std::vector<LoadedSection>&& sections);

  struct IoGroup {
    uint64_t offset;
    std::vector<folly::Range<char*>> ranges;
  };

  // Computes IO group boundaries using coalesceIo, then builds preadv
  // ranges for each group with null-data ranges for gaps.
  // 'readRanges' provides the writable destination for each section.
  std::vector<IoGroup> computeIoGroups(
      const std::vector<LoadedSection>& sections,
      const std::vector<uint32_t>& loadIndices,
      const std::vector<folly::Range<char*>>& readRanges);

  // Dispatches IO groups in parallel via AsyncSource when executor is
  // available, records IO stats, and propagates the first error.
  void executeIoGroups(std::vector<IoGroup>& ioGroups);

  // Returns the uncompressed size, or nullopt if unknown (old files
  // without uncompressed_size — will be deprecated).
  static std::optional<uint32_t> resolveUncompressedSize(
      const MetadataSection& section);

  // Records coalesced IO stats (payload, total read, overread).
  void recordCoalescedIoStats(const velox::CoalesceIoStats& ioStats);

  // RAII guard that measures elapsed time and records it as query thread
  // IO latency when the scope exits.
  class QueryIoLatencyGuard {
   public:
    explicit QueryIoLatencyGuard(velox::io::IoStatistics* ioStats)
        : ioStats_{ioStats}, timer_{&usecs_} {}

    ~QueryIoLatencyGuard() {
      ioStats_->queryThreadIoLatencyUs().increment(usecs_);
    }

   private:
    velox::io::IoStatistics* const ioStats_;
    uint64_t usecs_{0};
    velox::MicrosecondTimer timer_;
  };

  velox::ReadFile* const file_;
  velox::memory::MemoryPool* const pool_;
  // Max gap in bytes between consecutive sections to merge into one IO.
  const int32_t maxCoalesceDistance_;
  // Max total bytes per coalesced IO batch.
  const int64_t maxCoalesceBytes_;
  folly::Executor* const executor_;
  const std::shared_ptr<velox::io::IoStatistics> ioStats_;
};

/// Direct metadata loading with IO coalescing, no caching.
class DirectMetadataInput : public MetadataInput {
 public:
  std::vector<std::shared_ptr<MetadataBuffer>> load(
      std::span<const MetadataSection> sections) override;

 protected:
  std::shared_ptr<MetadataBuffer> store(
      const MetadataSection& section,
      velox::BufferPtr&& decompressed) override;

 private:
  void prepareBuffers(
      const std::vector<LoadedSection>& sections,
      ReadBuffers& readBuffers);

  friend class MetadataInput;

  DirectMetadataInput(velox::ReadFile* file, const Options& options);
};

/// Cached metadata loading with IO coalescing and AsyncDataCache.
/// Checks memory cache and SSD cache before file IO.
/// Caches decompressed metadata in contiguous cache entries.
/// Cache key is the compressed file offset.
class CachedMetadataInput : public MetadataInput {
 public:
  bool cached() const override {
    return true;
  }

  std::vector<std::shared_ptr<MetadataBuffer>> load(
      std::span<const MetadataSection> sections) override;

  std::unique_ptr<MetadataBuffer> findCachedMetadata(uint64_t offset) override;

  void cacheMetadata(
      uint64_t cacheOffset,
      std::span<const std::string_view> ranges) override;

 protected:
  std::shared_ptr<MetadataBuffer> store(
      const MetadataSection& section,
      velox::BufferPtr&& decompressed) override;

 private:
  // Checks memory cache for all sections. Returns indices of sections
  // not found in cache. Pre-claims exclusive pins for known-size misses.
  std::vector<uint32_t> loadFromCache(
      std::vector<LoadedSection>& sections,
      std::vector<std::optional<velox::cache::CachePin>>& cachePins);

  // Checks SSD cache for memory misses, batch-loads SSD hits into
  // memory cache. Removes loaded indices from loadIndices in-place.
  void loadFromSsd(
      std::vector<LoadedSection>& sections,
      std::vector<uint32_t>& loadIndices,
      std::vector<std::optional<velox::cache::CachePin>>& cachePins);

  // Decompresses loaded buffers and stores results into sections.
  // For sections with cache pins, decompresses into the pin and promotes
  // to shared. For sections without pins, uses decompressAndStore.
  void processLoadedBuffers(
      const std::vector<uint32_t>& loadIndices,
      std::vector<LoadedSection>& sections,
      std::vector<std::optional<velox::cache::CachePin>>& cachePins,
      ReadBuffers& readBuffers);

  // For uncompressed sections with cache pins, reads directly into the
  // pin's contiguous data. Otherwise allocates a temp buffer.
  void prepareBuffers(
      const std::vector<LoadedSection>& sections,
      const std::vector<uint32_t>& loadIndices,
      const std::vector<std::optional<velox::cache::CachePin>>& cachePins,
      ReadBuffers& readBuffers);

  // Returns whether an entry of 'size' is eligible for this cache.
  bool cacheable(uint64_t size) const;

  // Waits for a cache pin to become available via findOrCreate.
  // Blocks until the pin is no longer exclusively held by another thread.
  velox::cache::CachePin acquireCachePin(
      const velox::cache::RawFileCacheKey& key,
      uint32_t size);

  // If the pin is shared (cache hit), validates the entry, records ramHit
  // stats, and returns the MetadataBuffer. Returns nullptr on cache miss.
  std::shared_ptr<MetadataBuffer> tryCacheHit(
      uint32_t expectedSize,
      velox::cache::CachePin& pin);

  // Validates an exclusive cache pin entry, promotes it to shared, and
  // returns a MetadataBuffer. Records the given IO counter stats.
  std::shared_ptr<MetadataBuffer> promoteCachePin(
      const MetadataSection& section,
      velox::cache::CachePin pin,
      velox::io::IoCounter& ioCounter);

  friend class MetadataInput;

  CachedMetadataInput(
      velox::ReadFile* file,
      velox::StringIdLease fileId,
      velox::cache::AsyncDataCache* cache,
      const Options& options);

  velox::cache::AsyncDataCache* const cache_;
  const velox::StringIdLease fileId_;
  const std::optional<uint32_t> maxCacheEntrySize_;
};

} // namespace facebook::nimble
