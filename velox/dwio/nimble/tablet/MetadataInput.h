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

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <optional>
#include "folly/Range.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/file/File.h"
#include "velox/common/file/Region.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/BlockCache.h"
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
///   - CachedMetadataInput: IO coalescing + BlockCache for decompressed
///     metadata
class MetadataInput {
 public:
  struct Options {
    velox::memory::MemoryPool* pool;

    /// Receives the IO counters. rawBytesRead counts each section that load()
    /// returns at its size in the file. It counts sections served from a cache
    /// too, as velox counts every byte a reader touches.
    std::shared_ptr<velox::io::IoStatistics> ioStats;
    int32_t maxCoalesceDistance{
        velox::io::ReaderOptions::kDefaultCoalesceDistance};
    int64_t maxCoalesceBytes{velox::io::ReaderOptions::kDefaultCoalesceBytes};

    /// Runs the coalesced reads of a load in parallel when set. A direct-I/O
    /// load of CachedMetadataInput is one batched read instead.
    folly::Executor* executor{nullptr};

    /// fileHandle->uuid is the id of the file in the keys of 'blockCache', so
    /// 'blockCache' requires fileHandle. Files whose bytes differ, including
    /// files that reuse a path, must use different ids. Only create() reads the
    /// handle, so it need not outlive create(). The input keeps a copy of the
    /// id, so the id stays registered for as long as the input can create cache
    /// entries, which the BlockCache from getBlockCache() needs.
    const velox::FileHandle* fileHandle{nullptr};

    /// When set, metadata entries above this size in bytes bypass
    /// 'blockCache'. A lookup in 'blockCache' that does not know the entry size
    /// still returns an entry above it that another input cached:
    /// findCachedMetadata(), and the load of a section whose uncompressed size
    /// the file does not record.
    std::optional<uint32_t> maxCacheEntrySize{};
    velox::FileIoContext fileIoContext{};

    /// When set, creates CachedMetadataInput, which caches decompressed
    /// metadata here. Otherwise creates DirectMetadataInput. Must outlive the
    /// input and every MetadataBuffer it returns, including clones and slices,
    /// since each may pin a cache entry.
    BlockCache* blockCache{nullptr};
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
  /// Counts the found bytes as raw bytes read. Requires cached().
  virtual std::unique_ptr<MetadataBuffer> findCachedMetadata(
      uint64_t /*offset*/) {
    NIMBLE_UNREACHABLE("findCachedMetadata requires CachedMetadataInput");
  }

  /// Raw tracked read for bootstrap IO (e.g. speculative footer read)
  /// before section boundaries are known. Records bytes and latency into
  /// the same ioStats used by load().
  void readRaw(uint64_t offset, uint64_t size, void* dest);

  /// Caches data at the given offset from multiple contiguous ranges.
  /// Ranges are written sequentially into a single cache entry. Requires
  /// cached().
  virtual void cacheMetadata(
      uint64_t /*cacheOffset*/,
      std::span<const std::string_view> /*ranges*/) {
    NIMBLE_UNREACHABLE("cacheMetadata requires CachedMetadataInput");
  }

 protected:
  MetadataInput(velox::ReadFile* file, const Options& options);

  struct LoadedSection {
    explicit LoadedSection(MetadataSection _section) : section{_section} {}

    MetadataSection section;
    std::shared_ptr<MetadataBuffer> buffer;
  };

  // Per-section read state produced by prepareBuffers().
  // buffers[i] owns the allocated memory.
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

  /// Decompresses the buffer into a MetadataBuffer. For uncompressed data,
  /// passes the buffer directly (zero copy).
  std::shared_ptr<MetadataBuffer> decompressAndStore(
      const MetadataSection& section,
      velox::BufferPtr&& buffer);

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
  void executeIoGroups(std::vector<IoGroup>& ioGroups) const;

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
  const velox::FileIoContext fileIoContext_;
};

/// Direct metadata loading with IO coalescing, no caching.
class DirectMetadataInput : public MetadataInput {
 public:
  std::vector<std::shared_ptr<MetadataBuffer>> load(
      std::span<const MetadataSection> sections) override;

 private:
  void prepareBuffers(
      const std::vector<LoadedSection>& sections,
      ReadBuffers& readBuffers);

  friend class MetadataInput;

  DirectMetadataInput(velox::ReadFile* file, const Options& options);
};

/// Metadata input that caches decompressed sections in Options::blockCache,
/// keyed by (Options::fileHandle->uuid, compressed file offset). A section
/// whose uncompressed size the file does not record is cached after its read,
/// at the size its compressed bytes start with. An empty section, and a section
/// above Options::maxCacheEntrySize, bypass the cache. A section too large for
/// any cache entry means a corrupt file, and fails. Without direct I/O, reads
/// an uncompressed section that misses straight into its new cache entry, so
/// caching it takes no copy.
///
/// Reads the sections that miss the way DataInput reads stripe data. Each
/// load() coalesces the sections it must read from the file, aligns the reads
/// for direct I/O, and submits them as one positioned vector read, which a file
/// backed by RocksDB serves with a single MultiRead. Each section then gets its
/// own allocation, so no read padding outlives the load. Without direct I/O,
/// each merged range is instead one scatter read into the buffers of its
/// sections, and Options::executor runs them in parallel. A file without a
/// native batched preadv() reads the regions of a direct-I/O load one after
/// another.
class CachedMetadataInput final : public MetadataInput {
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

 private:
  // A merged read covering the sections at positions [begin, end) of the
  // indices passed to coalesce().
  struct ReadGroup {
    // File range to read, aligned for direct I/O, so it can run past the end of
    // the file.
    velox::common::Region region;

    // Offset of this read's bytes in the shared read buffer.
    uint64_t destBufferOffset{0};

    // Position of the first section this read covers in the indices passed to
    // coalesce().
    size_t begin{0};

    // Position one past the last section this read covers.
    size_t end{0};
  };

  // Returns the sealed entry of 'section' on a hit. On a miss, returns a new
  // entry for the read to fill, or an empty entry when the section is cached
  // after the read, or not cached at all.
  BlockCache::Entry find(const MetadataSection& section) const;

  // Decompresses 'data', the bytes of 'section', into 'entry', a new entry, and
  // seals it.
  std::shared_ptr<MetadataBuffer> fill(
      const MetadataSection& section,
      std::string_view data,
      BlockCache::Entry& entry);

  // Caches 'section', whose uncompressed size the file does not record, now
  // that its bytes 'data' are read: they start with that size. Returns the
  // decompressed section.
  std::shared_ptr<MetadataBuffer> cacheSection(
      const MetadataSection& section,
      std::string_view data);

  // Returns the entry of 'size' bytes at 'offset', without counting an existing
  // entry as a RAM hit.
  BlockCache::Entry getOrCreateWithoutHit(uint64_t offset, uint32_t size) const;

  // Returns 'section', decompressed from 'data', its bytes in the file, and
  // caches it: in 'entry' when find() created one, or in an entry created now
  // when the file does not record the uncompressed size. A section that cannot
  // be cached is returned uncached, in 'readBuffer' itself when that holds
  // 'data' and the section is uncompressed.
  std::shared_ptr<MetadataBuffer> cacheSection(
      const MetadataSection& section,
      std::string_view data,
      BlockCache::Entry& entry,
      velox::BufferPtr readBuffer);

  // Reads sections[i] for each i in 'missIndices', which must be in file
  // offset order, in one positioned vector read, and sets buffers[i] to what
  // cacheSection() returns for it and newEntries[i].
  void loadDirectIo(
      std::span<const MetadataSection> sections,
      std::span<const uint32_t> missIndices,
      std::span<BlockCache::Entry> newEntries,
      std::span<std::shared_ptr<MetadataBuffer>> buffers);

  // Requires !directIo(). Like loadDirectIo(), but reads each merged range in
  // one scatter read, which skips the gaps between its sections.
  void loadBufferedIo(
      std::span<const MetadataSection> sections,
      std::span<const uint32_t> missIndices,
      std::span<BlockCache::Entry> newEntries,
      std::span<std::shared_ptr<MetadataBuffer>> buffers);

  // Returns 'data', compressed with 'compressionType', decompressed into a new
  // pool buffer.
  std::shared_ptr<MetadataBuffer> decompress(
      std::string_view data,
      CompressionType compressionType) const;

  // Returns 'data', compressed with 'compressionType', decompressed into a new
  // pool buffer, or 'data' itself when uncompressed.
  std::shared_ptr<MetadataBuffer> decompress(
      velox::BufferPtr data,
      CompressionType compressionType) const;

  // Returns true if reads from the file use direct I/O. A direct-I/O read
  // covers whole aligned blocks, so it cannot land in the buffer of a section.
  bool directIo() const {
    return alignment_ > 1;
  }

  // Requires !directIo(). Reads sections[i] for each i in 'sortedIndices',
  // which must be in file offset order, into destinations[i], which must hold
  // sections[i].size() bytes. Each merged range is one scatter read, which
  // skips the gaps between its sections. Options::executor, when set, runs
  // every scatter read but the first.
  void readSections(
      std::span<const MetadataSection> sections,
      std::span<const uint32_t> sortedIndices,
      std::span<const folly::Range<char*>> destinations) const;

  // Merges the reads of the sections at 'sortedIndices'. Reads that share an
  // aligned block always merge, so the returned reads never overlap.
  std::vector<ReadGroup> coalesce(
      std::span<const MetadataSection> sections,
      std::span<const uint32_t> sortedIndices) const;

  // Reads 'groups' into 'buffer' in one positioned vector read. 'payloadBytes'
  // is the section bytes they cover.
  void readGroups(
      std::span<const ReadGroup> groups,
      char* buffer,
      uint64_t payloadBytes) const;

  // Records the reads of one load: 'readBytes' bytes read from the file in
  // 'storageReadNs', of which 'payloadBytes' are section bytes.
  void recordReadStats(
      uint64_t storageReadNs,
      uint64_t readBytes,
      uint64_t payloadBytes) const;

  // Records the bytes of the reads of one load, whose latencies are recorded
  // separately: 'readBytes' bytes read from the file, of which 'payloadBytes'
  // are section bytes.
  void recordReadBytes(uint64_t readBytes, uint64_t payloadBytes) const;

  CachedMetadataInput(velox::ReadFile* file, const Options& options);

  // Holds the decompressed sections, and the metadata that cacheMetadata()
  // stores.
  BlockCache* const blockCache_;

  // Identity of the file in cache keys, held so that the id stays registered.
  const velox::StringIdLease fileId_;

  // Entries above this size bypass the cache.
  const std::optional<uint32_t> maxCacheEntrySize_;

  // Alignment of read offsets, buffers, and lengths; 1 without direct I/O.
  const uint64_t alignment_;

  friend class MetadataInput;
};

} // namespace facebook::nimble
