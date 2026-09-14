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
#include <span>
#include <string_view>
#include <vector>

#include <fmt/core.h>

#include "velox/common/file/File.h"
#include "velox/common/file/Region.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::cache {
class AsyncDataCache;
class CachePin;
} // namespace facebook::velox::cache

namespace facebook::nimble {

/// Base class for data I/O with an enqueue + batch load pattern.
///
/// Reads are organized into groups (one per stripe). Within each group,
/// reads are sorted by file offset for coalescing. Across groups,
/// relative order is preserved (stripes are already in file order).
/// After load(), bufferRef() provides zero-copy access to loaded data.
///
/// Implementations:
///   - DirectDataInput: coalesced I/O, no caching
///   - CachedDataInput: whole-group caching through AsyncDataCache
class DataInput {
 public:
  using Region = velox::common::Region;

  /// Reference to a loaded buffer region. Points into a contiguous
  /// allocation.
  struct BufferRef {
    const char* data{nullptr};
    uint64_t length{0};
    /// Index of the canonical BufferRef for this loaded region.
    /// Unique regions refer to themselves (`canonicalIndex == i`). Exact
    /// duplicate regions refer to the first enqueued copy and share that copy's
    /// `data` pointer, allowing callers to reuse `bufferRef(canonicalIndex)`
    /// instead of emitting the duplicate again. Only valid after load().
    uint32_t canonicalIndex{0};
  };

  /// Opaque handle keeping all loaded allocations alive.
  /// Callers must hold this until done with all BufferRefs.
  using Handle = std::shared_ptr<void>;

  virtual ~DataInput() = default;

  /// Pre-allocate internal storage for 'numRegions' enqueued reads.
  /// Avoids reallocation during enqueue().
  virtual void reserve(uint32_t numRegions) = 0;

  /// Starts a new group (one per stripe). Reads within a group are sorted by
  /// offset; across groups, relative order is preserved. `groupRegion` is the
  /// full physical region used by whole-group cache implementations. It is
  /// required by CachedDataInput and ignored by DirectDataInput.
  virtual void startGroup(std::optional<Region> groupRegion = std::nullopt) = 0;

  /// Enqueue a read in the current group. Returns an index for
  /// bufferRef() after load().
  virtual uint32_t enqueue(Region region) = 0;

  /// Load all enqueued requests via coalesced I/O. Returns a handle
  /// that keeps the loaded data alive. Caller must hold this until
  /// done with all BufferRefs (e.g., capture in IOBuf destructors).
  virtual Handle load() = 0;

  /// Get the loaded buffer for a previously enqueued request. The returned
  /// BufferRef::canonicalIndex identifies the stored copy when the region is an
  /// exact duplicate of an earlier enqueued request (see BufferRef). Only valid
  /// after load().
  virtual const BufferRef& bufferRef(uint32_t index) const = 0;

  /// Release all state (loaded buffers, enqueued requests).
  virtual void clear() = 0;
};

/// Direct data loading with IO coalescing and alignment for direct I/O.
///
/// After coalescing, each I/O group is backed by a non-contiguous
/// Allocation (page-aligned runs). Reads are submitted through the file's
/// positioned vector read API.
///
/// NOTE: DirectDataInput is not thread-safe. Each thread must use its
/// own instance.
class DirectDataInput : public DataInput {
 public:
  static constexpr int32_t kDefaultCoalesceDistance{4 << 10};
  static constexpr int32_t kMaxCoalesceDistance{256 << 10};

  struct Options {
    velox::memory::MemoryPool* pool{nullptr};
    std::shared_ptr<velox::io::IoStatistics> ioStats;
    int32_t maxCoalesceDistance{kDefaultCoalesceDistance};
    int64_t maxCoalesceBytes{velox::io::ReaderOptions::kDefaultCoalesceBytes};
  };

  DirectDataInput(velox::ReadFile* file, const Options& options);

  void reserve(uint32_t numRegions) override;

  void startGroup(std::optional<Region> groupRegion = std::nullopt) override;

  uint32_t enqueue(Region region) override;

  Handle load() override;

  const BufferRef& bufferRef(uint32_t index) const override;

  void clear() override;

  enum class State { kInit, kEnqueuing, kLoaded };

  static std::string_view stateName(State state);

 private:
  struct EnqueuedRegion {
    bool sameRegionAs(const EnqueuedRegion& other) const {
      return region.offset == other.region.offset &&
          region.length == other.region.length;
    }

    uint32_t enqueueIndex;
    Region region;
  };

  // A coalesced I/O operation covering one or more enqueued regions.
  struct IoGroup {
    // Aligned file offset and size for the coalesced read.
    uint64_t readOffset{0};
    uint64_t readSize{0};
    // Unique physical bytes requested by the logical regions, excluding
    // exact duplicate ranges and alignment/coalescing overread.
    uint64_t payloadSize{0};
    // Byte offset within the shared read buffer.
    uint64_t bufferOffset{0};
    std::span<const EnqueuedRegion> regions;
  };

  // Coalesces sorted regions into physical IO groups. Exact duplicate ranges
  // can share one physical read when the caller maps them to the same bytes.
  std::vector<IoGroup> computeIoGroups(
      const std::vector<EnqueuedRegion>& sortedRegions);

  // Computes the unique payload size for a sorted span of regions. Exact
  // duplicate ranges do not add payload bytes. Returns {payloadSize, lastEnd}
  // where lastEnd is the end offset of the last accepted range.
  static std::pair<uint64_t, uint64_t> computePayloadSize(
      std::span<const EnqueuedRegion> sortedRegions);

  // Computes total read and payload bytes, and sets each group's buffer offset
  // in the shared read buffer. Returns {readBytes, payloadBytes}.
  static std::pair<uint64_t, uint64_t> computeIoSizes(
      std::vector<IoGroup>& ioGroups);

  // Points each logical buffer ref at its slice inside the shared read buffer.
  void populateBufferRefs(
      const std::vector<IoGroup>& ioGroups,
      const char* buffer);

  // Allocates the shared aligned read buffer and returns a handle that frees
  // it.
  std::pair<char*, Handle> allocateBuffer(uint64_t bytes);

  // Executes all physical IO groups through ReadFile's positioned vector API.
  void executeIoGroups(
      std::vector<IoGroup>& ioGroups,
      char* buffer,
      uint64_t bufferSize);

  // File to read from.
  velox::ReadFile* const file_;
  // Memory pool for allocating the read buffer.
  velox::memory::MemoryPool* const pool_;
  // IO statistics for tracking read bytes, latency, and gaps.
  const std::shared_ptr<velox::io::IoStatistics> ioStats_;
  // I/O alignment for file offsets and read sizes.
  const uint64_t alignment_;
  // Allocation alignment passed to allocateAligned, clamped to
  // MemoryAllocator::kMinAlignment.
  const uint64_t allocAlignment_;
  // Max gap in bytes between regions to bridge by coalescing.
  const int32_t maxCoalesceDistance_;
  // Max total bytes per coalesced I/O group.
  const int64_t maxCoalesceBytes_;

  State state_{State::kInit};

  // --- Enqueue state (flat CSR layout) ---
  // All enqueued regions in a single flat vector. groupOffsets_[i] marks
  // where group i begins; regions within each group are sorted by offset
  // during load().
  std::vector<EnqueuedRegion> regions_;
  // Start index in regions_ for each group.
  std::vector<uint32_t> groupOffsets_;
  // Populated by load() with pointers into the aligned buffer.
  std::vector<BufferRef> bufferRefs_;
};

/// Loads and caches one contiguous entry per group. Enqueued regions return
/// zero-copy slices into their group's cached entry.
///
/// NOTE: CachedDataInput is not thread-safe. Each thread must use its own
/// instance, while the underlying AsyncDataCache may be shared.
class CachedDataInput final : public DataInput {
 public:
  /// Configures the cache, source-file identity, and memory accounting.
  struct Options {
    /// Pool used for direct-I/O staging buffers.
    velox::memory::MemoryPool* pool{nullptr};
    /// Shared cache receiving one entry for each loaded group.
    velox::cache::AsyncDataCache* cache{nullptr};
    /// Stable identity of the source file in the cache key space.
    uint64_t fileId{0};
    /// Statistics receiving storage-read and memory-hit counters.
    std::shared_ptr<velox::io::IoStatistics> ioStats;
  };

  /// Constructs a grouped cache reader for `file`.
  CachedDataInput(velox::ReadFile* file, const Options& options);

  /// Reserves storage for the requested stream regions.
  void reserve(uint32_t numRegions) override;

  /// Starts a cache group. `groupRegion` must contain the complete physical
  /// byte range to cache, including bytes not requested by enqueue().
  void startGroup(std::optional<Region> groupRegion) override;

  /// Adds a requested stream region to the current cache group.
  uint32_t enqueue(Region region) override;

  /// Loads all groups and returns ownership of their cache pins. BufferRef
  /// pointers remain valid until the returned handle is released.
  Handle load() override;

  /// Returns the loaded bytes for a previously enqueued stream region.
  const BufferRef& bufferRef(uint32_t index) const override;

  /// Clears request state without releasing pins owned by a returned handle.
  void clear() override;

 private:
  struct EnqueuedRegion {
    // Position of this request in bufferRefs_.
    uint32_t enqueueIndex{0};
    // Logical file range requested by the caller.
    Region region;
  };

  struct Group {
    // Complete physical region stored in one cache entry.
    Region groupRegion;
    // First request for this group in regions_.
    uint32_t enqueuedRegionOffset{0};
  };

  struct CacheMissGroup {
    // Position shared by groups_ and the vector of cache pins.
    size_t groupIndex{0};
    // Physical range submitted to storage, including direct-I/O alignment.
    Region readRegion;
    // Offset of this read's destination in the aligned staging allocation.
    uint64_t stagingBufferOffset{0};
    // Distinct bytes requested from this group by the caller.
    uint64_t payloadBytes{0};
  };

  // Returns the number of distinct requested bytes in the group.
  uint64_t populateBufferRefs(
      const Group& group,
      uint32_t endRegion,
      const char* buffer);

  // Loads cache-miss groups in one positioned read batch.
  void loadCacheMissGroups(
      std::span<const CacheMissGroup> cacheMissGroups,
      uint64_t stagingBufferSize,
      const std::vector<velox::cache::CachePin>& cachePins);

  // File to read from on a cache miss.
  velox::ReadFile* const file_;
  // Memory pool for allocating direct-I/O staging buffers.
  velox::memory::MemoryPool* const pool_;
  // Shared cache used for full-group entries.
  velox::cache::AsyncDataCache* const cache_;
  // Stable file identity used in cache keys.
  const uint64_t fileId_;
  // IO statistics for storage reads, cache hits, and logical bytes.
  const std::shared_ptr<velox::io::IoStatistics> ioStats_;
  // Required alignment for file offsets, read sizes, and destination buffers.
  const uint64_t alignment_;
  // Allocation alignment accepted by the memory pool.
  const uint64_t allocationAlignment_;

  // True between load() and clear().
  bool loaded_{false};
  // Requested stream regions in enqueue order.
  std::vector<EnqueuedRegion> regions_;
  // Cache groups in increasing physical file-offset order.
  std::vector<Group> groups_;
  // Loaded zero-copy slices corresponding to regions_.
  std::vector<BufferRef> bufferRefs_;
};

} // namespace facebook::nimble

template <>
struct fmt::formatter<facebook::nimble::DirectDataInput::State>
    : fmt::formatter<std::string_view> {
  auto format(
      facebook::nimble::DirectDataInput::State state,
      fmt::format_context& ctx) const {
    return fmt::formatter<std::string_view>::format(
        facebook::nimble::DirectDataInput::stateName(state), ctx);
  }
};
