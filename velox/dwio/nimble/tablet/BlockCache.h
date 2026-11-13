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
#include <utility>

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatistics.h"

namespace facebook::velox::cache {
class AsyncDataCache;
} // namespace facebook::velox::cache

namespace facebook::nimble {

/// Stores contiguous byte ranges keyed by (fileId, offset): one entry per
/// CachedDataInput load group, and one per metadata section or footer that
/// CachedMetadataInput caches.
///
/// Exists so the same Input can run over velox's AsyncDataCache or a foreign
/// cache such as RocksDB's block cache, which differ in who allocates the
/// bytes: AsyncDataCache hands back a buffer it owns, while RocksDB expects the
/// caller to allocate and then transfer ownership on seal. Entry hides that
/// difference behind a writable pointer plus an explicit seal step.
///
/// Implementations must outlive every Entry they hand out.
class BlockCache {
 public:
  /// Pins one cache entry. Move-only; the pin is released on destruction, after
  /// which data() dangles.
  class Entry {
   public:
    Entry() = default;

    /// Constructed by BlockCache implementations. 'handle' is opaque and is
    /// passed back to BlockCache::release(). 'data' points at the entry's
    /// 'size' bytes. 'sealed' is false when this lookup created the entry, so
    /// the caller owns filling data() and calling BlockCache::seal() to make
    /// it visible to other readers.
    Entry(
        BlockCache* cache,
        void* handle,
        char* data,
        uint32_t size,
        bool sealed)
        : cache_{velox::checkedNotNull(cache)},
          handle_{handle},
          data_{data},
          size_{size},
          sealed_{sealed} {}

    Entry(Entry&& other) noexcept {
      *this = std::move(other);
    }

    Entry& operator=(Entry&& other) noexcept;

    Entry(const Entry&) = delete;
    Entry& operator=(const Entry&) = delete;

    ~Entry() {
      reset();
    }

    /// Start of the entry's bytes. Writable until seal(); read-only once the
    /// entry is sealed, either by this caller or by a prior cache hit.
    char* data() const {
      return data_;
    }

    /// Number of bytes at data(). On a cache hit this is the size the entry was
    /// created with, which can differ from the size requested.
    uint32_t size() const {
      return size_;
    }

    /// False when this lookup created the entry, so the caller owns filling it
    /// and sealing it. An entry left unsealed is discarded rather than cached.
    bool sealed() const {
      return sealed_;
    }

    /// True on a default-constructed or moved-from Entry. An empty Entry holds
    /// no cache pin; data(), size(), handle(), and sealed() are not meaningful
    /// on it.
    bool empty() const {
      return cache_ == nullptr;
    }

    /// Releases the pin and clears the entry.
    void reset() noexcept;

    /// Opaque implementation handle. For BlockCache implementations.
    void* handle() const {
      return handle_;
    }

    /// Marks the entry sealed once it is visible to other readers. For
    /// BlockCache implementations.
    void seal() {
      sealed_ = true;
    }

   private:
    BlockCache* cache_{nullptr};
    void* handle_{nullptr};
    char* data_{nullptr};
    uint32_t size_{0};
    // True once this entry's bytes are complete and visible to other readers.
    // Fresh entries returned by a cache miss start out false; a sealed entry
    // moves between an exclusive filler and shared readers.
    bool sealed_{false};
  };

  virtual ~BlockCache() = default;

  /// Returns a pinned entry for (fileId, offset). On a miss the entry holds
  /// 'size' bytes, sealed() is false, and the caller must fill data() and then
  /// seal(). On a hit the bytes are valid, sealed() is true, and size() is at
  /// least 'size': a longer entry cached at the offset is returned as is. A
  /// shorter one is never returned; an implementation replaces it with a new
  /// entry, as on a miss, or rejects it. An implementation may reject a longer
  /// entry too. May block while another thread fills the same entry, depending
  /// on the implementation.
  /// 'ioStats' receives the request's ramHit counter, plus cache-wait counters
  /// for implementations that block on concurrent fills, and SSD read counters
  /// for implementations that load entries from SSD.
  virtual Entry getOrCreate(
      uint64_t fileId,
      uint64_t offset,
      uint32_t size,
      velox::io::IoStatistics& ioStats) = 0;

  /// Returns the sealed entry for (fileId, offset), or an empty Entry when none
  /// is cached. Unlike getOrCreate(), never returns an entry for the caller to
  /// fill, so callers can probe a key whose size they do not know and read it
  /// from Entry::size().
  /// May block while another thread fills the same entry, depending on the
  /// implementation. 'ioStats' receives the same counters as getOrCreate().
  virtual Entry
  find(uint64_t fileId, uint64_t offset, velox::io::IoStatistics& ioStats) = 0;

  /// Seals a filled entry so it is visible to other readers. Call exactly once
  /// per entry whose sealed() is false, after its bytes are written. Sealing
  /// may fail to cache the entry under memory pressure; the bytes stay valid
  /// either way.
  virtual void seal(Entry& entry) = 0;

 protected:
  // Releases the opaque pin stored in Entry::handle(). Called from ~Entry, so
  // it must not throw. Entry is a member of this class and so reaches it
  // without a friend declaration.
  virtual void release(void* pin) noexcept = 0;
};

/// Returns the BlockCache backed by velox's AsyncDataCache `cache`. Each
/// AsyncDataCache gets one BlockCache: the first call for it constructs the
/// BlockCache, and every later call returns the same pointer. When `cache` is
/// null, uses `velox::cache::AsyncDataCache::getInstance()`, which is the
/// production path. Thread-safe. The returned pointer is owned by a
/// process-wide registry and must not be deleted; the backing AsyncDataCache
/// must outlive every entry it hands out. Entries are contiguous. When the
/// AsyncDataCache has an SSD cache, entries up to 8 MiB, the largest an SsdRun
/// can describe, are saved to it. A lookup that misses memory then loads the
/// entry from SSD: getOrCreate() only an entry of the requested size, and
/// find() an entry of any size. getOrCreate() and find() block while another
/// thread fills the same entry, so concurrent misses on one key cost a single
/// read.
BlockCache* getBlockCache(velox::cache::AsyncDataCache* cache = nullptr);

/// Destroys the BlockCaches that getBlockCache() returned, so the next call
/// rebuilds them. Intended for test fixtures that construct a fresh
/// AsyncDataCache per test; production code should not call this.
void testingResetBlockCache();

} // namespace facebook::nimble
