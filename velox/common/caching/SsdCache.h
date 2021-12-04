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

#include "velox/common/caching/AsyncDataCache.h"

namespace facebook::velox::cache {

// A 64 bit word describing a SSD cache entry in an SsdFile. The low
// 23 bits are the size, for a maximum entry size of 8MB. The high
// bits are the offset.
class SsdRun {
 public:
  static constexpr int32_t kSizeBits = 23;

  SsdRun() : bits_(0) {}

  SsdRun(uint64_t offset, uint32_t size)
      : bits_((offset << kSizeBits) | ((size - 1))) {
    VELOX_CHECK_LT(offset, 1L << (64 - kSizeBits));
    VELOX_CHECK_LT(size - 1, 1 << kSizeBits);
  }

  SsdRun(const SsdRun& other) = default;
  SsdRun(SsdRun&& other) = default;

  void operator=(const SsdRun& other) {
    bits_ = other.bits_;
  }
  void operator=(SsdRun&& other) {
    bits_ = other.bits_;
  }

  uint64_t offset() const {
    return (bits_ >> kSizeBits);
  }

  uint32_t size() const {
    return (bits_ & ((1 << kSizeBits) - 1)) + 1;
  }

 private:
  uint64_t bits_;
};

// Key for SsdFile lookup. The key is the file number in storage and
// the offset in the file. SSD cache sizes align to the RAM cache
// quantized sizes for cached streams from the original file.
struct SsdKey {
  StringIdLease file;
  uint64_t offset;

  bool operator==(const SsdKey& other) const {
    return offset == other.offset && file.id() == other.file.id();
  }
};

} // namespace facebook::velox::cache
namespace std {
template <>
struct hash<::facebook::velox::cache::SsdKey> {
  size_t operator()(const ::facebook::velox::cache::SsdKey& key) const {
    return facebook::velox::bits::hashMix(key.file.id(), key.offset);
  }
};

} // namespace std

namespace facebook::velox::cache {

// Represents an SsdFile entry that is planned for load or being
// loaded. This is destroyed after load. Destruction decrements the
// pin count of the corresponding region of 'file_'. While there are
// pins, the region cannot be evicted.
class SsdPin {
 public:
  SsdPin() : file_(nullptr) {}

  // Constructs a pin referencing 'run' in 'file'. The region must be
  // pinned before constructing the pin.
  SsdPin(SsdFile& file, SsdRun run);

  SsdPin(const SsdPin& other) = delete;

  SsdPin(SsdPin&& other) {
    run_ = other.run_;
    file_ = other.file_;
    other.file_ = nullptr;
  }

  ~SsdPin();

  void operator=(SsdPin&&);

  bool empty() const {
    return file_ == nullptr;
  }
  SsdFile* file() const {
    return file_;
  }

  SsdRun run() const {
    return run_;
  }

 private:
  SsdFile* file_;
  SsdRun run_;
};

// Metrics for SSD cache. Maintained by SsdFile and aggregated by SsdCache.
struct SsdCacheStats {
  uint64_t entriesWritten{0};
  uint64_t bytesWritten{0};
  uint64_t entriesRead{0};
  uint64_t bytesRead{0};
  uint64_t entriesCached{0};
  uint64_t bytesCached{0};
};

// A shard of SsdCache. Corresponds to one file on SSD.  The data
// backed by each SsdFile is selected on a hash of the storage file
// number of the cached data. Each file consists of an integer number
// of 64MB regions. Each region has a pin count and an read
// count. Cache replacement takes place region by region, preferring
// regions with a smaller read count. Entries do not span
// regions. Otherwise entries are consecutive byte ranges in side
// their region.
class SsdFile {
 public:
  static constexpr uint64_t kMaxSize = 1UL << 36; // 64G
  static constexpr uint64_t kRegionSize = 1 << 26; // 64MB
  static constexpr int32_t kNumRegions = kMaxSize / kRegionSize;

  // Constructs a cache backed by filename. Discards any previous
  // contents of filename.
  SsdFile(
      const std::string& filename,
      SsdCache& cache,
      int32_t ordinal,
      int32_t maxRegions);

  // Adds entries of  'pins'  to this file. 'pins' must be in read mode and
  // those pins that are successfully added to SSD are marked as being on SSD.
  // The file of the entries must be a file that is backed by 'this'.
  void store(std::vector<CachePin>& pins);

  // Finds an entry for 'key'. If no entry is found, the returned pin is empty.
  SsdPin find(RawFileCacheKey key);

  // Copies the data at 'run' into 'entry'. Checks that the entry
  // and run sizes match. The quantization of SSD cache matches that
  // of the memory cache.
  void load(SsdRun run, AsyncDataCacheEntry& entry);

  // Reads the backing file as per ReadFile::preadv(). 'numEntries' is
  // used only for updating statistics. Allows implementing coalesced
  // read of nearby entries.
  void read(
      uint64_t offset,
      const std::vector<folly::Range<char*>> buffers,
      int32_t numEntries);

  // Increments the pin count of the region of 'offset'.
  void pinRegion(uint64_t offset);

  // Increments the pin count of the region of 'offset'. Caller must hold
  // 'mutex_'.
  void pinRegionLocked(uint64_t offset) {
    ++regionPins_[regionIndex(offset)];
  }

  // Decrements the pin count of the region of 'offset'. If the pin
  // count goes to zero and evict is due, starts the evict.
  void unpinRegion(uint64_t offset);

  // Asserts that the region of 'offset' is pinned. This is called by
  // the pin holder. The pin count can be read without mutex.
  void checkPinned(uint64_t offset) const {
    VELOX_CHECK_LT(0, regionPins_[regionIndex(offset)]);
  }

  // Returns the region number corresponding to offset.
  static int32_t regionIndex(uint64_t offset) {
    return offset / kRegionSize;
  }

  // Updates the read count of a region.
  void regionUsed(int32_t region, int32_t size) {
    regionScore_[region] += size;
  }

  int32_t maxRegions() const {
    return maxRegions_;
  }

  int32_t ordinal() const {
    return ordinal_;
  }

  // Adds 'stats_' to 'stats'.
  void updateStats(SsdCacheStats& stats);

  // Resets [this' to a post-construction empty state. See SsdCache::clear().
  void clear();

 private:
  static constexpr int32_t kDecayInterval = 1000;

  // Returns [start, size] of contiguous space for storing data of a
  // number of contiguous 'pins' starting at 'startIndex'.  Returns a
  // run of 0 bytes if there is no space
  std::pair<uint64_t, int32_t> getSpace(
      const std::vector<CachePin>& pins,
      int32_t begin);

  // Removes all 'entries_' that reference data in 'toErase'. 'toErase is a set
  // of region indices.
  void clearRegionEntriesLocked(const std::vector<int32_t>& toErase);

  // Clears one or more  regions for accommodating new entries. The regions are
  // added to 'writableRegions_'. Returns true if regions could be cleared.
  bool evictLocked();

  // Increments event count and periodically decays scores.
  void newEventLocked();

  // Verifies that 'entry' has the data at 'run'.
  void verifyWrite(AsyncDataCacheEntry& entry, SsdRun run);

  // Serializes access to all private data members.
  std::mutex mutex_;

  // The containing SsdCache.
  SsdCache& cache_;

  // Stripe number within 'cache_'.
  int32_t ordinal_;

  // Number of kRegionSize regions in the file.
  int32_t numRegions_{0};

  // True if stopped serving traffic. Happens if no evictions are
  // possible due to everything being pinned. Clears when pins
  // decrease and space can be cleared.
  bool suspended_{false};

  // Maximum size of the backing file in kRegionSize units.
  const int32_t maxRegions_;

  // Number of used bytes in in each region. A new entry must fit
  // between the offset and the end of the region.
  std::vector<uint32_t> regionSize_;

  // Region numbers available for writing new entries.
  std::vector<int32_t> writableRegions_;

  // Count of bytes read from the corresponding region. Decays with time.
  std::vector<uint64_t> regionScore_;

  // Pin count for each region.
  std::vector<int32_t> regionPins_;

  // Map of file number and offset to location in file.
  folly::F14FastMap<SsdKey, SsdRun> entries_;

  // Count of reads and writes. The scores are decayed every time e count goes
  // over kDecayInterval or half 'entries_' size, wichever comes first.
  uint64_t numEvents_{0};

  // Name of backing file.
  const std::string filename_;

  // File descriptor.
  int32_t fd_;

  // Size of the backing file in bytes. Must be multiple of kRegionSize.
  uint64_t fileSize_{0};

  // ReadFile made from 'fd_'.
  std::unique_ptr<ReadFile> readFile_;

  // Counters.
  SsdCacheStats stats_;
};

class SsdCache {
 public:
  //  Constructs a cache with backing files at path
  //  'filePrefix'.<ordinal>. <ordinal> ranges from 0 to 'numShards' -
  //  1.. '. 'maxBytes' is the total capacity of the cache. This is
  //  rounded up to the next multiple of kRegionSize * 'numShards'.
  SsdCache(
      std::string_view filePrefix,
      uint64_t maxBytes,
      int32_t numShards_ = 32);

  // Returns the shard corresponding to 'fileId'.
  SsdFile& file(uint64_t fileId);

  uint64_t maxBytes() const {
    return files_[0]->maxRegions() * files_.size() * SsdFile::kRegionSize;
  }

  // Returns true if no store s in progress. Atomically sets a store
  // to be in progress. store() must be called after this. The storing
  // state is reset asynchronously after writing to SSD finishes.
  bool startStore();

  // Stores the entries of 'pins' into the corresponding files. Sets
  // the file for the successfully stored entries. May evict existing
  // entries from unpinned regions.
  void store(std::vector<CachePin> pins);

  // Returns  stats aggregated from all shards.
  SsdCacheStats stats() const;

  FileGroupStats& groupStats() const {
    return *groupStats_;
  }

  // Drops all entries. Outstanding pins become invalid but reading
  // them will mostly succeed since the files will not be rewritten
  // until new content is stord.
  void clear();

  std::string toString() const;

 private:
  std::string filePrefix_;
  const int32_t numShards_;
  std::vector<std::unique_ptr<SsdFile>> files_;

  // Count of shards with unfinished stores.
  std::atomic<int32_t> storesInProgress_{0};

  // Stats for selecting entries to save from AsyncDataCache.
  std::unique_ptr<FileGroupStats> groupStats_;
};

} // namespace facebook::velox::cache
