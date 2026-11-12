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
#include "velox/dwio/nimble/tablet/BlockCache.h"

#include <exception>
#include <vector>

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/executors/QueuedImmediateExecutor.h>
#include <glog/logging.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/caching/SsdFile.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

namespace {

// Largest entry the SSD cache can hold, since an SsdRun stores the entry size
// in kSizeBits bits.
constexpr uint32_t kMaxSsdEntrySize{1U << velox::cache::SsdRun::kSizeBits};

// Waits until another thread's exclusive pin on an entry is released.
void waitForFill(
    folly::SemiFuture<bool> waitFuture,
    velox::io::IoStatistics& ioStats) {
  NIMBLE_CHECK(waitFuture.valid(), "Cache wait future is invalid");
  uint64_t waitUs{0};
  {
    velox::MicrosecondTimer timer(&waitUs);
    std::move(waitFuture)
        .via(&folly::QueuedImmediateExecutor::instance())
        .wait();
  }
  ioStats.queryThreadIoLatencyUs().increment(waitUs);
  ioStats.cacheWaitLatencyUs().increment(waitUs);
}

// Fills 'pin', a new entry of 'key', from 'ssdPin' and seals it. If the load
// fails, erases the SSD entry, so that later lookups read storage, and
// returns false with the entry unfilled.
bool loadFromSsd(
    const velox::cache::RawFileCacheKey& key,
    velox::cache::SsdPin ssdPin,
    velox::cache::CachePin& pin,
    velox::io::IoStatistics& ioStats) {
  auto* ssdFile = ssdPin.file();

  // SsdFile::load() takes vectors of pins, so 'pin' moves into one and back.
  std::vector<velox::cache::SsdPin> ssdPins;
  ssdPins.push_back(std::move(ssdPin));
  std::vector<velox::cache::CachePin> pins;
  pins.push_back(std::move(pin));
  uint64_t loadUs{0};
  try {
    velox::MicrosecondTimer timer(&loadUs);
    ssdFile->load(ssdPins, pins);
  } catch (const std::exception& e) {
    LOG(ERROR) << "IOERR: Failed to load a block cache entry from SSD, file "
               << velox::fileIds().string(key.fileNum) << ", offset "
               << key.offset << ": " << e.what();
    ssdFile->erase(key);
    pin = std::move(pins[0]);
    return false;
  }
  pin = std::move(pins[0]);
  auto* entry = pin.checkedEntry();
  ioStats.ssdRead().increment(entry->size());
  ioStats.queryThreadIoLatencyUs().increment(loadUs);
  ioStats.ssdCacheReadLatencyUs().increment(loadUs);
  entry->setExclusiveToShared();
  return true;
}

// Counts 'hitBytes' as RAM hits if the lookup that pinned 'entry' found it
// sealed in memory.
void recordRamHitBytes(
    velox::cache::AsyncDataCacheEntry& entry,
    uint32_t hitBytes,
    velox::io::IoStatistics& ioStats) {
  // The first read of a prefetched entry is not a hit: the prefetch already
  // paid for the IO. A new entry also starts with the first-use flag, which is
  // cleared here so that its next read counts.
  const bool firstUse = entry.getAndClearFirstUseFlag();
  if (entry.isShared() && !firstUse) {
    ioStats.ramHit().increment(hitBytes);
  }
}

// Wraps a non-empty pin in an Entry that 'cache' releases.
BlockCache::Entry toEntry(BlockCache* cache, velox::cache::CachePin pin) {
  auto* entry = pin.checkedEntry();

  // TODO: Support non-contiguous entries, so that CachedDataInput and
  // CachedMetadataInput can cache large entries in page runs instead of
  // one contiguous allocation.
  NIMBLE_CHECK(
      entry->hasContiguousData(), "Cache entry must have contiguous data");

  // An exclusive pin means the caller must fill the entry; a shared pin
  // means the bytes are already sealed in cache.
  const bool sealed = !entry->isExclusive();
  char* const data = entry->contiguousData();
  const auto size = static_cast<uint32_t>(entry->size());
  return BlockCache::Entry{
      cache, new velox::cache::CachePin(std::move(pin)), data, size, sealed};
}

// BlockCache over velox's AsyncDataCache. Entries are contiguous, and those up
// to kMaxSsdEntrySize are saved to its SSD cache when it has one. getOrCreate()
// and find() block while another thread fills the same entry, so concurrent
// misses on one key cost a single read.
class BlockCacheImpl final : public BlockCache {
 public:
  explicit BlockCacheImpl(velox::cache::AsyncDataCache* cache)
      : cache_{velox::checkedNotNull(cache)} {}

  Entry getOrCreate(
      uint64_t fileId,
      uint64_t offset,
      uint32_t size,
      velox::io::IoStatistics& ioStats) override {
    const velox::cache::RawFileCacheKey key{fileId, offset};
    auto pin = findOrCreate(key, size, ioStats);

    // A new entry takes the bytes of an SSD entry only of the same size. A
    // metadata entry holds a decompressed section, which a prefix of a longer
    // entry is not.
    if (pin.checkedEntry()->isExclusive()) {
      auto ssdPin = lookupSsd(key);
      if (!ssdPin.empty() && ssdPin.run().size() == size) {
        loadFromSsd(key, std::move(ssdPin), pin, ioStats);
      }
    }
    return toEntry(this, std::move(pin));
  }

  Entry find(uint64_t fileId, uint64_t offset, velox::io::IoStatistics& ioStats)
      override {
    const velox::cache::RawFileCacheKey key{fileId, offset};
    for (;;) {
      folly::SemiFuture<bool> waitFuture{false};
      auto pin = cache_->find(key, &waitFuture);
      if (!pin.has_value()) {
        return findSsd(key, ioStats);
      }
      if (!pin->empty()) {
        auto& entry = *pin->checkedEntry();
        recordRamHitBytes(entry, static_cast<uint32_t>(entry.size()), ioStats);
        return toEntry(this, std::move(*pin));
      }

      // As in findOrCreate(), the future only signals that the fill finished.
      waitForFill(std::move(waitFuture), ioStats);
    }
  }

  void seal(Entry& entry) override {
    NIMBLE_CHECK(!entry.sealed(), "Only an unsealed cache entry can be sealed");
    auto* pin = static_cast<velox::cache::CachePin*>(entry.handle());
    pin->checkedEntry()->setExclusiveToShared(
        /*ssdSavable=*/entry.size() <= kMaxSsdEntrySize);
    entry.seal();
  }

 protected:
  void release(void* pin) noexcept override {
    NIMBLE_CHECK_NOT_NULL(pin);
    delete static_cast<velox::cache::CachePin*>(pin);
  }

 private:
  // Returns a pin on the entry of 'key', which a miss creates with 'size'
  // bytes. Waits while another thread fills the entry.
  velox::cache::CachePin findOrCreate(
      const velox::cache::RawFileCacheKey& key,
      uint32_t size,
      velox::io::IoStatistics& ioStats) {
    for (;;) {
      folly::SemiFuture<bool> waitFuture{false};
      auto pin =
          cache_->findOrCreate(key, size, /*contiguous=*/true, &waitFuture);
      if (!pin.empty()) {
        recordRamHitBytes(*pin.checkedEntry(), size, ioStats);
        return pin;
      }
      // An empty pin means another thread holds the entry exclusively. The
      // future only signals that its load finished, so findOrCreate must run
      // again to acquire a shared or exclusive pin.
      waitForFill(std::move(waitFuture), ioStats);
    }
  }

  // Returns the SsdPin of 'key', or an empty pin when there is no SSD cache or
  // it does not hold the key.
  velox::cache::SsdPin lookupSsd(
      const velox::cache::RawFileCacheKey& key) const {
    auto* ssdCache = cache_->ssdCache();
    if (ssdCache == nullptr) {
      return {};
    }
    return ssdCache->file(key.fileNum).find(key);
  }

  // Returns the entry of 'key' loaded from SSD at the size the SSD cache
  // records, or an empty Entry when the SSD cache does not hold it or the load
  // fails. A loaded entry is also cached in memory, so later lookups of 'key'
  // hit memory.
  Entry findSsd(
      const velox::cache::RawFileCacheKey& key,
      velox::io::IoStatistics& ioStats) {
    auto ssdPin = lookupSsd(key);
    if (ssdPin.empty()) {
      return {};
    }
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::BlockCacheImpl::findSsd", this);
    auto pin = findOrCreate(key, ssdPin.run().size(), ioStats);

    // Another thread may have cached the key since the memory lookup missed.
    if (pin.checkedEntry()->isExclusive() &&
        !loadFromSsd(key, std::move(ssdPin), pin, ioStats)) {
      return {};
    }
    return toEntry(this, std::move(pin));
  }

  velox::cache::AsyncDataCache* const cache_;
};

using BlockCacheRegistry = folly::Synchronized<folly::F14FastMap<
    velox::cache::AsyncDataCache*,
    std::unique_ptr<BlockCacheImpl>>>;

// Process-wide BlockCaches, one per AsyncDataCache. The first call to
// getBlockCache() for an AsyncDataCache constructs its BlockCache; subsequent
// calls return the same pointer. testingResetBlockCache() clears them so the
// next call rebuilds -- test-only.
BlockCacheRegistry& blockCacheRegistry() {
  static auto* registry = new BlockCacheRegistry();
  return *registry;
}

} // namespace

BlockCache::Entry& BlockCache::Entry::operator=(Entry&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  cache_ = std::exchange(other.cache_, nullptr);
  handle_ = std::exchange(other.handle_, nullptr);
  data_ = std::exchange(other.data_, nullptr);
  size_ = std::exchange(other.size_, 0);
  sealed_ = std::exchange(other.sealed_, false);
  return *this;
}

void BlockCache::Entry::reset() noexcept {
  if (cache_ != nullptr) {
    cache_->release(handle_);
  }
  cache_ = nullptr;
  handle_ = nullptr;
  data_ = nullptr;
  size_ = 0;
  sealed_ = false;
}

BlockCache* getBlockCache(velox::cache::AsyncDataCache* cache) {
  auto* backing =
      cache != nullptr ? cache : velox::cache::AsyncDataCache::getInstance();
  NIMBLE_CHECK_NOT_NULL(
      backing,
      "getBlockCache called with no cache and no process-wide AsyncDataCache");
  {
    auto locked = blockCacheRegistry().rlock();
    const auto it = locked->find(backing);
    if (it != locked->end()) {
      return it->second.get();
    }
  }
  auto locked = blockCacheRegistry().wlock();
  auto& blockCache = (*locked)[backing];
  if (blockCache == nullptr) {
    blockCache = std::make_unique<BlockCacheImpl>(backing);
  }
  return blockCache.get();
}

void testingResetBlockCache() {
  blockCacheRegistry().wlock()->clear();
}

} // namespace facebook::nimble
