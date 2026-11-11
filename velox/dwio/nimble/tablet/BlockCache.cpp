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

#include <folly/Synchronized.h>
#include <folly/executors/QueuedImmediateExecutor.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

BlockCache::Entry& BlockCache::Entry::operator=(Entry&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  cache_ = std::exchange(other.cache_, nullptr);
  handle_ = std::exchange(other.handle_, nullptr);
  data_ = std::exchange(other.data_, nullptr);
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
  sealed_ = false;
}

namespace {

// BlockCache over velox's AsyncDataCache. Entries are contiguous and never
// saved to the SSD tier. getOrCreate() blocks while another thread fills the
// same entry, so concurrent misses on one key cost a single read.
class BlockCacheImpl final : public BlockCache {
 public:
  explicit BlockCacheImpl(velox::cache::AsyncDataCache* cache)
      : cache_{velox::checkedNotNull(cache)} {}

  Entry getOrCreate(
      uint64_t fileId,
      uint64_t offset,
      uint32_t size,
      velox::io::IoStatistics& ioStats) override {
    velox::cache::CachePin pin;
    for (;;) {
      folly::SemiFuture<bool> waitFuture{false};
      pin = cache_->findOrCreate(
          velox::cache::RawFileCacheKey{fileId, offset},
          size,
          /*contiguous=*/true,
          &waitFuture);
      if (!pin.empty()) {
        break;
      }
      // An empty pin means another thread holds the entry exclusively. The
      // future only signals that its load finished, so findOrCreate must run
      // again to acquire a shared or exclusive pin.
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

    auto* entry = pin.checkedEntry();
    NIMBLE_CHECK(
        entry->hasContiguousData(), "Cache entry must have contiguous data");
    // An exclusive pin means the caller must fill the entry; a shared pin
    // means the bytes are already sealed in cache.
    const bool sealed = !entry->isExclusive();
    // The first read of a prefetched entry is not a hit: the prefetch already
    // paid for the IO.
    const bool firstUse = entry->getAndClearFirstUseFlag();
    if (sealed && !firstUse) {
      ioStats.ramHit().increment(size);
    }
    char* const data = entry->contiguousData();
    return Entry{
        this, new velox::cache::CachePin(std::move(pin)), data, sealed};
  }

  void seal(Entry& entry) override {
    NIMBLE_CHECK(!entry.sealed(), "Only an unsealed cache entry can be sealed");
    auto* pin = static_cast<velox::cache::CachePin*>(entry.handle());
    pin->checkedEntry()->setExclusiveToShared(/*ssdSavable=*/false);
    entry.seal();
  }

 protected:
  void release(void* pin) noexcept override {
    NIMBLE_CHECK_NOT_NULL(pin);
    delete static_cast<velox::cache::CachePin*>(pin);
  }

 private:
  velox::cache::AsyncDataCache* const cache_;
};

} // namespace

namespace {
// Process-wide singleton BlockCache. First call to getBlockCache() constructs
// it; subsequent calls return the same pointer. testingResetBlockCache()
// clears it so the next call rebuilds -- test-only.
folly::Synchronized<std::unique_ptr<BlockCacheImpl>>& blockCacheSingleton() {
  static auto* singleton =
      new folly::Synchronized<std::unique_ptr<BlockCacheImpl>>();
  return *singleton;
}
} // namespace

BlockCache* getBlockCache(velox::cache::AsyncDataCache* cache) {
  auto locked = blockCacheSingleton().wlock();
  if (*locked == nullptr) {
    auto* backing =
        cache != nullptr ? cache : velox::cache::AsyncDataCache::getInstance();
    NIMBLE_CHECK_NOT_NULL(
        backing,
        "getBlockCache called with no cache and no process-wide AsyncDataCache");
    *locked = std::make_unique<BlockCacheImpl>(backing);
  }
  return locked->get();
}

void testingResetBlockCache() {
  blockCacheSingleton().wlock()->reset();
}

} // namespace facebook::nimble
