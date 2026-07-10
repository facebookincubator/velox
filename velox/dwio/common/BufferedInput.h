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

#include <atomic>
#include <list>

#include <folly/Executor.h>
#include <folly/futures/Future.h>

#include "folly/io/IOBuf.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/memory/AllocationPool.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/StreamIdentifier.h"

// Use WS VRead API to load
DECLARE_bool(wsVRLoad);

namespace facebook::velox::cache {
class CoalescedLoad;
}

namespace facebook::velox::dwio::common {

/// Provides read-only access to cached data without copying. Holds a
/// shared-mode pin on the cache entry, keeping it alive while the caller
/// accesses the data buffers.
class CachedRegion {
 public:
  /// The pin must be non-empty and in shared (non-exclusive) mode.
  explicit CachedRegion(cache::CachePin pin);

  uint64_t size() const {
    return size_;
  }

  /// Returns buffer ranges covering the cached data. For small entries the
  /// result is a single contiguous range; for larger entries there may be
  /// multiple non-contiguous ranges from the backing allocation.
  const std::vector<folly::Range<const char*>>& ranges() const {
    return ranges_;
  }

  /// Returns an IOBuf chain wrapping the cached data ranges without copying.
  /// The returned IOBuf references memory owned by this CachedRegion, so
  /// the caller must not outlive this object.
  folly::IOBuf toIOBuf() const;

 private:
  cache::CachePin pin_;
  // Cached data size in bytes.
  uint64_t size_{0};
  std::vector<folly::Range<const char*>> ranges_;
};

/// Shared bookkeeping for the outstanding-prefetch-bytes budget.
///
/// One instance is created by the root `BufferedInput`; clones (per-RG
/// inputs created by `BufferedInput::clone()` and used by
/// `StructColumnReader::loadRowGroup`) share the same instance via
/// `shared_ptr`. Without sharing, each cloned input would enforce the
/// 256 MB cap independently, multiplying the effective budget by the
/// number of prefetched row groups (e.g. 8 × 256 MB = 2 GB resident).
struct PrefetchBudget {
  // Per-file prefetch cap. <= 0 disables budget checks.
  // Stored in shared state so clones inherit the same cap automatically.
  std::atomic<int64_t> maxBytes{256 << 20};
  std::atomic<int64_t> outstandingBytes{0};
  std::atomic<int64_t> peakBytes{0};
  std::atomic<uint64_t> skipsAtCap{0};

  // Self-contained release: matches `BufferedInput::releasePrefetchBytes`
  // but takes no dependency on a BufferedInput instance. Async
  // continuations that need to release a reservation capture this
  // struct by `shared_ptr` value (see BufferedInput::prefetchBudget()),
  // so they remain safe to fire even if the originating BufferedInput
  // has already been destroyed. This is what makes the async-release
  // chains in CachedBufferedInput/DirectBufferedInput lifetime-safe
  // without requiring the destructor to drain every in-flight
  // continuation.
  void release(int64_t bytes) {
    if (maxBytes.load(std::memory_order_relaxed) <= 0 || bytes <= 0) {
      return;
    }
    const int64_t prev =
        outstandingBytes.fetch_sub(bytes, std::memory_order_acq_rel);
    DCHECK_GE(prev, bytes) << "PrefetchBudget::release underflow: prev=" << prev
                           << " release=" << bytes
                           << " (double-release or release-without-reserve)";
  }
};

class BufferedInput {
 public:
  constexpr static uint64_t kMaxMergeDistance = 1024 * 1024 * 1.25;

  BufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      const MetricsLogPtr& metricsLog = MetricsLog::voidLog(),
      IoStatistics* stats = nullptr,
      velox::IoStats* ioStats = nullptr,
      uint64_t maxMergeDistance = kMaxMergeDistance,
      std::optional<bool> wsVRLoad = std::nullopt,
      folly::F14FastMap<std::string, std::string> fileReadOps = {},
      bool cacheable = false)
      : BufferedInput(
            std::make_shared<ReadFileInputStream>(
                std::move(readFile),
                metricsLog,
                stats,
                ioStats,
                std::move(fileReadOps),
                cacheable),
            pool,
            maxMergeDistance,
            wsVRLoad) {}

  BufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      memory::MemoryPool& pool,
      uint64_t maxMergeDistance = kMaxMergeDistance,
      std::optional<bool> wsVRLoad = std::nullopt)
      : input_{std::move(input)},
        pool_{&pool},
        maxMergeDistance_{maxMergeDistance},
        wsVRLoad_{wsVRLoad},
        allocPool_{std::make_unique<memory::AllocationPool>(&pool)} {}

  BufferedInput(BufferedInput&&) = default;
  virtual ~BufferedInput();

  BufferedInput(const BufferedInput&) = delete;
  BufferedInput& operator=(const BufferedInput&) = delete;
  BufferedInput& operator=(BufferedInput&&) = delete;

  virtual const std::string& getName() const {
    return input_->getName();
  }

  /// The previous API was taking a vector of regions. Now we allow callers to
  /// enqueue region any time/place and we do final load into buffer in 2 steps
  /// (enqueue....load). 'si' allows tracking which streams actually get read.
  /// This may control read-ahead and caching for BufferedInput implementations
  /// supporting these.
  virtual std::unique_ptr<SeekableInputStream> enqueue(
      velox::common::Region region,
      const StreamIdentifier* sid = nullptr);

  /// Preloads the entire file into memory for fast sub-region access.
  /// Each subclass stores the preloaded data in its own native format.
  /// For small files (<= filePreloadThreshold), this eliminates separate
  /// footer and stripe data reads.
  virtual void preload() {
    enqueue({0, input_->getLength()});
    load(LogType::FILE);
    preloaded_ = true;
  }

  /// Returns true if the file has been preloaded.
  virtual bool preloaded() const {
    return preloaded_;
  }

  /// load all regions to be read in an optimized way (IO efficiency)
  virtual void load(const LogType);

  virtual bool isBuffered(uint64_t offset, uint64_t length) const {
    return !!readBuffer(offset, length);
  }

  virtual std::unique_ptr<SeekableInputStream>
  read(uint64_t offset, uint64_t length, LogType logType) const {
    auto ret = readBuffer(offset, length);
    if (ret != nullptr) {
      return ret;
    }
    VLOG(1) << "Unplanned read. Offset: " << offset << ", Length: " << length;
    // We cannot do enqueue/load here because load() clears previously
    // loaded data. TODO: figure out how we can use the data cache for
    // this access.
    return std::make_unique<SeekableFileInputStream>(
        input_, offset, length, *pool_, logType, input_->getNaturalReadSize());
  }

  // True if there is free memory for prefetching the stripe. This is
  // called to check if a stripe that is not next for read should be
  // prefetched. 'numPages' is added to the already enqueued pages, so
  // that this can be called also before enqueueing regions.
  virtual bool shouldPreload(int32_t /*numPages*/ = 0) {
    return false;
  }

  // True if caller should enqueue and load regions for stripe
  // metadata after reading a file footer. The stripe footers are
  // likely to be hit and should be read ahead of demand if
  // BufferedInput has background load.
  virtual bool shouldPrefetchStripes() const {
    return false;
  }

  virtual void setNumStripes(int32_t /*numStripes*/) {}

  // Create a new (clean) instance of BufferedInput sharing the same
  // underlying file and memory pool.  The enqueued regions are NOT copied.
  // Subclass overrides MUST propagate `prefetchBudget_` so per-RG clones
  // share one budget state (cap + counters) instead of multiplying
  // effective budget by prefetch fan-out.
  virtual std::unique_ptr<BufferedInput> clone() const {
    auto cloned = std::make_unique<BufferedInput>(
        input_, *pool_, maxMergeDistance_, wsVRLoad_);
    cloned->setPrefetchBudget(prefetchBudget_);
    return cloned;
  }

  std::unique_ptr<SeekableInputStream> loadCompleteFile() {
    auto stream = enqueue({0, input_->getLength()});
    load(dwio::common::LogType::FILE);
    return stream;
  }

  const std::shared_ptr<ReadFile>& getReadFile() const {
    return input_->getReadFile();
  }

  // Internal API, do not use outside Velox.
  const std::shared_ptr<ReadFileInputStream>& getInputStream() const {
    return input_;
  }

  virtual folly::Executor* executor() const {
    return nullptr;
  }

  /// Returns true if this BufferedInput has a backing cache (e.g.,
  /// AsyncDataCache). When true, callers may skip their own caching of raw
  /// bytes since the BufferedInput will handle caching.
  virtual bool hasCache() const {
    return false;
  }

  /// Offers pre-read data for a file region to the backing cache. Throws if
  /// hasCache() is false. Override in subclasses with a backing cache.
  virtual void cacheRegion(
      uint64_t /*offset*/,
      uint64_t /*length*/,
      std::string_view /*data*/) {
    VELOX_UNSUPPORTED("cacheRegion requires a backing cache");
  }

  /// Overload that copies from an IOBuf (possibly chained) into the cache
  /// entry, avoiding the need to coalesce the IOBuf first. 'bufferOffset' is
  /// the byte offset within the IOBuf chain where the region data starts.
  /// Throws if hasCache() is false.
  virtual void cacheRegion(
      uint64_t /*offset*/,
      uint64_t /*length*/,
      const folly::IOBuf& /*buffer*/,
      uint64_t /*bufferOffset*/) {
    VELOX_UNSUPPORTED("cacheRegion requires a backing cache");
  }

  /// Finds a cached region at the given offset. Returns a CachedRegion holding
  /// a shared-mode pin on the cache entry if found, or std::nullopt on cache
  /// miss. Throws if hasCache() is false.
  virtual std::optional<CachedRegion> findCachedRegion(
      uint64_t /*offset*/) const {
    VELOX_UNSUPPORTED("findCachedRegion requires a backing cache");
  }

  virtual uint64_t nextFetchSize() const;

  /// Resets the buffered input for reuse. This is used by index lookup which
  /// reuses the same BufferedInput across different index lookups. For
  /// instance, Nimble file format with cluster index supports index lookup and
  /// needs to reset the buffered input state between lookups.
  virtual void reset();

  /// Outstanding-prefetch-bytes budget (Velox async-IO redesign step 5).
  ///
  /// Bounds the total resident bytes of in-flight async prefetch loads
  /// dispatched through this BufferedInput. The cap exists so that wide
  /// schemas / large RG fan-outs cannot blow memory by speculatively
  /// scheduling far ahead of the consumer.
  ///
  /// Usage contract for async paths (e.g. CachedBufferedInput's
  /// preadvAsync submission, future DirectBufferedInput async path):
  ///
  ///   if (tryReservePrefetchBytes(coalescedLoad.size())) {
  ///     // submit preadvAsync; release in continuation.
  ///   } else {
  ///     // leave load in kPlanned; consumer's first read() / next
  ///     // scheduleRowGroups tick re-attempts (skip-not-queue).
  ///   }
  ///
  /// Implementation uses CAS-style fetch_add then check, with rollback
  /// (fetch_sub) on failure. This makes the gate safe under concurrent
  /// submission: two threads cannot both observe "under cap" pre-add and
  /// both submit.
  ///
  /// A maxOutstandingPrefetchBytes <= 0 disables the cap entirely,
  /// preserving pre-budget behavior. The default is set in
  /// io::ReaderOptions::kDefaultMaxOutstandingPrefetchBytes.
  bool tryReservePrefetchBytes(int64_t bytes) {
    const int64_t maxBytes = maxOutstandingPrefetchBytes();
    if (maxBytes <= 0) {
      return true;
    }
    if (bytes <= 0) {
      return true;
    }
    auto& outstanding = prefetchBudget_->outstandingBytes;
    // Single-oversize-request safety: if nothing is in flight, allow a
    // request larger than the cap through. Without this, a misconfiguration
    // where max-coalesced-bytes > max-outstanding-prefetch-bytes would
    // silently stop the prefetch path forever (every load fails the gate,
    // stays kPlanned, never gets re-issued — only the on-demand consumer
    // path could pick them up, defeating prefetch entirely).
    //
    // CAS rather than load-then-add: the bypass must admit exactly one
    // oversize load at a time. A naive `load() == 0 && fetch_add(bytes)`
    // races — two threads both observe 0, both fetch_add, both submit,
    // and the cap is silently doubled (or worse). Use compare_exchange
    // to atomically transition 0 → bytes; if another thread won the race
    // we fall through to the normal under-cap path (which will correctly
    // reject and roll back this oversize request).
    if (bytes > maxBytes) {
      int64_t expected = 0;
      if (outstanding.compare_exchange_strong(
              expected,
              bytes,
              std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        updatePeakOutstandingPrefetchBytes(bytes);
        return true;
      }
      // Lost the race; fall through. The normal path below will roll
      // back since `after` will exceed the cap.
    }
    const int64_t after =
        outstanding.fetch_add(bytes, std::memory_order_acq_rel) + bytes;
    if (after > maxBytes) {
      // Roll back: another submitter may have raced us over the cap.
      outstanding.fetch_sub(bytes, std::memory_order_acq_rel);
      prefetchBudget_->skipsAtCap.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    updatePeakOutstandingPrefetchBytes(after);
    return true;
  }

  void releasePrefetchBytes(int64_t bytes) {
    // Forwards to PrefetchBudget::release; semantics identical to the
    // prior in-line implementation. Async continuations that outlive
    // `this` should instead capture the shared_ptr from
    // `prefetchBudget()` and call `->release(bytes)` directly — that
    // path has no `this` dependency.
    prefetchBudget_->release(bytes);
  }

  int64_t outstandingPrefetchBytes() const {
    return prefetchBudget_->outstandingBytes.load(std::memory_order_acquire);
  }

  int64_t peakOutstandingPrefetchBytes() const {
    return prefetchBudget_->peakBytes.load(std::memory_order_relaxed);
  }

  uint64_t prefetchSkipsAtCap() const {
    return prefetchBudget_->skipsAtCap.load(std::memory_order_relaxed);
  }

  /// Subclass hook: set the prefetch byte cap (default copied from
  /// io::ReaderOptions::kDefaultMaxOutstandingPrefetchBytes). Subclasses
  /// (e.g. CachedBufferedInput) call this from their constructor after
  /// reading the cap from io::ReaderOptions.
  void setMaxOutstandingPrefetchBytes(int64_t bytes) {
    prefetchBudget_->maxBytes.store(bytes, std::memory_order_relaxed);
  }

  int64_t maxOutstandingPrefetchBytes() const {
    return prefetchBudget_->maxBytes.load(std::memory_order_relaxed);
  }

  /// Returns the shared prefetch-budget bookkeeping. Subclasses propagate
  /// this to clones so per-RG cloned inputs share one counter rather than
  /// each enforcing the cap independently.
  const std::shared_ptr<PrefetchBudget>& prefetchBudget() const {
    return prefetchBudget_;
  }

  void setPrefetchBudget(std::shared_ptr<PrefetchBudget> budget) {
    VELOX_CHECK_NOT_NULL(budget);
    prefetchBudget_ = std::move(budget);
  }

 protected:
  /// Submits a single planned coalesced load to a native-async backend
  /// (preadvAsync) and registers the budget-release continuation in
  /// `inflightAsyncReleases_`. For prefetch loads this first reserves the
  /// outstanding-prefetch-bytes budget; if the budget is exhausted the load
  /// is left untouched (kPlanned) and NOT submitted (skip-not-queue), so a
  /// later tick can retry it. The release continuation captures the
  /// `PrefetchBudget` by `shared_ptr` (never `this`), so it is lifetime-safe
  /// even if this BufferedInput is destroyed before the backend IO completes;
  /// the Future is retained here and drained by the destructor. Shared by
  /// CachedBufferedInput::readRegions and DirectBufferedInput::readRegions.
  ///
  /// `ssdSavable` is forwarded to CoalescedLoad::loadOrFutureAsync and is the
  /// only value that differs between the two callers (Cached passes
  /// `!noCacheRetention()`, Direct passes `false`).
  void submitAsyncLoad(
      const std::shared_ptr<cache::CoalescedLoad>& load,
      bool prefetch,
      bool ssdSavable,
      folly::Executor* continuationExec);

  /// Drops already-completed entries from `inflightAsyncReleases_` to keep the
  /// retention list bounded across many load() cycles on the same instance.
  /// Call at the start of each readRegions() pass.
  void pruneReadyAsyncReleases();

  static int adjustedReadPct(const cache::TrackingData& trackingData) {
    // When this method is called, there is one more reference that is already
    // counted, but the corresponding read (if exists) has not happened yet.  So
    // we must count one fewer reference at this point.
    const auto referencedBytes =
        trackingData.referencedBytes - trackingData.lastReferencedBytes;
    if (referencedBytes == 0) {
      return 0;
    }
    const int pct = trackingData.readBytes / referencedBytes * 100;
    VELOX_CHECK_LE(0, pct, "Bad read percentage: {}", pct);
    // It is possible to seek back or clone the stream and read the same data
    // multiple times, or because of unplanned read, so pct could be larger than
    // 100.  This should be rare in production though.
    return pct;
  }

  // Move the requests in `noPrefetch' to `prefetch' if it is already covered by
  // coalescing in `prefetch'.
  template <typename Request, typename GetRegionOffset, typename GetRegionEnd>
  static void moveCoalesced(
      std::vector<Request>& prefetch,
      std::vector<int32_t>& ends,
      std::vector<Request>& noPrefetch,
      GetRegionOffset getRegionOffset,
      GetRegionEnd getRegionEnd) {
    auto numOldPrefetch = prefetch.size();
    prefetch.resize(prefetch.size() + noPrefetch.size());
    std::copy_backward(
        prefetch.data(), prefetch.data() + numOldPrefetch, prefetch.end());
    auto* oldPrefetch = prefetch.data() + noPrefetch.size();
    int numMoved = 0;
    int i = 0; // index into noPrefetch for read
    int j = 0; // index into oldPrefetch
    int k = 0; // index into prefetch
    int l = 0; // index into noPrefetch for write
    for (auto& end : ends) {
      prefetch[k++] = oldPrefetch[j++];
      while (j < end) {
        auto coalesceStart = getRegionEnd(oldPrefetch[j - 1]);
        auto coalesceEnd = getRegionOffset(oldPrefetch[j]);
        while (i < noPrefetch.size() &&
               getRegionOffset(noPrefetch[i]) < coalesceStart) {
          noPrefetch[l++] = noPrefetch[i++];
        }
        while (i < noPrefetch.size() &&
               getRegionEnd(noPrefetch[i]) <= coalesceEnd) {
          if (getRegionOffset(noPrefetch[i]) >= coalesceStart) {
            coalesceStart = getRegionEnd(noPrefetch[i]);
            prefetch[k++] = noPrefetch[i++];
            ++numMoved;
          } else {
            noPrefetch[l++] = noPrefetch[i++];
          }
        }
        prefetch[k++] = oldPrefetch[j++];
      }
      end += numMoved;
    }
    while (i < noPrefetch.size()) {
      noPrefetch[l++] = noPrefetch[i++];
    }
    VELOX_CHECK_EQ(k, numOldPrefetch + numMoved);
    prefetch.resize(k);
    VELOX_CHECK_EQ(l + numMoved, noPrefetch.size());
    noPrefetch.resize(l);
  }

  const std::shared_ptr<ReadFileInputStream> input_;
  memory::MemoryPool* const pool_;

 private:
  // Bump prefetchBudget_->peakBytes to `value` if it's the new high.
  // Lock-free CAS loop; tolerates concurrent updaters.
  void updatePeakOutstandingPrefetchBytes(int64_t value) {
    auto& peak = prefetchBudget_->peakBytes;
    int64_t prev = peak.load(std::memory_order_relaxed);
    while (
        value > prev &&
        !peak.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
    }
  }

  std::unique_ptr<SeekableInputStream> readBuffer(
      uint64_t offset,
      uint64_t length) const;

  std::tuple<const char*, uint64_t> readInternal(
      uint64_t offset,
      uint64_t length,
      std::optional<size_t> i = std::nullopt) const;

  void readToBuffer(
      uint64_t offset,
      folly::Range<char*> allocated,
      const LogType logType);

  folly::Range<char*> allocate(const velox::common::Region& region) {
    // Save the file offset and the buffer to which we'll read it
    offsets_.push_back(region.offset);
    buffers_.emplace_back(
        allocPool_->allocateFixed(region.length), region.length);
    return folly::Range<char*>(buffers_.back().data(), region.length);
  }

  bool useVRead() const;
  void sortRegions();
  void mergeRegions();

  // tries and merges WS read regions into one
  bool tryMerge(
      velox::common::Region& first,
      const velox::common::Region& second);

  bool preloaded_{false};
  uint64_t maxMergeDistance_;
  std::optional<bool> wsVRLoad_;
  std::unique_ptr<memory::AllocationPool> allocPool_;

  // Shared bookkeeping for the budget (cap + counters). Constructed by the root
  // BufferedInput; subclass clone() implementations propagate this same
  // shared_ptr so per-RG cloned inputs share one budget state and the cap
  // is enforced globally, not per-clone.
  std::shared_ptr<PrefetchBudget> prefetchBudget_{
      std::make_shared<PrefetchBudget>()};

  // Holds the Future returned by `.via(continuationExec)` for each async
  // coalesced load submitted by submitAsyncLoad(). Folly drops a deferred
  // chain at `.detach()` time unless the chosen executor owns the work in its
  // task queue; with the InlineExecutor fallback (no Velox IO executor
  // registered) we instead retain the Future here so the budget-release
  // continuation actually fires when the backend's preadvAsync promise
  // resolves on its IO completion thread. The destructor drains this list
  // before destroying captured state; pruneReadyAsyncReleases() trims
  // completed entries at the start of each readRegions() call.
  //
  // Unsynchronized by design: the only mutators are submitAsyncLoad(),
  // pruneReadyAsyncReleases() and ~BufferedInput(), all driven by the single
  // reader thread that owns this per-split BufferedInput (readRegions() is
  // never called concurrently on one instance, and destruction happens-after
  // all readRegions() calls). The async release continuations never touch
  // this list -- they only release the PrefetchBudget via its own atomic --
  // so there is no cross-thread list mutation to guard. Completions race only
  // on each Future's own (thread-safe) core.
  std::list<folly::Future<folly::Unit>> inflightAsyncReleases_;

  // Regions enqueued for reading
  std::vector<velox::common::Region> regions_;

  // Offsets in the file to which the corresponding Region belongs
  std::vector<uint64_t> offsets_;

  // Buffers allocated for reading each Region.
  std::vector<folly::Range<char*>> buffers_;

  // Maps the position in which the Region was originally enqueued to the
  // position that it went to after sorting and merging. Thus this maps from the
  // enqueued position to its corresponding buffer offset.
  std::vector<size_t> enqueuedToBufferOffset_;
};

} // namespace facebook::velox::dwio::common
