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

#include <folly/Executor.h>

#include "velox/common/caching/FileGroupStats.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/CacheInputStream.h"
#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/IoStatistics.h"
#include "velox/dwio/common/Options.h"

DECLARE_int32(cache_load_quantum);

namespace facebook::velox::dwio::common {

struct CacheRequest {
  CacheRequest(
      cache::RawFileCacheKey _key,
      uint64_t _size,
      cache::TrackingId _trackingId)
      : key(_key), size(_size), trackingId(_trackingId) {}

  cache::RawFileCacheKey key;
  uint64_t size;
  cache::TrackingId trackingId;
  cache::CachePin pin;
  cache::SsdPin ssdPin;

  bool processed{false};

  // True if this should be coalesced into a CoalescedLoad with other
  // nearby requests with a similar load probability. This is false
  // for sparsely accessed large columns where hitting one piece
  // should not load the adjacent pieces.
  bool coalesces{true};
  const SeekableInputStream* FOLLY_NONNULL stream;
};

class CachedBufferedInput : public BufferedInput {
 public:
  CachedBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      const MetricsLogPtr& metricsLog,
      uint64_t fileNum,
      cache::AsyncDataCache* FOLLY_NONNULL cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      uint64_t groupId,
      std::shared_ptr<IoStatistics> ioStats,
      folly::Executor* FOLLY_NULLABLE executor,
      int32_t loadQuantum,
      int32_t maxCoalesceDistance)
      : BufferedInput(std::move(readFile), pool, metricsLog),
        cache_(cache),
        fileNum_(fileNum),
        tracker_(std::move(tracker)),
        groupId_(groupId),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        fileSize_(input_->getLength()),
        loadQuantum_(loadQuantum),
        maxCoalesceDistance_(maxCoalesceDistance) {}

  CachedBufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      memory::MemoryPool& pool,
      uint64_t fileNum,
      cache::AsyncDataCache* FOLLY_NONNULL cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      uint64_t groupId,
      std::shared_ptr<IoStatistics> ioStats,
      folly::Executor* FOLLY_NULLABLE executor,
      int32_t loadQuantum,
      int32_t maxCoalesceDistance)
      : BufferedInput(std::move(input), pool),
        cache_(cache),
        fileNum_(fileNum),
        tracker_(std::move(tracker)),
        groupId_(groupId),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        fileSize_(input_->getLength()),
        loadQuantum_(loadQuantum),
        maxCoalesceDistance_(maxCoalesceDistance) {}

  ~CachedBufferedInput() override {
    for (auto& load : allCoalescedLoads_) {
      load->cancel();
    }
  }

  std::unique_ptr<SeekableInputStream> enqueue(
      Region region,
      const StreamIdentifier* FOLLY_NULLABLE si) override;

  void load(const LogType) override;

  bool isBuffered(uint64_t offset, uint64_t length) const override;

  std::unique_ptr<SeekableInputStream>
  read(uint64_t offset, uint64_t length, LogType logType) const override;

  /// Schedules load of 'region' on 'executor_'. Fails silently if no memory or
  /// if shouldPreload() is false.
  bool prefetch(Region region);

  bool shouldPreload(int32_t numPages = 0) override;

  bool shouldPrefetchStripes() const override {
    return true;
  }

  void setNumStripes(int32_t numStripes) override {
    auto stats = tracker_->fileGroupStats();
    if (stats) {
      stats->recordFile(fileNum_, groupId_, numStripes);
    }
  }

  virtual std::unique_ptr<BufferedInput> clone() const override {
    return std::make_unique<CachedBufferedInput>(
        input_,
        pool_,
        fileNum_,
        cache_,
        tracker_,
        groupId_,
        ioStats_,
        executor_,
        loadQuantum_,
        maxCoalesceDistance_);
  }

  cache::AsyncDataCache* FOLLY_NONNULL cache() const {
    return cache_;
  }

  // Returns the CoalescedLoad that contains the correlated loads for
  // 'stream' or nullptr if none. Returns nullptr on all but first
  // call for 'stream' since the load is to be triggered by the first
  // access.
  std::shared_ptr<cache::CoalescedLoad> coalescedLoad(
      const SeekableInputStream* FOLLY_NONNULL stream);

  folly::Executor* FOLLY_NULLABLE executor() const override {
    return executor_;
  }

 private:
  // Sorts requests and makes CoalescedLoads for nearby requests. If 'prefetch'
  // is true, starts background loading.
  void makeLoads(std::vector<CacheRequest*> requests, bool prefetch);

  // Makes a CoalescedLoad for 'requests' to be read together, coalescing
  // IO is appropriate. If 'prefetch' is set, schedules the CoalescedLoad
  // on 'executor_'. Links the CoalescedLoad  to all CacheInputStreams that it
  // concerns.

  void readRegion(std::vector<CacheRequest*> requests, bool prefetch);

  cache::AsyncDataCache* FOLLY_NONNULL cache_;
  const uint64_t fileNum_;
  std::shared_ptr<cache::ScanTracker> tracker_;
  const uint64_t groupId_;
  std::shared_ptr<IoStatistics> ioStats_;
  folly::Executor* const FOLLY_NULLABLE executor_;

  // Regions that are candidates for loading.
  std::vector<CacheRequest> requests_;

  // Coalesced loads spanning multiple cache entries in one IO.
  folly::Synchronized<folly::F14FastMap<
      const SeekableInputStream*,
      std::shared_ptr<cache::CoalescedLoad>>>
      coalescedLoads_;

  // Distinct coalesced loads in 'coalescedLoads_'.
  std::vector<std::shared_ptr<cache::CoalescedLoad>> allCoalescedLoads_;

  const uint64_t fileSize_;
  const int32_t loadQuantum_;
  const int32_t maxCoalesceDistance_;
};

} // namespace facebook::velox::dwio::common
