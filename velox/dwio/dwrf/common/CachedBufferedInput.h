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

#include "velox/common/caching/GroupTracker.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/dwrf/common/BufferedInput.h"
#include "velox/dwio/dwrf/common/CacheInputStream.h"

#include <folly/Executor.h>

DECLARE_int32(cache_load_quantum);

namespace facebook::velox::dwrf {

// Abstract class for owning an InputStream and related structures
// like pins into file handle caches. TODO: Make file handle cache
// produce shared_ptr<InputStream> and write all in terms of these.
class AbstractInputStreamHolder {
 public:
  virtual ~AbstractInputStreamHolder() = default;
  virtual dwio::common::InputStream& get() = 0;
};

// Function type for making copies of InputStream for running
// parammel loads. We cannot pass the one non-owned InputStream to
// other threads because the BufferedInput and its owner could be
// destroyed out of sequence and the streams are owned via
// unique_ptr. TODO: Make all streams owned via shared_ptr.
using StreamSource =
    std::function<std::unique_ptr<AbstractInputStreamHolder>()>;

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

  // True if this should be coalesced into a FusedLoad with other
  // nearby requests with a similar load probability. This is false
  // for sparsely accessed large columns where hitting one piece
  // should not load the adjacent pieces.
  bool coalesces{true};
  const SeekableInputStream* stream;
};

class CachedBufferedInput : public BufferedInput {
 public:
  CachedBufferedInput(
      dwio::common::InputStream& input,
      memory::MemoryPool& pool,
      dwio::common::DataCacheConfig* dataCacheConfig,
      cache::AsyncDataCache* cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      uint64_t groupId,
      StreamSource streamSource,
      std::shared_ptr<dwio::common::IoStatistics> ioStats,
      folly::Executor* executor)
      : BufferedInput(input, pool, dataCacheConfig),
        cache_(cache),
        fileNum_(dataCacheConfig->filenum),
        tracker_(std::move(tracker)),
        groupId_(groupId),
        streamSource_(streamSource),
        ioStats_(std::move(ioStats)),
        executor_(executor),
        fileSize_(input.getLength()) {
    tracker_->setLoadQuantum(FLAGS_cache_load_quantum);
  }

  ~CachedBufferedInput() override {
    for (auto& load : allFusedLoads_) {
      load->cancel();
    }
  }

  const std::string& getName() const override {
    return input_.getName();
  }

  std::unique_ptr<SeekableInputStream> enqueue(
      dwio::common::Region region,
      const StreamIdentifier* si) override;

  void load(const dwio::common::LogType) override;

  bool isBuffered(uint64_t offset, uint64_t length) const override;

  std::unique_ptr<SeekableInputStream> read(
      uint64_t offset,
      uint64_t length,
      dwio::common::LogType logType) const override;

  bool shouldPreload() override;

  bool shouldPrefetchStripes() const override {
    return true;
  }

  void setNumStripes(int32_t numStripes) override {
    if (tracker_->fileGroupStats()) {
      tracker_->fileGroupStats()->recordFile(fileNum_, groupId_, numStripes);
    }
  }

  cache::AsyncDataCache* cache() const {
    return cache_;
  }

  // Returns the FusedLoad that contains the correlated loads for
  // 'stream' or nullptr if none. Returns nullptr on all but first
  // call for 'stream' since the load is to be triggered by the first
  // access.
  std::shared_ptr<cache::FusedLoad> fusedLoad(
      const SeekableInputStream* stream);

 private:
  // Sorts requests and makes FusedLoads for nearby requests. If 'prefetch' is
  // true, starts background loading.
  void makeLoads(std::vector<CacheRequest*> requests, bool prefetch);

  // Makes a FusedLoad for 'requests' to be read together, coalescing
  // IO is appropriate. If 'prefetch' is set, schedules the FusedLoad
  // on 'executor_'. Links the FusedLoad  to all CacheInputStreams tat it
  // concers.
  void readRegion(std::vector<CacheRequest*> requests, bool prefetch);

  cache::AsyncDataCache* cache_;
  const uint64_t fileNum_;
  std::shared_ptr<cache::ScanTracker> tracker_;
  const uint64_t groupId_;
  StreamSource streamSource_;
  std::shared_ptr<dwio::common::IoStatistics> ioStats_;
  folly::Executor* const executor_;

  // Regions that are candidates for loading.
  std::vector<CacheRequest> requests_;

  // Coalesced loads spanning multiple cache entries in one IO.
  folly::Synchronized<folly::F14FastMap<
      const SeekableInputStream*,
      std::shared_ptr<cache::FusedLoad>>>
      fusedLoads_;

  // Distinct fused loads in 'fusedLoads_'.
  std::vector<std::shared_ptr<cache::FusedLoad>> allFusedLoads_;

  const uint64_t fileSize_;
};

class CachedBufferedInputFactory : public BufferedInputFactory {
 public:
  CachedBufferedInputFactory(
      cache::AsyncDataCache* cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      uint64_t groupId,
      StreamSource streamSource,
      std::shared_ptr<dwio::common::IoStatistics> ioStats,
      folly::Executor* executor)
      : cache_(cache),
        tracker_(std::move(tracker)),
        groupId_(groupId),
        streamSource_(streamSource),
        ioStats_(ioStats),
        executor_(executor) {}

  std::unique_ptr<BufferedInput> create(
      dwio::common::InputStream& input,
      velox::memory::MemoryPool& pool,
      dwio::common::DataCacheConfig* dataCacheConfig = nullptr) const override {
    return std::make_unique<CachedBufferedInput>(
        input,
        pool,
        dataCacheConfig,
        cache_,
        tracker_,
        groupId_,
        streamSource_,
        ioStats_,
        executor_);
  }

  std::string toString() const {
    if (tracker_) {
      return tracker_->toString();
    }
    return "";
  }

 private:
  cache::AsyncDataCache* const cache_;
  std::shared_ptr<cache::ScanTracker> tracker_;
  const uint64_t groupId_;
  StreamSource streamSource_;
  std::shared_ptr<dwio::common::IoStatistics> ioStats_;
  folly::Executor* executor_;
};
} // namespace facebook::velox::dwrf
