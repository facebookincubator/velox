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

#include "velox/common/io/CacheInputStream.h"
#include "velox/dwio/common/CachedBufferedInput.h"

namespace facebook::velox::dwio::common {

template <typename DerivedCacheInputStream, typename DerivedCachedBufferedInput>
class CovariantCacheInputStream : public facebook::velox::io::CacheInputStream {
 public:
  CovariantCacheInputStream(
      DerivedCachedBufferedInput* cache,
      IoStatistics* ioStats,
      const velox::common::Region& region,
      std::shared_ptr<ReadFileInputStream> input,
      uint64_t fileNum,
      std::shared_ptr<cache::ScanTracker> tracker,
      cache::TrackingId trackingId,
      uint64_t groupId,
      int32_t loadQuantum)
      : CacheInputStream(
            static_cast<io::CachedBufferedInput*>(cache),
            ioStats,
            region,
            input,
            fileNum,
            tracker,
            trackingId,
            groupId,
            loadQuantum) {}

  // Define the clone function with a covariant return type.
  std::unique_ptr<DerivedCacheInputStream> clone() {
    auto copy = std::make_unique<DerivedCacheInputStream>(
        static_cast<CachedBufferedInput*>(bufferedInput_),
        ioStats_,
        region_,
        input_,
        fileNum_,
        tracker_,
        trackingId_,
        groupId_,
        loadQuantum_);
    copy->position_ = position_;
    return copy;
  }
};

class CacheInputStream
    : public CovariantCacheInputStream<CacheInputStream, CachedBufferedInput> {
 public:
  CacheInputStream(
      CachedBufferedInput* cache,
      IoStatistics* ioStats,
      const velox::common::Region& region,
      std::shared_ptr<ReadFileInputStream> input,
      uint64_t fileNum,
      std::shared_ptr<cache::ScanTracker> tracker,
      cache::TrackingId trackingId,
      uint64_t groupId,
      int32_t loadQuantum)
      : CovariantCacheInputStream<CacheInputStream, CachedBufferedInput>(
            cache,
            ioStats,
            region,
            input,
            fileNum,
            tracker,
            trackingId,
            groupId,
            loadQuantum) {}
};
} // namespace facebook::velox::dwio::common