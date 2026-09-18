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

#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/SeekableInputStream.h"

namespace facebook::velox::dwio::common {

class DirectBufferedInput;
struct LoadedBuffer;

/// An input stream over possibly coalesced loads. Created by
/// DirectBufferedInput. Similar to CacheInputStream but does not use cache.
class DirectInputStream : public SeekableInputStream {
 public:
  DirectInputStream(
      DirectBufferedInput* bufferedInput,
      IoStatistics* ioStats,
      const velox::common::Region& region,
      std::shared_ptr<ReadFileInputStream> input,
      uint64_t fileNum,
      std::shared_ptr<cache::ScanTracker> tracker,
      cache::TrackingId trackingId,
      uint64_t groupId,
      int32_t loadQuantum);

  bool Next(const void** data, int* size) override;
  void BackUp(int count) override;
  bool SkipInt64(int64_t count) override;
  int64_t ByteCount() const override;

  void seekToPosition(PositionProvider& position) override;
  std::string getName() const override;
  size_t positionSize() const override;

  /// Testing function to access loaded state.
  void testingData(
      velox::common::Region& loadedRegion,
      memory::Allocation*& data,
      std::string*& tinyData) {
    loadedRegion = loadedRegion_;
    data = &loadedData_.owned;
    tinyData = &loadedData_.tiny;
  }

 private:
  // Ensures that the current position is covered by 'loadedData_'.
  void loadPosition();

  // Synchronously sets 'loadedData_' to cover 'loadedRegion_'.
  void loadSync();

  DirectBufferedInput* const bufferedInput_;
  IoStatistics* const ioStats_;
  const std::shared_ptr<ReadFileInputStream> input_;
  // The region of 'input' 'this' ranges over.
  const velox::common::Region region_;
  const uint64_t fileNum_;
  std::shared_ptr<cache::ScanTracker> tracker_;
  const cache::TrackingId trackingId_;
  const uint64_t groupId_;

  // Maximum number of bytes read from 'input' at a time.
  const int32_t loadQuantum_;

  // The part of 'region_' that is loaded into 'loadedData_'. Relative to file
  // start.
  velox::common::Region loadedRegion_;

  // The loaded bytes for 'loadedRegion_', held in exactly one of three
  // representations.
  struct LoadedData {
    // Allocation with loaded data. Has space for region.length or loadQuantum_
    // bytes, whichever is less.
    memory::Allocation owned;

    // Contains the data if the range is too small for Allocation.
    std::string tiny;

    // Borrowed slice of the load's shared allocation plus a hold on the owning
    // load, bundled so the pointer and keep-alive can never desync. Null when
    // the bytes are not backed by the shared allocation.
    const char* sharedPtr{nullptr};
    std::shared_ptr<void> sharedHolder;

    // Adopts the buffers of a completed coalesced load. 'load' is retained only
    // when the bytes are a borrowed slice of that load's shared allocation, so
    // the slice cannot outlive its owner.
    void set(LoadedBuffer&& loaded, std::shared_ptr<void> load);

    // True when at most one representation holds bytes. The priority dispatch
    // in loadPosition() picks the first live one, so a double-set would be
    // silently masked rather than caught.
    bool valid() const;

    // Drops the borrowed slice. 'owned' is deliberately left in place: an
    // Allocation is only freeable through its pool, and loadSync() reuses it
    // whenever it is already large enough, so clearing it here would force a
    // fresh allocation on every quantum advance.
    void resetShared();

    bool hasShared() const {
      return sharedPtr != nullptr;
    }
  };
  LoadedData loadedData_;

  // Pointer to start of current run in 'entry->nonContiguousData()' or
  // 'entry->contiguousData()'.
  uint8_t* run_{nullptr};

  // Offset of current run from start of 'loadedData_.owned'
  uint64_t offsetOfRun_;

  // Position of stream relative to 'run_'.
  int offsetInRun_{0};

  // Index of run in 'loadedData_.owned'
  int runIndex_ = -1;

  // Number of valid bytes starting at 'run_'
  uint32_t runSize_ = 0;
  // Position relative to 'region_.offset'.
  uint64_t offsetInRegion_ = 0;

  // Set to true when data is first loaded.
  bool loaded_{false};
};

} // namespace facebook::velox::dwio::common
