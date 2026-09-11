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
#include "velox/dwio/nimble/tablet/DataInput.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

#include <folly/executors/QueuedImmediateExecutor.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/CoalesceIo.h"
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

// --- DirectDataInput ---

namespace {

velox::ReadFile* checkedFile(velox::ReadFile* file) {
  NIMBLE_CHECK_NOT_NULL(file);
  return file;
}

uint64_t readAlignment(const velox::ReadFile& file) {
  uint64_t alignment{0};
  return file.directIo(/*alignment=*/alignment) ? alignment : 1;
}

DataInput::Region alignedRegion(DataInput::Region region, uint64_t alignment) {
  NIMBLE_DCHECK_GT(alignment, 0);
  NIMBLE_DCHECK(velox::bits::isPowerOfTwo(alignment));
  const auto alignedOffset = region.offset & ~(alignment - 1);
  const auto alignedEnd =
      (region.offset + region.length + alignment - 1) & ~(alignment - 1);
  return {alignedOffset, alignedEnd - alignedOffset};
}

} // namespace

/*static*/ std::string_view DirectDataInput::stateName(State state) {
  switch (state) {
    case State::kInit:
      return "kInit";
    case State::kEnqueuing:
      return "kEnqueuing";
    case State::kLoaded:
      return "kLoaded";
    default:
      NIMBLE_UNREACHABLE(
          fmt::format("Unknown State: {}", static_cast<int>(state)));
  }
}

DirectDataInput::DirectDataInput(velox::ReadFile* file, const Options& options)
    : file_{checkedFile(file)},
      pool_{options.pool},
      ioStats_{options.ioStats},
      alignment_{readAlignment(*file_)},
      allocAlignment_{std::max(
          alignment_,
          static_cast<uint64_t>(
              velox::memory::MemoryAllocator::kMinAlignment))},
      maxCoalesceDistance_{
          std::min(options.maxCoalesceDistance, kMaxCoalesceDistance)},
      maxCoalesceBytes_{options.maxCoalesceBytes} {
  NIMBLE_CHECK_NOT_NULL(pool_);
  NIMBLE_CHECK_NOT_NULL(ioStats_);
  NIMBLE_CHECK_GT(alignment_, 0, "alignment must be positive");
  NIMBLE_CHECK_LE(
      alignment_,
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
      "alignment exceeds the memory allocator limit");
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(alignment_), "alignment must be a power of 2");
}

void DirectDataInput::reserve(uint32_t numRegions) {
  NIMBLE_CHECK_EQ(state_, State::kInit);
  regions_.reserve(numRegions);
  bufferRefs_.reserve(numRegions);
}

void DirectDataInput::startGroup(std::optional<Region> /*groupRegion*/) {
  NIMBLE_CHECK_NE(state_, State::kLoaded);
  NIMBLE_CHECK(
      groupOffsets_.empty() ||
          groupOffsets_.back() < static_cast<uint32_t>(regions_.size()),
      "Previous group is empty");
  state_ = State::kEnqueuing;
  groupOffsets_.push_back(static_cast<uint32_t>(regions_.size()));
}

uint32_t DirectDataInput::enqueue(Region region) {
  NIMBLE_CHECK_EQ(state_, State::kEnqueuing);
  NIMBLE_CHECK_GT(region.length, 0, "Read region length must be positive");
  const auto index = static_cast<uint32_t>(bufferRefs_.size());
  bufferRefs_.emplace_back();
  regions_.emplace_back(EnqueuedRegion{index, region});
  return index;
}

/*static*/ std::pair<uint64_t, uint64_t> DirectDataInput::computeIoSizes(
    std::vector<IoGroup>& ioGroups) {
  uint64_t readBytes = 0;
  uint64_t payloadBytes = 0;
  for (auto& group : ioGroups) {
    group.bufferOffset = readBytes;
    readBytes += group.readSize;
    payloadBytes += group.payloadSize;
  }
  return {readBytes, payloadBytes};
}

/*static*/ std::pair<uint64_t, uint64_t> DirectDataInput::computePayloadSize(
    std::span<const EnqueuedRegion> sortedRegions) {
  NIMBLE_CHECK(!sortedRegions.empty(), "IO group must contain a region");

  const auto& firstRegion = sortedRegions.front().region;
  uint64_t prevRegionEnd{firstRegion.offset + firstRegion.length};
  uint64_t payloadSize{firstRegion.length};
  for (size_t i = 1; i < sortedRegions.size(); ++i) {
    const auto& region = sortedRegions[i].region;
    const auto regionEnd = region.offset + region.length;
    if (regionEnd == prevRegionEnd) {
      const auto& prevRegion = sortedRegions[i - 1].region;
      // coalesceIo only keeps exact duplicate ranges in the same IO group so
      // they can share the physical read. Contained and partially overlapping
      // ranges are emitted as separate groups.
      NIMBLE_CHECK_EQ(
          region.offset,
          prevRegion.offset,
          "Only exact duplicate ranges can end at the previous range end");
      NIMBLE_CHECK_EQ(
          region.length,
          prevRegion.length,
          "Only exact duplicate ranges can end at the previous range end");
      continue;
    }

    NIMBLE_CHECK_GE(
        region.offset,
        prevRegionEnd,
        "IO group cannot contain overlapping ranges");
    payloadSize += region.length;
    prevRegionEnd = regionEnd;
  }
  return {payloadSize, prevRegionEnd};
}

std::vector<DirectDataInput::IoGroup> DirectDataInput::computeIoGroups(
    const std::vector<EnqueuedRegion>& sortedRegions) {
  std::vector<int32_t> items(sortedRegions.size());
  std::iota(items.begin(), items.end(), 0);

  int64_t coalescedBytes = 0;
  std::vector<IoGroup> ioGroups;
  ioGroups.reserve(sortedRegions.size());

  const auto coalesceStats = velox::coalesceIo<int32_t, char>(
      items,
      /*maxGap=*/maxCoalesceDistance_,
      /*rangesPerIo=*/std::numeric_limits<int32_t>::max(),
      /*offsetFunc=*/
      [&](int32_t i) -> uint64_t { return sortedRegions[i].region.offset; },
      /*sizeFunc=*/
      [&](int32_t i) -> int32_t {
        return static_cast<int32_t>(sortedRegions[i].region.length);
      },
      /*numRanges=*/
      [&](int32_t i) -> int32_t {
        const auto size = static_cast<int64_t>(sortedRegions[i].region.length);
        if (coalescedBytes + size > maxCoalesceBytes_) {
          coalescedBytes = 0;
          return velox::kNoCoalesce;
        }
        coalescedBytes += size;
        return 1;
      },
      /*addRanges=*/
      [](const int32_t& /*i*/, std::vector<char>& ranges) {
        ranges.push_back(0);
      },
      /*skipRange=*/
      [](int32_t /*gap*/, std::vector<char>& /*ranges*/) {},
      /*ioFunc=*/
      [&](const std::vector<int32_t>& /*items*/,
          int32_t begin,
          int32_t end,
          uint64_t /*offset*/,
          const std::vector<char>& /*ranges*/) {
        coalescedBytes = 0;
        IoGroup group;
        const auto groupRegions = std::span<const EnqueuedRegion>(
            &sortedRegions[begin], static_cast<size_t>(end - begin));
        const auto [payloadSize, lastEnd] = computePayloadSize(groupRegions);
        const auto readRegion = alignedRegion(
            Region{
                sortedRegions[begin].region.offset,
                lastEnd - sortedRegions[begin].region.offset,
            },
            alignment_);
        group.readOffset = readRegion.offset;
        group.readSize = readRegion.length;
        group.payloadSize = payloadSize;
        group.regions = groupRegions;
        ioGroups.emplace_back(group);
      });

  ioStats_->readGap().merge(coalesceStats.gaps);
  ioStats_->incDuplicateRead(
      coalesceStats.duplicateRegions, coalesceStats.duplicateBytes);
  return ioGroups;
}

void DirectDataInput::populateBufferRefs(
    const std::vector<IoGroup>& ioGroups,
    const char* buffer) {
  for (const auto& group : ioGroups) {
    // group.regions is sorted by offset and then enqueueIndex. Duplicate read
    // regions are therefore contiguous, with the earliest enqueued request
    // first. The BufferRef for the first request in each run is canonical:
    // it points canonicalIndex to itself, and later duplicates point
    // canonicalIndex to it while sharing the same buffer slice. Since enqueue
    // order matches consumer order, the canonical ref is processed before any
    // duplicate that reuses it.
    const EnqueuedRegion* shared = nullptr;
    for (const auto& region : group.regions) {
      auto& bufferRef = bufferRefs_[region.enqueueIndex];
      bufferRef.data = buffer + group.bufferOffset +
          (region.region.offset - group.readOffset);
      bufferRef.length = region.region.length;
      if (shared != nullptr && region.sameRegionAs(*shared)) {
        bufferRef.canonicalIndex = shared->enqueueIndex;
      } else {
        bufferRef.canonicalIndex = region.enqueueIndex;
        shared = &region;
      }
    }
  }
}

std::pair<char*, DataInput::Handle> DirectDataInput::allocateBuffer(
    uint64_t bytes) {
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::DirectDataInput::allocateBuffer", &bytes);
  auto* buffer = static_cast<char*>(
      pool_->allocateAligned(static_cast<int64_t>(bytes), allocAlignment_));
  NIMBLE_DCHECK(
      reinterpret_cast<uintptr_t>(buffer) % alignment_ == 0,
      "allocateAligned returned unaligned buffer");
  auto handle = Handle(
      buffer, [pool = pool_, bytes, allocAlignment = allocAlignment_](void* p) {
        pool->freeAligned(p, static_cast<int64_t>(bytes), allocAlignment);
      });
  return {buffer, std::move(handle)};
}

void DirectDataInput::executeIoGroups(
    std::vector<IoGroup>& ioGroups,
    char* buffer,
    uint64_t bufferSize) {
  NIMBLE_CHECK(!ioGroups.empty());
  std::vector<velox::common::Region> readRegions;
  readRegions.reserve(ioGroups.size());
  std::vector<folly::Range<char*>> readBuffers;
  readBuffers.reserve(ioGroups.size());
  for (const auto& ioGroup : ioGroups) {
    readRegions.emplace_back(ioGroup.readOffset, ioGroup.readSize);
    readBuffers.emplace_back(
        buffer + ioGroup.bufferOffset, static_cast<size_t>(ioGroup.readSize));
  }
  const auto bytesRead = file_->preadv(
      folly::Range<const velox::common::Region*>(
          readRegions.data(), readRegions.size()),
      folly::Range<const folly::Range<char*>*>(
          readBuffers.data(), readBuffers.size()));
  NIMBLE_CHECK_EQ(bytesRead, bufferSize, "preadv returned a short read");
}

DataInput::Handle DirectDataInput::load() {
  NIMBLE_CHECK_EQ(state_, State::kEnqueuing);
  NIMBLE_CHECK(!bufferRefs_.empty(), "No regions enqueued");
  state_ = State::kLoaded;

  // Sort within each group segment in-place. Groups are expected to be
  // enqueued in non-overlapping file order, so boundary checks ensure the
  // flat vector remains globally sorted without sorting across groups.
  const auto numGroups = groupOffsets_.size();
  uint64_t prevGroupEnd{0};
  for (size_t i = 0; i < numGroups; ++i) {
    const auto start = groupOffsets_[i];
    const auto end = (i + 1 < numGroups)
        ? groupOffsets_[i + 1]
        : static_cast<uint32_t>(regions_.size());
    NIMBLE_CHECK_LT(start, end, "Read group must contain a region");
    std::sort(
        regions_.begin() + start,
        regions_.begin() + end,
        [](const EnqueuedRegion& a, const EnqueuedRegion& b) {
          if (a.region.offset != b.region.offset) {
            return a.region.offset < b.region.offset;
          }
          // Tiebreak equal extents by enqueue index so the first member of each
          // duplicate run is its stored copy (smallest enqueue index = earliest
          // processed by the consumer).
          return a.enqueueIndex < b.enqueueIndex;
        });

    if (i > 0) {
      NIMBLE_CHECK_GE(
          regions_[start].region.offset,
          prevGroupEnd,
          "Read groups must be sorted and non-overlapping by file offset");
    }

    const auto& lastRegion = regions_[end - 1].region;
    prevGroupEnd = lastRegion.offset + lastRegion.length;
  }

  auto ioGroups = computeIoGroups(regions_);

  const auto [readBytes, payloadBytes] = computeIoSizes(ioGroups);
  NIMBLE_CHECK_GE(readBytes, payloadBytes);
  NIMBLE_CHECK_EQ(readBytes % alignment_, 0);

  auto [buffer, handle] = allocateBuffer(readBytes);

  uint64_t ioUs{0};
  {
    velox::MicrosecondTimer ioTimer(&ioUs);
    executeIoGroups(ioGroups, buffer, readBytes);
  }

  for (const auto& group : ioGroups) {
    ioStats_->read().increment(group.readSize);
    ioStats_->incRawBytesRead(group.payloadSize);
    ioStats_->incRawOverreadBytes(group.readSize - group.payloadSize);
  }
  ioStats_->storageReadLatencyUs().increment(ioUs);
  ioStats_->incTotalScanTimeNs(ioUs * 1'000);

  populateBufferRefs(ioGroups, buffer);

  regions_.clear();
  return handle;
}

const DataInput::BufferRef& DirectDataInput::bufferRef(uint32_t index) const {
  NIMBLE_CHECK_EQ(state_, State::kLoaded);
  NIMBLE_DCHECK_LT(index, bufferRefs_.size());
  return bufferRefs_[index];
}

void DirectDataInput::clear() {
  state_ = State::kInit;
  regions_.clear();
  groupOffsets_.clear();
  bufferRefs_.clear();
}

// --- CachedDataInput ---

CachedDataInput::CachedDataInput(velox::ReadFile* file, const Options& options)
    : file_{checkedFile(file)},
      pool_{options.pool},
      cache_{options.cache},
      fileId_{options.fileId},
      ioStats_{options.ioStats},
      alignment_{readAlignment(*file_)},
      allocationAlignment_{std::max(
          alignment_,
          static_cast<uint64_t>(
              velox::memory::MemoryAllocator::kMinAlignment))} {
  NIMBLE_CHECK_NOT_NULL(pool_);
  NIMBLE_CHECK_NOT_NULL(cache_);
  NIMBLE_CHECK_NOT_NULL(ioStats_);
  NIMBLE_CHECK_GT(alignment_, 0, "alignment must be positive");
  NIMBLE_CHECK_LE(
      alignment_,
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
      "alignment exceeds the memory allocator limit");
  NIMBLE_CHECK(
      velox::bits::isPowerOfTwo(alignment_), "alignment must be a power of 2");
}

void CachedDataInput::reserve(uint32_t numRegions) {
  NIMBLE_CHECK(!loaded_, "Cannot reserve after load");
  regions_.reserve(numRegions);
  bufferRefs_.reserve(numRegions);
}

void CachedDataInput::startGroup(std::optional<Region> groupRegion) {
  NIMBLE_CHECK(!loaded_, "Cannot start a group after load");
  NIMBLE_CHECK(
      groups_.empty() || groups_.back().enqueuedRegionOffset < regions_.size(),
      "Previous group is empty");
  NIMBLE_CHECK(groupRegion.has_value(), "Cached group region is required");
  NIMBLE_CHECK_GT(
      groupRegion->length, 0, "Cached group length must be positive");
  NIMBLE_CHECK_LE(
      groupRegion->length,
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
      "Cached group exceeds the maximum cache entry size");
  if (!groups_.empty()) {
    const auto& previous = groups_.back().groupRegion;
    NIMBLE_CHECK_GE(
        groupRegion->offset,
        previous.offset + previous.length,
        "Cached groups must be sorted and non-overlapping by file offset");
  }
  groups_.push_back(
      Group{
          .groupRegion = *groupRegion,
          .enqueuedRegionOffset = static_cast<uint32_t>(regions_.size()),
      });
}

uint32_t CachedDataInput::enqueue(Region region) {
  NIMBLE_CHECK(!loaded_, "Cannot enqueue after load");
  NIMBLE_CHECK(!groups_.empty(), "startGroup() must precede enqueue()");
  NIMBLE_CHECK_GT(region.length, 0, "Read region length must be positive");
  const auto& groupRegion = groups_.back().groupRegion;
  NIMBLE_CHECK_GE(
      region.offset,
      groupRegion.offset,
      "Read region begins before its cached group");
  NIMBLE_CHECK_LE(
      region.offset + region.length,
      groupRegion.offset + groupRegion.length,
      "Read region extends beyond its cached group");
  const auto index = static_cast<uint32_t>(bufferRefs_.size());
  bufferRefs_.emplace_back();
  regions_.push_back(EnqueuedRegion{index, region});
  return index;
}

uint64_t CachedDataInput::populateBufferRefs(
    const Group& group,
    uint32_t endRegion,
    const char* buffer) {
  std::vector<const EnqueuedRegion*> sortedRegions;
  sortedRegions.reserve(endRegion - group.enqueuedRegionOffset);
  for (uint32_t i = group.enqueuedRegionOffset; i < endRegion; ++i) {
    sortedRegions.push_back(&regions_[i]);
  }
  std::sort(
      sortedRegions.begin(),
      sortedRegions.end(),
      [](const EnqueuedRegion* lhs, const EnqueuedRegion* rhs) {
        if (lhs->region.offset != rhs->region.offset) {
          return lhs->region.offset < rhs->region.offset;
        }
        if (lhs->region.length != rhs->region.length) {
          return lhs->region.length < rhs->region.length;
        }
        return lhs->enqueueIndex < rhs->enqueueIndex;
      });

  uint64_t payloadBytes{0};
  uint64_t coveredEnd{group.groupRegion.offset};
  const EnqueuedRegion* canonicalRegion{nullptr};
  for (const auto* enqueued : sortedRegions) {
    auto& ref = bufferRefs_[enqueued->enqueueIndex];
    ref.data = buffer + enqueued->region.offset - group.groupRegion.offset;
    ref.length = enqueued->region.length;
    if (canonicalRegion != nullptr &&
        canonicalRegion->region.offset == enqueued->region.offset &&
        canonicalRegion->region.length == enqueued->region.length) {
      ref.canonicalIndex = canonicalRegion->enqueueIndex;
    } else {
      canonicalRegion = enqueued;
      ref.canonicalIndex = enqueued->enqueueIndex;
    }

    const auto regionEnd = enqueued->region.offset + enqueued->region.length;
    if (regionEnd > coveredEnd) {
      payloadBytes += regionEnd - std::max(coveredEnd, enqueued->region.offset);
      coveredEnd = regionEnd;
    }
  }
  return payloadBytes;
}

void CachedDataInput::loadCacheMissGroups(
    std::span<const CacheMissGroup> cacheMissGroups,
    uint64_t stagingBufferSize,
    const std::vector<velox::cache::CachePin>& cachePins) {
  char* stagingBuffer{nullptr};
  Handle stagingHandle;
  if (alignment_ > 1) {
    stagingBuffer = static_cast<char*>(pool_->allocateAligned(
        static_cast<int64_t>(stagingBufferSize),
        static_cast<uint32_t>(allocationAlignment_)));
    stagingHandle = Handle(
        stagingBuffer,
        [pool = pool_,
         stagingBufferSize,
         allocationAlignment = allocationAlignment_](void* buffer) {
          pool->freeAligned(
              buffer,
              static_cast<int64_t>(stagingBufferSize),
              static_cast<uint32_t>(allocationAlignment));
        });
  }

  std::vector<Region> readRegions;
  std::vector<folly::Range<char*>> readBuffers;
  readRegions.reserve(cacheMissGroups.size());
  readBuffers.reserve(cacheMissGroups.size());
  uint64_t expectedReadBytes{0};
  for (const auto& cacheMissGroup : cacheMissGroups) {
    readRegions.push_back(cacheMissGroup.readRegion);
    auto* entry = cachePins.at(cacheMissGroup.groupIndex).checkedEntry();
    auto* destination = alignment_ == 1
        ? entry->contiguousData()
        : stagingBuffer + cacheMissGroup.stagingBufferOffset;
    readBuffers.emplace_back(destination, cacheMissGroup.readRegion.length);
    expectedReadBytes += cacheMissGroup.readRegion.length;
  }

  uint64_t storageReadUs{0};
  uint64_t bytesRead{0};
  {
    velox::MicrosecondTimer timer(&storageReadUs);
    bytesRead = file_->preadv(
        folly::Range<const Region*>(readRegions.data(), readRegions.size()),
        folly::Range<const folly::Range<char*>*>(
            readBuffers.data(), readBuffers.size()));
  }
  NIMBLE_CHECK_EQ(bytesRead, expectedReadBytes, "preadv returned a short read");
  ioStats_->queryThreadIoLatencyUs().increment(storageReadUs);
  ioStats_->storageReadLatencyUs().increment(storageReadUs);
  ioStats_->incTotalScanTimeNs(static_cast<int64_t>(storageReadUs) * 1'000);

  for (const auto& cacheMissGroup : cacheMissGroups) {
    auto* entry = cachePins.at(cacheMissGroup.groupIndex).checkedEntry();
    const auto& group = groups_[cacheMissGroup.groupIndex];
    if (alignment_ > 1) {
      std::memcpy(
          entry->contiguousData(),
          stagingBuffer + cacheMissGroup.stagingBufferOffset +
              group.groupRegion.offset - cacheMissGroup.readRegion.offset,
          group.groupRegion.length);
    }
    ioStats_->read().increment(cacheMissGroup.readRegion.length);
    ioStats_->incRawOverreadBytes(
        static_cast<int64_t>(
            cacheMissGroup.readRegion.length - cacheMissGroup.payloadBytes));
    entry->setExclusiveToShared(/*ssdSavable=*/false);
  }
}

DataInput::Handle CachedDataInput::load() {
  NIMBLE_CHECK(!loaded_, "Data is already loaded");
  NIMBLE_CHECK(!bufferRefs_.empty(), "No regions enqueued");
  NIMBLE_CHECK(
      groups_.back().enqueuedRegionOffset < regions_.size(),
      "Read group must contain a region");
  loaded_ = true;

  // Releasing a pin removes an unfinished exclusive cache entry and wakes its
  // waiters if any subsequent cache-fill work throws.
  auto cachePins = std::make_shared<std::vector<velox::cache::CachePin>>();
  cachePins->reserve(groups_.size());
  std::vector<CacheMissGroup> cacheMissGroups;
  cacheMissGroups.reserve(groups_.size());
  uint64_t stagingBufferSize{0};
  for (size_t groupIndex{0}; groupIndex < groups_.size(); ++groupIndex) {
    const auto& group = groups_[groupIndex];
    const auto endRegion = groupIndex + 1 < groups_.size()
        ? groups_[groupIndex + 1].enqueuedRegionOffset
        : static_cast<uint32_t>(regions_.size());

    velox::cache::CachePin pin;
    do {
      folly::SemiFuture<bool> waitFuture{false};
      pin = cache_->findOrCreate(
          velox::cache::RawFileCacheKey{fileId_, group.groupRegion.offset},
          group.groupRegion.length,
          /*contiguous=*/true,
          &waitFuture);
      if (pin.empty()) {
        NIMBLE_CHECK(waitFuture.valid(), "Cache wait future is invalid");
        uint64_t waitUs{0};
        {
          velox::MicrosecondTimer timer(&waitUs);
          std::move(waitFuture)
              .via(&folly::QueuedImmediateExecutor::instance())
              .wait();
        }
        ioStats_->queryThreadIoLatencyUs().increment(waitUs);
        ioStats_->cacheWaitLatencyUs().increment(waitUs);
        // The future only signals that the competing exclusive load finished;
        // findOrCreate must run again to acquire a shared or exclusive pin.
      }
    } while (pin.empty());

    auto* entry = pin.checkedEntry();
    const bool firstUse = entry->getAndClearFirstUseFlag();
    const bool loadedFromStorage = entry->isExclusive();
    if (!loadedFromStorage && !firstUse) {
      ioStats_->ramHit().increment(group.groupRegion.length);
    }

    const auto payloadBytes =
        populateBufferRefs(group, endRegion, entry->contiguousData());
    NIMBLE_CHECK_LE(payloadBytes, group.groupRegion.length);
    ioStats_->incRawBytesRead(static_cast<int64_t>(payloadBytes));
    if (loadedFromStorage) {
      const auto readRegion = alignedRegion(group.groupRegion, alignment_);
      cacheMissGroups.push_back(
          CacheMissGroup{
              .groupIndex = groupIndex,
              .readRegion = readRegion,
              .stagingBufferOffset = stagingBufferSize,
              .payloadBytes = payloadBytes,
          });
      stagingBufferSize += readRegion.length;
    }
    cachePins->push_back(std::move(pin));
  }

  if (!cacheMissGroups.empty()) {
    loadCacheMissGroups(cacheMissGroups, stagingBufferSize, *cachePins);
  }

  regions_.clear();
  return std::static_pointer_cast<void>(cachePins);
}

const DataInput::BufferRef& CachedDataInput::bufferRef(uint32_t index) const {
  NIMBLE_CHECK(loaded_, "Data has not been loaded");
  NIMBLE_CHECK_LT(index, bufferRefs_.size());
  return bufferRefs_[index];
}

void CachedDataInput::clear() {
  loaded_ = false;
  regions_.clear();
  groups_.clear();
  bufferRefs_.clear();
}

} // namespace facebook::nimble
