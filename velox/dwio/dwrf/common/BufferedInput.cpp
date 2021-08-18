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

#include "velox/dwio/dwrf/common/BufferedInput.h"

#include <fmt/format.h>

DEFINE_bool(wsVRLoad, false, "Use WS VRead API to load");

namespace facebook::velox::dwrf {

using dwio::common::LogType;
using dwio::common::Region;

namespace {

// Because only individual streams are ever enqueue'd we can uniquely identify
// a stream by the filenum and offset.
std::string CacheKey(uint64_t filenum, Region region) {
  return fmt::format("{}_{}", filenum, region.offset);
}

} // namespace

void BufferedInput::load(const LogType logType) {
  // no regions to load
  if (regions_.size() == 0) {
    return;
  }

  offsets_.clear();
  offsets_.reserve(regions_.size());
  buffers_.clear();
  buffers_.reserve(regions_.size());

  // sorting the regions from low to high
  std::sort(regions_.begin(), regions_.end());

  if (UNLIKELY(FLAGS_wsVRLoad)) {
    std::vector<void*> buffers;
    std::vector<Region> regions;
    uint64_t sizeToRead = 0;
    loadWithAction(
        logType,
        [&buffers, &regions, &sizeToRead](
            void* buf, uint64_t length, uint64_t offset, LogType) {
          buffers.push_back(buf);
          regions.emplace_back(offset, length);
          sizeToRead += length;
        });

    // Now we have all buffers and regions, load it in parallel
    input_.vread(buffers, regions, logType);
  } else {
    loadWithAction(
        logType,
        [this](void* buf, uint64_t length, uint64_t offset, LogType type) {
          input_.read(buf, length, offset, type);
        });
  }

  if (dataCacheConfig_) {
    for (auto& region : regions_) {
      const std::string cacheKey = CacheKey(dataCacheConfig_->filenum, region);
      auto buffer = readInternal(region.offset, region.length);
      const std::string_view cacheValue(std::get<0>(buffer), region.length);
      dataCacheConfig_->cache->put(cacheKey, cacheValue);
    }
  }

  // clear the loaded regions
  regions_.clear();
}

std::unique_ptr<SeekableInputStream> BufferedInput::enqueue(
    Region region,
    const StreamIdentifier* /*si*/) {
  if (region.length == 0) {
    return std::make_unique<SeekableArrayInputStream>(
        static_cast<const char*>(nullptr), 0);
  }

  // if the region is already in buffer - such as metadata
  auto ret = readBuffer(region.offset, region.length);
  if (ret) {
    return ret;
  }

  // If we have a DataCache we may be able to load the region from there.
  if (dataCacheConfig_) {
    // TODO: if these allocations show up on a profile, add a 'canGet'
    // method to the DataCache.
    std::unique_ptr<char[]> cachedStream(new char[region.length]);
    const std::string cacheKey = CacheKey(dataCacheConfig_->filenum, region);
    if (dataCacheConfig_->cache->get(
            cacheKey, region.length, cachedStream.get())) {
      auto result = std::make_unique<SeekableArrayInputStream>(
          std::move(cachedStream), region.length);
      return result;
    }
  }

  // push to region pool and give the caller the callback
  regions_.push_back(region);
  return std::make_unique<SeekableArrayInputStream>(
      [region, this]() { return readInternal(region.offset, region.length); });
}

void BufferedInput::loadWithAction(
    const LogType logType,
    std::function<void(void*, uint64_t, uint64_t, LogType)> action) {
  Region last;
  for (const auto& region : regions_) {
    DWIO_ENSURE_GT(region.length, 0, "invalid region");
    if (last.length == 0) {
      // first region
      last = region;
    } else {
      if (!tryMerge(last, region)) {
        readRegion(last, logType, action);
        last = region;
      }
    }
  }

  // handle last region
  readRegion(last, logType, action);
}

bool BufferedInput::tryMerge(
    dwio::common::Region& first,
    const dwio::common::Region& second) {
  DWIO_ENSURE_GE(second.offset, first.offset, "regions should be sorted.");
  int64_t gap = second.offset - first.offset - first.length;

  // compare with 0 since it's comparison in different types
  if (gap < 0 || gap <= kMaxMergeDistance) {
    // ensure try merge will handle duplicate regions (extension==0)
    int64_t extension = gap + second.length;

    // the second region is inside first one if extension is negative
    if (extension > 0) {
      first.length += extension;
      if ((input_.getStats() != nullptr) && gap > 0) {
        input_.getStats()->incRawOverreadBytes(gap);
      }
    }

    return true;
  }

  return false;
}

std::unique_ptr<SeekableInputStream> BufferedInput::readBuffer(
    uint64_t offset,
    uint64_t length) const {
  const auto result = readInternal(offset, length);

  auto size = std::get<1>(result);
  if (size == dwio::common::MAX_UINT64) {
    return {};
  }

  return std::make_unique<SeekableArrayInputStream>(std::get<0>(result), size);
}

std::tuple<const char*, uint64_t> BufferedInput::readInternal(
    uint64_t offset,
    uint64_t length) const {
  // return dummy one for zero length stream
  if (length == 0) {
    return std::make_tuple(nullptr, 0);
  }

  uint64_t index = 0;
  while (index < offsets_.size() && offsets_[index] <= offset) {
    ++index;
  }
  if (index >= 1) {
    index -= 1;
    uint64_t bufferOffset = offsets_[index];
    const auto& buffer = buffers_[index];
    if (bufferOffset + buffer.size() >= offset + length) {
      DWIO_ENSURE_LE(bufferOffset, offset, "Invalid offset for readInternal");
      DWIO_ENSURE_LE(
          (offset - bufferOffset) + length,
          buffer.size(),
          "Invalid readOffset for read Internal ",
          fmt::format(
              "{} {} {} {}", offset, bufferOffset, length, buffer.size()));

      return std::make_tuple(buffer.data() + (offset - bufferOffset), length);
    }
  }

  return std::make_tuple(nullptr, dwio::common::MAX_UINT64);
}

//  static
BufferedInputFactory* BufferedInputFactory::baseFactory() {
  static auto instance = std::make_unique<BufferedInputFactory>();
  return instance.get();
}

} // namespace facebook::velox::dwrf
