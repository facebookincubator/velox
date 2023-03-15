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

#include <fmt/format.h>

#include "velox/dwio/common/BufferedInput.h"

DEFINE_bool(wsVRLoad, false, "Use WS VRead API to load");

namespace facebook::velox::dwio::common {

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
    loadParallel(buffers, regions, logType);
  } else {
    loadWithAction(
        logType,
        [this](void* buf, uint64_t length, uint64_t offset, LogType type) {
          auto readStartMicros = getCurrentTimeMicro();
          input_->read(buf, length, offset, type);
          if (ioStats_) {
            ioStats_->incRawBytesRead(length);
            ioStats_->incTotalScanTime(
                (getCurrentTimeMicro() - readStartMicros) * 1000);
          }
        });
  }

  // clear the loaded regions
  regions_.clear();
}

std::unique_ptr<SeekableInputStream> BufferedInput::enqueue(
    Region region,
    const dwio::common::StreamIdentifier* /*si*/) {
  if (region.length == 0) {
    return std::make_unique<SeekableArrayInputStream>(
        static_cast<const char*>(nullptr), 0);
  }

  // if the region is already in buffer - such as metadata
  auto ret = readBuffer(region.offset, region.length);
  if (ret) {
    return ret;
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

bool BufferedInput::tryMerge(Region& first, const Region& second) {
  DWIO_ENSURE_GE(second.offset, first.offset, "regions should be sorted.");
  int64_t gap = second.offset - first.offset - first.length;

  // compare with 0 since it's comparison in different types
  if (gap < 0 || gap <= ReaderOptions::kMaxMergeDistance) {
    // ensure try merge will handle duplicate regions (extension==0)
    int64_t extension = gap + second.length;

    // the second region is inside first one if extension is negative
    if (extension > 0) {
      first.length += extension;
      if (ioStats_ && gap > 0) {
        ioStats_->incRawOverreadBytes(gap);
      }
    }

    return true;
  }

  return false;
}

void BufferedInput::splitRegion(
    const uint64_t length,
    const int32_t loadQuantum,
    std::vector<std::tuple<uint64_t, uint64_t>>& range) {
  uint64_t cursor = 0;
  while (cursor + loadQuantum < length) {
    range.emplace_back(cursor, loadQuantum);
    cursor += loadQuantum;
  }

  if ((length - cursor) > (loadQuantum / 2)) {
    range.emplace_back(cursor, (length - cursor));
  } else {
    auto last = range.back();
    range.pop_back();
    range.emplace_back(
        std::get<0>(last), std::get<1>(last) + (length - cursor));
  }
}

void BufferedInput::loadParallel(
    const std::vector<void*>& buffers,
    const std::vector<Region>& regions,
    const LogType purpose) {
  DWIO_ENSURE_NOT_NULL(executor_, "parallel load need executor");
  const auto size = buffers.size();
  DWIO_ENSURE_GT(size, 0, "invalid parameters");
  DWIO_ENSURE_EQ(
      regions.size(), size, "mismatched size of regions and buffers");

  int64_t totalReadLength = 0;
  for (const auto& region : regions) {
    totalReadLength += region.length;
  }
  auto readStartMicros = getCurrentTimeMicro();

  if (size == 1) {
    const auto& region = regions[0];
    input_->read(buffers[0], region.length, region.offset, purpose);
  } else {
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < size; ++i) {
      const auto& region = regions[i];
      const auto& buffer = buffers[i];
      if (region.length > loadQuantum_) {
        std::vector<std::tuple<uint64_t, uint64_t>> ranges;
        splitRegion(region.length, loadQuantum_, ranges);
        for (size_t idx = 0; idx < ranges.size(); idx++) {
          auto cursor = std::get<0>(ranges[idx]);
          auto length = std::get<1>(ranges[idx]);
          auto future = folly::via(
              executor_, [this, buffer, region, cursor, length, &purpose]() {
                char* b = reinterpret_cast<char*>(buffer);
                input_->read(
                    b + cursor, length, region.offset + cursor, purpose);
              });
          futures.push_back(std::move(future));
        }
      } else {
        auto future = folly::via(executor_, [this, buffer, region, &purpose]() {
          input_->read(buffer, region.length, region.offset, purpose);
        });
        futures.push_back(std::move(future));
      }
    }

    for (int64_t i = futures.size() - 1; i >= 0; --i) {
      futures[i].wait();
    }
  }

  if (ioStats_) {
    ioStats_->incRawBytesRead(totalReadLength);
    ioStats_->incTotalScanTime(
        (getCurrentTimeMicro() - readStartMicros) * 1000);
  }
}

std::unique_ptr<SeekableInputStream> BufferedInput::readBuffer(
    uint64_t offset,
    uint64_t length) const {
  const auto result = readInternal(offset, length);

  auto size = std::get<1>(result);
  if (size == MAX_UINT64) {
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

  return std::make_tuple(nullptr, MAX_UINT64);
}

} // namespace facebook::velox::dwio::common
