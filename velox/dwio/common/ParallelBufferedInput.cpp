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

#include "velox/dwio/common/ParallelBufferedInput.h"

namespace facebook::velox::dwio::common {

void ParallelBufferedInput::load(const LogType logType) {
  if (regions_.size() == 0) {
    return;
  }
  DWIO_ENSURE_NOT_NULL(executor_, "parallel load need executor");

  offsets_.clear();
  offsets_.reserve(regions_.size());
  buffers_.clear();
  buffers_.reserve(regions_.size());

  // Sorting the regions from low to high
  std::sort(regions_.begin(), regions_.end());

  uint64_t sizeBeforeMerge = 0;
  for (const auto& region : regions_) {
    sizeBeforeMerge += region.length;
  }

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

  auto readStartMicros = getCurrentTimeMicro();

  // Now we have all buffers and regions, load it in parallel
  loadParallel(buffers, regions, logType);

  if (ioStats_) {
    ioStats_->incRawBytesRead(sizeToRead);
    ioStats_->incRawOverreadBytes(sizeToRead - sizeBeforeMerge);
    ioStats_->incTotalScanTime(
        (getCurrentTimeMicro() - readStartMicros) * 1000);
  }

  // Clear the loaded regions
  regions_.clear();
}

void ParallelBufferedInput::splitRegion(
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

void ParallelBufferedInput::loadParallel(
    const std::vector<void*>& buffers,
    const std::vector<Region>& regions,
    const LogType purpose) {
  const auto size = buffers.size();
  DWIO_ENSURE_GT(size, 0, "invalid parameters");
  DWIO_ENSURE_EQ(
      regions.size(), size, "mismatched size of regions and buffers");

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
}
} // namespace facebook::velox::dwio::common
