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

#include "velox/common/io/IoStatisticsRuntimeStats.h"

#include <fmt/format.h>

#include "velox/common/io/IoStatistics.h"

namespace facebook::velox::io {

namespace {

void addIoCounterMetric(
    IoCounter& counter,
    const std::string& key,
    std::unordered_map<std::string, RuntimeMetric>& runtimeStats) {
  if (counter.count() > 0) {
    runtimeStats.insert({key, RuntimeMetric(counter.count())});
  }
}

void addIoCounterMetric(
    uint64_t value,
    const std::string& key,
    RuntimeCounter::Unit unit,
    std::unordered_map<std::string, RuntimeMetric>& runtimeStats) {
  if (value > 0) {
    runtimeStats.insert({key, RuntimeMetric(value, unit)});
  }
}

void addIoStatsMetric(
    IoCounter& counter,
    const std::string& key,
    RuntimeCounter::Unit unit,
    std::unordered_map<std::string, RuntimeMetric>& runtimeStats) {
  if (counter.count() > 0) {
    runtimeStats.insert(
        {key,
         RuntimeMetric(
             saturateCast(counter.sum()),
             counter.count(),
             saturateCast(counter.min()),
             saturateCast(counter.max()),
             unit)});
  }
}

void addIoLatencyMetric(
    IoCounter& counter,
    const std::string& key,
    std::unordered_map<std::string, RuntimeMetric>& runtimeStats) {
  if (counter.count() > 0) {
    runtimeStats.insert(
        {key,
         RuntimeMetric(
             saturateCast(counter.sum() * 1'000),
             counter.count(),
             saturateCast(counter.min() * 1'000),
             saturateCast(counter.max() * 1'000),
             RuntimeCounter::Unit::kNanos)});
  }
}

} // namespace

void addIoStatsToRuntimeStats(
    IoStatistics& ioStats,
    std::string_view prefix,
    std::unordered_map<std::string, RuntimeMetric>& runtimeStats) {
  auto key = [&](std::string_view name) {
    return prefix.empty() ? std::string(name)
                          : fmt::format("{}.{}", prefix, name);
  };

  addIoLatencyMetric(
      ioStats.queryThreadIoLatencyUs(), key(kIoWaitWallNanos), runtimeStats);
  addIoLatencyMetric(
      ioStats.storageReadLatencyUs(), key(kStorageReadWallNanos), runtimeStats);
  addIoLatencyMetric(
      ioStats.ssdCacheReadLatencyUs(),
      key(kSsdCacheReadWallNanos),
      runtimeStats);
  addIoLatencyMetric(
      ioStats.cacheWaitLatencyUs(), key(kCacheWaitWallNanos), runtimeStats);
  addIoLatencyMetric(
      ioStats.coalescedSsdLoadLatencyUs(),
      key(kCoalescedSsdLoadWallNanos),
      runtimeStats);
  addIoLatencyMetric(
      ioStats.coalescedStorageLoadLatencyUs(),
      key(kCoalescedStorageLoadWallNanos),
      runtimeStats);

  addIoCounterMetric(ioStats.prefetch(), key(kNumPrefetch), runtimeStats);
  addIoStatsMetric(
      ioStats.prefetch(),
      key(kPrefetchBytes),
      RuntimeCounter::Unit::kBytes,
      runtimeStats);
  addIoCounterMetric(
      ioStats.totalScanTimeNs(),
      key(kTotalScanTime),
      RuntimeCounter::Unit::kNanos,
      runtimeStats);
  addIoCounterMetric(
      ioStats.rawOverreadBytes(),
      key(kOverreadBytes),
      RuntimeCounter::Unit::kBytes,
      runtimeStats);

  addIoStatsMetric(
      ioStats.read(),
      key(kStorageReadBytes),
      RuntimeCounter::Unit::kBytes,
      runtimeStats);
  addIoCounterMetric(ioStats.ssdRead(), key(kNumLocalRead), runtimeStats);
  addIoStatsMetric(
      ioStats.ssdRead(),
      key(kLocalReadBytes),
      RuntimeCounter::Unit::kBytes,
      runtimeStats);
  addIoCounterMetric(ioStats.ramHit(), key(kNumRamRead), runtimeStats);
  addIoStatsMetric(
      ioStats.ramHit(),
      key(kRamReadBytes),
      RuntimeCounter::Unit::kBytes,
      runtimeStats);
  addIoStatsMetric(
      ioStats.readGap(),
      key(kReadGapBytes),
      RuntimeCounter::Unit::kBytes,
      runtimeStats);
}

} // namespace facebook::velox::io
