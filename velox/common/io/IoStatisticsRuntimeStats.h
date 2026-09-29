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

#include <string>
#include <string_view>
#include <unordered_map>

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox::io {

class IoStatistics;

inline constexpr std::string_view kIoWaitWallNanos{"ioWaitWallNanos"};
inline constexpr std::string_view kStorageReadWallNanos{"storageReadWallNanos"};
inline constexpr std::string_view kSsdCacheReadWallNanos{
    "ssdCacheReadWallNanos"};
inline constexpr std::string_view kCacheWaitWallNanos{"cacheWaitWallNanos"};
inline constexpr std::string_view kCoalescedSsdLoadWallNanos{
    "coalescedSsdLoadWallNanos"};
inline constexpr std::string_view kCoalescedStorageLoadWallNanos{
    "coalescedStorageLoadWallNanos"};
inline constexpr std::string_view kNumPrefetch{"numPrefetch"};
inline constexpr std::string_view kPrefetchBytes{"prefetchBytes"};
inline constexpr std::string_view kTotalScanTime{"totalScanTime"};
inline constexpr std::string_view kOverreadBytes{"overreadBytes"};
inline constexpr std::string_view kStorageReadBytes{"storageReadBytes"};
inline constexpr std::string_view kNumLocalRead{"numLocalRead"};
inline constexpr std::string_view kLocalReadBytes{"localReadBytes"};
inline constexpr std::string_view kNumRamRead{"numRamRead"};
inline constexpr std::string_view kRamReadBytes{"ramReadBytes"};
inline constexpr std::string_view kReadGapBytes{"readGapBytes"};

/// Adds IoStatistics counters and latency histograms to runtime stats.
/// The optional prefix distinguishes data and metadata I/O.
void addIoStatsToRuntimeStats(
    IoStatistics& ioStats,
    std::string_view prefix,
    std::unordered_map<std::string, RuntimeMetric>& runtimeStats);

} // namespace facebook::velox::io
