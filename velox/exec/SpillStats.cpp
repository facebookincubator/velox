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

#include "velox/exec/SpillStats.h"
#include <folly/system/HardwareConcurrency.h>
#include <sstream>
#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox::exec {
namespace {
std::vector<SpillStats>& allSpillStats() {
  static std::vector<SpillStats> spillStatsList(folly::hardware_concurrency());
  return spillStatsList;
}

SpillStats& localSpillStats() {
  const auto idx = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto& spillStatsVector = allSpillStats();
  return spillStatsVector[idx % spillStatsVector.size()];
}
} // namespace

SpillStats::SpillStats(
    uint64_t _spillRuns,
    uint64_t _spilledInputBytes,
    uint64_t _spilledBytes,
    uint64_t _spilledRows,
    uint32_t _spilledPartitions,
    uint64_t _spilledFiles,
    uint64_t _spillFillTimeNanos,
    uint64_t _spillSortTimeNanos,
    uint64_t _spillExtractVectorTimeNanos,
    uint64_t _spillSerializationTimeNanos,
    uint64_t _spillWrites,
    uint64_t _spillFlushTimeNanos,
    uint64_t _spillWriteTimeNanos,
    uint64_t _spillMaxLevelExceededCount,
    uint64_t _spillReadBytes,
    uint64_t _spillReads,
    uint64_t _spillReadTimeNanos,
    uint64_t _spillDeserializationTimeNanos) {
  spillRuns.store(_spillRuns, std::memory_order_relaxed);
  spilledInputBytes.store(_spilledInputBytes, std::memory_order_relaxed);
  spilledBytes.store(_spilledBytes, std::memory_order_relaxed);
  spilledRows.store(_spilledRows, std::memory_order_relaxed);
  spilledPartitions.store(_spilledPartitions, std::memory_order_relaxed);
  spilledFiles.store(_spilledFiles, std::memory_order_relaxed);
  spillFillTimeNanos.store(_spillFillTimeNanos, std::memory_order_relaxed);
  spillSortTimeNanos.store(_spillSortTimeNanos, std::memory_order_relaxed);
  spillExtractVectorTimeNanos.store(
      _spillExtractVectorTimeNanos, std::memory_order_relaxed);
  spillSerializationTimeNanos.store(
      _spillSerializationTimeNanos, std::memory_order_relaxed);
  spillWrites.store(_spillWrites, std::memory_order_relaxed);
  spillFlushTimeNanos.store(_spillFlushTimeNanos, std::memory_order_relaxed);
  spillWriteTimeNanos.store(_spillWriteTimeNanos, std::memory_order_relaxed);
  spillMaxLevelExceededCount.store(
      _spillMaxLevelExceededCount, std::memory_order_relaxed);
  spillReadBytes.store(_spillReadBytes, std::memory_order_relaxed);
  spillReads.store(_spillReads, std::memory_order_relaxed);
  spillReadTimeNanos.store(_spillReadTimeNanos, std::memory_order_relaxed);
  spillDeserializationTimeNanos.store(
      _spillDeserializationTimeNanos, std::memory_order_relaxed);
}

SpillStats::SpillStats(const SpillStats& other) {
  copyFrom(other);
}

SpillStats& SpillStats::operator=(const SpillStats& other) {
  if (this != &other) {
    copyFrom(other);
  }
  return *this;
}

SpillStats& SpillStats::operator+=(const SpillStats& other) {
  spillRuns.fetch_add(
      other.spillRuns.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledInputBytes.fetch_add(
      other.spilledInputBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledBytes.fetch_add(
      other.spilledBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledRows.fetch_add(
      other.spilledRows.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledPartitions.fetch_add(
      other.spilledPartitions.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledFiles.fetch_add(
      other.spilledFiles.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillFillTimeNanos.fetch_add(
      other.spillFillTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillSortTimeNanos.fetch_add(
      other.spillSortTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillExtractVectorTimeNanos.fetch_add(
      other.spillExtractVectorTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillSerializationTimeNanos.fetch_add(
      other.spillSerializationTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillWrites.fetch_add(
      other.spillWrites.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillFlushTimeNanos.fetch_add(
      other.spillFlushTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillWriteTimeNanos.fetch_add(
      other.spillWriteTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillMaxLevelExceededCount.fetch_add(
      other.spillMaxLevelExceededCount.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillReadBytes.fetch_add(
      other.spillReadBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillReads.fetch_add(
      other.spillReads.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillReadTimeNanos.fetch_add(
      other.spillReadTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillDeserializationTimeNanos.fetch_add(
      other.spillDeserializationTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  ioStats.merge(other.ioStats);
  return *this;
}

bool SpillStats::empty() const {
  return spilledBytes.load(std::memory_order_relaxed) == 0;
}

void SpillStats::copyFrom(const SpillStats& other) {
  spillRuns.store(
      other.spillRuns.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledInputBytes.store(
      other.spilledInputBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledBytes.store(
      other.spilledBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledRows.store(
      other.spilledRows.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledPartitions.store(
      other.spilledPartitions.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spilledFiles.store(
      other.spilledFiles.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillFillTimeNanos.store(
      other.spillFillTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillSortTimeNanos.store(
      other.spillSortTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillExtractVectorTimeNanos.store(
      other.spillExtractVectorTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillSerializationTimeNanos.store(
      other.spillSerializationTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillWrites.store(
      other.spillWrites.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillFlushTimeNanos.store(
      other.spillFlushTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillWriteTimeNanos.store(
      other.spillWriteTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillMaxLevelExceededCount.store(
      other.spillMaxLevelExceededCount.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillReadBytes.store(
      other.spillReadBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillReads.store(
      other.spillReads.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillReadTimeNanos.store(
      other.spillReadTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  spillDeserializationTimeNanos.store(
      other.spillDeserializationTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  ioStats.merge(other.ioStats);
}

SpillStats SpillStats::operator-(const SpillStats& other) const {
  SpillStats result;
  result.spillRuns.store(
      spillRuns.load(std::memory_order_relaxed) -
          other.spillRuns.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spilledInputBytes.store(
      spilledInputBytes.load(std::memory_order_relaxed) -
          other.spilledInputBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spilledBytes.store(
      spilledBytes.load(std::memory_order_relaxed) -
          other.spilledBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spilledRows.store(
      spilledRows.load(std::memory_order_relaxed) -
          other.spilledRows.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spilledPartitions.store(
      spilledPartitions.load(std::memory_order_relaxed) -
          other.spilledPartitions.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spilledFiles.store(
      spilledFiles.load(std::memory_order_relaxed) -
          other.spilledFiles.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillFillTimeNanos.store(
      spillFillTimeNanos.load(std::memory_order_relaxed) -
          other.spillFillTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillSortTimeNanos.store(
      spillSortTimeNanos.load(std::memory_order_relaxed) -
          other.spillSortTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillExtractVectorTimeNanos.store(
      spillExtractVectorTimeNanos.load(std::memory_order_relaxed) -
          other.spillExtractVectorTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillSerializationTimeNanos.store(
      spillSerializationTimeNanos.load(std::memory_order_relaxed) -
          other.spillSerializationTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillWrites.store(
      spillWrites.load(std::memory_order_relaxed) -
          other.spillWrites.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillFlushTimeNanos.store(
      spillFlushTimeNanos.load(std::memory_order_relaxed) -
          other.spillFlushTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillWriteTimeNanos.store(
      spillWriteTimeNanos.load(std::memory_order_relaxed) -
          other.spillWriteTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillMaxLevelExceededCount.store(
      spillMaxLevelExceededCount.load(std::memory_order_relaxed) -
          other.spillMaxLevelExceededCount.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillReadBytes.store(
      spillReadBytes.load(std::memory_order_relaxed) -
          other.spillReadBytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillReads.store(
      spillReads.load(std::memory_order_relaxed) -
          other.spillReads.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillReadTimeNanos.store(
      spillReadTimeNanos.load(std::memory_order_relaxed) -
          other.spillReadTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  result.spillDeserializationTimeNanos.store(
      spillDeserializationTimeNanos.load(std::memory_order_relaxed) -
          other.spillDeserializationTimeNanos.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  return result;
}

bool SpillStats::operator==(const SpillStats& other) const {
  return spillRuns.load(std::memory_order_relaxed) ==
      other.spillRuns.load(std::memory_order_relaxed) &&
      spilledInputBytes.load(std::memory_order_relaxed) ==
      other.spilledInputBytes.load(std::memory_order_relaxed) &&
      spilledBytes.load(std::memory_order_relaxed) ==
      other.spilledBytes.load(std::memory_order_relaxed) &&
      spilledRows.load(std::memory_order_relaxed) ==
      other.spilledRows.load(std::memory_order_relaxed) &&
      spilledPartitions.load(std::memory_order_relaxed) ==
      other.spilledPartitions.load(std::memory_order_relaxed) &&
      spilledFiles.load(std::memory_order_relaxed) ==
      other.spilledFiles.load(std::memory_order_relaxed) &&
      spillFillTimeNanos.load(std::memory_order_relaxed) ==
      other.spillFillTimeNanos.load(std::memory_order_relaxed) &&
      spillSortTimeNanos.load(std::memory_order_relaxed) ==
      other.spillSortTimeNanos.load(std::memory_order_relaxed) &&
      spillExtractVectorTimeNanos.load(std::memory_order_relaxed) ==
      other.spillExtractVectorTimeNanos.load(std::memory_order_relaxed) &&
      spillSerializationTimeNanos.load(std::memory_order_relaxed) ==
      other.spillSerializationTimeNanos.load(std::memory_order_relaxed) &&
      spillWrites.load(std::memory_order_relaxed) ==
      other.spillWrites.load(std::memory_order_relaxed) &&
      spillFlushTimeNanos.load(std::memory_order_relaxed) ==
      other.spillFlushTimeNanos.load(std::memory_order_relaxed) &&
      spillWriteTimeNanos.load(std::memory_order_relaxed) ==
      other.spillWriteTimeNanos.load(std::memory_order_relaxed) &&
      spillMaxLevelExceededCount.load(std::memory_order_relaxed) ==
      other.spillMaxLevelExceededCount.load(std::memory_order_relaxed) &&
      spillReadBytes.load(std::memory_order_relaxed) ==
      other.spillReadBytes.load(std::memory_order_relaxed) &&
      spillReads.load(std::memory_order_relaxed) ==
      other.spillReads.load(std::memory_order_relaxed) &&
      spillReadTimeNanos.load(std::memory_order_relaxed) ==
      other.spillReadTimeNanos.load(std::memory_order_relaxed) &&
      spillDeserializationTimeNanos.load(std::memory_order_relaxed) ==
      other.spillDeserializationTimeNanos.load(std::memory_order_relaxed);
}

void SpillStats::reset() {
  spillRuns.store(0, std::memory_order_relaxed);
  spilledInputBytes.store(0, std::memory_order_relaxed);
  spilledBytes.store(0, std::memory_order_relaxed);
  spilledRows.store(0, std::memory_order_relaxed);
  spilledPartitions.store(0, std::memory_order_relaxed);
  spilledFiles.store(0, std::memory_order_relaxed);
  spillFillTimeNanos.store(0, std::memory_order_relaxed);
  spillSortTimeNanos.store(0, std::memory_order_relaxed);
  spillExtractVectorTimeNanos.store(0, std::memory_order_relaxed);
  spillSerializationTimeNanos.store(0, std::memory_order_relaxed);
  spillWrites.store(0, std::memory_order_relaxed);
  spillFlushTimeNanos.store(0, std::memory_order_relaxed);
  spillWriteTimeNanos.store(0, std::memory_order_relaxed);
  spillMaxLevelExceededCount.store(0, std::memory_order_relaxed);
  spillReadBytes.store(0, std::memory_order_relaxed);
  spillReads.store(0, std::memory_order_relaxed);
  spillReadTimeNanos.store(0, std::memory_order_relaxed);
  spillDeserializationTimeNanos.store(0, std::memory_order_relaxed);
  ioStats = IoStats();
}

std::string SpillStats::toString() const {
  std::stringstream ss;
  ss << "spillRuns[" << spillRuns.load(std::memory_order_relaxed) << "] "
     << "spilledInputBytes["
     << succinctBytes(spilledInputBytes.load(std::memory_order_relaxed)) << "] "
     << "spilledBytes["
     << succinctBytes(spilledBytes.load(std::memory_order_relaxed)) << "] "
     << "spilledRows[" << spilledRows.load(std::memory_order_relaxed) << "] "
     << "spilledPartitions["
     << spilledPartitions.load(std::memory_order_relaxed) << "] "
     << "spilledFiles[" << spilledFiles.load(std::memory_order_relaxed) << "] "
     << "spillFillTimeNanos["
     << succinctNanos(spillFillTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "spillSortTimeNanos["
     << succinctNanos(spillSortTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "spillExtractVectorTime["
     << succinctNanos(
            spillExtractVectorTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "spillSerializationTimeNanos["
     << succinctNanos(
            spillSerializationTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "spillWrites[" << spillWrites.load(std::memory_order_relaxed) << "] "
     << "spillFlushTimeNanos["
     << succinctNanos(spillFlushTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "spillWriteTimeNanos["
     << succinctNanos(spillWriteTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "maxSpillExceededLimitCount["
     << spillMaxLevelExceededCount.load(std::memory_order_relaxed) << "] "
     << "spillReadBytes["
     << succinctBytes(spillReadBytes.load(std::memory_order_relaxed)) << "] "
     << "spillReads[" << spillReads.load(std::memory_order_relaxed) << "] "
     << "spillReadTimeNanos["
     << succinctNanos(spillReadTimeNanos.load(std::memory_order_relaxed))
     << "] "
     << "spillReadDeserializationTimeNanos["
     << succinctNanos(
            spillDeserializationTimeNanos.load(std::memory_order_relaxed))
     << "]";

  const auto ioStatsMap = ioStats.stats();
  if (!ioStatsMap.empty()) {
    ss << " ioStats[";
    bool first = true;
    for (const auto& [name, metric] : ioStatsMap) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      ss << name << ":{sum:" << metric.sum << ", count:" << metric.count
         << ", min:" << metric.min << ", max:" << metric.max << "}";
    }
    ss << "]";
  }
  return ss.str();
}

void updateGlobalSpillRunStats(uint64_t numRuns) {
  localSpillStats().spillRuns.fetch_add(numRuns, std::memory_order_relaxed);
}

void updateGlobalSpillAppendStats(
    uint64_t numRows,
    uint64_t serializationTimeNs) {
  RECORD_METRIC_VALUE(kMetricSpilledRowsCount, numRows);
  RECORD_HISTOGRAM_METRIC_VALUE(
      kMetricSpillSerializationTimeMs, serializationTimeNs / 1'000'000);
  auto& stats = localSpillStats();
  stats.spilledRows.fetch_add(numRows, std::memory_order_relaxed);
  stats.spillSerializationTimeNanos.fetch_add(
      serializationTimeNs, std::memory_order_relaxed);
}

void incrementGlobalSpilledPartitionStats() {
  localSpillStats().spilledPartitions.fetch_add(1, std::memory_order_relaxed);
}

void updateGlobalSpillFillTime(uint64_t timeNs) {
  RECORD_HISTOGRAM_METRIC_VALUE(kMetricSpillFillTimeMs, timeNs / 1'000'000);
  localSpillStats().spillFillTimeNanos.fetch_add(
      timeNs, std::memory_order_relaxed);
}

void updateGlobalSpillSortTime(uint64_t timeNs) {
  RECORD_HISTOGRAM_METRIC_VALUE(kMetricSpillSortTimeMs, timeNs / 1'000'000);
  localSpillStats().spillSortTimeNanos.fetch_add(
      timeNs, std::memory_order_relaxed);
}

void updateGlobalSpillExtractVectorTime(uint64_t timeNs) {
  RECORD_HISTOGRAM_METRIC_VALUE(
      kMetricSpillExtractVectorTimeMs, timeNs / 1'000'000);
  localSpillStats().spillExtractVectorTimeNanos.fetch_add(
      timeNs, std::memory_order_relaxed);
}

void updateGlobalSpillWriteStats(
    uint64_t spilledBytes,
    uint64_t flushTimeNs,
    uint64_t writeTimeNs) {
  RECORD_METRIC_VALUE(kMetricSpillWritesCount);
  RECORD_METRIC_VALUE(kMetricSpilledBytes, spilledBytes);
  RECORD_HISTOGRAM_METRIC_VALUE(
      kMetricSpillFlushTimeMs, flushTimeNs / 1'000'000);
  RECORD_HISTOGRAM_METRIC_VALUE(
      kMetricSpillWriteTimeMs, writeTimeNs / 1'000'000);
  auto& stats = localSpillStats();
  stats.spillWrites.fetch_add(1, std::memory_order_relaxed);
  stats.spilledBytes.fetch_add(spilledBytes, std::memory_order_relaxed);
  stats.spillFlushTimeNanos.fetch_add(flushTimeNs, std::memory_order_relaxed);
  stats.spillWriteTimeNanos.fetch_add(writeTimeNs, std::memory_order_relaxed);
}

void updateGlobalSpillReadStats(
    uint64_t spillReads,
    uint64_t spillReadBytes,
    uint64_t spillReadTimeNs) {
  auto& stats = localSpillStats();
  stats.spillReads.fetch_add(spillReads, std::memory_order_relaxed);
  stats.spillReadBytes.fetch_add(spillReadBytes, std::memory_order_relaxed);
  stats.spillReadTimeNanos.fetch_add(
      spillReadTimeNs, std::memory_order_relaxed);
}

void updateGlobalSpillMemoryBytes(uint64_t spilledInputBytes) {
  RECORD_METRIC_VALUE(kMetricSpilledInputBytes, spilledInputBytes);
  localSpillStats().spilledInputBytes.fetch_add(
      spilledInputBytes, std::memory_order_relaxed);
}

void incrementGlobalSpilledFiles() {
  RECORD_METRIC_VALUE(kMetricSpilledFilesCount);
  localSpillStats().spilledFiles.fetch_add(1, std::memory_order_relaxed);
}

void updateGlobalMaxSpillLevelExceededCount(
    uint64_t maxSpillLevelExceededCount) {
  localSpillStats().spillMaxLevelExceededCount.fetch_add(
      maxSpillLevelExceededCount, std::memory_order_relaxed);
}

void updateGlobalSpillDeserializationTimeNs(uint64_t timeNs) {
  localSpillStats().spillDeserializationTimeNanos.fetch_add(
      timeNs, std::memory_order_relaxed);
}

SpillStats globalSpillStats() {
  SpillStats gSpillStats;
  for (auto& stats : allSpillStats()) {
    gSpillStats += stats;
  }
  return gSpillStats;
}
} // namespace facebook::velox::exec
