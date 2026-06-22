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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <folly/dynamic.h>
#include "velox/common/base/IoCounter.h"

namespace facebook::velox::io {

struct OperationCounters {
  uint64_t resourceThrottleCount{0};
  uint64_t localThrottleCount{0};
  uint64_t networkThrottleCount{0};
  uint64_t globalThrottleCount{0};
  uint64_t fullThrottleCount{0};
  uint64_t partialThrottleCount{0};
  uint64_t retryCount{0};
  uint64_t latencyInMs{0};
  uint64_t requestCount{0};
  uint64_t delayInjectedInSecs{0};

  void merge(const OperationCounters& other);
};

class IoStatistics {
 public:
  uint64_t rawBytesRead() const;
  uint64_t rawOverreadBytes() const;
  uint64_t rawBytesWritten() const;
  uint64_t inputBatchSize() const;
  uint64_t outputBatchSize() const;
  uint64_t totalScanTimeNs() const;
  uint64_t writeIOTimeUs() const;

  uint64_t incRawBytesRead(int64_t);
  uint64_t incRawOverreadBytes(int64_t);
  uint64_t incRawBytesWritten(int64_t);
  uint64_t incInputBatchSize(int64_t);
  uint64_t incOutputBatchSize(int64_t);
  uint64_t incTotalScanTimeNs(int64_t);
  uint64_t incWriteIOTimeUs(int64_t);

  IoCounter& prefetch() {
    return prefetch_;
  }

  IoCounter& read() {
    return read_;
  }

  IoCounter& ssdRead() {
    return ssdRead_;
  }

  IoCounter& ramHit() {
    return ramHit_;
  }

  IoCounter& queryThreadIoLatencyUs() {
    return queryThreadIoLatencyUs_;
  }

  IoCounter& storageReadLatencyUs() {
    return storageReadLatencyUs_;
  }

  IoCounter& ssdCacheReadLatencyUs() {
    return ssdCacheReadLatencyUs_;
  }

  IoCounter& cacheWaitLatencyUs() {
    return cacheWaitLatencyUs_;
  }

  IoCounter& coalescedSsdLoadLatencyUs() {
    return coalescedSsdLoadLatencyUs_;
  }

  IoCounter& coalescedStorageLoadLatencyUs() {
    return coalescedStorageLoadLatencyUs_;
  }

  /// Distribution of gaps (in bytes) between consecutive read regions
  /// before coalescing. Measures data locality on disk.
  IoCounter& readGap() {
    return readGap_;
  }

  const IoCounter& readGap() const {
    return readGap_;
  }

  void incOperationCounters(
      const std::string& operation,
      const uint64_t resourceThrottleCount,
      const uint64_t localThrottleCount,
      const uint64_t networkThrottleCount,
      const uint64_t globalThrottleCount,
      const uint64_t retryCount,
      const uint64_t latencyInMs,
      const uint64_t delayInjectedInSecs,
      const uint64_t fullThrottleCount = 0,
      const uint64_t partialThrottleCount = 0);

  std::unordered_map<std::string, OperationCounters> operationStats() const;

  void merge(const IoStatistics& other);

  folly::dynamic getOperationStatsSnapshot() const;

 private:
  std::atomic<uint64_t> rawBytesRead_{0};
  std::atomic<uint64_t> rawBytesWritten_{0};
  std::atomic<uint64_t> inputBatchSize_{0};
  std::atomic<uint64_t> outputBatchSize_{0};
  std::atomic<uint64_t> rawOverreadBytes_{0};
  std::atomic<uint64_t> totalScanTimeNs_{0};
  std::atomic<uint64_t> writeIOTimeUs_{0};

  // Planned read from storage or SSD.
  IoCounter prefetch_;

  // Read from storage, for sparsely accessed columns.
  IoCounter read_;

  // Hits from RAM cache. Does not include first use of prefetched data.
  IoCounter ramHit_;

  // Read from SSD cache instead of storage. Includes both random and planned
  // reads.
  IoCounter ssdRead_;

  // Time spent by a query processing thread waiting for synchronously issued IO
  // or for an in-progress read-ahead to finish.
  IoCounter queryThreadIoLatencyUs_;

  // Breakdown of queryThreadIoLatencyUs_ by I/O type:

  // Time spent waiting for remote storage reads (S3, HDFS, etc.)
  IoCounter storageReadLatencyUs_;

  // Time spent waiting for SSD cache reads
  IoCounter ssdCacheReadLatencyUs_;

  // Time spent waiting for EXCLUSIVE cache entries (another thread is loading)
  IoCounter cacheWaitLatencyUs_;

  // Time spent waiting for coalesced loads from SSD cache
  IoCounter coalescedSsdLoadLatencyUs_;

  // Time spent waiting for coalesced loads from remote storage
  IoCounter coalescedStorageLoadLatencyUs_;

  // Gap between consecutive read regions before coalescing.
  IoCounter readGap_;

  std::unordered_map<std::string, OperationCounters> operationStats_;
  mutable std::mutex operationStatsMutex_;
};

} // namespace facebook::velox::io
