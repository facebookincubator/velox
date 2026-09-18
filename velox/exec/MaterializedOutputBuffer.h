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

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <folly/concurrency/UnboundedQueue.h>
#include <folly/io/IOBuf.h>
#include "velox/common/future/VeloxPromise.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/ExchangeSink.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

/// Shared buffer between materialized output operators and an ExchangeSink.
/// Appenders enqueue without locking. A CAS-elected flusher serializes sink
/// appends for each partition, while cooperative high/low-watermark
/// backpressure bounds the total queued bytes.
class MaterializedOutputBuffer {
 public:
  enum class State : uint8_t {
    kActive,
    kDraining,
    kClosed,
    kAborted,
  };

  /// Returns the name of the buffer state.
  static std::string stateName(State state);

  static constexpr int64_t kDefaultDrainThreshold = 128L * 1024;

  MaterializedOutputBuffer(
      int32_t numPartitions,
      std::shared_ptr<ExchangeSink> sink,
      int64_t maxBufferedBytes,
      int64_t partitionDrainThreshold = kDefaultDrainThreshold,
      std::shared_ptr<memory::MemoryPool> pool = nullptr);

  ~MaterializedOutputBuffer();

  /// Enqueues a serialized RowGroup for a partition.
  void enqueue(int32_t partition, std::unique_ptr<folly::IOBuf> rowGroup);

  /// Drains at the high watermark, then parks until below the low watermark.
  BlockingReason isBlocked(ContinueFuture* future);

  /// Best-effort drain of one partition without closing it.
  int64_t drainPartition(int32_t partition);

  /// Best-effort drain of every partition without closing them.
  uint64_t drainAll();

  /// Drains all remaining data and commits the sink.
  void noMoreData();

  /// Discards buffered data and aborts the sink.
  void abort();

  State state() const {
    return state_.load();
  }

  int64_t bufferedBytes() const {
    return bufferedBytes_;
  }

  int64_t partitionDrainThreshold() const {
    return partitionDrainThreshold_;
  }

  int64_t highWatermarkBytes() const {
    return highWatermarkBytes_;
  }

  int64_t lowWatermarkBytes() const {
    return lowWatermarkBytes_;
  }

  /// Returns true when buffered bytes reach the producer-blocking high
  /// watermark.
  bool isBufferFull() const;

  /// Records the number of drivers. Only the first call takes effect.
  void setNumDrivers(uint32_t numDrivers);

  /// Marks one driver complete and commits when the final driver finishes.
  bool noMoreDrivers();

  /// Returns sink and buffer runtime counters.
  folly::F14FastMap<std::string, int64_t> stats() const;

  /// Returns committed output locations after noMoreData().
  folly::F14FastMap<int32_t, std::string> outputLocations() const;

  /// Allocates a RowGroup IOBuf tracked by the configured memory pool.
  std::unique_ptr<folly::IOBuf> allocateTrackedIOBuf(size_t size);

  int32_t numPartitions() const {
    return numPartitions_;
  }

 private:
  class PartitionBuffer {
   public:
    PartitionBuffer(
        int32_t partition,
        int64_t drainThreshold,
        MaterializedOutputBuffer* buffer);

    /// Lock-free append that opportunistically drains above the threshold.
    int64_t enqueue(std::unique_ptr<folly::IOBuf> rowGroup);

   private:
    friend class MaterializedOutputBuffer;

    bool tryAcquireFlushing();

    void releaseFlushing();

    int64_t drainAndFlush();

    int64_t tryDrainPartition(int64_t targetBytes);

    int64_t finish(bool success);

    folly::UMPMCQueue<std::unique_ptr<folly::IOBuf>, false> rowGroupsQueue_;
    std::atomic_int64_t bufferedBytes_{0};
    std::atomic<bool> flushing_{false};
    std::atomic<bool> closed_{false};
    const int32_t partition_;
    const int64_t drainThreshold_;
    MaterializedOutputBuffer* const buffer_;
  };

  void initializePartitions();

  int64_t drainPartitionInternal(int32_t partition, int64_t targetBytes);

  uint64_t tryDrainPartitionsInternal();

  void tryDrainForBackpressure();

  uint64_t reclaimableBufferedBytes() const;

  void addBlockedPromise(ContinueFuture* future);

  void maybeWakeBlockedDrivers();

  void updateDrainStats(int64_t drainedBytes);

  void flushDrained(
      int32_t partition,
      std::deque<std::unique_ptr<folly::IOBuf>>& rowGroups);

  static void freeTrackedIOBuf(void* buffer, void* userData);

  const int32_t numPartitions_;
  const int64_t maxBufferedBytes_;
  const int64_t partitionDrainThreshold_;
  const int64_t reclaimDrainThresholdBytes_;
  const int64_t drainChunkThresholdBytes_;
  const int64_t highWatermarkBytes_;
  const int64_t lowWatermarkBytes_;
  const std::shared_ptr<memory::MemoryPool> pool_;
  const std::shared_ptr<ExchangeSink> sink_;

  std::atomic<State> state_{State::kActive};
  std::atomic_int64_t bufferedBytes_{0};
  std::vector<std::unique_ptr<PartitionBuffer>> partitionBuffers_;

  std::atomic_int64_t drainedBytes_{0};
  std::atomic_int64_t drainCount_{0};
  std::atomic_int64_t peakBufferedBytes_{0};
  std::atomic_int64_t appendCount_{0};
  std::atomic_int64_t concurrentAppendCount_{0};
  std::atomic_int64_t flushAcquireCount_{0};
  std::atomic_int64_t backpressureBlockCount_{0};
  std::atomic_int64_t backpressureWakeCount_{0};
  std::atomic_int64_t backpressureDrainedBytes_{0};

  folly::Synchronized<std::vector<ContinuePromise>> blockedPromises_;

  mutable std::mutex lifecycleMutex_;
  std::optional<CommittedExchangeOutput> output_;

  std::mutex driverMutex_;
  uint32_t numDrivers_{0};
  uint32_t numFinishedDrivers_{0};
};

} // namespace facebook::velox::exec

template <>
struct fmt::formatter<facebook::velox::exec::MaterializedOutputBuffer::State>
    : formatter<std::string> {
  auto format(
      facebook::velox::exec::MaterializedOutputBuffer::State state,
      format_context& context) const {
    return formatter<std::string>::format(
        facebook::velox::exec::MaterializedOutputBuffer::stateName(state),
        context);
  }
};
