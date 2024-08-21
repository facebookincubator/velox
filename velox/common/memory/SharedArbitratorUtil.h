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

#include "velox/common/memory/MemoryArbitrator.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::memory {

class SharedArbitrator;
class ArbitrationCandidate;
class ArbitrationOperation;
class ScopedArbitrationPool;

class ArbitrationPool : public std::enable_shared_from_this<ArbitrationPool> {
 public:
  struct Config {
    uint64_t minCapacity;
    uint64_t fastExponentialGrowthCapacityLimit;
    double slowCapacityGrowPct;
    uint64_t deprecatedMinGrowCapacity;
    uint64_t minFreeCapacity;
    double minFreeCapacityPct;

    Config(
        uint64_t _minCapacity,
        uint64_t _fastExponentialGrowthCapacityLimit,
        double _slowCapacityGrowPct,
        uint64_t _deprecatedMinGrowCapacity,
        uint64_t _minFreeCapacity,
        double _minFreeCapacityPct);
  };

  ArbitrationPool(
      uint64_t id,
      const std::shared_ptr<MemoryPool>& memPool,
      const Config* config);

  ~ArbitrationPool();

  std::string name() const {
    return memPool_->name();
  }

  uint64_t capacity() const {
    return memPool_->capacity();
  }

  uint64_t maxCapacity() const {
    return maxCapacity_;
  }

  uint64_t minCapacity() const {
    return config_->minCapacity;
  }

  /// The capacity growth target is set to have a coarser granularity. It can
  /// help to reduce the number of future grow calls, and hence reducing the
  /// number of unnecessary memory arbitration requests.
  void getGrowTargets(
      uint64_t requestBytes,
      uint64_t& maxGrowBytes,
      uint64_t& minGrowBytes) const;

  /// Checks if the memory pool can grow 'requestBytes' from its current
  /// capacity under the max capacity limit.
  bool checkCapacityGrowth(uint64_t requestBytes) const;

  void checkIfAborted() const {
    std::lock_guard<std::mutex> l(stateLock_);
    if (aborted_) {
      VELOX_MEM_POOL_ABORTED("The memory pool has been aborted");
    }
  }

  bool aborted() const {
    std::lock_guard<std::mutex> l(stateLock_);
    return aborted_;
  }

  uint64_t id() const {
    return id_;
  }

  MemoryPool* memPool() const {
    return memPool_;
  }

  size_t createTimeUs() const {
    return createTimeUs_;
  }

  ScopedArbitrationPool get();

  /// Invoked to grow 'pool' capacity by 'growBytes' and commit used reservation
  /// by 'reservationBytes'. The function throws if the growth fails.
  bool grow(uint64_t growBytes, uint64_t reservationBytes);

  uint64_t shrink(bool reclaimAll);

  /// Invoked to reclaim used memory from this memory pool with specified
  /// 'targetBytes'. The function returns the actually freed capacity.
  /// 'selfReclaim' is true when the reclaim is requested by the memory pool
  /// itself.
  uint64_t reclaim(
      uint64_t targetBytes,
      uint64_t maxWaitTimeMs,
      bool selfReclaim) noexcept;

  void abort(const std::exception_ptr& error);

  // Invoked to start next memory arbitration request, and it will wait for
  // the serialized execution if there is a running or other waiting
  // arbitration requests.
  void startArbitration(ArbitrationOperation* op);

  // Invoked by a finished memory arbitration request to kick off the next
  // arbitration request execution if there are any ones waiting.
  void finishArbitration(ArbitrationOperation* op);

  struct Stats {
    std::string name;
    size_t createTimeUs;
    uint32_t numReclaims;
    uint32_t numShrinks;
    uint32_t numGrows;

    std::string toString() const;
  };
  Stats stats() const;

  // Returns the free memory capacity that can be reclaimed from 'pool' by
  // shrink. If 'selfReclaim' true, we reclaim memory from the request pool
  // itself so that we can bypass the reserved free capacity reclaim
  // restriction.
  uint64_t reclaimableFreeCapacity() const;

  // Returns the used memory capacity that can be reclaimed from 'pool' by
  // disk spill. If 'isSelfReclaim' true, we reclaim memory from the request
  // pool itself so that we can bypass the reserved free capacity reclaim
  // restriction.
  uint64_t reclaimableUsedCapacity() const;

 private:
  uint64_t maxReclaimableCapacity() const;

  // The capacity shrink target is adjusted from request shrink bytes to give
  // the memory pool more headroom free capacity after shrink. It can help to
  // reduce the number of future grow calls, and hence reducing the number of
  // unnecessary memory arbitration requests.
  uint64_t maxShrinkCapacity() const;

  // Returns the max capacity to grow of this memory pool. The calculation is
  // based on the memory pool's max capacity and its current capacity.
  uint64_t maxGrowCapacity() const;
  // Returns the minimal amount of memory capacity to grow for 'pool' to have
  // the reserved capacity as specified by 'memoryPoolReservedCapacity_'.
  uint64_t minGrowCapacity() const;

  void abortLocked(const std::exception_ptr& error);

  const uint64_t id_;
  const std::weak_ptr<MemoryPool> memPoolPtr_;
  MemoryPool* const memPool_;
  const Config* const config_;
  const uint64_t maxCapacity_;
  const size_t createTimeUs_;

  mutable std::mutex stateLock_;
  bool aborted_{false};

  /// Points to the current running arbitration.
  ArbitrationOperation* runningOp_;

  /// The promises of the arbitration requests from the same query pool waiting
  /// for the serial execution.
  std::deque<ContinuePromise> waitOpPromises_;

  tsan_atomic<uint32_t> numReclaims_{0};
  tsan_atomic<uint32_t> numShrinks_{0};
  tsan_atomic<uint32_t> numGrows_{0};
  tsan_atomic<uint64_t> reclaimedUsedBytes_{0};
  tsan_atomic<uint64_t> reclaimedFreeBytes_{0};
  tsan_atomic<uint64_t> numNonReclaimableAttempts_{0};

  std::mutex reclaimLock_;
};

class ScopedArbitrationPool {
 public:
  ScopedArbitrationPool() : ScopedArbitrationPool(nullptr, nullptr) {}

  ScopedArbitrationPool(
      std::shared_ptr<ArbitrationPool> arbitrationPool,
      std::shared_ptr<MemoryPool> memPool);

  bool valid() const {
    return arbitrationPool_ != nullptr;
  }

  ArbitrationPool* operator->() const {
    return arbitrationPool_.get();
  }

  ArbitrationPool& operator*() const {
    return *arbitrationPool_;
  }

  ArbitrationPool& operator()() const {
    return *arbitrationPool_;
  }

  ArbitrationPool* get() const {
    return arbitrationPool_.get();
  }

 private:
  std::shared_ptr<ArbitrationPool> arbitrationPool_;
  std::shared_ptr<MemoryPool> memPool_;
};

/// Contains the execution state of an arbitration operation.
class ArbitrationOperation {
 public:
  ArbitrationOperation(
      ScopedArbitrationPool pool,
      uint64_t requestBytes,
      uint64_t timeoutMs);

  uint64_t id() const {
    return pool_->id();
  }

  size_t startTimeUs() const {
    return startTimeUs_;
  }

  size_t startTimeMs() const {
    return startTimeUs_ / 1'000;
  }

  bool hasTimeout() const {
    return executionTimeMs() >= startTimeMs();
  }

  size_t executionTimeMs() const {
    const auto currentTimeMs = getCurrentTimeMs();
    VELOX_CHECK_GE(currentTimeMs, startTimeMs());
    return currentTimeMs - startTimeMs();
  }

  size_t timeoutMs() const {
    const auto execTimeMs = executionTimeMs();
    if (execTimeMs >= timeoutMs_) {
      return 0;
    }
    return timeoutMs_ - execTimeMs;
  }

  ScopedArbitrationPool& pool() {
    return pool_;
  }

  void setGrowTargets();

  uint64_t requestBytes() const {
    return requestBytes_;
  }

  uint64_t maxGrowBytes() const {
    return maxGrowBytes_;
  }

  uint64_t minGrowBytes() const {
    return minGrowBytes_;
  }

  bool aborted() const {
    return pool_->aborted();
  }

  void start();

  void finish();

  uint64_t& arbitratedBytes() {
    return arbitratedBytes_;
  }

  void setLocalArbitrationWaitTimeUs(uint64_t waitTimeUs) {
    VELOX_CHECK_EQ(localArbitrationWaitTimeUs_, 0);
    localArbitrationWaitTimeUs_ = waitTimeUs;
  }

  uint64_t localArbitrationWaitTimeUs() const {
    return localArbitrationWaitTimeUs_;
  }

  void setLocalArbitrationExecutionTimeUs(uint64_t execTimeUs) {
    VELOX_CHECK_EQ(localArbitrationExecTimeUs_, 0);
    localArbitrationExecTimeUs_ = execTimeUs;
  }

  void setGlobalArbitrationWaitTimeUs(uint64_t waitTimeUs) {
    VELOX_CHECK_EQ(globalArbitrationWaitTimeUs_, 0);
    globalArbitrationWaitTimeUs_ = waitTimeUs;
  }

  uint64_t globalArbitrationWaitTimeUs() const {
    return globalArbitrationWaitTimeUs_;
  }

 private:
  const uint64_t requestBytes_;
  const uint64_t timeoutMs_;

  // The start time of this arbitration operation.
  const size_t startTimeUs_;

  ScopedArbitrationPool pool_;

  uint64_t maxGrowBytes_{0};
  uint64_t minGrowBytes_{0};

  uint64_t arbitratedBytes_{0};

  // The time that waits in local arbitration queue.
  uint64_t localArbitrationWaitTimeUs_{0};
  // The time that spent in local arbitration execution.
  uint64_t localArbitrationExecTimeUs_{0};
  // The time that waits for global arbitration.
  uint64_t globalArbitrationWaitTimeUs_{0};
};

/// The candidate memory pool stats used by arbitration.
struct ArbitrationCandidate {
  ScopedArbitrationPool pool;
  int64_t reclaimableBytes{0};
  int64_t freeBytes{0};
  int64_t reservedBytes{0};

  ArbitrationCandidate(ScopedArbitrationPool&& _pool, bool freeCapacityOnly);

  std::string toString() const;
};
} // namespace facebook::velox::memory
