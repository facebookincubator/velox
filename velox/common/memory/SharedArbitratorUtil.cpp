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

#include "velox/common/memory/SharedArbitratorUtil.h"
#include <mutex>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::memory {

using namespace facebook::velox::memory;

namespace {
std::string memoryPoolAbortMessage(
    MemoryPool* victim,
    MemoryPool* requestor,
    size_t growBytes) {
  std::stringstream out;
  VELOX_CHECK(victim->isRoot());
  VELOX_CHECK(requestor->isRoot());
  if (requestor == victim) {
    out << "\nFailed memory pool '" << victim->name()
        << "' aborted by itself when tried to grow " << succinctBytes(growBytes)
        << "\n";
  } else {
    out << "\nFailed memory pool '" << victim->name()
        << "' aborted when requestor '" << requestor->name()
        << "' tried to grow " << succinctBytes(growBytes) << "\n";
  }
  out << "Memory usage of the failed memory pool:\n"
      << victim->treeMemoryUsage();
  return out.str();
}
} // namespace

ArbitrationOperation::ArbitrationOperation(
    ScopedArbitrationPool pool,
    uint64_t requestBytes,
    uint64_t timeoutMs)
    : requestBytes_(requestBytes),
      timeoutMs_(timeoutMs),
      startTimeUs_(getCurrentTimeMicro()),
      pool_(std::move(pool)) {
  VELOX_CHECK_GT(requestBytes_, 0);
  VELOX_CHECK(pool_.valid());
}

void ArbitrationOperation::start() {
  pool_->startArbitration(this);
}

void ArbitrationOperation::finish() {
  pool_->finishArbitration(this);
}

void ArbitrationOperation::setGrowTargets() {
  VELOX_CHECK_EQ(maxGrowBytes_, 0);
  VELOX_CHECK_EQ(minGrowBytes_, 0);
  pool_->getGrowTargets(requestBytes_, maxGrowBytes_, minGrowBytes_);
  VELOX_CHECK_LE(requestBytes_, maxGrowBytes_);
}

ArbitrationPool::Config::Config(
    uint64_t _minCapacity,
    uint64_t _fastExponentialGrowthCapacityLimit,
    double _slowCapacityGrowPct,
    uint64_t _deprecatedMinGrowCapacity,
    uint64_t _minFreeCapacity,
    double _minFreeCapacityPct)
    : minCapacity(_minCapacity),
      fastExponentialGrowthCapacityLimit(_fastExponentialGrowthCapacityLimit),
      slowCapacityGrowPct(_slowCapacityGrowPct),
      deprecatedMinGrowCapacity(_deprecatedMinGrowCapacity),
      minFreeCapacity(_minFreeCapacity),
      minFreeCapacityPct(_minFreeCapacityPct) {
  VELOX_CHECK_GE(slowCapacityGrowPct, 0);
  VELOX_CHECK_EQ(
      fastExponentialGrowthCapacityLimit == 0,
      slowCapacityGrowPct == 0,
      "fastExponentialGrowthCapacityLimit_ {} and slowCapacityGrowPct_ {} "
      "both need to be set (non-zero) at the same time to enable growth capacity "
      "adjustment.",
      fastExponentialGrowthCapacityLimit,
      slowCapacityGrowPct);

  VELOX_CHECK_GE(minFreeCapacityPct, 0);
  VELOX_CHECK_LE(minFreeCapacityPct, 1);
  VELOX_CHECK_EQ(
      minFreeCapacity == 0,
      minFreeCapacityPct == 0,
      "memoryPoolMinFreeCapacity_ {} and memoryPoolMinFreeCapacityPct_ {} both "
      "need to be set (non-zero) at the same time to enable shrink capacity "
      "adjustment.",
      minFreeCapacity,
      minFreeCapacityPct);
}

ArbitrationPool::ArbitrationPool(
    uint64_t id,
    const std::shared_ptr<MemoryPool>& memPool,
    const Config* config)
    : id_(id),
      memPoolPtr_(memPool),
      memPool_(memPool.get()),
      config_(config),
      maxCapacity_(memPool_->maxCapacity()),
      createTimeUs_(getCurrentTimeMicro()) {}

ArbitrationPool::~ArbitrationPool() {
  VELOX_CHECK_NULL(runningOp_);
}

ScopedArbitrationPool ArbitrationPool::get() {
  auto sharedPtr = memPoolPtr_.lock();
  if (sharedPtr == nullptr) {
    return {};
  }
  return ScopedArbitrationPool(shared_from_this(), std::move(sharedPtr));
}

uint64_t ArbitrationPool::maxGrowCapacity() const {
  const auto capacity = memPool_->capacity();
  VELOX_CHECK_LE(capacity, maxCapacity_);
  return maxCapacity_ - capacity;
}

uint64_t ArbitrationPool::minGrowCapacity() const {
  const auto capacity = memPool_->capacity();
  if (capacity >= config_->minCapacity) {
    return 0;
  }
  return config_->minCapacity - capacity;
}

bool ArbitrationPool::checkCapacityGrowth(uint64_t requestBytes) const {
  return maxGrowCapacity() >= requestBytes;
}

void ArbitrationPool::getGrowTargets(
    uint64_t requestBytes,
    uint64_t& maxGrowBytes,
    uint64_t& minGrowBytes) const {
  checkCapacityGrowth(requestBytes);

  const uint64_t capacity = memPool_->capacity();
  if (config_->fastExponentialGrowthCapacityLimit == 0 &&
      config_->slowCapacityGrowPct == 0) {
    maxGrowBytes = config_->deprecatedMinGrowCapacity;
  } else {
    if (capacity * 2 <= config_->fastExponentialGrowthCapacityLimit) {
      maxGrowBytes = capacity;
    } else {
      maxGrowBytes = capacity * config_->slowCapacityGrowPct;
    }
  }
  maxGrowBytes = std::max(requestBytes, maxGrowBytes);
  maxGrowBytes = std::min(maxGrowCapacity(), maxGrowBytes);

  minGrowBytes = minGrowCapacity();
  if (requestBytes > maxGrowBytes) {
    LOG(ERROR) << "bad";
  }
  VELOX_CHECK_LE(requestBytes, maxGrowBytes);
}

uint64_t ArbitrationPool::maxShrinkCapacity() const {
  const uint64_t capacity = memPool_->capacity();
  const uint64_t freeBytes = memPool_->freeBytes();
  if (config_->minFreeCapacity != 0) {
    const uint64_t minFreeBytes = std::min(
        static_cast<uint64_t>(capacity * config_->minFreeCapacityPct),
        config_->minFreeCapacity);
    if (freeBytes <= minFreeBytes) {
      return 0;
    } else {
      return freeBytes - minFreeBytes;
    }
  } else {
    return freeBytes;
  }
}

void ArbitrationPool::startArbitration(ArbitrationOperation* op) {
  ContinueFuture waitPromise{ContinueFuture::makeEmpty()};
  {
    std::lock_guard<std::mutex> l(stateLock_);
    if (runningOp_ != nullptr) {
      waitOpPromises_.emplace_back(
          fmt::format("Wait for arbitration {}", op->pool()->name()));
      waitPromise = waitOpPromises_.back().getSemiFuture();
    } else {
      runningOp_ = op;
    }
  }

  TestValue::adjust(
      "facebook::velox::memory::ArbitrationPool::startArbitration", this);

  if (waitPromise.valid()) {
    uint64_t waitTimeUs{0};
    {
      MicrosecondTimer timer(&waitTimeUs);
      waitPromise.wait();
    }
    op->setLocalArbitrationWaitTimeUs(waitTimeUs);
  }
}

void ArbitrationPool::finishArbitration(ArbitrationOperation* op) {
  ContinuePromise resumePromise{ContinuePromise::makeEmpty()};
  {
    std::lock_guard<std::mutex> l(stateLock_);
    VELOX_CHECK_EQ(static_cast<void*>(op), static_cast<void*>(runningOp_));
    if (!waitOpPromises_.empty()) {
      resumePromise = std::move(waitOpPromises_.front());
      waitOpPromises_.pop_front();
    }
    runningOp_ = nullptr;
  }
  if (resumePromise.valid()) {
    resumePromise.setValue();
  }
}

uint64_t ArbitrationPool::reclaim(
    uint64_t targetBytes,
    uint64_t maxWaitTimeMs,
    bool selfReclaim) noexcept {
  std::lock_guard<std::mutex> l(reclaimLock_);
  int64_t bytesToReclaim =
      std::min<uint64_t>(targetBytes, maxReclaimableCapacity());
  if (bytesToReclaim == 0) {
    return 0;
  }

  uint64_t reclaimedUsedBytes{0};
  uint64_t reclaimedFreeBytes{0};
  MemoryReclaimer::Stats reclaimStats;
  try {
    reclaimedFreeBytes = shrink(selfReclaim);
    if (reclaimedFreeBytes < bytesToReclaim) {
      ++numReclaims_;
      memPool_->reclaim(
          bytesToReclaim - reclaimedFreeBytes, maxWaitTimeMs, reclaimStats);
    }
  } catch (const std::exception& e) {
    VELOX_MEM_LOG(ERROR) << "Failed to reclaim from memory pool "
                         << memPool_->name() << ", aborting it: " << e.what();
    abortLocked(std::current_exception());
    reclaimedUsedBytes = shrink(/*reclaimAll=*/true);
  }
  reclaimedUsedBytes += shrink(selfReclaim);

  reclaimedUsedBytes_ += reclaimedUsedBytes;
  reclaimedFreeBytes_ += reclaimedFreeBytes;
  numNonReclaimableAttempts_ += reclaimStats.numNonReclaimableAttempts;
  return reclaimedUsedBytes + reclaimedFreeBytes;
}

bool ArbitrationPool::grow(uint64_t growBytes, uint64_t reservationBytes) {
  std::lock_guard<std::mutex> l(stateLock_);
  ++numGrows_;
  return memPool_->grow(growBytes, reservationBytes);
}

uint64_t ArbitrationPool::shrink(bool reclaimAll) {
  std::lock_guard<std::mutex> l(stateLock_);
  ++numShrinks_;
  return memPool_->shrink(reclaimAll ? 0 : reclaimableFreeCapacity());
}

void ArbitrationPool::abort(const std::exception_ptr& error) {
  std::lock_guard<std::mutex> l(reclaimLock_);
  abortLocked(error);
}

void ArbitrationPool::abortLocked(const std::exception_ptr& error) {
  {
    std::lock_guard<std::mutex> l(stateLock_);
    if (aborted_) {
      return;
    }
    aborted_ = true;
  }
  try {
    memPool_->abort(error);
  } catch (const std::exception& e) {
    VELOX_MEM_LOG(WARNING) << "Failed to abort memory pool "
                           << memPool_->toString() << ", error: " << e.what();
  }
  // NOTE: no matter memory pool abort throws or not, it should have been marked
  // as aborted to prevent any new memory arbitration triggered from the aborted
  // memory pool.
  VELOX_CHECK(memPool_->aborted());
}

uint64_t ArbitrationPool::reclaimableFreeCapacity() const {
  return std::min(maxShrinkCapacity(), maxReclaimableCapacity());
}

uint64_t ArbitrationPool::maxReclaimableCapacity() const {
  // Checks if a query memory pool has finished processing or not. If it has
  // finished, then we don't have to respect the memory pool reserved capacity
  // limit check.
  //
  // NOTE: for query system like Prestissimo, it holds a finished query
  // state in minutes for query stats fetch request from the Presto
  // coordinator.
  if (memPool_->reservedBytes() == 0 && memPool_->peakBytes() != 0) {
    return memPool_->capacity();
  }
  const uint64_t capacityBytes = memPool_->capacity();
  if (capacityBytes < config_->minCapacity) {
    return 0;
  }
  return capacityBytes - config_->minCapacity;
}

uint64_t ArbitrationPool::reclaimableUsedCapacity() const {
  const auto maxReclaimableBytes = maxReclaimableCapacity();
  const auto reclaimableBytes = memPool_->reclaimableBytes();
  return std::min<int64_t>(maxReclaimableBytes, reclaimableBytes.value_or(0));
}

ScopedArbitrationPool::ScopedArbitrationPool(
    std::shared_ptr<ArbitrationPool> arbitrationPool,
    std::shared_ptr<MemoryPool> memPool)
    : arbitrationPool_(std::move(arbitrationPool)),
      memPool_(std::move(memPool)) {
  VELOX_CHECK_EQ(arbitrationPool_ == nullptr, memPool_ == nullptr);
}

ArbitrationCandidate::ArbitrationCandidate(
    ScopedArbitrationPool&& _pool,
    bool freeCapacityOnly)
    : pool(std::move(_pool)),
      reclaimableBytes(freeCapacityOnly ? 0 : pool->reclaimableUsedCapacity()),
      freeBytes(pool->reclaimableFreeCapacity()) {}

std::string ArbitrationCandidate::toString() const {
  return fmt::format(
      "CANDIDATE[{}] RECLAIMABLE_BYTES[{}] FREE_BYTES[{}]]",
      pool->name(),
      succinctBytes(reclaimableBytes),
      succinctBytes(freeBytes));
}
} // namespace facebook::velox::memory
