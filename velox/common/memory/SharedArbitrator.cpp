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

#include "velox/common/memory/SharedArbitrator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::memory {
namespace {
// Returns the max capacity to grow of memory 'pool'. The calculation is based
// on a memory pool's max capacity and its current capacity.
uint64_t maxGrowBytes(const MemoryPool& pool) {
  return pool.maxCapacity() - pool.capacity();
}

// Returns the capacity of the memory pool with the specified growth target.
uint64_t capacityAfterGrowth(const MemoryPool& pool, uint64_t targetBytes) {
  return pool.capacity() + targetBytes;
}
} // namespace

std::string SharedArbitrator::Candidate::toString() const {
  return fmt::format(
      "CANDIDATE[{} RECLAIMABLE[{}] RECLAIMABLE_BYTES[{}] FREE_BYTES[{}]]",
      pool->root()->name(),
      reclaimable,
      succinctBytes(reclaimableBytes),
      succinctBytes(freeBytes));
}

void SharedArbitrator::sortCandidatesByFreeCapacity(
    std::vector<Candidate>& candidates) const {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](const Candidate& lhs, const Candidate& rhs) {
        return lhs.freeBytes > rhs.freeBytes;
      });

  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::sortCandidatesByFreeCapacity",
      &candidates);
}

void SharedArbitrator::sortCandidatesByReclaimableMemory(
    std::vector<Candidate>& candidates) const {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const Candidate& lhs, const Candidate& rhs) {
        if (!lhs.reclaimable) {
          return false;
        }
        if (!rhs.reclaimable) {
          return true;
        }
        return lhs.reclaimableBytes > rhs.reclaimableBytes;
      });

  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::sortCandidatesByReclaimableMemory",
      &candidates);
}

const SharedArbitrator::Candidate&
SharedArbitrator::findCandidateWithLargestCapacity(
    MemoryPool* requestor,
    uint64_t targetBytes,
    const std::vector<Candidate>& candidates) const {
  VELOX_CHECK(!candidates.empty());
  int32_t candidateIdx{-1};
  int64_t maxCapacity{-1};
  for (int32_t i = 0; i < candidates.size(); ++i) {
    const bool isCandidate = candidates[i].pool == requestor;
    // For capacity comparison, the requestor's capacity should include both its
    // current capacity and the capacity growth.
    const int64_t capacity =
        candidates[i].pool->capacity() + (isCandidate ? targetBytes : 0);
    if (i == 0) {
      candidateIdx = 0;
      maxCapacity = capacity;
      continue;
    }
    if (maxCapacity > capacity) {
      continue;
    }
    if (capacity < maxCapacity) {
      candidateIdx = i;
      continue;
    }
    // With the same amount of capacity, we prefer to kill the requestor itself
    // without affecting the other query.
    if (isCandidate) {
      candidateIdx = i;
    }
  }
  VELOX_CHECK_NE(candidateIdx, -1);
  return candidates[candidateIdx];
}

SharedArbitrator::~SharedArbitrator() {
  VELOX_CHECK_EQ(freeCapacity_, capacity_, "{}", toString());
}

void SharedArbitrator::reserveMemory(MemoryPool* pool, uint64_t /*unused*/) {
  const int64_t bytesToReserve =
      std::min<int64_t>(maxGrowBytes(*pool), initMemoryPoolCapacity_);
  std::lock_guard<std::mutex> l(mutex_);
  if (running_) {
    // NOTE: if there is a running memory arbitration, then we shall skip
    // reserving the free memory for the newly created memory pool but let it
    // grow its capacity on-demand later through the memory arbitration.
    return;
  }
  const uint64_t reserveBytes = decrementFreeCapacityLocked(bytesToReserve);
  pool->grow(reserveBytes);
}

void SharedArbitrator::releaseMemory(MemoryPool* pool) {
  std::lock_guard<std::mutex> l(mutex_);
  const uint64_t freedBytes = pool->shrink(0);
  incrementFreeCapacityLocked(freedBytes);
}

std::vector<SharedArbitrator::Candidate> SharedArbitrator::getCandidateStats(
    const std::vector<std::shared_ptr<MemoryPool>>& pools) {
  std::vector<SharedArbitrator::Candidate> candidates;
  candidates.reserve(pools.size());
  for (const auto& pool : pools) {
    uint64_t reclaimableBytes;
    const bool reclaimable = pool->reclaimableBytes(reclaimableBytes);
    candidates.push_back(
        {reclaimable, reclaimableBytes, pool->freeBytes(), pool.get()});
  }
  return candidates;
}

bool SharedArbitrator::growMemory(
    MemoryPool* pool,
    const std::vector<std::shared_ptr<MemoryPool>>& candidatePools,
    uint64_t targetBytes) {
  ScopedArbitration scopedArbitration(pool, this);
  MemoryPool* requestor = pool->root();
  if (FOLLY_UNLIKELY(requestor->aborted())) {
    ++numFailures_;
    VELOX_MEM_POOL_ABORTED(requestor);
  }

  if (FOLLY_UNLIKELY(!ensureCapacity(requestor, targetBytes))) {
    ++numFailures_;
    VELOX_MEM_LOG(ERROR) << "Can't grow " << requestor->name()
                         << " capacity to "
                         << succinctBytes(requestor->capacity() + targetBytes)
                         << " which exceeds its max capacity "
                         << succinctBytes(requestor->maxCapacity());
    return false;
  }

  std::vector<Candidate> candidates;
  candidates.reserve(candidatePools.size());
  int numRetries{0};
  for (;; ++numRetries) {
    // Get refreshed stats before the memory arbitration retry.
    candidates = getCandidateStats(candidatePools);
    if (arbitrateMemory(requestor, candidates, targetBytes)) {
      return true;
    }
    if (!retryArbitrationFailure_ || numRetries > 0) {
      break;
    }
    VELOX_CHECK(!requestor->aborted());
    handleOOM(requestor, targetBytes, candidates);
  }
  VELOX_MEM_LOG(ERROR)
      << "Failed to arbitrate sufficient memory for memory pool "
      << requestor->name() << ", request " << succinctBytes(targetBytes)
      << " after " << numRetries
      << " retries, Arbitrator state: " << toString();
  ++numFailures_;
  return false;
}

bool SharedArbitrator::checkCapacityGrowth(
    const MemoryPool& pool,
    uint64_t targetBytes) const {
  return (maxGrowBytes(pool) >= targetBytes) &&
      (capacityAfterGrowth(pool, targetBytes) <= capacity_);
}

bool SharedArbitrator::ensureCapacity(
    MemoryPool* requestor,
    uint64_t targetBytes) {
  if ((targetBytes > capacity_) || (targetBytes > requestor->maxCapacity())) {
    return false;
  }
  if (checkCapacityGrowth(*requestor, targetBytes)) {
    return true;
  }
  const uint64_t reclaimedBytes = reclaim(requestor, targetBytes);
  // NOTE: return the reclaimed bytes back to the arbitrator and let the memory
  // arbitration process to grow the requestor's memory capacity accordingly.
  incrementFreeCapacity(reclaimedBytes);
  // Check if the requestor has been aborted in reclaim operation above.
  if (requestor->aborted()) {
    ++numFailures_;
    VELOX_MEM_POOL_ABORTED(requestor);
  }
  return checkCapacityGrowth(*requestor, targetBytes);
}

void SharedArbitrator::handleOOM(
    MemoryPool* requestor,
    uint64_t targetBytes,
    std::vector<Candidate>& candidates) {
  MemoryPool* victim =
      findCandidateWithLargestCapacity(requestor, targetBytes, candidates).pool;
  if (requestor != victim) {
    VELOX_MEM_LOG(WARNING) << "Aborting victim memory pool " << victim->name()
                           << " to free up memory for requestor "
                           << requestor->name();
  } else {
    VELOX_MEM_LOG(ERROR) << "Aborting requestor memory pool "
                         << requestor->name() << " itself to free up memory";
  }
  abort(victim);
  // Free up all the unused capacity from the aborted memory pool and gives back
  // to the arbitrator.
  incrementFreeCapacity(victim->shrink());
  if (requestor == victim) {
    ++numFailures_;
    VELOX_MEM_POOL_ABORTED(requestor);
  }
}

bool SharedArbitrator::arbitrateMemory(
    MemoryPool* requestor,
    std::vector<Candidate>& candidates,
    uint64_t targetBytes) {
  VELOX_CHECK(!requestor->aborted());

  const uint64_t growTarget = std::min(
      maxGrowBytes(*requestor),
      std::max(minMemoryPoolCapacityTransferSize_, targetBytes));
  uint64_t freedBytes = decrementFreeCapacity(growTarget);
  if (freedBytes >= targetBytes) {
    requestor->grow(freedBytes);
    return true;
  }
  VELOX_CHECK_LT(freedBytes, growTarget);

  auto freeGuard = folly::makeGuard([&]() {
    // Returns the unused freed memory capacity back to the arbitrator.
    if (freedBytes > 0) {
      incrementFreeCapacity(freedBytes);
    }
  });

  freedBytes +=
      reclaimFreeMemoryFromCandidates(candidates, growTarget - freedBytes);
  if (freedBytes >= targetBytes) {
    const uint64_t bytesToGrow = std::min(growTarget, freedBytes);
    requestor->grow(bytesToGrow);
    freedBytes -= bytesToGrow;
    return true;
  }

  VELOX_CHECK_LT(freedBytes, growTarget);
  freedBytes += reclaimUsedMemoryFromCandidates(
      requestor, candidates, growTarget - freedBytes);
  if (requestor->aborted()) {
    ++numFailures_;
    VELOX_MEM_POOL_ABORTED(requestor);
  }

  VELOX_CHECK(!requestor->aborted());

  if (freedBytes < targetBytes) {
    VELOX_MEM_LOG(WARNING)
        << "Failed to arbitrate sufficient memory for memory pool "
        << requestor->name() << ", request " << succinctBytes(targetBytes)
        << ", only " << succinctBytes(freedBytes)
        << " has been freed, Arbitrator state: " << toString();
    return false;
  }

  const uint64_t bytesToGrow = std::min(freedBytes, growTarget);
  requestor->grow(bytesToGrow);
  freedBytes -= bytesToGrow;
  return true;
}

uint64_t SharedArbitrator::reclaimFreeMemoryFromCandidates(
    std::vector<Candidate>& candidates,
    uint64_t targetBytes) {
  // Sort candidate memory pools based on their free capacity.
  sortCandidatesByFreeCapacity(candidates);

  uint64_t freedBytes{0};
  for (const auto& candidate : candidates) {
    VELOX_CHECK_LT(freedBytes, targetBytes);
    if (candidate.freeBytes == 0) {
      break;
    }
    const int64_t bytesToShrink =
        std::min<int64_t>(targetBytes - freedBytes, candidate.freeBytes);
    if (bytesToShrink <= 0) {
      break;
    }
    freedBytes += candidate.pool->shrink(bytesToShrink);
    if (freedBytes >= targetBytes) {
      break;
    }
  }
  numShrunkBytes_ += freedBytes;
  return freedBytes;
}

uint64_t SharedArbitrator::reclaimUsedMemoryFromCandidates(
    MemoryPool* requestor,
    std::vector<Candidate>& candidates,
    uint64_t targetBytes) {
  // Sort candidate memory pools based on their reclaimable memory.
  sortCandidatesByReclaimableMemory(candidates);

  int64_t freedBytes{0};
  for (const auto& candidate : candidates) {
    VELOX_CHECK_LT(freedBytes, targetBytes);
    if (!candidate.reclaimable || candidate.reclaimableBytes == 0) {
      break;
    }
    const int64_t bytesToReclaim = std::max<int64_t>(
        targetBytes - freedBytes, minMemoryPoolCapacityTransferSize_);
    VELOX_CHECK_GT(bytesToReclaim, 0);
    freedBytes += reclaim(candidate.pool, bytesToReclaim);
    if ((freedBytes >= targetBytes) || requestor->aborted()) {
      break;
    }
  }
  return freedBytes;
}

uint64_t SharedArbitrator::reclaim(
    MemoryPool* pool,
    uint64_t targetBytes) noexcept {
  const uint64_t oldCapacity = pool->capacity();
  uint64_t freedBytes{0};
  try {
    freedBytes = pool->shrink(targetBytes);
    if (freedBytes < targetBytes) {
      pool->reclaim(targetBytes - freedBytes);
    }
  } catch (const std::exception& e) {
    VELOX_MEM_LOG(ERROR) << "Failed to reclaim from memory pool "
                         << pool->name() << ", aborting it!";
    abort(pool);
    // Free up all the free capacity from the aborted pool as the associated
    // query has failed at this point.
    pool->shrink();
  }
  const uint64_t newCapacity = pool->capacity();
  VELOX_CHECK_GE(oldCapacity, newCapacity);
  const uint64_t reclaimedbytes = oldCapacity - newCapacity;
  numShrunkBytes_ += freedBytes;
  numReclaimedBytes_ += reclaimedbytes - freedBytes;
  return reclaimedbytes;
}

void SharedArbitrator::abort(MemoryPool* pool) {
  ++numAborted_;
  try {
    pool->abort();
  } catch (const std::exception& e) {
    VELOX_MEM_LOG(WARNING) << "Failed to abort memory pool "
                           << pool->toString();
  }
  // NOTE: no matter memory pool abort throws or not, it should have been marked
  // as aborted to prevent any new memory arbitration triggered from the aborted
  // memory pool.
  VELOX_CHECK(pool->aborted());
}

uint64_t SharedArbitrator::decrementFreeCapacity(uint64_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  return decrementFreeCapacityLocked(bytes);
}

uint64_t SharedArbitrator::decrementFreeCapacityLocked(uint64_t bytes) {
  const uint64_t targetBytes = std::min(freeCapacity_, bytes);
  VELOX_CHECK_LE(targetBytes, freeCapacity_);
  freeCapacity_ -= targetBytes;
  return targetBytes;
}

void SharedArbitrator::incrementFreeCapacity(uint64_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  incrementFreeCapacityLocked(bytes);
}

void SharedArbitrator::incrementFreeCapacityLocked(uint64_t bytes) {
  freeCapacity_ += bytes;
  if (FOLLY_UNLIKELY(freeCapacity_ > capacity_)) {
    VELOX_FAIL(
        "The free capacity {} is larger than the max capacity {}, {}",
        succinctBytes(freeCapacity_),
        succinctBytes(capacity_),
        toStringLocked());
  }
}

MemoryArbitrator::Stats SharedArbitrator::stats() const {
  std::lock_guard<std::mutex> l(mutex_);
  return statsLocked();
}

MemoryArbitrator::Stats SharedArbitrator::statsLocked() const {
  Stats stats;
  stats.numRequests = numRequests_;
  stats.numAborted = numAborted_;
  stats.numFailures = numFailures_;
  stats.queueTimeUs = queueTimeUs_;
  stats.arbitrationTimeUs = arbitrationTimeUs_;
  stats.numShrunkBytes = numShrunkBytes_;
  stats.numReclaimedBytes = numReclaimedBytes_;
  stats.maxCapacityBytes = capacity_;
  stats.freeCapacityBytes = freeCapacity_;
  return stats;
}

std::string SharedArbitrator::toString() const {
  std::lock_guard<std::mutex> l(mutex_);
  return toStringLocked();
}

std::string SharedArbitrator::toStringLocked() const {
  return fmt::format(
      "ARBITRATOR[{}] CAPACITY {} {}",
      kindString(kind_),
      succinctBytes(capacity_),
      statsLocked().toString());
}

SharedArbitrator::ScopedArbitration::ScopedArbitration(
    MemoryPool* requestor,
    SharedArbitrator* arbitrator)
    : requestor_(requestor),
      arbitrator_(arbitrator),
      startTime_(std::chrono::steady_clock::now()) {
  VELOX_CHECK_NOT_NULL(requestor_);
  VELOX_CHECK_NOT_NULL(arbitrator_);
  arbitrator_->startArbitration(requestor);
}

SharedArbitrator::ScopedArbitration::~ScopedArbitration() {
  requestor_->leaveArbitration();
  const auto arbitrationTime =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - startTime_);
  arbitrator_->arbitrationTimeUs_ += arbitrationTime.count();
  arbitrator_->finishArbitration();
}

void SharedArbitrator::startArbitration(MemoryPool* requestor) {
  requestor->enterArbitration();
  ContinueFuture waitPromise{ContinueFuture::makeEmpty()};
  {
    std::lock_guard<std::mutex> l(mutex_);
    ++numRequests_;
    if (running_) {
      waitPromises_.emplace_back(fmt::format(
          "Wait for arbitration, requestor: {}[{}]",
          requestor->name(),
          requestor->root()->name()));
      waitPromise = waitPromises_.back().getSemiFuture();
    } else {
      VELOX_CHECK(waitPromises_.empty());
      running_ = true;
    }
  }

  TestValue::adjust(
      "facebook::velox::memory::SharedArbitrator::startArbitration", requestor);

  if (waitPromise.valid()) {
    uint64_t waitTimeUs{0};
    {
      MicrosecondTimer timer(&waitTimeUs);
      waitPromise.wait();
    }
    queueTimeUs_ += waitTimeUs;
  }
}

void SharedArbitrator::finishArbitration() {
  ContinuePromise resumePromise{ContinuePromise::makeEmpty()};
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(running_);
    if (!waitPromises_.empty()) {
      resumePromise = std::move(waitPromises_.back());
      waitPromises_.pop_back();
    } else {
      running_ = false;
    }
  }
  if (resumePromise.valid()) {
    resumePromise.setValue();
  }
}
} // namespace facebook::velox::memory
