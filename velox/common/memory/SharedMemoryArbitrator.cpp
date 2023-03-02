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

#include "velox/common/memory/SharedMemoryArbitrator.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"

DECLARE_uint64(velox_memory_shared_arbitrator_min_reserve_bytes);

namespace facebook::velox::memory {
namespace {
void sortCandidatesByShrinkableBytes(
    MemoryPool* requestor,
    std::vector<MemoryPool*>& candidates) {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](MemoryPool* lhs, MemoryPool* rhs) {
        if (lhs == requestor) {
          return true;
        }
        if (lhs == requestor) {
          return false;
        }
        return lhs->shrinkableBytes() > rhs->shrinkableBytes();
      });
}

void sortCandidatesByReclaimableBytes(std::vector<MemoryPool*>& candidates) {
  bool reclaimable;
  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](MemoryPool* lhs, MemoryPool* rhs) {
        return lhs->reclaimableBytes(reclaimable) >
            rhs->reclaimableBytes(reclaimable);
      });
}
} // namespace

SharedMemoryArbitrator::Traits SharedMemoryArbitrator::traits() {
  static Traits traits{
      .minReserveBytes =
          FLAGS_velox_memory_shared_arbitrator_min_reserve_bytes};
  return traits;
}

SharedMemoryArbitrator::~SharedMemoryArbitrator() {
  VELOX_CHECK_EQ(freeCapacity_, capacity_);
}

int64_t SharedMemoryArbitrator::reserveMemory(int64_t targetBytes) {
  if (targetBytes == kMaxMemory) {
    return targetBytes;
  }
  std::lock_guard<std::mutex> l(mutex_);
  int64_t reserveBytes =
      std::max<int64_t>(targetBytes, traits().minReserveBytes);
  reserveBytes = std::min(reserveBytes, freeCapacity_);
  freeCapacity_ -= reserveBytes;
  return reserveBytes;
}

void SharedMemoryArbitrator::releaseMemory(MemoryPool* releasor) {
  VELOX_CHECK_EQ(releasor->currentBytes(), 0);
  const int64_t bytesToRelease = releasor->shrink();
  releaseFreeCapacity(bytesToRelease);
}

bool SharedMemoryArbitrator::growMemory(
    MemoryPool* requestor,
    const std::vector<MemoryPool*>& candidates,
    uint64_t targetBytes) {
  ScopedArbitration scopedArbitration(requestor, this);
  if (growMemoryFast(
          requestor, std::max(traits().minReserveBytes, targetBytes))) {
    return true;
  }
  return growMemorySlow(requestor->root(), candidates, targetBytes);
}

bool SharedMemoryArbitrator::growMemoryFast(
    MemoryPool* requestor,
    uint64_t targetBytes) {
  ContinueFuture future;
  bool wait{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    ++stats_.numRequests;
    if (running_) {
      wait = true;
      ++stats_.numWaits;
      promises_.emplace_back("Wait for arbitration run");
      future = promises_.back().getSemiFuture();
    } else {
      if (growMemoryFromFreeCapacityLocked(requestor, targetBytes)) {
        return true;
      }
      running_ = true;
    }
  }

  if (wait) {
    uint64_t waitTimeUs{0};
    {
      MicrosecondTimer timer(&waitTimeUs);
      VELOX_CHECK(future.valid());
      future.wait();
    }
    stats_.waitTimeUs += waitTimeUs;
  }
  return false;
}

bool SharedMemoryArbitrator::growMemorySlow(
    MemoryPool* requestor,
    const std::vector<MemoryPool*>& candidates,
    uint64_t targetBytes) {
  uint64_t arbitrationTimeUs{0};
  MicrosecondTimer timer(&arbitrationTimeUs);
  const bool success = arbitrate(requestor, candidates, targetBytes);
  ++stats_.numArbitrations;
  stats_.arbitrationTimeUs += arbitrationTimeUs;
  if (!success) {
    ++stats_.numFailures;
  }
  return success;
}

bool SharedMemoryArbitrator::growMemoryFromFreeCapacityLocked(
    MemoryPool* requestor,
    uint64_t targetBytes) {
  if (freeCapacity_ < targetBytes) {
    return false;
  }
  chargeFreeCapacityLocked(targetBytes);
  requestor->grow(targetBytes);
  return true;
}

bool SharedMemoryArbitrator::arbitrate(
    MemoryPool* requestor,
    std::vector<MemoryPool*> candidates,
    uint64_t targetBytes) {
  int64_t remainingBytes = std::max(traits().minReserveBytes, targetBytes);
  int64_t freedBytes{0};
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(running_);
    if (growMemoryFromFreeCapacityLocked(requestor, remainingBytes)) {
      return true;
    }
    freedBytes = chargeFreeCapacityLocked(freeCapacity_);
    remainingBytes -= freedBytes;
  }
  VELOX_CHECK_GT(remainingBytes, 0);

  freedBytes += shrinkFromCandidates(requestor, candidates, remainingBytes);
  if (freedBytes >= targetBytes) {
    requestor->grow(freedBytes);
    return true;
  }

  remainingBytes = targetBytes - freedBytes;
  VELOX_CHECK_GT(remainingBytes, 0);
  freedBytes += reclaimFromCandidates(
      candidates,
      remainingBytes,
      std::max(traits().minReserveBytes, targetBytes) - freedBytes);
  if (freedBytes >= targetBytes) {
    requestor->grow(freedBytes);
    return true;
  }
  VELOX_MEM_LOG(WARNING) << "Failed to arbitrate " << succinctBytes(targetBytes)
                         << " for memory pool " << requestor->name() << " only "
                         << succinctBytes(freedBytes)
                         << " has been freed, Arbitrator state: {}"
                         << toString();
  releaseFreeCapacity(freedBytes);
  return false;
}

int64_t SharedMemoryArbitrator::shrinkFromCandidates(
    MemoryPool* requestor,
    std::vector<MemoryPool*> candidates,
    uint64_t targetBytes) {
  // Sort based on the shrinkable bytes.
  sortCandidatesByShrinkableBytes(requestor, candidates);

  int64_t remainingBytes = targetBytes;
  int64_t freedBytes{0};
  for (auto* candidate : candidates) {
    VELOX_CHECK_GT(remainingBytes, 0);
    const bool requestorSelf = (candidate == requestor);
    const int64_t bytesToShrink = std::min<int64_t>(
        remainingBytes,
        candidate->shrinkableBytes() -
            (requestorSelf ? 0 : traits().minReserveBytes));
    if (bytesToShrink <= 0) {
      if (requestorSelf) {
        continue;
      }
      break;
    }
    freedBytes += candidate->shrink(bytesToShrink);
    if (freedBytes >= targetBytes) {
      break;
    }
    remainingBytes = targetBytes - freedBytes;
  }
  stats_.numShrinkedBytes += freedBytes;
  return freedBytes;
}

int64_t SharedMemoryArbitrator::reclaimFromCandidates(
    std::vector<MemoryPool*> candidates,
    uint64_t minTargetBytes,
    uint64_t maxTargetBytes) {
  // Sort based on the reclaimable bytes.
  sortCandidatesByReclaimableBytes(candidates);

  int64_t remainingBytes = maxTargetBytes;
  int64_t freedBytes{0};
  for (auto& candidate : candidates) {
    VELOX_CHECK_GT(remainingBytes, 0);
    bool reclaimable;
    const int64_t bytesToReclaim = std::min<int64_t>(
        remainingBytes, candidate->reclaimableBytes(reclaimable));
    VELOX_CHECK(reclaimable || bytesToReclaim == 0);
    if (bytesToReclaim <= 0) {
      break;
    }
    candidate->reclaim(
        std::max<int64_t>(traits().minReclaimBytes, bytesToReclaim));
    const int64_t bytesToShrink = std::min<int64_t>(
        remainingBytes,
        candidate->shrinkableBytes() - traits().minReserveBytes);
    freedBytes += candidate->shrink(bytesToShrink);
    if (freedBytes >= minTargetBytes) {
      break;
    }
    remainingBytes = maxTargetBytes - freedBytes;
  }
  stats_.numReclaimedBytes = freedBytes;
  return freedBytes;
}

int64_t SharedMemoryArbitrator::chargeFreeCapacityLocked(int64_t bytes) {
  int64_t chargedBytes = bytes;
  if (bytes == 0) {
    chargedBytes = freeCapacity_;
  }
  freeCapacity_ -= chargedBytes;
  return chargedBytes;
}

void SharedMemoryArbitrator::releaseFreeCapacity(int64_t bytes) {
  std::lock_guard<std::mutex> l(mutex_);
  releaseFreeCapacityLocked(bytes);
}

void SharedMemoryArbitrator::releaseFreeCapacityLocked(int64_t bytes) {
  freeCapacity_ += bytes;
  VELOX_CHECK_LE(freeCapacity_, capacity_);
}

void SharedMemoryArbitrator::resume() {
  ContinuePromise promise = ContinuePromise::makeEmpty();
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(running_);
    if (!promises_.empty()) {
      promise = std::move(promises_.back());
      promises_.pop_back();
    }
    running_ = false;
  }
  if (promise.valid()) {
    promise.setValue();
  }
}

std::string SharedMemoryArbitrator::toString() const {
  return fmt::format(
      "ARBITRATOR {} CAPACITY {} STATS {}",
      kindString(kind_),
      succinctBytes(capacity_),
      stats().toString());
}

SharedMemoryArbitrator::ScopedArbitration::ScopedArbitration(
    MemoryPool* requestor,
    SharedMemoryArbitrator* arbitrator)
    : requestor_(requestor), arbitrator_(arbitrator) {
  VELOX_CHECK_NOT_NULL(requestor_);
  VELOX_CHECK_NOT_NULL(arbitrator_);
  requestor_->enterArbitration();
  resume_ = true;
}

SharedMemoryArbitrator::ScopedArbitration::~ScopedArbitration() {
  requestor_->leaveArbitration();
  if (resume_) {
    arbitrator_->resume();
  }
}
} // namespace facebook::velox::memory
