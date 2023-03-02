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

#include "velox/common/memory/MemoryArbitrator.h"

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedMemoryArbitrator.h"

namespace facebook::velox::memory {
std::string MemoryArbitrator::kindString(Kind kind) {
  switch (kind) {
    case Kind::kFixed:
      return "FIXED";
    case Kind::kShared:
      return "SHARED";
    default:
      return fmt::format("UNKNOWN: {}", static_cast<int>(kind));
  }
}

std::ostream& operator<<(
    std::ostream& out,
    const MemoryArbitrator::Kind& kind) {
  out << MemoryArbitrator::kindString(kind);
  return out;
}

std::unique_ptr<MemoryArbitrator> MemoryArbitrator::create(
    const Config& config) {
  switch (config.kind) {
    case Kind::kFixed:
      return std::make_unique<FixedMemoryArbitrator>(config);
    case Kind::kShared:
      return std::make_unique<SharedMemoryArbitrator>(config);
    default:
      VELOX_UNREACHABLE(kindString(config.kind));
  }
}

int64_t FixedMemoryArbitrator::reserveMemory(int64_t targetBytes) {
  if (targetBytes == kMaxMemory) {
    return targetBytes;
  }
  return std::min<int64_t>(targetBytes, capacity_);
}

std::string FixedMemoryArbitrator::toString() const {
  return fmt::format(
      "ARBITRATOR {} CAPACITY {}", kindString(kind_), succinctBytes(capacity_));
}

std::shared_ptr<MemoryReclaimer> MemoryReclaimer::create() {
  return std::shared_ptr<MemoryReclaimer>(new MemoryReclaimer());
}

bool MemoryReclaimer::canReclaim(const MemoryPool& pool) const {
  return pool.kind() == MemoryPool::Kind::kLeaf ? false : true;
}

uint64_t MemoryReclaimer::reclaimableBytes(
    const MemoryPool& pool,
    bool& reclaimable) const {
  if (pool.kind() == MemoryPool::Kind::kLeaf) {
    reclaimable = canReclaim(pool);
    return 0;
  }
  reclaimable = false;
  uint64_t reclaimableBytes{0};
  pool.visitChildren([&](MemoryPool* pool) {
    bool poolReclaimable;
    reclaimableBytes += pool->reclaimableBytes(poolReclaimable);
    reclaimable |= poolReclaimable;
    return true;
  });
  return reclaimableBytes;
}

uint64_t MemoryReclaimer::reclaim(MemoryPool* pool, uint64_t targetBytes) {
  if (pool->kind() == MemoryPool::Kind::kLeaf) {
    return 0;
  }
  uint64_t reclaimedBytes{0};
  pool->visitChildren([&](MemoryPool* pool) {
    const auto bytes = pool->reclaim(targetBytes);
    reclaimedBytes += bytes;
    if (bytes >= targetBytes) {
      return false;
    }
    targetBytes -= bytes;
    return true;
  });
  return reclaimedBytes;
}

std::string MemoryArbitrator::Stats::toString() const {
  return fmt::format(
      "STATS[numRequests {} numArbitrations {} numFailures {} numRetries {} numShrinkedBytes {} numReclaimedBytes {} numWaits {} waitTimeUs {} arbitrationTimeUs {}]",
      numRequests,
      numArbitrations,
      numFailures,
      numRetries,
      numShrinkedBytes,
      numReclaimedBytes,
      numWaits,
      waitTimeUs,
      arbitrationTimeUs);
}
} // namespace facebook::velox::memory
