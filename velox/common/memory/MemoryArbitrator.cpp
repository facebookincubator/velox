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
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TestValue.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::memory {
std::string MemoryArbitrator::kindString(Kind kind) {
  switch (kind) {
    case Kind::kNoOp:
      return "NOOP";
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
    case Kind::kNoOp:
      return nullptr;
    case Kind::kShared:
      return std::make_unique<SharedArbitrator>(config);
    default:
      VELOX_UNREACHABLE(kindString(config.kind));
  }
}

std::shared_ptr<MemoryReclaimer> MemoryReclaimer::create() {
  return std::shared_ptr<MemoryReclaimer>(new MemoryReclaimer());
}

bool MemoryReclaimer::reclaimableBytes(
    const MemoryPool& pool,
    uint64_t& reclaimableBytes) const {
  reclaimableBytes = 0;
  if (pool.kind() == MemoryPool::Kind::kLeaf) {
    return false;
  }
  bool reclaimable{false};
  pool.visitChildren([&](MemoryPool* pool) {
    uint64_t poolReclaimableBytes{0};
    reclaimable |= pool->reclaimableBytes(poolReclaimableBytes);
    reclaimableBytes += poolReclaimableBytes;
    return true;
  });
  VELOX_CHECK(reclaimable || reclaimableBytes == 0);
  return reclaimable;
}

uint64_t MemoryReclaimer::reclaim(MemoryPool* pool, uint64_t targetBytes) {
  if (pool->kind() == MemoryPool::Kind::kLeaf) {
    return 0;
  }
  std::vector<std::shared_ptr<MemoryPool>> children;
  {
    folly::SharedMutex::ReadHolder guard{pool->childrenMutex_};
    children.reserve(pool->children_.size());
    for (auto& entry : pool->children_) {
      auto child = entry.second.lock();
      if (child != nullptr) {
        children.push_back(std::move(child));
      }
    }
  }

  std::vector<ArbitrationCandidate> candidates(children.size());
  for (uint32_t i = 0; i < children.size(); ++i) {
    candidates[i].pool = children[i].get();
    candidates[i].reclaimable =
        children[i]->reclaimableBytes(candidates[i].reclaimableBytes);
  }
  sortCandidatesByReclaimableMemory(candidates);

  uint64_t reclaimedBytes{0};
  for (const auto& candidate : candidates) {
    if (!candidate.reclaimable || candidate.reclaimableBytes == 0) {
      break;
    }
    const auto bytes = candidate.pool->reclaim(targetBytes);
    reclaimedBytes += bytes;
    if (targetBytes != 0) {
      if (bytes >= targetBytes) {
        break;
      }
      targetBytes -= bytes;
    }
  }
  return reclaimedBytes;
}

std::string MemoryArbitrator::Stats::toString() const {
  return fmt::format(
      "STATS[numRequests {} numFailures {} queueTime {} arbitrationTime {} shrunkMemory {} reclaimedMemory {} maxCapacity {} freeCapacity {}]",
      numRequests,
      numFailures,
      succinctMicros(queueTimeUs),
      succinctMicros(arbitrationTimeUs),
      succinctBytes(numShrunkBytes),
      succinctBytes(numReclaimedBytes),
      succinctBytes(maxCapacityBytes),
      succinctBytes(freeCapacityBytes));
}

std::string ArbitrationCandidate::toString() const {
  return fmt::format(
      "[reclaimable {} reclaimableBytes {} freeBytes {} pool {}]",
      reclaimable,
      succinctBytes(reclaimableBytes),
      succinctBytes(freeBytes),
      reinterpret_cast<uint64_t>(pool));
}

void sortCandidatesByReclaimableMemory(
    std::vector<ArbitrationCandidate>& candidates) {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const ArbitrationCandidate& lhs, const ArbitrationCandidate& rhs) {
        if (!lhs.reclaimable) {
          return false;
        }
        if (!rhs.reclaimable) {
          return true;
        }
        return lhs.reclaimableBytes > rhs.reclaimableBytes;
      });
  TestValue::adjust(
      "facebook::velox::memory::sortCandidatesByReclaimableMemory",
      &candidates);
}

void sortCandidatesByFreeCapacity(
    std::vector<ArbitrationCandidate>& candidates) {
  std::sort(
      candidates.begin(),
      candidates.end(),
      [&](const ArbitrationCandidate& lhs, const ArbitrationCandidate& rhs) {
        return lhs.freeBytes > rhs.freeBytes;
      });

  TestValue::adjust(
      "facebook::velox::memory::sortCandidatesByFreeCapacity", &candidates);
}
} // namespace facebook::velox::memory
