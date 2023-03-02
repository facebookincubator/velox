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

#include "velox/common/future/VeloxPromise.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::memory {
class SharedMemoryArbitrator : public MemoryArbitrator {
 public:
  SharedMemoryArbitrator(const Config& config)
      : MemoryArbitrator(config), freeCapacity_(capacity_) {
    VELOX_CHECK_EQ(kind_, Kind::kShared);
  }

  ~SharedMemoryArbitrator() override;

  virtual int64_t reserveMemory(int64_t targetBytes) final;

  void releaseMemory(MemoryPool* releasor) final;

  bool growMemory(
      MemoryPool* requestor,
      const std::vector<MemoryPool*>& candidates,
      uint64_t targetBytes) final;

  Stats stats() const final {
    // TODO: add implementation later.
    return {};
  }

  std::string toString() const final;

 private:
  struct Traits {
    uint64_t minReserveBytes;
    uint64_t minReclaimBytes;
  };
  static Traits traits();

  class ScopedArbitration {
   public:
    ScopedArbitration(
        MemoryPool* requestor,
        SharedMemoryArbitrator* arbitrator);

    ~ScopedArbitration();

   private:
    MemoryPool* const requestor_;
    SharedMemoryArbitrator* const arbitrator_;
    bool resume_{false};
  };

  bool growMemoryFast(MemoryPool* requestor, uint64_t targetBytes);

  bool growMemorySlow(
      MemoryPool* requestor,
      const std::vector<MemoryPool*>& candidates,
      uint64_t targetBytes);

  bool arbitrate(
      MemoryPool* requestor,
      std::vector<MemoryPool*> candidates,
      uint64_t targetBytes);

  bool growMemoryFromFreeCapacityLocked(
      MemoryPool* requestor,
      uint64_t targetBytes);

  int64_t shrinkFromCandidates(
      MemoryPool* requestor,
      std::vector<MemoryPool*> candidates,
      uint64_t targetBytes);

  int64_t reclaimFromCandidates(
      std::vector<MemoryPool*> candidates,
      uint64_t minTargetBytes,
      uint64_t maxTargetBytes);

  int64_t chargeFreeCapacityLocked(int64_t bytes);
  void releaseFreeCapacity(int64_t bytes);
  void releaseFreeCapacityLocked(int64_t bytes);

  void resume();

  mutable std::mutex mutex_;
  int64_t freeCapacity_{0};
  bool running_{false};
  std::vector<ContinuePromise> promises_;
};
} // namespace facebook::velox::memory
