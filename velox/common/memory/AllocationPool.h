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

#include "velox/common/memory/Memory.h"

namespace facebook::velox {
// A set of Allocations holding the fixed width payload
// rows. The Runs are filled to the end except for the last one. This
// is used for iterating over the payload for rehashing, returning
// results etc. This is used via HashStringAllocator for variable length
// allocation for backing ByteStreams for complex objects. In that case, there
// is a current run that is appended to and when this is exhausted a new run is
// started.
class AllocationPool {
 public:
  static constexpr int32_t kMinPages = 16;

  explicit AllocationPool(memory::MemoryPool* pool) : pool_(pool) {}

  ~AllocationPool() = default;

  void clear();

  // Allocate a buffer from this pool, optionally aligned.  The alignment can
  // only be power of 2.
  char* allocateFixed(uint64_t bytes, int32_t alignment = 1);

  // Starts a new run for variable length allocation. The actual size
  // is at least one machine page. Throws std::bad_alloc if no space.
  void newRun(int32_t preferredSize);

  int32_t numTotalAllocations() const {
    return numSmallAllocations() + numLargeAllocations();
  }

  int32_t numSmallAllocations() const {
    return 1 + allocations_.size();
  }

  int32_t numLargeAllocations() const {
    return largeAllocations_.size();
  }

  const memory::Allocation* allocationAt(int32_t index) const {
    return index == allocations_.size() ? &allocation_
                                        : allocations_[index].get();
  }

  const memory::ContiguousAllocation* largeAllocationAt(int32_t index) const {
    return largeAllocations_[index].get();
  }

  int32_t currentRunIndex() const {
    return currentRun_;
  }

  int64_t currentOffset() const {
    return currentOffset_;
  }

  int64_t allocatedBytes() const {
    int32_t totalPages = allocation_.numPages();
    for (auto& allocation : allocations_) {
      totalPages += allocation->numPages();
    }
    for (auto& largeAllocation : largeAllocations_) {
      totalPages += largeAllocation->numPages();
    }
    return totalPages * memory::AllocationTraits::kPageSize;
  }

  // Returns number of bytes left at the end of the current run.
  int32_t availableInRun() const {
    if (!allocation_.numRuns()) {
      return 0;
    }
    return currentRun().numBytes() - currentOffset_;
  }

  // Returns pointer to first unallocated byte in the current run.
  char* firstFreeInRun() {
    VELOX_DCHECK(availableInRun() > 0);
    return currentRun().data<char>() + currentOffset_;
  }

  // Sets the first free position in the current run.
  void setFirstFreeInRun(const char* firstFree) {
    auto run = currentRun();
    auto offset = firstFree - run.data<char>();
    VELOX_CHECK(
        offset >= 0 && offset <= run.numBytes(),
        "Trying to set end of allocation outside of last allocated run");
    currentOffset_ = offset;
  }

  memory::MemoryPool* pool() const {
    return pool_;
  }

 private:
  memory::Allocation::PageRun currentRun() const {
    return allocation_.runAt(currentRun_);
  }

  void newRunImpl(memory::MachinePageCount numPages);

  memory::MemoryPool* pool_;
  std::vector<std::unique_ptr<memory::Allocation>> allocations_;
  std::vector<std::unique_ptr<memory::ContiguousAllocation>> largeAllocations_;
  memory::Allocation allocation_;
  int32_t currentRun_ = 0;
  int32_t currentOffset_ = 0;
};

} // namespace facebook::velox
