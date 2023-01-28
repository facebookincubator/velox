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

#include "velox/common/memory/AllocationPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox {

void AllocationPool::clear() {
  // Trigger Allocation's destructor to free allocated memory
  auto copy = std::move(allocation_);
  allocations_.clear();
  auto copyLarge = std::move(largeAllocations_);
  largeAllocations_.clear();
}

char* AllocationPool::allocateFixed(uint64_t bytes) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");

  auto numPages = bits::roundUp(bytes, memory::AllocationTraits::kPageSize) /
      memory::AllocationTraits::kPageSize;

  // Use contiguous allocations from mapped memory if allocation size is large
  if (numPages > pool_->largestSizeClass()) {
    auto largeAlloc = std::make_unique<memory::ContiguousAllocation>();
    pool_->allocateContiguous(numPages, *largeAlloc);
    largeAllocations_.emplace_back(std::move(largeAlloc));
    auto res = largeAllocations_.back()->data<char>();
    VELOX_CHECK_NOT_NULL(
        res, "Unexpected nullptr for large contiguous allocation");
    return res;
  }

  if (availableInRun() < bytes) {
    newRunImpl(numPages);
  }
  auto run = currentRun();
  uint64_t size = run.numBytes();
  VELOX_CHECK_LE(bytes + currentOffset_, size);
  currentOffset_ += bytes;
  return reinterpret_cast<char*>(run.data() + currentOffset_ - bytes);
}

void AllocationPool::newRunImpl(memory::MachinePageCount numPages) {
  ++currentRun_;
  if (currentRun_ >= allocation_.numRuns()) {
    if (allocation_.numRuns()) {
      allocations_.push_back(
          std::make_unique<memory::Allocation>(std::move(allocation_)));
    }
    pool_->allocateNonContiguous(
        std::max<int32_t>(kMinPages, numPages), allocation_, numPages);
    currentRun_ = 0;
  }
  currentOffset_ = 0;
}

void AllocationPool::newRun(int32_t preferredSize) {
  auto numPages =
      bits::roundUp(preferredSize, memory::AllocationTraits::kPageSize) /
      memory::AllocationTraits::kPageSize;
  newRunImpl(numPages);
}

} // namespace facebook::velox
