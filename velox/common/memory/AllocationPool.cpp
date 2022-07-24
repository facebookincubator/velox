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
#include <velox/common/base/Exceptions.h>
#include <velox/common/memory/MappedMemory.h>
#include "velox/common/base/BitUtil.h"

namespace facebook::velox {

void AllocationPool::clear() {
  // Trigger Allocation's destructor to free allocated memory
  if (allocation_ != nullptr) {
    auto copy = std::move(allocation_);
  }
  allocations_.clear();
}

char* AllocationPool::allocateFixed(uint64_t bytes) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");
  auto numPages = bits::roundUp(bytes, memory::MappedMemory::kPageSize) /
      memory::MappedMemory::kPageSize;
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
  // If we don't have another run in the current allocation or we have but the
  // run's size is not enough, we do a new allocation
  if (allocation_ == nullptr || currentRun_ >= allocation_->numRuns() ||
      currentRun().numPages() < numPages) {
    if (allocation_ != nullptr) {
      VELOX_CHECK_GE(allocation_->numRuns(), 1);
      allocations_.push_back(std::move(allocation_));
    }

    // Use contiguous allocations from mapped memory if allocation size is large
    if (numPages > mappedMemory_->largestSizeClass()) {
      auto largeAlloc =
          std::make_unique<memory::MappedMemory::ContiguousAllocation>(
              mappedMemory_);
      largeAlloc->reset(nullptr, 0);
      if (!mappedMemory_->allocateContiguous(numPages, nullptr, *largeAlloc)) {
        throw std::bad_alloc();
      }
      allocation_ = std::move(largeAlloc);
    } else {
      allocation_ =
          std::make_unique<memory::MappedMemory::Allocation>(mappedMemory_);
      if (!mappedMemory_->allocate(
              std::max<int32_t>(kMinPages, numPages),
              owner_,
              *allocation_,
              nullptr,
              numPages)) {
        throw std::bad_alloc();
      }
    }
    currentRun_ = 0;
  }
  currentOffset_ = 0;
}

void AllocationPool::newRun(int32_t preferredSize) {
  auto numPages =
      bits::roundUp(preferredSize, memory::MappedMemory::kPageSize) /
      memory::MappedMemory::kPageSize;
  newRunImpl(numPages);
}

} // namespace facebook::velox
