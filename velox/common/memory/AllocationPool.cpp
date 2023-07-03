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

DEFINE_int32(min_apool_mb, 0, "Min MB for alloc pool");

namespace facebook::velox {

void AllocationPool::clear() {
  allocations_.clear();
  largeAllocations_.clear();
  startOfRun_ = nullptr;
  bytesInRun_ = 0;
  currentOffset_ = 0;
}

char* AllocationPool::allocateFixed(uint64_t bytes, int32_t alignment) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");
  if (availableInRun() >= bytes && alignment == 1) {
    auto* result = startOfRun_ + currentOffset_;
    currentOffset_ += bytes;
    return result;
  }
  VELOX_CHECK_EQ(
      __builtin_popcount(alignment), 1, "Alignment can only be power of 2");

  auto numPages = memory::AllocationTraits::numPages(bytes + alignment - 1);

  // Use contiguous allocations from mapped memory if allocation size is large
  if (numPages > pool_->largestSizeClass()) {
    auto largeAlloc = std::make_unique<memory::ContiguousAllocation>();
    pool_->allocateContiguous(numPages, *largeAlloc);
    largeAllocations_.emplace_back(std::move(largeAlloc));
    auto result = largeAllocations_.back()->data<char>();
    VELOX_CHECK_NOT_NULL(
        result, "Unexpected nullptr for large contiguous allocation");
    // Should be at page boundary and always aligned.
    VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(result) % alignment, 0);
    return result;
  }

  if (availableInRun() == 0) {
    newRunImpl(numPages);
  } else {
    auto alignedBytes =
        bytes + memory::alignmentPadding(firstFreeInRun(), alignment);
    if (availableInRun() < alignedBytes) {
      newRunImpl(numPages);
    }
  }
  currentOffset_ += memory::alignmentPadding(firstFreeInRun(), alignment);
  VELOX_CHECK_LE(bytes + currentOffset_, bytesInRun_);
  auto* result = startOfRun_ + currentOffset_;
  VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(result) % alignment, 0);
  currentOffset_ += bytes;
  return result;
}

void AllocationPool::newRunImpl(memory::MachinePageCount numPages) {
  if (FLAGS_min_apool_mb > 0) {
    auto largeAlloc = std::make_unique<memory::ContiguousAllocation>();
    pool_->allocateContiguous(FLAGS_min_apool_mb * 256, *largeAlloc);
    startOfRun_ = largeAlloc->data<char>();
    bytesInRun_ = largeAlloc->size();
    largeAllocations_.emplace_back(std::move(largeAlloc));
    currentOffset_ = 0;
    return;
  }
  memory::Allocation allocation;
  auto roundedPages = std::max<int32_t>(kMinPages, numPages);
  pool_->allocateNonContiguous(roundedPages, allocation, roundedPages);
  VELOX_CHECK_EQ(allocation.numRuns(), 1);
  startOfRun_ = allocation.runAt(0).data<char>();
  bytesInRun_ = allocation.runAt(0).numBytes();
  currentOffset_ = 0;
  allocations_.push_back(
      std::make_unique<memory::Allocation>(std::move(allocation)));
}

void AllocationPool::newRun(int32_t preferredSize) {
  newRunImpl(memory::AllocationTraits::numPages(preferredSize));
}

} // namespace facebook::velox
