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

#include <algorithm>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryAllocator.h"

namespace facebook::velox::memory {

folly::Range<char*> AllocationPool::rangeAt(int32_t index) const {
  if (index < allocations_.size()) {
    auto run = allocations_[index].runAt(0);
    return folly::Range<char*>(
        run.data<char>(),
        run.data<char>() == startOfRun_ ? currentOffset_ : run.numBytes());
  }
  const auto largeIndex = index - allocations_.size();
  if (largeIndex < largeAllocations_.size()) {
    auto range = largeAllocations_[largeIndex].hugePageRange().value();
    if (range.data() == startOfRun_) {
      return folly::Range<char*>(range.data(), currentOffset_);
    }
    return range;
  }
  VELOX_FAIL("Out of range index for rangeAt(): {}", index);
}

void AllocationPool::clear() {
  allocations_.clear();
  largeAllocations_.clear();
  startOfRun_ = nullptr;
  bytesInRun_ = 0;
  currentOffset_ = 0;
  usedBytes_ = 0;
}

std::vector<AllocationPool::Relocation> AllocationPool::clone(
    const AllocationPool& src) {
  VELOX_CHECK(
      allocations_.empty() && largeAllocations_.empty(),
      "clone requires an empty destination pool");
  const auto numSmall = static_cast<int32_t>(src.allocations_.size());
  const auto numRanges = src.numRanges();
  std::vector<Relocation> relocations;
  relocations.reserve(numRanges);
  // rangeAt() lists the small runs first, then the large ones, in allocation
  // order, so the source's current run (startOfRun_) is the last range; its
  // used extent is 'currentOffset_', every earlier run is full.
  for (auto i = 0; i < numRanges; ++i) {
    const auto srcRange = src.rangeAt(i);
    char* const srcBegin = srcRange.data();
    const int64_t used = srcRange.size();
    const bool isLast = srcBegin == src.startOfRun_;
    char* destBegin;
    if (i < numSmall) {
      // Small non-contiguous run: the same page count reproduces the byte size.
      const auto numPages = src.allocations_[i].numPages();
      Allocation allocation;
      pool_->allocateNonContiguous(numPages, allocation, numPages);
      VELOX_CHECK_EQ(allocation.numRuns(), 1);
      const auto run = allocation.runAt(0);
      VELOX_CHECK(
          isLast || run.numBytes() == src.allocations_[i].runAt(0).numBytes(),
          "Cloned run size does not match the source run");
      destBegin = run.data<char>();
      startOfRun_ = destBegin;
      bytesInRun_ = run.numBytes();
      usedBytes_ += bytesInRun_;
      allocations_.push_back(std::move(allocation));
    } else {
      // Large contiguous run: reproduce the committed and reserved sizes so the
      // hugePageRange() the row iterator spans matches.
      const auto& srcLarge = src.largeAllocations_[i - numSmall];
      ContiguousAllocation allocation;
      pool_->allocateContiguous(
          AllocationTraits::numPages(srcLarge.size()),
          allocation,
          AllocationTraits::numPages(srcLarge.maxSize()));
      const auto range = allocation.hugePageRange().value();
      VELOX_CHECK(
          isLast || range.size() == srcRange.size(),
          "Cloned huge-page run size does not match the source run");
      destBegin = range.data();
      startOfRun_ = destBegin;
      bytesInRun_ = range.size();
      usedBytes_ += srcLarge.size();
      largeAllocations_.push_back(std::move(allocation));
    }
    ::memcpy(destBegin, srcBegin, used);
    // The last range becomes the destination's current run; its used extent is
    // the live write offset for any future allocation.
    currentOffset_ = used;
    relocations.push_back(Relocation{srcBegin, used, destBegin - srcBegin});
  }
  std::sort(
      relocations.begin(),
      relocations.end(),
      [](const Relocation& a, const Relocation& b) {
        return a.sourceBegin < b.sourceBegin;
      });
  return relocations;
}

char* AllocationPool::allocateFixed(uint64_t bytes, int32_t alignment) {
  VELOX_CHECK_GT(bytes, 0, "Cannot allocate zero bytes");
  if (freeAddressableBytes() >= bytes && alignment == 1) {
    auto* result = startOfRun_ + currentOffset_;
    maybeGrowLastAllocation(bytes);
    return result;
  }
  VELOX_CHECK_EQ(
      __builtin_popcount(alignment), 1, "Alignment can only be power of 2");

  auto numPages = AllocationTraits::numPages(bytes + alignment - 1);

  if (freeAddressableBytes() == 0) {
    newRunImpl(numPages);
  } else {
    auto alignedBytes = bytes + alignmentPadding(firstFreeInRun(), alignment);
    if (freeAddressableBytes() < alignedBytes) {
      newRunImpl(numPages);
    }
  }
  currentOffset_ += alignmentPadding(firstFreeInRun(), alignment);
  VELOX_CHECK_LE(bytes + currentOffset_, bytesInRun_);
  auto* result = startOfRun_ + currentOffset_;
  VELOX_CHECK_EQ(reinterpret_cast<uintptr_t>(result) % alignment, 0);
  maybeGrowLastAllocation(bytes);
  return result;
}

void AllocationPool::maybeGrowLastAllocation(uint64_t bytesRequested) {
  const auto updateOffset = currentOffset_ + bytesRequested;
  if (updateOffset > endOfReservedRun()) {
    VELOX_CHECK_GT(bytesInRun_, AllocationTraits::kHugePageSize);
    const auto bytesToReserve = bits::roundUp(
        updateOffset - endOfReservedRun(), AllocationTraits::kHugePageSize);
    largeAllocations_.back().grow(AllocationTraits::numPages(bytesToReserve));
    usedBytes_ += bytesToReserve;
  }
  // Only update currentOffset_ once it points to valid data.
  currentOffset_ = updateOffset;
}

void AllocationPool::newRunImpl(MachinePageCount numPages) {
  if (usedBytes_ >= hugePageThreshold_ ||
      numPages > pool_->sizeClasses().back()) {
    // At least 16 huge pages, no more than kMaxMmapBytes. The next is
    // double the previous. Because the previous is a hair under the
    // power of two because of fractional pages at ends of allocation,
    // add an extra huge page size.
    int64_t nextSize = std::min(
        kMaxMmapBytes,
        std::max<int64_t>(
            16 * AllocationTraits::kHugePageSize,
            bits::nextPowerOfTwo(
                usedBytes_ + AllocationTraits::kHugePageSize)));
    // Round 'numPages' to no of pages in huge page. Allocating this plus an
    // extra huge page guarantees that 'numPages' worth of contiguous aligned
    // huge pages will be founfd in the allocation.
    numPages = bits::roundUp(numPages, AllocationTraits::numPagesInHugePage());
    if (AllocationTraits::pageBytes(numPages) +
            AllocationTraits::kHugePageSize >
        nextSize) {
      // Extra large single request.
      nextSize = AllocationTraits::pageBytes(numPages) +
          AllocationTraits::kHugePageSize;
    }

    ContiguousAllocation largeAlloc;
    const MachinePageCount pagesToAlloc =
        AllocationTraits::numPagesInHugePage();
    pool_->allocateContiguous(
        pagesToAlloc, largeAlloc, AllocationTraits::numPages(nextSize));

    auto range = largeAlloc.hugePageRange().value();
    startOfRun_ = range.data();
    bytesInRun_ = range.size();
    largeAllocations_.emplace_back(std::move(largeAlloc));
    currentOffset_ = 0;
    usedBytes_ += AllocationTraits::pageBytes(pagesToAlloc);
    return;
  }

  Allocation allocation;
  auto roundedPages = std::max<int32_t>(kMinPages, numPages);
  pool_->allocateNonContiguous(roundedPages, allocation, roundedPages);
  VELOX_CHECK_EQ(allocation.numRuns(), 1);
  startOfRun_ = allocation.runAt(0).data<char>();
  bytesInRun_ = allocation.runAt(0).numBytes();
  currentOffset_ = 0;
  allocations_.push_back(std::move(allocation));
  usedBytes_ += bytesInRun_;
}

void AllocationPool::newRun(int64_t preferredSize) {
  newRunImpl(AllocationTraits::numPages(preferredSize));
}

} // namespace facebook::velox::memory
