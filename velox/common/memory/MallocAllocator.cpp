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

#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"

#include <sys/mman.h>

namespace facebook::velox::memory {
MallocAllocator::MallocAllocator(size_t capacity)
    : kind_(MemoryAllocator::Kind::kMalloc), capacity_(capacity) {}

bool MallocAllocator::allocateNonContiguousWithoutRetry(
    MachinePageCount numPages,
    Allocation& out,
    ReservationCallback reservationCB,
    MachinePageCount minSizeClass) {
  const uint64_t freedBytes = freeNonContiguous(out);
  if (numPages == 0) {
    if (freedBytes != 0 && reservationCB != nullptr) {
      reservationCB(freedBytes, false);
    }
    return true;
  }
  const SizeMix mix = allocationSize(numPages, minSizeClass);
  const auto totalBytes = AllocationTraits::pageBytes(mix.totalPages);
  if (!incrementUsage(totalBytes)) {
    if (freedBytes != 0 && reservationCB != nullptr) {
      reservationCB(freedBytes, false);
    }
    return false;
  }

  uint64_t bytesToAllocate = 0;
  if (reservationCB != nullptr) {
    bytesToAllocate = AllocationTraits::pageBytes(mix.totalPages) - freedBytes;
    try {
      reservationCB(bytesToAllocate, true);
    } catch (std::exception& e) {
      VELOX_MEM_LOG(WARNING)
          << "Failed to reserve " << succinctBytes(bytesToAllocate)
          << " for non-contiguous allocation of " << numPages
          << " pages, then release " << succinctBytes(freedBytes)
          << " from the old allocation";
      // If the new memory reservation fails, we need to release the memory
      // reservation of the freed memory of previously allocation.
      reservationCB(freedBytes, false);
      decrementUsage(totalBytes);
      std::rethrow_exception(std::current_exception());
    }
  }

  std::vector<void*> pages;
  pages.reserve(mix.numSizes);
  for (int32_t i = 0; i < mix.numSizes; ++i) {
    // Trigger allocation failure by breaking out the loop.
    if (testingHasInjectedFailure(InjectedFailure::kAllocate)) {
      break;
    }
    MachinePageCount numSizeClassPages =
        mix.sizeCounts[i] * sizeClassSizes_[mix.sizeIndices[i]];
    void* ptr;
    stats_.recordAllocate(
        AllocationTraits::pageBytes(sizeClassSizes_[mix.sizeIndices[i]]),
        mix.sizeCounts[i],
        [&]() {
          ptr = ::malloc(
              AllocationTraits::pageBytes(numSizeClassPages)); // NOLINT
        });
    if (ptr == nullptr) {
      // Failed to allocate memory from memory.
      break;
    }
    pages.emplace_back(ptr);
    out.append(reinterpret_cast<uint8_t*>(ptr), numSizeClassPages); // NOLINT
  }

  if (pages.size() != mix.numSizes) {
    // Failed to allocate memory using malloc. Free any malloced pages and
    // return false.
    for (auto ptr : pages) {
      ::free(ptr);
    }
    out.clear();
    if (reservationCB != nullptr) {
      VELOX_MEM_LOG(WARNING)
          << "Failed to allocate memory for non-contiguous allocation of "
          << numPages << " pages, then release "
          << succinctBytes(bytesToAllocate + freedBytes)
          << " of memory reservation including the old allocation";
      reservationCB(bytesToAllocate + freedBytes, false);
    }
    decrementUsage(totalBytes);
    return false;
  }

  {
    std::lock_guard<std::mutex> l(mallocsMutex_);
    mallocs_.insert(pages.begin(), pages.end());
  }

  // Successfully allocated all pages.
  numAllocated_.fetch_add(mix.totalPages);
  return true;
}

bool MallocAllocator::allocateContiguousWithoutRetry(
    MachinePageCount numPages,
    Allocation* collateral,
    ContiguousAllocation& allocation,
    ReservationCallback reservationCB,
    MachinePageCount maxPages) {
  bool result;
  stats_.recordAllocate(AllocationTraits::pageBytes(numPages), 1, [&]() {
    result = allocateContiguousImpl(
        numPages, collateral, allocation, reservationCB, maxPages);
  });
  return result;
}

bool MallocAllocator::allocateContiguousImpl(
    MachinePageCount numPages,
    Allocation* collateral,
    ContiguousAllocation& allocation,
    ReservationCallback reservationCB,
    MachinePageCount maxPages) {
  if (maxPages == 0) {
    maxPages = numPages;
  } else {
    VELOX_CHECK_LE(numPages, maxPages);
  }
  MachinePageCount numCollateralPages = 0;
  if (collateral != nullptr) {
    numCollateralPages =
        freeNonContiguous(*collateral) / AllocationTraits::kPageSize;
  }
  auto numContiguousCollateralPages = allocation.numPages();
  if (numContiguousCollateralPages > 0) {
    useHugePages(allocation, false);
    if (::munmap(allocation.data(), allocation.maxSize()) < 0) {
      VELOX_MEM_LOG(ERROR) << "munmap got " << folly::errnoStr(errno) << "for "
                           << allocation.data() << ", " << allocation.size();
    }
    numMapped_.fetch_sub(numContiguousCollateralPages);
    numAllocated_.fetch_sub(numContiguousCollateralPages);
    decrementUsage(AllocationTraits::pageBytes(numContiguousCollateralPages));
    allocation.clear();
  }
  const auto totalCollateralPages =
      numCollateralPages + numContiguousCollateralPages;
  const auto totalCollateralBytes =
      AllocationTraits::pageBytes(totalCollateralPages);
  if (numPages == 0) {
    if (totalCollateralBytes != 0 && reservationCB != nullptr) {
      reservationCB(totalCollateralBytes, false);
    }
    return true;
  }

  const auto totalBytes = AllocationTraits::pageBytes(numPages);
  if (!incrementUsage(totalBytes)) {
    if (totalCollateralBytes != 0 && reservationCB != nullptr) {
      reservationCB(totalCollateralBytes, false);
    }
    return false;
  }
  const int64_t numNeededPages = numPages - totalCollateralPages;
  if (reservationCB != nullptr) {
    try {
      reservationCB(AllocationTraits::pageBytes(numNeededPages), true);
    } catch (std::exception& e) {
      // If the new memory reservation fails, we need to release the memory
      // reservation of the freed contiguous and non-contiguous memory.
      VELOX_MEM_LOG(WARNING)
          << "Failed to reserve " << AllocationTraits::pageBytes(numNeededPages)
          << " bytes for contiguous allocation of " << numPages
          << " pages, then release " << succinctBytes(totalCollateralBytes)
          << " from the old allocations";
      reservationCB(totalCollateralBytes, false);
      decrementUsage(totalBytes);
      std::rethrow_exception(std::current_exception());
    }
  }
  numAllocated_.fetch_add(numPages);
  numMapped_.fetch_add(numPages);
  void* data = ::mmap(
      nullptr,
      AllocationTraits::pageBytes(maxPages),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  // TODO: add handling of MAP_FAILED.
  allocation.set(
      data,
      AllocationTraits::pageBytes(numPages),
      AllocationTraits::pageBytes(maxPages));
  useHugePages(allocation, true);
  return true;
}

int64_t MallocAllocator::freeNonContiguous(Allocation& allocation) {
  if (allocation.empty()) {
    return 0;
  }
  MachinePageCount numFreed = 0;
  for (int32_t i = 0; i < allocation.numRuns(); ++i) {
    Allocation::PageRun run = allocation.runAt(i);
    numFreed += run.numPages();
    void* ptr = run.data();
    {
      std::lock_guard<std::mutex> l(mallocsMutex_);
      const auto ret = mallocs_.erase(ptr);
      VELOX_CHECK_EQ(ret, 1, "Bad free page pointer: {}", ptr);
    }
    stats_.recordFree(
        std::min<int64_t>(
            AllocationTraits::pageBytes(sizeClassSizes_.back()),
            AllocationTraits::pageBytes(run.numPages())),
        [&]() {
          ::free(ptr); // NOLINT
        });
  }

  const auto freedBytes = AllocationTraits::pageBytes(numFreed);
  decrementUsage(freedBytes);
  numAllocated_.fetch_sub(numFreed);
  allocation.clear();
  return freedBytes;
}

void MallocAllocator::freeContiguous(ContiguousAllocation& allocation) {
  stats_.recordFree(
      allocation.size(), [&]() { freeContiguousImpl(allocation); });
}

void MallocAllocator::freeContiguousImpl(ContiguousAllocation& allocation) {
  if (allocation.empty()) {
    return;
  }
  useHugePages(allocation, false);
  const auto bytes = allocation.size();
  const auto numPages = allocation.numPages();
  if (::munmap(allocation.data(), allocation.maxSize()) < 0) {
    VELOX_MEM_LOG(ERROR) << "Error for munmap(" << allocation.data() << ", "
                         << succinctBytes(bytes) << "): '"
                         << folly::errnoStr(errno) << "'";
  }
  numMapped_.fetch_sub(numPages);
  numAllocated_.fetch_sub(numPages);
  decrementUsage(bytes);
  allocation.clear();
}

bool MallocAllocator::growContiguousWithoutRetry(
    MachinePageCount increment,
    ContiguousAllocation& allocation,
    ReservationCallback reservationCB) {
  VELOX_CHECK_LE(
      allocation.size() + increment * AllocationTraits::kPageSize,
      allocation.maxSize());
  if (reservationCB != nullptr) {
    // May throw. If does, there is nothing to revert.
    reservationCB(AllocationTraits::pageBytes(increment), true);
  }
  if (!incrementUsage(AllocationTraits::pageBytes(increment))) {
    if (reservationCB != nullptr) {
      reservationCB(AllocationTraits::pageBytes(increment), false);
    }
    return false;
  }
  numAllocated_ += increment;
  numMapped_ += increment;
  allocation.set(
      allocation.data(),
      allocation.size() + AllocationTraits::kPageSize * increment,
      allocation.maxSize());
  return true;
}

void* MallocAllocator::allocateBytesWithoutRetry(
    uint64_t bytes,
    uint16_t alignment) {
  if (!incrementUsage(bytes)) {
    return nullptr;
  }
  if (!isAlignmentValid(bytes, alignment)) {
    decrementUsage(bytes);
    VELOX_FAIL(
        "Alignment check failed, allocateBytes {}, alignmentBytes {}",
        bytes,
        alignment);
  }
  void* result = (alignment > kMinAlignment) ? ::aligned_alloc(alignment, bytes)
                                             : ::malloc(bytes);
  if (FOLLY_UNLIKELY(result == nullptr)) {
    VELOX_MEM_LOG(ERROR) << "Failed to allocateBytes " << succinctBytes(bytes)
                         << " with " << alignment << " alignment";
  }
  return result;
}

void* MallocAllocator::allocateZeroFilledWithoutRetry(uint64_t bytes) {
  if (!incrementUsage(bytes)) {
    return nullptr;
  }
  void* result = std::calloc(1, bytes);
  if (FOLLY_UNLIKELY(result == nullptr)) {
    VELOX_MEM_LOG(ERROR) << "Failed to allocateZeroFilled "
                         << succinctBytes(bytes);
  }
  return result;
}

void MallocAllocator::freeBytes(void* p, uint64_t bytes) noexcept {
  ::free(p); // NOLINT
  decrementUsage(bytes);
}

bool MallocAllocator::checkConsistency() const {
  const auto allocatedBytes = allocatedBytes_.load();
  return allocatedBytes >= 0 && allocatedBytes <= capacity_;
}

std::string MallocAllocator::toString() const {
  std::stringstream out;
  out << "Memory Allocator[" << kindString(kind_) << " capacity "
      << ((capacity_ == kMaxMemory) ? "UNLIMITED" : succinctBytes(capacity_))
      << " allocated bytes " << allocatedBytes_ << " allocated pages "
      << numAllocated_ << " mapped pages " << numMapped_ << "]";
  return out.str();
}
} // namespace facebook::velox::memory
