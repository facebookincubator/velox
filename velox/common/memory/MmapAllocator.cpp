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

#include "velox/common/memory/MmapAllocator.h"

#include <sys/mman.h>

#include "velox/common/base/Portability.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::memory {
MmapAllocator::MmapAllocator(const Options& options)
    : kind_(MemoryAllocator::Kind::kMmap),
      useMmapArena_(options.useMmapArena),
      maxMallocBytes_(options.maxMallocBytes),
      mallocReservedBytes_(
          maxMallocBytes_ == 0
              ? 0
              : options.capacity * options.smallAllocationReservePct / 100),
      capacity_(bits::roundUp(
          AllocationTraits::numPages(options.capacity - mallocReservedBytes_),
          64 * sizeClassSizes_.back())) {
  for (const auto& size : sizeClassSizes_) {
    sizeClasses_.push_back(std::make_unique<SizeClass>(capacity_ / size, size));
  }

  if (useMmapArena_) {
    const auto arenaSizeBytes = bits::roundUp(
        AllocationTraits::pageBytes(capacity_) / options.mmapArenaCapacityRatio,
        AllocationTraits::kPageSize);
    managedArenas_ = std::make_unique<ManagedMmapArenas>(
        std::max<uint64_t>(arenaSizeBytes, MmapArena::kMinCapacityBytes));
  }
}

MmapAllocator::~MmapAllocator() {
  VELOX_CHECK(
      (numAllocated_ == 0) && (numExternalMapped_ == 0), "{}", toString());
}

bool MmapAllocator::allocateNonContiguous(
    MachinePageCount numPages,
    Allocation& out,
    ReservationCallback reservationCB,
    MachinePageCount minSizeClass) {
  VELOX_CHECK_GT(numPages, 0);

  const SizeMix mix = allocationSize(numPages, minSizeClass);

  const int64_t numFreed = freeInternal(out);
  if (numFreed != 0) {
    numAllocated_.fetch_sub(numFreed);
  }

  if (testingHasInjectedFailure(InjectedFailure::kCap)) {
    if ((numFreed != 0) && (reservationCB != nullptr)) {
      reservationCB(AllocationTraits::pageBytes(numFreed), false);
    }
    return false;
  }

  if (numAllocated_ + mix.totalPages > capacity_) {
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000)
        << "Exceeding memory allocator limit when allocate " << mix.totalPages
        << " pages with capacity of " << capacity_;
    if ((numFreed != 0) && (reservationCB != nullptr)) {
      reservationCB(AllocationTraits::pageBytes(numFreed), false);
    }
    return false;
  }
  if (numAllocated_.fetch_add(mix.totalPages) + mix.totalPages > capacity_) {
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000)
        << "Exceeded memory allocator limit when allocate " << mix.totalPages
        << " pages with capacity of " << capacity_;
    numAllocated_.fetch_sub(mix.totalPages);
    if ((numFreed != 0) && (reservationCB != nullptr)) {
      reservationCB(AllocationTraits::pageBytes(numFreed), false);
    }
    return false;
  }

  ++numAllocations_;
  numAllocatedPages_ += mix.totalPages;
  const int64_t numNeededPages = mix.totalPages - numFreed;
  if (reservationCB != nullptr) {
    try {
      reservationCB(AllocationTraits::pageBytes(numNeededPages), true);
    } catch (const std::exception& e) {
      VELOX_MEM_LOG_EVERY_MS(WARNING, 1000)
          << "Exceeded memory reservation limit when reserve " << numNeededPages
          << " new pages when allocate " << mix.totalPages << " pages";
      numAllocated_.fetch_sub(mix.totalPages);
      reservationCB(AllocationTraits::pageBytes(numFreed), false);
      std::rethrow_exception(std::current_exception());
    }
  }
  MachinePageCount newMapsNeeded = 0;
  for (int i = 0; i < mix.numSizes; ++i) {
    bool success;
    stats_.recordAllocate(
        AllocationTraits::pageBytes(sizeClassSizes_[mix.sizeIndices[i]]),
        mix.sizeCounts[i],
        [&]() {
          success = sizeClasses_[mix.sizeIndices[i]]->allocate(
              mix.sizeCounts[i], newMapsNeeded, out);
        });
    if (success && ((i > 0) || (mix.numSizes == 1)) &&
        testingHasInjectedFailure(InjectedFailure::kAllocate)) {
      // Trigger memory allocation failure in the middle of the size class
      // allocation series.
      success = false;
    }
    if (!success) {
      // This does not normally happen since any size class can accommodate
      // all the capacity. 'allocatedPages_' must be out of sync.
      VELOX_MEM_LOG(WARNING) << "Failed allocation in size class " << i
                             << " for " << mix.sizeCounts[i] << " pages";
      const auto failedPages = mix.totalPages - out.numPages();
      freeNonContiguous(out);
      numAllocated_.fetch_sub(failedPages);
      if (reservationCB != nullptr) {
        reservationCB(AllocationTraits::pageBytes(mix.totalPages), false);
      }
      return false;
    }
  }
  if (newMapsNeeded == 0) {
    return true;
  }
  if (ensureEnoughMappedPages(newMapsNeeded)) {
    markAllMapped(out);
    return true;
  }

  VELOX_MEM_LOG(WARNING) << "Could not advise away enough for " << newMapsNeeded
                         << " pages with total allocation of << "
                         << mix.totalPages << " pages";
  freeNonContiguous(out);
  if (reservationCB != nullptr) {
    reservationCB(AllocationTraits::pageBytes(mix.totalPages), false);
  }
  return false;
}

bool MmapAllocator::ensureEnoughMappedPages(int32_t newMappedNeeded) {
  std::lock_guard<std::mutex> l(sizeClassBalanceMutex_);
  if (testingHasInjectedFailure(InjectedFailure::kMadvise)) {
    return false;
  }
  const auto totalMaps =
      numMapped_.fetch_add(newMappedNeeded) + newMappedNeeded;
  if (totalMaps <= capacity_) {
    // We are not at capacity. No need to advise away.
    return true;
  }
  // We need to advise away a number of pages or we fail the alloc.
  const auto target = totalMaps - capacity_;
  const auto numAdvised = adviseAway(target);
  numAdvisedPages_ += numAdvised;
  if (numAdvised >= target) {
    numMapped_.fetch_sub(numAdvised);
    return true;
  }
  numMapped_.fetch_sub(numAdvised + newMappedNeeded);
  return false;
}

int64_t MmapAllocator::freeNonContiguous(Allocation& allocation) {
  const auto numFreed = freeInternal(allocation);
  numAllocated_.fetch_sub(numFreed);
  return AllocationTraits::pageBytes(numFreed);
}

MachinePageCount MmapAllocator::freeInternal(Allocation& allocation) {
  MachinePageCount numFreed = 0;
  if (allocation.empty()) {
    return numFreed;
  }

  for (auto i = 0; i < sizeClasses_.size(); ++i) {
    auto& sizeClass = sizeClasses_[i];
    int32_t pages = 0;
    uint64_t clocks = 0;
    {
      ClockTimer timer(clocks);
      pages = sizeClass->free(allocation);
    }
    if ((pages > 0) && FLAGS_velox_time_allocations) {
      // Increment the free time only if the allocation contained
      // pages in the class. Note that size class indices in the
      // allocator are not necessarily the same as in the stats.
      const auto sizeIndex =
          Stats::sizeIndex(AllocationTraits::pageBytes(sizeClassSizes_[i]));
      stats_.sizes[sizeIndex].freeClocks += clocks;
    }
    numFreed += pages;
  }
  allocation.clear();
  return numFreed;
}

bool MmapAllocator::allocateContiguousImpl(
    MachinePageCount numPages,
    Allocation* collateral,
    ContiguousAllocation& allocation,
    ReservationCallback reservationCB) {
  MachinePageCount numCollateralPages = 0;
  // 'collateral' and 'allocation' get freed anyway. But the counters are not
  // updated to reflect this. Rather, we add the delta that is needed on top of
  // collaterals to the allocation and mapped counters. In this way another
  // thread will not see the temporary dip in allocation, and we are sure to
  // succeed if 'collateral' and 'allocation' together cover 'numPages'. If we
  // need more space and fail to get this, then we subtract 'collateral' and
  // 'allocation' from the counters.
  //
  // Specifically, we do not subtract anything from counters with a resource
  // reservation semantic, i.e. 'numAllocated_' and 'numMapped_' except at the
  // end where the outcome of the operation is clear. Otherwise, we could not
  // have the guarantee that the operation succeeds if 'collateral' and
  // 'allocation' cover the new size, as other threads might grab the
  // transiently free pages.
  if (collateral != nullptr) {
    numCollateralPages = freeInternal(*collateral);
  }
  const auto numLargeCollateralPages = allocation.numPages();
  if (numLargeCollateralPages > 0) {
    if (useMmapArena_) {
      std::lock_guard<std::mutex> l(arenaMutex_);
      managedArenas_->free(allocation.data(), allocation.size());
    } else {
      if (::munmap(allocation.data(), allocation.size()) < 0) {
        VELOX_MEM_LOG(ERROR) << "munmap got " << folly::errnoStr(errno)
                             << " for " << allocation.toString();
      }
    }
    allocation.clear();
  }

  const auto totalCollateralPages =
      numCollateralPages + numLargeCollateralPages;
  const auto numCollateralUnmap = numLargeCollateralPages;
  const int64_t newPages = numPages - totalCollateralPages;
  if (reservationCB != nullptr) {
    try {
      reservationCB(AllocationTraits::pageBytes(newPages), true);
    } catch (const std::exception& e) {
      VELOX_MEM_LOG_EVERY_MS(WARNING, 1000)
          << "Exceeded memory reservation limit when reserve " << newPages
          << " new pages when allocate " << numPages << " pages";
      numAllocated_ -= totalCollateralPages;
      numMapped_ -= numCollateralUnmap;
      numExternalMapped_ -= numCollateralUnmap;

      // We failed to grow by 'newPages. So we record the freeing off the whole
      // collateral and the unmap of former 'allocation'.
      reservationCB(AllocationTraits::pageBytes(totalCollateralPages), false);
      throw;
    }
  }

  // Rolls back the counters on failure. 'mappedDecrement' is subtracted from
  // 'numMapped_' on top of other adjustment.
  auto rollbackAllocation = [&](int64_t mappedDecrement) {
    // The previous allocation and collateral were both freed but not counted as
    // freed.
    numAllocated_ -= numPages;
    // Incremented by numPages - numLargeCollateralPages. On failure,
    // numLargeCollateralPages are freed and numPages - numLargeCollateralPages
    // were never allocated.
    numExternalMapped_ -= numPages;
    numMapped_ -= numCollateralUnmap + mappedDecrement;

    if (reservationCB != nullptr) {
      reservationCB(AllocationTraits::pageBytes(numPages), false);
    }
  };

  numExternalMapped_ += numPages - numCollateralUnmap;
  auto numAllocated = numAllocated_.fetch_add(newPages) + newPages;
  // Check if went over the limit. But a net decrease always succeeds even if
  // ending up over the limit because some other thread might be transiently
  // over the limit.
  if (newPages > 0 &&
      (numAllocated > capacity_ ||
       testingHasInjectedFailure(InjectedFailure::kCap))) {
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000)
        << "Exceeded memory allocator limit when allocate " << newPages
        << " new pages for total allocation of " << numPages
        << " pages, the memory allocator capacity is " << capacity_;
    rollbackAllocation(0);
    return false;
  }
  // Make sure there are free backing pages for the size minus what we just
  // unmapped.
  const int64_t numToMap = numPages - numCollateralUnmap;
  if (numToMap > 0) {
    if (!ensureEnoughMappedPages(numToMap)) {
      VELOX_MEM_LOG(WARNING)
          << "Could not advise away enough for " << numToMap
          << " pages for total allocation of " << numPages << " pages";
      rollbackAllocation(0);
      return false;
    }
  } else {
    // We exchange a large mmap for a smaller one. Add negative delta.
    numMapped_ += numToMap;
  }

  void* data;
  if (testingHasInjectedFailure(InjectedFailure::kMmap)) {
    data = nullptr;
  } else {
    if (useMmapArena_) {
      std::lock_guard<std::mutex> l(arenaMutex_);
      data = managedArenas_->allocate(AllocationTraits::pageBytes(numPages));
    } else {
      data = ::mmap(
          nullptr,
          AllocationTraits::pageBytes(numPages),
          PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS,
          -1,
          0);
    }
  }
  // TODO: add handling of MAP_FAILED.
  if (data == nullptr) {
    VELOX_MEM_LOG(ERROR) << "Mmap failed with " << numPages
                         << " pages, use MmapArena "
                         << (useMmapArena_ ? "true" : "false");
    // If the mmap failed, we have unmapped former 'allocation' and the extra to
    // be mapped.
    rollbackAllocation(numToMap);
    return false;
  }
  allocation.set(data, AllocationTraits::pageBytes(numPages));
  return true;
}

void MmapAllocator::freeContiguousImpl(ContiguousAllocation& allocation) {
  if (allocation.empty()) {
    return;
  }
  if (useMmapArena_) {
    std::lock_guard<std::mutex> l(arenaMutex_);
    managedArenas_->free(allocation.data(), allocation.size());
  } else {
    if (::munmap(allocation.data(), allocation.size()) < 0) {
      VELOX_MEM_LOG(ERROR) << "munmap returned " << folly::errnoStr(errno)
                           << " for " << allocation.toString();
    }
  }
  numMapped_ -= allocation.numPages();
  numExternalMapped_ -= allocation.numPages();
  numAllocated_ -= allocation.numPages();
  allocation.clear();
}

void* MmapAllocator::allocateBytes(uint64_t bytes, uint16_t alignment) {
  alignmentCheck(bytes, alignment);

  if (useMalloc(bytes)) {
    auto* result = alignment > kMinAlignment ? ::aligned_alloc(alignment, bytes)
                                             : ::malloc(bytes);
    if (FOLLY_UNLIKELY(result == nullptr)) {
      VELOX_MEM_LOG(ERROR) << "Failed to allocateBytes " << bytes
                           << " bytes with " << alignment << " alignment";
    } else {
      numMallocBytes_.fetch_add(bytes);
    }
    return result;
  }

  if (bytes <= AllocationTraits::pageBytes(sizeClassSizes_.back())) {
    Allocation allocation;
    const auto numPages = roundUpToSizeClassSize(bytes, sizeClassSizes_);
    if (!allocateNonContiguous(numPages, allocation, nullptr, numPages)) {
      return nullptr;
    }
    auto run = allocation.runAt(0);
    VELOX_CHECK_EQ(
        1,
        allocation.numRuns(),
        "A size class allocateBytes must produce one run");
    allocation.clear();
    return run.data<char>();
  }

  ContiguousAllocation allocation;
  auto numPages = bits::roundUp(bytes, AllocationTraits::kPageSize) /
      AllocationTraits::kPageSize;
  if (!allocateContiguous(numPages, nullptr, allocation)) {
    return nullptr;
  }

  char* data = allocation.data<char>();
  allocation.clear();
  return data;
}

void MmapAllocator::freeBytes(void* p, uint64_t bytes) noexcept {
  if (useMalloc(bytes)) {
    ::free(p); // NOLINT
    numMallocBytes_.fetch_sub(bytes);
    return;
  }

  if (bytes <= AllocationTraits::pageBytes(sizeClassSizes_.back())) {
    Allocation allocation;
    auto numPages = roundUpToSizeClassSize(bytes, sizeClassSizes_);
    allocation.append(reinterpret_cast<uint8_t*>(p), numPages);
    freeNonContiguous(allocation);
    return;
  }

  ContiguousAllocation allocation;
  allocation.set(p, bytes);
  freeContiguous(allocation);
}

void MmapAllocator::markAllMapped(const Allocation& allocation) {
  for (auto& sizeClass : sizeClasses_) {
    sizeClass->setAllMapped(allocation, true);
  }
}

MachinePageCount MmapAllocator::adviseAway(MachinePageCount target) {
  MachinePageCount numAway = 0;
  for (int32_t i = sizeClasses_.size() - 1; i >= 0; --i) {
    numAway += sizeClasses_[i]->adviseAway(target - numAway);
    if (numAway >= target) {
      break;
    }
  }
  return numAway;
}

MmapAllocator::SizeClass::SizeClass(size_t capacity, MachinePageCount unitSize)
    : capacity_(capacity),
      unitSize_(unitSize),
      byteSize_(AllocationTraits::pageBytes(capacity_ * unitSize_)),
      pageBitmapSize_(capacity_ / 64),
      // Min 8 words + 1 bit for every 512 bits in 'pageAllocated_'.
      mappedFreeLookup_((capacity_ / kPagesPerLookupBit / 64) + kSimdTail),
      pageAllocated_(pageBitmapSize_ + kSimdTail),
      pageMapped_(pageBitmapSize_ + kSimdTail) {
  VELOX_CHECK_EQ(
      capacity_ % 64,
      0,
      "Sizeclass {} must have a multiple of 64 capacity",
      unitSize_);
  void* ptr = mmap(
      nullptr,
      AllocationTraits::pageBytes(capacity_ * unitSize_),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
  if (ptr == MAP_FAILED || ptr == nullptr) {
    VELOX_FAIL(
        "Could not allocate working memory "
        "mmap failed with {} for sizeClass {}",
        folly::errnoStr(errno),
        unitSize_);
  }
  address_ = reinterpret_cast<uint8_t*>(ptr);
}

MmapAllocator::SizeClass::~SizeClass() {
  munmap(address_, byteSize_);
}
ClassPageCount MmapAllocator::SizeClass::checkConsistency(
    ClassPageCount& numMapped,
    int32_t& numErrors) const {
  constexpr int32_t kWordsPerLookupBit = kPagesPerLookupBit / 64;
  int count = 0;
  int mappedCount = 0;
  int mappedFreeCount = 0;
  for (int i = 0; i < pageBitmapSize_; ++i) {
    count += __builtin_popcountll(pageAllocated_[i]);
    mappedCount += __builtin_popcountll(pageMapped_[i]);
    mappedFreeCount +=
        __builtin_popcountll(~pageAllocated_[i] & pageMapped_[i]);

    if (i % kWordsPerLookupBit == 0) {
      int32_t mappedFreeInGroup = 0;
      // There is at least kSimdTail words of zeros after the last meaningful
      // word.
      for (auto j = i;
           j < std::min<int32_t>(pageBitmapSize_, i + kWordsPerLookupBit);
           ++j) {
        mappedFreeInGroup +=
            __builtin_popcountll(pageMapped_[j] & ~pageAllocated_[j]);
      }
      if (bits::isBitSet(mappedFreeLookup_.data(), i / kWordsPerLookupBit)) {
        if (!mappedFreeInGroup) {
          VELOX_MEM_LOG(WARNING) << "Extra mapped free bit for group at " << i;
          ++numErrors;
        }
      } else if (mappedFreeInGroup) {
        ++numErrors;
        VELOX_MEM_LOG(WARNING) << "Missing lookup bit for group at " << i;
      }
    }
  }
  if (mappedFreeCount != numMappedFreePages_) {
    ++numErrors;
    VELOX_MEM_LOG(WARNING)
        << "Mismatched count of mapped free pages in size class " << unitSize_
        << ". Actual= " << mappedFreeCount
        << " vs recorded= " << numMappedFreePages_
        << ". Total mapped=" << mappedCount;
  }
  numMapped = mappedCount;
  return count;
}

std::string MmapAllocator::SizeClass::toString() const {
  std::stringstream out;
  int count = 0;
  int mappedCount = 0;
  int mappedFreeCount = 0;
  for (int i = 0; i < pageBitmapSize_; ++i) {
    count += __builtin_popcountll(pageAllocated_[i]);
    mappedCount += __builtin_popcountll(pageMapped_[i]);
    mappedFreeCount +=
        __builtin_popcountll(~pageAllocated_[i] & pageMapped_[i]);
  }
  auto mb = (AllocationTraits::pageBytes(count * unitSize_)) >> 20;
  out << "[size " << unitSize_ << ": " << count << "(" << mb << "MB) allocated "
      << mappedCount << " mapped";
  if (mappedFreeCount != numMappedFreePages_) {
    out << "Mismatched count of mapped free pages "
        << ". Actual= " << mappedFreeCount
        << " vs recorded= " << numMappedFreePages_
        << ". Total mapped=" << mappedCount;
  }
  out << "]";
  return out.str();
}

bool MmapAllocator::SizeClass::allocate(
    ClassPageCount numPages,
    MachinePageCount& numUnmapped,
    Allocation& out) {
  std::lock_guard<std::mutex> l(mutex_);
  return allocateLocked(numPages, &numUnmapped, out);
}

bool MmapAllocator::SizeClass::allocateLocked(
    const ClassPageCount numPages,
    MachinePageCount* numUnmapped,
    Allocation& out) {
  const size_t numWords = pageBitmapSize_;
  ClassPageCount considerMappedOnly = std::min(numMappedFreePages_, numPages);
  auto numPagesToAllocate = numPages;
  if (considerMappedOnly > 0) {
    const auto previousPages = out.numPages();
    allocateFromMappedFree(considerMappedOnly, out);
    const auto numAllocated = (out.numPages() - previousPages) / unitSize_;
    VELOX_CHECK_EQ(
        numAllocated,
        considerMappedOnly,
        "Allocated different number of pages");
    numMappedFreePages_ -= numAllocated;
    numPagesToAllocate -= numAllocated;
  }
  if (numPagesToAllocate == 0) {
    return true;
  }
  if (numUnmapped == nullptr) {
    return false;
  }
  uint32_t cursor = clockHand_;
  int32_t numWordsTried = 0;
  for (;;) {
    auto previousCursor = cursor;
    if (++cursor >= numWords) {
      cursor = 0;
    }
    if (++numWordsTried > numWords) {
      return false;
    }
    uint64_t bits = pageAllocated_[cursor];
    if (bits != kAllSet) {
      const auto previousToAllocate = numPagesToAllocate;
      allocateAny(cursor, numPagesToAllocate, *numUnmapped, out);
      numAllocatedUnmapped_ += previousToAllocate - numPagesToAllocate;
      if (numPagesToAllocate == 0) {
        clockHand_ = previousCursor;
        return true;
      }
    }
  }
}

namespace {
bool isAllZero(xsimd::batch<uint64_t> bits) {
  return simd::allSetBitMask<uint64_t>() ==
      simd::toBitMask(bits == xsimd::broadcast<uint64_t>(0));
}
} // namespace

uint32_t MmapAllocator::SizeClass::findMappedFreeGroup() {
  constexpr int32_t kWidth = xsimd::batch<int64_t>::size;
  int32_t index = lastLookupIndex_;
  if (index == kNoLastLookup) {
    index = 0;
  }
  auto lookupSize = mappedFreeLookup_.size() + kSimdTail;
  for (auto counter = 0; counter <= lookupSize; counter += kWidth) {
    auto candidates = xsimd::load_unaligned(mappedFreeLookup_.data() + index);
    auto bits = simd::allSetBitMask<int64_t>() ^
        simd::toBitMask(candidates == xsimd::broadcast<uint64_t>(0LL));
    if (!bits) {
      index = index + kWidth <= mappedFreeLookup_.size() - kWidth
          ? index + kWidth
          : 0;
      continue;
    }
    lastLookupIndex_ = index;
    auto wordIndex = count_trailing_zeros(bits);
    auto word = mappedFreeLookup_[index + wordIndex];
    auto bit = count_trailing_zeros(word);
    return (index + wordIndex) * 64 + bit;
  }
  ClassPageCount ignore = 0;
  checkConsistency(ignore, ignore);
  VELOX_MEM_LOG(FATAL) << "Inconsistent mapped free lookup class " << unitSize_;
}

xsimd::batch<uint64_t> MmapAllocator::SizeClass::mappedFreeBits(int32_t index) {
  return (xsimd::load_unaligned(pageAllocated_.data() + index) ^
          xsimd::broadcast<uint64_t>(~0UL)) &
      xsimd::load_unaligned(pageMapped_.data() + index);
}

void MmapAllocator::SizeClass::allocateFromMappedFree(
    int32_t numPages,
    Allocation& allocation) {
  constexpr int32_t kWidth = xsimd::batch<int64_t>::size;
  constexpr int32_t kWordsPerGroup = kPagesPerLookupBit / 64;
  int needed = numPages;
  for (;;) {
    const auto group = findMappedFreeGroup();
    const auto startWord = group * kWordsPerGroup;
    const auto endWord = startWord + kWordsPerGroup;
    bool anyFound = false;
    for (auto word = startWord; word < endWord; word += kWidth) {
      const auto bits = mappedFreeBits(word);
      uint16_t wordMask = simd::allSetBitMask<int64_t>() ^
          simd::toBitMask(bits == xsimd::broadcast<uint64_t>(0));
      if (wordMask == 0) {
        continue;
      }

      anyFound = true;
      const auto firstWord = bits::getAndClearLastSetBit(wordMask);
      const auto wordAllUsed = bits::testBits(
          reinterpret_cast<const uint64_t*>(&bits),
          firstWord * 64,
          sizeof(bits) * 8,
          true,
          [&](int32_t bit) {
            if (!needed) {
              return false;
            }
            const auto page = word * 64 + bit;
            bits::setBit(pageAllocated_.data(), page);
            allocation.append(
                address_ + AllocationTraits::pageBytes(page * unitSize_),
                unitSize_);
            --needed;
            return true;
          });

      if (needed == 0) {
        if (!wordAllUsed) {
          return;
        }
        for (word += kWidth; word < endWord; word += kWidth) {
          if (!isAllZero(mappedFreeBits(word))) {
            return;
          }
        }
        bits::setBit(mappedFreeLookup_.data(), group, false);
        return;
      }
    }

    bits::setBit(mappedFreeLookup_.data(), group, false);
    if (!anyFound) {
      VELOX_MEM_LOG(ERROR) << "Lookup bit set but no free mapped pages class "
                           << unitSize_;
      return;
    }
  }
}

MachinePageCount MmapAllocator::SizeClass::adviseAway(
    MachinePageCount numPages) {
  // Allocate as many mapped free pages as needed and advise them away.
  ClassPageCount target = bits::roundUp(numPages, unitSize_) / unitSize_;
  Allocation allocation;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (numMappedFreePages_ == 0) {
      return 0;
    }
    target = std::min(target, numMappedFreePages_);
    allocateLocked(target, nullptr, allocation);
    VELOX_CHECK_EQ(allocation.numPages(), target * unitSize_);
    numAllocatedMapped_ -= target;
    numAdvisedAway_ += target;
  }
  // Outside of 'mutex_'.
  adviseAway(allocation);
  free(allocation);
  allocation.clear();
  return unitSize_ * target;
}

bool MmapAllocator::SizeClass::isInRange(uint8_t* ptr) const {
  if (ptr >= address_ && ptr < address_ + byteSize_) {
    // See that ptr falls on a page boundary.
    if ((ptr - address_) % AllocationTraits::pageBytes(unitSize_) != 0) {
      VELOX_FAIL("Pointer is in a SizeClass but not at page boundary");
    }
    return true;
  }
  return false;
}

void MmapAllocator::SizeClass::setAllMapped(
    const Allocation& allocation,
    bool value) {
  for (int i = 0; i < allocation.numRuns(); ++i) {
    Allocation::PageRun run = allocation.runAt(i);
    if (!isInRange(run.data())) {
      continue;
    }
    std::lock_guard<std::mutex> l(mutex_);
    setMappedBits(run, value);
  }
}

void MmapAllocator::SizeClass::adviseAway(const Allocation& allocation) {
  for (int i = 0; i < allocation.numRuns(); ++i) {
    Allocation::PageRun run = allocation.runAt(i);
    if (!isInRange(run.data())) {
      continue;
    }
    if (::madvise(
            run.data(),
            AllocationTraits::pageBytes(run.numPages()),
            MADV_DONTNEED) < 0) {
      VELOX_MEM_LOG(ERROR) << "madvise got errno " << folly::errnoStr(errno);
    } else {
      std::lock_guard<std::mutex> l(mutex_);
      setMappedBits(run, false);
    }
  }
}

void MmapAllocator::SizeClass::setMappedBits(
    const Allocation::PageRun run,
    bool value) {
  const uint8_t* runAddress = run.data();
  VELOX_CHECK_EQ(
      (runAddress - address_) % (AllocationTraits::kPageSize * unitSize_),
      0,
      "Unaligned allocation in setting mapped bits");
  const auto firstBit =
      (runAddress - address_) / AllocationTraits::pageBytes(unitSize_);
  const auto numPages = run.numPages() / unitSize_;
  for (int32_t page = firstBit; page < firstBit + numPages; ++page) {
    bits::setBit(pageMapped_.data(), page, value);
  }
}

MachinePageCount MmapAllocator::SizeClass::free(Allocation& allocation) {
  int32_t firstRunInClass = -1;
  // Check if there are any runs in 'this' outside of 'mutex_'.
  for (int32_t i = 0; i < allocation.numRuns(); ++i) {
    Allocation::PageRun run = allocation.runAt(i);
    uint8_t* runAddress = run.data();
    if (isInRange(runAddress)) {
      firstRunInClass = i;
      break;
    }
  }
  if (firstRunInClass == -1) {
    return 0;
  }
  MachinePageCount numFreed = 0;
  std::lock_guard<std::mutex> l(mutex_);
  for (int i = firstRunInClass; i < allocation.numRuns(); ++i) {
    Allocation::PageRun run = allocation.runAt(i);
    uint8_t* runAddress = run.data();
    if (!isInRange(runAddress)) {
      continue;
    }
    const ClassPageCount numPages = run.numPages() / unitSize_;
    const int firstBit =
        (runAddress - address_) / (AllocationTraits::kPageSize * unitSize_);
    for (auto page = firstBit; page < firstBit + numPages; ++page) {
      if (!bits::isBitSet(pageAllocated_.data(), page)) {
        // TODO: change this to a velox failure to catch the bug.
        VELOX_MEM_LOG(ERROR)
            << "Double free: page = " << page << " sizeclass = " << unitSize_;
        continue;
      }
      if (bits::isBitSet(pageMapped_.data(), page)) {
        ++numMappedFreePages_;
        markMappedFree(page);
      }
      bits::clearBit(pageAllocated_.data(), page);
      numFreed += unitSize_;
    }
  }
  return numFreed;
}

void MmapAllocator::SizeClass::allocateAny(
    int32_t wordIndex,
    ClassPageCount& numPages,
    MachinePageCount& numUnmapped,
    Allocation& allocation) {
  uint64_t freeBits = ~pageAllocated_[wordIndex];
  const auto toAlloc = std::min(numPages, __builtin_popcountll(freeBits));
  for (int32_t i = 0; i < toAlloc; ++i) {
    const int bit = __builtin_ctzll(freeBits);
    bits::setBit(&pageAllocated_[wordIndex], bit);
    if (!(pageMapped_[wordIndex] & (1UL << bit))) {
      numUnmapped += unitSize_;
    } else {
      --numMappedFreePages_;
    }
    allocation.append(
        address_ +
            AllocationTraits::kPageSize * unitSize_ * (bit + wordIndex * 64),
        unitSize_);
    freeBits &= freeBits - 1;
  }
  numPages -= toAlloc;
}

bool MmapAllocator::checkConsistency() const {
  int count = 0;
  int mappedCount = 0;
  int32_t numErrors = 0;
  for (auto& sizeClass : sizeClasses_) {
    int mapped = 0;
    count +=
        sizeClass->checkConsistency(mapped, numErrors) * sizeClass->unitSize();
    mappedCount += mapped * sizeClass->unitSize();
  }
  if (count != numAllocated_ - numExternalMapped_) {
    ++numErrors;
    VELOX_MEM_LOG(WARNING) << "Allocated count out of sync. Actual= " << count
                           << " recorded= "
                           << numAllocated_ - numExternalMapped_;
  }
  if (mappedCount != numMapped_ - numExternalMapped_) {
    ++numErrors;
    VELOX_MEM_LOG(WARNING) << "Mapped count out of sync. Actual= "
                           << mappedCount + numExternalMapped_
                           << " recorded= " << numMapped_;
  }
  if (numErrors) {
    VELOX_MEM_LOG(ERROR) << "MmapAllocator::checkConsistency(): " << numErrors
                         << " errors";
  }
  return numErrors == 0;
}

bool MmapAllocator::useMalloc(uint64_t bytes) {
  return (maxMallocBytes_ != 0) && (bytes <= maxMallocBytes_);
}

std::string MmapAllocator::toString() const {
  std::stringstream out;
  out << "[Memory capacity " << capacity_ << " allocated " << numAllocated_
      << " mapped " << numMapped_ << " external mapped " << numExternalMapped_
      << std::endl;
  for (auto& sizeClass : sizeClasses_) {
    out << sizeClass->toString() << std::endl;
  }
  out << "]" << std::endl;
  return out.str();
}

} // namespace facebook::velox::memory
