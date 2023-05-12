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

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MmapArena.h"

namespace facebook::velox::memory {

/// Denotes a number of pages of one size class, i.e. one page consists
/// of a size class dependent number of consecutive machine pages.
using ClassPageCount = int32_t;

/// Implementation of MemoryAllocator with mmap and madvise. Each size class is
/// mmapped for the whole capacity. Each size class has a bitmap of allocated
/// entries and entries that are backed by memory. If a size class does not have
/// an entry that is free and backed by memory, we allocate an entry that is
/// free and we advise away pages from other size classes where the
/// corresponding entry is free. In this way, any combination of sizes that adds
/// up to the capacity can be allocated without fragmentation. If a size needs
/// to be allocated that does not correspond to size classes, we advise away
/// enough pages from other size classes to cover for it and then make a new
/// mmap of the requested size (ContiguousAllocation). Small contiguous memory
/// allocations less than 3/4 of smallest size class are still delegated to
/// malloc.
class MmapAllocator : public MemoryAllocator {
 public:
  struct Options {
    ///  Capacity in bytes, default 512MB
    uint64_t capacity = 1L << 29;

    /// If set true, allocations larger than largest size class size will be
    /// delegated to ManagedMmapArena. Otherwise a system mmap call will be
    /// issued for each such allocation.
    bool useMmapArena = false;

    /// Used to determine MmapArena capacity. The ratio represents system memory
    /// capacity to single MmapArena capacity ratio.
    int32_t mmapArenaCapacityRatio = 10;

    /// If not zero, reserve 'smallAllocationReservePct'% of space from
    /// 'capacity' for ad hoc small allocations. And those allocations are
    /// delegated to std::malloc.
    ///
    /// NOTE: if 'maxMallocBytes' is 0, this value will be disregarded.
    uint32_t smallAllocationReservePct = 0;

    /// The allocation threshold less than which an allocation is delegated to
    /// std::malloc().
    ///
    /// NOTE: if it is zero, then we don't delegate any allocation std::malloc,
    /// and 'smallAllocationReservePct' will be automatically set to 0
    /// disregarding any passed in value.
    int32_t maxMallocBytes = 3072;
  };

  explicit MmapAllocator(const Options& options);

  ~MmapAllocator();

  Kind kind() const override {
    return kind_;
  }

  bool allocateNonContiguous(
      MachinePageCount numPages,
      Allocation& out,
      ReservationCallback reservationCB = nullptr,
      MachinePageCount minSizeClass = 0) override;

  int64_t freeNonContiguous(Allocation& allocation) override;

  bool allocateContiguous(
      MachinePageCount numPages,
      Allocation* collateral,
      ContiguousAllocation& allocation,
      ReservationCallback reservationCB = nullptr) override {
    VELOX_CHECK_GT(numPages, 0);
    bool result;
    stats_.recordAllocate(numPages * AllocationTraits::kPageSize, 1, [&]() {
      result = allocateContiguousImpl(
          numPages, collateral, allocation, reservationCB);
    });
    return result;
  }

  void freeContiguous(ContiguousAllocation& allocation) override {
    stats_.recordFree(
        allocation.size(), [&]() { freeContiguousImpl(allocation); });
  }

  /// Allocates 'bytes' contiguous bytes and returns the pointer to the first
  /// byte. If 'bytes' is less than 'maxMallocBytes_', delegates the allocation
  /// to malloc. If the size is above that and below the largest size classes'
  /// size, allocates one element of the next size classes' size. If 'size' is
  /// greater than the largest size classes' size, calls allocateContiguous().
  /// Returns nullptr if there is no space. The amount to allocate is subject to
  /// the size limit of 'this'. This function is not virtual but calls the
  /// virtual functions allocateNonContiguous and allocateContiguous, which can
  /// track sizes and enforce caps etc. If 'alignment' is not kMinAlignment,
  /// then 'bytes' must be a multiple of 'alignment'.
  ///
  /// NOTE: 'alignment' must be power of two and in range of [kMinAlignment,
  /// kMaxAlignment].
  void* allocateBytes(uint64_t bytes, uint16_t alignment) override;

  void freeBytes(void* p, uint64_t bytes) noexcept override;

  /// Checks internal consistency of allocation data structures. Returns true if
  /// OK. May return false if there are concurrent allocations and frees during
  /// the consistency check. This is a false positive but not dangerous.
  ///
  /// Checks that the totals of mapped free and mapped and allocated pages match
  /// the data in the bitmaps in the size classes.
  bool checkConsistency() const override;

  MachinePageCount capacity() const {
    return capacity_;
  }

  int32_t maxMallocBytes() const {
    return maxMallocBytes_;
  }

  size_t mallocReservedBytes() const {
    return mallocReservedBytes_;
  }

  MachinePageCount numAllocated() const override {
    return numAllocated_;
  }

  MachinePageCount numMapped() const override {
    return numMapped_;
  }

  MachinePageCount numExternalMapped() const {
    return numExternalMapped_;
  }

  uint64_t numMallocBytes() const {
    return numMallocBytes_;
  }

  Stats stats() const override {
    auto stats = stats_;
    stats.numAdvise = numAdvisedPages_;
    return stats;
  }

  std::string toString() const override;

 private:
  static constexpr uint64_t kAllSet = 0xffffffffffffffff;

  // Represents a range of virtual addresses used for allocating entries of
  // 'unitSize_' machine pages.
  class SizeClass {
   public:
    SizeClass(size_t capacity, MachinePageCount unitSize);

    ~SizeClass();

    MachinePageCount unitSize() const {
      return unitSize_;
    }

    // Allocates 'numPages' from 'this' and appends these to *out.
    // '*numUnmapped' is incremented by the number of pages that are not backed
    // by memory.
    bool allocate(
        ClassPageCount numPages,
        MachinePageCount& numUnmapped,
        Allocation& out);

    // Frees all pages of 'allocation' that fall in this size
    // class. Erases the corresponding runs from 'allocation'.
    MachinePageCount free(Allocation& allocation);

    // Checks that allocation and map counts match the corresponding bitmaps.
    ClassPageCount checkConsistency(
        ClassPageCount& numMapped,
        int32_t& numErrors) const;

    // Advises away backing for 'numPages' worth of unallocated mapped class
    // pages. This needs to make an Allocation, for which it needs the
    // containing MmapAllocator.
    MachinePageCount adviseAway(MachinePageCount numPages);

    // Sets the mapped bits for the runs in 'allocation' to 'value' for the
    // addresses that fall in the range of 'this'
    void setAllMapped(const Allocation& allocation, bool value);

    // Sets the mapped flag for the class pages in 'run' to 'value'
    void setMappedBits(const Allocation::PageRun run, bool value);

    // True if 'ptr' is in the address range of 'this'. Checks that ptr is at a
    // size class page boundary.
    bool isInRange(uint8_t* ptr) const;

    std::string toString() const;

   private:
    static constexpr int32_t kNoLastLookup = -1;
    // Number of bits in 'mappedPages_' for one bit in 'mappedFreeLookup_'.
    static constexpr int32_t kPagesPerLookupBit =
        xsimd::batch<int64_t>::size * 128;
    // Number of extra 0 uint64's at te end of allocation bitmaps for SIMD
    // checks.
    static constexpr int32_t kSimdTail = 8;

    // Same as allocate, except that this must be called inside
    // 'mutex_'. If 'numUnmapped' is nullptr, the allocated pages must
    // all be backed by memory. Otherwise numUnmapped is updated to be
    // the count of machine pages needing backing memory to back the
    // allocation.
    bool allocateLocked(
        ClassPageCount numPages,
        MachinePageCount* numUnmapped,
        Allocation& out);

    // Returns the bit offset of the first bit of a 512 bit group in
    // 'pageAllocated_'/'pageMapped_'  that contains at least one mapped free
    // page.
    uint32_t findMappedFreeGroup();

    // Returns a word of 256 bits with a set bit for each mapped free page in
    // the range. 'index' is an index of a word in
    // 'pageAllocated_'/'pageMapped_'. In other words, accesses 4 adjacent
    // elements for performance. The arrays are appropriately padded at
    // construction.
    xsimd::batch<uint64_t> mappedFreeBits(int32_t index);

    // Adds 'numPages' mapped free pages of this size class to 'allocation'. May
    // only be called if 'mappedFreePages_' >= 'numPages'.
    void allocateFromMappedFree(int32_t numPages, Allocation& allocation);

    // Marks that 'page' is free and mapped. Called when freeing the page.
    // 'page' is a page number iin this class.
    void markMappedFree(ClassPageCount page) {
      bits::setBit(mappedFreeLookup_.data(), page / kPagesPerLookupBit);
    }

    // Advises away the machine pages of 'this' size class contained in
    // 'allocation'.
    void adviseAway(const Allocation& allocation);

    // Allocates up to 'numPages' of mapped or unmapped pages from the
    // free/mapped word at 'wordIndex'. 'numPages' is decremented by the number
    // of allocated class pages, numUnmapped is incremented by the count of
    // machine pages needed to back the unmapped part of  the new allocated
    // runs. The memorry runs are added to 'allocation'
    void allocateAny(
        int32_t wordIndex,
        ClassPageCount& numPages,
        MachinePageCount& numUnmapped,
        Allocation& allocation);

    // Number of size class pages. Number of valid bits in
    const uint64_t capacity_;

    // Size of one size class page in machine pages.
    const MachinePageCount unitSize_;

    // Size in bytes of the address range.
    const size_t byteSize_;

    // Number of meaningful words in 'pageAllocated_'/'pageMapped'. The arrays
    // themselves are padded with extra zeros for SIMD access.
    const int32_t pageBitmapSize_;

    // Serializes access to all data members and private methods.
    std::mutex mutex_;

    // Start of address range.
    uint8_t* address_;

    // Index of last modified word in 'pageAllocated_'. Sweeps over
    // the bitmaps when looking for free pages.
    int32_t clockHand_ = 0;

    // Count of free pages backed by memory.
    ClassPageCount numMappedFreePages_ = 0;

    // Last used index in 'mappedFreeLookup_'.
    int32_t lastLookupIndex_{kNoLastLookup};

    // has a set bit if the corresponding 8 word range in
    // pageAllocated_/pageMapped_ has at least one mapped free bit. Contains 1
    // bit for each 8 words of pageAllocated_/pageMapped_.
    std::vector<uint64_t> mappedFreeLookup_;

    // Has a 1 bit if the corresponding size class page is allocated.
    std::vector<uint64_t> pageAllocated_;

    // Has a 1 bit if the corresponding size class page is backed by memory.
    std::vector<uint64_t> pageMapped_;

    // Cumulative count of allocated pages for which there was backing memory.
    uint64_t numAllocatedMapped_ = 0;

    // Cumulative count of allocated pages for which there was no backing
    // memory.
    uint64_t numAllocatedUnmapped_ = 0;

    // Cumulative count of madvise for pages of 'this'
    uint64_t numAdvisedAway_ = 0;
  };

  bool allocateContiguousImpl(
      MachinePageCount numPages,
      Allocation* collateral,
      ContiguousAllocation& allocation,
      ReservationCallback reservationCB);

  void freeContiguousImpl(ContiguousAllocation& allocation);

  // Ensures that there are at least 'newMappedNeeded' pages that are
  // not backing any existing allocation. If capacity_ - numMapped_ <
  // newMappedNeeded, advises away enough pages backing freed slots in
  // the size classes to make sure that the new pages can be used
  // while staying within 'capacity"'.
  // success. Returns false if cannot advise away enough free but backed pages
  // from the size classes.
  bool ensureEnoughMappedPages(int32_t newMappedNeeded);

  // Frees 'allocation and returns the number of freed pages. Does not
  // update 'numAllocated'.
  MachinePageCount freeInternal(Allocation& allocation);

  void markAllMapped(const Allocation& allocation);

  // Finds at least  'target' unallocated pages in different size classes and
  // advises them away. Returns the number of pages advised away.
  MachinePageCount adviseAway(MachinePageCount target);

  bool useMalloc(uint64_t bytes);

  const Kind kind_;

  // If set true, allocations larger than the largest size class size will be
  // delegated to ManagedMmapArena. Otherwise, a system mmap call will be
  // issued for each such allocation.
  const bool useMmapArena_;

  // Serializes moving capacity between size classes
  std::mutex sizeClassBalanceMutex_;

  // Number of pages allocated and explicitly mmap'd by the
  // application via allocateContiguous, outside of
  // 'sizeClasses'. These pages are counted in 'numAllocated_' and
  // 'numMapped_'. Allocation requests are decided against
  // 'numAllocated_' and 'numMapped_'. This counter is informational
  // only.
  std::atomic<MachinePageCount> numExternalMapped_{0};

  // Allocations smaller than 'maxMallocBytes' will be delegated to
  // std::malloc().
  //
  // NOTE: if it is zero, then there is no delegation to std::malloc.
  const int32_t maxMallocBytes_ = 0;

  // Reserved capacity for small allocations made by MmapAllocator through
  // std::malloc().
  //
  // TODO: we don't put limit on the ad hoc small memory allocation usage for
  // now. Might consider to add later after analyzing memory metrics in prod if
  // required.
  const size_t mallocReservedBytes_ = 0;

  // Capacity for allocations made by MmapAllocator excluding the ones delegated
  // to std::malloc().
  const MachinePageCount capacity_ = 0;

  std::vector<std::unique_ptr<SizeClass>> sizeClasses_;

  // Statistics.
  std::atomic<uint64_t> numAllocations_ = 0;
  std::atomic<uint64_t> numAllocatedPages_ = 0;
  std::atomic<uint64_t> numAdvisedPages_ = 0;
  std::atomic<uint64_t> numMallocBytes_ = 0;

  // Allocations that are larger than largest size classes will be delegated to
  // ManagedMmapArenas, to avoid calling mmap on every allocation.
  std::mutex arenaMutex_;
  std::unique_ptr<ManagedMmapArenas> managedArenas_;

  Stats stats_;
};

} // namespace facebook::velox::memory
