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
#include <memory>
#include <mutex>
#include <unordered_set>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/MappedMemory.h"

namespace facebook::velox::memory {

// Denotes a number of pages of one size class, i.e. one page consists
// of a size class dependent number of consecutive machine pages.
using ClassPageCount = int32_t;

struct MmapAllocatorOptions {
  //  Capacity in bytes, default 512MB
  uint64_t capacity = 1L << 29;
};
// Implementation of MappedMemory with mmap and madvise. Each size
// class is mmapped for the whole capacity. Each size class has a
// bitmap of allocated entries and entries that are backed by
// memory. If a size class does not have an entry that is free and
// backed by memory, we allocate an entry that is free and we advise
// away pages from other size classes where the corresponding entry
// is free. In this way, any combination of sizes that adds up to
// the capacity can be allocated without fragmentation. If a size
// needs to be allocated that does not correspond to size classes,
// we advise away enough pages from other size classes to cover for
// it and then make a new mmap of the requested size
// (ContiguousAllocation).
class MmapAllocator : public MappedMemory {
 public:
  explicit MmapAllocator(const MmapAllocatorOptions& options);

  bool allocate(
      MachinePageCount numPages,
      int32_t owner,
      Allocation& out,
      std::function<void(int64_t)> beforeAllocCB = nullptr,
      MachinePageCount minSizeClass = 0) override;

  int64_t free(Allocation& allocation) override;

  bool allocateContiguous(
      MachinePageCount numPages,
      Allocation* FOLLY_NULLABLE collateral,
      ContiguousAllocation& allocation,
      std::function<void(int64_t)> beforeAllocCB = nullptr) override;

  void freeContiguous(ContiguousAllocation& allocation) override;

  // Checks internal consistency of allocation data
  // structures. Returns true if OK. May return false if there are
  // concurrent alocations and frees during the consistency check. This
  // is a false positive but not dangerous.
  bool checkConsistency() const override;

  MachinePageCount capacity() const {
    return capacity_;
  }

  MachinePageCount numAllocated() const override {
    return numAllocated_;
  }

  MachinePageCount numMapped() const override {
    return numMapped_;
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

    // Allocates 'numPages' from 'this' and appends these to
    // *out. '*numUnmapped' is incremented by the number of pages that
    // are not backed by memory.
    bool allocate(
        ClassPageCount numPages,
        int32_t owner,
        MachinePageCount& numUnmapped,
        MappedMemory::Allocation& out);

    // Frees all pages of 'allocation' that fall in this size
    // class. Erases the corresponding runs from 'allocation'.
    MachinePageCount free(Allocation& allocation);

    // Checks that allocation and map counts match the corresponding bitmaps.
    ClassPageCount checkConsistency(ClassPageCount& numMapped) const;

    // Advises away backing for 'numPages' worth of unallocated mapped class
    // pages. This needs to make an Allocation, for which it needs the
    // containing MmapAllocator.
    MachinePageCount adviseAway(
        MachinePageCount numPages,
        MmapAllocator* FOLLY_NONNULL allocator);

    // Sets the mapped bits for the runs in 'allocation' to 'value' for the
    // addresses that fall in the range of 'this'
    void setAllMapped(const Allocation& allocation, bool value);

    // Sets the mapped flag for the class pages in 'run' to 'value'
    void setMappedBits(const MappedMemory::PageRun run, bool value);

    // True if 'ptr' is in the address range of 'this'. Checks that ptr is at a
    // size class page boundary.
    bool isInRange(uint8_t* FOLLY_NONNULL ptr) const;

    std::string toString() const;

   private:
    // Same as allocate, except that this must be called inside
    // 'mutex_'. If 'numUnmapped' is nullptr, the allocated pages must
    // all be backed by memory. Otherwise numUnmapped is updated to be
    // the count of machine pages needeing backing memory to back the
    // allocation.
    bool allocateLocked(
        ClassPageCount numPages,
        int32_t owner,
        MachinePageCount* FOLLY_NULLABLE numUnmapped,
        MappedMemory::Allocation& out);

    // Advises away the machine pages of 'this' size class contained in
    // 'allocation'.
    void adviseAway(const Allocation& allocation);

    // Allocates up to 'numPages' of mapped pages from the free/mapped word at
    // 'wordIndex'. 'candidates' has a bit set for free and mapped pages. The
    // memory ranges are added to 'allocation'. 'numPages' is decremented by the
    // count of allocated class pages.
    void allocateMapped(
        int32_t wordIndex,
        uint64_t candidates,
        ClassPageCount& numPages,
        MappedMemory::Allocation& allocation);

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

    // Serializes access to all data members and private methods.
    std::mutex mutex_;

    // Number of size class pages. Number of valid bits in
    const uint64_t capacity_;

    // Size of one size class page in machine pages.
    const MachinePageCount unitSize_;

    // Start of address range.
    uint8_t* FOLLY_NONNULL address_;

    // Size in bytes of the address range.
    const size_t byteSize_;

    // Index of last modified word in 'pageAllocated_'. Sweeps over
    // the bitmaps when looking for free pages.
    int32_t clockHand_ = 0;

    // Count of free pages backed by memory.
    ClassPageCount numMappedFreePages_ = 0;

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

  // Marks all the pages backing 'out' to be mapped if one can safely
  // write up to 'numMappedNeeded' pages that have no backing
  // memory. Returns true on success. Returns false if enough backed
  // but unused pages from other size classes cannot be advised
  // away. On failure, also frees 'out'.

  bool ensureEnoughMappedPages(int32_t newMappedNeeded, Allocation& out);

  // Frees 'allocation and returns the number of freed pages. Does not
  // update 'numAllocated'.
  MachinePageCount freeInternal(Allocation& allocation);

  void markAllMapped(const Allocation& allocation);

  // Finds at least  'target' unallocated pages in different size classes and
  // advises them away. Returns the number of pages advised away.
  MachinePageCount adviseAway(MachinePageCount target);

  // Serializes moving capacity between size classes
  std::mutex sizeClassBalanceMutex_;

  // Number of allocated pages. Allocation succeeds if an atomic
  // increment of this by the desired amount is <= 'capacity_'.
  std::atomic<MachinePageCount> numAllocated_;

  //  Number of machine pages backed by memory in the
  // address ranges in 'sizeClasses_'
  std::atomic<MachinePageCount> numMapped_;

  // Number of pages allocated and explicitly mmap'd by the
  // application, outside of 'sizeClasses'. These count towards 'numAllocated_'
  // but not towards 'numMapped_'.
  std::atomic<MachinePageCount> numExternalMapped_{0};
  MachinePageCount capacity_ = 0;

  std::vector<std::unique_ptr<SizeClass>> sizeClasses_;

  // Statistics. Not atomic.
  uint64_t numAllocations_ = 0;
  uint64_t numAllocatedPages_ = 0;
  uint64_t numAdvisedPages_ = 0;
};

} // namespace facebook::velox::memory
