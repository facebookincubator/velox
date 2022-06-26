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

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/MappedMemory.h"

namespace facebook::velox::memory {

// Denotes a number of pages of one size class, i.e. one page consists
// of a size class dependent number of consecutive machine pages.
using ClassPageCount = int32_t;

struct MmapAllocatorOptions {
  //  Capacity in bytes, default 512MB
  uint64_t capacity = 1L << 29;
};

static constexpr uint64_t kMinGrainSizeBytes = 1024 * 1024;

class Arena {
 public:
  Arena(size_t capacityBytes);
  ~Arena();

  void* allocate(uint64_t bytes);
  void free(void* address, uint64_t bytes);
  void* address() const {
    return reinterpret_cast<void*>(address_);
  }

  uint64_t byteSize() const {
    return byteSize_;
  }

  const std::map<uint64_t, uint64_t>& freeList() const {
    return freeList_;
  }

  const std::map<uint64_t, std::unordered_set<uint64_t>>& freeLookup() const {
    return freeLookup_;
  }

  uint64_t freeBytes() {
    return freeBytes_;
  }

  // translate lookup table to a string for debugging purpose only.
  std::string freeLookupStr() {
    std::string lookupStr;
    for (auto itr = freeLookup_.begin(); itr != freeLookup_.end(); itr++) {
      lookupStr.append("\n{").append(std::to_string(itr->first)).append("->[");
      for (auto itrInner = itr->second.begin(); itrInner != itr->second.end();
           itrInner++) {
        lookupStr.append(std::to_string(*itrInner)).append(", ");
      }
      lookupStr.append("]}\n");
    }
    return lookupStr;
  }

 private:
  // Rounds up size to the next power of 2 if the asked size is larger than 3
  // quarters of the next power of 2. Otherwise rounds up to the 3 quarters.
  static uint64_t roundBytes(uint64_t bytes);

  std::map<uint64_t, uint64_t>::iterator addFreeBlock(
      uint64_t addr,
      uint64_t bytes);

  void removeFromLookup(uint64_t addr, uint64_t bytes);

  void removeFreeBlock(uint64_t addr, uint64_t bytes);

  void removeFreeBlock(std::map<uint64_t, uint64_t>::iterator& itr);

  // Starting address of this arena
  uint8_t* FOLLY_NONNULL address_;

  // Total size of this arena
  const uint64_t byteSize_;

  std::atomic<uint64_t> freeBytes_;

  // A sorted list with each entry mapping from free block address to size of
  // the free block
  std::map<uint64_t, uint64_t> freeList_;

  // A sorted look up structure that stores the block size as key and a set of
  // address of that size as value.
  std::map<uint64_t, std::unordered_set<uint64_t>> freeLookup_;
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
  enum class Failure { kNone, kMadvise, kMmap };

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
      std::function<void(int64_t)> beforeAllocCB = nullptr) override {
    bool result;
    stats_.recordAlloc(numPages * kPageSize, 1, [&]() {
      result = allocateContiguousImpl(
          numPages, collateral, allocation, beforeAllocCB);
    });
    return result;
  }

  void freeContiguous(ContiguousAllocation& allocation) override {
    stats_.recordFree(
        allocation.size(), [&]() { freeContiguousImpl(allocation); });
  }

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

  // Causes 'failure' to occur in next call. This is a test-only
  // function for validating otherwise unreachable error paths.
  void injectFailure(Failure failure) {
    injectedFailure_ = failure;
  }

  MachinePageCount numExternalMapped() const {
    return numExternalMapped_;
  }

  std::string toString() const override;

  Stats stats() const override {
    auto stats = stats_;
    stats.numAdvise = numAdvisedPages_;
    return stats;
  }

 private:
  static constexpr uint64_t kAllSet = 0xffffffffffffffff;
  static constexpr int32_t kNoLastLookup = -1;
  // Number of bits in 'mappedPages_' for one bit in
  // 'mappedFreeLookup_'.
  static constexpr int32_t kPagesPerLookupBit = 512;
  
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

    int32_t findMappedFreeGroup();

    xsimd::batch<uint64_t> mappedFreeBits(int32_t index);
    void   allocateFromMappdFree(int32_t numPages, Allocation&allocation);

    
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

    // Returns an index into 'pageAllocated_' wiht free mapped pages. Call only if 'numMappedFreePages_' > 0.
    int32_t findCandidateWord();

    void markMappedFree(int32_t page) {
      bits::setBit(mappedFreeLookup_.data(), page / kPagesPerLookupBit);
    }

    // Clears the mappedFreeLokup_ bit if there are no mapped free
    // pages left in the range of 'page'.
    void markAllocated(int32_t page);
    

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

    // Last used index in 'mappedFreeLookup_' -1 if none.
    int32_t lastLookupIndex_{kNoLastLookup};
    // has a 1 bit if the corresponding 8 word range in
    // pageAllocated/pageMapped_ has at least one mapped free bit. Has
    // 1 bit for each 8 words of pageAllocated/pageMapped.
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
      Allocation* FOLLY_NULLABLE collateral,
      ContiguousAllocation& allocation,
      std::function<void(int64_t)> beforeAllocCB = nullptr);

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

  // Serializes moving capacity between size classes
  std::mutex sizeClassBalanceMutex_;

  // Number of allocated pages. Allocation succeeds if an atomic
  // increment of this by the desired amount is <= 'capacity_'.
  std::atomic<MachinePageCount> numAllocated_;

  // Number of machine pages backed by memory in the address ranges in
  // 'sizeClasses_'. It includes pages that are already freed. Hence it should
  // be larger than numAllocated_
  std::atomic<MachinePageCount> numMapped_;

  // Number of pages allocated and explicitly mmap'd by the
  // application via allocateContiguous, outside of
  // 'sizeClasses'. These pages are counted in 'numAllocated_' and
  // 'numMapped_'. Allocation requests are decided against
  // 'numAllocated_' and 'numMapped_'. This counter is informational
  // only.
  std::atomic<MachinePageCount> numExternalMapped_{0};
  MachinePageCount capacity_ = 0;

  std::vector<std::unique_ptr<SizeClass>> sizeClasses_;

  std::mutex arenaMutex_;
  std::unique_ptr<Arena> arena_;

  // Statistics. Not atomic.
  uint64_t numAllocations_ = 0;
  uint64_t numAllocatedPages_ = 0;
  uint64_t numAdvisedPages_ = 0;
  Failure injectedFailure_{Failure::kNone};
  Stats stats_;
};

} // namespace facebook::velox::memory
