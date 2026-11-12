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

#include <thread>

#include "velox/common/base/ConcurrentCounter.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryAllocator.h"

DECLARE_bool(velox_memory_leak_check_enabled);

namespace facebook::velox::memory {
/// The implementation of MemoryAllocator using malloc.
class MallocAllocator : public MemoryAllocator {
 public:
  explicit MallocAllocator(const Options& options);

  ~MallocAllocator() override;

  void registerCache(const std::shared_ptr<Cache>& cache) override {
    VELOX_CHECK_NULL(cache_);
    VELOX_CHECK_NOT_NULL(cache);
    VELOX_CHECK(cache->allocator() == this);
    cache_ = cache;
  }

  Cache* cache() const override {
    return cache_.get();
  }

  Kind kind() const override {
    return kind_;
  }

  size_t capacity() const override {
    return capacity_;
  }

  void freeContiguous(ContiguousAllocation& allocation) override;

  int64_t freeNonContiguous(Allocation& allocation) override;

  bool growContiguousWithoutRetry(
      MachinePageCount increment,
      ContiguousAllocation& allocation) override;

  void freeBytes(void* p, uint64_t bytes) noexcept override;

  MachinePageCount unmap(MachinePageCount targetPages) override {
    // NOTE: MallocAllocator doesn't support unmap as it delegates all the
    // memory allocations to std::malloc.
    return 0;
  }

  size_t totalUsedBytes() const override {
    return allocatedBytes_ - static_cast<int64_t>(readReservations());
  }

  MachinePageCount numAllocated() const override {
    return numAllocated_;
  }

  MachinePageCount numMapped() const override {
    return numMapped_;
  }

  MachinePageCount numExternalMapped() const override {
    return numExternalMapped_;
  }

  bool checkConsistency() const override;

  std::string toString() const override;

 private:
  bool allocateNonContiguousWithoutRetry(
      const SizeMix& sizeMix,
      Allocation& out) override;

  bool allocateContiguousWithoutRetry(
      MachinePageCount numPages,
      Allocation* collateral,
      ContiguousAllocation& allocation,
      MachinePageCount maxPages = 0) override;

  bool allocateContiguousImpl(
      MachinePageCount numPages,
      Allocation* collateral,
      ContiguousAllocation& allocation,
      MachinePageCount maxPages);

  // Allocates 'maxBytes' of contiguous memory using malloc or mmap depending
  // on 'mallocContiguousEnabled_'. Returns the allocated pointer, or nullptr
  // on failure.
  void* dispatchAllocateContiguous(size_t maxBytes);

  // Frees contiguous memory previously allocated by
  // dispatchAllocateContiguous.
  void dispatchFreeContiguous(ContiguousAllocation& allocation);

  void freeContiguousImpl(ContiguousAllocation& allocation);

  void* allocateBytesWithoutRetry(uint64_t bytes, uint16_t alignment) override;

  void* allocateZeroFilledWithoutRetry(uint64_t bytes) override;

  /// Attempts in-place reallocation via ::realloc(). Returns nullptr when
  /// ::realloc() cannot satisfy the request, in which case the caller falls
  /// back to allocate + memcpy + free. Preconditions for using ::realloc():
  /// 1. p must be non-null. ::realloc(nullptr, n) would behave like malloc,
  ///    but we route nullptr through allocateBytes so the caller picks up
  ///    cache eviction on the fallback.
  /// 2. The requested alignment must be at most kMinAlignment. ::realloc()
  ///    guarantees only kMinAlignment on the returned pointer (and may move
  ///    the data to a new address), so larger alignments cannot be honored.
  /// 3. p must already be aligned to the requested alignment. MemoryAllocator
  ///    is a standalone module that may be used outside MemoryPool, so we
  ///    cannot rely on the caller to pass an alignment matching the original
  ///    allocation.
  void* reallocateBytesWithoutRetry(
      void* p,
      uint64_t oldSize,
      uint64_t newSize,
      uint16_t alignment) override;

  // Increments current usage and check current 'allocatedBytes_' counter to
  // make sure current usage does not go above 'capacity_'. If it goes above
  // 'capacity_', the increment will not be applied. Returns true if within
  // capacity, false otherwise.
  //
  // NOTE: This method should always be called BEFORE the actual allocation.
  inline bool incrementUsage(int64_t bytes) {
    if (bytes < reservationByteLimit_) {
      return incrementUsageWithReservation(bytes);
    }
    return incrementUsageWithoutReservation(bytes);
  }

  // Increments the memory usage in the local sharded counter from
  // 'reservations_' for memory allocation with size < 'reservationByteLimit_'
  // without updating the global 'allocatedBytes_' counter. If there is not
  // enough reserved bytes in local sharded counter, then 'reserveFunc_' is
  // called to reserve 'reservationByteLimit_' bytes from the global counter at
  // a time.
  inline bool incrementUsageWithReservation(uint32_t bytes) {
    return reservations_.update(bytes, reserveFunc_);
  }

  inline bool incrementUsageWithReservationFunc(
      std::atomic<uint64_t>& counter,
      uint64_t increment) {
    VELOX_CHECK_LT(increment, reservationByteLimit_);
    while (true) {
      uint64_t current = loadReservationCounter(counter);
      if (current > increment) {
        if (counter.compare_exchange_weak(
                current, current - increment, std::memory_order_relaxed)) {
          return true;
        }
        continue;
      }
      // Hide the depleted shard until the matching global reservation has
      // been reflected locally, so another allocation cannot miss the credit.
      if (!counter.compare_exchange_weak(
              current,
              current | kReservationTransferInProgress,
              std::memory_order_relaxed)) {
        continue;
      }
      if (!incrementUsageWithoutReservation(reservationByteLimit_)) {
        counter.store(current, std::memory_order_release);
        return false;
      }
      const uint64_t desired = current + reservationByteLimit_ - increment;
      VELOX_CHECK_GT(desired, 0);
      counter.store(desired, std::memory_order_release);
      return true;
    }
  }

  // Increments the memory usage from the global 'allocatedBytes_' counter
  // directly.
  inline bool incrementUsageWithoutReservation(int64_t bytes) {
    VELOX_CHECK_GE(bytes, reservationByteLimit_);
    const auto originalBytes = allocatedBytes_.fetch_add(bytes);
    // We don't do the check when capacity_ is 0, meaning unlimited capacity.
    if (capacity_ != 0 && originalBytes + bytes > capacity_) {
      allocatedBytes_.fetch_sub(bytes);
      return false;
    }
    return true;
  }

  // Decrements current usage and check current 'allocatedBytes_' counter to
  // make sure current usage does not go below 0. Throws if usage goes below 0.
  //
  // NOTE: This method should always be called AFTER actual free.
  inline void decrementUsage(int64_t bytes) {
    if (bytes < reservationByteLimit_) {
      decrementUsageWithReservation(bytes);
      return;
    }
    decrementUsageWithoutReservation(bytes);
  }

  // Decrements the memory usage in the local sharded counter from
  // 'reservations_' for memory free with size < 'reservationByteLimit_'
  // without updating the global 'allocatedBytes_' counter. If there is more
  // than 2 * 'reservationByteLimit_' free reserved bytes in local sharded
  // counter, then 'releaseFunc_' is called to release 'reservationByteLimit_'
  // bytes back to the global counter.
  inline void decrementUsageWithReservation(int64_t bytes) {
    reservations_.update(bytes, releaseFunc_);
  }

  inline void decrementUsageWithReservationFunc(
      std::atomic<uint64_t>& counter,
      uint64_t decrement) {
    VELOX_CHECK_LT(decrement, reservationByteLimit_);
    uint64_t desired;
    while (true) {
      uint64_t current = loadReservationCounter(counter);
      const uint64_t combined = current + decrement;
      if (combined >= uint64_t{2} * reservationByteLimit_) {
        desired = combined - reservationByteLimit_;
        if (tryReleaseReservation(counter, current, desired)) {
          break;
        }
      } else {
        desired = combined;
        if (counter.compare_exchange_weak(
                current, desired, std::memory_order_relaxed)) {
          break;
        }
      }
    }
    VELOX_CHECK_LT(desired, uint64_t{2} * reservationByteLimit_);
  }

  static constexpr uint64_t kReservationTransferInProgress = uint64_t{1} << 63;

  static uint64_t loadReservationCounter(const std::atomic<uint64_t>& counter) {
    while (true) {
      const auto current = counter.load(std::memory_order_acquire);
      if ((current & kReservationTransferInProgress) == 0) {
        return current;
      }
      // Only a shard-to-global transfer sets the marker, and it spans one
      // global atomic update. Waiting preserves the old mutex ordering.
      std::this_thread::yield();
    }
  }

  uint64_t readReservations() const {
    return reservations_.read([](const std::atomic<uint64_t>& counter) {
      return loadReservationCounter(counter);
    });
  }

  bool tryReleaseReservation(
      std::atomic<uint64_t>& counter,
      uint64_t& current,
      uint64_t desired) {
    // Keep the new shard value hidden until its matching global release is
    // visible, so another allocation cannot miss both sources of capacity.
    if (!counter.compare_exchange_weak(
            current,
            desired | kReservationTransferInProgress,
            std::memory_order_relaxed)) {
      return false;
    }
    try {
      decrementUsageWithoutReservation(reservationByteLimit_);
    } catch (...) {
      counter.store(desired + reservationByteLimit_, std::memory_order_release);
      throw;
    }
    counter.store(desired, std::memory_order_release);
    return true;
  }

  // Decrements the memory usage from the global 'allocatedBytes_' counter
  // directly.
  inline void decrementUsageWithoutReservation(int64_t bytes) {
    const auto originalBytes = allocatedBytes_.fetch_sub(bytes);
    if (originalBytes - bytes < 0) {
      // In case of inconsistency while freeing memory, do not revert in this
      // case because free is guaranteed to happen.
      VELOX_MEM_ALLOC_ERROR(
          fmt::format(
              "Trying to free {} bytes, which is larger than current allocated "
              "bytes {}",
              bytes,
              originalBytes))
    }
  }

  const Kind kind_;

  // If true, use malloc for contiguous allocations instead of mmap/munmap.
  const bool mallocContiguousEnabled_;

  // Capacity in bytes. Total allocation byte is not allowed to exceed this
  // value.
  const size_t capacity_;
  const uint32_t reservationByteLimit_;

  const ConcurrentCounter<uint64_t>::UpdateFn reserveFunc_;
  const ConcurrentCounter<uint64_t>::UpdateFn releaseFunc_;

  ConcurrentCounter<uint64_t> reservations_;

  // Current total allocated bytes by this 'MallocAllocator'.
  std::atomic<int64_t> allocatedBytes_{0};

  std::shared_ptr<Cache> cache_;
};
} // namespace facebook::velox::memory
