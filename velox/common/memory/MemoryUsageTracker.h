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
#include <memory>
#include <optional>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::memory {
constexpr std::string_view MEM_CAP_EXCEEDED_ERROR_FORMAT =
    "Exceeded memory cap of {} when requesting {}";
constexpr int64_t kMaxMemory = std::numeric_limits<int64_t>::max();

#define VELOX_MEM_CAP_EXCEEDED(errorMessage)                        \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kMemCapExceeded.c_str(),       \
      /* isRetriable */ true,                                       \
      "{}",                                                         \
      errorMessage);

/// Keeps track of currently outstanding and peak outstanding memory and
/// cumulative allocation volume for a MemoryPool or MappedMemory allocator.
/// This is supplied at construction when creating a child MemoryPool or
/// MappedMemory. These trackers form a tree. The tracking information is
/// aggregated from leaf to root. Each level in the tree may specify a maximum
/// allocation. Updating the allocated amount past the maximum will throw. This
/// is why the update should precede the actual allocation. A tracker can
/// specify a reservation. Making a reservation means that no memory is
/// allocated but the allocated size is updated so as to make sure that at least
/// the reserved amount can be allocated. Allocations that fit within the
/// reserved are not counted as new because the counting was done when making
/// the reservation.  Allocating past the reservation is possible and will
/// increase the reservation by a size-dependent quantum. This may fail if the
/// new size in some parent exceeds a cap. Freeing data within the reservation
/// drops the usage but not the reservation. Automatically updated reservations
/// are freed when enough memory is freed. Explicitly made reservations must be
/// freed with release(). Parents are updated only at reservation time to avoid
/// cache coherence traffic that arises when every thread constantly updates the
/// same top level allocation counter.
class MemoryUsageTracker
    : public std::enable_shared_from_this<MemoryUsageTracker> {
 public:
  /// The memory usage tracker's execution stats.
  struct Stats {
    uint64_t peakBytes{0};
    /// The accumulated reserved memory bytes.
    uint64_t cumulativeBytes{0};
    uint64_t numAllocs{0};
    uint64_t numFrees{0};
    uint64_t numReserves{0};
    uint64_t numReleases{0};
    /// The number of memory reservation collisions caused by concurrent memory
    /// requests.
    uint64_t numCollisions{0};
    /// The number of created child memory trackers.
    uint64_t numChildren{0};

    bool operator==(const Stats& rhs) const;

    std::string toString() const;
  };

  /// Function to increase a MemoryUsageTracker's limits. This is called when an
  /// allocation would exceed the tracker's size limit. The usage in 'tracker'
  /// at the time of call is as if the allocation had succeeded.
  /// If this returns true, this must set the limits to be >= the usage in
  /// 'tracker'. If this returns false, this should not modify 'tracker'. The
  /// caller will revert the allocation that invoked this and signal an error.
  ///
  /// This may be called on one tracker from several threads. This is
  /// responsible for serializing these. When this is called, the 'tracker's
  /// 'mutex_' must not be held by the caller. A GrowCallback must not throw.
  using GrowCallback =
      std::function<bool(int64_t size, MemoryUsageTracker& tracker)>;

  /// This function will be used when a MEM_CAP_EXCEEDED error is thrown during
  /// a call to increment reservation. It returns a string that will be appended
  /// to the error's message and should ideally be used to add additional
  /// details to it. This may be called to add details when the error is thrown
  /// or when encountered during a recursive call to increment reservation from
  /// a parent tracker.
  using MakeMemoryCapExceededMessage =
      std::function<std::string(MemoryUsageTracker& tracker)>;

  /// Create default usage tracker which is a 'root' tracker.
  static std::shared_ptr<MemoryUsageTracker> create(
      int64_t maxMemory = kMaxMemory) {
    return create(nullptr, maxMemory);
  }

  ~MemoryUsageTracker();

  /// Increments the reservation for 'this' so that we can allocate at least
  /// 'size' bytes on top of the current allocation. This is used when a memory
  /// user needs to allocate more memory and needs a guarantee of at least
  /// 'size' being available. If less memory ends up being needed, the unused
  /// reservation should be released with release(). If the new reserved amount
  /// exceeds the usage limit, an exception will be thrown.
  void reserve(uint64_t size);

  /// Checks if it is likely that the reservation on 'this' can be incremented
  /// by 'increment'. Returns false if this seems unlikely. Otherwise attempts
  /// the reservation increment and returns true if succeeded.
  bool maybeReserve(uint64_t increment);

  /// If a minimum reservation has been set with reserve(), resets the minimum
  /// reservation. If the current usage is below the minimum reservation,
  /// decreases reservation and usage down to the rounded actual usage.
  void release();

  /// Increments outstanding memory by 'size', which is positive for allocation
  /// and negative for free. If there is no reservation or the new allocated
  /// amount exceeds the reservation, propagates the change upward. If 'size' is
  /// zero, the function does nothing.
  void update(int64_t size);

  /// Returns the current memory usage.
  ///
  /// TODO: we will add tracker type to calculate the current bytes based on the
  /// tracker type explicitly.
  int64_t currentBytes() const {
    return adjustByReservation(reservationBytes_);
  }

  /// Returns the unused reservations in bytes.
  int64_t availableReservation() const {
    return std::max<int64_t>(
        0, grantedReservationBytes_ - usedReservationBytes_);
  }

  /// Returns the number of allocations.
  ///
  /// TODO: deprecate this API and get the number of allocations through
  /// stats().
  int64_t numAllocs() const {
    return numAllocs_;
  }

  /// Returns the peak memory usage in bytes including unused reservations.
  ///
  /// TODO: deprecate this API and get peak bytes through stats().
  int64_t peakBytes() const {
    return peakBytes_;
  }

  /// Returns the sum of the size of all allocations. cumulativeBytes() /
  /// numAllocs() is the average size of allocation.
  ///
  /// TODO: deprecate this API and get cumulative bytes through stats().
  int64_t cumulativeBytes() const {
    return cumulativeBytes_;
  }

  /// Returns the total memory reservation size including unused reservation.
  int64_t reservedBytes() const {
    return reservationBytes_;
  }

  /// Returns the actual used memory reservation size which is only meaningful
  /// for a leaf memory tracker.
  int64_t usedReservationBytes() const {
    return usedReservationBytes_;
  }

  int64_t maxMemory() const {
    return parent_ != nullptr ? parent_->maxMemory() : maxMemory_;
  }

  std::shared_ptr<MemoryUsageTracker> addChild() {
    ++numChildren_;
    return create(shared_from_this());
  }

  void setGrowCallback(GrowCallback func) {
    VELOX_CHECK_NULL(
        parent_, "Only root tracker allows to set memory grow callback");
    growCallback_ = func;
  }

  void setMakeMemoryCapExceededMessage(MakeMemoryCapExceededMessage func) {
    makeMemoryCapExceededMessage_ = func;
  }

  /// Returns the stats of this memory usage tracker.
  Stats stats() const;

  std::string toString() const;

  void testingUpdateMaxMemory(int64_t maxMemory) {
    maxMemory_ = maxMemory;
  }

 private:
  static constexpr int64_t kMB = 1 << 20;

  static std::shared_ptr<MemoryUsageTracker> create(
      const std::shared_ptr<MemoryUsageTracker>& parent,
      int64_t maxMemory = kMaxMemory);

  MemoryUsageTracker(
      const std::shared_ptr<MemoryUsageTracker>& parent,
      int64_t maxMemory)
      : parent_(parent), maxMemory_{maxMemory} {
    // NOTE: only the root memory tracker enforces the memory limit check.
    VELOX_CHECK(parent_ == nullptr || maxMemory_ == kMaxMemory);
  }

  void reserve(uint64_t size, bool reserveOnly);
  void release(uint64_t size);

  void maybeUpdatePeakBytes(int64_t newPeak);

  // Increments the reservation and checks against limits at root tracker. Calls
  // root tracker's 'growCallback_' if it is set and limit exceeded. Should be
  // called without holding 'mutex_'. This throws if a limit is exceeded and
  // there is no corresponding GrowCallback or the GrowCallback fails.
  void incrementReservation(uint64_t size);

  //  Decrements the reservation in 'this' and parents.
  void decrementReservation(uint64_t size) noexcept;

  int64_t adjustByReservation(int64_t total) const {
    return grantedReservationBytes_ > 0
        ? std::max<int64_t>(total - availableReservation(), 0)
        : std::max<int64_t>(total, 0);
  }

  // Returns the needed reservation size. If there is sufficient unused memory
  // reservation, this function returns zero.
  int64_t reservationSizeLocked(int64_t size);

  // Returns a rounded up delta based on adding 'delta' to 'size'. Adding the
  // rounded delta to 'size' will result in 'size' a quantized size, rounded to
  // the MB or 8MB for larger sizes.
  static int64_t roundedDelta(int64_t size, int64_t delta) {
    return quantizedSize(size + delta) - size;
  }

  // Returns the next higher quantized size. Small sizes are at MB granularity,
  // larger ones at coarser granularity.
  static uint64_t quantizedSize(uint64_t size) {
    if (size < 16 * kMB) {
      return bits::roundUp(size, kMB);
    }
    if (size < 64 * kMB) {
      return bits::roundUp(size, 4 * kMB);
    }
    return bits::roundUp(size, 8 * kMB);
  }

  void checkNonNegativeSizes(const char* FOLLY_NONNULL message) const;

  void sanityCheckLocked() const;

  // Serializes updates on 'grantedReservationBytes_', 'usedReservationBytes_'
  // and 'minReservationBytes_' to make reservation decision on a consistent
  // read/write of those counters. incrementReservation()/decrementReservation()
  // work based on atomic 'reservationBytes_' without mutex as children updating
  // the same parent do not have to be serialized.
  std::mutex mutex_;
  std::shared_ptr<MemoryUsageTracker> parent_;

  // The memory limit in bytes to enforce.
  int64_t maxMemory_;

  std::atomic<int64_t> peakBytes_{0};
  std::atomic<int64_t> cumulativeBytes_{0};

  // The number of reservation bytes propagated up to the parent for memory
  // limit check at the root tracker.
  std::atomic<int64_t> reservationBytes_{0};

  // The number of granted reservation bytes which is maintained at the leaf
  // tracker and protected by mutex for consistent memory reservation/release
  // decisions.
  //
  // NOTE: after making memory reservation decision, we first try to propagate
  // 'reservationBytes_' increment up to the root tracker without mutex. If the
  // increment succeeds, then update 'grantedReservationBytes_',
  // 'usedReservationBytes_' and 'minReservationBytes_' accordingly with mutex.
  // Correspondingly, a memory release first updates 'grantedReservationBytes_',
  // 'usedReservationBytes_' and 'minReservationBytes_' under mutex, and then
  // propagates the 'reservationBytes_' decrement up to the root tracker without
  // mutex.
  std::atomic<int64_t> grantedReservationBytes_{0};

  // The number of used reservation bytes which is maintained at the leaf
  // tracker and protected by mutex for consistent memory reservation/release
  // decisions.
  std::atomic<int64_t> usedReservationBytes_{0};

  // Minimum amount of reserved memory in bytes to hold until explicit
  // release().
  std::atomic<int64_t> minReservationBytes_{0};

  GrowCallback growCallback_{};

  MakeMemoryCapExceededMessage makeMemoryCapExceededMessage_{};

  // Stats counters.
  // The number of memory allocations through update() including the failed
  // ones.
  std::atomic<uint64_t> numAllocs_{0};

  // The number of memory frees through update().
  std::atomic<uint64_t> numFrees_{0};

  // The number of memory reservations through reserve() and maybeReserve()
  // including the failed ones.
  std::atomic<uint64_t> numReserves_{0};

  // The number of memory releases through update().
  std::atomic<uint64_t> numReleases_{0};

  // The number of internal memory reservation collisions caused by concurrent
  // memory requests.
  std::atomic<uint64_t> numCollisions_{0};

  // The number of created child memory trackers.
  std::atomic<uint64_t> numChildren_{0};
};

std::ostream& operator<<(
    std::ostream& os,
    const MemoryUsageTracker::Stats& stats);
} // namespace facebook::velox::memory
