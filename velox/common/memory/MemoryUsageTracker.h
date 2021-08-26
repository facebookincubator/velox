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

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::memory {
constexpr int64_t kMaxMemory = std::numeric_limits<int64_t>::max();

#define VELOX_MEM_CAP_EXCEEDED()                                    \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kMemCapExceeded.c_str(),       \
      /* isRetriable */ true);

struct MemoryUsageConfig {
  std::optional<int64_t> maxUserMemory;
  std::optional<int64_t> maxSystemMemory;
  std::optional<int64_t> maxTotalMemory;
  std::optional<int64_t> maxRecoverableMemory;
  bool allowOvercommit{false};
  bool isRecoverable{false};
  bool forMemoryManager{false};
};

struct MemoryUsageConfigBuilder {
  MemoryUsageConfig config;

  MemoryUsageConfigBuilder(MemoryUsageConfig c = MemoryUsageConfig())
      : config(c) {}

  MemoryUsageConfigBuilder& maxUserMemory(int64_t max) {
    config.maxUserMemory = max;
    return *this;
  }

  MemoryUsageConfigBuilder& maxSystemMemory(int64_t max) {
    config.maxSystemMemory = max;
    return *this;
  }

  MemoryUsageConfigBuilder& maxTotalMemory(int64_t max) {
    config.maxTotalMemory = max;
    return *this;
  }

  MemoryUsageConfigBuilder& maxRecoverableMemory(int64_t max) {
    config.maxRecoverableMemory = max;
    return *this;
  }

  MemoryUsageConfigBuilder& allowOvercommit(bool overcommit) {
    config.allowOvercommit = overcommit;
    return *this;
  }

  MemoryUsageConfigBuilder& isRecoverable(bool recoverable) {
    config.isRecoverable = recoverable;
    return *this;
  }

  MemoryUsageConfigBuilder& forMemoryManager(bool forMemMgr) {
    config.forMemoryManager = forMemMgr;
    return *this;
  }

  MemoryUsageConfig build() {
    return config;
  }
};

enum class UsageType : int {
  kUserMem = 0,
  kSystemMem = 1,
  kTotalMem = 2,
  kRecoverableMem = 3
};

// Keeps track of currently outstanding and peak outstanding memory
// and cumulative allocation volume for a MemoryPool or MappedMemory
// allocator. This is supplied at construction when creating a child
// MemoryPool or MappedMemory. These trackers form a tree. The
// tracking information is aggregated from leaf to root. Each level
// in the tree may specify a maximum allocation. Updating the
// allocated amount past the maximum will call an optional grow callback and
// throw if the callback did not make more space. This is why the update should
// precede the actual allocation. A tracker can specify a reservation. Making a
// reservation means that no memory is allocated but the allocated size  is
// updated so as to make sure that at least the reserved  amount can be
// allocated. Allocations that fit within the reserved are not
// counted as new because the counting  was done when
// making the reservation.  Allocating past the reservation is possible, but
// then allocation is recorded  in the tracker and it can fail. Freeing data
// when above reservation counts as free up to the reservation size. Freeing
// data within the reservation drops the usage but not the reservation.
// release() frees unused reserved capacity.
class MemoryUsageTracker
    : public std::enable_shared_from_this<MemoryUsageTracker> {
 public:
  // Function to increase size limit. Returns true if limit
  // increased. On success, increases the limits in 'limit' so as to
  // have 'size' more bytes of 'type'. This is responsible for updating all
  // parent trackers of 'limit'.
  using GrowCallback = std::function<
      bool(UsageType type, int64_t size, MemoryUsageTracker* limit)>;

  // Create default usage tracker. It aggregates both 'user' and 'system' memory
  // from its children and tracks the allocations as 'user' memory. It returns a
  // 'root' tracker.
  static std::shared_ptr<MemoryUsageTracker> create(
      const MemoryUsageConfig& config = MemoryUsageConfig()) {
    return create(nullptr, UsageType::kUserMem, config);
  }

  // FIXME(venkatra): Remove this once presto_cpp is updated to use above
  // function.
  static std::shared_ptr<MemoryUsageTracker> create(
      int64_t maxUserMemory,
      int64_t maxSystemMemory,
      int64_t maxTotalMemory) {
    return create(
        nullptr,
        UsageType::kUserMem,
        MemoryUsageConfigBuilder()
            .maxUserMemory(maxUserMemory)
            .maxSystemMemory(maxSystemMemory)
            .maxTotalMemory(maxTotalMemory)
            .build());
  }

  static std::shared_ptr<MemoryUsageTracker> create(
      const std::shared_ptr<MemoryUsageTracker>& parent,
      UsageType type,
      const MemoryUsageConfig& config);

  int64_t getOvercommittedMemory() const {
    return getCurrentTotalBytes() + getCurrentRecoverableBytes() -
        getMaxTotalBytes();
  }

  // Increments the reservation for 'this' so that we can allocate
  // at least 'size' bytes on top of the current allocation. This is
  // used when an a memory user needs to allocate more memory and
  // needs a guarantee of at least 'size' being available. If less
  // memory ends up being needed, the unused reservation should be released with
  // release().
  //   If the new reserved amount exceeds the
  // usage limit, an exception will be thrown.  Note that this
  // function is not thread-safe. We reserve on a
  // leaf memory pool or mapped memory, which is accessed within a
  // single thread.
  void reserve(int64_t size) {
    int64_t actualSize = size - (reservation_ - usedReservation_);
    if (actualSize > 0) {
      update(type_, actualSize);
      reservation_ += actualSize;
    }
  }

  // Release unused reservation. Used reservation will be released as the
  // allocations are freed. Note that this function is not thread-safe.
  void release() {
    int64_t remaining = reservation_ - usedReservation_;
    if (remaining) {
      update(type_, -remaining);
    }
    reservation_ = 0;
    usedReservation_ = 0;
  }

  // Increments outstanding memory by 'size', which is positive for
  // allocation and negative for free. If there is no reservation or
  // the new allocated amount exceeds the reservation, propagates the
  // change upward.
  void update(int64_t size) {
    if (int64_t increment = updateUsed(size)) {
      try {
        update(type_, increment);
      } catch (const VeloxRuntimeError& e) {
        // Revert the increment to reservation usage.
        usedReservation_.fetch_sub(size);
        std::rethrow_exception(std::current_exception());
      }
    }
  }

  int64_t getCurrentUserBytes() const {
    return currentUsageInBytes_[static_cast<int>(UsageType::kUserMem)];
  }
  int64_t getCurrentSystemBytes() const {
    return currentUsageInBytes_[static_cast<int>(UsageType::kSystemMem)];
  }
  int64_t getCurrentTotalBytes() const {
    return getCurrentUserBytes() + getCurrentSystemBytes() +
        getCurrentRecoverableBytes();
  }
  int64_t getCurrentRecoverableBytes() const {
    return currentUsageInBytes_[static_cast<int>(UsageType::kRecoverableMem)];
  }
  int64_t getPeakUserBytes() const {
    return peakUsageInBytes_[static_cast<int>(UsageType::kUserMem)];
  }
  int64_t getPeakSystemBytes() const {
    return peakUsageInBytes_[static_cast<int>(UsageType::kSystemMem)];
  }
  int64_t getPeakTotalBytes() const {
    return peakUsageInBytes_[static_cast<int>(UsageType::kTotalMem)];
  }
  int64_t getPeakRecoverableBytes() const {
    return peakUsageInBytes_[static_cast<int>(UsageType::kRecoverableMem)];
  }
  int64_t getMaxTotalBytes() const {
    return maxMemory_[static_cast<int>(UsageType::kTotalMem)];
  }

  int64_t getAvailableReservation() const {
    return std::max<int64_t>(0, reservation_ - usedReservation_);
  }

  int64_t getNumAllocs() const {
    return numAllocs_[static_cast<int>(UsageType::kTotalMem)];
  }

  int64_t getCumulativeBytes() const {
    return cumulativeBytes_[static_cast<int>(UsageType::kTotalMem)];
  }

  std::shared_ptr<MemoryUsageTracker> addChild(
      bool trackSystemMem = false,
      const MemoryUsageConfig& config = MemoryUsageConfig()) {
    return create(
        shared_from_this(),
        trackSystemMem ? UsageType::kSystemMem : UsageType::kUserMem,
        config);
  }

  void updateConfig(const MemoryUsageConfig& config) {
    if (config.maxUserMemory.has_value()) {
      maxMemory_[static_cast<int>(UsageType::kUserMem)] =
          config.maxUserMemory.value();
    }
    if (config.maxSystemMemory.has_value()) {
      maxMemory_[static_cast<int>(UsageType::kSystemMem)] =
          config.maxUserMemory.value();
    }
    if (config.maxTotalMemory.has_value()) {
      maxMemory_[static_cast<int>(UsageType::kTotalMem)] =
          config.maxTotalMemory.value();
    }
    if (config.maxRecoverableMemory.has_value()) {
      maxMemory_[static_cast<int>(UsageType::kRecoverableMem)] =
          config.maxRecoverableMemory.value();
    }
  }

  void increment(UsageType type, int64_t size) {
    currentUsageInBytes_[static_cast<int32_t>(UsageType::kTotalMem)].fetch_add(
        size);

    currentUsageInBytes_[static_cast<int32_t>(type)].fetch_add(size);
  }

  int64_t getIfCan(UsageType type, int64_t size) {
    int64_t limit = maxMemory_[static_cast<int32_t>(UsageType::kTotalMem)];

    if (currentUsageInBytes_[static_cast<int32_t>(UsageType::kTotalMem)]
                .fetch_add(size) +
            size >
        limit) {
      currentUsageInBytes_[static_cast<int32_t>(UsageType::kTotalMem)]
          .fetch_sub(size);
      return false;
    }
    currentUsageInBytes_[static_cast<int32_t>(type)].fetch_add(size);
    return true;
  }

  UsageType type() const {
    return type_;
  }

  void setGrowCallback(GrowCallback func) {
    growCallback_ = func;
  }

  std::string toString() const;

 private:
  GrowCallback growCallback_;
  explicit MemoryUsageTracker(
      const std::shared_ptr<MemoryUsageTracker>& parent,
      UsageType type,
      const MemoryUsageConfig& config)
      : parent_(parent),
        type_(config.isRecoverable ? UsageType::kRecoverableMem : type),
        maxMemory_{
            config.maxUserMemory.value_or(kMaxMemory),
            config.maxSystemMemory.value_or(kMaxMemory),
            config.maxTotalMemory.value_or(kMaxMemory)},
        allowOvercommit_{config.allowOvercommit},
        forMemoryManager_{config.forMemoryManager} {}

  void maySetMax(UsageType type, int64_t newPeak) {
    auto& peakUsage = peakUsageInBytes_[static_cast<int>(type)];
    int64_t oldPeak = peakUsage;
    while (oldPeak < newPeak &&
           !peakUsage.compare_exchange_weak(oldPeak, newPeak)) {
      oldPeak = peakUsage;
    }
  }

  void update(UsageType type, int64_t size);

  // Increments the amount of 'usedReservation_' by 'size'.  Returns the
  // amount by which current size must be incremented. If both old and
  // new values are below the reservation, there is no increment. If
  // the increment crosses reservation, then the increment is, in the
  // case of allocation, the amount that goes above the reservation
  // and for deallocation the amount that takes the current size to
  // reservation but not below this. If both new and old amounts used
  // are above reservation, the increment is the delta between the
  // two.
  int64_t updateUsed(int64_t size) {
    int64_t increment = size;
    int64_t reservation = reservation_;
    if (reservation > 0) {
      int64_t oldUsed = 0;
      int64_t newUsed = 0;
      do {
        oldUsed = usedReservation_;
        newUsed = oldUsed + size;
        if (newUsed <= reservation && oldUsed <= reservation_) {
          // The usage stays below reservation. No change to allocated size.
          increment = 0;
        } else if (newUsed < reservation_ && oldUsed > reservation_) {
          // Usage goes below reservation due to a free.
          increment = reservation_ - oldUsed;
        } else if (newUsed > reservation_ && oldUsed < reservation_) {
          // Usage goes above reservation due to allocation.
          increment = newUsed - reservation_;
        } else {
          // Old and new are both above reservation. The delta is directly
          // reflected in allocated size.
          increment = size;
        }
      } while (!usedReservation_.compare_exchange_weak(oldUsed, newUsed));
    }
    return increment;
  }

  std::shared_ptr<MemoryUsageTracker> parent_;
  UsageType type_;
  std::array<std::atomic<int64_t>, 4> currentUsageInBytes_{};
  std::array<std::atomic<int64_t>, 4> peakUsageInBytes_{};
  std::array<int64_t, 3> maxMemory_;
  std::array<int64_t, 3> numAllocs_{};
  std::array<int64_t, 3> cumulativeBytes_{};

  int64_t reservation_{0};
  std::atomic<int64_t> usedReservation_{};

  bool allowOvercommit_;
  bool forMemoryManager_;
};
} // namespace facebook::velox::memory
