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
#include <queue>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"

namespace facebook::velox::memory {

class MemoryManager;

constexpr int64_t kMaxMemory = std::numeric_limits<int64_t>::max();

/// This class provides the memory allocation interfaces for a query execution.
/// Each query execution entity creates a dedicated memory pool object. The
/// memory pool objects from a query are organized as a tree with four levels
/// which reflects the query's physical execution plan:
///
/// The top level is a single root pool object (query pool) associated with the
/// query. The query pool is created on the first executed query task and owned
/// by QueryCtx. Note that the query pool is optional as not all the engines
/// using memory pool are creating multiple tasks for the same query in the same
/// process.
///
/// The second level is a number of intermediate pool objects (task pool) with
/// one per each query task. The query pool is the parent of all the task pools
/// of the same query. The task pool is created by the query task and owned by
/// Task.
///
/// The third level is a number of intermediate pool objects (node pool) with
/// one per each query plan node. The task pool is the parent of all the node
/// pools from the task's physical query plan fragment. The node pool is created
/// by the first operator instantiated for the corresponding plan node. It is
/// owned by Task via 'childPools_'
///
/// The bottom level consists of per-operator pools. These are children of the
/// node pool that corresponds to the plan node from which the operator is
/// created. Operator and node pools are owned by the Task via 'childPools_'.
///
/// The query pool is created from IMemoryManager::getChild() as a child of a
/// singleton root pool object (system pool). There is only one system pool for
/// a velox process. Hence each query pool objects forms a subtree rooted from
/// the system pool.
///
/// Each child pool object holds a shared reference to its parent pool object.
/// The parent object tracks its child pool objects through the raw pool object
/// pointer protected by a mutex. The child pool object destruction first
/// removes its raw pointer from its parent through dropChild() and then drops
/// the shared reference on the parent.
///
/// NOTE: for the users that integrate at expression evaluation level, we don't
/// need to build the memory pool hierarchy as described above. Users can either
/// create a single memory pool from IMemoryManager::getChild() to share with
/// all the concurrent expression evaluations or create one dedicated memory
/// pool for each expression evaluation if they need per-expression memory quota
/// enforcement.
///
/// In addition to providing memory allocation functions, the memory pool object
/// also provides memory usage accounting through MemoryUsageTracker. This will
/// be merged into memory pool object later.
class MemoryPool : public std::enable_shared_from_this<MemoryPool> {
 public:
  enum class Kind {
    kLeaf = 0,
    kAggregate = 1,
  };
  static std::string kindString(Kind kind);

  struct Options {
    /// Specifies the memory allocation alignment through this memory pool.
    uint16_t alignment{MemoryAllocator::kMaxAlignment};
    /// Specifies the memory capacity of this memory pool.
    int64_t capacity{kMaxMemory};
    std::shared_ptr<MemoryReclaimer> reclaimer;
  };

  /// Constructs a named memory pool with specified 'parent'.
  MemoryPool(
      const std::string& name,
      Kind kind,
      std::shared_ptr<MemoryPool> parent,
      const Options& options);

  /// Removes this memory pool's tracking from its parent through dropChild().
  /// Drops the shared reference to its parent.
  virtual ~MemoryPool();

  /// Tree methods used to access and manage the memory hierarchy.
  /// Returns the name of this memory pool.
  virtual const std::string& name() const;

  virtual Kind kind() const;

  /// Returns the raw pointer to the parent pool. The root memory pool has
  /// no parent.
  ///
  /// NOTE: users are only safe to access the returned parent pool pointer while
  /// they hold the shared reference on this child memory pool. Otherwise, the
  /// parent memory pool might have been destroyed.
  virtual MemoryPool* parent() const;

  virtual MemoryPool* root() const;

  /// Returns the number of child memory pools.
  virtual size_t childrenCount() const;

  /// Invoked to traverse the memory pool subtree rooted at this, and calls
  /// 'visitor' on each visited child memory pool.
  virtual void visitChildren(std::function<bool(MemoryPool*)> visitor) const;

  /// Invoked to create a named child memory pool from this with specified
  /// 'cap'.
  virtual std::shared_ptr<MemoryPool> addChild(
      const std::string& name,
      Kind kind,
      std::shared_ptr<MemoryReclaimer> reclaimer = nullptr);

  /// Allocates a buffer with specified 'size'.
  virtual void* allocate(int64_t size) = 0;

  /// Allocates a zero-filled buffer with capacity that can store 'numEntries'
  /// entries with each size of 'sizeEach'.
  virtual void* allocateZeroFilled(int64_t numEntries, int64_t sizeEach) = 0;

  /// Re-allocates from an existing buffer with 'newSize' and update memory
  /// usage counting accordingly. If 'newSize' is larger than the current buffer
  /// 'size', the function will allocate a new buffer and free the old buffer.
  virtual void* reallocate(void* p, int64_t size, int64_t newSize) = 0;

  /// Frees an allocated buffer.
  virtual void free(void* p, int64_t size) = 0;

  /// Allocates one or more runs that add up to at least 'numPages', with the
  /// smallest run being at least 'minSizeClass' pages. 'minSizeClass' must be
  /// <= the size of the largest size class. The new memory is returned in 'out'
  /// and any memory formerly referenced by 'out' is freed. The function returns
  /// true if the allocation succeeded. If returning false, 'out' references no
  /// memory and any partially allocated memory is freed.
  virtual void allocateNonContiguous(
      MachinePageCount numPages,
      Allocation& out,
      MachinePageCount minSizeClass = 0) = 0;

  /// Frees non-contiguous 'allocation'. 'allocation' is empty on return.
  virtual void freeNonContiguous(Allocation& allocation) = 0;

  /// Returns the largest class size used by non-contiguous memory allocation.
  virtual MachinePageCount largestSizeClass() const = 0;

  /// Returns the list of supported size class sizes used by non-contiguous
  /// memory allocation.
  virtual const std::vector<MachinePageCount>& sizeClasses() const = 0;

  /// Makes a large contiguous mmap of 'numPages'. The new mapped pages are
  /// returned in 'out' on success. Any formly mapped pages referenced by
  /// 'out' is unmapped in all the cases even if the allocation fails.
  virtual void allocateContiguous(
      MachinePageCount numPages,
      ContiguousAllocation& out) = 0;

  /// Frees contiguous 'allocation'. 'allocation' is empty on return.
  virtual void freeContiguous(ContiguousAllocation& allocation) = 0;

  /// Rounds up to a power of 2 >= size, or to a size halfway between
  /// two consecutive powers of two, i.e 8, 12, 16, 24, 32, .... This
  /// coincides with JEMalloc size classes.
  virtual size_t preferredSize(size_t size);

  /// Returns the memory allocation alignment size applied internally by this
  /// memory pool object.
  virtual uint16_t alignment() const {
    return alignment_;
  }

  /// Resource governing methods used to track and limit the memory usage
  /// through this memory pool object.

  virtual int64_t capacity() const = 0;

  /// Returns the current used memory in bytes.
  virtual int64_t currentBytes() const = 0;

  /// Returns the unused reservations in bytes.
  virtual int64_t availableReservation() const = 0;

  /// Returns the total memory reservation including the used memory and unused
  /// reservations.
  virtual int64_t reservedBytes() const = 0;

  /// Tracks the externally allocated memory usage without doing a new
  /// allocation. If 'size' > 0, the functions reserves the memory
  virtual void reserve(uint64_t size) = 0;

  /// Sometimes in memory governance we want to mock an update for quota
  /// accounting purposes and different implementations can
  /// choose to accommodate this differently.
  virtual void release(uint64_t size) = 0;

  /// Checks if it is likely that the reservation on 'this' can be incremented
  /// by 'size'. Returns false if this seems unlikely. Otherwise attempts
  /// the reservation increment and returns true if succeeded.
  virtual bool maybeReserve(uint64_t size) = 0;

  /// If a minimum reservation has been set with reserve(), resets the minimum
  /// reservation. If the current usage is below the minimum reservation,
  /// decreases reservation and usage down to the rounded actual usage.
  virtual void release() = 0;

  /// Memory arbitration interfaces.
  virtual uint64_t shrinkableBytes() const = 0;
  virtual uint64_t shrink(uint64_t targetBytes = 0) = 0;
  virtual uint64_t grow(int64_t bytes) = 0;

  virtual bool canReclaim() const;
  virtual int64_t reclaimableBytes(bool& reclaimable) const;
  virtual int64_t reclaim(uint64_t targetBytes);

  virtual void enterArbitration();
  virtual void leaveArbitration();

  /// The memory pool's execution stats.
  struct Stats {
    uint64_t peakBytes{0};
    /// The accumulated reserved memory bytes.
    uint64_t cumulativeBytes{0};
    uint64_t numAllocs{0};
    uint64_t numFrees{0};
    uint64_t numReserves{0};
    uint64_t numReleases{0};
    uint64_t numShrinks{0};
    uint64_t numReclaims{0};
    /// The number of memory reservation collisions caused by concurrent memory
    /// requests.
    uint64_t numCollisions{0};
    /// The number of created child memory pools.
    uint64_t numChildren{0};

    bool operator==(const Stats& rhs) const;

    std::string toString() const;
  };

  /// Returns the stats of this memory usage tracker.
  virtual Stats stats() const = 0;

  virtual std::string toString() const = 0;

 protected:
  /// Invoked by addChild() to create a child memory pool object. 'parent' is
  /// a shared pointer created from this.
  virtual std::shared_ptr<MemoryPool> genChild(
      std::shared_ptr<MemoryPool> parent,
      const std::string& name,
      Kind kind,
      std::shared_ptr<MemoryReclaimer>) = 0;

  /// Invoked only on destruction to remove this memory pool from its parent's
  /// child memory pool tracking.
  virtual void dropChild(const MemoryPool* child);

  FOLLY_ALWAYS_INLINE virtual void checkAllocation() {
    VELOX_CHECK_EQ(
        kind_,
        Kind::kLeaf,
        "Memory allocation is only allowed on leaf memory pool: {}",
        toString());
  }

  const std::string name_;
  const Kind kind_;
  const uint16_t alignment_;
  const std::shared_ptr<MemoryPool> parent_;
  const std::shared_ptr<MemoryReclaimer> reclaimer_;

  mutable folly::SharedMutex childMutex_;
  std::unordered_map<std::string, std::weak_ptr<MemoryPool>> children_;
};

std::ostream& operator<<(std::ostream& out, MemoryPool::Kind kind);

std::ostream& operator<<(std::ostream& os, const MemoryPool::Stats& stats);

class MemoryPoolImpl : public MemoryPool {
 public:
  using DestructionCallback = std::function<void(MemoryPool*)>;

  MemoryPoolImpl(
      MemoryManager* manager,
      const std::string& name,
      Kind kind,
      std::shared_ptr<MemoryPool> parent,
      DestructionCallback destructionCb,
      const Options& options);

  ~MemoryPoolImpl();

  void* allocate(int64_t size) override;

  void* allocateZeroFilled(int64_t numEntries, int64_t sizeEach) override;

  void* reallocate(void* p, int64_t size, int64_t newSize) override;

  void free(void* p, int64_t size) override;

  void allocateNonContiguous(
      MachinePageCount numPages,
      Allocation& out,
      MachinePageCount minSizeClass = 0) override;

  void freeNonContiguous(Allocation& allocation) override;

  MachinePageCount largestSizeClass() const override;

  const std::vector<MachinePageCount>& sizeClasses() const override;

  void allocateContiguous(MachinePageCount numPages, ContiguousAllocation& out)
      override;

  void freeContiguous(ContiguousAllocation& allocation) override;

  int64_t capacity() const override;

  int64_t currentBytes() const override;

  int64_t availableReservation() const override;

  int64_t reservedBytes() const override;

  void reserve(uint64_t size) override;

  void release(uint64_t size) override;

  bool maybeReserve(uint64_t size) override;

  void release() override;

  uint64_t shrinkableBytes() const override;
  uint64_t shrink(uint64_t targetBytes = 0) override;
  uint64_t grow(int64_t bytes) override;

  Stats stats() const override;

  std::string toString() const override {
    std::lock_guard<std::mutex> l(mutex_);
    return toStringLocked();
  }

  static FOLLY_ALWAYS_INLINE MemoryPoolImpl* fromPool(MemoryPool* pool) {
    return static_cast<MemoryPoolImpl*>(pool);
  }

  static FOLLY_ALWAYS_INLINE MemoryPoolImpl* fromPool(
      const std::shared_ptr<MemoryPool>& pool) {
    return static_cast<MemoryPoolImpl*>(pool.get());
  }

  struct MemoryUsage {
    std::string name;
    int64_t currentUsage;
    int64_t peakUsage;

    bool operator>(const MemoryUsage& other) const {
      return std::tie(currentUsage, peakUsage, name) >
          std::tie(other.currentUsage, other.peakUsage, other.name);
    }

    std::string toString() const {
      return fmt::format(
          "{} usage {} peak {}",
          name,
          succinctBytes(currentUsage),
          succinctBytes(peakUsage));
    }
  };

  struct MemoryUsageComp {
    bool operator()(const MemoryUsage& lhs, const MemoryUsage& rhs) const {
      return lhs > rhs;
    }
  };
  using MemoryUsageHeap = std::
      priority_queue<MemoryUsage, std::vector<MemoryUsage>, MemoryUsageComp>;

  void testingSetCapacity(int64_t bytes);

  MemoryAllocator* testingAllocator() const {
    return allocator_;
  }

 private:
  static constexpr uint64_t kMB = 1 << 20;

  std::shared_ptr<MemoryPool> genChild(
      std::shared_ptr<MemoryPool> parent,
      const std::string& name,
      Kind kind,
      std::shared_ptr<MemoryReclaimer>) override;

  // Increments the reservation and checks against limits at root tracker. Calls
  // root tracker's 'growCallback_' if it is set and limit exceeded. Should be
  // called without holding 'mutex_'. This throws if a limit is exceeded and
  // there is no corresponding GrowCallback or the GrowCallback fails.
  bool incrementReservation(MemoryPool* requestor, uint64_t size);

  // Decrements the reservation in 'this' and parents.
  void decrementReservation(uint64_t size) noexcept;

  void sanityCheckLocked() const;

  int64_t sizeAlign(int64_t size);

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

  void reserve(uint64_t size, bool reserveOnly);

  // Returns the needed reservation size. If there is sufficient unused memory
  // reservation, this function returns zero.
  int64_t reservationSizeLocked(int64_t size);

  void maybeUpdatePeakBytesLocked(int64_t newPeak);

  std::string capExceedingMessage(uint64_t incrementBytes);

  MemoryUsage memoryUsage() const;
  MemoryUsage memoryUsageLocked() const;

  void capExceedingMessage(
      size_t indent,
      MemoryUsageHeap& topLeafMemUsages,
      std::stringstream& out);

  int64_t currentBytesLocked() const;
  int64_t availableReservationLocked() const;

  std::string toStringLocked() const;

  MemoryManager* const memoryManager_;
  MemoryAllocator* const allocator_;
  const DestructionCallback destructionCb_;

  // Serializes updates on 'grantedReservationBytes_', 'usedReservationBytes_'
  // and 'minReservationBytes_' to make reservation decision on a consistent
  // read/write of those counters. incrementReservation()/decrementReservation()
  // work based on atomic 'reservationBytes_' without mutex as children updating
  // the same parent do not have to be serialized.
  mutable std::mutex mutex_;

  // The memory cap in bytes to enforce.
  int64_t capacity_;

  // The number of reservation bytes.
  int64_t reservationBytes_{0};

  // The number of used reservation bytes which is maintained at the leaf
  // tracker and protected by mutex for consistent memory reservation/release
  // decisions.
  int64_t usedReservationBytes_{0};

  // Minimum amount of reserved memory in bytes to hold until explicit
  // release().
  int64_t minReservationBytes_{0};

  int64_t peakBytes_{0};
  int64_t cumulativeBytes_{0};

  // Stats counters.
  // The number of memory allocations through update() including the failed
  // ones.
  uint64_t numAllocs_{0};

  // The number of memory frees through update().
  uint64_t numFrees_{0};

  // The number of memory reservations through reserve() and maybeReserve()
  // including the failed ones.
  uint64_t numReserves_{0};

  // The number of memory releases through update().
  uint64_t numReleases_{0};

  // The number of internal memory reservation collisions caused by concurrent
  // memory requests.
  uint64_t numCollisions_{0};

  uint64_t numReclaims_{0};
  uint64_t numShrinks_{0};
};

/// An Allocator backed by a memory pool for STL containers.
template <typename T>
class StlAllocator {
 public:
  typedef T value_type;
  MemoryPool& pool;

  /* implicit */ StlAllocator(MemoryPool& pool) : pool{pool} {}

  template <typename U>
  /* implicit */ StlAllocator(const StlAllocator<U>& a) : pool{a.pool} {}

  T* allocate(size_t n) {
    return static_cast<T*>(pool.allocate(checkedMultiply(n, sizeof(T))));
  }

  void deallocate(T* p, size_t n) {
    pool.free(p, checkedMultiply(n, sizeof(T)));
  }

  template <typename T1>
  bool operator==(const StlAllocator<T1>& rhs) const {
    if constexpr (std::is_same_v<T, T1>) {
      return &this->pool == &rhs.pool;
    }
    return false;
  }

  template <typename T1>
  bool operator!=(const StlAllocator<T1>& rhs) const {
    return !(*this == rhs);
  }
};
} // namespace facebook::velox::memory
