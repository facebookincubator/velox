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

#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/future/VeloxPromise.h"

namespace facebook::velox::memory {

class MemoryPool;

using MemoryArbitrationStateCheckCB = std::function<void(MemoryPool&)>;

/// The memory arbitrator interface. There is one memory arbitrator object per
/// memory manager which is responsible for arbitrating memory usage among the
/// query memory pools for query memory isolation. When a memory pool exceeds
/// its capacity limit, it sends a memory grow request to the memory manager
/// which forwards the request to the memory arbitrator behind along with all
/// the query memory pools as the arbitration candidates. The arbitrator tries
/// to free up the overused memory from the selected candidates according to the
/// supported query memory isolation policy. The memory arbitrator can free up
/// memory by either reclaiming the used memory from a running query through
/// techniques such as disk spilling or aborting a query. Different memory
/// arbitrator implementations achieve different query memory isolation policy
/// (see Kind definition below).
class MemoryArbitrator {
 public:
  struct Config {
    /// The string kind of this memory arbitrator.
    ///
    /// NOTE: If kind is not set, a noop arbitrator is created which grants the
    /// maximum capacity to each newly created memory pool.
    std::string kind{};

    /// The total memory capacity in bytes of all the running queries.
    ///
    /// NOTE: this should be same capacity as we set in the associated memory
    /// manager.
    int64_t capacity;

    /// The initial memory capacity to reserve for a newly created memory pool.
    uint64_t memoryPoolInitCapacity{256 << 20};

    /// The minimal memory capacity to transfer out of or into a memory pool
    /// during the memory arbitration.
    uint64_t memoryPoolTransferCapacity{32 << 20};

    /// Provided by the query system to validate the state after a memory pool
    /// enters arbitration if not null. For instance, Prestissimo provides
    /// callback to check if a memory arbitration request is issued from a
    /// driver thread, then the driver should be put in suspended state to avoid
    /// the potential deadlock when reclaim memory from the task of the request
    /// memory pool.
    MemoryArbitrationStateCheckCB arbitrationStateCheckCb{nullptr};
  };

  using Factory = std::function<std::unique_ptr<MemoryArbitrator>(
      const MemoryArbitrator::Config& config)>;

  /// Register factory for a specific 'kind' of memory arbitrator
  /// MemoryArbitrator::Create looks up the registry to find the factory to
  /// create arbitrator instance based on the kind specified in arbitrator
  /// config.
  ///
  /// NOTE: we only allow the same 'kind' of memory arbitrator to be registered
  /// once. The function throws an error if 'kind' is already registered.
  static void registerFactory(const std::string& kind, Factory factory);

  /// Unregister the registered factory for a specifc kind.
  ///
  /// NOTE: the function throws if the specified arbitrator 'kind' is not
  /// registered.
  static void unregisterFactory(const std::string& kind);

  /// Register all the supported memory arbitrator kinds.
  static void registerAllFactories();

  /// Unregister all the supported memory arbitrator kinds.
  static void unregisterAllFactories();

  /// Invoked by the memory manager to create an instance of memory arbitrator
  /// based on the kind specified in 'config'. The arbitrator kind must be
  /// registered through MemoryArbitrator::registerFactory(), otherwise the
  /// function throws a velox user exception error.
  ///
  /// NOTE: if arbitrator kind is not set in 'config', then the function returns
  /// nullptr, and the memory arbitration function is disabled.
  static std::unique_ptr<MemoryArbitrator> create(const Config& config);

  virtual std::string kind() const = 0;

  uint64_t capacity() const {
    return capacity_;
  }

  virtual ~MemoryArbitrator() = default;

  /// Invoked by the memory manager to reserve up to 'bytes' memory capacity
  /// without actually freeing memory for a newly created memory pool. The
  /// function will set the memory pool's capacity based on the actually
  /// reserved memory.
  ///
  /// NOTE: the memory arbitrator can decides how much memory capacity is
  /// actually reserved for a newly created memory pool. The latter can trigger
  /// the memory arbitration on demand when actual memory allocation happens.
  virtual void reserveMemory(MemoryPool* pool, uint64_t bytes) = 0;

  /// Invoked by the memory manager to return back all the reserved memory
  /// capacity of a destroying memory pool.
  virtual void releaseMemory(MemoryPool* pool) = 0;

  /// Invoked by the memory manager to grow a memory pool's capacity.
  /// 'pool' is the memory pool to request to grow. 'candidates' is a list
  /// of query root pools to participate in the memory arbitration. The memory
  /// arbitrator picks up a number of pools to either shrink its memory capacity
  /// without actually freeing memory or reclaim its used memory to free up
  /// enough memory for 'requestor' to grow. Different arbitrators use different
  /// policies to select the candidate pools. The shared memory arbitrator used
  /// by both Prestissimo and Prestissimo-on-Spark, selects the candidates with
  /// more memory capacity.
  ///
  /// NOTE: the memory manager keeps 'candidates' valid during the arbitration
  /// processing.
  virtual bool growMemory(
      MemoryPool* pool,
      const std::vector<std::shared_ptr<MemoryPool>>& candidatePools,
      uint64_t targetBytes) = 0;

  /// Invoked by the memory manager to shrink memory from a given list of memory
  /// pools. The freed memory capacity is given back to the arbitrator. The
  /// function returns the actual freed memory capacity in bytes.
  virtual uint64_t shrinkMemory(
      const std::vector<std::shared_ptr<MemoryPool>>& pools,
      uint64_t targetBytes) = 0;

  /// The internal execution stats of the memory arbitrator.
  struct Stats {
    /// The number of arbitration requests.
    uint64_t numRequests{0};
    /// The number of succeeded arbitration requests.
    uint64_t numSucceeded{0};
    /// The number of aborted arbitration requests.
    uint64_t numAborted{0};
    /// The number of arbitration request failures.
    uint64_t numFailures{0};
    /// The sum of all the arbitration request queue times in microseconds.
    uint64_t queueTimeUs{0};
    /// The sum of all the arbitration run times in microseconds.
    uint64_t arbitrationTimeUs{0};
    /// The amount of memory bytes freed by reducing the memory pool's capacity
    /// without actually freeing memory.
    uint64_t numShrunkBytes{0};
    /// The amount of memory bytes freed by memory reclamation.
    uint64_t numReclaimedBytes{0};
    /// The max memory capacity in bytes.
    uint64_t maxCapacityBytes{0};
    /// The free memory capacity in bytes.
    uint64_t freeCapacityBytes{0};
    /// The sum of all reclaim operation durations during arbitration in
    /// microseconds.
    uint64_t reclaimTimeUs{0};

    Stats(
        uint64_t _numRequests,
        uint64_t _numSucceeded,
        uint64_t _numAborted,
        uint64_t _numFailures,
        uint64_t _queueTimeUs,
        uint64_t _arbitrationTimeUs,
        uint64_t _numShrunkBytes,
        uint64_t _numReclaimedBytes,
        uint64_t _maxCapacityBytes,
        uint64_t _freeCapacityBytes,
        uint64_t _reclaimTimeUs);

    Stats() = default;

    Stats operator-(const Stats& other) const;
    bool operator==(const Stats& other) const;
    bool operator!=(const Stats& other) const;
    bool operator<(const Stats& other) const;
    bool operator>(const Stats& other) const;
    bool operator>=(const Stats& other) const;
    bool operator<=(const Stats& other) const;

    bool empty() const {
      return numRequests == 0;
    }

    /// Returns the debug string of this stats.
    std::string toString() const;
  };

  virtual Stats stats() const = 0;

  /// Returns the debug string of this memory arbitrator.
  virtual std::string toString() const = 0;

 protected:
  explicit MemoryArbitrator(const Config& config)
      : capacity_(config.capacity),
        memoryPoolInitCapacity_(config.memoryPoolInitCapacity),
        memoryPoolTransferCapacity_(config.memoryPoolTransferCapacity),
        arbitrationStateCheckCb_(config.arbitrationStateCheckCb) {}

  const uint64_t capacity_;
  const uint64_t memoryPoolInitCapacity_;
  const uint64_t memoryPoolTransferCapacity_;
  const MemoryArbitrationStateCheckCB arbitrationStateCheckCb_;
};

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& o,
    const MemoryArbitrator::Stats& stats) {
  return o << stats.toString();
}

/// The memory reclaimer interface is used by memory pool to participate in
/// the memory arbitration execution (enter/leave arbitration process) as well
/// as reclaim memory from the associated query object. We have default
/// implementation that always reclaim memory from the child memory pool with
/// most reclaimable memory. This is used by the query and plan node memory
/// pools which don't need customized operations. A task memory pool needs to
/// to pause a task execution before reclaiming memory from its child pools.
/// This avoids any potential race condition between concurrent memory
/// reclamation operation and the task activities. An operator memory pool needs
/// to to put the the associated task driver thread into suspension state before
/// entering into an arbitration process. It is because the memory arbitrator
/// needs to pause a task execution before reclaim memory from the task. It is
/// possible that the memory arbitration tries to reclaim memory from the task
/// which initiates the memory arbitration request. If we don't put the driver
/// thread into suspension state, then the memory arbitration process might
/// run into deadlock as the task will never be paused. The operator memory pool
/// also needs to reclaim the actually used memory from the associated operator
/// through techniques such as disks spilling.
class MemoryReclaimer {
 public:
  virtual ~MemoryReclaimer() = default;

  static std::unique_ptr<MemoryReclaimer> create();

  /// Invoked by the memory arbitrator before entering the memory arbitration
  /// processing. The default implementation does nothing but user can override
  /// this if needs. For example, an operator memory reclaimer needed to put the
  /// associated driver execution into suspension state so that the task which
  /// initiates the memory arbitration request, can be paused for memory
  /// reclamation during the memory arbitration processing.
  virtual void enterArbitration() {}

  /// Invoked by the memory arbitrator after finishes the memory arbitration
  /// processing and is used in pair with 'enterArbitration'. For example, an
  /// operator memory pool needs to moves the associated driver execution out of
  /// the suspension state.
  ///
  /// NOTE: it is guaranteed to be called also on failure path if
  /// enterArbitration has been called.
  virtual void leaveArbitration() noexcept {}

  /// Invoked by the memory arbitrator to get the amount of memory bytes that
  /// can be reclaimed from 'pool'. The function returns true if 'pool' is
  /// reclaimable and returns the estimated reclaimable bytes in
  /// 'reclaimableBytes'.
  virtual bool reclaimableBytes(
      const MemoryPool& pool,
      uint64_t& reclaimableBytes) const;

  /// Invoked by the memory arbitrator to reclaim from memory 'pool' with
  /// specified 'targetBytes'. It is expected to reclaim at least that amount of
  /// memory bytes but there is no guarantees. If 'targetBytes' is zero, then it
  /// reclaims all the reclaimable memory from the memory 'pool'. The function
  /// returns the actual reclaimed memory bytes.
  virtual uint64_t reclaim(MemoryPool* pool, uint64_t targetBytes);

  /// Invoked by the memory arbitrator to abort memory 'pool' and the associated
  /// query execution when encounters non-recoverable memory reclaim error or
  /// fails to reclaim enough free capacity. The abort is a synchronous
  /// operation and we expect most of used memory to be freed after the abort
  /// completes. 'error' should be passed in as the direct cause of the
  /// abortion. It will be propagated all the way to task level for accurate
  /// error exposure.
  virtual void abort(MemoryPool* pool, const std::exception_ptr& error);

 protected:
  MemoryReclaimer() = default;
};

/// The memory arbitration context which is set on per-thread local variable by
/// memory arbitrator. It is used to indicate a running thread is under memory
/// arbitration processing or not. This helps to enable sanity check such as all
/// the memory reservations during memory arbitration should come from the
/// spilling memory pool.
struct MemoryArbitrationContext {
  const MemoryPool& requestor;
};

/// Object used to set/restore the memory arbitration context when a thread is
/// under memory arbitration processing.
class ScopedMemoryArbitrationContext {
 public:
  explicit ScopedMemoryArbitrationContext(const MemoryPool& requestor);
  ~ScopedMemoryArbitrationContext();

 private:
  MemoryArbitrationContext* const savedArbitrationCtx_{nullptr};
  MemoryArbitrationContext currentArbitrationCtx_;
};

/// Returns the memory arbitration context set by a per-thread local variable if
/// the running thread is under memory arbitration processing.
MemoryArbitrationContext* memoryArbitrationContext();

/// Returns true if the running thread is under memory arbitration or not.
bool underMemoryArbitration();
} // namespace facebook::velox::memory
