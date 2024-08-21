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

#include <shared_mutex>

#include "velox/common/base/Counters.h"
#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitratorUtil.h"

namespace facebook::velox::memory {

class SharedArbitrator;
class ArbitrationOperation;

/// Used to achieve dynamic memory sharing among running queries. When a
/// memory pool exceeds its current memory capacity, the arbitrator tries to
/// grow its capacity by reclaim the overused memory from the query with
/// more memory usage. We can configure memory arbitrator the way to reclaim
/// memory. For Prestissimo, we can configure it to reclaim memory by
/// aborting a query. For Prestissimo-on-Spark, we can configure it to
/// reclaim from a running query through techniques such as disk-spilling,
/// partial aggregation or persistent shuffle data flushes.
class SharedArbitrator : public memory::MemoryArbitrator {
 public:
  struct ExtraConfig {
    /// The memory capacity reserved to ensure each running query has minimal
    /// capacity of 'memoryPoolReservedCapacity' to run.
    static constexpr std::string_view kReservedCapacity{"reserved-capacity"};
    static constexpr std::string_view kDefaultReservedCapacity{"0B"};
    static int64_t getReservedCapacity(
        const std::unordered_map<std::string, std::string>& configs);

    /// The initial memory capacity to reserve for a newly created query memory
    /// pool.
    static constexpr std::string_view kMemoryPoolInitialCapacity{
        "memory-pool-initial-capacity"};
    static constexpr std::string_view kDefaultMemoryPoolInitialCapacity{
        "256MB"};
    static uint64_t getMemoryPoolInitialCapacity(
        const std::unordered_map<std::string, std::string>& configs);

    /// The minimal amount of memory capacity reserved for each query to run.
    static constexpr std::string_view kMemoryPoolReservedCapacity{
        "memory-pool-reserved-capacity"};
    static constexpr std::string_view kDefaultMemoryPoolReservedCapacity{"0B"};
    static uint64_t getMemoryPoolReservedCapacity(
        const std::unordered_map<std::string, std::string>& configs);

    /// The minimal memory capacity to transfer out of or into a memory pool
    /// during the memory arbitration.
    static constexpr std::string_view kMemoryPoolTransferCapacity{
        "memory-pool-transfer-capacity"};
    static constexpr std::string_view kDefaultMemoryPoolTransferCapacity{
        "128MB"};
    static uint64_t getMemoryPoolTransferCapacity(
        const std::unordered_map<std::string, std::string>& configs);

    /// Specifies the max time to wait for memory reclaim by arbitration. The
    /// memory reclaim might fail if the max time has exceeded. This prevents
    /// the memory arbitration from getting stuck when the memory reclaim waits
    /// for a hanging query task to pause. If it is zero, then there is no
    /// timeout.
    static constexpr std::string_view kMemoryReclaimMaxWaitTime{
        "memory-reclaim-max-wait-time"};
    static constexpr std::string_view kDefaultMemoryReclaimMaxWaitTime{"0ms"};
    static uint64_t getMemoryReclaimMaxWaitTimeMs(
        const std::unordered_map<std::string, std::string>& configs);

    /// When shrinking capacity, the shrink bytes will be adjusted in a way such
    /// that AFTER shrink, the stricter (whichever is smaller) of the following
    /// conditions is met, in order to better fit the pool's current memory
    /// usage:
    /// - Free capacity is greater or equal to capacity *
    /// 'memoryPoolMinFreeCapacityPct'
    /// - Free capacity is greater or equal to 'memoryPoolMinFreeCapacity'
    ///
    /// NOTE: In the conditions when original requested shrink bytes ends up
    /// with more free capacity than above 2 conditions, the adjusted shrink
    /// bytes is not respected.
    ///
    /// NOTE: Capacity shrink adjustment is enabled when both
    /// 'memoryPoolMinFreeCapacityPct' and 'memoryPoolMinFreeCapacity' are set.
    static constexpr std::string_view kMemoryPoolMinFreeCapacity{
        "memory-pool-min-free-capacity"};
    static constexpr std::string_view kDefaultMemoryPoolMinFreeCapacity{
        "128MB"};
    static uint64_t getMemoryPoolMinFreeCapacity(
        const std::unordered_map<std::string, std::string>& configs);

    static constexpr std::string_view kMemoryPoolMinFreeCapacityPct{
        "memory-pool-min-free-capacity-pct"};
    static constexpr double kDefaultMemoryPoolMinFreeCapacityPct{0.25};
    static double getMemoryPoolMinFreeCapacityPct(
        const std::unordered_map<std::string, std::string>& configs);

    /// If true, it allows memory arbitrator to reclaim used memory cross query
    /// memory pools.
    static constexpr std::string_view kGlobalArbitrationEnabled{
        "global-arbitration-enabled"};
    static constexpr bool kDefaultGlobalArbitrationEnabled{false};
    static bool getGlobalArbitrationEnabled(
        const std::unordered_map<std::string, std::string>& configs);

    /// When growing capacity, the growth bytes will be adjusted in the
    /// following way:
    ///  - If 2 * current capacity is less than or equal to
    ///    'fastExponentialGrowthCapacityLimit', grow through fast path by at
    ///    least doubling the current capacity, when conditions allow (see below
    ///    NOTE section).
    ///  - If 2 * current capacity is greater than
    ///    'fastExponentialGrowthCapacityLimit', grow through slow path by
    ///    growing capacity by at least 'slowCapacityGrowPct' * current capacity
    ///    if allowed (see below NOTE section).
    ///
    /// NOTE: If original requested growth bytes is larger than the adjusted
    /// growth bytes or adjusted growth bytes reaches max capacity limit, the
    /// adjusted growth bytes will not be respected.
    ///
    /// NOTE: Capacity growth adjust is only enabled if both
    /// 'fastExponentialGrowthCapacityLimit' and 'slowCapacityGrowPct' are set,
    /// otherwise it is disabled.
    static constexpr std::string_view kFastExponentialGrowthCapacityLimit{
        "fast-exponential-growth-capacity-limit"};
    static constexpr std::string_view
        kDefaultFastExponentialGrowthCapacityLimit{"512MB"};
    static uint64_t getFastExponentialGrowthCapacityLimitBytes(
        const std::unordered_map<std::string, std::string>& configs);

    static constexpr std::string_view kSlowCapacityGrowPct{
        "slow-capacity-grow-pct"};
    static constexpr double kDefaultSlowCapacityGrowPct{0.25};
    static double getSlowCapacityGrowPct(
        const std::unordered_map<std::string, std::string>& configs);

    /// If true, do sanity check on the arbitrator state on destruction.
    ///
    /// TODO: deprecate this flag after all the existing memory leak use cases
    /// have been fixed.
    static constexpr std::string_view kCheckUsageLeak{"check-usage-leak"};
    static constexpr bool kDefaultCheckUsageLeak{true};
    static bool getCheckUsageLeak(
        const std::unordered_map<std::string, std::string>& configs);
  };

  explicit SharedArbitrator(const Config& config);

  ~SharedArbitrator() override;

  static void registerFactory();

  static void unregisterFactory();

  void addPool(const std::shared_ptr<MemoryPool>& memPool) final;

  void removePool(MemoryPool* memPool) final;

  bool growCapacity(MemoryPool* memPool, uint64_t requestBytes) final;

  uint64_t shrinkCapacity(MemoryPool* memPool, uint64_t requestBytes = 0) final;

  uint64_t shrinkCapacity(
      uint64_t requestBytes,
      bool allowSpill = true,
      bool force = false) override final;

  Stats stats() const final;

  std::string kind() const override;

  std::string toString() const final;

  /// Returns 'freeCapacity' back to the arbitrator for testing.
  void testingFreeCapacity(uint64_t freeCapacity);

  uint64_t testingNumRequests() const;

  /// Enables/disables global arbitration accordingly.
  void testingSetGlobalArbitration(bool enableGlobalArbitration) {
    *const_cast<bool*>(&globalArbitrationEnabled_) = enableGlobalArbitration;
  }

  /// Operator level runtime stats reported for an arbitration operation
  /// execution.
  static inline const std::string kMemoryArbitrationWallNanos{
      "memoryArbitrationWallNanos"};
  static inline const std::string kGlobalArbitrationCount{
      "globalArbitrationCount"};
  static inline const std::string kLocalArbitrationCount{
      "localArbitrationCount"};
  static inline const std::string kLocalArbitrationWaitWallNanos{
      "localArbitrationWaitWallNanos"};
  static inline const std::string kGlobalArbitrationWaitWallNanos{
      "globalArbitrationWaitWallNanos"};

 private:
  using ArbitrationId = uint64_t;

  // The kind string of shared arbitrator.
  inline static const std::string kind_{"SHARED"};

  // Used to start and finish an arbitration operation initiated from a memory
  // pool or memory capacity shrink request sent through shrinkPools() API.
  class ScopedArbitration {
   public:
    explicit ScopedArbitration(
        SharedArbitrator* arbitrator,
        ArbitrationOperation* op);

    ~ScopedArbitration();

   private:
    SharedArbitrator* arbitrator_;
    ArbitrationOperation* const operation_;
    const ScopedMemoryArbitrationContext arbitrationCtx_;
    const std::chrono::steady_clock::time_point startTime_;
  };

  ScopedArbitrationPool getPool(const std::string& name) const;

  ArbitrationOperation createArbitrationOperation(
      MemoryPool* memPool,
      uint64_t requestBytes);

  void startArbitration(ArbitrationOperation* op);

  void finishArbitration(ArbitrationOperation* op);

  // Invoked to check if the memory growth will exceed the memory pool's max
  // capacity limit or the arbitrator's node capacity limit.
  bool checkCapacityGrowth(ArbitrationOperation& op) const;

  // Invoked to ensure the memory growth request won't exceed the request memory
  // pool's max capacity as well as the arbitrator's node capacity. If it does,
  // then we first need to reclaim the used memory from the request memory pool
  // itself to ensure the memory growth won't exceed the capacity limit, and
  // then proceed with the memory arbitration process across queries.
  bool ensureCapacity(ArbitrationOperation& op);

  // Invoked to run local arbitration on the request memory pool. It first
  // ensures the memory growth is within both memory pool and arbitrator
  // capacity limits. This step might reclaim the used memory from the request
  // memory pool itself. Then it tries to obtain free capacity from the
  // arbitrator. At last, it tries to reclaim free memory from itself before it
  // falls back to the global arbitration. The local arbitration run is
  // protected by shared lock of 'arbitrationLock_' which can run in parallel
  // for different query pools. The free memory reclamation is protected by
  // arbitrator 'mutex_' which is an in-memory fast operation. The function
  // returns false on failure. Otherwise, it needs to further check if
  // 'needGlobalArbitration' is true or not. If true, needs to proceed with the
  // global arbitration run.

  void startArbitrationThread();
  void stopArbitrationThread();
  void wakeupArbitrationThread();

  void arbitrationThreadRun();

  uint64_t getArbitrationTarget();

  uint64_t getArbitrationTargetLocked();

  void runArbitrationLoop();

  // Invoked to run global arbitration to reclaim free or used memory from the
  // other queries. The global arbitration run is protected by the exclusive
  // lock of 'arbitrationLock_' for serial execution. The function returns true
  // on success, false on failure.
  bool startAndWaitArbitration(ArbitrationOperation& op);

  // Gets the mim/max memory capacity growth targets for 'op'. The min and max
  // targets are calculated based on memoryPoolReservedCapacity_ requirements
  // and the pool's max capacity.
  void getGrowTargets(
      ArbitrationOperation& op,
      uint64_t& maxGrowTarget,
      uint64_t& minGrowTarget);

  // Invoked to get or refresh the candidate memory pools for arbitration. If
  // 'freeCapacityOnly' is true, then we only get free capacity stats for each
  // candidate memory pool.
  std::vector<ArbitrationCandidate> getCandidates(
      bool freeCapacityOnly = false);

  // Sorts 'candidates' based on reclaimable free capacity in descending order.
  static void sortCandidatesByReclaimableFreeCapacity(
      std::vector<ArbitrationCandidate>& candidates);

  // Sorts 'candidates' based on reclaimable used capacity in descending order.
  static void sortCandidatesByReclaimableUsedCapacity(
      std::vector<ArbitrationCandidate>& candidates);

  // Sorts 'candidates' based on actual used memory in descending order.
  static void sortCandidatesByUsage(
      std::vector<ArbitrationCandidate>& candidates);

  // Finds the candidate with the largest capacity. For 'requestor', the
  // capacity for comparison including its current capacity and the capacity to
  // grow.
  ArbitrationCandidate findCandidateWithLargestCapacity();

  ArbitrationCandidate findCandidateWithLargestReclaimableBytes();

  // Invoked to reclaim unused memory capacity from memory pools without
  // actually freeing used memory.
  uint64_t reclaimUnusedCapacity();

  // Invoked to reclaim used memory capacity from 'candidates' by spilling.
  //
  // NOTE: the function might sort 'candidates' based on each candidate's
  // reclaimable memory internally.
  uint64_t reclaimUsedMemoryBySpill(uint64_t targetBytes);

  uint64_t reclaimUsedMemoryBySpill(
      ScopedArbitrationPool& pool,
      uint64_t targetBytes);

  // Invoked to handle the memory arbitration failure to abort the memory pool
  // with the largest capacity to free up memory. The function returns true on
  // success and false if the requestor itself has been selected as the
  // victim. We don't abort the requestor itself but just fails the
  // arbitration to let the user decide to either proceed with the query or
  // fail it.
  uint64_t reclaimUsedMemoryByAbort();

  uint64_t reclaimUsedMemoryByAbort(ScopedArbitrationPool& pool);

  // Checks if request pool has been aborted or not.
  void checkIfAborted(ArbitrationOperation& op);

  // Checks if the operation has timed out or not.
  void checkIfTimeout(ArbitrationOperation& op);

  // Checks if the request pool already has enough free capacity for the growth.
  // This could happen if there are multiple arbitration operations from the
  // same query. When the first served operation succeeds, it might have
  // reserved enough capacity for the followup operations.
  bool maybeGrowFromSelf(ArbitrationOperation& op);

  bool growWithFreeCapacity(ArbitrationOperation& op);

  // Invoked to grow 'pool' capacity by 'growBytes' and commit used reservation
  // by 'reservationBytes'. The function throws if the growth fails.
  void checkedGrow(
      ScopedArbitrationPool& pool,
      uint64_t growBytes,
      uint64_t reservationBytes);

  // Invoked to reclaim used memory from 'targetPool' with specified
  // 'targetBytes'. The function returns the actually freed capacity.
  // 'isLocalArbitration' is true when the reclaim attempt is within a local
  // arbitration.
  uint64_t reclaim(
      ScopedArbitrationPool& pool,
      uint64_t targetBytes,
      uint64_t timeoutMs,
      bool selfReclaim) noexcept;

  // Invoked to abort memory 'pool'.
  uint64_t abort(ScopedArbitrationPool& pool, const std::exception_ptr& error);

  // Decrements free capacity from the arbitrator with up to
  // 'maxBytesToReserve'. The arbitrator might have less free available
  // capacity. The function returns the actual decremented free capacity
  // bytes. If 'minBytesToReserve' is not zero and there is less than
  // 'minBytes' available in non-reserved capacity, then the arbitrator tries
  // to decrement up to 'minBytes' from the reserved capacity.
  bool allocateCapacity(
      uint64_t requestPoolId,
      uint64_t requestBytes,
      uint64_t maxAllocateBytes,
      uint64_t minAllocateBytes,
      uint64_t& allocatedBytes);

  bool allocateCapacityLocked(
      uint64_t requestPoolId,
      uint64_t requestBytes,
      uint64_t maxAllocateBytes,
      uint64_t minAllocateBytes,
      uint64_t& allocatedBytes);

  // Increment free capacity by 'bytes'.
  void freeCapacity(uint64_t bytes);

  void freeCapacityLocked(
      uint64_t bytes,
      std::vector<ContinuePromise>& resumes);
  // Increments the free reserved capacity up to 'bytes' until reaches to the
  // reserved capacity limit. 'bytes' is updated accordingly.
  void freeReservedCapacityLocked(uint64_t& bytes);

  void resumeArbitrationWaiters();
  void resumeArbitrationWaitersLocked(std::vector<ContinuePromise>& resumes);

  void removeArbitrationWaiter(uint64_t id);

  void incrementGlobalArbitrationCount();
  void incrementLocalArbitrationCount();

  std::string toStringLocked() const;

  Stats statsLocked() const;

  void updateArbitrationRequestStats();

  void updateArbitrationFailureStats();

  const uint64_t reservedCapacity_;
  const uint64_t memoryPoolInitialCapacity_;
  const uint64_t memoryPoolTransferCapacity_;
  const uint64_t memoryArbitrationTimeMs_;
  const ArbitrationPool::Config poolConfig_;
  const bool globalArbitrationEnabled_;
  const bool checkUsageLeak_;

  std::atomic_uint64_t nextPoolId_{0};

  mutable folly::SharedMutex poolLock_;
  std::unordered_map<std::string, std::shared_ptr<ArbitrationPool>>
      arbitrationPools_;

  // Lock used to protect the arbitrator internal state.
  mutable std::mutex stateLock_;

  bool shutdown_{false};
  std::condition_variable arbitrationThreadCv_;
  std::unique_ptr<std::thread> arbitrationThread_;

  tsan_atomic<uint64_t> freeReservedCapacity_{0};
  tsan_atomic<uint64_t> freeNonReservedCapacity_{0};

  std::unique_ptr<std::thread> globalArbitrationThread_;

  struct ArbitrationWait {
    ArbitrationOperation* op;
    ContinuePromise resumePromise;

    ArbitrationWait(ArbitrationOperation* _op, ContinuePromise _resumePromise)
        : op(_op), resumePromise(std::move(_resumePromise)) {}
  };
  std::map<uint64_t, ArbitrationWait> arbitrationWaiters_;

  std::atomic_uint64_t numRequests_{0};
  std::atomic_uint32_t numPending_{0};
  tsan_atomic<uint64_t> numAborted_{0};
  std::atomic_uint64_t numFailures_{0};
  std::atomic_uint64_t waitTimeUs_{0};
  tsan_atomic<uint64_t> arbitrationTimeUs_{0};
  std::atomic_uint64_t reclaimedFreeBytes_{0};
  std::atomic_uint64_t reclaimedUsedBytes_{0};
  std::atomic_uint64_t reclaimTimeUs_{0};
  tsan_atomic<uint64_t> numNonReclaimableAttempts_{0};
  tsan_atomic<uint64_t> numShrinks_{0};

  friend class ArbitrationOperation;
};
} // namespace facebook::velox::memory
