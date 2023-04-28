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

namespace facebook::velox::memory {

class MemoryPool;

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
  /// Defines the kind of memory arbitrators.
  enum class Kind {
    /// Used to enforce the fixed query memory isolation across running queries.
    /// When a memory pool exceeds the fixed capacity limit, the arbitrator
    /// fails the query with memory capacity exceeded error. This is used to
    /// implement the memory isolation mechanism currently used by Prestissimo.
    ///
    /// TODO: deprecate this legacy policy with kShared policy for Prestissimo
    /// later.
    kFixed,
    /// Used to achieve dynamic memory sharing among running queries. When a
    /// memory pool exceeds its current memory capacity, the arbitrator tries to
    /// grow its capacity by reclaim the overused memory from the query with
    /// more memory usage. We can configure memory arbitrator the way to reclaim
    /// memory. For Prestissimo, we can configure it to reclaim memory by
    /// aborting a query. For Prestissimo-on-Spark, we can configure it to
    /// reclaim from a running query through techniques such as disk-spilling,
    /// partial aggregation or persistent shuffle data flushes.
    kShared,
  };

  struct Config {
    Kind kind;
    /// The total memory capacity in bytes of all the running queries.
    ///
    /// NOTE: this should be same capacity as we set in the associated memory
    /// manager.
    int64_t capacity;
  };
  static std::unique_ptr<MemoryArbitrator> create(const Config& config);

  Kind kind() const {
    return kind_;
  }
  static std::string kindString(Kind kind);

  virtual ~MemoryArbitrator() = default;

  /// Indicates if this arbitrator allows to grow a memory pool's capacity or
  /// not.
  virtual bool canGrow() = 0;

  /// Invoked by the memory manager to reserve up to 'bytes' memory capacity
  /// without actually freeing memory for a newly created memory pool. The
  /// function returns the actual reserved memory capacity in bytes.
  ///
  /// NOTE: the memory arbitrator can decides how much memory capacity is
  /// actually reserved for a newly created memory pool. The latter can trigger
  /// the memory arbitration on demand when actual memory allocation happens.
  virtual int64_t reserveMemory(int64_t bytes) = 0;

  /// Invoked by the memory manager to grow a memory pool's capacity.
  /// 'requestor' is the memory pool to request to grow. 'candidates' is a list
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
      MemoryPool* requestor,
      const std::vector<MemoryPool*>& candidates,
      uint64_t targetBytes) = 0;

  /// Invoked by the memory manager to return back the reserved memory capacity
  /// of a destroying memory pool.
  virtual void releaseMemory(MemoryPool* releasor) = 0;

  /// The internal execution stats of the memory arbitrator.
  struct Stats {
    /// The number of arbitration requests.
    uint64_t numRequests{0};
    /// The number of arbitration request failures.
    uint64_t numFailures{0};
    /// The number of arbitration requests that has been queued waiting for
    /// execution.
    uint64_t numQueuedRequests{0};
    /// The sum of all the arbitration request queue times in microseconds.
    uint64_t queueTimeUs{0};
    /// The sum of all the arbitration run times in microseconds.
    uint64_t arbitrationTimeUs{0};
    /// The amount of memory bytes freed by reducing the memory pool's capacity
    /// without actually freeing memory.
    uint64_t numShrunkBytes{0};
    /// The amount of memory bytes freed by memory reclamation.
    uint64_t numReclaimedBytes{0};

    /// Returns the debug string of this stats.
    std::string toString() const;
  };
  virtual Stats stats() const = 0;

  /// Returns the debug string of this memory arbitrator.
  virtual std::string toString() const = 0;

 protected:
  explicit MemoryArbitrator(const Config& config)
      : kind_(config.kind), capacity_(config.capacity) {}

  const Kind kind_;
  const uint64_t capacity_;

  Stats stats_;
};

std::ostream& operator<<(std::ostream& out, const MemoryArbitrator::Kind& kind);

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

  static std::shared_ptr<MemoryReclaimer> create();

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

  /// Invoked by the memory arbitrator to check if the associated memory 'pool'
  /// can reclaim memory or not. As for now, only spillable operator's memory
  /// pool can reclaim memory. We will support to reclaim memory from partial
  /// aggregation and persistent shuffle writer by flushing the buffered data.
  virtual bool canReclaim(const MemoryPool& pool) const;

  /// Invoked by the memory arbitrator to get the amount of memory bytes that
  /// can be reclaimed from 'pool'.
  virtual uint64_t reclaimableBytes(const MemoryPool& pool) const;

  /// Invoked by the memory arbitrator to reclaim from memory 'pool' with
  /// specified 'targetBytes'. It is expected to reclaim at least that amount of
  /// memory bytes but there is no guarantees. If 'targetBytes' is zero, then it
  /// reclaims all the reclaimable memory from the memory 'pool'. The function
  /// returns the actual reclaimed memory bytes.
  virtual uint64_t reclaim(MemoryPool* pool, uint64_t targetBytes);

 protected:
  MemoryReclaimer() = default;
};
} // namespace facebook::velox::memory
