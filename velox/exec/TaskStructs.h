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

#include "velox/common/base/SkewedPartitionBalancer.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/connectors/Connector.h"
#include "velox/exec/LocalPartition.h"
#include "velox/exec/ScaledScanController.h"
#include "velox/exec/Split.h"
#include "velox/exec/TaskStats.h"

#include <folly/CPortability.h>
#include <folly/container/F14Map.h>

#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace facebook::velox::exec {

class Driver;
class ExchangeClient;
class JoinBridge;
class LocalExchangeMemoryManager;
class MergeSource;
class MergeJoinSource;

/// Corresponds to Presto TaskState, needed for reporting query completion.
enum class TaskState : int {
  kRunning = 0,
  kFinished = 1,
  kCanceled = 2,
  kAborted = 3,
  kFailed = 4
};

std::string taskStateString(TaskState state);

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    TaskState state) {
  os << taskStateString(state);
  return os;
}

struct BarrierState {
  /// Tracks the number of peers that have entered the current barrier round.
  uint32_t numRequested{0};
  /// Tracks the number of unfinished peers expected in the current round.
  uint32_t numPeers{0};
  /// Identifies the pipeline that owns this barrier.
  std::optional<uint32_t> pipelineId;
  /// Holds drivers waiting at the current barrier round.
  std::vector<std::shared_ptr<Driver>> drivers;
  /// Promises given to non-last peer drivers. After the last driver completes
  /// the barrier's coordinated work, the promises are fulfilled and the
  /// non-last drivers can continue.
  std::vector<ContinuePromise> allPeersArrivedPromises;
};

using ConnectorSplitPreloadFunc =
    std::function<void(const std::shared_ptr<connector::ConnectorSplit>&)>;

/// A split store that either can accumulate splits through addSplit() or
/// generating splits from its own source.
///
/// This object is not multi-thread safe.
class SplitsStore {
 public:
  explicit SplitsStore(bool remoteSplit) : remoteSplit_(remoteSplit) {}

  virtual ~SplitsStore() = default;

  /// Add a barrier marker for each participating driver supplied by Task.
  /// Driver IDs are not necessarily contiguous. Waiters on nextSplit() are
  /// returned in `promises` for the caller to notify.
  ///
  /// `promises` should be set by caller (potentially outside a lock), to notify
  /// any waiters on the splits.
  virtual void requestBarrier(
      const std::vector<uint32_t>& driverIds,
      std::vector<ContinuePromise>& promises);

  /// Returns the next split to process.
  ///
  /// @param driverId The driver id requesting the split. Barrier splits are
  /// delivered exactly once for each driver.
  /// @param maxPreloadSplits Maximum number of splits to preload.
  /// @param preload Function to preload connector splits.
  /// @param split Output parameter for the next split.
  /// @param future Output parameter for the future to wait on if no split is
  /// available.
  /// @return true if a split is set or there are no more splits; false if the
  /// caller should retry when the future is fulfilled.
  virtual bool nextSplit(
      uint32_t driverId,
      int maxPreloadSplits,
      const ConnectorSplitPreloadFunc& preload,
      Split& split,
      ContinueFuture& future) = 0;

  /// Return whether all splits has been consumed and there will be no more
  /// splits.
  virtual bool allSplitsConsumed() const = 0;

  /// Add new split into this store.  Some of the waiters previously waiting on
  /// the ContinueFuture after the nextSplit() call should be notified by the
  /// caller of this function by setting values on the `promises` output
  /// parameter.
  ///
  /// `promises` should be set by caller (potentially outside a lock), to notify
  /// any waiters on the splits.
  void addSplit(Split split, std::vector<ContinuePromise>& promises);

  /// Marks all splits acquired by the specified consumer as fully processed.
  void finishSplits(uint32_t driverId);

  /// Records one consumer closing and abandons its unfinished splits.
  /// When all 'numConsumers' consumers have closed, abandons queued splits
  /// and returns them for disposal outside the task lock. Appends outstanding
  /// split waiters to 'promises'. Each consumer
  /// must call this exactly once. Consumers that have not started still count.
  std::deque<Split> finishConsumer(
      uint32_t driverId,
      uint32_t numConsumers,
      std::vector<ContinuePromise>& promises);

  /// Signals no more splits and abandons the queue. Returns queued splits for
  /// disposal outside the task lock and appends split waiters to 'promises'.
  /// Acquired splits remain owned by their consumers until completion or close.
  std::deque<Split> discardQueuedSplits(std::vector<ContinuePromise>& promises);

  bool allConsumersFinished() const {
    return allConsumersFinished_;
  }

  /// Discard the driver's pending marker for the current barrier, if any.
  void removeBarrier(uint32_t driverId);

  /// Returns the splits added but not yet returned by nextSplit().
  const std::deque<Split>& queuedSplits() const {
    return splits_;
  }

  /// Return the number of waiters waiting for new splits.
  int numWaiters() const {
    return promises_.size();
  }

  /// Signal there will not be any more splits added to this store.
  std::vector<ContinuePromise> noMoreSplits() {
    noMoreSplits_ = true;
    return std::move(promises_);
  }

  void setTaskStats(TaskStats& taskStats) {
    taskStats_ = &taskStats;
  }

  void setPreloadingSplits(
      folly::F14FastSet<std::shared_ptr<connector::ConnectorSplit>>&
          preloadingSplits) {
    preloadingSplits_ = &preloadingSplits;
  }

 protected:
  // Removes a queued split and records its count and weight for 'driverId'.
  Split getSplit(
      uint32_t driverId,
      int maxPreloadSplits,
      const ConnectorSplitPreloadFunc& preload);

  ContinueFuture makeFuture();

  bool tryGetBarrier(uint32_t driverId, Split& split);

  // Arrived (added), but not distributed yet, splits.
  std::deque<Split> splits_;
  // Signal, that no more splits will arrive.
  bool noMoreSplits_{false};

 private:
  // Removes a driver's acquired work from running counters and returns its
  // split count.
  int32_t releaseRunningSplits(uint32_t driverId);

  // Removes one split's count and weight from the queued counters.
  void removeQueuedSplitStats(const Split& split);

  const bool remoteSplit_;
  TaskStats* taskStats_{};
  folly::F14FastSet<std::shared_ptr<connector::ConnectorSplit>>*
      preloadingSplits_{};

  // The map from driver id to barrier splits.
  std::unordered_map<uint32_t, Split> barrierSplits_;

  // Blocking promises given out when out of splits to distribute.
  std::vector<ContinuePromise> promises_;

  // Work acquired by a consumer until it completes or the consumer closes.
  struct RunningSplits {
    int32_t numSplits{0};
    // Excludes weights of remote and index splits.
    int64_t weight{0};
  };
  // Prevents one consumer's closure from releasing another consumer's work.
  folly::F14FastMap<uint32_t, RunningSplits> runningSplits_;

  // Consumers that have closed for this source and split group.
  std::unordered_set<uint32_t> finishedConsumers_;

  // Prevents accepting more work after the last consumer closes.
  bool allConsumersFinished_{false};
};

/// Structure contains the current info on splits for a particular plan node.
struct SplitsState {
  /// True if the source node is a table scan.
  bool sourceIsTableScan{false};

  /// Plan node-wide 'no more splits'.
  bool noMoreSplits{false};

  /// Keep the max added split's sequence id to deduplicate incoming splits.
  long maxSequenceId{std::numeric_limits<long>::min()};

  /// Map split group id -> split store.
  std::unordered_map<uint32_t, std::unique_ptr<SplitsStore>> groupSplitsStores;

  /// Exchange client of a leaf node that requires one, other than a
  /// MergeExchangeNode. Set when the task creates the client. The task
  /// forwards remote splits and 'no more splits' to it as they arrive. Null
  /// for other plan nodes.
  std::shared_ptr<ExchangeClient> exchangeClient;

  /// We need these due to having promises in the structure.
  SplitsState() = default;
  SplitsState(SplitsState const&) = delete;
  SplitsState& operator=(SplitsState const&) = delete;
};

/// Stores local exchange queues with the memory manager.
struct LocalExchangeState {
  std::shared_ptr<LocalExchangeMemoryManager> memoryManager;
  std::shared_ptr<LocalExchangeVectorPool> vectorPool;
  std::vector<std::shared_ptr<LocalExchangeQueue>> queues;
  std::shared_ptr<common::SkewedPartitionRebalancer>
      scaleWriterPartitionBalancer;
};

/// Stores inter-operator state (exchange, bridges) for split groups.
struct SplitGroupState {
  /// Map from the plan node id of the join to the corresponding JoinBridge.
  /// This map will contain only HashJoinBridge and NestedLoopJoinBridge.
  std::unordered_map<core::PlanNodeId, std::shared_ptr<JoinBridge>> bridges;
  /// This map will contain all other custom bridges.
  std::unordered_map<core::PlanNodeId, std::shared_ptr<JoinBridge>>
      customBridges;
  /// Holds states for Task::allPeersArrived.
  std::unordered_map<core::PlanNodeId, BarrierState> barriers;
  /// Tracks the number of operators that won't participate in future barriers,
  /// keyed by plan node ID and pipeline ID.
  ///
  /// NOTE: 'pipelineId' is needed here even though build and probe barriers
  /// alternate. Unlike the per-round barrier state, the finished count persists
  /// across spill rounds. Both pipelines share the same 'planNodeId', so keying
  /// only by 'planNodeId' would let a finished probe reduce the next build
  /// barrier's expected peer count, causing it to complete too early.
  std::unordered_map<core::PlanNodeId, std::unordered_map<uint32_t, uint32_t>>
      finishedBarrierPeers;

  /// Map of merge sources keyed on LocalMergeNode plan node ID.
  std::
      unordered_map<core::PlanNodeId, std::vector<std::shared_ptr<MergeSource>>>
          localMergeSources;

  /// Map of merge join sources keyed on MergeJoinNode plan node ID.
  std::unordered_map<core::PlanNodeId, std::shared_ptr<MergeJoinSource>>
      mergeJoinSources;

  /// Map of local exchanges keyed on LocalPartition plan node ID.
  std::unordered_map<core::PlanNodeId, LocalExchangeState> localExchanges;

  /// Map of scaled scan controllers keyed on TableScan plan node ID.
  std::unordered_map<core::PlanNodeId, std::shared_ptr<ScaledScanController>>
      scaledScanControllers;

  /// Drivers created and still running for this split group.
  /// The split group is finished when this numbers reaches zero.
  uint32_t numRunningDrivers{0};

  /// The number of completed drivers in the output pipeline. When all drivers
  /// in the output pipeline finish, the remaining running pipelines should stop
  /// processing and transition to finished state as well. This happens when
  /// there is a downstream operator that finishes before receiving all input,
  /// e.g. Limit.
  uint32_t numFinishedOutputDrivers{0};

  // True if the state contains structures used for connecting ungrouped
  // execution pipeline with grouped execution pipeline. In that case we don't
  // want to clean up some of these structures.
  bool mixedExecutionMode{false};

  /// Clears the state.
  void clear() {
    if (!mixedExecutionMode) {
      bridges.clear();
      customBridges.clear();
      barriers.clear();
      finishedBarrierPeers.clear();
    }
    localMergeSources.clear();
    mergeJoinSources.clear();
    localExchanges.clear();
  }
};

} // namespace facebook::velox::exec

template <>
struct fmt::formatter<facebook::velox::exec::TaskState>
    : formatter<std::string> {
  auto format(facebook::velox::exec::TaskState state, format_context& ctx)
      const {
    return formatter<std::string>::format(
        facebook::velox::exec::taskStateString(state), ctx);
  }
};
