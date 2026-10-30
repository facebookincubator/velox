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

#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

#include <cuda/stream>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>

namespace facebook::velox::ucx_exchange {

namespace detail {

/// Returns the row boundary for an equal partition. Kept as a small testable
/// helper because cudf::size_type is 32-bit and multiplying before widening
/// silently wraps for otherwise valid large tables.
cudf::size_type equalPartitionOffset(
    cudf::size_type numRows,
    size_t partition,
    size_t numPartitions);

/// Geometric retry policy used after a pre-partition allocation failure.
int64_t smallerBatchRowsAfterAllocationFailure(int64_t attemptedRows);

} // namespace detail

/// This is the cudf equivalent of the PartitionedOutput operator for cudf.
/// Instead of serializing and segmenting the partitioned data into an
/// OutputBuffer, the UcxPartitionedOutput operator transfers entire
/// cudf::packed_columns corresponding to CudfVectors to other workers.
class UcxPartitionedOutput : public exec::Operator,
                             public cudf_velox::NvtxHelper {
 public:
  // Default minimum rows to accumulate before flushing. Matches HTTP
  // PartitionedOutput's ~10,000 row target. The process-wide default comes from
  // CudfConfig::partitionedOutputBatchRows and may be overridden per query
  // using CudfConfig::kUcxPartitionedOutputBatchRows as the QueryConfig key.
  static constexpr int64_t kDefaultTargetRowsPerChunk = 10'000;

  /// @param eagerFlush Send every batch as soon as it arrives instead of
  /// waiting for kUcxPartitionedOutputBatchRows rows. LocalPlanner sets it
  /// below a small partial LIMIT; the partitioned_output_eager_flush session
  /// property has the same effect.
  /// @param queueManager Output queue manager the partitions are enqueued to.
  /// Comes from the same exec::OutputTransportEntry that builds this operator,
  /// and must be the process-wide instance used by UcxExchangeServer. Held
  /// weakly, matching exec::PartitionedOutput.
  UcxPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush,
      const std::shared_ptr<UcxOutputQueueManager>& queueManager);

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr. The action is to further process
  /// unprocessed input. If all input has been processed, 'this' is in
  /// a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  void close() override;

  /// The caller checks isBlocked before adding input, so this only needs to
  /// report whether the operator can still accept rows once unblocked.
  bool needsInput() const override {
    return !isTaskCancelled() && !finished_ && !noMoreInput_ &&
        !shouldDrainPending() &&
        blockingReason_ == exec::BlockingReason::kNotBlocked;
  }

  // Returns kWaitForConsumer when output queues are full.
  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  // The operaor is finished when the queue manager say the queues have all been
  // drained ?
  bool isFinished() override;

 private:
  std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
  sharedQueueManager();

  bool usesHashPartitioning() const;

  // Heuristic method to derive the partition keys from the PartitionNode
  // specification.
  void initPartitionKeys(
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  // Partitions the cudf table view using the partition keys and a hash
  // function using the given stream.
  void hashPartition(
      cudf::table_view tableView,
      cuda::stream_ref stream,
      int64_t estimatedBytes);

  // Splits the cudf table view into equal sizes. This is used when
  // RoundRobin partitioning is requested but round robin on a
  // row-by-row basis is not meaningful for UCX exchange.
  void equalPartition(
      cudf::table_view tableView,
      cuda::stream_ref stream,
      std::unique_ptr<cudf::table>& tableOwner,
      std::vector<cudf_velox::CudfVectorPtr>& vectorOwners,
      int64_t estimatedBytes);

  struct PendingPartitionedBatch {
    std::unique_ptr<cudf::table> tableOwner;
    std::vector<cudf_velox::CudfVectorPtr> vectorOwners;
    cudf::table_view tableView;
    std::vector<cudf::size_type> offsets;
    int64_t estimatedBytes{0};
    bool conservativeChunkSizing{false};
    cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
    size_t nextPartition{0};
    std::vector<cudf::size_type> nextRows;
    std::vector<uint64_t> nextPayloadBytes;
    std::vector<uint64_t> drainDeficits;
    size_t remainingPartitions{0};
  };

  struct PendingInputSegment {
    cudf_velox::CudfVectorPtr owner;
    vector_size_t nextRow{0};
    vector_size_t remainingRows{0};
    int64_t remainingEstimatedBytes{0};
  };

  struct PendingInputBatch {
    std::unique_ptr<cudf::table> tableOwner;
    std::vector<cudf_velox::CudfVectorPtr> vectorOwners;
    /// Exact source slices consumed when this batch was formed. They are kept
    /// only until partitioning publishes successor state, allowing a
    /// pre-publication allocation failure to restore the input transaction and
    /// retry fewer rows without loss or duplication.
    std::vector<PendingInputSegment> recoverySegments;
    cudf::table_view tableView;
    int64_t numRows{0};
    int64_t estimatedBytes{0};
    cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
  };

  struct PendingReplicatedBatch {
    std::unique_ptr<cudf::table> tableOwner;
    std::vector<cudf_velox::CudfVectorPtr> vectorOwners;
    cudf::table_view tableView;
    int64_t estimatedBytes{0};
    cuda::stream_ref stream{cudaStream_t{cudaStreamDefault}};
    size_t nextDestination{0};
    cudf::size_type nextRow{0};
    uint64_t nextPayloadBytes{0};
  };

  void partitionPendingInputBatch();

  /// Separates the any-null rows and one arbitrary row from the ordinarily
  /// routed rows. Returns true after installing replication state; 'input' may
  /// be replaced or reduced to the routed suffix, so the caller must return to
  /// the state-machine loop.
  bool prepareReplicateNullsAndAny(PendingInputBatch& input);

  bool drainPendingReplicatedBatch();

  bool drainPendingPartitionedBatch();

  bool tryDrainWithFullContiguousSplit(PendingPartitionedBatch& batch);

  bool tryDrainWithContiguousSplit(PendingPartitionedBatch& batch);

  void initializePartitionDrainState(PendingPartitionedBatch& batch) const;

  uint64_t estimatedPartitionBytes(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end) const;

  uint64_t exactPartitionBytes(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end) const;

  cudf::size_type rowsForPayloadTarget(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end,
      uint64_t targetBytes) const;

  bool shouldSplitPayload(
      const PendingPartitionedBatch& batch,
      cudf::size_type start,
      cudf::size_type end) const;

  uint64_t baseTransferWindowBytes(const PendingPartitionedBatch& batch) const;

  uint64_t normalTransferWindowBytes(
      const PendingPartitionedBatch& batch) const;

  uint64_t maxTransferWindowBytes(const PendingPartitionedBatch& batch) const;

  uint64_t transferWindowBytes(
      const PendingPartitionedBatch& batch,
      int destination,
      const std::shared_ptr<UcxOutputQueueManager>& queueManager) const;

  /// Applies flow-control backoff after allocation pressure. Returns true only
  /// when retained output registered a real wakeup for this operator. A false
  /// result must be converted to a terminal allocation failure; retrying it on
  /// the driver thread would busy-spin without any state change.
  bool recordAllocationPressure(uint64_t waitBytes);

  /// Lowers the persistent row cap after a recoverable, pre-publication
  /// allocation failure. Returns false once a one-row attempt is reached.
  bool reduceAllocationBatchRows(int64_t attemptedRows);

  /// Reconstructs the pending-input deque from the current batch's exact
  /// source slices, then lowers the row cap for the next driver call.
  bool downsizePendingInputBatch();

  bool isTaskCancelled() const;

  bool maybeFinishCancelled();

  void clearPending();

  const std::weak_ptr<UcxOutputQueueManager> queueManager_;
  std::vector<column_index_t> partitionKeyIndices_;
  const size_t numPartitions_;
  /// Broadcast and arbitrary queues fan out/reroute a single destination-zero
  /// payload internally and therefore must not receive per-destination
  /// transfer reservations.
  const bool isPartitionedKind_;
  const bool replicateNullsAndAny_;

  // Set once the arbitrary row has been replicated. Matches
  // exec::PartitionedOutput: one arbitrary row is replicated once per
  // operator instance, rather than once per flush.
  bool replicatedAnyRow_{false};

  const int pipelineId_;
  const int driverId_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;

  bool finished_{false};
  std::string spec_;
  bool isGatherPartition_{false};

  // Used for switching columns when column order differs between input and
  // output.
  bool needsRemap_{false};
  std::vector<uint32_t> remap_;

  /// Concatenates pending inputs and partitions/enqueues the merged result.
  void flushPending();

  bool shouldFlushPending() const;

  bool shouldDrainPending() const;

  /// Unconsumed input segments awaiting promotion into an output batch. A
  /// segment may retain the suffix of a CudfVector after a bounded prefix has
  /// been promoted.
  std::deque<PendingInputSegment> pendingInputs_;
  /// Total unconsumed rows across pendingInputs_.
  int64_t pendingRows_{0};
  /// Total estimated unconsumed bytes across pendingInputs_.
  int64_t pendingBytes_{0};
  /// Configured minimum row threshold for flushing (from QueryConfig).
  const int64_t flushThresholdRows_;
  // Flush every batch when requested by the planner or query config.
  const bool eagerFlush_;
  /// Runtime row cap installed after a pre-partition allocation failure. A
  /// geometric reduction makes at most O(log(rows)) retries before either one
  /// row succeeds or the allocation becomes terminal.
  int64_t allocationRetryMaxRows_{0};
  /// Initial target GPU payload size for geometrically chunked UCX messages.
  const uint64_t initialPayloadBytes_;

  /// Partitioned table waiting to be packed and enqueued destination-by-
  /// destination. This avoids materializing the full fanout before UCX output
  /// backpressure can take effect.
  std::optional<PendingPartitionedBatch> pendingPartitionedBatch_;

  /// Rows replicated to every destination, packed incrementally under the
  /// same transfer reservations as ordinarily partitioned rows.
  std::optional<PendingReplicatedBatch> pendingReplicatedBatch_;

  /// Cudf input that has been materialized enough to retry partitioning after
  /// GPU allocation pressure without asking the upstream driver for the row
  /// again.
  std::optional<PendingInputBatch> pendingInputBatch_;
};

} // namespace facebook::velox::ucx_exchange
