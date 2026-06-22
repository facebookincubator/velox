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

#include <memory>
#include <optional>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/Operator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
#include "velox/exec/rpc/RPCState.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

namespace facebook::velox::exec::rpc {

/// Single-operator implementation for async RPC execution in Velox.
///
/// Handles both dispatch (send RPCs in addInput()) and join (receive results
/// in isBlocked()/getOutput()) within the same operator.
///
/// Architecture:
///   TableScan -> RPCOperator -> downstream
///
/// The operator lifecycle:
/// 1. addInput(): Reads pre-computed argument columns, dispatches RPCs via
///    the function's dispatchPerRow() or accumulateBatch()+flushBatch().
/// 2. needsInput(): Returns false when under backpressure or when there are
///    ready results to output (prioritizes outputting over accepting input).
/// 3. isBlocked(): Checks RPCState for completed responses. Returns
///    kWaitForRPC when no results are ready yet.
/// 4. getOutput(): Builds output RowVector from completed RPC responses
///    combined with preserved input (passthrough) columns.
/// 5. noMoreInput(): In BATCH mode, flushes remaining accumulated rows.
///    Signals RPCState that no more rows will be dispatched.
///
/// Supports two streaming modes:
/// - PER_ROW: Rows emitted as individual RPCs complete (out-of-order).
///   Lower tail latency for high-variance workloads (e.g., LLM inference).
/// - BATCH: All rows in a batch complete before emitting. Lower overhead
///   for uniform-latency workloads. Supports pipelined dispatch via
///   dispatchBatchSize.
///
/// State is derived from data presence (no explicit state machine enum):
/// - Has output: claimedRows_ non-empty or claimedBatch_ has value
/// - Finished: noMoreInput_ && state_->isFinished() && no claimed data
///
/// Thread safety model:
/// - addInput(), getOutput(), needsInput(), isBlocked() are called from
///   a single driver thread (Velox guarantee). No synchronization needed
///   for operator-local state (e.g., globalRowIdCounter_, claimedRows_).
/// - Async RPC callbacks may run on any thread (transport executor pool).
///   All cross-thread coordination goes through RPCState, which is fully
///   mutex-protected (see RPCState.h for per-method annotations).
/// - RPCRateLimiter tokens use RAII: destruction (including from cancelled
///   futures) automatically decrements the pending count and notifies
///   waiters.
///
class RPCOperator : public exec::Operator {
 public:
  RPCOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::RPCNode> rpcNode);

  void initialize() override;

  void close() override;

  bool needsInput() const override;

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  bool startDrain() override;

  /// Runtime stat names.
  static inline const std::string kRpcRequestsDispatched{
      "rpcRequestsDispatched"};
  static inline const std::string kRpcResponsesReceived{"rpcResponsesReceived"};
  static inline const std::string kRpcErrorCount{"rpcErrorCount"};
  static inline const std::string kRpcWaitWallNanos{"rpcWaitWallNanos"};
  static inline const std::string kRpcBackpressureWaitNanos{
      "rpcBackpressureWaitNanos"};

 private:
  /// Flush accumulated batch rows via function_->flushBatch().
  /// Called when threshold is reached or at noMoreInput/drain time.
  /// @param maxRows Maximum rows to flush. 0 means flush all.
  void flushBatchRequests(int32_t maxRows = 0);

  /// Build output RowVector from ready rows (PER_ROW mode).
  /// Supports multiple rows via batched drain for pipeline efficiency.
  RowVectorPtr buildOutputFromReadyRows(
      std::vector<RPCState::ReadyRow>& readyRows);

  /// Build output RowVector from a ready batch (BATCH mode).
  RowVectorPtr buildOutputFromReadyBatch(RPCState::ReadyBatch& readyBatch);

  /// Common helper: build output vector from responses + input data lookup.
  RowVectorPtr buildOutputVector(
      const std::vector<RPCResponse>& responses,
      const std::vector<std::pair<int32_t, vector_size_t>>& locations);

  /// Precompute output column projections from source type to output type.
  /// Called once in initialize() to avoid repeated string lookups in
  /// buildOutputVector().
  void initOutputProjections();

  /// Record runtime stats into operator stats. Called from close().
  void recordRuntimeStats();

  std::shared_ptr<const core::RPCNode> rpcNode_;
  std::shared_ptr<RPCState> state_;
  std::shared_ptr<AsyncRPCFunction> function_;

  // Tier key for per-tier rate limiting (from function_->tierKey()).
  std::string tierKey_;

  // Precomputed argument column indices for reading from input in addInput().
  // Initialized in initialize() by looking up argumentColumns in source type.
  std::vector<column_index_t> argumentColumnIndices_;

  // Collected row locations for current batch (BATCH mode).
  // Passed to addPendingBatch() when the batch is flushed.
  std::vector<RPCState::RowLocation> batchRowLocations_;

  // Collected row IDs for current batch (BATCH mode).
  // Used to stamp rowIds onto responses at flush time.
  std::vector<int64_t> batchRowIds_;

  int64_t numRequestsDispatched_{0};
  int64_t numResponsesCollected_{0};
  int64_t numErrors_{0};

  // Global row ID counter for unique IDs across all input batches.
  int64_t globalRowIdCounter_{0};

  // Dispatch batch size for pipelined BATCH mode.
  // 0 = collect all rows, fire once in noMoreInput().
  // > 0 = fire flushBatch() every N rows during addInput().
  int32_t dispatchBatchSize_{0};

  // Claimed rows/batch from isBlocked() for use in getOutput().
  // State is derived from these: if non-empty, we have output ready.
  std::vector<RPCState::ReadyRow> claimedRows_;
  std::optional<RPCState::ReadyBatch> claimedBatch_;

  // Whether we've detected the finish condition.
  bool finished_{false};

  // Timeout for batch RPC calls (30 minutes).
  // This is a ceiling — the operator returns as soon as results are ready.
  // Batch LLM inference can take many minutes due to MetaGen queuing
  // and GPU scheduling, so the timeout needs generous headroom.
  static constexpr auto kBatchRpcTimeout = std::chrono::milliseconds(3'600'000);

  // Block wait time tracking for runtime stats.
  std::optional<uint64_t> blockWaitStartNs_;
  bool blockWaitIsBackpressure_{false};
  uint64_t totalBlockWaitNanos_{0};
  uint64_t totalBackpressureWaitNanos_{0};

  // Reusable indices buffer for dictionary wrapping in single-batch output
  // path.
  BufferPtr reusableIndices_;

  // Precomputed output column projections (initialized in initialize()).
  // Maps output column index to source column index for passthrough columns.
  // Avoids repeated string-based column lookups in buildOutputVector().
  struct OutputProjection {
    column_index_t outputChannel;
    column_index_t sourceChannel;
  };
  std::vector<OutputProjection> passthroughProjections_;
  column_index_t rpcResultOutputChannel_{0};
};

} // namespace facebook::velox::exec::rpc
