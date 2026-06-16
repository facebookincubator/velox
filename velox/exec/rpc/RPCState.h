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

#include <deque>
#include <mutex>
#include <optional>
#include <vector>

#include <folly/futures/Future.h>

#include "velox/common/future/VeloxPromise.h"
#include "velox/common/rpc/RPCTypes.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec::rpc {

// Import RPC types from velox/common/rpc into this namespace.
using velox::rpc::RPCResponse;
using velox::rpc::RPCStreamingMode;

/// A stored input batch with its rows referenced by index.
/// Instead of slicing individual rows, we keep the entire batch and
/// use row indices to look up passthrough columns at output time.
struct InputBatchRef {
  /// Flattened columns from the input batch (shared across all rows).
  std::vector<VectorPtr> flatColumns;

  /// Number of rows from this batch that are still in-flight or pending output.
  /// When this reaches 0, the batch can be released.
  int64_t activeRowCount{0};
};

/// Shared state between RPCOperator's driver thread and RPC completion
/// callbacks.
///
/// A mutex-protected state object owned by RPCOperator. It coordinates
/// async RPC dispatch (addPendingRow/addPendingBatch) with result
/// consumption (tryClaimOrWait/tryPollBatchOrWait) across the driver
/// thread and RPC client executor threads.
///
/// Thread safety: All public methods are thread-safe. The mutex_ protects
/// all mutable state. Completion callbacks from the RPC client's executor
/// threads call notifyWaitersLocked() to wake the driver thread.
///
/// Two streaming modes:
/// - PER_ROW: Rows are emitted as they complete individually (out-of-order).
///   Lower tail latency for high-variance workloads (e.g., LLM inference).
/// - BATCH: All rows in a batch complete before emitting. Lower overhead
///   for uniform-latency workloads.
class RPCState {
 public:
  // ===== Data structures =====

  /// Location of a row within an input batch.
  struct RowLocation {
    int32_t batchIndex{0};
    vector_size_t rowIndex{0};
  };

  /// A completed row with its RPC response and location in the input batch.
  struct ReadyRow {
    int64_t rowId{0};
    RowLocation location;
    RPCResponse response;
  };

  /// A batch of rows waiting for RPC response.
  struct PendingBatch {
    int64_t batchId;
    folly::SemiFuture<std::vector<RPCResponse>> future;
    /// Row locations for mapping responses back to input batch positions.
    /// Stored at batch level instead of per-row in a map, since callBatch()
    /// returns responses in the same order as requests.
    std::vector<RowLocation> rowLocations;
  };

  /// A batch with completed RPC responses.
  struct ReadyBatch {
    int64_t batchId{0};
    std::vector<RPCResponse> responses;
    std::optional<std::string> error;
    /// Row locations carried from PendingBatch for response-to-input mapping.
    std::vector<RowLocation> rowLocations;
  };

  RPCState() = default;

  // ===== Configuration =====
  // These must be called before any dispatch (single-threaded init phase).

  /// Set the streaming mode. Must be called before any dispatch.
  void setStreamingMode(RPCStreamingMode mode);

  /// Get the current streaming mode.
  RPCStreamingMode streamingMode() const;

  /// Set the maximum number of pending rows before backpressure.
  void setMaxPendingRows(int64_t maxPendingRows);

  /// Set the maximum number of pending batches before backpressure (BATCH
  /// mode).
  void setMaxPendingBatches(int64_t maxPendingBatches);

  // ===== Input batch storage =====
  // Called from the driver thread (addInput/getOutput). Thread-safe.

  /// Store an input batch and return its index. Thread-safe.
  /// The batch is reference-counted; call releaseRows() when rows are output.
  int32_t storeInputBatch(std::vector<VectorPtr> flatColumns, int64_t rowCount);

  /// Get the flattened columns for a stored input batch. Thread-safe.
  /// Returns by value to avoid returning a reference that outlives the lock.
  std::vector<VectorPtr> getInputBatchColumns(int32_t batchIndex) const;

  /// Release rows from an input batch. Thread-safe.
  /// When all rows are released, the batch columns are freed.
  void releaseRows(int32_t batchIndex, int64_t count);

  // ===== PER_ROW mode API =====

  /// Add a pending row with its RPC future and location in the input batch.
  /// Thread-safe. Called from the driver thread in addInput().
  ///
  /// Attaches a completion callback to the future that moves the response
  /// into readyRows_ and notifies waiting drivers. The callback runs on
  /// the RPC client's executor thread, acquiring mutex_ internally.
  ///
  /// @param selfPtr Shared pointer to this RPCState, captured by the callback
  ///        to prevent premature destruction.
  /// @param rowId Globally unique row ID for correlation.
  /// @param location Row's location in the stored input batch.
  /// @param future The SemiFuture from client->call().
  void addPendingRow(
      std::shared_ptr<RPCState> selfPtr,
      int64_t rowId,
      RowLocation location,
      folly::SemiFuture<RPCResponse> future);

  /// Atomically try to claim a ready row, check finish, or wait. Thread-safe.
  /// Called from the driver thread in isBlocked().
  ///
  /// All three checks happen under a single lock to prevent TOCTOU races.
  ///
  /// @param[out] future Set to a wait future if kMustWait.
  /// @param[out] claimedRow Set to the claimed row if kClaimed.
  /// @return kClaimed, kFinished, or kMustWait.
  enum class ClaimResult { kClaimed, kFinished, kMustWait };
  ClaimResult tryClaimOrWait(
      ContinueFuture* future,
      std::optional<ReadyRow>* claimedRow);

  /// Non-blocking claim of a ready row. Thread-safe.
  /// Returns nullopt if no ready rows.
  /// Unlike tryClaimOrWait(), does NOT create a promise on miss.
  /// Use this pre-noMoreInput to avoid accumulating orphaned promises.
  std::optional<ReadyRow> tryClaimReady();

  /// Drain all currently ready rows (non-blocking, up to maxRows). Thread-safe.
  /// Used for batched PER_ROW output to amortize RowVector allocation.
  void drainReadyRows(std::vector<ReadyRow>& out, int32_t maxRows);

  /// Returns the number of pending (in-flight) rows. Thread-safe.
  int64_t numPendingRows();

  // ===== BATCH mode API =====

  /// Add a pending batch future with row locations for response-to-input
  /// mapping. Thread-safe. Called from the driver thread in
  /// flushBatchRequests().
  ///
  /// Attaches a completion callback that notifies waiters on completion.
  /// The callback runs on the RPC client's executor thread, acquiring
  /// mutex_ internally.
  ///
  /// @param selfPtr Shared pointer to this RPCState (prevent destruction).
  /// @param future The SemiFuture from client->callBatch().
  /// @param rowLocations Locations mapping each request to its input batch
  ///        position. Stored on the PendingBatch and carried through to
  ///        ReadyBatch, eliminating per-row rowLocations_ map overhead.
  void addPendingBatch(
      std::shared_ptr<RPCState> selfPtr,
      folly::SemiFuture<std::vector<RPCResponse>> future,
      std::vector<RowLocation> rowLocations);

  /// Atomically try to poll a ready batch, check finish, or wait. Thread-safe.
  /// Called from the driver thread in isBlocked().
  ///
  /// @param[out] future Set to a wait future if kMustWait.
  /// @param[out] readyBatch Set to the ready batch if kGotBatch.
  /// @return kGotBatch, kFinished, or kMustWait.
  enum class BatchPollResult { kGotBatch, kFinished, kMustWait };
  BatchPollResult tryPollBatchOrWait(
      ContinueFuture* future,
      std::optional<ReadyBatch>* readyBatch);

  /// Non-blocking poll of a ready batch. Thread-safe.
  /// Returns nullopt if no batch ready.
  /// Unlike tryPollBatchOrWait(), does NOT create a promise on miss.
  /// Use this pre-noMoreInput to avoid accumulating orphaned promises.
  std::optional<ReadyBatch> tryPollReady();

  // ===== Common =====

  /// Signal that no more rows will be dispatched. Thread-safe.
  void setNoMoreInput();

  /// Returns true when all work is complete. Thread-safe.
  bool isFinished();

  /// Returns true if backpressure should be applied. Thread-safe.
  /// PER_ROW mode: pending rows >= maxPendingRows.
  /// BATCH mode: pending batches >= effectiveMaxPendingBatches
  /// (congestion-adjusted).
  bool isUnderBackpressure();

  /// Signal that a batch completed successfully (all responses non-empty).
  /// Increases the effective concurrency window by increment (additive
  /// increase). Thread-safe.
  void onBatchSuccess(int64_t increment = 2);

  /// Signal that a batch had errors (e.g., empty responses from overload).
  /// Halves the effective concurrency window (multiplicative decrease).
  /// Thread-safe.
  void onBatchError();

 private:
  /// Move a completed row into readyRows_ and notify waiters.
  /// Called from the RPC completion callback (runs on executor thread).
  void completeRow(int64_t rowId, RowLocation location, RPCResponse response);

  /// Fulfill all waiting promises and clear. Called under lock.
  void notifyWaitersLocked();

  mutable std::mutex mutex_;
  std::vector<ContinuePromise> promises_;

  // Input batch storage (shared across PER_ROW and BATCH modes)
  std::vector<InputBatchRef> inputBatches_;

  // PER_ROW state
  std::deque<ReadyRow> readyRows_;
  int64_t numPendingRows_{0};

  // BATCH state
  int64_t nextBatchId_{0};
  std::deque<PendingBatch> pendingBatches_;

  // Common
  bool noMoreInput_{false};
  RPCStreamingMode streamingMode_{RPCStreamingMode::kPerRow};
  int64_t maxPendingRows_{100};
  int64_t maxPendingBatches_{2};

  // Congestion control for BATCH mode.
  // effectiveMaxPendingBatches_ starts at maxPendingBatches_ and adjusts:
  //   - On success: min(effective + 1, maxPendingBatches_)  (additive increase)
  //   - On error:   max(effective / 2, 1)                   (multiplicative
  //   decrease)
  int64_t effectiveMaxPendingBatches_{2};
};

} // namespace facebook::velox::exec::rpc
