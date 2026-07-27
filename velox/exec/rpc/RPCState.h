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

#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <folly/futures/Future.h>

#include "velox/common/future/VeloxPromise.h"
#include "velox/common/rpc/RPCTypes.h"
#include "velox/exec/rpc/CongestionController.h"
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
    /// Round-trip latency (nanos) from dispatch to completion, used as the
    /// gradient congestion signal on success.
    int64_t rttNs{0};
  };

  /// A batch of rows waiting for RPC response.
  struct PendingBatch {
    int64_t batchId;
    folly::SemiFuture<std::vector<RPCResponse>> future;
    /// Row locations for mapping responses back to input batch positions.
    /// Stored at batch level instead of per-row in a map, since callBatch()
    /// returns responses in the same order as requests.
    std::vector<RowLocation> rowLocations;
    /// Monotonic dispatch time (nanos, steady_clock).
    int64_t dispatchTimeNs{0};
    /// Monotonic completion time (nanos), stamped by the future's completion
    /// callback (not at poll time) so the round-trip latency fed to the
    /// gradient excludes any in-operator queueing between completion and poll.
    /// Shared so the callback (executor thread) and the poll path (driver
    /// thread) refer to the same cell; 0 until the batch completes.
    std::shared_ptr<std::atomic<int64_t>> completionTimeNs;
  };

  /// A batch with completed RPC responses.
  struct ReadyBatch {
    int64_t batchId{0};
    std::vector<RPCResponse> responses;
    std::optional<std::string> error;
    /// Row locations carried from PendingBatch for response-to-input mapping.
    std::vector<RowLocation> rowLocations;
    /// Round-trip latency (nanos) from dispatch to completion, used as the
    /// gradient congestion signal on success.
    int64_t rttNs{0};
  };

  /// Snapshot of all operator-visible state at close() time, captured under a
  /// single lock acquisition for consistency.
  struct OperatorSnapshot {
    // Congestion controller.
    int64_t windowLimit{0};
    int64_t baselineRttNs{0};
    int64_t numShrinks{0};
    int64_t peakInFlight{0};
    // Transport RTT.
    int64_t rttMinNs{0};
    int64_t rttMaxNs{0};
    int64_t numRttSamples{0};
    // Streaming mode.
    RPCStreamingMode streamingMode{RPCStreamingMode::kPerRow};
  };

  RPCState() = default;

  // ===== Configuration =====
  // These must be called before any dispatch (single-threaded init phase).

  /// Set the streaming mode and the congestion-window tunables. Must be called
  /// before any dispatch.
  /// @param minWindow Floor the window may shrink to under overload (default
  /// 1).
  /// @param stepCoef Multiplier on the additive-increase headroom
  /// (default 1.0). Both default to the controller's defaults so callers and
  /// OSS behavior are unchanged; the RPCOperator threads runtime-tunable values
  /// from QueryConfig.
  /// @param maxWindow Ceiling for the congestion window (and, for PER_ROW, its
  /// starting value). 0 keeps the per-mode built-in ceiling (PER_ROW 100, BATCH
  /// 256). Raise it so a high-latency backend can run at high concurrency;
  /// admission-controlled dispatch makes this ceiling bind.
  void setStreamingMode(
      RPCStreamingMode mode,
      int64_t minWindow = 1,
      double stepCoef = 1.0,
      int64_t maxWindow = 0);

  /// Get the current streaming mode.
  RPCStreamingMode streamingMode() const;

  /// Set both the starting limit and the ceiling to maxWindow, overriding the
  /// per-mode default chosen in setStreamingMode(). This yields a fixed window
  /// only while no samples/errors are fed: onUnitError still halves it and
  /// onUnitSample still grows it via the sqrt headroom. That no-feed case is
  /// the deterministic window the unit tests rely on. Must be called AFTER
  /// setStreamingMode(), which always resets the window. Tests only; production
  /// relies on the per-mode defaults.
  void setMaxWindow(int64_t maxWindow);

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

  /// Returns the number of in-flight units of the active mode (rows in
  /// PER_ROW, batches in BATCH). Thread-safe.
  int64_t numInFlight() const;

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
  /// Both modes: in-flight units (rows for PER_ROW, batches for BATCH) >=
  /// the congestion window limit.
  bool isUnderBackpressure();

  /// Available dispatch headroom under the per-driver congestion window:
  /// max(0, window.limit() - inFlight). Admission-controlled dispatch takes the
  /// min of this and the process-global rate-limiter headroom to size each
  /// drip chunk, so a whole-vector blast can no longer overrun the window.
  /// Thread-safe.
  int64_t dispatchHeadroom();

  /// Report that a completed unit showed backend overload (rate limit /
  /// timeout). Multiplicative-decrease (halving) of the congestion window.
  /// Thread-safe.
  void onUnitError();

  /// Feed one completed unit's round-trip latency (nanos) into the
  /// latency-gradient window so it learns the in-flight sweet spot with no
  /// fixed ceiling. Thread-safe.
  void onUnitSample(int64_t rttNs);

  /// Feed many completed units' round-trip latencies under a single lock
  /// acquisition. The PER_ROW path drains up to ~1k rows per output and would
  /// otherwise lock once per row, contending with completion callbacks.
  /// Thread-safe.
  void onUnitSamples(const std::vector<int64_t>& rttNsList);

  /// Return a consistent snapshot of all operator-visible state under a single
  /// lock acquisition. Thread-safe.
  OperatorSnapshot operatorSnapshot() const;

 private:
  /// Move a completed row into readyRows_ and notify waiters.
  /// Called from the RPC completion callback (runs on executor thread).
  void completeRow(
      int64_t rowId,
      RowLocation location,
      RPCResponse response,
      int64_t rttNs);

  /// Fulfill all waiting promises and clear. Called under lock.
  void notifyWaitersLocked();

  /// Extract the ready batch referenced by `it`: compute its round-trip
  /// latency, move out the responses (capturing any error), erase the entry,
  /// and decrement inFlight_. Must be called under mutex_ with `it->future`
  /// ready.
  ReadyBatch extractReadyBatchLocked(
      const std::deque<PendingBatch>::iterator& it);

  mutable std::mutex mutex_;
  std::vector<ContinuePromise> promises_;

  // Input batch storage (shared across PER_ROW and BATCH modes)
  std::vector<InputBatchRef> inputBatches_;

  // PER_ROW state
  std::deque<ReadyRow> readyRows_;

  // BATCH state
  int64_t nextBatchId_{0};
  std::deque<PendingBatch> pendingBatches_;

  // Common
  bool noMoreInput_{false};
  RPCStreamingMode streamingMode_{RPCStreamingMode::kPerRow};

  // Dispatched-but-not-completed UNITS of the active mode: rows in PER_ROW,
  // in-flight batches in BATCH. Mode is fixed for the lifetime of an RPCState,
  // so exactly one dispatch path feeds this counter. Backpressure compares it
  // against window_.limit().
  int64_t inFlight_{0};

  // High-water mark of inFlight_ across the lifetime of this RPCState.
  int64_t peakInFlight_{0};

  // Accumulated RTT measurements across all completed units.
  int64_t rttMinNs_{std::numeric_limits<int64_t>::max()};
  int64_t rttMaxNs_{0};
  int64_t numRttSamples_{0};

  // Latency-gradient concurrency window, fed RTT via onUnitSample(). Both modes
  // use the same learner; setStreamingMode() only picks the (start, max) pair:
  // - PER_ROW {100, 100}: starts non-binding (per-row parallelism is bounded by
  //   the transport thread pool) but still shrinks under sustained overload and
  //   recovers back toward 100.
  // - BATCH {2, large}: starts at 2 and learns the in-flight-batch sweet spot,
  //   bounded only by a large safety ceiling.
  // Shrinks fast on overload (onUnitError).
  CongestionController window_;
};

} // namespace facebook::velox::exec::rpc
