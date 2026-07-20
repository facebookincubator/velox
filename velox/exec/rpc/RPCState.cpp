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

#include "velox/exec/rpc/RPCState.h"

#include <algorithm>
#include <chrono>

#include <folly/executors/InlineExecutor.h>

#include "velox/common/base/Exceptions.h"

#define RPC_STATE_LOG(severity) LOG(severity) << "[RPC_STATE] "
#define RPC_STATE_VLOG(level) VLOG(level) << "[RPC_STATE] "

namespace facebook::velox::exec::rpc {

namespace {
// Safety ceiling for the BATCH latency-gradient window. The gradient backs off
// as soon as queueing lifts RTT, well before this bound, so it caps
// pathological growth rather than tuning throughput.
constexpr int64_t kBatchMaxWindow = 256;

// Monotonic now() in nanos for RTT measurement. steady_clock (not wall-clock)
// so NTP/clock adjustments between dispatch and completion can't inject a bogus
// round-trip time that would skew the gradient.
int64_t steadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
} // namespace

// ===== Configuration =====
// These setters are called during RPCOperator construction, before any
// concurrent access. No lock needed — avoids lock-order-inversion with the
// Task mutex (TSAN: M0→M1→M0 cycle).

void RPCState::setStreamingMode(
    RPCStreamingMode mode,
    int64_t minWindow,
    double stepCoef,
    int64_t maxWindow) {
  streamingMode_ = mode;
  if (mode == RPCStreamingMode::kBatch) {
    // BATCH: the gradient window starts at 2 (the previously hand-tuned value)
    // and learns the in-flight-batch sweet spot from per-batch RTT, bounded
    // only by a large safety ceiling so it never leaves throughput on the
    // table.
    const int64_t maxW = maxWindow > 0 ? maxWindow : kBatchMaxWindow;
    window_ = CongestionController{/*startWindow*/ 2,
                                   /*maxWindow*/ maxW,
                                   minWindow,
                                   stepCoef};
  } else {
    // PER_ROW: start == max. Default ceiling 100; a high-latency backend can
    // raise it via rpc.congestion.max_window. Pinning start == max is
    // deliberate: the window starts at the ceiling and can only SHRINK from
    // there under overload. It must NOT ramp upward on its own, because a
    // rate-limit (429) storm is LOW-latency — the latency gradient can't see
    // it, so a probing-upward window would ramp straight into the storm. The
    // gradient plus the per-driver onUnitError halve (fed kRateLimited/kTimeout
    // by RPCOperator) shrink it under sustained overload.
    const int64_t maxW = maxWindow > 0 ? maxWindow : 100;
    window_ = CongestionController{
        /*startWindow*/ maxW, /*maxWindow*/ maxW, minWindow, stepCoef};
  }
}

RPCStreamingMode RPCState::streamingMode() const {
  return streamingMode_;
}

void RPCState::setMaxWindow(int64_t maxWindow) {
  window_ = CongestionController{maxWindow, maxWindow};
}

// ===== Input batch storage =====

int32_t RPCState::storeInputBatch(
    std::vector<VectorPtr> flatColumns,
    int64_t rowCount) {
  std::lock_guard<std::mutex> l(mutex_);
  auto batchIndex = static_cast<int32_t>(inputBatches_.size());
  inputBatches_.push_back(
      InputBatchRef{
          .flatColumns = std::move(flatColumns), .activeRowCount = rowCount});
  RPC_STATE_VLOG(2) << "storeInputBatch: batchIndex=" << batchIndex
                    << ", rowCount=" << rowCount;
  return batchIndex;
}

std::vector<VectorPtr> RPCState::getInputBatchColumns(
    int32_t batchIndex) const {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_LT(
      batchIndex,
      static_cast<int32_t>(inputBatches_.size()),
      "Invalid batchIndex {} for inputBatches_ of size {}",
      batchIndex,
      inputBatches_.size());
  return inputBatches_[batchIndex].flatColumns;
}

void RPCState::releaseRows(int32_t batchIndex, int64_t count) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_LT(
      batchIndex,
      static_cast<int32_t>(inputBatches_.size()),
      "Invalid batchIndex {} for inputBatches_ of size {}",
      batchIndex,
      inputBatches_.size());
  auto& batch = inputBatches_[batchIndex];
  VELOX_CHECK_GE(
      batch.activeRowCount,
      count,
      "Cannot release {} rows from batch {} with only {} active rows",
      count,
      batchIndex,
      batch.activeRowCount);
  batch.activeRowCount -= count;
  if (batch.activeRowCount == 0) {
    // Release the column vectors to free memory.
    batch.flatColumns.clear();
    RPC_STATE_VLOG(2) << "releaseRows: batch " << batchIndex
                      << " fully released";
  }
}

// ===== PER_ROW mode API =====

void RPCState::addPendingRow(
    std::shared_ptr<RPCState> selfPtr,
    int64_t rowId,
    RowLocation location,
    folly::SemiFuture<RPCResponse> future) {
  auto dispatchTimeNs = steadyNowNs();
  {
    std::lock_guard<std::mutex> l(mutex_);
    inFlight_++;
    peakInFlight_ = std::max(peakInFlight_, inFlight_);
    RPC_STATE_VLOG(2) << "addPendingRow: rowId=" << rowId
                      << ", inFlight=" << inFlight_;
  }

  // Attach completion callbacks that delegate to completeRow(). The dispatch
  // timestamp is captured so completion derives the round-trip latency for the
  // gradient window. We capture selfPtr to keep this RPCState alive until the
  // callback fires.
  auto stateForError = selfPtr;
  folly::futures::detachOn(
      folly::getKeepAliveToken(folly::InlineExecutor::instance()),
      std::move(future)
          .deferValue(
              [state = std::move(selfPtr), rowId, location, dispatchTimeNs](
                  RPCResponse response) mutable {
                const auto rttNs = steadyNowNs() - dispatchTimeNs;
                state->completeRow(rowId, location, std::move(response), rttNs);
              })
          .deferError(
              [state = std::move(stateForError),
               rowId,
               location,
               dispatchTimeNs](const folly::exception_wrapper& ew) mutable {
                RPC_STATE_LOG(ERROR)
                    << "RPC failed for rowId=" << rowId << ": " << ew.what();
                RPCResponse errorResponse;
                errorResponse.rowId = rowId;
                errorResponse.error = ew.what().toStdString();
                const auto rttNs = steadyNowNs() - dispatchTimeNs;
                state->completeRow(
                    rowId, location, std::move(errorResponse), rttNs);
              }));
}

void RPCState::completeRow(
    int64_t rowId,
    RowLocation location,
    RPCResponse response,
    int64_t rttNs) {
  std::lock_guard<std::mutex> l(mutex_);
  readyRows_.push_back(
      ReadyRow{
          .rowId = rowId,
          .location = location,
          .response = std::move(response),
          .rttNs = rttNs});
  inFlight_--;

  if (rttNs > 0) {
    rttMinNs_ = std::min(rttMinNs_, rttNs);
    rttMaxNs_ = std::max(rttMaxNs_, rttNs);
    ++numRttSamples_;
  }

  RPC_STATE_VLOG(2) << "Row completed: rowId=" << rowId
                    << ", readyRows=" << readyRows_.size()
                    << ", inFlight=" << inFlight_;

  notifyWaitersLocked();
}

RPCState::ClaimResult RPCState::tryClaimOrWait(
    ContinueFuture* future,
    std::optional<ReadyRow>* claimedRow) {
  std::lock_guard<std::mutex> l(mutex_);

  // Step 1: Try to claim a ready row.
  if (!readyRows_.empty()) {
    *claimedRow = std::move(readyRows_.front());
    readyRows_.pop_front();
    RPC_STATE_VLOG(1) << "tryClaimOrWait: claimed rowId="
                      << (*claimedRow)->rowId
                      << ", remaining ready=" << readyRows_.size();
    return ClaimResult::kClaimed;
  }

  // Step 2: Check finish condition.
  if (noMoreInput_ && inFlight_ == 0) {
    RPC_STATE_VLOG(1) << "tryClaimOrWait: finish condition met";
    return ClaimResult::kFinished;
  }

  // Step 3: Must wait — create a promise under the same lock to prevent
  // TOCTOU races (no completion can slip between the check and wait).
  RPC_STATE_VLOG(2) << "tryClaimOrWait: must wait (inFlight=" << inFlight_
                    << ")";
  promises_.emplace_back("RPCState::tryClaimOrWait");
  *future = promises_.back().getSemiFuture();
  return ClaimResult::kMustWait;
}

std::optional<RPCState::ReadyRow> RPCState::tryClaimReady() {
  std::lock_guard<std::mutex> l(mutex_);
  if (!readyRows_.empty()) {
    auto row = std::move(readyRows_.front());
    readyRows_.pop_front();
    RPC_STATE_VLOG(1) << "tryClaimReady: claimed rowId=" << row.rowId
                      << ", remaining ready=" << readyRows_.size();
    return row;
  }
  return std::nullopt;
}

void RPCState::drainReadyRows(std::vector<ReadyRow>& out, int32_t maxRows) {
  std::lock_guard<std::mutex> l(mutex_);
  while (!readyRows_.empty() && static_cast<int32_t>(out.size()) < maxRows) {
    out.push_back(std::move(readyRows_.front()));
    readyRows_.pop_front();
  }
}

int64_t RPCState::numInFlight() const {
  std::lock_guard<std::mutex> l(mutex_);
  return inFlight_;
}

// ===== BATCH mode API =====

void RPCState::addPendingBatch(
    std::shared_ptr<RPCState> selfPtr,
    folly::SemiFuture<std::vector<RPCResponse>> future,
    std::vector<RowLocation> rowLocations) {
  // Build the callback chain OUTSIDE the lock. .via(InlineExecutor) may drive
  // the chain inline if the future is already resolved, and the
  // thenValue/thenError callbacks acquire mutex_ to notify waiters — holding
  // mutex_ here would self-deadlock. RPCState is per-driver (owned by
  // RPCOperator, never shared), so if the chain completes inline before the
  // batch is inserted below, promises_ is empty and the notify is a harmless
  // no-op; the batch is found ready by the next tryPollBatchOrWait.
  //
  // Capture the dispatch time BEFORE attaching the callback so an already-ready
  // future driven inline cannot stamp completion ahead of dispatch (which would
  // yield a negative RTT the gradient then drops).
  auto dispatchTimeNs = steadyNowNs();

  // Shared cell the completion callback stamps with the monotonic completion
  // time, so the poll path measures dispatch->completion RTT rather than
  // dispatch->poll (which would fold in-operator queueing into the sample).
  auto completionTimeNs = std::make_shared<std::atomic<int64_t>>(0);
  auto callbackFuture =
      std::move(future)
          .via(folly::getKeepAliveToken(folly::InlineExecutor::instance()))
          .thenValue([state = selfPtr,
                      completionTimeNs](std::vector<RPCResponse> responses) {
            completionTimeNs->store(steadyNowNs(), std::memory_order_relaxed);
            RPC_STATE_VLOG(1)
                << "Batch completed with " << responses.size() << " responses";
            {
              std::lock_guard<std::mutex> l(state->mutex_);
              state->notifyWaitersLocked();
            }
            return responses;
          })
          .thenError([state = selfPtr,
                      completionTimeNs](folly::exception_wrapper ew) {
            completionTimeNs->store(steadyNowNs(), std::memory_order_relaxed);
            RPC_STATE_LOG(ERROR) << "Batch failed: " << ew.what();
            {
              std::lock_guard<std::mutex> l(state->mutex_);
              state->notifyWaitersLocked();
            }
            return folly::makeSemiFuture<std::vector<RPCResponse>>(
                std::move(ew));
          })
          .semi();

  {
    std::lock_guard<std::mutex> l(mutex_);
    auto batchId = nextBatchId_++;
    inFlight_++;
    peakInFlight_ = std::max(peakInFlight_, inFlight_);
    pendingBatches_.push_back(
        PendingBatch{
            .batchId = batchId,
            .future = std::move(callbackFuture),
            .rowLocations = std::move(rowLocations),
            .dispatchTimeNs = dispatchTimeNs,
            .completionTimeNs = std::move(completionTimeNs)});

    RPC_STATE_VLOG(1) << "addPendingBatch: batchId=" << batchId
                      << ", totalPending=" << pendingBatches_.size();
  }
}

RPCState::ReadyBatch RPCState::extractReadyBatchLocked(
    const std::deque<PendingBatch>::iterator& it) {
  ReadyBatch result;
  result.batchId = it->batchId;
  result.rowLocations = std::move(it->rowLocations);
  // Latency stamped at completion (in the future callback), not now, so
  // in-operator poll delay between completion and this drain is not folded in.
  // The relaxed load is safe: we only reach here after future.isReady(), and
  // the completion callback stored completionTimeNs before fulfilling the
  // future, so its value is published to this thread by that happens-before
  // edge. A not-yet-stamped 0 would yield a negative rttNs, which onSample
  // drops — never a torn read.
  result.rttNs = it->completionTimeNs->load(std::memory_order_relaxed) -
      it->dispatchTimeNs;

  try {
    result.responses = std::move(it->future).get();
    RPC_STATE_VLOG(1) << "extractReadyBatchLocked: batchId=" << result.batchId
                      << " ready with " << result.responses.size()
                      << " responses";
  } catch (const std::exception& e) {
    result.error = e.what();
    result.responses = {};
    RPC_STATE_LOG(ERROR) << "extractReadyBatchLocked: batchId="
                         << result.batchId << " failed: " << e.what();
  }

  if (result.rttNs > 0) {
    rttMinNs_ = std::min(rttMinNs_, result.rttNs);
    rttMaxNs_ = std::max(rttMaxNs_, result.rttNs);
    ++numRttSamples_;
  }

  pendingBatches_.erase(it);
  inFlight_--;
  return result;
}

RPCState::BatchPollResult RPCState::tryPollBatchOrWait(
    ContinueFuture* future,
    std::optional<ReadyBatch>* readyBatch) {
  std::lock_guard<std::mutex> l(mutex_);

  // Step 1: Check for a ready batch (out-of-order: first ready wins).
  for (auto it = pendingBatches_.begin(); it != pendingBatches_.end(); ++it) {
    if (it->future.isReady()) {
      *readyBatch = extractReadyBatchLocked(it);
      return BatchPollResult::kGotBatch;
    }
  }

  // Step 2: Check finish condition.
  if (noMoreInput_ && pendingBatches_.empty()) {
    RPC_STATE_VLOG(1) << "tryPollBatchOrWait: finish condition met";
    return BatchPollResult::kFinished;
  }

  // Step 3: Must wait.
  RPC_STATE_VLOG(2) << "tryPollBatchOrWait: must wait";
  promises_.emplace_back("RPCState::tryPollBatchOrWait");
  *future = promises_.back().getSemiFuture();
  return BatchPollResult::kMustWait;
}

std::optional<RPCState::ReadyBatch> RPCState::tryPollReady() {
  std::lock_guard<std::mutex> l(mutex_);
  for (auto it = pendingBatches_.begin(); it != pendingBatches_.end(); ++it) {
    if (it->future.isReady()) {
      return extractReadyBatchLocked(it);
    }
  }
  return std::nullopt;
}

// ===== Common =====

void RPCState::setNoMoreInput() {
  std::lock_guard<std::mutex> l(mutex_);
  noMoreInput_ = true;
  RPC_STATE_VLOG(1) << "setNoMoreInput: inFlight=" << inFlight_
                    << ", pendingBatches=" << pendingBatches_.size()
                    << ", readyRows=" << readyRows_.size();
  notifyWaitersLocked();
}

bool RPCState::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  // inFlight_ == 0 covers both pending rows and pending batches; readyRows_ is
  // PER_ROW-only and always empty in BATCH.
  return noMoreInput_ && inFlight_ == 0 && readyRows_.empty();
}

bool RPCState::isUnderBackpressure() {
  std::lock_guard<std::mutex> l(mutex_);
  // Unified across modes: inFlight_ counts the active mode's units (rows for
  // PER_ROW, batches for BATCH) and window_ carries the gradient limit.
  return inFlight_ >= window_.limit();
}

int64_t RPCState::dispatchHeadroom() {
  std::lock_guard<std::mutex> l(mutex_);
  return std::max<int64_t>(0, window_.limit() - inFlight_);
}

void RPCState::onUnitError() {
  std::lock_guard<std::mutex> l(mutex_);
  const auto prev = window_.limit();
  window_.onError();
  if (window_.limit() != prev) {
    RPC_STATE_LOG(WARNING) << "RPC congestion: overload, window " << prev
                           << " -> " << window_.limit();
  }
}

void RPCState::onUnitSample(int64_t rttNs) {
  std::lock_guard<std::mutex> l(mutex_);
  const auto prev = window_.limit();
  window_.onSample(rttNs);
  if (window_.limit() != prev) {
    RPC_STATE_LOG(INFO) << "RPC congestion: gradient sample rtt=" << rttNs
                        << "ns, window " << prev << " -> " << window_.limit();
  }
}

void RPCState::onUnitSamples(const std::vector<int64_t>& rttNsList) {
  std::lock_guard<std::mutex> l(mutex_);
  const auto prev = window_.limit();
  for (auto rttNs : rttNsList) {
    window_.onSample(rttNs);
  }
  if (window_.limit() != prev) {
    RPC_STATE_LOG(INFO) << "RPC congestion: " << rttNsList.size()
                        << " gradient samples, window " << prev << " -> "
                        << window_.limit();
  }
}

void RPCState::notifyWaitersLocked() {
  // Fulfill all promises to wake up blocked drivers.
  // Called while mutex_ is held.
  for (auto& promise : promises_) {
    promise.setValue();
  }
  promises_.clear();
}

RPCState::OperatorSnapshot RPCState::operatorSnapshot() const {
  std::lock_guard<std::mutex> l(mutex_);
  return OperatorSnapshot{
      .windowLimit = window_.limit(),
      .baselineRttNs = window_.baselineRttNs(),
      .numShrinks = window_.numShrinks(),
      .peakInFlight = peakInFlight_,
      .rttMinNs = numRttSamples_ > 0 ? rttMinNs_ : 0,
      .rttMaxNs = rttMaxNs_,
      .numRttSamples = numRttSamples_,
      .streamingMode = streamingMode_,
  };
}

} // namespace facebook::velox::exec::rpc
