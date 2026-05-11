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

#include <folly/executors/InlineExecutor.h>

#include "velox/common/base/Exceptions.h"

#define RPC_STATE_LOG(severity) LOG(severity) << "[RPC_STATE] "
#define RPC_STATE_VLOG(level) VLOG(level) << "[RPC_STATE] "

namespace facebook::velox::exec::rpc {

// ===== Configuration =====
// These setters are called during RPCOperator construction, before any
// concurrent access. No lock needed — avoids lock-order-inversion with the
// Task mutex (TSAN: M0→M1→M0 cycle).

void RPCState::setStreamingMode(RPCStreamingMode mode) {
  streamingMode_ = mode;
}

RPCStreamingMode RPCState::streamingMode() const {
  return streamingMode_;
}

void RPCState::setMaxPendingRows(int64_t maxPendingRows) {
  maxPendingRows_ = maxPendingRows;
}

void RPCState::setMaxPendingBatches(int64_t maxPendingBatches) {
  maxPendingBatches_ = maxPendingBatches;
  effectiveMaxPendingBatches_ = maxPendingBatches;
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
  {
    std::lock_guard<std::mutex> l(mutex_);
    numPendingRows_++;
    RPC_STATE_VLOG(2) << "addPendingRow: rowId=" << rowId
                      << ", pendingCount=" << numPendingRows_;
  }

  // Attach completion callbacks that delegate to completeRow().
  // We capture selfPtr to keep this RPCState alive until the callback fires.
  auto stateForError = selfPtr;
  folly::futures::detachOn(
      folly::getKeepAliveToken(folly::InlineExecutor::instance()),
      std::move(future)
          .deferValue([state = std::move(selfPtr), rowId, location](
                          RPCResponse response) mutable {
            state->completeRow(rowId, location, std::move(response));
          })
          .deferError([state = std::move(stateForError), rowId, location](
                          const folly::exception_wrapper& ew) mutable {
            RPC_STATE_LOG(ERROR)
                << "RPC failed for rowId=" << rowId << ": " << ew.what();
            RPCResponse errorResponse;
            errorResponse.rowId = rowId;
            errorResponse.error = ew.what().toStdString();
            state->completeRow(rowId, location, std::move(errorResponse));
          }));
}

void RPCState::completeRow(
    int64_t rowId,
    RowLocation location,
    RPCResponse response) {
  std::lock_guard<std::mutex> l(mutex_);
  readyRows_.push_back(
      ReadyRow{
          .rowId = rowId,
          .location = location,
          .response = std::move(response)});
  numPendingRows_--;

  RPC_STATE_VLOG(2) << "Row completed: rowId=" << rowId
                    << ", readyRows=" << readyRows_.size()
                    << ", pendingCount=" << numPendingRows_;

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
  if (noMoreInput_ && numPendingRows_ == 0) {
    RPC_STATE_VLOG(1) << "tryClaimOrWait: finish condition met";
    return ClaimResult::kFinished;
  }

  // Step 3: Must wait — create a promise under the same lock to prevent
  // TOCTOU races (no completion can slip between the check and wait).
  RPC_STATE_VLOG(2) << "tryClaimOrWait: must wait (pending=" << numPendingRows_
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

int64_t RPCState::numPendingRows() {
  std::lock_guard<std::mutex> l(mutex_);
  return numPendingRows_;
}

// ===== BATCH mode API =====

void RPCState::addPendingBatch(
    std::shared_ptr<RPCState> selfPtr,
    folly::SemiFuture<std::vector<RPCResponse>> future,
    std::vector<RowLocation> rowLocations) {
  std::lock_guard<std::mutex> l(mutex_);

  int64_t batchId = nextBatchId_++;

  // Attach a completion callback that notifies waiters when the batch
  // completes. We use .via().thenValue() to run the callback eagerly.
  auto callbackFuture =
      std::move(future)
          .via(folly::getKeepAliveToken(folly::InlineExecutor::instance()))
          .thenValue([state = selfPtr](std::vector<RPCResponse> responses) {
            RPC_STATE_VLOG(1)
                << "Batch completed with " << responses.size() << " responses";
            {
              std::lock_guard<std::mutex> l(state->mutex_);
              state->notifyWaitersLocked();
            }
            return responses;
          })
          .thenError([state = selfPtr](folly::exception_wrapper ew) {
            RPC_STATE_LOG(ERROR) << "Batch failed: " << ew.what();
            {
              std::lock_guard<std::mutex> l(state->mutex_);
              state->notifyWaitersLocked();
            }
            return folly::makeSemiFuture<std::vector<RPCResponse>>(
                std::move(ew));
          })
          .semi();

  pendingBatches_.push_back(
      PendingBatch{
          .batchId = batchId,
          .future = std::move(callbackFuture),
          .rowLocations = std::move(rowLocations)});

  RPC_STATE_VLOG(1) << "addPendingBatch: batchId=" << batchId
                    << ", totalPending=" << pendingBatches_.size();
}

RPCState::BatchPollResult RPCState::tryPollBatchOrWait(
    ContinueFuture* future,
    std::optional<ReadyBatch>* readyBatch) {
  std::lock_guard<std::mutex> l(mutex_);

  // Step 1: Check for a ready batch (out-of-order: first ready wins).
  for (auto it = pendingBatches_.begin(); it != pendingBatches_.end(); ++it) {
    if (it->future.isReady()) {
      ReadyBatch result;
      result.batchId = it->batchId;
      result.rowLocations = std::move(it->rowLocations);

      try {
        result.responses = std::move(it->future).get();
        RPC_STATE_VLOG(1) << "tryPollBatchOrWait: batchId=" << result.batchId
                          << " ready with " << result.responses.size()
                          << " responses";
      } catch (const std::exception& e) {
        result.error = e.what();
        result.responses = {};
        RPC_STATE_LOG(ERROR) << "tryPollBatchOrWait: batchId=" << result.batchId
                             << " failed: " << e.what();
      }

      pendingBatches_.erase(it);
      *readyBatch = std::move(result);
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
      ReadyBatch result;
      result.batchId = it->batchId;
      result.rowLocations = std::move(it->rowLocations);
      try {
        result.responses = std::move(it->future).get();
        RPC_STATE_VLOG(1) << "tryPollReady: batchId=" << result.batchId
                          << " ready with " << result.responses.size()
                          << " responses";
      } catch (const std::exception& e) {
        result.error = e.what();
        result.responses = {};
        RPC_STATE_LOG(ERROR) << "tryPollReady: batchId=" << result.batchId
                             << " failed: " << e.what();
      }
      pendingBatches_.erase(it);
      return result;
    }
  }
  return std::nullopt;
}

// ===== Common =====

void RPCState::setNoMoreInput() {
  std::lock_guard<std::mutex> l(mutex_);
  noMoreInput_ = true;
  RPC_STATE_VLOG(1) << "setNoMoreInput: pendingRows=" << numPendingRows_
                    << ", pendingBatches=" << pendingBatches_.size()
                    << ", readyRows=" << readyRows_.size();
  notifyWaitersLocked();
}

bool RPCState::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return noMoreInput_ && numPendingRows_ == 0 && readyRows_.empty() &&
      pendingBatches_.empty();
}

bool RPCState::isUnderBackpressure() {
  std::lock_guard<std::mutex> l(mutex_);
  if (streamingMode_ == RPCStreamingMode::kBatch) {
    return static_cast<int64_t>(pendingBatches_.size()) >=
        effectiveMaxPendingBatches_;
  }
  return numPendingRows_ >= maxPendingRows_;
}

void RPCState::onBatchSuccess(int64_t increment) {
  std::lock_guard<std::mutex> l(mutex_);
  if (effectiveMaxPendingBatches_ < maxPendingBatches_) {
    effectiveMaxPendingBatches_ =
        std::min(effectiveMaxPendingBatches_ + increment, maxPendingBatches_);
    RPC_STATE_LOG(INFO) << "RPC congestion: batch success, window increased to "
                        << effectiveMaxPendingBatches_ << "/"
                        << maxPendingBatches_;
  }
}

void RPCState::onBatchError() {
  std::lock_guard<std::mutex> l(mutex_);
  auto prev = effectiveMaxPendingBatches_;
  effectiveMaxPendingBatches_ =
      std::max<int64_t>(effectiveMaxPendingBatches_ / 2, 1);
  if (effectiveMaxPendingBatches_ < prev) {
    RPC_STATE_LOG(WARNING)
        << "RPC congestion: batch error, window decreased from " << prev
        << " to " << effectiveMaxPendingBatches_;
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

} // namespace facebook::velox::exec::rpc
