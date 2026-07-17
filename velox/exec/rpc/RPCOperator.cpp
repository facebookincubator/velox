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

#include "velox/exec/rpc/RPCOperator.h"

#include <algorithm>

#include "velox/common/time/CpuWallTimer.h"
#include "velox/common/time/Timer.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

#define RPC_OP_LOG(severity) LOG(severity) << "[RPC_OP] "
#define RPC_OP_VLOG(level) VLOG(level) << "[RPC_OP] "

namespace facebook::velox::exec::rpc {

RPCOperator::RPCOperator(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::RPCNode> rpcNode)
    : exec::Operator(
          driverCtx,
          rpcNode->outputType(),
          operatorId,
          rpcNode->id(),
          "RPC"),
      rpcNode_(std::move(rpcNode)),
      state_(std::make_shared<RPCState>()),
      dispatchBatchSize_(rpcNode_->dispatchBatchSize()) {
  // Configure RPCState with the streaming mode and the congestion-window
  // tunables. The two knobs are registered QueryConfig properties
  // (rpc.congestion.min_window / rpc.congestion.step_coef) with safe defaults,
  // so they can be retuned via SET SESSION / configerator with no code change;
  // unset means the controller's defaults (floor 1, headroom 1.0x).
  const auto& queryConfig = driverCtx->queryConfig();
  state_->setStreamingMode(
      rpcNode_->streamingMode(),
      queryConfig.rpcCongestionMinWindow(),
      queryConfig.rpcCongestionStepCoef(),
      queryConfig.rpcCongestionMaxWindow());
}

void RPCOperator::initialize() {
  Operator::initialize();

  // Resolve the AsyncRPCFunction by name from the registry.
  function_ = AsyncRPCFunctionRegistry::create(rpcNode_->functionName());
  VELOX_CHECK(
      function_,
      "Unknown RPC function '{}'. Ensure it is registered via "
      "AsyncRPCFunctionRegistry::registerFunction() before query execution.",
      rpcNode_->functionName());

  // Initialize the function with query config, argument types, and constants.
  // The function creates/caches its own transport and clients internally.
  function_->initialize(
      operatorCtx_->driverCtx()->queryConfig(),
      rpcNode_->argumentTypes(),
      rpcNode_->constantInputs());

  tierKey_ = function_->tierKey();

  // Configure the process-global adaptive rate limiter from QueryConfig. This
  // is idempotent and cluster-default-driven; off by default (static cap).
  const auto& queryConfig = operatorCtx_->driverCtx()->queryConfig();
  RPCRateLimiter::setAdaptiveConfig(
      queryConfig.rpcRateLimiterAdaptiveEnabled(),
      queryConfig.rpcRateLimiterMinLimit(),
      queryConfig.rpcRateLimiterDecreaseFactor());
  // Raise the per-tier ceiling so a high-latency backend can run at high
  // concurrency; admission-controlled dispatch makes this cap bind, and the
  // adaptive limiter shrinks from here under overload. 0 keeps the built-in 20.
  if (const auto rlMax = queryConfig.rpcRateLimiterMaxLimit(); rlMax > 0) {
    RPCRateLimiter::setMaxPending(tierKey_, rlMax);
  }

  RPC_OP_VLOG(1) << "Created operator for function '"
                 << rpcNode_->functionName() << "', planNodeId=" << planNodeId()
                 << ", operatorId=" << operatorId() << ", streamingMode="
                 << (rpcNode_->streamingMode() == RPCStreamingMode::kBatch
                         ? "BATCH"
                         : "PER_ROW");

  // Precompute argument column indices for addInput().
  const auto& argCols = rpcNode_->argumentColumns();
  if (!argCols.empty()) {
    auto sourceType = rpcNode_->source()->outputType();
    argumentColumnIndices_.reserve(argCols.size());
    for (const auto& colName : argCols) {
      auto idx = sourceType->getChildIdx(colName);
      argumentColumnIndices_.push_back(static_cast<column_index_t>(idx));
    }
    RPC_OP_VLOG(1) << "Initialized with " << argCols.size()
                   << " argument columns";
  } else {
    RPC_OP_VLOG(1) << "Initialized with no argument columns "
                   << "(fallback to all input columns)";
  }

  // Precompute output column projections to avoid string lookups in
  // buildOutputVector().
  initOutputProjections();
}

bool RPCOperator::needsInput() const {
  if (noMoreInput_ || isDraining()) {
    return false;
  }

  // Don't accept input if we have results ready to output.
  if (!claimedRows_.empty() || claimedBatch_.has_value()) {
    return false;
  }

  // Don't take a new input vector until the current one's buffered rows
  // are fully dispatched (drained under admission control).
  if (hasPendingRows()) {
    return false;
  }

  // Check per-state backpressure.
  if (state_->isUnderBackpressure()) {
    return false;
  }

  return true;
}

void RPCOperator::addInput(RowVectorPtr input) {
  if (!input || input->size() == 0) {
    RPC_OP_VLOG(2) << "addInput received empty input";
    return;
  }

  RPC_OP_VLOG(1) << "addInput received " << input->size() << " rows with "
                 << input->childrenSize() << " input columns";

  SelectivityVector rows(input->size());

  // Read pre-computed argument columns by precomputed index.
  std::vector<VectorPtr> args;
  if (!argumentColumnIndices_.empty()) {
    args.reserve(argumentColumnIndices_.size());
    for (auto idx : argumentColumnIndices_) {
      args.push_back(input->childAt(idx));
    }
  } else {
    // Fallback: use all input columns as arguments.
    for (auto i = 0; i < input->childrenSize(); ++i) {
      args.push_back(input->childAt(i));
    }
  }

  // Flatten/load all columns upfront to avoid issues with lazy vectors.
  std::vector<VectorPtr> flattenedColumns;
  flattenedColumns.reserve(input->childrenSize());
  for (int32_t j = 0; j < input->childrenSize(); ++j) {
    auto column = BaseVector::loadedVectorShared(input->childAt(j));
    BaseVector::flattenVector(column);
    flattenedColumns.push_back(column);
  }

  auto streamingMode = state_->streamingMode();

  if (streamingMode == RPCStreamingMode::kPerRow) {
    // PER_ROW: buffer the input and drip its rows out under admission
    // control (dispatchPendingRows) instead of dispatching the whole vector at
    // once, which would overrun both the per-driver window and the per-tier
    // rate limiter. needsInput() returns false until this buffer is drained, so
    // exactly one input vector is buffered at a time.
    pendingArgs_ = std::move(args);
    pendingNumRows_ = static_cast<vector_size_t>(input->size());
    pendingCursor_ = 0;
    pendingBatchIndex_ = state_->storeInputBatch(
        std::move(flattenedColumns), static_cast<int64_t>(pendingNumRows_));
    dispatchPendingRows();
  } else {
    // BATCH: function accumulates typed data internally.
    auto rowIndices = function_->accumulateBatch(rows, args);

    auto batchIndex = state_->storeInputBatch(
        flattenedColumns, static_cast<int64_t>(rowIndices.size()));
    numRequestsDispatched_ += static_cast<int64_t>(rowIndices.size());

    for (auto originalRowIndex : rowIndices) {
      auto rowId = globalRowIdCounter_++;
      batchRowLocations_.push_back(
          RPCState::RowLocation{batchIndex, originalRowIndex});
      batchRowIds_.push_back(rowId);
    }

    if (dispatchBatchSize_ > 0 &&
        function_->pendingBatchSize() >= dispatchBatchSize_) {
      // Flush in chunks of dispatchBatchSize_ to avoid sending one
      // giant batch_predict call that overwhelms the server.
      while (function_->pendingBatchSize() >= dispatchBatchSize_ &&
             !state_->isUnderBackpressure()) {
        flushBatchRequests(dispatchBatchSize_);
      }
    }
  }
}

void RPCOperator::dispatchPendingRows() {
  while (hasPendingRows()) {
    // Admission headroom = min(per-driver congestion window, process-global
    // per-tier rate limiter). Both must have room; the tighter one binds. This
    // is what makes rpcPeakInFlight track the cap instead of the vector size.
    const int64_t headroom = std::min(
        state_->dispatchHeadroom(),
        RPCRateLimiter::availableHeadroom(tierKey_));
    if (headroom <= 0) {
      break;
    }
    const auto remaining =
        static_cast<int64_t>(pendingNumRows_ - pendingCursor_);
    const auto numRowsInChunk =
        static_cast<vector_size_t>(std::min(headroom, remaining));

    // Select this chunk's rows [pendingCursor_, pendingCursor_ +
    // numRowsInChunk).
    SelectivityVector chunk(pendingNumRows_, false);
    chunk.setValidRange(pendingCursor_, pendingCursor_ + numRowsInChunk, true);
    chunk.updateBounds();

    auto futures = function_->dispatchPerRow(chunk, pendingArgs_);
    // Contract: one future per selected row. We advance pendingCursor_ by n and
    // the stored batch's activeRowCount was set to the full row count, so a
    // short return would leave the batch buffer un-released (leak) and drop
    // rows from the output. Enforce it rather than fail silently.
    VELOX_CHECK_EQ(
        futures.size(),
        static_cast<size_t>(numRowsInChunk),
        "dispatchPerRow returned {} futures for {} selected rows",
        futures.size(),
        numRowsInChunk);
    numRequestsDispatched_ += static_cast<int64_t>(futures.size());
    for (auto& [originalRowIndex, future] : futures) {
      auto rowId = globalRowIdCounter_++;
      auto token = std::make_shared<RPCRateLimiter::Token>(
          RPCRateLimiter::acquire(tierKey_));
      auto wrapped =
          std::move(future)
              .within(kBatchRpcTimeout)
              .deferValue([rowId, token](RPCResponse resp) {
                resp.rowId = rowId;
                return resp;
              })
              .deferError([token](folly::exception_wrapper ew) {
                return folly::makeSemiFuture<RPCResponse>(std::move(ew));
              });
      state_->addPendingRow(
          state_,
          rowId,
          RPCState::RowLocation{pendingBatchIndex_, originalRowIndex},
          std::move(wrapped));
    }
    pendingCursor_ += numRowsInChunk;
  }
  if (!hasPendingRows()) {
    // Buffer fully dripped; drop references so needsInput() accepts the next
    // input vector.
    pendingArgs_.clear();
    pendingNumRows_ = 0;
    pendingCursor_ = 0;
    pendingBatchIndex_ = -1;
  }
}

void RPCOperator::flushBatchRequests(int32_t maxRows) {
  if (function_->pendingBatchSize() == 0) {
    VELOX_CHECK(
        batchRowLocations_.empty(),
        "Operator has {} accumulated batch rows but function reports "
        "pendingBatchSize=0. Function must override pendingBatchSize() "
        "when using BATCH mode.",
        batchRowLocations_.size());
    return;
  }

  // Determine how many rows to flush.
  auto flushCount = maxRows > 0
      ? std::min(static_cast<int32_t>(batchRowLocations_.size()), maxRows)
      : static_cast<int32_t>(batchRowLocations_.size());

  RPC_OP_LOG(INFO) << "Flushing batch with " << flushCount << " of "
                   << function_->pendingBatchSize() << " accumulated rows";

  // Split off the rows to flush.
  std::vector<RPCState::RowLocation> rowLocations(
      batchRowLocations_.begin(), batchRowLocations_.begin() + flushCount);
  std::vector<int64_t> rowIds(
      batchRowIds_.begin(), batchRowIds_.begin() + flushCount);
  batchRowLocations_.erase(
      batchRowLocations_.begin(), batchRowLocations_.begin() + flushCount);
  batchRowIds_.erase(batchRowIds_.begin(), batchRowIds_.begin() + flushCount);

  auto future = function_->flushBatch(maxRows);

  // Count each flushBatch() as 1 pending unit in the rate limiter.
  auto token = std::make_shared<RPCRateLimiter::Token>(
      RPCRateLimiter::acquire(tierKey_));

  // Share rowIds across both continuations. Order matters: deferError runs
  // BEFORE deferValue, so a whole-batch backend failure is first converted into
  // one errored response per row (in batch-position order), and then flows
  // through the same scatter as real responses. This keeps the scatter's
  // invariant checks (below) FATAL for genuine function-contract violations
  // (wrong response count / duplicate / out-of-range rowId) — those must still
  // hard-fail the query, not be silently degraded to NULL rows.
  auto rowIdsPtr = std::make_shared<std::vector<int64_t>>(std::move(rowIds));
  auto wrapped =
      std::move(future)
          .within(kBatchRpcTimeout)
          .deferError([rowIdsPtr, token](folly::exception_wrapper ew) {
            // A whole-batch failure (e.g. an operator-level batch/RPC timeout)
            // degrades to per-row errored responses so the per-row error policy
            // (meta_ai_on_error) applies downstream, instead of hard-failing
            // the whole query. Mirrors the client-layer fan-out, but covers all
            // backends and the operator-level timeout uniformly. Both AIMD
            // controllers still back off via evaluateCongestion (a batch
            // failure is overload). Responses carry batch-position rowId
            // (0..N-1) so the scatter below stamps global rowIds identically to
            // the success path.
            const auto& rowIds = *rowIdsPtr;
            RPC_OP_LOG(ERROR)
                << "RPC batch failed, " << rowIds.size()
                << " rows will carry a per-row error: " << ew.what();
            std::vector<RPCResponse> errResponses(rowIds.size());
            for (size_t i = 0; i < rowIds.size(); ++i) {
              errResponses[i].rowId = static_cast<int64_t>(i);
              errResponses[i].error = std::string("[RPC_BATCH] batch error: ") +
                  ew.what().toStdString();
              errResponses[i].errorKind =
                  velox::rpc::RPCErrorKind::kBackendError;
            }
            return errResponses;
          })
          // Scatter responses into batch-position order using each response's
          // function-assigned rowId (its position within the batch), then stamp
          // the global rowIds. Functions may return results out of order (e.g.,
          // MetaGen's batchDialogCompletion streams results in arbitrary
          // order). Without this, responses[i] would be paired with
          // rowLocations[i] in buildOutputFromReadyBatch, silently mis-mapping
          // results to wrong passthrough rows. Invariant violations here are
          // fatal by design.
          .deferValue([rowIdsPtr, token](std::vector<RPCResponse> resps) {
            const auto& rowIds = *rowIdsPtr;
            VELOX_CHECK_EQ(
                resps.size(),
                rowIds.size(),
                "RPC batch response count ({}) does not match row count ({})",
                resps.size(),
                rowIds.size());
            std::vector<RPCResponse> sorted(resps.size());
            std::vector<bool> seen(resps.size(), false);
            for (auto& resp : resps) {
              auto batchIdx = resp.rowId;
              VELOX_CHECK_GE(batchIdx, 0);
              VELOX_CHECK_LT(
                  static_cast<size_t>(batchIdx),
                  rowIds.size(),
                  "RPC batch response rowId ({}) out of range (0-{})",
                  batchIdx,
                  rowIds.size() - 1);
              VELOX_CHECK(
                  !seen[static_cast<size_t>(batchIdx)],
                  "Duplicate batch response rowId ({})",
                  batchIdx);
              seen[static_cast<size_t>(batchIdx)] = true;
              resp.rowId = rowIds[static_cast<size_t>(batchIdx)];
              sorted[static_cast<size_t>(batchIdx)] = std::move(resp);
            }
            return sorted;
          });

  state_->addPendingBatch(state_, std::move(wrapped), std::move(rowLocations));
}

void RPCOperator::noMoreInput() {
  exec::Operator::noMoreInput();

  RPC_OP_VLOG(1) << "noMoreInput: totalRequestsDispatched="
                 << numRequestsDispatched_;

  if (state_->streamingMode() == RPCStreamingMode::kBatch) {
    // Flush any remaining accumulated rows in chunks.
    while (function_->pendingBatchSize() > 0) {
      flushBatchRequests(dispatchBatchSize_ > 0 ? dispatchBatchSize_ : 0);
    }
  }

  state_->setNoMoreInput();
}

RowVectorPtr RPCOperator::getOutput() {
  auto streamingMode = state_->streamingMode();

  if (streamingMode == RPCStreamingMode::kPerRow) {
    // Drip more buffered rows now that in-flight completions may have
    // freed window / rate-limiter headroom.
    dispatchPendingRows();

    if (claimedRows_.empty()) {
      // If draining and nothing left to output, check finish.
      if (isDraining() && state_->isFinished()) {
        finished_ = true;
        finishDrain();
      }
      return nullptr;
    }

    // Drain additional ready rows (non-blocking) for batched output.
    // This amortizes RowVector allocation across multiple completed rows.
    state_->drainReadyRows(claimedRows_, 1'024);

    // Materialize responses, locations, and round-trip latencies once — reused
    // for the congestion signal and the output vector (no extra copy).
    auto numRows = static_cast<int64_t>(claimedRows_.size());
    std::vector<RPCResponse> responses;
    std::vector<std::pair<int32_t, vector_size_t>> locations;
    std::vector<int64_t> roundTripTimesNs;
    responses.reserve(claimedRows_.size());
    locations.reserve(claimedRows_.size());
    roundTripTimesNs.reserve(claimedRows_.size());
    for (auto& row : claimedRows_) {
      const bool hasError = row.response.hasError();
      if (hasError) {
        numErrors_++;
        recordErrorKind(row.response.errorKind);
      }
      locations.emplace_back(row.location.batchIndex, row.location.rowIndex);
      // Only successful rows feed the gradient. Errored rows (e.g. null_input,
      // client-side rejections) complete without a real round trip, so their
      // artificially small RTT would pull down the per-window minimum and skew
      // the gradient/baseline.
      if (!hasError) {
        roundTripTimesNs.push_back(row.rttNs);
      }
      responses.push_back(std::move(row.response));
    }

    // Two AIMD controllers at different scopes, BOTH backing off on the
    // function's overload verdict (see RPCRateLimiter.h /
    // CongestionController.h / the function's CongestionPolicy):
    //  - Window (per-driver): halve on overload; otherwise feed the successful
    //    rows' RTTs to the latency gradient.
    //  - Rate limiter (process-global per tier): halve the cap on overload,
    //    additive-recover on success.
    // The policy classifies overload as rate-limit / timeout / majority error
    // (ignoring null_input). Both scopes must back off on it: a rate-limit
    // storm is LOW-latency, so the latency gradient alone is blind to it — the
    // error verdict is what makes the per-driver window back off, not just
    // latency.
    const auto signal = function_->evaluateCongestion(responses);
    if (signal == AsyncRPCFunction::CongestionSignal::kError) {
      state_->onUnitError();
      RPCRateLimiter::onRateLimited(tierKey_);
    } else if (signal == AsyncRPCFunction::CongestionSignal::kSuccess) {
      // Feed the whole drained batch of successful RTTs to the gradient in one
      // lock acquisition; its size is the success count driving AIMD recovery.
      state_->onUnitSamples(roundTripTimesNs);
      RPCRateLimiter::onSuccess(
          tierKey_, static_cast<int64_t>(roundTripTimesNs.size()));
    }

    auto output = buildOutputVector(responses, locations);
    numResponsesReceived_ += numRows;
    claimedRows_.clear();
    return output;
  } else {
    if (!claimedBatch_.has_value()) {
      // If draining and nothing left to output, check finish.
      if (isDraining() && state_->isFinished()) {
        finished_ = true;
        finishDrain();
      }
      return nullptr;
    }

    // Fail loudly on batch errors instead of silently dropping rows.
    if (claimedBatch_->error.has_value()) {
      auto error = claimedBatch_->error.value();
      claimedBatch_.reset();
      VELOX_FAIL("RPC batch failed: {}", error);
    }

    auto numRows = static_cast<int64_t>(claimedBatch_->responses.size());
    int64_t batchErrors = 0;
    for (const auto& response : claimedBatch_->responses) {
      if (response.hasError()) {
        numErrors_++;
        ++batchErrors;
        recordErrorKind(response.errorKind);
      }
    }

    // Both AIMD controllers back off on the function's overload verdict (see
    // PER_ROW above): the window (per-driver) halves on overload, else feeds
    // the batch RTT to the latency gradient; the rate limiter (global) halves
    // the cap on overload and recovers on success.
    const auto signal = function_->evaluateCongestion(claimedBatch_->responses);
    if (signal == AsyncRPCFunction::CongestionSignal::kError) {
      state_->onUnitError();
      RPCRateLimiter::onRateLimited(tierKey_);
    } else if (signal == AsyncRPCFunction::CongestionSignal::kSuccess) {
      // Feed the measured round-trip latency to the gradient window so it
      // learns the in-flight-batch sweet spot without a fixed ceiling.
      state_->onUnitSample(claimedBatch_->rttNs);
      // Successful rows in this batch drive AIMD recovery of the per-tier cap.
      RPCRateLimiter::onSuccess(tierKey_, numRows - batchErrors);
    }

    auto output = buildOutputFromReadyBatch(*claimedBatch_);
    numResponsesReceived_ += numRows;
    claimedBatch_.reset();
    return output;
  }
}

exec::BlockingReason RPCOperator::isBlocked(ContinueFuture* future) {
  // End any previous block wait measurement.
  if (blockWaitStartNs_.has_value()) {
    auto elapsed = getCurrentTimeNano() - blockWaitStartNs_.value();
    if (blockWaitIsBackpressure_) {
      totalBackpressureWaitNanos_ += elapsed;
    } else {
      totalBlockWaitNanos_ += elapsed;
    }
    blockWaitStartNs_ = std::nullopt;
  }

  // Emit ready output / report finished BEFORE any backpressure gate: a driver
  // holding completed rows (or with its own in-flight completions to harvest)
  // must never park behind the shared per-tier cap held by OTHER drivers. The
  // per-tier backpressure wait is applied as a last resort below, only when
  // this operator has buffered rows and nothing in-flight (i.e. it is genuinely
  // blocked on the global cap, with no local completion to wake it).
  if (!claimedRows_.empty() || claimedBatch_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  // If finished, don't block.
  if (finished_) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto streamingMode = state_->streamingMode();

  if (streamingMode == RPCStreamingMode::kPerRow) {
    if (!noMoreInput_ && !isDraining()) {
      // A completion may have freed headroom — try to drip more.
      dispatchPendingRows();
      auto claimedRow = state_->tryClaimReady();
      if (claimedRow) {
        claimedRows_.push_back(std::move(*claimedRow));
        return exec::BlockingReason::kNotBlocked;
      }
      // No ready output. Wait on the per-state completion future ONLY when this
      // operator has in-flight rows — those are guaranteed to fire that future.
      // If rows are buffered but nothing is in-flight, they are blocked solely
      // on the process-global per-tier rate limiter (its cap held by other
      // drivers); the per-state future would never resolve, so returning
      // kWaitForRPC here would hang the driver. Instead fall through to
      // kNotBlocked and let the next isBlocked() re-check
      // RPCRateLimiter::checkBackpressure(), which a slot-freeing decrement on
      // any driver wakes. needsInput() stays false while the buffer is
      // non-empty, so no new input arrives meanwhile.
      if (state_->numInFlight() > 0) {
        std::optional<RPCState::ReadyRow> waited;
        ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
        auto result = state_->tryClaimOrWait(&waitFuture, &waited);
        switch (result) {
          case RPCState::ClaimResult::kClaimed:
            claimedRows_.push_back(std::move(*waited));
            return exec::BlockingReason::kNotBlocked;
          case RPCState::ClaimResult::kFinished:
            return exec::BlockingReason::kNotBlocked;
          case RPCState::ClaimResult::kMustWait:
            *future = std::move(waitFuture);
            blockWaitStartNs_ = getCurrentTimeNano();
            blockWaitIsBackpressure_ = false;
            return exec::BlockingReason::kWaitForRPC;
        }
      }
      // Buffered rows but nothing in-flight: this operator is blocked solely on
      // the process-global per-tier cap (its slots held by other drivers). Park
      // on the limiter's waiter — woken by any driver's slot-freeing decrement
      // — rather than busy-spinning via repeated kNotBlocked.
      if (hasPendingRows()) {
        if (auto bp = RPCRateLimiter::checkBackpressure(tierKey_)) {
          *future = std::move(*bp);
          blockWaitStartNs_ = getCurrentTimeNano();
          blockWaitIsBackpressure_ = true;
          return exec::BlockingReason::kWaitForRPC;
        }
      }
      return exec::BlockingReason::kNotBlocked;
    }

    std::optional<RPCState::ReadyRow> claimedRow;
    ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
    auto result = state_->tryClaimOrWait(&waitFuture, &claimedRow);

    switch (result) {
      case RPCState::ClaimResult::kClaimed:
        claimedRows_.push_back(std::move(*claimedRow));
        return exec::BlockingReason::kNotBlocked;

      case RPCState::ClaimResult::kFinished:
        finished_ = true;
        return exec::BlockingReason::kNotBlocked;

      case RPCState::ClaimResult::kMustWait:
        *future = std::move(waitFuture);
        blockWaitStartNs_ = getCurrentTimeNano();
        blockWaitIsBackpressure_ = false;
        return exec::BlockingReason::kWaitForRPC;
    }
  } else {
    // BATCH mode
    if (!noMoreInput_ && !isDraining()) {
      auto readyBatch = state_->tryPollReady();
      if (readyBatch) {
        if (readyBatch->error.has_value()) {
          RPC_OP_LOG(WARNING)
              << "Received batch with error: " << readyBatch->error.value();
        }
        claimedBatch_ = std::move(*readyBatch);
      }
      return exec::BlockingReason::kNotBlocked;
    }

    std::optional<RPCState::ReadyBatch> readyBatch;
    ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
    auto result = state_->tryPollBatchOrWait(&waitFuture, &readyBatch);

    switch (result) {
      case RPCState::BatchPollResult::kGotBatch:
        if (readyBatch->error.has_value()) {
          RPC_OP_LOG(WARNING)
              << "Received batch with error: " << readyBatch->error.value();
        }
        claimedBatch_ = std::move(*readyBatch);
        return exec::BlockingReason::kNotBlocked;

      case RPCState::BatchPollResult::kFinished:
        finished_ = true;
        return exec::BlockingReason::kNotBlocked;

      case RPCState::BatchPollResult::kMustWait:
        *future = std::move(waitFuture);
        blockWaitStartNs_ = getCurrentTimeNano();
        blockWaitIsBackpressure_ = false;
        return exec::BlockingReason::kWaitForRPC;
    }
  }

  VELOX_UNREACHABLE();
}

bool RPCOperator::isFinished() {
  return finished_ && claimedRows_.empty() && !claimedBatch_.has_value();
}

bool RPCOperator::startDrain() {
  VELOX_CHECK(isDraining());
  VELOX_CHECK(!noMoreInput_);

  // Flush any undispatched accumulated rows.
  if (function_->pendingBatchSize() > 0) {
    flushBatchRequests();
  }

  // Signal RPCState that no more rows will be dispatched so it can
  // detect the finish condition once in-flight RPCs complete.
  state_->setNoMoreInput();

  // If we have claimed output or pending in-flight RPCs, there is
  // buffered data to drain.
  if (!claimedRows_.empty() || claimedBatch_.has_value()) {
    return true;
  }
  if (state_ && !state_->isFinished()) {
    return true;
  }
  return false;
}

void RPCOperator::close() {
  recordRuntimeStats();

  // Release resources explicitly. RPCState may be held alive by in-flight
  // RPC callbacks (via shared_ptr capture), but we release our reference
  // so that input batch memory can be freed as soon as possible.
  state_.reset();
  function_.reset();
  claimedRows_.clear();
  claimedBatch_.reset();
  batchRowLocations_.clear();
  batchRowIds_.clear();
  reusableIndices_.reset();

  Operator::close();
}

void RPCOperator::initOutputProjections() {
  const auto& outputColumn = rpcNode_->outputColumn();
  const auto& outputType = rpcNode_->outputType();
  auto sourceType = rpcNode_->source()->outputType();

  for (int32_t i = 0; i < static_cast<int32_t>(outputType->size()); ++i) {
    const auto& colName = outputType->nameOf(i);
    if (colName == outputColumn) {
      rpcResultOutputChannel_ = static_cast<column_index_t>(i);
    } else {
      auto colIdx = sourceType->getChildIdxIfExists(colName);
      if (colIdx.has_value()) {
        passthroughProjections_.push_back(
            OutputProjection{
                .outputChannel = static_cast<column_index_t>(i),
                .sourceChannel = static_cast<column_index_t>(colIdx.value())});
      }
    }
  }

  RPC_OP_VLOG(1) << "initOutputProjections: rpcResultChannel="
                 << rpcResultOutputChannel_ << ", passthroughProjections="
                 << passthroughProjections_.size();
}

void RPCOperator::recordErrorKind(velox::rpc::RPCErrorKind kind) {
  switch (kind) {
    case velox::rpc::RPCErrorKind::kRateLimited:
      ++numErrorsRateLimited_;
      break;
    case velox::rpc::RPCErrorKind::kTimeout:
      ++numErrorsTimeout_;
      break;
    case velox::rpc::RPCErrorKind::kBackendError:
      ++numErrorsBackend_;
      break;
    case velox::rpc::RPCErrorKind::kNone:
    case velox::rpc::RPCErrorKind::kNullInput:
    case velox::rpc::RPCErrorKind::kEmptyResponse:
    // A rejected request is a permanent client-side error, not a congestion
    // signal, so it is not counted among the overload kinds above (it is
    // tracked separately via a dedicated invalid-request counter).
    case velox::rpc::RPCErrorKind::kInvalidRequest:
      break;
  }
}

void RPCOperator::recordRuntimeStats() {
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      kRpcRequestsDispatched, RuntimeCounter(numRequestsDispatched_));
  lockedStats->addRuntimeStat(
      kRpcResponsesReceived, RuntimeCounter(numResponsesReceived_));
  lockedStats->addRuntimeStat(kRpcErrorCount, RuntimeCounter(numErrors_));
  if (totalBlockWaitNanos_ > 0) {
    lockedStats->addRuntimeStat(
        kRpcWaitWallNanos,
        RuntimeCounter(
            static_cast<int64_t>(totalBlockWaitNanos_),
            RuntimeCounter::Unit::kNanos));
  }
  if (totalBackpressureWaitNanos_ > 0) {
    lockedStats->addRuntimeStat(
        kRpcBackpressureWaitNanos,
        RuntimeCounter(
            static_cast<int64_t>(totalBackpressureWaitNanos_),
            RuntimeCounter::Unit::kNanos));
  }

  if (totalBlockWaitNanos_ > 0 || numResponsesReceived_ > 0) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(numResponsesReceived_), totalBlockWaitNanos_, 0};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }

  if (state_) {
    auto snapshot = state_->operatorSnapshot();
    lockedStats->addRuntimeStat(
        kRpcCongestionWindowFinal, RuntimeCounter(snapshot.windowLimit));
    lockedStats->addRuntimeStat(
        kRpcPeakInFlight, RuntimeCounter(snapshot.peakInFlight));
    if (snapshot.numShrinks > 0) {
      lockedStats->addRuntimeStat(
          kRpcCongestionShrinks, RuntimeCounter(snapshot.numShrinks));
    }
    if (snapshot.baselineRttNs > 0) {
      lockedStats->addRuntimeStat(
          kRpcBaselineRttNanos,
          RuntimeCounter(snapshot.baselineRttNs, RuntimeCounter::Unit::kNanos));
    }

    if (snapshot.numRttSamples > 0) {
      lockedStats->addRuntimeStat(
          kRpcRttMinWallNanos,
          RuntimeCounter(snapshot.rttMinNs, RuntimeCounter::Unit::kNanos));
      lockedStats->addRuntimeStat(
          kRpcRttMaxWallNanos,
          RuntimeCounter(snapshot.rttMaxNs, RuntimeCounter::Unit::kNanos));
      lockedStats->addRuntimeStat(
          kRpcRttCount, RuntimeCounter(snapshot.numRttSamples));
    }

    lockedStats->addRuntimeStat(
        kRpcStreamingMode,
        RuntimeCounter(
            snapshot.streamingMode == RPCStreamingMode::kBatch ? 1 : 0));
  }

  if (numErrorsRateLimited_ > 0) {
    lockedStats->addRuntimeStat(
        kRpcErrorKindRateLimited, RuntimeCounter(numErrorsRateLimited_));
  }
  if (numErrorsTimeout_ > 0) {
    lockedStats->addRuntimeStat(
        kRpcErrorKindTimeout, RuntimeCounter(numErrorsTimeout_));
  }
  if (numErrorsBackend_ > 0) {
    lockedStats->addRuntimeStat(
        kRpcErrorKindBackendError, RuntimeCounter(numErrorsBackend_));
  }

  // Process-global per-tier rate-limiter cap trajectory (the shared cap this
  // operator drips against). Distinct from the per-driver rpcCongestion*
  // window. Emitted unconditionally — including for the empty/default tier,
  // which is the bucket the meta.ai per-row-key path uses; gating on a
  // non-empty tierKey_ would hide the cap on exactly that main path.
  lockedStats->addRuntimeStat(
      kRpcRateLimiterCap,
      RuntimeCounter(RPCRateLimiter::currentLimit(tierKey_)));
  lockedStats->addRuntimeStat(
      kRpcRateLimiterPeakPending,
      RuntimeCounter(RPCRateLimiter::peakPending(tierKey_)));
  const auto minCap = RPCRateLimiter::minLimitReached(tierKey_);
  if (minCap > 0) {
    lockedStats->addRuntimeStat(kRpcRateLimiterMinCap, RuntimeCounter(minCap));
  }
}

RowVectorPtr RPCOperator::buildOutputFromReadyBatch(
    RPCState::ReadyBatch& readyBatch) {
  std::vector<std::pair<int32_t, vector_size_t>> locations;
  locations.reserve(readyBatch.rowLocations.size());
  for (const auto& loc : readyBatch.rowLocations) {
    locations.emplace_back(loc.batchIndex, loc.rowIndex);
  }

  return buildOutputVector(readyBatch.responses, locations);
}

RowVectorPtr RPCOperator::buildOutputVector(
    const std::vector<RPCResponse>& responses,
    const std::vector<std::pair<int32_t, vector_size_t>>& locations) {
  const auto numRows = static_cast<vector_size_t>(responses.size());
  auto* pool = operatorCtx_->pool();

  const auto& outputType = rpcNode_->outputType();

  // Use AsyncRPCFunction to build RPC result column.
  auto responseVector = function_->buildOutput(responses, pool);

  // Check if all rows come from the same batch (common for BATCH mode).
  bool singleBatch = true;
  if (numRows > 0) {
    int32_t firstBatch = locations[0].first;
    for (vector_size_t i = 1; i < numRows; ++i) {
      if (locations[i].first != firstBatch) {
        singleBatch = false;
        break;
      }
    }
  }

  std::vector<VectorPtr> outputChildren(outputType->size());

  // Set RPC result column using precomputed index.
  outputChildren[rpcResultOutputChannel_] = responseVector;

  // Set passthrough columns using precomputed projections.
  if (numRows == 0) {
    for (const auto& proj : passthroughProjections_) {
      outputChildren[proj.outputChannel] =
          BaseVector::create(outputType->childAt(proj.outputChannel), 0, pool);
    }
  } else if (singleBatch) {
    // All rows from same batch: use dictionary wrapping (zero-copy).
    const auto indicesByteSize = numRows * sizeof(vector_size_t);
    if (!reusableIndices_ || !reusableIndices_->unique() ||
        reusableIndices_->capacity() < indicesByteSize) {
      reusableIndices_ = allocateIndices(numRows, pool);
    }
    reusableIndices_->setSize(indicesByteSize);
    auto rawIndices = reusableIndices_->asMutable<vector_size_t>();
    for (vector_size_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
      rawIndices[rowIdx] = locations[rowIdx].second;
    }

    const auto batchCols = state_->getInputBatchColumns(locations[0].first);
    for (const auto& proj : passthroughProjections_) {
      if (proj.sourceChannel < static_cast<column_index_t>(batchCols.size())) {
        outputChildren[proj.outputChannel] = BaseVector::wrapInDictionary(
            nullptr, reusableIndices_, numRows, batchCols[proj.sourceChannel]);
      } else {
        outputChildren[proj.outputChannel] = BaseVector::createNullConstant(
            outputType->childAt(proj.outputChannel), numRows, pool);
      }
    }
  } else {
    // Rows from multiple batches: fetch columns once per batch.
    std::unordered_map<int32_t, std::vector<VectorPtr>> batchColsCache;
    for (const auto& proj : passthroughProjections_) {
      auto combined = BaseVector::create(
          outputType->childAt(proj.outputChannel), numRows, pool);
      for (vector_size_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        const auto& [batchIdx, rowInBatch] = locations[rowIdx];
        auto it = batchColsCache.find(batchIdx);
        if (it == batchColsCache.end()) {
          it = batchColsCache
                   .emplace(batchIdx, state_->getInputBatchColumns(batchIdx))
                   .first;
        }
        const auto& batchCols = it->second;
        if (proj.sourceChannel <
            static_cast<column_index_t>(batchCols.size())) {
          combined->copy(
              batchCols[proj.sourceChannel].get(), rowIdx, rowInBatch, 1);
        } else {
          combined->setNull(rowIdx, true);
        }
      }
      outputChildren[proj.outputChannel] = combined;
    }
  }

  // Fill any remaining nullptr entries with null constants.
  for (int32_t i = 0; i < static_cast<int32_t>(outputChildren.size()); ++i) {
    if (!outputChildren[i]) {
      outputChildren[i] =
          BaseVector::createNullConstant(outputType->childAt(i), numRows, pool);
    }
  }

  // Release rows from their input batches.
  std::unordered_map<int32_t, int64_t> batchReleaseCounts;
  for (const auto& loc : locations) {
    batchReleaseCounts[loc.first]++;
  }
  for (const auto& [batchIdx, count] : batchReleaseCounts) {
    state_->releaseRows(batchIdx, count);
  }

  return std::make_shared<RowVector>(
      pool, outputType, nullptr, numRows, std::move(outputChildren));
}

} // namespace facebook::velox::exec::rpc
