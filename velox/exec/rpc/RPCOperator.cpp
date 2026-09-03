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
#include "velox/exec/rpc/BackendErrorSummary.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

#define RPC_OP_LOG(severity) LOG(severity) << "[RPC_OP] "
#define RPC_OP_VLOG(level) VLOG(level) << "[RPC_OP] "

namespace facebook::velox::exec::rpc {

namespace {

// Identifies the operator-level batch fan-out as the source of a row's error.
constexpr std::string_view kBatchErrorPrefix = "[RPC_BATCH] batch error: ";

} // namespace

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

  // Walk the RPC call's argument expressions once, in order, to build:
  //  - argumentSources_: how addInput() sources each arg (column vs constant),
  //  - inputTypes: each argument's Velox type,
  //  - constantInputs: single-element constant vector for constant args and
  //    nullptr for column args. This keeps AsyncRPCFunction::initialize()'s
  //    interface unchanged (types + aligned constant values).
  auto sourceType = rpcNode_->source()->outputType();
  const auto& callInputs = rpcNode_->call()->inputs();
  argumentSources_.reserve(callInputs.size());
  std::vector<TypePtr> inputTypes;
  std::vector<VectorPtr> constantInputs;
  inputTypes.reserve(callInputs.size());
  constantInputs.reserve(callInputs.size());
  auto* pool = operatorCtx_->pool();
  for (const auto& input : callInputs) {
    inputTypes.push_back(input->type());
    if (auto* field = input->asUnchecked<core::FieldAccessTypedExpr>()) {
      const auto idx = sourceType->getChildIdx(field->name());
      argumentSources_.push_back(
          ArgumentSource{
              .isConstant = false,
              .sourceChannel = static_cast<column_index_t>(idx),
          });
      constantInputs.push_back(nullptr);
    } else if (auto* constant = input->asUnchecked<core::ConstantTypedExpr>()) {
      auto constantValue = constant->toConstantVector(pool);
      constantInputs.push_back(constantValue);
      argumentSources_.push_back(
          ArgumentSource{
              .isConstant = true,
              .constantValue = std::move(constantValue),
          });
    } else {
      VELOX_FAIL(
          "RPC call argument must be a FieldAccessTypedExpr or "
          "ConstantTypedExpr, got: {}",
          input->toString());
    }
  }

  // Initialize the function with query config, argument types, and constants.
  // The function creates/caches its own transport and clients internally.
  function_->initialize(
      operatorCtx_->driverCtx()->queryConfig(), inputTypes, constantInputs);

  tierKey_ = function_->tierKey();

  const auto& queryConfig = operatorCtx_->driverCtx()->queryConfig();

  // Size output vectors from config; see getOutput().
  outputBatchRows_ = queryConfig.preferredOutputBatchRows();

  limiter_ = &RPCRateLimiterRegistry::global().get(tierKey_);
  // A backend's configuration is fixed by the first query to reach it and is
  // shared by every query after. The limiter is a controller: its policy has
  // to hold still while it adapts, and the adaptation itself is learned from
  // all of them jointly. A later query asking for something different is
  // logged and ignored rather than allowed to move the target mid-flight.
  //
  // One call, so one query's whole view lands or none of it does. Applying the
  // function's option and the session properties as two writes left a window
  // where a second query could interleave and fix a mixture of the two.
  limiter_->initializeOnce([this,
                            &queryConfig](RPCRateLimiter::Config& config) {
    if (const auto fnCeiling = function_->configuredCeiling(); fnCeiling > 0) {
      config.ceiling = fnCeiling;
    }
    config.adaptive = queryConfig.rpcRateLimiterAdaptiveEnabled();
    config.floor = queryConfig.rpcRateLimiterMinLimit();
    config.decreaseFactor = queryConfig.rpcRateLimiterDecreaseFactor();
    // 0 keeps whatever the function asked for; any positive value wins.
    if (const auto rlMax = queryConfig.rpcRateLimiterMaxLimit(); rlMax > 0) {
      config.ceiling = rlMax;
    }
  });

  RPC_OP_VLOG(1) << "Created operator for function '"
                 << rpcNode_->functionName() << "', planNodeId=" << planNodeId()
                 << ", operatorId=" << operatorId() << ", streamingMode="
                 << (rpcNode_->streamingMode() == RPCStreamingMode::kBatch
                         ? "BATCH"
                         : "PER_ROW");

  if (!argumentSources_.empty()) {
    RPC_OP_VLOG(1) << "Initialized with " << argumentSources_.size()
                   << " call arguments";
  } else {
    RPC_OP_VLOG(1) << "Initialized with no call arguments "
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

  // Don't take more input while this driver already holds as much as it
  // should buffer. This bounds memory; whether the tier can take a flush
  // right now is isBlocked()'s question, not this one.
  if (inputBufferIsFull()) {
    return false;
  }

  // Check per-state backpressure.
  if (state_->isUnderBackpressure()) {
    // NOTE: in BATCH mode this back-pressure is not paired with a blocking
    // isBlocked() future -- see the TODO in isBlocked().
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

  // Build per-call arguments in call()->inputs() order: a column argument reads
  // the source column; a constant argument wraps its single-element constant to
  // the batch row count, so the function sees the same arg list (order + count)
  // as it would for a column. Keeps null-input handling identical.
  std::vector<VectorPtr> args;
  if (!argumentSources_.empty()) {
    args.reserve(argumentSources_.size());
    for (const auto& argSource : argumentSources_) {
      if (argSource.isConstant) {
        args.push_back(
            BaseVector::wrapInConstant(
                input->size(), 0, argSource.constantValue));
      } else {
        args.push_back(input->childAt(argSource.sourceChannel));
      }
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
    // control (dispatchRowsUnderAdmission) instead of dispatching the whole
    // vector at once, which would overrun both the per-driver window and the
    // shared admission control. needsInput() returns false until this buffer is
    // drained, so exactly one input vector is buffered at a time.
    pendingArgs_ = std::move(args);
    pendingNumRows_ = static_cast<vector_size_t>(input->size());
    pendingCursor_ = 0;
    pendingBatchIndex_ = state_->storeInputBatch(
        std::move(flattenedColumns), static_cast<int64_t>(pendingNumRows_));
    dispatchRowsUnderAdmission();
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

    // Flush in chunks of dispatchBatchSize_ rather than one giant
    // batch_predict call that would overwhelm the server.
    dispatchBatchUnderAdmission(DispatchScope::kFullChunksOnly);
  }
}

void RPCOperator::dispatchRowsUnderAdmission() {
  while (hasPendingRows()) {
    // The per-driver congestion window bounds this driver; the tier's capacity
    // bounds every driver sharing the backend. Both must have room.
    const int64_t windowHeadroom = state_->dispatchHeadroom();
    if (windowHeadroom <= 0) {
      break;
    }
    const auto remaining =
        static_cast<int64_t>(pendingNumRows_ - pendingCursor_);
    const int64_t want = std::min(windowHeadroom, remaining);

    // Reserve the tier's slots BEFORE dispatching, one token per row, and send
    // exactly what was granted. Sizing the chunk against available() and
    // acquiring after the RPC is out lets N drivers each measure the same free
    // capacity and all dispatch against it, overshooting the cap by roughly
    // the driver count.
    // One grant for the whole chunk: reserving row by row would take the
    // backend's exclusive lock once per row, and that lock also carries
    // available(), waitForCapacity() and the adaptation callbacks.
    auto reserved = limiter_->tryAcquireUpTo(want);
    if (reserved.empty()) {
      break;
    }
    const auto numRowsInChunk = static_cast<vector_size_t>(reserved.size());

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
    size_t reservedIndex = 0;
    for (auto& [originalRowIndex, future] : futures) {
      auto rowId = globalRowIdCounter_++;
      // One pre-reserved slot per row, handed to the continuation so it is
      // released when the row completes.
      auto token = std::make_shared<RPCRateLimiter::Token>(
          std::move(reserved[reservedIndex++]));
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

bool RPCOperator::flushBatchRequests(int32_t maxRows) {
  if (function_->pendingBatchSize() == 0) {
    VELOX_CHECK(
        batchRowLocations_.empty(),
        "Operator has {} accumulated batch rows but function reports "
        "pendingBatchSize=0. Function must override pendingBatchSize() "
        "when using BATCH mode.",
        batchRowLocations_.size());
    return false;
  }

  // Reserve the tier slot before touching any state: the accumulator is only
  // split and handed to flushBatch() once this flush is admitted. Checking
  // available() and acquiring after the call is out lets concurrent drivers
  // overshoot the cap.
  auto reserved = limiter_->tryAcquireUpTo(1);
  if (reserved.empty()) {
    return false;
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

  // Each flushBatch() is 1 pending unit against tier capacity, reserved above.
  auto token =
      std::make_shared<RPCRateLimiter::Token>(std::move(reserved.front()));

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
            // Summarize the backend text once, here, rather than copying a
            // server-side stack trace into every row (see
            // summarizeBackendError). The untruncated text stays in the log
            // line above.
            auto errResponses = makeBatchErrorResponses(
                rowIds.size(), kBatchErrorPrefix, ew.what().toStdString());
            for (size_t i = 0; i < errResponses.size(); ++i) {
              errResponses[i].rowId = static_cast<int64_t>(i);
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
  return true;
}

void RPCOperator::noMoreInput() {
  exec::Operator::noMoreInput();

  RPC_OP_VLOG(1) << "noMoreInput: totalRequestsDispatched="
                 << numRequestsDispatched_;

  // BATCH flushes its accumulator here; PER_ROW has nothing buffered at this
  // point (see hasUndispatchedWork()).
  drainPending();

  // Only declare the input closed once nothing is left undispatched. Anything
  // admission held back goes out from isBlocked() as slots free, and
  // signalling here would let the finish condition fire while rows are still
  // waiting to be sent.
  if (!hasUndispatchedWork()) {
    state_->setNoMoreInput();
  }
}

bool RPCOperator::inputBufferIsFull() const {
  if (state_->streamingMode() == RPCStreamingMode::kPerRow) {
    return hasPendingRows();
  }
  // Depth only. Deliberately not gated on the tier's free capacity, which
  // churns as tokens release: throttling intake on an instantaneous zero both
  // stalls the pipeline and ignores this driver's own in-flight work that is
  // about to free a slot. dispatchBatchSize_ of 0 means buffer-everything, so
  // there is no depth to bound.
  return dispatchBatchSize_ > 0 &&
      function_->pendingBatchSize() >= kBufferedChunks * dispatchBatchSize_;
}

bool RPCOperator::hasUndispatchedWork() const {
  return state_->streamingMode() == RPCStreamingMode::kPerRow
      ? hasPendingRows()
      : function_->pendingBatchSize() > 0;
}

void RPCOperator::drainPending() {
  if (state_->streamingMode() == RPCStreamingMode::kPerRow) {
    dispatchRowsUnderAdmission();
  } else {
    dispatchBatchUnderAdmission(DispatchScope::kEverything);
  }
}

std::optional<exec::BlockingReason> RPCOperator::drainOrParkOnAdmission(
    ContinueFuture* future) {
  if (!hasUndispatchedWork()) {
    return std::nullopt;
  }
  drainPending();
  if (!hasUndispatchedWork()) {
    // Everything went out, so the input can finally be declared closed --
    // noMoreInput() and startDrain() defer that until exactly this point.
    state_->setNoMoreInput();
    return std::nullopt;
  }
  if (state_->numInFlight() == 0) {
    // Nothing of ours is in flight, so only another driver's release can help.
    // admitOrWait() decides and enrols under one lock, so a slot freeing here
    // cannot leave us waiting on nothing, and this never falls through to a
    // wait that nothing could fulfil.
    auto admission = limiter_->admitOrWait();
    if (admission.admitted) {
      // Room appeared; come back round and dispatch into it.
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(admission.wait);
    blockWaitStartNs_ = getCurrentTimeNano();
    blockWaitIsBackpressure_ = true;
    return exec::BlockingReason::kWaitForRPC;
  }
  // Our own completions will free slots and wake the wait below.
  return std::nullopt;
}

void RPCOperator::dispatchBatchUnderAdmission(DispatchScope scope) {
  int32_t minRows = 1;
  if (scope == DispatchScope::kFullChunksOnly) {
    if (dispatchBatchSize_ <= 0) {
      // Dispatch-once mode: nothing is a full chunk until the input closes.
      return;
    }
    minRows = dispatchBatchSize_;
  }
  const auto chunk = dispatchBatchSize_ > 0 ? dispatchBatchSize_ : 0;
  // Both gates, the same pair dispatchRowsUnderAdmission() applies: the
  // per-driver window, and the tier's shared capacity. Checking only the window
  // lets a driver whose window is open keep flushing while other drivers have
  // already exhausted the tier, which is the over-admission this exists to
  // stop. The tier slot is reserved inside flushBatchRequests(), which returns
  // false when the backend is full, so this stops instead of overshooting.
  while (function_->pendingBatchSize() >= minRows &&
         !state_->isUnderBackpressure() && flushBatchRequests(chunk)) {
  }
}

RowVectorPtr RPCOperator::getOutput() {
  auto streamingMode = state_->streamingMode();

  if (streamingMode == RPCStreamingMode::kPerRow) {
    // Drip more buffered rows now that in-flight completions may have
    // freed window / tier capacity.
    dispatchRowsUnderAdmission();

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
    state_->drainReadyRows(claimedRows_, outputBatchRows_);

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
    //  - Rate limiter (one per provisioned capacity): halve the cap on
    //    overload,
    //    additive-recover on success.
    // The policy classifies overload as rate-limit / timeout / majority error
    // (ignoring null_input). Both scopes must back off on it: a rate-limit
    // storm is LOW-latency, so the latency gradient alone is blind to it — the
    // error verdict is what makes the per-driver window back off, not just
    // latency.
    const auto signal = function_->evaluateCongestion(responses);
    if (signal == AsyncRPCFunction::CongestionSignal::kError) {
      state_->onUnitError();
      limiter_->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    } else if (signal == AsyncRPCFunction::CongestionSignal::kSuccess) {
      // Feed the whole drained batch of successful RTTs to the gradient in one
      // lock acquisition; its size is the success count driving AIMD recovery.
      state_->onUnitSamples(roundTripTimesNs);
      limiter_->onOutcome(
          RPCRateLimiter::Outcome::kSuccess,
          static_cast<int64_t>(roundTripTimesNs.size()));
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
    // the batch RTT to the latency gradient; tier capacity (shared) halves
    // the cap on overload and recovers on success.
    const auto signal = function_->evaluateCongestion(claimedBatch_->responses);
    if (signal == AsyncRPCFunction::CongestionSignal::kError) {
      state_->onUnitError();
      limiter_->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    } else if (signal == AsyncRPCFunction::CongestionSignal::kSuccess) {
      // Feed the measured round-trip latency to the gradient window so it
      // learns the in-flight-batch sweet spot without a fixed ceiling.
      state_->onUnitSample(claimedBatch_->rttNs);
      // Successful rows in this batch drive AIMD recovery of the backend's
      // shared cap.
      limiter_->onOutcome(
          RPCRateLimiter::Outcome::kSuccess, numRows - batchErrors);
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
  // must never park behind the backend's shared cap held by OTHER drivers.
  // That wait is applied as a last resort below, only when
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
      dispatchRowsUnderAdmission();
      auto claimedRow = state_->tryClaimReady();
      if (claimedRow) {
        claimedRows_.push_back(std::move(*claimedRow));
        return exec::BlockingReason::kNotBlocked;
      }
      // No ready output. Wait on the per-state completion future ONLY when this
      // operator has in-flight rows — those are guaranteed to fire that future.
      // If rows are buffered but nothing is in-flight, they are blocked solely
      // on the backend's shared admission capacity (held by other
      // drivers); the per-state future would never resolve, so returning
      // kWaitForRPC here would hang the driver. Instead fall through to
      // kNotBlocked and let the next isBlocked() re-check
      // RPCRateLimiter::admitOrWait(), which a slot-freeing release on
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
      // the backend's shared cap (its slots held by other drivers). Park
      // on the tier's waiter queue — woken by any driver's slot-freeing release
      // — rather than busy-spinning via repeated kNotBlocked.
      if (hasPendingRows()) {
        auto admission = limiter_->admitOrWait();
        if (admission.admitted) {
          // Room appeared; come back round and dispatch into it.
          return exec::BlockingReason::kNotBlocked;
        }
        *future = std::move(admission.wait);
        blockWaitStartNs_ = getCurrentTimeNano();
        blockWaitIsBackpressure_ = true;
        return exec::BlockingReason::kWaitForRPC;
      }
      return exec::BlockingReason::kNotBlocked;
    }

    // Resume the drain: rows admission held back still have to go out, and
    // noMoreInput()/startDrain() deferred the end-of-input signal until they
    // do. Mirrors the BATCH path below. Completions free slots, and this runs
    // on the same wake-ups that deliver them.
    if (auto reason = drainOrParkOnAdmission(future)) {
      return reason.value();
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
      // A completion may have freed a slot. BATCH otherwise only flushes on
      // new input, so once needsInput() stops taking input a full accumulator
      // would sit here indefinitely with capacity going unused.
      dispatchBatchUnderAdmission(DispatchScope::kFullChunksOnly);

      auto readyBatch = state_->tryPollReady();
      if (readyBatch) {
        if (readyBatch->error.has_value()) {
          RPC_OP_LOG(WARNING)
              << "Received batch with error: " << readyBatch->error.value();
        }
        claimedBatch_ = std::move(*readyBatch);
        return exec::BlockingReason::kNotBlocked;
      }
      // No ready batch. Under back-pressure (in-flight batches at the window
      // limit, so needsInput() returns false), PARK on an in-flight batch
      // rather than returning kNotBlocked: otherwise a driver that halts its
      // upstream walk at this full operator (transitive back-pressure) would
      // busy-spin here until a batch completes, monopolizing its thread and
      // starving co-scheduled queries. When not under back-pressure we can
      // still accept input, so report not-blocked and let the driver call
      // needsInput()/addInput(). tryPollBatchOrWait registers a waiter that a
      // batch completion fulfills, so this cannot hang while batches are
      // in-flight (guaranteed by isUnderBackpressure()).
      if (state_->isUnderBackpressure()) {
        ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
        std::optional<RPCState::ReadyBatch> polledBatch;
        switch (state_->tryPollBatchOrWait(&waitFuture, &polledBatch)) {
          case RPCState::BatchPollResult::kGotBatch:
            if (polledBatch->error.has_value()) {
              RPC_OP_LOG(WARNING) << "Received batch with error: "
                                  << polledBatch->error.value();
            }
            claimedBatch_ = std::move(*polledBatch);
            return exec::BlockingReason::kNotBlocked;
          case RPCState::BatchPollResult::kMustWait:
            *future = std::move(waitFuture);
            blockWaitStartNs_ = getCurrentTimeNano();
            blockWaitIsBackpressure_ = false;
            return exec::BlockingReason::kWaitForRPC;
          case RPCState::BatchPollResult::kFinished:
            // Not expected mid-stream (noMoreInput_ is false); fall through to
            // not-blocked defensively.
            break;
        }
      }

      // Admission refused a flush this driver is otherwise ready to make.
      // Yield the thread rather than report not-blocked: needsInput() also
      // refuses while the accumulator holds an unflushable chunk, so the
      // driver would come straight back here with nothing to do.
      if (dispatchBatchSize_ > 0 &&
          function_->pendingBatchSize() >= dispatchBatchSize_ &&
          limiter_->available() == 0) {
        // Same order as the PER_ROW path: prefer this driver's own in-flight
        // batches, whose completion both frees a slot and is guaranteed to
        // fire. Only with nothing of ours in flight is the cap held entirely
        // by other drivers, and the tier's waiter queue the one thing that can
        // wake us.
        if (state_->numInFlight() > 0) {
          ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
          std::optional<RPCState::ReadyBatch> polledBatch;
          switch (state_->tryPollBatchOrWait(&waitFuture, &polledBatch)) {
            case RPCState::BatchPollResult::kGotBatch:
              if (polledBatch->error.has_value()) {
                RPC_OP_LOG(WARNING) << "Received batch with error: "
                                    << polledBatch->error.value();
              }
              claimedBatch_ = std::move(*polledBatch);
              return exec::BlockingReason::kNotBlocked;
            case RPCState::BatchPollResult::kMustWait:
              *future = std::move(waitFuture);
              blockWaitStartNs_ = getCurrentTimeNano();
              blockWaitIsBackpressure_ = true;
              return exec::BlockingReason::kWaitForRPC;
            case RPCState::BatchPollResult::kFinished:
              break;
          }
        } else {
          auto admission = limiter_->admitOrWait();
          if (admission.admitted) {
            // Room appeared; come back round and dispatch into it.
            return exec::BlockingReason::kNotBlocked;
          }
          *future = std::move(admission.wait);
          blockWaitStartNs_ = getCurrentTimeNano();
          blockWaitIsBackpressure_ = true;
          return exec::BlockingReason::kWaitForRPC;
        }
      }
      return exec::BlockingReason::kNotBlocked;
    }

    // Resume the drain: noMoreInput() dispatched only what admission allowed,
    // so anything still accumulated is waiting on a slot. Completions free
    // slots, and this runs on the same wake-ups that deliver them.
    if (auto reason = drainOrParkOnAdmission(future)) {
      return reason.value();
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

  // Send what admission allows; whatever does not fit goes out from
  // isBlocked() as slots free.
  drainPending();

  // Same rule as noMoreInput(): declare the input closed only once nothing is
  // left undispatched, or the finish condition could fire while rows are
  // still waiting to be sent.
  if (!hasUndispatchedWork()) {
    state_->setNoMoreInput();
  } else {
    // Those rows still have to go out, so there is buffered data to drain.
    return true;
  }

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

  // Release resources explicitly. RPCState may be held alive by in-flight RPC
  // callbacks (via shared_ptr capture), so dropping our reference is not enough
  // to free the input vectors: those belong to upstream operators' memory pools
  // and must be released here, on the driver thread, while those pools are
  // still alive. Otherwise the retained reservation makes the arbitrator's
  // reservedBytes() == 0 check throw from ~MemoryPoolImpl() and terminate the
  // worker, or a late callback frees into pools that are already gone.
  if (state_ != nullptr) {
    state_->releaseAllInputBatches();
  }
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

  // Per-tier admission capacity trajectory (the shared capacity this
  // operator drips against). Distinct from the per-driver rpcCongestion*
  // window. Emitted unconditionally — including for the empty/default tier,
  // which is the bucket the meta.ai per-row-key path uses; gating on a
  // non-empty tierKey_ would hide the cap on exactly that main path.
  const auto limiterStats = limiter_->stats();
  lockedStats->addRuntimeStat(
      kRpcRateLimiterCap, RuntimeCounter(limiterStats.capacity));
  lockedStats->addRuntimeStat(
      kRpcRateLimiterPeakPending, RuntimeCounter(limiterStats.peakPending));
  if (limiterStats.lowWaterCapacity > 0) {
    lockedStats->addRuntimeStat(
        kRpcRateLimiterMinCap, RuntimeCounter(limiterStats.lowWaterCapacity));
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
