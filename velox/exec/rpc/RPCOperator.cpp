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
      const auto index = sourceType->getChildIdx(field->name());
      argumentSources_.push_back(
          ArgumentSource{
              .isConstant = false,
              .sourceChannel = static_cast<column_index_t>(index),
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
  // The instruction goes in with everything else the function needs: it
  // resolves its backend and how it will serve the instruction on that backend
  // in one place, and the framework never learns which path it picked.
  function_->initialize(
      operatorCtx_->driverCtx()->queryConfig(),
      inputTypes,
      constantInputs,
      rpcNode_->streamingMode());

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
                         : "PER_ROW")
                 << ", dispatchPath="
                 << RpcDispatchPathName::toName(function_->dispatchPath());

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
  std::vector<VectorPtr> arguments;
  if (!argumentSources_.empty()) {
    arguments.reserve(argumentSources_.size());
    for (const auto& argumentSource : argumentSources_) {
      if (argumentSource.isConstant) {
        arguments.push_back(
            BaseVector::wrapInConstant(
                input->size(), 0, argumentSource.constantValue));
      } else {
        arguments.push_back(input->childAt(argumentSource.sourceChannel));
      }
    }
  } else {
    // Fallback: use all input columns as arguments.
    for (auto i = 0; i < input->childrenSize(); ++i) {
      arguments.push_back(input->childAt(i));
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
    pendingArgs_ = std::move(arguments);
    pendingNumRows_ = static_cast<vector_size_t>(input->size());
    pendingCursor_ = 0;
    pendingBatchIndex_ = state_->storeInputBatch(
        std::move(flattenedColumns), static_cast<int64_t>(pendingNumRows_));
    dispatchRowsUnderAdmission();
  } else {
    // BATCH: function accumulates typed data internally.
    auto rowIndices = function_->accumulateBatch(rows, arguments);

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
    // available(), admitOrWait() and the adaptation callbacks.
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
              .deferError([token](folly::exception_wrapper error) {
                return folly::makeSemiFuture<RPCResponse>(std::move(error));
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

namespace {

// Turns a whole-batch failure into one errored response per row, so the per-row
// error policy (meta_ai_on_error) applies downstream instead of the query
// hard-failing. Responses carry batch-position rowIds, so the scatter stamps
// global ids identically to the success path.
std::vector<RPCResponse> degradeBatchFailureToRowErrors(
    const std::vector<int64_t>& rowIds,
    const folly::exception_wrapper& error) {
  // Mirrors the client-layer fan-out but covers every backend and the
  // operator-level timeout uniformly. Both AIMD controllers still back off,
  // since evaluateCongestion reads a batch failure as overload.
  RPC_OP_LOG(ERROR) << "RPC batch failed, " << rowIds.size()
                    << " rows will carry a per-row error: " << error.what();
  std::vector<RPCResponse> errored(rowIds.size());
  for (size_t i = 0; i < rowIds.size(); ++i) {
    // Batch-position rowId, so the scatter stamps global ids the same way it
    // does on the success path.
    errored[i].rowId = static_cast<int64_t>(i);
    errored[i].error =
        std::string("[RPC_BATCH] batch error: ") + error.what().toStdString();
    errored[i].errorKind = velox::rpc::RPCErrorKind::kBackendError;
  }
  return errored;
}

// Reorders responses into batch position using each response's
// function-assigned rowId, then stamps the global rowIds. Functions may return
// out of order, and pairing responses[i] with rowLocations[i] would silently
// mis-map results onto the wrong passthrough rows. Invariant violations are
// fatal by design.
std::vector<RPCResponse> scatterIntoBatchOrder(
    std::vector<RPCResponse> responses,
    const std::vector<int64_t>& rowIds) {
  VELOX_CHECK_EQ(
      responses.size(),
      rowIds.size(),
      "RPC batch response count ({}) does not match row count ({})",
      responses.size(),
      rowIds.size());
  std::vector<RPCResponse> sorted(responses.size());
  std::vector<bool> seen(responses.size(), false);
  for (auto& response : responses) {
    const auto batchIndex = response.rowId;
    VELOX_CHECK_GE(batchIndex, 0);
    VELOX_CHECK_LT(
        static_cast<size_t>(batchIndex),
        rowIds.size(),
        "RPC batch response rowId ({}) out of range (0-{})",
        batchIndex,
        rowIds.size() - 1);
    VELOX_CHECK(
        !seen[static_cast<size_t>(batchIndex)],
        "Duplicate batch response rowId ({})",
        batchIndex);
    seen[static_cast<size_t>(batchIndex)] = true;
    response.rowId = rowIds[static_cast<size_t>(batchIndex)];
    sorted[static_cast<size_t>(batchIndex)] = std::move(response);
  }
  return sorted;
}

} // namespace

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
  // Order matters: deferError runs BEFORE deferValue, so a whole-batch failure
  // is first turned into one errored response per row and then flows through
  // the same scatter as real responses. That keeps the scatter's invariant
  // checks fatal for genuine function-contract violations -- wrong response
  // count, duplicate or out-of-range rowId -- rather than degrading them to
  // NULL rows. Both lambdas hold 'token' only to keep the tier slot reserved
  // until the batch settles.
  auto wrapped =
      std::move(future)
          .within(kBatchRpcTimeout)
          .deferError([rowIdsPtr, token](folly::exception_wrapper error) {
            return degradeBatchFailureToRowErrors(*rowIdsPtr, error);
          })
          .deferValue([rowIdsPtr, token](std::vector<RPCResponse> responses) {
            return scatterIntoBatchOrder(std::move(responses), *rowIdsPtr);
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
    // parkOnTierCapacity() decides and enrols under one lock, so a slot
    // freeing here cannot leave us waiting on nothing, and this never falls
    // through to a wait that nothing could fulfil.
    return parkOnTierCapacity(future);
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

RowVectorPtr RPCOperator::finishIfDrained() {
  if (isDraining() && state_->isFinished()) {
    finished_ = true;
    finishDrain();
  }
  return nullptr;
}

void RPCOperator::recordCongestion(
    AsyncRPCFunction::CongestionSignal signal,
    const std::vector<int64_t>& roundTripTimesNs,
    int64_t successUnits) {
  // Two AIMD controllers at different scopes, both driven by the function's
  // overload verdict (see RPCRateLimiter.h / CongestionController.h and the
  // function's CongestionPolicy):
  //  - the per-driver window halves on overload, else feeds the successful
  //    round trips to its latency gradient;
  //  - the backend's shared cap halves on overload and recovers additively on
  //    success.
  // The policy classifies overload as rate-limit / timeout / majority error,
  // ignoring null_input. Both scopes must back off on it: a rate-limit storm
  // is LOW-latency, so the gradient alone is blind to it and the error verdict
  // is what makes the window shrink.
  if (signal == AsyncRPCFunction::CongestionSignal::kError) {
    state_->onUnitError();
    limiter_->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    return;
  }
  if (signal == AsyncRPCFunction::CongestionSignal::kSuccess) {
    state_->onUnitSamples(roundTripTimesNs);
    limiter_->onOutcome(RPCRateLimiter::Outcome::kSuccess, successUnits);
  }
}

RowVectorPtr RPCOperator::outputPerRow() {
  // Drip more buffered rows now that in-flight completions may have freed
  // window / tier capacity.
  dispatchRowsUnderAdmission();

  if (claimedRows_.empty()) {
    return finishIfDrained();
  }

  // Drain additional ready rows (non-blocking) for batched output. This
  // amortizes RowVector allocation across multiple completed rows.
  state_->drainReadyRows(claimedRows_, outputBatchRows_);

  // Materialize responses, locations and round-trip latencies once -- reused
  // for the congestion signal and the output vector (no extra copy).
  const auto numRows = static_cast<int64_t>(claimedRows_.size());
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
    // Only successful rows feed the gradient. Errored rows (e.g. null_input,
    // client-side rejections) complete without a real round trip, so their
    // artificially small RTT would pull down the per-window minimum and skew
    // the gradient/baseline.
    if (!hasError) {
      roundTripTimesNs.push_back(row.rttNs);
    }
    responses.push_back(std::move(row.response));
    locations.emplace_back(row.location.batchIndex, row.location.rowIndex);
  }

  recordCongestion(
      function_->evaluateCongestion(responses),
      roundTripTimesNs,
      static_cast<int64_t>(roundTripTimesNs.size()));

  auto output = buildOutputVector(responses, locations);
  numResponsesReceived_ += numRows;
  claimedRows_.clear();
  return output;
}

RowVectorPtr RPCOperator::outputBatch() {
  if (!claimedBatch_.has_value()) {
    return finishIfDrained();
  }

  // Fail loudly on batch errors instead of silently dropping rows.
  if (claimedBatch_->error.has_value()) {
    auto error = claimedBatch_->error.value();
    claimedBatch_.reset();
    VELOX_FAIL("RPC batch failed: {}", error);
  }

  const auto numRows = static_cast<int64_t>(claimedBatch_->responses.size());
  for (const auto& response : claimedBatch_->responses) {
    if (response.hasError()) {
      numErrors_++;
      recordErrorKind(response.errorKind);
    }
  }

  // One measured round trip for the whole batch, so the gradient gets a single
  // sample. The cap recovers by one unit, not by the row count: BATCH reserves
  // one slot per flushBatch() regardless of rows, and onSuccess() steps by
  // units/capacity, so crediting rows against a batch-denominated capacity
  // makes recovery accelerate as capacity shrinks.
  const std::vector<int64_t> roundTripTimesNs{claimedBatch_->rttNs};
  recordCongestion(
      function_->evaluateCongestion(claimedBatch_->responses),
      roundTripTimesNs,
      /*successUnits=*/1);

  auto output = buildOutputFromReadyBatch(*claimedBatch_);
  numResponsesReceived_ += numRows;
  claimedBatch_.reset();
  return output;
}

RowVectorPtr RPCOperator::getOutput() {
  return state_->streamingMode() == RPCStreamingMode::kPerRow ? outputPerRow()
                                                              : outputBatch();
}

void RPCOperator::endBlockWait() {
  if (!blockWaitStartNs_.has_value()) {
    return;
  }
  const auto elapsed = getCurrentTimeNano() - blockWaitStartNs_.value();
  if (blockWaitIsBackpressure_) {
    totalBackpressureWaitNanos_ += elapsed;
  } else {
    totalBlockWaitNanos_ += elapsed;
  }
  blockWaitStartNs_ = std::nullopt;
}

exec::BlockingReason RPCOperator::park(
    ContinueFuture* future,
    ContinueFuture waitFuture,
    bool isBackpressure) {
  *future = std::move(waitFuture);
  blockWaitStartNs_ = getCurrentTimeNano();
  blockWaitIsBackpressure_ = isBackpressure;
  return exec::BlockingReason::kWaitForRPC;
}

void RPCOperator::claimBatch(RPCState::ReadyBatch batch) {
  if (batch.error.has_value()) {
    RPC_OP_LOG(WARNING) << "Received batch with error: " << batch.error.value();
  }
  claimedBatch_ = std::move(batch);
}

bool RPCOperator::hasUndispatchableWork() const {
  if (state_->streamingMode() == RPCStreamingMode::kPerRow) {
    return hasPendingRows();
  }
  // A chunk is ready to flush and the tier is refusing it. dispatchBatchSize_
  // of 0 means buffer-everything, so nothing flushes until noMoreInput().
  return dispatchBatchSize_ > 0 &&
      function_->pendingBatchSize() >= dispatchBatchSize_ &&
      limiter_->available() == 0;
}

exec::BlockingReason RPCOperator::parkOnTierCapacity(ContinueFuture* future) {
  auto admission = limiter_->admitOrWait();
  if (admission.admitted) {
    // Room appeared; come back round and dispatch into it.
    return exec::BlockingReason::kNotBlocked;
  }
  return park(future, std::move(admission.wait), /*isBackpressure=*/true);
}

std::optional<exec::BlockingReason> RPCOperator::tryClaimOrParkOnRow(
    ContinueFuture* future) {
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimed;
  switch (state_->tryClaimOrWait(&waitFuture, &claimed)) {
    case RPCState::ClaimResult::kClaimed:
      claimedRows_.push_back(std::move(*claimed));
      return exec::BlockingReason::kNotBlocked;
    case RPCState::ClaimResult::kMustWait:
      return park(future, std::move(waitFuture), /*isBackpressure=*/false);
    case RPCState::ClaimResult::kFinished:
      return std::nullopt;
  }
  VELOX_UNREACHABLE();
}

std::optional<exec::BlockingReason> RPCOperator::tryClaimOrParkOnBatch(
    ContinueFuture* future,
    bool isBackpressure) {
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> polled;
  switch (state_->tryPollBatchOrWait(&waitFuture, &polled)) {
    case RPCState::BatchPollResult::kGotBatch:
      claimBatch(std::move(*polled));
      return exec::BlockingReason::kNotBlocked;
    case RPCState::BatchPollResult::kMustWait:
      return park(future, std::move(waitFuture), isBackpressure);
    case RPCState::BatchPollResult::kFinished:
      return std::nullopt;
  }
  VELOX_UNREACHABLE();
}

exec::BlockingReason RPCOperator::blockedInPerRow(ContinueFuture* future) {
  if (noMoreInput_ || isDraining()) {
    // Resume the drain: rows admission held back still have to go out, and
    // noMoreInput()/startDrain() defer the end-of-input signal until they do.
    // Mirrors blockedInBatch(); defensive for the same reason described on the
    // guard in noMoreInput().
    if (auto reason = drainOrParkOnAdmission(future)) {
      return reason.value();
    }

    // Finishing: nothing more will be dispatched, so the only outcomes are a
    // completed row, a wait on one, or the stream running out.
    if (auto reason = tryClaimOrParkOnRow(future)) {
      return reason.value();
    }
    finished_ = true;
    return exec::BlockingReason::kNotBlocked;
  }

  // A completion may have freed headroom -- try to drip more.
  dispatchRowsUnderAdmission();
  if (auto claimed = state_->tryClaimReady()) {
    claimedRows_.push_back(std::move(*claimed));
    return exec::BlockingReason::kNotBlocked;
  }

  // Nothing ready. Park on whatever can wake us; with rows buffered and
  // nothing in flight that is the tier's queue, since the per-state future
  // would never fire. needsInput() stays false meanwhile, so no new input
  // arrives. Finished is not expected mid-stream, so it falls through.
  if (state_->numInFlight() > 0 || hasUndispatchableWork()) {
    // Prefer this driver's own in-flight work: that completion is guaranteed
    // to fire AND frees a slot, whereas the tier's queue depends on another
    // driver releasing. blockedInBatch() applies the same order.
    if (state_->numInFlight() > 0) {
      if (auto reason = tryClaimOrParkOnRow(future)) {
        return reason.value();
      }
    } else {
      return parkOnTierCapacity(future);
    }
  }
  return exec::BlockingReason::kNotBlocked;
}

exec::BlockingReason RPCOperator::blockedInBatch(ContinueFuture* future) {
  if (noMoreInput_ || isDraining()) {
    // Resume the drain: noMoreInput() dispatched only what admission allowed,
    // so anything still accumulated is waiting on a slot. Completions free
    // slots, and this runs on the same wake-ups that deliver them.
    if (auto reason = drainOrParkOnAdmission(future)) {
      return reason.value();
    }

    if (auto reason = tryClaimOrParkOnBatch(future, /*isBackpressure=*/false)) {
      return reason.value();
    }
    finished_ = true;
    return exec::BlockingReason::kNotBlocked;
  }

  // A completion may have freed a slot. BATCH otherwise only flushes on new
  // input, so once needsInput() stops taking input a full accumulator would
  // sit here indefinitely with capacity going unused.
  dispatchBatchUnderAdmission(DispatchScope::kFullChunksOnly);

  if (auto ready = state_->tryPollReady()) {
    claimBatch(std::move(*ready));
    return exec::BlockingReason::kNotBlocked;
  }

  // Under window back-pressure needsInput() is false, so a driver that halted
  // its upstream walk here would busy-spin until a batch completes,
  // monopolizing its thread. Park on an in-flight batch instead. Off
  // back-pressure we can still accept input, so report not-blocked and let the
  // driver call needsInput()/addInput().
  if (state_->isUnderBackpressure()) {
    if (auto reason = tryClaimOrParkOnBatch(future, /*isBackpressure=*/false)) {
      return reason.value();
    }
    // Finished is not expected mid-stream; fall through defensively.
  }

  // Admission refused a flush this driver is otherwise ready to make. Yield
  // the thread rather than report not-blocked: needsInput() also refuses while
  // the accumulator holds an unflushable chunk, so the driver would come
  // straight back here with nothing to do.
  if (hasUndispatchableWork()) {
    // Same order as blockedInPerRow(): this driver's own in-flight batches
    // first, the tier's queue only when it has none.
    if (state_->numInFlight() > 0) {
      if (auto reason =
              tryClaimOrParkOnBatch(future, /*isBackpressure=*/true)) {
        return reason.value();
      }
    } else {
      return parkOnTierCapacity(future);
    }
  }
  return exec::BlockingReason::kNotBlocked;
}

exec::BlockingReason RPCOperator::isBlocked(ContinueFuture* future) {
  endBlockWait();

  // Emit ready output / report finished BEFORE any backpressure gate: a driver
  // holding completed rows, or with its own completions to harvest, must never
  // park behind the backend's shared cap held by OTHER drivers. That wait is
  // a last resort, taken only when this operator has buffered work and nothing
  // in flight of its own to wake it.
  if (!claimedRows_.empty() || claimedBatch_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }
  if (finished_) {
    return exec::BlockingReason::kNotBlocked;
  }

  return state_->streamingMode() == RPCStreamingMode::kPerRow
      ? blockedInPerRow(future)
      : blockedInBatch(future);
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

  // The framework owns the destination type; the function owns the mapping
  // onto it. Nothing checked they agree, so a function wired to a node
  // declaring a different type produced a RowVector whose child disagreed
  // with its own declared type, and the failure surfaced downstream.
  const auto& declaredType = outputType->childAt(rpcResultOutputChannel_);
  VELOX_CHECK(
      declaredType->equivalent(*function_->resultType()),
      "RPC function '{}' returns {} but the plan declares column '{}' as {}",
      function_->name(),
      function_->resultType()->toString(),
      outputColumn,
      declaredType->toString());

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

  // The backend's admission capacity trajectory: the capacity this operator
  // shares with every other driver dispatching to the same backend, as
  // distinct from the per-driver rpcCongestion* window. Emitted for every
  // backend including the default one, whose key is empty; gating on a
  // non-empty key would hide the cap on exactly the most common path.
  //
  // limiter_ is resolved in initialize(), but Driver::closeOperators() closes
  // every operator whether or not initializeOperators() ran -- a task that
  // terminates during setup reaches close() first. There is no limiter to
  // report in that case, and no stats worth reporting either.
  if (limiter_ != nullptr) {
    const auto limiterStats = limiter_->stats();
    lockedStats->addRuntimeStat(
        kRpcRateLimiterCap, RuntimeCounter(limiterStats.capacity));
    lockedStats->addRuntimeStat(
        kRpcRateLimiterPeakPending, RuntimeCounter(limiterStats.peakPending));
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
        const auto& [batchIndex, rowInBatch] = locations[rowIdx];
        auto it = batchColsCache.find(batchIndex);
        if (it == batchColsCache.end()) {
          it =
              batchColsCache
                  .emplace(batchIndex, state_->getInputBatchColumns(batchIndex))
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
  for (const auto& [batchIndex, count] : batchReleaseCounts) {
    state_->releaseRows(batchIndex, count);
  }

  return std::make_shared<RowVector>(
      pool, outputType, nullptr, numRows, std::move(outputChildren));
}

} // namespace facebook::velox::exec::rpc
