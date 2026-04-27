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
  // Configure RPCState with streaming mode.
  state_->setStreamingMode(rpcNode_->streamingMode());
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
    // PER_ROW: function dispatches individual RPCs and returns futures.
    auto futures = function_->dispatchPerRow(rows, args);

    auto batchIndex = state_->storeInputBatch(
        flattenedColumns, static_cast<int64_t>(futures.size()));
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
          RPCState::RowLocation{batchIndex, originalRowIndex},
          std::move(wrapped));
    }
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

  // Stamp rowIds onto responses.
  auto wrapped =
      std::move(future)
          .within(kBatchRpcTimeout)
          .deferValue([rowIds = std::move(rowIds),
                       token](std::vector<RPCResponse> resps) {
            VELOX_CHECK_EQ(
                resps.size(),
                rowIds.size(),
                "RPC batch response count ({}) does not match row count ({})",
                resps.size(),
                rowIds.size());
            for (size_t i = 0; i < resps.size(); ++i) {
              resps[i].rowId = rowIds[i];
            }
            return resps;
          })
          .deferError([token](folly::exception_wrapper ew) {
            RPC_OP_LOG(ERROR) << "RPC batch failed: " << ew.what();
            return folly::makeSemiFuture<std::vector<RPCResponse>>(
                std::move(ew));
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
    state_->drainReadyRows(claimedRows_, 1024);

    auto numRows = static_cast<int64_t>(claimedRows_.size());
    for (const auto& row : claimedRows_) {
      if (row.response.hasError()) {
        numErrors_++;
      }
    }
    auto output = buildOutputFromReadyRows(claimedRows_);
    numResponsesCollected_ += numRows;
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
    for (const auto& response : claimedBatch_->responses) {
      if (response.hasError()) {
        numErrors_++;
      }
    }

    // Delegate congestion evaluation to the function.
    // The function knows its domain-specific error semantics.
    auto signal = function_->evaluateCongestion(claimedBatch_->responses);
    if (signal == AsyncRPCFunction::CongestionSignal::kError) {
      state_->onBatchError();
    } else if (signal == AsyncRPCFunction::CongestionSignal::kSuccess) {
      state_->onBatchSuccess(function_->congestionRecoveryIncrement());
    }

    auto output = buildOutputFromReadyBatch(*claimedBatch_);
    numResponsesCollected_ += numRows;
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

  // Check per-tier backpressure first.
  if (auto backpressureFuture = RPCRateLimiter::checkBackpressure(tierKey_)) {
    RPC_OP_VLOG(1) << "Backpressure applied for tier '" << tierKey_
                   << "', pending=" << RPCRateLimiter::pendingCount(tierKey_);
    *future = std::move(*backpressureFuture);
    blockWaitStartNs_ = getCurrentTimeNano();
    blockWaitIsBackpressure_ = true;
    return exec::BlockingReason::kWaitForRPC;
  }

  // If we already have output ready, don't block.
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
      auto claimedRow = state_->tryClaimReady();
      if (claimedRow) {
        claimedRows_.push_back(std::move(*claimedRow));
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

void RPCOperator::recordRuntimeStats() {
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      kRpcRequestsDispatched, RuntimeCounter(numRequestsDispatched_));
  lockedStats->addRuntimeStat(
      kRpcResponsesReceived, RuntimeCounter(numResponsesCollected_));
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

  if (totalBlockWaitNanos_ > 0 || numResponsesCollected_ > 0) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(numResponsesCollected_), totalBlockWaitNanos_, 0};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }
}

RowVectorPtr RPCOperator::buildOutputFromReadyRows(
    std::vector<RPCState::ReadyRow>& readyRows) {
  std::vector<RPCResponse> responses;
  responses.reserve(readyRows.size());

  std::vector<std::pair<int32_t, vector_size_t>> locations;
  locations.reserve(readyRows.size());

  for (auto& row : readyRows) {
    responses.push_back(std::move(row.response));
    locations.emplace_back(row.location.batchIndex, row.location.rowIndex);
  }

  return buildOutputVector(responses, locations);
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
