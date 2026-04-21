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
#include "velox/experimental/ucx-exchange/UcxExchange.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

using facebook::velox::exec::Operator;
using facebook::velox::exec::RemoteConnectorSplit;
// Required by VELOX_NVTX_OPERATOR_FUNC_RANGE macro in NvtxHelper.h which
// references extractClassAndFunction and VeloxDomain unqualified.
using namespace facebook::velox::cudf_velox; // NOLINT

namespace facebook::velox::ucx_exchange {

// --- Implementation of the UcxExchange operator.

UcxExchange::UcxExchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::PlanNode>& planNode,
    std::shared_ptr<UcxExchangeClient> ucxExchangeClient,
    std::string_view operatorType)
    : SourceOperator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          operatorType),
      NvtxHelper(
          nvtx3::rgb{0, 191, 255},
          operatorId,
          fmt::format("[{}]", planNode->id())),
      preferredOutputBatchBytes_{
          driverCtx->queryConfig().preferredOutputBatchBytes()},
      processSplits_{driverCtx->driverId == 0},
      pipelineId_{driverCtx->pipelineId},
      driverId_{driverCtx->driverId} {
  if (ucxExchangeClient) {
    // UcxExchangeClient is provided externally when this is a "plain"
    // UcxExchange.
    exchangeClient_ = std::move(ucxExchangeClient);
  } else {
    // UcxExchangeClient is nullptr when this UcxExchange is used to
    // implement a MergeExchange. Create a new UCX exchange client.
    auto task = operatorCtx_->task();
    exchangeClient_ = std::make_shared<UcxExchangeClient>(
        task->taskId(),
        task->destination(),
        1 // number of consumers, is always 1.
    );
  }
}

UcxExchange::~UcxExchange() {
  close();
}

void UcxExchange::addRemoteTaskIds(std::vector<std::string>& remoteTaskIds) {
  std::shuffle(std::begin(remoteTaskIds), std::end(remoteTaskIds), rng_);
  for (const std::string& remoteTaskId : remoteTaskIds) {
    exchangeClient_->addRemoteTaskId(remoteTaskId);
  }
  stats_.wlock()->numSplits += remoteTaskIds.size();
}

void UcxExchange::getSplits(ContinueFuture* future) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " getSplits called for task: " << taskId();
  if (!processSplits_) {
    VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
            << " getSplits: Not allowed to process splits for task: "
            << taskId();
    return;
  }
  if (noMoreSplits_) {
    VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
            << " getSplits: No more splits for task: " << taskId();
    return;
  }
  std::vector<std::string> remoteTaskIds;
  // loop until we get an end marker.
  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->driverId,
        operatorCtx_->driverCtx()->splitGroupId,
        planNodeId(),
        /*maxPreloadSplits=*/0,
        /*preload=*/nullptr,
        split,
        *future);
    if (reason != BlockingReason::kNotBlocked) {
      // we are blocked. Add the splits collected so far (if any) and return.
      addRemoteTaskIds(remoteTaskIds);
      return;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          std::dynamic_pointer_cast<RemoteConnectorSplit>(split.connectorSplit);
      VELOX_CHECK_NOT_NULL(remoteSplit, "Wrong type of split");
      remoteTaskIds.push_back(remoteSplit->taskId);
      // check for more splits.
      continue;
    }

    // not blocked but also no connector split.
    // we've reached the end.
    addRemoteTaskIds(remoteTaskIds);
    exchangeClient_->noMoreRemoteTasks();
    noMoreSplits_ = true;
    if (atEnd_) {
      operatorCtx_->task()->multipleSplitsFinished(
          false, stats_.rlock()->numSplits, 0);
      recordExchangeClientStats();
    }
    return;
  }
}

BlockingReason UcxExchange::isBlocked(ContinueFuture* future) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (currentData_ || atEnd_) {
    return BlockingReason::kNotBlocked;
  }

  // Get splits from the task if no splits are outstanding.
  if (!splitFuture_.valid()) {
    getSplits(&splitFuture_);
  }

  ContinueFuture dataFuture = ContinueFuture::makeEmpty();
  currentData_ = exchangeClient_->next(driverId_, &atEnd_, &dataFuture);
  if (currentData_ || atEnd_) {
    // got some data or reached the end.
    if (atEnd_ && noMoreSplits_) {
      const auto numSplits = stats_.rlock()->numSplits;
      operatorCtx_->task()->multipleSplitsFinished(false, numSplits, 0);
    }
    recordExchangeClientStats();
    return BlockingReason::kNotBlocked;
  }

  if (splitFuture_.valid()) {
    // Combine the futures and block until data becomes available or more splits
    // arrive.
    std::vector<ContinueFuture> futures;
    futures.push_back(std::move(splitFuture_));
    futures.push_back(std::move(dataFuture));
    *future = folly::collectAny(futures).unit();
    return BlockingReason::kWaitForSplit;
  }

  // Block until data becomes available.
  *future = std::move(dataFuture);
  return BlockingReason::kWaitForProducer;
}

bool UcxExchange::isFinished() {
  return atEnd_ && !currentData_;
}

RowVectorPtr UcxExchange::getOutputFromPackedTable() {
  if (!currentData_) {
    return nullptr;
  }

  // Get the packed_table and stream from the PackedTableWithStream
  PackedTableWithStream& data = *currentData_;
  auto numRows = data.packedTable->table.num_rows();
  auto gpuDataSize = data.gpuDataSize();

  // Use the stream that was allocated in UcxExchangeSource::onMetadata
  // and the packed_table constructor of CudfVector to avoid copying data.
  auto result = std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType_, numRows, std::move(data.packedTable), data.stream);

  recordInputStats(gpuDataSize, result);
  // free the memory owned by PackedTableWithStream and set it to nullptr;
  currentData_.reset();

  return result;
}

RowVectorPtr UcxExchange::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  return getOutputFromPackedTable();
}

void UcxExchange::recordInputStats(
    uint64_t rawInputBytes,
    const RowVectorPtr& result) {
  auto lockedStats = stats_.wlock();
  lockedStats->rawInputBytes += rawInputBytes;
  lockedStats->rawInputPositions += result->size();
  lockedStats->addInputVector(result->estimateFlatSize(), result->size());
}

void UcxExchange::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  SourceOperator::close();
  currentData_.reset();
  if (exchangeClient_) {
    recordExchangeClientStats();
    exchangeClient_->close();
  }
  exchangeClient_ = nullptr;
}

void UcxExchange::recordExchangeClientStats() {
  if (!processSplits_) {
    return;
  }

  auto lockedStats = stats_.wlock();
  const auto exchangeClientStats = exchangeClient_->stats();
  for (const auto& [name, value] : exchangeClientStats) {
    lockedStats->runtimeStats.erase(name);
    lockedStats->runtimeStats.insert({name, value});
  }

  const auto iter = exchangeClientStats.find(Operator::kBackgroundCpuTimeNanos);
  if (iter != exchangeClientStats.end()) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(iter->second.count),
        0,
        static_cast<uint64_t>(iter->second.sum)};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }
}

} // namespace facebook::velox::ucx_exchange
