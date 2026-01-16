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
#include "velox/experimental/cudf-exchange/CudfExchange.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::core;

namespace facebook::velox::cudf_exchange {

CudfExchange::CudfExchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::PlanNode>& planNode,
    std::shared_ptr<CudfExchangeClient> exchangeClient,
    const std::string& operatorType)
    : SourceOperator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          operatorType),
      processSplits_{operatorCtx_->driverCtx()->driverId == 0},
      driverId_{driverCtx->driverId} {
  if (exchangeClient) {
    // exchangeClient is provided externally when this is a "plain"
    // CudfExchange.
    exchangeClient_ = std::move(exchangeClient);
  } else {
    // exchangeClient is nullptr when this CudfExchange is used to implement a
    // MergeExchange. Create a new exchange client.
    auto task = operatorCtx_->task();
    exchangeClient_ = std::make_shared<CudfExchangeClient>(
        task->taskId(), task->destination(), 1);
  }
}

void CudfExchange::addRemoteTaskIds(std::vector<std::string>& remoteTaskIds) {
  // shuffle is just so that the order is random to avoid over load

  std::shuffle(std::begin(remoteTaskIds), std::end(remoteTaskIds), rng_);
  for (const std::string& remoteTaskId : remoteTaskIds) {
    exchangeClient_->addRemoteTaskId(remoteTaskId);
    VLOG(3) << "@" << taskId()
            << " CudfExchange::addRemoteTasksIds: " << remoteTaskId;
  }
  stats_.wlock()->numSplits += remoteTaskIds.size();
}

bool CudfExchange::getSplits(ContinueFuture* future) {
  if (!processSplits_) {
    return false;
  }
  if (noMoreSplits_) {
    return false;
  }
  std::vector<std::string> remoteTaskIds;
  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId(), split, *future);
    if (reason == BlockingReason::kNotBlocked) {
      if (split.hasConnectorSplit()) {
        auto remoteSplit = std::dynamic_pointer_cast<RemoteConnectorSplit>(
            split.connectorSplit);
        VELOX_CHECK_NOT_NULL(remoteSplit, "Wrong type of split");
        remoteTaskIds.push_back(remoteSplit->taskId);
      } else {
        addRemoteTaskIds(remoteTaskIds);
        exchangeClient_->noMoreRemoteTasks();
        noMoreSplits_ = true;
        if (atEnd_) {
          operatorCtx_->task()->multipleSplitsFinished(
              false, stats_.rlock()->numSplits, 0);
          recordExchangeClientStats();
        }
        return false;
      }
    } else {
      addRemoteTaskIds(remoteTaskIds);
      return true;
    }
  }
}

BlockingReason CudfExchange::isBlocked(ContinueFuture* future) {
  // check whether there is data or whether this is at the end.
  if (currentData_ != nullptr || atEnd_) {
    return BlockingReason::kNotBlocked;
  }

  // Get splits from the task if no splits are outstanding.
  if (!splitFuture_.valid()) {
    getSplits(&splitFuture_);
  }

  // No data! Ask the client for the next packed table.
  VELOX_CHECK_NULL(currentData_);
  ContinueFuture dataFuture;
  currentData_ = exchangeClient_->next(driverId_, &atEnd_, &dataFuture);
  if (currentData_ != nullptr || atEnd_) {
    if (atEnd_ && noMoreSplits_) {
      const auto numSplits = stats_.rlock()->numSplits;
      operatorCtx_->task()->multipleSplitsFinished(false, numSplits, 0);
    }
    recordExchangeClientStats();
    return BlockingReason::kNotBlocked;
  }

  VELOX_CHECK(dataFuture.valid());
  if (splitFuture_.valid()) {
    // Block until data becomes available or more splits arrive.
    std::vector<ContinueFuture> futures;
    futures.push_back(std::move(splitFuture_));
    futures.push_back(std::move(dataFuture));
    *future = folly::collectAny(futures).unit();
    return BlockingReason::kWaitForSplit;
  }

  // Not waiting for more splits.
  // Block until data becomes available.
  *future = std::move(dataFuture);
  return BlockingReason::kWaitForProducer;
}

bool CudfExchange::isFinished() {
  return atEnd_ && currentData_ == nullptr;
}

RowVectorPtr CudfExchange::getOutput() {
  VLOG(3) << "@" << taskId() << " CudfExchange::getOutput() has data: "
          << (currentData_ != nullptr);
  if (currentData_ == nullptr) {
    return nullptr;
  }

  // Get the packed_table and stream from the PackedTableWithStream
  auto numRows = currentData_->packedTable->table.num_rows();
  auto gpuDataSize = currentData_->gpuDataSize();

  // Use the stream that was allocated in CudfExchangeSource::onMetadata
  // and the packed_table constructor of CudfVector to avoid copying data.
  auto result = std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      numRows,
      std::move(currentData_->packedTable),
      currentData_->stream);

  recordInputStats(gpuDataSize, result);
  // free the memory owned by PackedTableWithStream and set it to nullptr;
  currentData_.reset();

  return result;
}

void CudfExchange::recordInputStats(
    uint64_t rawInputBytes,
    RowVectorPtr result) {
  auto lockedStats = stats_.wlock();
  lockedStats->rawInputBytes += rawInputBytes;
  // size(): number of rows in result_ vector
  lockedStats->rawInputPositions += result->size();
  lockedStats->addInputVector(result->estimateFlatSize(), result->size());
}

void CudfExchange::close() {
  SourceOperator::close();
  if (currentData_ != nullptr) {
    currentData_.reset();
  }
  if (exchangeClient_) {
    recordExchangeClientStats();
    exchangeClient_->close();
  }
  exchangeClient_ = nullptr;
}

void CudfExchange::recordExchangeClientStats() {
  if (!processSplits_) {
    return;
  }

  auto lockedStats = stats_.wlock();
  const auto exchangeClientStats = exchangeClient_->stats();
  for (const auto& [name, value] : exchangeClientStats) {
    lockedStats->runtimeStats.erase(name);
    lockedStats->runtimeStats.insert({name, value});
  }

  auto backgroundCpuTimeMs =
      exchangeClientStats.find(ExchangeClient::kBackgroundCpuTimeMs);
  if (backgroundCpuTimeMs != exchangeClientStats.end()) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(backgroundCpuTimeMs->second.count),
        0,
        static_cast<uint64_t>(backgroundCpuTimeMs->second.sum) *
            Timestamp::kNanosecondsInMillisecond};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }
}

} // namespace facebook::velox::cudf_exchange
