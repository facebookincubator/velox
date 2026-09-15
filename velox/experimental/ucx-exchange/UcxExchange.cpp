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
#include "velox/common/Casts.h"
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
      driverId_{driverCtx->driverId},
      planNodeBoundary_{
          planNode->is<core::MergeExchangeNode>()
              ? core::PlanNode::Boundary::kInput
              : core::PlanNode::Boundary::kBoth},
      forwardsSplits_{planNode->is<core::MergeExchangeNode>()} {
  VELOX_CHECK_NOT_NULL(
      ucxExchangeClient,
      "UCX exchange client is null, plan node: {}",
      planNode->id());
  exchangeClient_ = std::move(ucxExchangeClient);
}

UcxExchange::~UcxExchange() {
  close();
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
      return;
    }

    if (split.hasConnectorSplit()) {
      if (forwardsSplits_) {
        exchangeClient_->addRemoteTaskId(
            checkedPointerCast<RemoteConnectorSplit>(split.connectorSplit)
                ->taskId);
      }
      ++stats_.wlock()->numSplits;
      // check for more splits.
      continue;
    }

    // not blocked but also no connector split.
    // we've reached the end.
    if (forwardsSplits_) {
      exchangeClient_->noMoreRemoteTasks();
    }
    noMoreSplits_ = true;
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
    if (atEnd_ && !noMoreSplits_) {
      // The task stores 'no more splits' before it tells the client, so the
      // remaining splits are in the store. When the task terminates, the client
      // can report the end first; the splits are then not reported finished.
      splitFuture_ = ContinueFuture::makeEmpty();
      getSplits(&splitFuture_);
    }
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
  // The producer's count, not table.num_rows(): a packed table with no columns
  // reports zero rows however many it holds, which is what an exchange
  // fragment with an empty output layout sends.
  const auto numRows = data.numRows;
  const auto& tableView = data.packedTable->table;
  VELOX_CHECK(
      tableView.num_columns() == 0 || tableView.num_rows() == numRows,
      "Row count from the exchange disagrees with the received table: {} vs. {}",
      numRows,
      tableView.num_rows());
  auto gpuDataSize = data.gpuDataSize();
  auto stream = data.stream;

  if (numRows == 0) {
    // An empty page carries no rows to hand on, and Operator::getOutput() must
    // never return an empty vector. Drop it and let the next isBlocked() fetch
    // the following page or reach the end of the stream. Producers do not
    // enqueue empty pages, so this is a defence rather than a live path.
    releaseCurrentData();
    return nullptr;
  }

  cudf_velox::CudfVector::ReleaseCallback releaseCallback;
  if (exchangeClient_->tracksInFlightReceiveBytes()) {
    std::weak_ptr<UcxExchangeClient> client{exchangeClient_};
    releaseCallback = [client, gpuDataSize](std::exception_ptr error) {
      if (auto locked = client.lock()) {
        if (error) {
          locked->failReceive(
              "Failed to synchronize a released UCX receive buffer");
        } else {
          locked->releaseInFlightReceiveBytes(gpuDataSize);
        }
      }
    };
  }

  // Use the stream that was allocated in UcxExchangeSource::onMetadata
  // and the packed_table constructor of CudfVector to avoid copying data.
  RowVectorPtr result;
  try {
    result = std::make_shared<cudf_velox::CudfVector>(
        pool(),
        outputType_,
        numRows,
        std::move(data.packedTable),
        stream,
        std::move(releaseCallback));
  } catch (...) {
    // The vector callback is not guaranteed to run when its constructor
    // throws. Ensure stream-ordered cleanup is complete before returning the
    // dequeued table's credit.
    currentData_.reset();
    bool cleanupSynchronized = true;
    try {
      stream.sync();
    } catch (const std::exception& e) {
      cleanupSynchronized = false;
      LOG(ERROR) << "Failed to synchronize UCX receive cleanup: " << e.what();
      exchangeClient_->failReceive("Failed to synchronize UCX receive cleanup");
    } catch (...) {
      cleanupSynchronized = false;
      LOG(ERROR) << "Failed to synchronize UCX receive cleanup";
      exchangeClient_->failReceive("Failed to synchronize UCX receive cleanup");
    }
    if (cleanupSynchronized && exchangeClient_->tracksInFlightReceiveBytes()) {
      exchangeClient_->releaseInFlightReceiveBytes(gpuDataSize);
    }
    throw;
  }

  recordInputStats(gpuDataSize, result);
  // free the memory owned by PackedTableWithStream and set it to nullptr;
  currentData_.reset();

  return result;
}

void UcxExchange::releaseCurrentData() {
  if (!currentData_) {
    return;
  }

  const auto bytes = currentData_->gpuDataSize();
  const auto stream = currentData_->stream;
  currentData_.reset();
  if (!exchangeClient_ || !exchangeClient_->tracksInFlightReceiveBytes()) {
    return;
  }

  try {
    stream.sync();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to synchronize dropped UCX receive: " << e.what();
    exchangeClient_->failReceive(
        "Failed to synchronize a dropped UCX receive buffer");
    return;
  } catch (...) {
    LOG(ERROR) << "Failed to synchronize dropped UCX receive";
    exchangeClient_->failReceive(
        "Failed to synchronize a dropped UCX receive buffer");
    return;
  }
  exchangeClient_->releaseInFlightReceiveBytes(bytes);
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
  releaseCurrentData();
  if (exchangeClient_) {
    recordExchangeClientStats();
  }
  // The client is shared by all consumers of this pipeline and owned by the
  // Task, which closes it on termination. Only drop this operator's reference.
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
