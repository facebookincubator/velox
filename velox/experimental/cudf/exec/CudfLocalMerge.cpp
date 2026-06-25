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

#include "velox/experimental/cudf/exec/CudfLocalMerge.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/Task.h"

#include <cudf/merge.hpp>

namespace facebook::velox::cudf_velox {

CudfLocalMerge::CudfLocalMerge(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::LocalMergeNode>& localMergeNode)
    : CudfSourceOperatorBase(
          operatorId,
          driverCtx,
          localMergeNode->outputType(),
          localMergeNode->id(),
          "CudfLocalMerge",
          nvtx3::rgb{0, 206, 209}) { // Dark Turquoise
  VELOX_CHECK_EQ(
      operatorCtx_->driverCtx()->driverId,
      0,
      "CudfLocalMerge needs to run single-threaded");

  const auto& sortingKeys = localMergeNode->sortingKeys();
  const auto& sortingOrders = localMergeNode->sortingOrders();
  const auto numSortingKeys = sortingKeys.size();
  sortKeys_.reserve(numSortingKeys);
  columnOrder_.reserve(numSortingKeys);
  nullOrder_.reserve(numSortingKeys);

  for (size_t i = 0; i < numSortingKeys; ++i) {
    const auto channel = exec::exprToChannel(sortingKeys[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "LocalMerge doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    const auto& order = sortingOrders[i];
    columnOrder_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfLocalMerge::addMergeSources() {
  if (!sourcesAdded_) {
    sources_ = operatorCtx_->task()->getLocalMergeSources(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    sourceData_.resize(sources_.size());
    sourceDone_.resize(sources_.size(), false);
    sourcesAdded_ = true;
  }
}

exec::BlockingReason CudfLocalMerge::isBlocked(ContinueFuture* future) {
  addMergeSources();

  if (sources_.empty()) {
    finished_ = true;
    return exec::BlockingReason::kNotBlocked;
  }

  if (!sourcesStarted_) {
    for (auto& source : sources_) {
      source->start();
    }
    sourcesStarted_ = true;
  }

  if (!blockingFutures_.empty()) {
    *future = std::move(blockingFutures_.back());
    blockingFutures_.pop_back();
    return exec::BlockingReason::kWaitForProducer;
  }

  return exec::BlockingReason::kNotBlocked;
}

bool CudfLocalMerge::isFinished() {
  return finished_;
}

RowVectorPtr CudfLocalMerge::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sourceDone_[i]) {
      continue;
    }

    while (true) {
      RowVectorPtr data;
      ContinueFuture future;
      bool drained = false;
      auto reason = sources_[i]->next(data, &future, drained);

      if (reason != exec::BlockingReason::kNotBlocked) {
        blockingFutures_.push_back(std::move(future));
        return nullptr;
      }

      if (!data || data->size() == 0) {
        sourceDone_[i] = true;
        break;
      }

      auto cudfData = std::dynamic_pointer_cast<CudfVector>(data);
      VELOX_CHECK_NOT_NULL(
          cudfData,
          "CudfLocalMerge expected CudfVector input from MergeSource");
      sourceData_[i].push_back(std::move(cudfData));
    }
  }

  bool allDone = std::all_of(
      sourceDone_.begin(), sourceDone_.end(), [](bool d) { return d; });
  if (!allDone) {
    return nullptr;
  }

  // All sources drained. Collect non-empty per-source tables and merge.
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();

  auto numNonEmptySources =
      std::count_if(sourceData_.begin(), sourceData_.end(), [](const auto& v) {
        return !v.empty();
      });
  if (numNonEmptySources == 0) {
    finished_ = true;
    return nullptr;
  } else if (numNonEmptySources == 1) {
    finished_ = true;
    auto it =
        std::find_if(sourceData_.begin(), sourceData_.end(), [](const auto& v) {
          return !v.empty();
        });
    VELOX_CHECK(it != sourceData_.end());
    if (it->size() == 1) {
      return std::move(it->front());
    }
    auto concatenated =
        getConcatenatedTable(std::move(*it), outputType_, stream, mr);
    auto numRows = concatenated->num_rows();
    return std::make_shared<CudfVector>(
        pool(), outputType_, numRows, std::move(concatenated), stream);
  }

  size_t numBatches = 0;
  for (const auto& batches : sourceData_) {
    numBatches += batches.size();
  }

  std::vector<cudf::table_view> tableViews;
  std::vector<rmm::cuda_stream_view> inputStreams;
  std::vector<CudfVectorPtr> inputs;
  tableViews.reserve(numBatches);
  inputStreams.reserve(numBatches);
  inputs.reserve(numBatches);
  for (auto& batches : sourceData_) {
    for (auto& tbl : batches) {
      tableViews.push_back(tbl->getTableView());
      inputStreams.push_back(tbl->stream());
      inputs.push_back(std::move(tbl));
    }
  }
  sourceData_.clear();

  cudf::detail::join_streams(inputStreams, stream);
  auto mergedTable =
      cudf::merge(tableViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);
  // Order the source deallocations after the merge by rebinding their buffers
  // to `stream` (with an event-wait fallback). This frees the inputs promptly
  // without forcing the pooled producer streams to wait on the merge.
  orderCudfVectorDeallocationsAfterStream(inputs, inputStreams, stream);

  auto numRows = mergedTable->num_rows();

  finished_ = true;
  return std::make_shared<CudfVector>(
      pool(), outputType_, numRows, std::move(mergedTable), stream);
}

void CudfLocalMerge::doClose() {
  for (auto& source : sources_) {
    source->close();
  }
  sourceData_.clear();
  CudfSourceOperatorBase::doClose();
}

} // namespace facebook::velox::cudf_velox
