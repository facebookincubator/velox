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

#include <boost/circular_buffer.hpp>

#include "velox/exec/Merge.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

Merge::Merge(
    int32_t operatorId,
    DriverCtx* ctx,
    RowTypePtr outputType,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    const std::string& planNodeId,
    const std::string& operatorType)
    : SourceOperator(
          ctx,
          std::move(outputType),
          operatorId,
          planNodeId,
          operatorType),
      comparator_(Comparator(outputType_, sortingKeys, sortingOrders)) {}

BlockingReason Merge::isBlocked(ContinueFuture* future) {
  auto reason = addMergeSources(future);
  if (reason != BlockingReason::kNotBlocked) {
    return reason;
  }

  if (sourceCursors_.empty()) {
    std::vector<std::unique_ptr<SourceCursor>> sourceCursors;
    sourceCursors.reserve(sources_.size());
    sourceCursors_.reserve(sources_.size());
    for (auto& source : sources_) {
      sourceCursors.push_back(
          std::make_unique<SourceCursor>(source.get(), sourceBlockingFutures_));
      sourceCursors_.push_back(sourceCursors.back().get());
    }

    treeOfLoosers_ = std::make_unique<TreeOfLosers<SourceRow, SourceCursor>>(
        std::move(sourceCursors));
  } else if (sourceBlockingFutures_.empty()) {
    for (auto& cursor : sourceCursors_) {
      cursor->isReady();
    }
  }

  if (!sourceBlockingFutures_.empty()) {
    *future = std::move(sourceBlockingFutures_.back());
    sourceBlockingFutures_.pop_back();
    //    *future = folly::collectAny(std::move(sourceBlockingFutures_))
    //                  .deferValue([](auto /*unused*/) { return true; });
    return BlockingReason::kWaitForExchange;
  }

  return BlockingReason::kNotBlocked;
}

bool Merge::isFinished() {
  return finished_;
}

RowVectorPtr Merge::getOutput() {
  if (finished_) {
    return nullptr;
  }

  // TODO Fetch from config.
  const vector_size_t kOutputBatchSize = 1024;

  if (!output_) {
    output_ = std::dynamic_pointer_cast<RowVector>(BaseVector::create(
        outputType_, kOutputBatchSize, operatorCtx_->pool()));
    for (auto& child : output_->children()) {
      child->resize(kOutputBatchSize);
    }
  }

  for (;;) {
    auto outputRow = treeOfLoosers_->next(comparator_);

    if (!outputRow.has_value()) {
      output_->resize(outputSize_);
      finished_ = true;
      return std::move(output_);
    }

    for (auto i = 0; i < outputType_->size(); ++i) {
      output_->childAt(i)->copy(
          outputRow->vector->childAt(i).get(),
          outputSize_,
          outputRow->index,
          1);
    }
    ++outputSize_;

    if (outputSize_ == kOutputBatchSize) {
      outputSize_ = 0;
      return std::move(output_);
    }

    if (!sourceBlockingFutures_.empty()) {
      return nullptr;
    }
  }
}

Merge::Comparator::Comparator(
    const RowTypePtr& type,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders) {
  auto numKeys = sortingKeys.size();
  for (int i = 0; i < numKeys; ++i) {
    auto channel = exprToChannel(sortingKeys[i].get(), type);
    VELOX_CHECK_NE(
        channel,
        kConstantChannel,
        "Merge doesn't allow constant grouping keys");
    keyInfo_.emplace_back(
        channel,
        CompareFlags{
            sortingOrders[i].isNullsFirst(),
            sortingOrders[i].isAscending(),
            false});
  }
}

LocalMerge::LocalMerge(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::LocalMergeNode>& localMergeNode)
    : Merge(
          operatorId,
          driverCtx,
          localMergeNode->outputType(),
          localMergeNode->sortingKeys(),
          localMergeNode->sortingOrders(),
          localMergeNode->id(),
          "LocalMerge") {
  VELOX_CHECK_EQ(
      operatorCtx_->driverCtx()->driverId,
      0,
      "LocalMerge needs to run single-threaded");
}

BlockingReason LocalMerge::addMergeSources(ContinueFuture* /* future */) {
  if (sources_.empty()) {
    sources_ = operatorCtx_->task()->getLocalMergeSources(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  }
  return BlockingReason::kNotBlocked;
}

MergeExchange::MergeExchange(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MergeExchangeNode>& mergeExchangeNode)
    : Merge(
          operatorId,
          driverCtx,
          mergeExchangeNode->outputType(),
          mergeExchangeNode->sortingKeys(),
          mergeExchangeNode->sortingOrders(),
          mergeExchangeNode->id(),
          "MergeExchange") {}

BlockingReason MergeExchange::addMergeSources(ContinueFuture* future) {
  if (operatorCtx_->driverCtx()->driverId != 0) {
    // When there are multiple pipelines, a single operator, the one from
    // pipeline 0, is responsible for merging pages.
    return BlockingReason::kNotBlocked;
  }
  if (noMoreSplits_) {
    return BlockingReason::kNotBlocked;
  }
  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId(), split, *future);
    if (reason == BlockingReason::kNotBlocked) {
      if (split.hasConnectorSplit()) {
        auto remoteSplit = std::dynamic_pointer_cast<RemoteConnectorSplit>(
            split.connectorSplit);
        VELOX_CHECK(remoteSplit, "Wrong type of split");

        sources_.emplace_back(
            MergeSource::createMergeExchangeSource(this, remoteSplit->taskId));
        ++numSplits_;
      } else {
        noMoreSplits_ = true;
        // TODO Delay this call until all input data has been processed.
        operatorCtx_->task()->multipleSplitsFinished(numSplits_);
        return BlockingReason::kNotBlocked;
      }
    } else {
      return reason;
    }
  }
}

} // namespace facebook::velox::exec
