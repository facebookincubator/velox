/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/MixedUnion.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

MixedUnion::MixedUnion(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MixedUnionNode>& unionNode)
    : SourceOperator(
          driverCtx,
          unionNode->outputType(),
          operatorId,
          unionNode->id(),
          "MixedUnion"),
      unionNode_(unionNode),
      maxOutputBatchRows_(outputBatchRows()),
      maxOutputBatchBytes_(
          driverCtx->queryConfig().preferredOutputBatchBytes()) {}

BlockingReason MixedUnion::addMergeSources(ContinueFuture* /* future */) {
  if (sources_.empty()) {
    // Get merge sources from the task
    sources_ = operatorCtx_->task()->getLocalMergeSources(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());

    // Initialize tracking vectors
    const auto numSources = sources_.size();
    pendingData_.resize(numSources);
    sourcesFinished_.resize(numSources, false);
  }
  return BlockingReason::kNotBlocked;
}

void MixedUnion::startSources() {
  if (sourcesStarted_) {
    return;
  }

  // Start all sources
  for (auto& source : sources_) {
    source->start();
  }
  sourcesStarted_ = true;
}

BlockingReason MixedUnion::isBlocked(ContinueFuture* future) {
  // Get sources from the task if not already acquired
  const auto reason = addMergeSources(future);
  if (reason != BlockingReason::kNotBlocked) {
    return reason;
  }

  // If task terminated early with no sources, mark as finished
  if (sources_.empty()) {
    finished_ = true;
    return BlockingReason::kNotBlocked;
  }

  // Start sources if not already started
  startSources();

  // Try to fetch data from each source that doesn't have pending data
  sourceBlockingFutures_.clear();
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sourcesFinished_[i] || pendingData_[i]) {
      // Source is finished or already has data
      continue;
    }

    ContinueFuture sourceFuture;
    RowVectorPtr data;
    const auto blockingReason = sources_[i]->next(data, &sourceFuture);

    if (blockingReason != BlockingReason::kNotBlocked) {
      // Source is blocked, add future to the list
      sourceBlockingFutures_.push_back(std::move(sourceFuture));
    } else if (data) {
      // Got data from this source
      pendingData_[i] = std::move(data);
    } else {
      // Source is finished
      sourcesFinished_[i] = true;
    }
  }

  // If any source is blocked, return a blocking future
  if (!sourceBlockingFutures_.empty()) {
    *future = std::move(sourceBlockingFutures_.back());
    sourceBlockingFutures_.pop_back();
    return BlockingReason::kWaitForProducer;
  }

  return BlockingReason::kNotBlocked;
}

bool MixedUnion::isFinished() {
  return finished_;
}

RowVectorPtr MixedUnion::getOutput() {
  if (finished_) {
    return nullptr;
  }

  return getOutputMixed();
}

RowVectorPtr MixedUnion::getOutputMixed() {
  // Collect all available data from sources
  std::vector<RowVectorPtr> validInputs;
  for (size_t i = 0; i < pendingData_.size(); ++i) {
    if (pendingData_[i]) {
      validInputs.push_back(std::move(pendingData_[i]));
      pendingData_[i] = nullptr;
    }
  }

  // If we have no data, check if all sources are finished
  if (validInputs.empty()) {
    bool allFinished = true;
    for (bool isFinished : sourcesFinished_) {
      if (!isFinished) {
        allFinished = false;
        break;
      }
    }
    if (allFinished) {
      finished_ = true;
    }
    return nullptr;
  }

  // Combine results from all sources
  return combineResults(validInputs);
}

RowVectorPtr MixedUnion::combineResults(std::vector<RowVectorPtr>& results) {
  if (results.empty()) {
    return nullptr;
  }

  if (results.size() == 1) {
    auto result = std::move(results[0]);
    // Record output statistics
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addOutputVector(result->estimateFlatSize(), result->size());
    }
    return result;
  }

  // Calculate total number of rows
  vector_size_t totalRows = 0;
  for (const auto& result : results) {
    totalRows += result->size();
  }

  if (totalRows == 0) {
    return nullptr;
  }

  // Create combined output vector
  auto combinedResult =
      BaseVector::create<RowVector>(outputType_, totalRows, pool());

  // Copy data from all input vectors
  vector_size_t currentOffset = 0;
  for (const auto& result : results) {
    if (result->size() > 0) {
      for (auto i = 0; i < outputType_->size(); ++i) {
        // Copy column data
        std::vector<BaseVector::CopyRange> ranges;
        ranges.push_back({0, currentOffset, result->size()});

        combinedResult->childAt(i)->copyRanges(
            result->childAt(i).get(), ranges);
      }
      currentOffset += result->size();
    }
  }

  // Record output statistics
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(
        combinedResult->estimateFlatSize(), combinedResult->size());
  }

  return combinedResult;
}

bool MixedUnion::hasDataFromAllSources() const {
  for (size_t i = 0; i < pendingData_.size(); ++i) {
    // If source is not finished and has no data, we don't have all sources
    if (!sourcesFinished_[i] && !pendingData_[i]) {
      return false;
    }
  }
  return true;
}

void MixedUnion::close() {
  // Close all sources
  for (auto& source : sources_) {
    source->close();
  }
  pendingData_.clear();
  Operator::close();
}

} // namespace facebook::velox::exec
