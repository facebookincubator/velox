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

BlockingReason MixedUnion::addMergeSources(ContinueFuture* future) {
  if (sources_.empty()) {
    // Get merge sources from the task
    sources_ = operatorCtx_->task()->getLocalMergeSources(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());

    // Initialize tracking vectors
    const auto numSources = sources_.size();
    pendingData_.resize(numSources);
    sourcesFinished_.resize(numSources, false);
    sourcesDrained_.resize(numSources, false);
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

  // Return any pending blocking reason from getOutput()
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(blockingFuture_);
    auto savedReason = blockingReason_;
    blockingReason_ = BlockingReason::kNotBlocked;
    return savedReason;
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

  if (isDraining()) {
    return getOutputDraining();
  }

  return getOutputMixed();
}

bool MixedUnion::startDrain() {
  VELOX_CHECK(isDraining());

  // Note: We don't call source->drain() here because the producer's
  // CallbackSink has already called it when it entered drain mode.
  // We just need to check if there's pending data to drain and drain any
  // remaining data from sources.
  drainSignaledToSources_ = true;

  // Check if there's any pending data to drain.
  for (size_t i = 0; i < pendingData_.size(); ++i) {
    if (pendingData_[i]) {
      return true;
    }
  }

  // Check if any source still has data to drain (hasn't signaled drained yet).
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (!sourcesDrained_[i] && !sourcesFinished_[i]) {
      return true;
    }
  }

  // No data to drain. Reset state for next barrier cycle.
  drainSignaledToSources_ = false;
  std::fill(sourcesDrained_.begin(), sourcesDrained_.end(), false);

  return false;
}

void MixedUnion::maybeFinishDrain() {
  if (!isDraining()) {
    return;
  }

  // Check if we have drained all pending data.
  for (const auto& data : pendingData_) {
    if (data) {
      return;
    }
  }

  // Check if all sources have been drained.
  for (bool drained : sourcesDrained_) {
    if (!drained) {
      return;
    }
  }

  finishDrain();
}

void MixedUnion::finishDrain() {
  VELOX_CHECK(drainSignaledToSources_);

  // Reset drain state for next barrier.
  drainSignaledToSources_ = false;
  std::fill(sourcesDrained_.begin(), sourcesDrained_.end(), false);

  Operator::finishDrain();
}

RowVectorPtr MixedUnion::getOutputDraining() {
  // First output any pending data.
  std::vector<RowVectorPtr> validInputs;
  for (size_t i = 0; i < pendingData_.size(); ++i) {
    if (pendingData_[i]) {
      validInputs.push_back(std::move(pendingData_[i]));
      pendingData_[i] = nullptr;
    }
  }

  if (!validInputs.empty()) {
    auto result = combineResults(validInputs);
    maybeFinishDrain();
    return result;
  }

  // Try to get remaining data from sources.
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sourcesDrained_[i]) {
      continue;
    }

    // Try to get data from the source.
    ContinueFuture unusedFuture;
    RowVectorPtr data;
    bool sourceDrained{false};
    const auto blockingReason =
        sources_[i]->next(data, &unusedFuture, sourceDrained);

    if (sourceDrained) {
      sourcesDrained_[i] = true;
      continue;
    }

    if (blockingReason != BlockingReason::kNotBlocked) {
      // Source is blocked - in drain mode we don't wait, just check next.
      continue;
    }

    if (data) {
      validInputs.push_back(std::move(data));
    }
  }

  auto result = combineResults(validInputs);
  maybeFinishDrain();
  return result;
}

RowVectorPtr MixedUnion::getOutputMixed() {
  // Collect any available data from sources.
  std::vector<RowVectorPtr> validInputs;
  std::vector<ContinueFuture> blockingFutures;

  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sourcesFinished_[i] || sourcesDrained_[i]) {
      continue;
    }

    ContinueFuture sourceFuture;
    RowVectorPtr data;
    bool drained{false};
    const auto blockingReason = sources_[i]->next(data, &sourceFuture, drained);

    if (blockingReason != BlockingReason::kNotBlocked) {
      blockingFutures.push_back(std::move(sourceFuture));
    } else if (data) {
      validInputs.push_back(std::move(data));
    } else if (drained) {
      sourcesDrained_[i] = true;
    } else {
      sourcesFinished_[i] = true;
    }
  }

  // If we got some data, return it.
  if (!validInputs.empty()) {
    return combineResults(validInputs);
  }

  // Check if all sources are finished.
  bool allFinished = true;
  bool allDrained = true;
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (!sourcesFinished_[i]) {
      allFinished = false;
    }
    if (!sourcesFinished_[i] && !sourcesDrained_[i]) {
      allDrained = false;
    }
  }

  if (allFinished) {
    finished_ = true;
    return nullptr;
  }

  // If some sources are blocked, save blocking state for isBlocked().
  if (!blockingFutures.empty()) {
    blockingReason_ = BlockingReason::kWaitForProducer;
    blockingFuture_ = folly::collectAny(std::move(blockingFutures)).unit();
    return nullptr;
  }

  // If all sources have drained (but not finished), trigger barrier processing.
  // Only trigger if the driver is under barrier processing.
  if (allDrained && operatorCtx_->driver()->hasBarrier() && !isDraining()) {
    operatorCtx_->driver()->drainOutput();
  }

  return nullptr;
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
