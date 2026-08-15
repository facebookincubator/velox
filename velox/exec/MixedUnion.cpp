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
#include "velox/exec/OperatorType.h"
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
          OperatorType::kMixedUnion),
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
  const auto reason = addMergeSources(future);
  if (reason != BlockingReason::kNotBlocked) {
    return reason;
  }

  if (sources_.empty()) {
    finished_ = true;
    return BlockingReason::kNotBlocked;
  }

  startSources();

  // After this operator has been drained in the current barrier cycle, don't
  // poll sources — they won't produce data until the next barrier.
  if (hasDrained()) {
    return BlockingReason::kNotBlocked;
  }

  // Pre-fetch data from each source into pendingData_. This is done in both
  // normal and drain modes so that getOutputMixed() can uniformly drain
  // pendingData_ without polling sources directly.
  std::vector<ContinueFuture> blockingFutures;
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sourcesFinished_[i] || sourcesDrained_[i] || pendingData_[i]) {
      continue;
    }

    ContinueFuture sourceFuture;
    RowVectorPtr data;
    bool drained{false};
    const auto blockingReason = sources_[i]->next(data, &sourceFuture, drained);

    if (blockingReason != BlockingReason::kNotBlocked) {
      blockingFutures.push_back(std::move(sourceFuture));
    } else if (data) {
      pendingData_[i] = std::move(data);
    } else if (drained) {
      sourcesDrained_[i] = true;
    } else {
      sourcesFinished_[i] = true;
    }
  }

  if (!blockingFutures.empty()) {
    // Wait for ALL pending sources so getOutput() can emit batches from
    // every active source in round-robin order. This ensures deterministic
    // interleaving. Finished and drained sources are skipped above and
    // never contribute a future, so collectAll will not hang.
    *future = folly::collectAll(std::move(blockingFutures)).unit();
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

bool MixedUnion::hasPendingDrainData() const {
  // Check if there's any pending data to drain.
  for (const auto& data : pendingData_) {
    if (data != nullptr) {
      return true;
    }
  }

  // Check if any source still has data to drain.
  // A source is considered drained if it's finished OR has signaled drained.
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (!sourcesFinished_[i] && !sourcesDrained_[i]) {
      return true;
    }
  }

  return false;
}

bool MixedUnion::startDrain() {
  VELOX_CHECK(isDraining());

  // Note: We don't call source->drain() here because the producer's
  // CallbackSink has already called it when it entered drain mode.
  // We just need to check if there's pending data to drain and drain any
  // remaining data from sources.

  if (hasPendingDrainData()) {
    return true;
  }

  // No data to drain. Reset state for next barrier cycle.
  std::fill(sourcesDrained_.begin(), sourcesDrained_.end(), false);

  return false;
}

void MixedUnion::maybeFinishDrain() {
  if (!isDraining()) {
    return;
  }

  if (hasPendingDrainData()) {
    return;
  }

  finishDrain();
}

void MixedUnion::finishDrain() {
  VELOX_CHECK(isDraining());

  // Reset drain state for next barrier.
  std::fill(sourcesDrained_.begin(), sourcesDrained_.end(), false);

  Operator::finishDrain();
}

RowVectorPtr MixedUnion::getOutputMixed() {
  if (hasDrained()) {
    return nullptr;
  }

  // Combine all pending data from all sources into a single output batch.
  // Sources are processed in index order (deterministic), with each source
  // contributing one batch per cycle. This matches the koski engine's
  // combineResults behavior.
  const auto numSources = pendingData_.size();

  // Collect all pending batches
  std::vector<RowVectorPtr> batches;
  vector_size_t totalRows = 0;
  for (size_t i = 0; i < numSources; ++i) {
    if (pendingData_[i]) {
      totalRows += pendingData_[i]->size();
      batches.push_back(std::move(pendingData_[i]));
    }
  }

  if (!batches.empty()) {
    RowVectorPtr result;
    if (batches.size() == 1) {
      result = std::move(batches[0]);
    } else {
      // Combine multiple batches into one by copying rows
      const auto outputType = unionNode_->outputType();
      auto pool = operatorCtx_->pool();
      std::vector<VectorPtr> children(outputType->size());
      for (auto i = 0; i < outputType->size(); ++i) {
        children[i] =
            BaseVector::create(outputType->childAt(i), totalRows, pool);
      }
      result = std::make_shared<RowVector>(
          pool, outputType, nullptr, totalRows, std::move(children));

      vector_size_t offset = 0;
      for (const auto& batch : batches) {
        for (auto i = 0; i < outputType->size(); ++i) {
          result->childAt(i)->copy(
              batch->childAt(i).get(), offset, 0, batch->size());
        }
        offset += batch->size();
      }
    }

    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(result->estimateFlatSize(), result->size());
    return result;
  }

  // No pending data. Check termination conditions.
  if (isDraining()) {
    maybeFinishDrain();
    return nullptr;
  }

  bool allFinished = true;
  bool allDrained = true;
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sourcesFinished_[i]) {
      continue;
    }
    allFinished = false;
    if (!sourcesDrained_[i]) {
      allDrained = false;
    }
  }

  if (allFinished) {
    finished_ = true;
    return nullptr;
  }

  if (allDrained && operatorCtx_->driver()->hasBarrier()) {
    operatorCtx_->driver()->drainOutput();
  }

  return nullptr;
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
