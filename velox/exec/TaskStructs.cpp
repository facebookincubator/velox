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

#include "velox/exec/TaskStructs.h"

namespace facebook::velox::exec {

void SplitsStore::addSplit(
    Split split,
    std::vector<ContinuePromise>& promises) {
  VELOX_CHECK(!noMoreSplits_);
  VELOX_CHECK(!(remoteSplit_ && split.isBarrier()));
  VELOX_CHECK(barrierSplits_.empty());
  if (split.isBarrier()) {
    for (auto i = 0; i < split.barrier->numDrivers; ++i) {
      barrierSplits_[i] = Split::createBarrier();
    }
    VELOX_CHECK_LE(promises_.size(), split.barrier->numDrivers);
    // A barrier is assigned to every driver; wake up all currently blocked
    // drivers to process it.
    std::move(promises_.begin(), promises_.end(), std::back_inserter(promises));
    promises_.clear();
  } else {
    splits_.push_back(std::move(split));
    if (!promises_.empty()) {
      promises.push_back(std::move(promises_.back()));
      promises_.pop_back();
    }
  }
}

ContinueFuture SplitsStore::makeFuture() {
  auto [promise, future] =
      makeVeloxContinuePromiseContract("SplitsStore::makeFuture");
  promises_.push_back(std::move(promise));
  return std::move(future);
}

std::deque<Split> SplitsStore::finishConsumer(
    uint32_t driverId,
    uint32_t numConsumers,
    std::vector<ContinuePromise>& promises) {
  VELOX_CHECK(finishedConsumers_.insert(driverId).second);
  VELOX_CHECK_LE(finishedConsumers_.size(), numConsumers);
  taskStats_->numAbandonedSplits += releaseRunningSplits(driverId);
  if (finishedConsumers_.size() != numConsumers) {
    return {};
  }
  allConsumersFinished_ = true;
  return discardQueuedSplits(promises);
}

void SplitsStore::finishSplits(uint32_t driverId) {
  taskStats_->numFinishedSplits += releaseRunningSplits(driverId);
}

int32_t SplitsStore::releaseRunningSplits(uint32_t driverId) {
  const auto it = runningSplits_.find(driverId);
  if (it == runningSplits_.end()) {
    return 0;
  }
  const auto& running = it->second;
  const auto numSplits = running.numSplits;
  taskStats_->numRunningSplits -= running.numSplits;
  if (!remoteSplit_) {
    taskStats_->numRunningTableScanSplits -= running.numSplits;
    taskStats_->runningTableScanSplitWeights -= running.weight;
  }
  runningSplits_.erase(it);
  return numSplits;
}

std::deque<Split> SplitsStore::discardQueuedSplits(
    std::vector<ContinuePromise>& promises) {
  auto waiters = noMoreSplits();
  std::move(waiters.begin(), waiters.end(), std::back_inserter(promises));
  barrierSplits_.clear();

  std::deque<Split> abandoned;
  abandoned.swap(splits_);
  for (const auto& split : abandoned) {
    ++taskStats_->numAbandonedSplits;
    removeQueuedSplitStats(split);
  }
  return abandoned;
}

void SplitsStore::removeQueuedSplitStats(const Split& split) {
  --taskStats_->numQueuedSplits;
  if (!remoteSplit_ && split.hasConnectorSplit()) {
    --taskStats_->numQueuedTableScanSplits;
    taskStats_->queuedTableScanSplitWeights -=
        split.connectorSplit->splitWeight;
  }
}

Split SplitsStore::getSplit(
    uint32_t driverId,
    int maxPreloadSplits,
    const ConnectorSplitPreloadFunc& preload) {
  VELOX_CHECK_EQ(finishedConsumers_.count(driverId), 0);
  int readySplitIndex = -1;
  if (maxPreloadSplits > 0) {
    for (int i = 0, end = std::min<size_t>(maxPreloadSplits, splits_.size());
         i < end;
         ++i) {
      if (splits_[i].isBarrier()) {
        VELOX_CHECK(!remoteSplit_);
        continue;
      }
      auto& connectorSplit = splits_[i].connectorSplit;
      if (!connectorSplit->dataSource) {
        // Initializes split->dataSource.
        preload(connectorSplit);
        preloadingSplits_->insert(connectorSplit);
      } else if (
          readySplitIndex == -1 && connectorSplit->dataSource->hasValue()) {
        readySplitIndex = i;
        preloadingSplits_->erase(connectorSplit);
      }
    }
  }
  if (readySplitIndex == -1) {
    readySplitIndex = 0;
  }
  VELOX_CHECK(!splits_.empty());
  auto split = std::move(splits_[readySplitIndex]);
  splits_.erase(splits_.begin() + readySplitIndex);
  removeQueuedSplitStats(split);
  ++taskStats_->numRunningSplits;
  auto& running = runningSplits_[driverId];
  ++running.numSplits;
  if (!remoteSplit_ && split.connectorSplit) {
    running.weight += split.connectorSplit->splitWeight;
    ++taskStats_->numRunningTableScanSplits;
    taskStats_->runningTableScanSplitWeights +=
        split.connectorSplit->splitWeight;
  }
  taskStats_->lastSplitStartTimeMs =
      EpochTime::now<std::chrono::milliseconds>();
  if (taskStats_->firstSplitStartTimeMs == 0) {
    taskStats_->firstSplitStartTimeMs = taskStats_->lastSplitStartTimeMs;
  }
  return split;
}

bool SplitsStore::tryGetBarrier(uint32_t driverId, Split& split) {
  // Delivers a barrier exactly once for each driver from the same plan node.
  auto it = barrierSplits_.find(driverId);
  if (it == barrierSplits_.end()) {
    return false;
  }
  split = it->second;
  barrierSplits_.erase(it);
  return true;
}

} // namespace facebook::velox::exec
