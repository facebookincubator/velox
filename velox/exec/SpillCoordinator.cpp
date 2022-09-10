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

#include "velox/exec/SpillCoordinator.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

void SpillCoordinator::join(Operator* op, SpillCoordinator::SpillRunner runner) {
  VELOX_CHECK_NOT_NULL(runner);
  std::lock_guard<std::mutex> l(mutex_);
  operators_.push_back(op);
  if (FOLLY_UNLIKELY(spillRunner_ == nullptr)) {
    spillRunner_ = std::move(runner);
  }
}

bool SpillCoordinator::requestSpill(Operator* op, ContinueFuture* future) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(numWaitingOperators_, numSpillingOperatorsLocked());
    VELOX_CHECK_LT(noMoreSpillOperators_.size(), operators_.size());
    hasPendingSpill_ = true;
    if (waitSpillLocked(op, promises, future)) {
      VELOX_CHECK(future->valid());
      return true;
    }
  }
  runSpill(promises);
  return false;
}

bool SpillCoordinator::waitSpill(Operator* op, ContinueFuture* future) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_LT(noMoreSpillOperators_.size(), operators_.size());
    if (!hasPendingSpill_) {
      VELOX_CHECK_EQ(numWaitingOperators_, 0);
      return false;
    }
    if (waitSpillLocked(op, promises, future)) {
      return true;
    }
  }
  runSpill(promises);
  return false;
}

bool SpillCoordinator::waitSpillLocked(
    Operator* op,
    std::vector<ContinuePromise>& promises,
    ContinueFuture* future) {
  VELOX_CHECK_LT(numWaitingOperators_, numSpillingOperatorsLocked());
  if (++numWaitingOperators_ == numSpillingOperatorsLocked()) {
    promises = std::move(promises_);
    return false;
  }
  promises_.emplace_back(ContinuePromise(fmt::format(
      "SpillCoordinator::waitSpillLocked {}/{}/{}/{}",
      taskId_,
      planNodeId_,
      splitGroupId_,
      op->stats().operatorId)));
  *future = promises_.back().getSemiFuture();
  return true;
}

void SpillCoordinator::noMoreSpill(Operator* op) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_DCHECK(!noMoreSpillOperators_.contains(op));
    noMoreSpillOperators_.insert(op);
    VELOX_DCHECK_LE(noMoreSpillOperators_.size(), operators_.size());
    if (FOLLY_LIKELY(!hasPendingSpill_)) {
      return;
    }
    if (numWaitingOperators_ < numSpillingOperatorsLocked()) {
      return;
    }
    promises = std::move(promises_);
  }
  runSpill(promises);
}

void SpillCoordinator::reset() {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_EQ(noMoreSpillOperators_.size(), operators_.size());
  // Reset this coordinator state if all the participated operators have
  // finished memory intensive operations and won't trigger any more spill.
  VELOX_CHECK(!hasPendingSpill_);
  VELOX_CHECK_EQ(numWaitingOperators_, 0);
  VELOX_CHECK(promises_.empty());
  noMoreSpillOperators_.clear();
}

void SpillCoordinator::runSpill(std::vector<ContinuePromise>& promises) {
  VELOX_CHECK(hasPendingSpill_);
  spillRunner_(operators_);
  hasPendingSpill_ = false;
  numWaitingOperators_ = 0;
  for (auto& promise : promises) {
    promise.setValue();
  }
}
} // namespace facebook::velox::exec
