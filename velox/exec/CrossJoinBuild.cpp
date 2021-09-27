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
#include "velox/exec/CrossJoinBuild.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

void CrossJoinBridge::setData(std::vector<VectorPtr> data) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!data_.has_value(), "setData may be called only once");
  data_ = std::move(data);
  notifyConsumersLocked();
}

std::optional<std::vector<VectorPtr>> CrossJoinBridge::dataOrFuture(
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (data_.has_value()) {
    return std::move(data_);
  }
  promises_.emplace_back("CrossJoinBridge::tableOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

CrossJoinBuild::CrossJoinBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::CrossJoinNode> joinNode)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          "CrossJoinBuild") {}

void CrossJoinBuild::addInput(RowVectorPtr input) {
  if (input->size() > 0) {
    // Load lazy vectors before storing.
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    data_.emplace_back(std::move(input));
  }
}

BlockingReason CrossJoinBuild::isBlocked(ContinueFuture* future) {
  if (!hasFuture_) {
    return BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  hasFuture_ = false;
  return BlockingReason::kWaitForJoinBuild;
}

void CrossJoinBuild::finish() {
  Operator::finish();
  std::vector<VeloxPromise<bool>> promises;
  std::vector<std::shared_ptr<Driver>> peers;
  // The last Driver to hit CrossJoinBuild::finish gathers the data from
  // all build Drivers and hands it over to the probe side. At this
  // point all build Drivers are continued and will free their
  // state. allPeersFinished is true only for the last Driver of the
  // build pipeline.
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    hasFuture_ = true;
    return;
  }

  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CrossJoinBuild*>(op);
    VELOX_CHECK(build);
    data_.insert(data_.begin(), build->data_.begin(), build->data_.end());
  }

  // Realize the promises so that the other Drivers (which were not
  // the last to finish) can continue from the barrier and finish.
  peers.clear();
  for (auto& promise : promises) {
    promise.setValue(true);
  }

  operatorCtx_->task()
      ->getCrossJoinBridge(planNodeId())
      ->setData(std::move(data_));
}
} // namespace facebook::velox::exec
