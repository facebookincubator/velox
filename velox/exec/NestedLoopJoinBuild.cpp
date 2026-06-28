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
#include "velox/exec/NestedLoopJoinBuild.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"

#include <algorithm>

namespace facebook::velox::exec {
namespace {
const SpillPartitionId kNestedLoopJoinSpillPartitionId{0};
} // namespace

void NestedLoopJoinBridge::setData(NestedLoopJoinBuildData buildData) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_NULL(buildData_, "setData must be called only once");
    buildData_ =
        std::make_shared<const NestedLoopJoinBuildData>(std::move(buildData));
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::shared_ptr<const NestedLoopJoinBuildData>
NestedLoopJoinBridge::dataOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (buildData_ != nullptr) {
    return buildData_;
  }
  promises_.emplace_back("NestedLoopJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return nullptr;
}

NestedLoopJoinBuild::NestedLoopJoinBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::NestedLoopJoinNode> joinNode)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          OperatorType::kNestedLoopJoinBuild,
          joinNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(
                    operatorId,
                    OperatorType::kNestedLoopJoinBuild)
              : std::nullopt),
      buildType_(joinNode->sources()[1]->outputType()) {}

void NestedLoopJoinBuild::addInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  {
    Operator::ReclaimableSectionGuard guard(this);
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    if (canSpill() && spiller_ == nullptr) {
      // Charge retained input to the build pool so arbitration can reclaim it.
      auto copy =
          BaseVector::create<RowVector>(buildType_, input->size(), pool());
      copy->copy(input.get(), 0, 0, input->size());
      input = std::move(copy);
    }
  }
  // Copying input may have triggered reclamation and started spilling.
  if (spiller_ != nullptr) {
    spiller_->spill(kNestedLoopJoinSpillPartitionId, input);
    return;
  }
  dataVectors_.emplace_back(std::move(input));
}

BlockingReason NestedLoopJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return BlockingReason::kWaitForJoinBuild;
}

// Merge adjacent vectors to larger vectors as long as the result do not exceed
// the size limit.  This is important for performance because each small vector
// here would be duplicated by the number of rows on probe side, result in huge
// number of small vectors in the output.
std::vector<RowVectorPtr> NestedLoopJoinBuild::mergeDataVectors() const {
  const auto maxBatchRows =
      operatorCtx_->task()->queryCtx()->queryConfig().maxOutputBatchRows();
  std::vector<RowVectorPtr> merged;
  for (int i = 0; i < dataVectors_.size();) {
    // convert int32_t to int64_t to avoid sum overflow
    int64_t batchSize = dataVectors_[i]->size();
    auto j = i + 1;
    while (j < dataVectors_.size() &&
           batchSize + dataVectors_[j]->size() <= maxBatchRows) {
      batchSize += dataVectors_[j++]->size();
    }
    if (j == i + 1) {
      merged.push_back(dataVectors_[i++]);
    } else {
      auto batch = BaseVector::create<RowVector>(
          dataVectors_[i]->type(), batchSize, pool());
      batchSize = 0;
      while (i < j) {
        auto* source = dataVectors_[i++].get();
        batch->copy(source, batchSize, 0, source->size());
        batchSize += source->size();
      }
      merged.push_back(std::move(batch));
    }
  }
  return merged;
}

void NestedLoopJoinBuild::spillBufferedVectors() {
  if (dataVectors_.empty()) {
    return;
  }
  if (spiller_ == nullptr) {
    spiller_ = std::make_unique<NoRowContainerSpiller>(
        buildType_,
        std::nullopt,
        HashBitRange{},
        spillConfig(),
        spillStats_.get());
  }
  for (auto& vector : dataVectors_) {
    spiller_->spill(kNestedLoopJoinSpillPartitionId, vector);
    vector.reset();
  }
  dataVectors_.clear();
  pool()->release();
}

void NestedLoopJoinBuild::reclaim(
    uint64_t /*targetBytes*/,
    memory::MemoryReclaimer::Stats& /*stats*/) {
  VELOX_CHECK(canSpill());
  VELOX_CHECK(!nonReclaimableSection_);
  spillBufferedVectors();
}

void NestedLoopJoinBuild::noMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Operator>> peerOperators;
  // The last Driver to hit NestedLoopJoinBuild::finish gathers the data from
  // all build Drivers and hands it over to the probe side. At this
  // point all build Drivers are continued and will free their
  // state. allPeersFinished is true only for the last Driver of the
  // build pipeline.
  if (!operatorCtx_->allPeersFinished(&future_, promises, peerOperators)) {
    return;
  }

  {
    SCOPE_EXIT {
      // Realize the promises so that the other Drivers (which were not
      // the last to finish) can continue from the barrier and finish.
      peerOperators.clear();
      for (auto& promise : promises) {
        promise.setValue();
      }
    };

    std::vector<NestedLoopJoinBuild*> builds{this};
    for (const auto& peer : peerOperators) {
      auto* build = peer->as<NestedLoopJoinBuild>();
      VELOX_CHECK_NOT_NULL(build);
      builds.push_back(build);
    }

    const bool anySpilled =
        std::any_of(builds.begin(), builds.end(), [](auto* build) {
          return build->spiller_ != nullptr;
        });

    if (anySpilled) {
      SpillPartitionSet spillPartitions;
      for (auto* build : builds) {
        build->spillBufferedVectors();
        if (build->spiller_ != nullptr) {
          build->spiller_->finishSpill(spillPartitions);
          build->spiller_.reset();
        }
      }

      operatorCtx_->task()
          ->getNestedLoopJoinBridge(
              operatorCtx_->driverCtx()->splitGroupId, planNodeId())
          ->setData(
              {{},
               spillPartitions.at(kNestedLoopJoinSpillPartitionId)->files()});
      return;
    }

    for (const auto& peer : peerOperators) {
      auto* build = peer->as<NestedLoopJoinBuild>();
      VELOX_CHECK_NOT_NULL(build);
      dataVectors_.insert(
          dataVectors_.begin(),
          build->dataVectors_.begin(),
          build->dataVectors_.end());
    }
  }

  dataVectors_ = mergeDataVectors();
  operatorCtx_->task()
      ->getNestedLoopJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId())
      ->setData({std::move(dataVectors_), {}});
}

void NestedLoopJoinBuild::close() {
  dataVectors_.clear();
  spiller_.reset();
  Operator::close();
}

bool NestedLoopJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}
} // namespace facebook::velox::exec
