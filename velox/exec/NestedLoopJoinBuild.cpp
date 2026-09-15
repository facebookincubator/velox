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

std::shared_ptr<const NestedLoopJoinBuildData>
NestedLoopJoinBuildData::createInMemory(std::vector<RowVectorPtr> vectors) {
  auto data = std::make_shared<NestedLoopJoinBuildData>();
  data->vectors = std::move(vectors);
  return data;
}

std::shared_ptr<const NestedLoopJoinBuildData>
NestedLoopJoinBuildData::createSpilled(RowTypePtr type, SpillFiles files) {
  auto data = std::make_shared<NestedLoopJoinBuildData>();
  data->spilled = true;
  data->type = std::move(type);
  data->spillFiles = std::move(files);
  return data;
}

void NestedLoopJoinBridge::setData(
    std::shared_ptr<const NestedLoopJoinBuildData> buildData) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_NULL(buildData_, "setData must be called only once");
    buildData_ = std::move(buildData);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::shared_ptr<const NestedLoopJoinBuildData>
NestedLoopJoinBridge::dataOrFuture(
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (buildData_ != nullptr) {
    return buildData_;
  }
  promises_.emplace_back("NestedLoopJoinBridge::tableOrFuture");
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
  if (input->size() > 0) {
    {
      Operator::ReclaimableSectionGuard guard(this);
      // Load lazy vectors before storing or spilling.
      for (auto& child : input->children()) {
        child->loadedVector();
      }
      if (canSpill()) {
        input = copyToOperatorPool(input);
      }
    }

    if (spilled_) {
      spillVector(std::move(input));
      pool()->release();
      return;
    }
    dataVectors_.emplace_back(std::move(input));
  }
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

void NestedLoopJoinBuild::ensureSpiller() {
  VELOX_CHECK(canSpill());
  if (spiller_ != nullptr) {
    return;
  }
  spiller_ = std::make_unique<NoRowContainerSpiller>(
      buildType_,
      std::nullopt,
      HashBitRange{},
      spillConfig(),
      spillStats_.get());
}

void NestedLoopJoinBuild::spillVector(RowVectorPtr vector) {
  if (vector == nullptr || vector->size() == 0) {
    return;
  }
  ensureSpiller();
  spiller_->spill(kNestedLoopJoinSpillPartitionId, vector);
  spilled_ = true;
}

void NestedLoopJoinBuild::spillBufferedVectors() {
  if (dataVectors_.empty()) {
    return;
  }
  for (auto& vector : dataVectors_) {
    spillVector(std::move(vector));
  }
  dataVectors_.clear();
  pool()->release();
}

void NestedLoopJoinBuild::finishSpill(SpillPartitionSet& spillPartitions) {
  if (spiller_ == nullptr) {
    return;
  }
  spiller_->finishSpill(spillPartitions);
  spiller_.reset();
}

RowVectorPtr NestedLoopJoinBuild::copyToOperatorPool(
    const RowVectorPtr& input) {
  auto copy =
      BaseVector::create<RowVector>(input->type(), input->size(), pool());
  copy->copy(input.get(), 0, 0, input->size());
  return copy;
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
  std::vector<std::shared_ptr<Driver>> peers;
  // The last Driver to hit NestedLoopJoinBuild::finish gathers the data from
  // all build Drivers and hands it over to the probe side. At this
  // point all build Drivers are continued and will free their
  // state. allPeersFinished is true only for the last Driver of the
  // build pipeline.
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  {
    auto promisesGuard = folly::makeGuard([&]() {
      // Realize the promises so that the other Drivers (which were not
      // the last to finish) can continue from the barrier and finish.
      peers.clear();
      for (auto& promise : promises) {
        promise.setValue();
      }
    });

    std::vector<NestedLoopJoinBuild*> builds{this};
    for (auto& peer : peers) {
      auto op = peer->findOperator(planNodeId());
      auto* build = dynamic_cast<NestedLoopJoinBuild*>(op);
      VELOX_CHECK_NOT_NULL(build);
      builds.push_back(build);
    }

    const bool anySpilled =
        std::any_of(builds.begin(), builds.end(), [](auto* build) {
          return build->spilled_;
        });

    if (anySpilled) {
      SpillPartitionSet spillPartitions;
      for (auto* build : builds) {
        build->spillBufferedVectors();
        build->finishSpill(spillPartitions);
      }

      SpillFiles spillFiles;
      auto it = spillPartitions.find(kNestedLoopJoinSpillPartitionId);
      if (it != spillPartitions.end()) {
        spillFiles = it->second->files();
      }

      operatorCtx_->task()
          ->getNestedLoopJoinBridge(
              operatorCtx_->driverCtx()->splitGroupId, planNodeId())
          ->setData(NestedLoopJoinBuildData::createSpilled(
              buildType_, std::move(spillFiles)));
      return;
    }

    for (auto& peer : peers) {
      auto op = peer->findOperator(planNodeId());
      auto* build = dynamic_cast<NestedLoopJoinBuild*>(op);
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
      ->setData(NestedLoopJoinBuildData::createInMemory(
          std::move(dataVectors_)));
}

bool NestedLoopJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}
} // namespace facebook::velox::exec
