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

#include "velox/exec/SpatialJoinBuild.h"
#include <limits>
#include "velox/common/geospatial/GeometryConstants.h"
#ifdef VELOX_ENABLE_GEO
#include "velox/common/geospatial/GeometrySerde.h"
#endif
#include "velox/exec/OperatorType.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

using velox::common::geospatial::GeometrySerializationType;

void SpatialJoinBridge::setData(SpatialBuildResult buildResult) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!buildResult_.has_value(), "setData must be called only once");
    buildResult_ = std::move(buildResult);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<SpatialBuildResult> SpatialJoinBridge::dataOrFuture(
    ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(!cancelled_, "Getting data after the build side is aborted");
  if (buildResult_.has_value()) {
    return buildResult_.value();
  }
  promises_.emplace_back("SpatialJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

SpatialJoinBuild::SpatialJoinBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::SpatialJoinNode> joinNode)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          OperatorType::kSpatialJoinBuild) {
  const auto& buildType = joinNode->rightNode()->outputType();
  buildGeometryChannel_ =
      buildType->getChildIdx(joinNode->buildGeometry()->name());
  VELOX_CHECK_EQ(
      buildType->childAt(buildGeometryChannel_),
      joinNode->buildGeometry()->type());
  if (joinNode->radius().has_value()) {
    auto radiusVar = joinNode->radius().value();
    uint32_t radiusChannel = buildType->getChildIdx(radiusVar->name());
    VELOX_CHECK_EQ(buildType->childAt(radiusChannel), radiusVar->type());
    radiusChannel_ = radiusChannel;
  }
}

void SpatialJoinBuild::addInput(RowVectorPtr input) {
  if (input->size() > 0) {
    // Load lazy vectors before storing.
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    dataVectors_.emplace_back(std::move(input));
  }
}

BlockingReason SpatialJoinBuild::isBlocked(ContinueFuture* future) {
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
std::vector<RowVectorPtr> SpatialJoinBuild::mergeDataVectors() const {
  const auto maxBatchRows =
      operatorCtx_->task()->queryCtx()->queryConfig().maxOutputBatchRows();
  std::vector<RowVectorPtr> merged;
  for (size_t i = 0; i < dataVectors_.size();) {
    // convert int32_t to int64_t to avoid sum overflow
    int64_t batchSize = static_cast<int64_t>(dataVectors_[i]->size());
    auto j = i + 1;
    while (j < dataVectors_.size() &&
           batchSize + dataVectors_[j]->size() <= maxBatchRows) {
      batchSize += dataVectors_[j++]->size();
    }
    if (j == i + 1) {
      merged.push_back(dataVectors_[i++]);
    } else {
      auto batch = BaseVector::create<RowVector>(
          dataVectors_[i]->type(),
          static_cast<vector_size_t>(batchSize),
          pool());
      batchSize = 0;
      while (i < j) {
        auto* source = dataVectors_[i++].get();
        batch->copy(
            source, static_cast<vector_size_t>(batchSize), 0, source->size());
        batchSize += source->size();
      }
      merged.push_back(std::move(batch));
    }
  }
  return merged;
}

Envelope SpatialJoinBuild::readEnvelope(
    const StringView& geometryBytes,
    double radius) {
#ifdef VELOX_ENABLE_GEO
  radius = std::max(radius, 0.0);
  auto geosEnvelope =
      common::geospatial::GeometryDeserializer::deserializeEnvelope(
          geometryBytes);
  if (geosEnvelope->isNull()) {
    return Envelope::empty();
  } else {
    return Envelope::from(
        geosEnvelope->getMinX() - radius,
        geosEnvelope->getMinY() - radius,
        geosEnvelope->getMaxX() + radius,
        geosEnvelope->getMaxY() + radius);
  }
#else
  // When VELOX_ENABLE_GEO is not set, return an envelope of infinite area
  // to ensure all geometries are considered for spatial join
  return Envelope::from(
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity());
#endif
}

SpatialIndex SpatialJoinBuild::buildSpatialIndex(
    const std::vector<RowVectorPtr>& data,
    column_index_t geometryIdx,
    std::optional<column_index_t> radiusIdx) {
  size_t numRows = 0;
  for (auto& vector : data) {
    numRows += vector->size();
  }
  std::vector<Envelope> envelopes;
  // TODO: Chunk the data to avoid allocating a large vector.
  envelopes.reserve(numRows);

  DecodedVector radiusCol;
  DecodedVector geometryCol;
  vector_size_t offset = 0;
  Envelope bounds = Envelope::empty();

  for (auto& vector : data) {
    const auto& rawGeometryCol =
        vector->childAt(geometryIdx)->asChecked<SimpleVector<StringView>>();
    geometryCol.decode(*rawGeometryCol);

    auto constantZero = velox::BaseVector::createConstant(
        velox::DOUBLE(), 0.0, vector->size(), pool());
    if (radiusIdx.has_value()) {
      const auto& rawRadiusCol =
          vector->childAt(radiusIdx.value())->asChecked<SimpleVector<double>>();
      radiusCol.decode(*rawRadiusCol);
    } else {
      radiusCol.decode(*constantZero);
    }

    // TODO: Make a selectivity vector based on nulls and use for DecodedVector.
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      if (geometryCol.isNullAt(i) || radiusCol.isNullAt(i)) {
        // If geometry or radius is null, it will not match the predicate and so
        // we should skip the envelope.
        continue;
      }
      double radius = radiusCol.valueAt<double>(i);
      const StringView geometryBytes = geometryCol.valueAt<StringView>(i);
      Envelope envelope = SpatialJoinBuild::readEnvelope(geometryBytes, radius);
      if (FOLLY_UNLIKELY(envelope.isEmpty())) {
        continue;
      }
      envelope.rowIndex = offset + geometryCol.index(i);
      bounds.merge(envelope);
      envelopes.push_back(std::move(envelope));
    }
    offset += vector->size();
  }
  return SpatialIndex(std::move(bounds), std::move(envelopes));
}

void SpatialJoinBuild::noMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Driver>> peers;
  // The last Driver to hit SpatialJoinBuild::finish gathers the data from
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

    for (auto& peer : peers) {
      auto op = peer->findOperator(planNodeId());
      auto* build = dynamic_cast<SpatialJoinBuild*>(op);
      VELOX_CHECK_NOT_NULL(build);
      dataVectors_.insert(
          dataVectors_.end(),
          std::make_move_iterator(build->dataVectors_.begin()),
          std::make_move_iterator(build->dataVectors_.end()));
    }
  }

  dataVectors_ = mergeDataVectors();
  SpatialIndex spatialIndex =
      buildSpatialIndex(dataVectors_, buildGeometryChannel_, radiusChannel_);
  SpatialBuildResult buildResult;
  buildResult.spatialIndex =
      std::make_shared<SpatialIndex>(std::move(spatialIndex));
  buildResult.buildVectors = std::move(dataVectors_);

  operatorCtx_->task()
      ->getSpatialJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId())
      ->setData(std::move(buildResult));
}

bool SpatialJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}
} // namespace facebook::velox::exec
