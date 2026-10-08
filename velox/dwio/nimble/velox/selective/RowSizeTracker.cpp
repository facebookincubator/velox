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

#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include <algorithm>

#include "velox/common/base/BitUtil.h"

namespace facebook::nimble {

/* explicit */ RowSizeTracker::RowSizeTracker(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& typeWithId)
    : typeWithId_{typeWithId} {
  const auto totalNodes = typeWithId->maxId() + 1;
  variableLengthNodes_.resize(totalNodes);
  projectedNodes_.resize(totalNodes, false);
  initFromSchema(*typeWithId->type(), 0, false);
  cellSizes_.resize(totalNodes, std::nullopt);
}

size_t RowSizeTracker::initFromSchema(
    const facebook::velox::Type& type,
    size_t nodeId,
    bool isParentVariableLength) {
  size_t nextId = nodeId + 1;
  if (type.isRow()) {
    variableLengthNodes_.setValid(nodeId, isParentVariableLength);
    for (const auto& child : type.asRow().children()) {
      nextId = initFromSchema(*child, nextId, isParentVariableLength);
    }
    return nextId;
  }

  if (type.isArray()) {
    auto& arrayType = type.asArray();
    variableLengthNodes_.setValid(nodeId, true);
    return initFromSchema(*arrayType.childAt(0), nextId, true);
  }

  if (type.isMap()) {
    auto& mapType = type.asMap();
    variableLengthNodes_.setValid(nodeId, true);
    nextId = initFromSchema(*mapType.childAt(0), nextId, true);
    return initFromSchema(*mapType.childAt(1), nextId, true);
  }

  if (type.isVarchar() || type.isVarbinary()) {
    variableLengthNodes_.setValid(nodeId, true);
    return nextId;
  }

  variableLengthNodes_.setValid(nodeId, isParentVariableLength);
  return nextId;
}

void RowSizeTracker::update(
    size_t nodeIdx,
    size_t memoryFootprint,
    velox::vector_size_t rowCount) {
  VELOX_CHECK(
      projectionFinalized_, "Must finalize projected columns before updates");
  if (rowCount <= 0) {
    return;
  }

  auto& currentColumnMax = cellSizes_[nodeIdx];
  auto updateValue = memoryFootprint / rowCount;
  if (!currentColumnMax.has_value()) {
    maxRowSize_ += updateValue;
    currentColumnMax = updateValue;
    return;
  }

  // The below logic handles each column's max row size independently.
  // This allows us to tolerate out of order updates to different rows
  // to some extent. The only remaining edge case is if the tracked row size
  // is polled too early (before any subcolumns can materialize).
  if (currentColumnMax < updateValue) {
    maxRowSize_ = maxRowSize_ - currentColumnMax.value() + updateValue;
    currentColumnMax = updateValue;
  }
}

void RowSizeTracker::applyProjection(size_t nodeIdx, bool projected) {
  projectedNodes_.setValid(nodeIdx, projected);
}

void RowSizeTracker::finalizeProjection() {
  if (projectionFinalized_) {
    return;
  }
  velox::bits::andBits(
      const_cast<uint64_t*>(variableLengthNodes_.allBits()),
      variableLengthNodes_.allBits(),
      projectedNodes_.allBits(),
      0,
      typeWithId_->maxId() + 1);
  projectionFinalized_ = true;
}

size_t RowSizeTracker::getCurrentMaxRowSize() const {
  VELOX_CHECK(
      projectionFinalized_, "Must finalize projected columns before access");
  bool allVariableLengthMaterialized = true;
  velox::bits::forEachBit(
      variableLengthNodes_.allBits(),
      0,
      variableLengthNodes_.end(),
      true,
      [&](velox::vector_size_t col) {
        allVariableLengthMaterialized = cellSizes_[col].has_value();
        VLOG(1) << fmt::format(
            "variable length col = {}, cell size: {}",
            col,
            cellSizes_[col].has_value() ? cellSizes_[col].value() : -1);
      });

  // TODO: handle the edge case where the tracked row size
  // is polled too early (before a lot of primitive subcolumns can
  // materialize). In this case we can either interpolate or supply some
  // heuristics based on primitive types. This logic would be similar to the
  // reader row size estimates logic to begin with. In a future iteration, we
  // can combine the 2 sources of estimates.

  // We return the same conservative estimate if we still have a lot of
  // unmaterialized variable length columns.
  return allVariableLengthMaterialized ? maxRowSize_
                                       : std::max(1UL << 20, maxRowSize_);
}

} // namespace facebook::nimble
