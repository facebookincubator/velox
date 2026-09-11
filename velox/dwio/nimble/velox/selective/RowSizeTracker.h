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

#pragma once

#include <memory>

#include "velox/dwio/common/TypeWithId.h"
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"

#include "velox/dwio/common/ScanSpec.h"

namespace facebook::nimble {

class RowSizeTracker {
 public:
  explicit RowSizeTracker(
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& typeWithId);

  ~RowSizeTracker() = default;

  void
  update(size_t nodeIdx, size_t memoryFootprint, velox::vector_size_t rowCount);

  void applyProjection(size_t nodeIdx, bool projected);

  void finalizeProjection();

  size_t getCurrentMaxRowSize() const;

 private:
  // Populates cell sizes and marks variable length nodes.
  // Returns the next node index (or the size of the current type subtree).
  size_t initFromSchema(
      const facebook::velox::Type& type,
      size_t nodeId,
      bool isParentVariableLength);

  const std::shared_ptr<const velox::dwio::common::TypeWithId>& typeWithId_;
  std::vector<std::optional<size_t>> cellSizes_;
  // Avoid providing aggressive row sizes when we don't have any materialized
  // variable length nodes.
  velox::SelectivityVector variableLengthNodes_;
  velox::SelectivityVector projectedNodes_;
  bool projectionFinalized_{false};
  size_t maxRowSize_{0};

  friend class RowSizeTrackerTest;
};

} // namespace facebook::nimble
