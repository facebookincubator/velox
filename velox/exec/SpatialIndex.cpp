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

#include <algorithm>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/SpatialIndex.h"

namespace facebook::velox::exec {

SpatialIndex::SpatialIndex(std::vector<Envelope> envelopes) {
  std::ranges::sort(envelopes, {}, &Envelope::minX);

  minXs_.reserve(envelopes.size());
  minYs_.reserve(envelopes.size());
  maxXs_.reserve(envelopes.size());
  maxYs_.reserve(envelopes.size());
  rowIndices_.reserve(envelopes.size());

  for (const auto& env : envelopes) {
    bounds_.maxX = std::max(bounds_.maxX, env.maxX);
    bounds_.maxY = std::max(bounds_.maxY, env.maxY);
    bounds_.minX = std::min(bounds_.minX, env.minX);
    bounds_.minY = std::min(bounds_.minY, env.minY);
    minXs_.push_back(env.minX);
    minYs_.push_back(env.minY);
    maxXs_.push_back(env.maxX);
    maxYs_.push_back(env.maxY);
    rowIndices_.push_back(env.rowIndex);
  }
}

std::vector<int32_t> SpatialIndex::query(const Envelope& queryEnv) const {
  std::vector<int32_t> result;
  if (!Envelope::intersects(queryEnv, bounds_)) {
    return result;
  }

  // Find the last minX that is <= queryEnv.maxX . These first envelopes
  // are the only ones that can intersect the query envelope.
  // `it` is _one past_ the last element, so we iterate up to it - 1.
  auto it = std::upper_bound(minXs_.begin(), minXs_.end(), queryEnv.maxX);
  if (it == minXs_.begin()) {
    return result;
  }

  auto lastIdx = std::distance(minXs_.begin(), it);
  VELOX_CHECK_GT(lastIdx, 0);

  for (size_t idx = 0; idx < lastIdx; ++idx) {
    bool intersects = (queryEnv.maxY >= minYs_[idx]) &&
        (queryEnv.minX <= maxXs_[idx]) && (queryEnv.minY <= maxYs_[idx]);
    if (intersects) {
      result.push_back(rowIndices_[idx]);
    }
  }

  return result;
}

Envelope SpatialIndex::bounds() const {
  return bounds_;
}

} // namespace facebook::velox::exec
