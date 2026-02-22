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

#include <algorithm>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/HilbertIndex.h"
#include "velox/exec/SpatialIndex.h"

namespace facebook::velox::exec {

std::vector<size_t> RTreeLevel::query(
    const Envelope& queryEnv,
    const std::vector<size_t>& branchIndices) const {
  std::vector<size_t> result;

  for (size_t branchIdx : branchIndices) {
    size_t startIdx = branchIdx * branchSize_;
    size_t endIdx = std::min(startIdx + branchSize_, minXs_.size());
    for (size_t idx = startIdx; idx < endIdx; ++idx) {
      bool intersects = (queryEnv.maxX >= minXs_[idx]) &&
          (queryEnv.maxY >= minYs_[idx]) && (queryEnv.minX <= maxXs_[idx]) &&
          (queryEnv.minY <= maxYs_[idx]);
      if (intersects) {
        result.push_back(idx);
      }
    }
  }

  return result;
}

namespace {
std::pair<RTreeLevel, std::vector<Envelope>> buildLevel(
    uint32_t branchSize,
    const std::vector<Envelope>& envelopes) {
  std::vector<float> minXs;
  minXs.reserve(envelopes.size());
  std::vector<float> minYs;
  minYs.reserve(envelopes.size());
  std::vector<float> maxXs;
  maxXs.reserve(envelopes.size());
  std::vector<float> maxYs;
  maxYs.reserve(envelopes.size());

  std::vector<Envelope> parentEnvelopes;
  parentEnvelopes.reserve((envelopes.size() + branchSize - 1) / branchSize);
  Envelope currentBounds = Envelope::empty();

  uint32_t idx = 0;
  for (const auto& env : envelopes) {
    ++idx;
    currentBounds.maxX = std::max(currentBounds.maxX, env.maxX);
    currentBounds.maxY = std::max(currentBounds.maxY, env.maxY);
    currentBounds.minX = std::min(currentBounds.minX, env.minX);
    currentBounds.minY = std::min(currentBounds.minY, env.minY);
    if (idx % branchSize == 0) {
      parentEnvelopes.push_back(currentBounds);
      currentBounds = Envelope::empty();
    }

    minXs.push_back(env.minX);
    minYs.push_back(env.minY);
    maxXs.push_back(env.maxX);
    maxYs.push_back(env.maxY);
  }

  if (!currentBounds.isEmpty()) {
    parentEnvelopes.push_back(currentBounds);
  }

  return {
      RTreeLevel(
          branchSize,
          std::move(minXs),
          std::move(minYs),
          std::move(maxXs),
          std::move(maxYs)),
      std::move(parentEnvelopes)};
}
} // namespace

SpatialIndex::SpatialIndex(
    Envelope bounds,
    std::vector<Envelope> envelopes,
    uint32_t branchSize)
    : branchSize_(branchSize), bounds_(std::move(bounds)) {
  VELOX_CHECK_GT(branchSize_, 1);

  if (!bounds_.isEmpty()) {
    HilbertIndex hilbert(
        bounds_.minX, bounds_.minY, bounds_.maxX, bounds_.maxY);

    std::sort(
        envelopes.begin(), envelopes.end(), [&](const auto& a, const auto& b) {
          return hilbert.indexOf(a.minX, a.minY) <
              hilbert.indexOf(b.minX, b.minY);
        });
  }

  rowIndices_.reserve(envelopes.size());
  for (const auto& env : envelopes) {
    VELOX_CHECK(env.minX >= bounds_.minX);
    VELOX_CHECK(env.minY >= bounds_.minY);
    VELOX_CHECK(env.maxX <= bounds_.maxX);
    VELOX_CHECK(env.maxY <= bounds_.maxY);
    rowIndices_.push_back(env.rowIndex);
  }

  if (envelopes.size() > 0) {
    size_t numLevels =
        std::ceil(std::log(envelopes.size()) / std::log(branchSize_));
    levels_.reserve(numLevels);
  }

  while (envelopes.size() > branchSize_) {
    auto [level, parentEnvelopes] = buildLevel(branchSize_, envelopes);
    levels_.push_back(std::move(level));
    envelopes = std::move(parentEnvelopes);
  }

  if (envelopes.size() > 1 || levels_.empty()) {
    levels_.push_back(buildLevel(branchSize_, envelopes).first);
  }

  VELOX_CHECK_GT(branchSize_ + 1, levels_.back().size());
}

std::vector<vector_size_t> SpatialIndex::query(const Envelope& queryEnv) const {
  std::vector<vector_size_t> result;
  if (!Envelope::intersects(queryEnv, bounds_)) {
    return result;
  }

  size_t thisLevel = levels_.size() - 1;
  VELOX_CHECK_GT(levels_[thisLevel].size(), 0);
  VELOX_CHECK_GT(branchSize_ + 1, levels_[thisLevel].size());

  // The top level should have only one branch.
  std::vector<size_t> childIndices = {0};
  for (; thisLevel > 0; --thisLevel) {
    // Avoiding thisLevel = 0 due to int underflow
    childIndices = levels_[thisLevel].query(queryEnv, childIndices);
    // If we have no matches, return.
    if (childIndices.empty()) {
      return result;
    }
  }

  // We're at level 0 now.  The indices index into rowIndices.
  VELOX_DCHECK_EQ(thisLevel, 0);
  childIndices = levels_[thisLevel].query(queryEnv, childIndices);
  result.reserve(childIndices.size());
  for (auto idx : childIndices) {
    result.push_back(rowIndices_[idx]);
  }

  return result;
}

Envelope SpatialIndex::bounds() const {
  return bounds_;
}

} // namespace facebook::velox::exec
