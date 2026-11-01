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

#pragma once

#include <limits>

#include "velox/common/caching/EvictionPolicy.h"

namespace facebook::velox::cache {

/// Holds per-shard state for ApproxLrfuEvictionPolicy. evictionThreshold is
/// the maximum retainable score; entries scoring at or above it are
/// evictable.
class ApproxLrfuShardState : public EvictionPolicyShardState {
 public:
  int32_t evictionThreshold{std::numeric_limits<int32_t>::max()};
};

/// Implements the adaptive approx-LRFU policy. Retention score is
/// (now - lastUse) / (1 + numUses); the eviction threshold is the
/// evictionPercentile-th percentile of maxEvictionSamples sampled scores,
/// recalibrated when hit or scan counters exceed shard-size fractions.
class ApproxLrfuEvictionPolicy : public EvictionPolicy {
 public:
  static constexpr int32_t kNoThreshold = std::numeric_limits<int32_t>::max();
  static constexpr int32_t kDefaultMaxEvictionSamples = 10;
  static constexpr int32_t kDefaultEvictionPercentile = 80;

  explicit ApproxLrfuEvictionPolicy(
      int32_t maxEvictionSamples = kDefaultMaxEvictionSamples,
      int32_t evictionPercentile = kDefaultEvictionPercentile);

  std::unique_ptr<EvictionPolicyShardState> makeShardState() const override;

  void onEntryAccessLocked(
      AsyncDataCacheEntry& /*entry*/,
      EvictionPolicyShardState& /*shardState*/) const override {}

  void onEntryInsertedLocked(
      AsyncDataCacheEntry& /*entry*/,
      EvictionPolicyShardState& /*shardState*/) const override {}

  void onEntryRemovedLocked(
      AsyncDataCacheEntry& /*entry*/,
      EvictionPolicyShardState& /*shardState*/) const override {}

  std::unique_ptr<EvictionCandidateCursor> createEvictionCursorLocked(
      EvictionPolicyShardState& shardState,
      const std::deque<std::unique_ptr<AsyncDataCacheEntry>>& entries,
      uint32_t& clockHand,
      uint32_t& eventCounter,
      bool evictAllUnpinned) const override;

  int32_t maxEvictionSamples() const {
    return maxEvictionSamples_;
  }

  int32_t evictionPercentile() const {
    return evictionPercentile_;
  }

 private:
  const int32_t maxEvictionSamples_;
  const int32_t evictionPercentile_;
};

} // namespace facebook::velox::cache
