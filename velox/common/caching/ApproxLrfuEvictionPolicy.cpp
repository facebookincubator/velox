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

#include "velox/common/caching/ApproxLrfuEvictionPolicy.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/caching/AsyncDataCache.h"

namespace facebook::velox::cache {

namespace {

class ApproxLrfuCursor : public EvictionCandidateCursor {
 public:
  ApproxLrfuCursor(
      ApproxLrfuShardState& state,
      const std::deque<std::unique_ptr<AsyncDataCacheEntry>>& entries,
      uint32_t& clockHand,
      uint32_t& eventCounter,
      int32_t maxEvictionSamples,
      int32_t evictionPercentile,
      bool evictAllUnpinned)
      : state_(state),
        entries_(entries),
        clockHand_(clockHand),
        eventCounter_(eventCounter),
        maxEvictionSamples_(maxEvictionSamples),
        evictionPercentile_(evictionPercentile),
        evictAllUnpinned_(evictAllUnpinned),
        size_(entries.size()),
        entryIndex_(size_ == 0 ? 0 : clockHand % size_),
        iter_(size_ == 0 ? entries.begin() : entries.begin() + entryIndex_),
        now_(accessTime()) {}

  Candidate next() override {
    if (size_ == 0) {
      return {};
    }
    while (++counter_ <= size_) {
      advance();
      auto* candidate = iter_->get();
      if (candidate == nullptr) {
        continue;
      }
      ++numChecked_;
      if (state_.evictionThreshold == ApproxLrfuEvictionPolicy::kNoThreshold ||
          eventCounter_ > size_ / 4 || numChecked_ > size_ / 8) {
        now_ = accessTime();
        calibrateThreshold();
        numChecked_ = 0;
        eventCounter_ = 0;
      }
      if (candidate->numPins() != 0) {
        continue;
      }
      if (evictAllUnpinned_ || !candidate->key().fileNum.hasValue()) {
        return {candidate, static_cast<int32_t>(entryIndex_), 0};
      }
      const int32_t score = candidate->score(now_);
      if (score >= state_.evictionThreshold) {
        return {candidate, static_cast<int32_t>(entryIndex_), score};
      }
    }
    return {};
  }

 private:
  void advance() {
    if (++iter_ == entries_.end()) {
      iter_ = entries_.begin();
      entryIndex_ = 0;
    } else {
      ++entryIndex_;
    }
    ++clockHand_;
  }

  void calibrateThreshold() {
    const auto numSamples = std::min<int32_t>(maxEvictionSamples_, size_);
    const auto sampleNow = accessTime();
    auto sampleIndex = clockHand_ % size_;
    const auto step = size_ / numSamples;
    auto sampleIter = entries_.begin() + sampleIndex;
    state_.evictionThreshold = percentile<int32_t>(
        [&]() -> int32_t {
          AsyncDataCacheEntry* element = sampleIter->get();
          const int32_t s = element ? element->score(sampleNow) : 0;
          if (sampleIndex + step >= size_) {
            sampleIndex = (sampleIndex + step) % size_;
            sampleIter = entries_.begin() + sampleIndex;
          } else {
            sampleIndex += step;
            sampleIter += step;
          }
          return s;
        },
        numSamples,
        evictionPercentile_);
  }

  ApproxLrfuShardState& state_;
  const std::deque<std::unique_ptr<AsyncDataCacheEntry>>& entries_;
  uint32_t& clockHand_;
  uint32_t& eventCounter_;
  const int32_t maxEvictionSamples_;
  const int32_t evictionPercentile_;
  const bool evictAllUnpinned_;
  const size_t size_;
  size_t entryIndex_;
  std::deque<std::unique_ptr<AsyncDataCacheEntry>>::const_iterator iter_;
  AccessTime now_;
  size_t counter_{0};
  int32_t numChecked_{0};
};

} // namespace

ApproxLrfuEvictionPolicy::ApproxLrfuEvictionPolicy(
    int32_t maxEvictionSamples,
    int32_t evictionPercentile)
    : maxEvictionSamples_(maxEvictionSamples),
      evictionPercentile_(evictionPercentile) {
  VELOX_CHECK_GT(maxEvictionSamples_, 0);
  VELOX_CHECK_GT(evictionPercentile_, 0);
  VELOX_CHECK_LE(evictionPercentile_, 100);
}

std::unique_ptr<EvictionPolicyShardState>
ApproxLrfuEvictionPolicy::makeShardState() const {
  return std::make_unique<ApproxLrfuShardState>();
}

std::unique_ptr<EvictionCandidateCursor>
ApproxLrfuEvictionPolicy::createEvictionCursorLocked(
    EvictionPolicyShardState& baseState,
    const std::deque<std::unique_ptr<AsyncDataCacheEntry>>& entries,
    uint32_t& clockHand,
    uint32_t& eventCounter,
    bool evictAllUnpinned) const {
  auto& state = static_cast<ApproxLrfuShardState&>(baseState);
  return std::make_unique<ApproxLrfuCursor>(
      state,
      entries,
      clockHand,
      eventCounter,
      maxEvictionSamples_,
      evictionPercentile_,
      evictAllUnpinned);
}

} // namespace facebook::velox::cache
