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

#include "velox/exec/rpc/CongestionController.h"

#include <algorithm>
#include <cmath>

namespace facebook::velox::exec::rpc {

void CongestionController::onError() {
  const auto prevEffective = effective_;
  effective_ = std::max<int64_t>(effective_ / 2, minWindow_);
  if (effective_ < prevEffective) {
    ++numShrinks_;
  }
  // Drop the in-progress sample window so partial pre-overload samples don't
  // contaminate the next gradient computation. baselineRttNs_ is deliberately
  // preserved (not reset): it represents the unloaded RTT, and keeping it makes
  // the next gradient stay < 1 while the backend is still degraded (so the
  // window stays small) yet recover once latency drops back toward the
  // baseline. Re-seeding the baseline from a post-overload (elevated) RTT would
  // instead treat the degraded latency as "normal" and let the window grow back
  // into the overload.
  numWindowSamples_ = 0;
  windowMinRttNs_ = 0;
}

void CongestionController::onSample(int64_t rttNs) {
  if (rttNs <= 0) {
    return;
  }

  // Track the minimum RTT across the current sample window.
  if (numWindowSamples_ == 0 || rttNs < windowMinRttNs_) {
    windowMinRttNs_ = rttNs;
  }
  if (++numWindowSamples_ < kSamplesPerWindow) {
    return;
  }

  const int64_t observedRttNs = windowMinRttNs_;
  numWindowSamples_ = 0;
  windowMinRttNs_ = 0;

  if (baselineRttNs_ == 0) {
    // The first full window establishes the unloaded baseline; hold steady.
    baselineRttNs_ = observedRttNs;
    return;
  }

  // gradient < 1 when queueing has lifted the observed RTT above the baseline;
  // clamp so a single window at most halves or holds the window.
  const double gradient = std::clamp(
      static_cast<double>(baselineRttNs_) / static_cast<double>(observedRttNs),
      0.5,
      1.0);
  // sqrt headroom keeps probing upward when latency is flat (gradient ~ 1), so
  // the window is never pinned by a fixed ceiling. stepCoef_ scales how hard it
  // probes (1.0 = the plain sqrt headroom).
  const double headroom =
      stepCoef_ * std::sqrt(static_cast<double>(effective_));
  const auto newWindow = static_cast<int64_t>(
      static_cast<double>(effective_) * gradient + headroom);
  const auto prevEffective = effective_;
  effective_ = std::clamp(newWindow, minWindow_, maxWindow_);
  if (effective_ < prevEffective) {
    ++numShrinks_;
  }

  // Track the baseline slowly toward the observed RTT so the controller settles
  // at the knee under sustained load instead of shrinking forever.
  baselineRttNs_ = static_cast<int64_t>(
      static_cast<double>(baselineRttNs_) * (1.0 - kBaselineSmoothing) +
      static_cast<double>(observedRttNs) * kBaselineSmoothing);
}

} // namespace facebook::velox::exec::rpc
