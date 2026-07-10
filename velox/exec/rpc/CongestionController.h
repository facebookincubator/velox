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

#include <algorithm>
#include <cstdint>

namespace facebook::velox::exec::rpc {

/// A latency-gradient concurrency window shared by both RPC streaming modes.
///
/// The window auto-learns the throughput sweet spot from round-trip latency,
/// with no hand-tuned ceiling. It keeps a slow baseline of the unloaded
/// round-trip time and, each sample window, computes
/// gradient = baselineRtt / observedRtt:
///   - gradient < 1 means queueing is building (observed RTT above the unloaded
///     baseline), so the window shrinks multiplicatively *before* requests time
///     out;
///   - when latency is flat (gradient ~ 1) the window probes upward by a
///     stepCoef * sqrt(window) headroom term, so it is never pinned by a fixed
///     limit.
/// Overload (onError) halves the window instantly as a fast path. The latency
/// signal is the per-window *minimum* RTT, which is robust to per-request size
/// variance (e.g. variable LLM output length) because only queueing lifts the
/// fastest request in a window.
///
/// A window constructed with startWindow == maxWindow that is never fed a
/// sample stays fixed at that value — this is how callers pin a deterministic
/// window (tests/config) without a separate code path.
///
/// The window is unit-agnostic: the owner decides whether one "unit" is a row
/// (PER_ROW) or a batch (BATCH).
///
/// Not thread-safe; the owner (RPCState) serializes all access under its mutex.
class CongestionController {
 public:
  /// Default-constructs an inert {1, 1} window. RPCState always reassigns the
  /// window in setStreamingMode() before any dispatch, so this default is never
  /// used in production; {1, 1} just keeps the unconfigured state safe.
  CongestionController() = default;

  /// @param startWindow Initial admission limit (the starting parallelism).
  /// @param maxWindow Safety ceiling the window may grow toward; not a tuning
  ///        knob, since the gradient self-limits well below it via latency.
  /// @param minWindow Floor the window may shrink to under overload (default 1,
  ///        i.e. never fully stall). A higher floor refuses to back off below
  ///        it; keep at 1 unless a deployment proves otherwise. Clamped to
  ///        [1, maxWindow]: these may come from user-settable session
  ///        properties, and minWindow <= 0 would stall dispatch while
  ///        minWindow > maxWindow would make the internal std::clamp undefined.
  /// @param stepCoef Multiplier on the sqrt(window) additive-increase headroom
  ///        (default 1.0); lower values converge tighter at small windows.
  ///        Clamped to >= 0 (a negative coefficient would shrink on probe).
  ///        A value < 1 can pin the window at small sizes because the update
  ///        truncates to an integer (e.g. stepCoef 0.5 at window 1: 1 + 0.5 ->
  ///        1); keep >= 1 unless minimal ramp-up is intended.
  /// maxWindow is clamped to >= 1 first so the minWindow clamp range is always
  /// valid (a < 1 ceiling would make std::clamp undefined). maxWindow is
  /// internal and never < 1 at any current call site; this is defense in depth.
  CongestionController(
      int64_t startWindow,
      int64_t maxWindow,
      int64_t minWindow = 1,
      double stepCoef = 1.0)
      : maxWindow_{std::max<int64_t>(maxWindow, 1)},
        minWindow_{std::clamp<int64_t>(minWindow, int64_t{1}, maxWindow_)},
        stepCoef_{std::max(0.0, stepCoef)},
        // Clamp the starting window into [minWindow_, maxWindow_] so limit()
        // is in range from construction, before the first onError/onSample.
        effective_{std::clamp<int64_t>(startWindow, minWindow_, maxWindow_)} {}

  /// Returns the current admission limit (max in-flight units before
  /// backpressure).
  int64_t limit() const {
    return effective_;
  }

  /// Returns the learned baseline RTT (nanos), or 0 before the first full
  /// sample window.
  int64_t baselineRttNs() const {
    return baselineRttNs_;
  }

  /// Returns the number of window-shrink events (onError halving + gradient
  /// shrinks) since construction.
  int64_t numShrinks() const {
    return numShrinks_;
  }

  /// Halves the window, floored at 1 so dispatch never fully stalls. The fast
  /// overload path (rate limit / timeout).
  void onError();

  /// Feeds one completed unit's round-trip latency (nanos) into the gradient
  /// learner, recomputing the window once per kSamplesPerWindow samples.
  void onSample(int64_t rttNs);

 private:
  // Samples accumulated before each gradient recomputation.
  static constexpr int64_t kSamplesPerWindow = 8;
  // EMA weight applied to the observed RTT when tracking the baseline.
  static constexpr double kBaselineSmoothing = 0.1;

  // Safety ceiling the window may grow toward.
  int64_t maxWindow_{1};
  // Floor the window may shrink to under overload.
  int64_t minWindow_{1};
  // Multiplier on the sqrt(window) additive-increase headroom.
  double stepCoef_{1.0};
  // Current admission limit (the value limit() returns).
  int64_t effective_{1};

  // Slow EMA of per-window minimum RTT (nanos); 0 until the first window.
  int64_t baselineRttNs_{0};
  // Minimum RTT (nanos) seen so far in the current sample window, and the
  // number of samples accumulated in it.
  int64_t windowMinRttNs_{0};
  int64_t numWindowSamples_{0};

  // Count of window-shrink events (onError halving + gradient shrinks).
  int64_t numShrinks_{0};
};

} // namespace facebook::velox::exec::rpc
