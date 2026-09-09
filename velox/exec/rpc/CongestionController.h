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
#include <cmath>
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
/// The window is carried as a double and reported as its floor. That is load
/// bearing, not a style choice: one recomputation changes the window by
/// stepCoef*sqrt(w) - w*(1 - gradient), which is well under 1 for most (w,
/// gradient) pairs. Rounding that to an integer every window discards the
/// increment, so an integer-valued window stalls whenever
///   stepCoef*sqrt(w) < w*(1 - gradient) + 1,
/// which at w = 1 with the default stepCoef holds for every gradient below 1.
/// Accumulating in floating point lets the sub-unit steps add up, and the
/// window tracks the law instead of freezing. While the baseline is
/// held, the law has the fixed point
///   w* = stepCoef^2 / (1 - gradient)^2,
/// which is how to size stepCoef against a backend that needs at least N
/// concurrent requests to reach its throughput knee: stepCoef >= sqrt(N) *
/// (1 - gradient). That fixed point governs the transient only — under
/// sustained elevation the baseline EMA absorbs the new latency, the gradient
/// returns toward 1, and the window resumes probing upward by design.
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
  ///        The window accumulates in floating point, so a value < 1 still
  ///        grows; it just takes more windows to cross each integer. Size it
  ///        against the concurrency a backend needs to reach its throughput
  ///        knee -- see the fixed point in the class comment above.
  /// maxWindow is clamped into [1, kMaxWindowCeiling] first so the minWindow
  /// clamp range is always valid (a < 1 ceiling would make std::clamp
  /// undefined) and so the ceiling round-trips through the double accumulator
  /// exactly. Both bounds are defense in depth: maxWindow is internal and no
  /// current call site is outside them.
  CongestionController(
      int64_t startWindow,
      int64_t maxWindow,
      int64_t minWindow = 1,
      double stepCoef = 1.0)
      : maxWindow_{std::clamp<int64_t>(maxWindow, 1, kMaxWindowCeiling)},
        minWindow_{std::clamp<int64_t>(minWindow, int64_t{1}, maxWindow_)},
        stepCoef_{std::max(0.0, stepCoef)},
        // Clamp the starting window into [minWindow_, maxWindow_] so limit()
        // is in range from construction, before the first onError/onSample.
        effective_{std::clamp<double>(
            static_cast<double>(startWindow),
            static_cast<double>(minWindow_),
            static_cast<double>(maxWindow_))} {}

  /// Returns the current admission limit (max in-flight units before
  /// backpressure): the floor of the accumulated window, never below
  /// minWindow.
  int64_t limit() const {
    return static_cast<int64_t>(std::floor(effective_));
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
  // Largest ceiling the window may take. 2^53 is the last integer a double
  // represents exactly, so effective_ never exceeds a value that converts back
  // to int64_t. rpc.congestion.max_window is user-settable and unbounded;
  // INT64_MAX would round *up* to 2^63 as a double, and converting that back is
  // undefined (x86 yields INT64_MIN, which would stall dispatch outright).
  static constexpr int64_t kMaxWindowCeiling = int64_t{1} << 53;

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
  // Accumulated admission window. Fractional so sub-unit growth is not lost
  // between recomputations; limit() reports its floor.
  double effective_{1.0};

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
