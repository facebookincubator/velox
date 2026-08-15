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

#include <gtest/gtest.h>

namespace facebook::velox::exec::rpc {
namespace {

// Feeds a full sample window (kSamplesPerWindow = 8) of identical RTTs.
void feedWindow(CongestionController& window, int64_t rttNs) {
  for (int i = 0; i < 8; ++i) {
    window.onSample(rttNs);
  }
}

TEST(CongestionControllerTest, startsAtStartWindow) {
  CongestionController window{4, 64};
  EXPECT_EQ(window.limit(), 4);
}

TEST(CongestionControllerTest, multiplicativeDecreaseFloorsAtOne) {
  CongestionController window{8, 64};
  window.onError();
  EXPECT_EQ(window.limit(), 4);
  window.onError();
  EXPECT_EQ(window.limit(), 2);
  window.onError();
  EXPECT_EQ(window.limit(), 1);
  // Floored at 1 so dispatch never fully stalls.
  window.onError();
  EXPECT_EQ(window.limit(), 1);
}

TEST(CongestionControllerTest, customFloorStopsDecreaseAboveOne) {
  // A minWindow > 1 stops the multiplicative decrease at the floor instead
  // of 1.
  CongestionController window{16, 64, /*minWindow*/ 4};
  window.onError();
  EXPECT_EQ(window.limit(), 8);
  window.onError();
  EXPECT_EQ(window.limit(), 4);
  // Floored at minWindow, not 1.
  window.onError();
  EXPECT_EQ(window.limit(), 4);
}

TEST(CongestionControllerTest, fixedWhenNeverFedSamples) {
  // A window that is never fed a sample stays at its start value — this is how
  // callers pin a deterministic window (start == max) without a separate path.
  CongestionController window{2, 2};
  EXPECT_EQ(window.limit(), 2);
  // onError still applies as the overload fast path.
  window.onError();
  EXPECT_EQ(window.limit(), 1);
}

TEST(CongestionControllerTest, holdsThenGrowsUnderFlatLatency) {
  auto window = CongestionController{4, 64};
  // The first full window establishes the baseline RTT; the window is
  // unchanged.
  feedWindow(window, 1'000'000);
  EXPECT_EQ(window.limit(), 4);
  // Flat latency -> gradient ~ 1, so the window grows by the sqrt(4) = 2
  // headroom term: 4 * 1 + 2 = 6.
  feedWindow(window, 1'000'000);
  EXPECT_EQ(window.limit(), 6);
}

TEST(CongestionControllerTest, stepCoefScalesGrowthHeadroom) {
  // stepCoef scales the additive-increase headroom. With coef 2.0 a
  // flat-latency window grows by 2.0 * sqrt(4) = 4 (vs the default 2): 4 * 1 +
  // 4 = 8.
  auto window = CongestionController{4, 64, /*minWindow*/ 1, /*stepCoef*/ 2.0};
  feedWindow(window, 1'000'000); // baseline
  feedWindow(window, 1'000'000); // flat latency -> grows by the scaled headroom
  EXPECT_EQ(window.limit(), 8);
}

TEST(CongestionControllerTest, clampsNonPositiveMinWindowToOne) {
  // minWindow may come from a user-settable session property; <= 0 would let
  // the window collapse to 0 and stall dispatch. It is clamped to 1.
  CongestionController window{8, 64, /*minWindow*/ 0};
  for (int i = 0; i < 5; ++i) {
    window.onError();
  }
  EXPECT_EQ(window.limit(), 1);
}

TEST(CongestionControllerTest, clampsMinWindowAboveMaxToMax) {
  // minWindow > maxWindow would make the internal std::clamp(newWindow, min,
  // max) undefined. It is clamped to maxWindow, so the window stays
  // well-defined and never exceeds the ceiling.
  CongestionController window{8, 64, /*minWindow*/ 100};
  window.onError();
  EXPECT_EQ(window.limit(), 64);
  // A full flat-latency window must not push it above the ceiling (no UB).
  feedWindow(window, 1'000'000); // baseline
  feedWindow(window, 1'000'000);
  EXPECT_EQ(window.limit(), 64);
}

TEST(CongestionControllerTest, clampsNegativeStepCoefToZero) {
  // A negative stepCoef would make the additive-increase term shrink the window
  // on a healthy probe. It is clamped to 0 (no growth), never negative.
  CongestionController window{4, 64, /*minWindow*/ 1, /*stepCoef*/ -2.0};
  feedWindow(window, 1'000'000); // baseline
  feedWindow(
      window, 1'000'000); // flat latency -> headroom 0 -> holds, no shrink
  EXPECT_EQ(window.limit(), 4);
}

TEST(CongestionControllerTest, clampsMaxWindowBelowOneToOne) {
  // A maxWindow < 1 would make both the ctor's minWindow clamp and onSample's
  // window clamp undefined (lo > hi). maxWindow is clamped to >= 1, so the
  // window stays well-defined and collapses to 1 rather than hitting UB.
  CongestionController window{2, /*maxWindow*/ 0};
  feedWindow(window, 1'000'000); // baseline
  feedWindow(
      window, 1'000'000); // grows, then clamps to the corrected ceiling 1
  EXPECT_EQ(window.limit(), 1);
}

TEST(CongestionControllerTest, clampsStartWindowIntoRange) {
  // The starting window is clamped into [minWindow, maxWindow] so limit() is in
  // range from construction (before any sample/error).
  CongestionController above{/*startWindow*/ 500, /*maxWindow*/ 256};
  EXPECT_EQ(above.limit(), 256); // start above the ceiling -> clamped down
  CongestionController below{/*startWindow*/ 1, /*maxWindow*/ 256, /*min*/ 4};
  EXPECT_EQ(below.limit(), 4); // start below the floor -> clamped up
}

TEST(CongestionControllerTest, shrinksWhenLatencyRises) {
  auto window = CongestionController{16, 64};
  feedWindow(window, 1'000'000); // baseline
  // 4x latency -> gradient clamps to 0.5: 16 * 0.5 + sqrt(16) = 8 + 4 = 12.
  feedWindow(window, 4'000'000);
  EXPECT_EQ(window.limit(), 12);
}

TEST(CongestionControllerTest, neverExceedsMax) {
  auto window = CongestionController{60, 64};
  feedWindow(window, 1'000'000); // baseline
  // 60 * 1 + sqrt(60) ~ 67 -> clamped to the safety ceiling of 64.
  feedWindow(window, 1'000'000);
  EXPECT_EQ(window.limit(), 64);
}

TEST(CongestionControllerTest, errorHalvesIndependentOfGradient) {
  auto window = CongestionController{16, 64};
  window.onError();
  EXPECT_EQ(window.limit(), 8);
}

TEST(CongestionControllerTest, ignoresPartialWindow) {
  auto window = CongestionController{4, 64};
  // Fewer than a full sample window -> no recomputation, window unchanged.
  for (int i = 0; i < 7; ++i) {
    window.onSample(1'000'000);
  }
  EXPECT_EQ(window.limit(), 4);
}

TEST(CongestionControllerTest, shrinksTowardFloorUnderSustainedLatency) {
  auto window = CongestionController{16, 64};
  feedWindow(window, 1'000'000); // baseline
  // Sustained high latency drives repeated multiplicative back-off.
  for (int round = 0; round < 5; ++round) {
    feedWindow(window, 100'000'000);
  }
  EXPECT_LT(window.limit(), 16);
  EXPECT_GE(window.limit(), 1);
}

TEST(CongestionControllerTest, ignoresNonPositiveSamples) {
  auto window = CongestionController{4, 64};
  for (int i = 0; i < 16; ++i) {
    window.onSample(0);
    window.onSample(-5);
  }
  EXPECT_EQ(window.limit(), 4);
}

TEST(CongestionControllerTest, doesNotShrinkWhenMinRttStable) {
  // The signal is the per-window MINIMUM RTT, so high per-request variance does
  // not shrink the window as long as the minimum tracks the baseline (no
  // queueing). Guards the core robustness claim against, e.g., variable LLM
  // output length being mistaken for overload.
  auto window = CongestionController{16, 64};
  feedWindow(window, 1'000'000); // baseline
  const int64_t varied[8] = {
      1'000'000,
      5'000'000,
      2'000'000,
      1'000'000,
      8'000'000,
      3'000'000,
      1'000'000,
      4'000'000};
  for (auto rttNs : varied) {
    window.onSample(rttNs);
  }
  // min == baseline -> gradient ~ 1 -> grows by headroom, never shrinks.
  EXPECT_GE(window.limit(), 16);
}

TEST(CongestionControllerTest, errorDiscardsPartialSampleWindow) {
  // onError() drops the in-progress sample window so pre-overload RTTs do not
  // contaminate the next gradient computation. Here a stale low sample would,
  // if leaked, make the next window grow; discarded, the next window shrinks.
  auto window = CongestionController{16, 64};
  feedWindow(window, 1'000'000); // baseline = 1e6, window 16
  window.onSample(1); // partial window with an artificially low min
  window.onError(); // 16 -> 8, partial window (min=1) discarded
  EXPECT_EQ(window.limit(), 8);
  // A full window at high latency: with the stale min=1 discarded the observed
  // min is high -> gradient clamps to 0.5 -> 8*0.5 + sqrt(8) = 6.
  feedWindow(window, 10'000'000);
  EXPECT_EQ(window.limit(), 6);
}

TEST(CongestionControllerTest, baselineTracksSustainedLatencyAndRecovers) {
  // The baseline EMA tracks the observed RTT toward a sustained level. Without
  // it, a constant 2x latency keeps the gradient pinned at 0.5 and the window
  // settles at the sqrt-headroom equilibrium (exactly 4: 4*0.5 + sqrt(4) = 4).
  // With the EMA the baseline rises toward the sustained level, the gradient
  // returns to ~1, and the window recovers well above that plain equilibrium.
  // Guards against a frozen-baseline regression (shrink-forever under load).
  auto window = CongestionController{16, 64};
  feedWindow(window, 1'000'000); // establish baseline at 1e6
  // Sustained 2x latency over many windows: the baseline EMA converges to ~2e6.
  for (int round = 0; round < 60; ++round) {
    feedWindow(window, 2'000'000);
  }
  EXPECT_GT(window.limit(), 4);
}

} // namespace
} // namespace facebook::velox::exec::rpc
