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

/// RPCRateLimiterTest - Tests for per-backend RPC limiter control.
///
/// RPCRateLimiter provides per-backend concurrency limits with FIFO waiter
/// notification and RAII token-based slot management.
///
/// Tests cover:
/// - acquireAndRelease: Token acquire increments, destruction decrements.
/// - backpressureWhenAtLimit: admitOrWait waits once at capacity.
/// - backpressureReliefOnRelease: Waiter notified when token released.
/// - fifoWaiterNotification: Multiple waiters notified in FIFO order.
/// - perBackendIsolation: Different backends have independent limits.
/// - perBackendMaxPending: a per-backend ceiling overrides the process default.
/// - defaultMaxPending: the process default applies to unconfigured backends.
/// - tokenMoveSemantics: Move constructor/assignment transfer ownership.
/// - testingResetClearsStateInPlace: Reset restores defaults and zeroes
///   pending without destroying a backend an outstanding token points at.
/// - adaptiveIsPerBackend: Two backends adapt independently.
/// - adaptiveFactorsAreIndependent: Each backend uses its own decrease factor.

#include <folly/futures/Future.h>
#include <folly/futures/ManualTimekeeper.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/rpc/RPCRateLimiter.h"

namespace facebook::velox::exec::rpc {
namespace {

using namespace std::chrono_literals;

// RPCRateLimiter takes a whole Config per backend. These helpers amend one
// field group at a time so each test states only the setting it cares about.
RPCRateLimiter& limiterFor(const std::string& admissionKey) {
  return RPCRateLimiterRegistry::global().get(admissionKey);
}

void setCeiling(const std::string& admissionKey, int64_t ceiling) {
  auto& limiter = limiterFor(admissionKey);
  auto config = limiter.config();
  config.ceiling = ceiling;
  limiter.configure(config);
}

void setAdaptive(
    const std::string& admissionKey,
    bool enabled,
    double floor,
    double decreaseFactor) {
  auto& limiter = limiterFor(admissionKey);
  auto config = limiter.config();
  config.adaptive = enabled;
  config.floor = floor;
  config.decreaseFactor = decreaseFactor;
  limiter.configure(config);
}

class ManualPacingClock {
 public:
  std::chrono::steady_clock::time_point now() const {
    return now_;
  }

  folly::Timekeeper* timekeeper() {
    return &timekeeper_;
  }

  template <typename Rep, typename Period>
  void advanceClockOnly(std::chrono::duration<Rep, Period> duration) {
    now_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        duration);
  }

  template <typename Rep, typename Period>
  void advance(std::chrono::duration<Rep, Period> duration) {
    advanceClockOnly(duration);
    timekeeper_.advance(duration);
  }

 private:
  std::chrono::steady_clock::time_point now_{};
  folly::ManualTimekeeper timekeeper_;
};

class ThrowingTimekeeper final : public folly::Timekeeper {
 public:
  [[noreturn]] folly::SemiFuture<folly::Unit> after(
      folly::HighResDuration /* duration */) override {
    throw std::runtime_error("timer scheduling failed");
  }
};

class FailOncePacingClock final : public folly::Timekeeper {
 public:
  std::chrono::steady_clock::time_point now() const {
    return now_;
  }

  folly::SemiFuture<folly::Unit> after(
      folly::HighResDuration duration) override {
    if (!firstTimerScheduled_) {
      firstTimerScheduled_ = true;
      return firstTimer_.getSemiFuture();
    }
    return timekeeper_.after(duration);
  }

  void failFirstTimer() {
    firstTimer_.setException(std::runtime_error("timer failed"));
  }

  template <typename Rep, typename Period>
  void advance(std::chrono::duration<Rep, Period> duration) {
    now_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        duration);
    timekeeper_.advance(duration);
  }

 private:
  std::chrono::steady_clock::time_point now_{};
  folly::ManualTimekeeper timekeeper_;
  folly::Promise<folly::Unit> firstTimer_;
  bool firstTimerScheduled_{false};
};

std::shared_ptr<RPCRateLimiter> makePacingLimiter(
    std::string admissionKey,
    ManualPacingClock& clock) {
  return std::make_shared<RPCRateLimiter>(
      std::move(admissionKey),
      [&clock] { return clock.now(); },
      clock.timekeeper(),
      [] { return 1.0; });
}

void enableFractionalLimit(RPCRateLimiter& limiter, double minimumLimit) {
  limiter.configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = minimumLimit,
          .decreaseFactor = 0.5,
      });
}

class RPCRateLimiterTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    common::testutil::TestValue::enable();
  }

  void SetUp() override {
    RPCRateLimiterRegistry::global().testingReset();
  }

  void TearDown() override {
    RPCRateLimiterRegistry::global().testingReset();
  }
};

TEST_F(RPCRateLimiterTest, flowControlPolicyNamesParseCaseInsensitively) {
  const std::vector<std::pair<std::string_view, RPCFlowControlPolicy>> policies{
      {"WINDOW_AIMD", RPCFlowControlPolicy::kWindowAimd},
      {"SMART_PROBING", RPCFlowControlPolicy::kSmartProbing},
  };
  for (const auto& [name, policy] : policies) {
    SCOPED_TRACE(name);
    EXPECT_EQ(parseRPCFlowControlPolicy(name), policy);
    EXPECT_EQ(rpcFlowControlPolicyName(policy), name);
    EXPECT_EQ(
        parseRPCFlowControlPolicy(rpcFlowControlPolicyName(policy)), policy);
  }
  EXPECT_EQ(static_cast<int>(RPCFlowControlPolicy::kWindowAimd), 0);
  EXPECT_EQ(static_cast<int>(RPCFlowControlPolicy::kSmartProbing), 1);
  EXPECT_EQ(
      parseRPCFlowControlPolicy("smart_probing"),
      RPCFlowControlPolicy::kSmartProbing);
  EXPECT_EQ(
      parseRPCFlowControlPolicy("sMaRt_PrObInG"),
      RPCFlowControlPolicy::kSmartProbing);
  EXPECT_EQ(
      parseRPCFlowControlPolicy("window_aimd"),
      RPCFlowControlPolicy::kWindowAimd);
  for (const std::string_view invalid :
       {"", "unknown", "1", " SMART_PROBING", "SMART_PROBING "}) {
    SCOPED_TRACE(invalid);
    const auto expectedMessage =
        "RPC flow control policy must be WINDOW_AIMD or SMART_PROBING; received '" +
        std::string{invalid} + "'";
    VELOX_ASSERT_USER_THROW(
        parseRPCFlowControlPolicy(invalid), expectedMessage);
  }
}

TEST_F(RPCRateLimiterTest, rpcProbingDefaultAndAdaptiveDisabledFallback) {
  ManualPacingClock clock;
  auto omitted = makePacingLimiter("test.probing.omitted", clock);
  auto explicitWindow =
      makePacingLimiter("test.probing.explicit-window", clock);
  const RPCRateLimiter::Config defaults;
  EXPECT_EQ(defaults.flowControlPolicy, RPCFlowControlPolicy::kWindowAimd);
  EXPECT_EQ(
      omitted->stats().flowControlPolicy, RPCFlowControlPolicy::kWindowAimd);
  explicitWindow->configure(defaults);
  EXPECT_EQ(omitted->stats().capacity, explicitWindow->stats().capacity);
  EXPECT_EQ(omitted->stats().limitMilli, explicitWindow->stats().limitMilli);
  EXPECT_EQ(
      omitted->tryAcquireUpTo(20).size(),
      explicitWindow->tryAcquireUpTo(20).size());

  const auto requested = parseRPCFlowControlPolicy("SMART_PROBING");
  RPCRateLimiter::Config fixed{.ceiling = 20, .flowControlPolicy = requested};
  auto& registry = RPCRateLimiterRegistry::global();
  const auto fallback =
      registry.getForPolicy("test.probing.fixed-fallback", fixed);
  EXPECT_EQ(static_cast<int>(requested), 1);
  EXPECT_EQ(static_cast<int>(fallback.limiter->stats().flowControlPolicy), 0);
  EXPECT_EQ(
      fallback.limiter->config().flowControlPolicy,
      RPCFlowControlPolicy::kWindowAimd);
  EXPECT_FALSE(fallback.limiter->config().adaptive);
  EXPECT_FALSE(fallback.limiter->canHandleRetryBackoff());
  fixed.flowControlPolicy = RPCFlowControlPolicy::kWindowAimd;
  EXPECT_EQ(
      registry.getForPolicy("test.probing.fixed-fallback", fixed).limiter,
      fallback.limiter);
}

TEST_F(
    RPCRateLimiterTest,
    smartProbingWindowColdStartChunksFillUnchangedCeiling) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.smart-probing-window.chunks", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 20,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  const auto startedAt = clock.now();
  EXPECT_EQ(limiter->stats().limitMilli, 20'000);
  EXPECT_EQ(limiter->stats().capacity, 20);
  EXPECT_FALSE(limiter->usesAttributedAdmissionControl());
  EXPECT_TRUE(limiter->tryAcquireUpTo(0).empty());
  std::vector<RPCRateLimiter::Token> held;
  for (int i = 0; i < 5; ++i) {
    auto chunk = limiter->tryAcquireUpTo(20);
    ASSERT_EQ(chunk.size(), 4);
    for (auto& token : chunk) {
      held.push_back(std::move(token));
    }
    EXPECT_EQ(limiter->stats().limitMilli, 20'000);
    EXPECT_EQ(limiter->stats().pending, 4 * (i + 1));
    EXPECT_EQ(limiter->stats().nextPermitDelayNanos, 0);
    EXPECT_FALSE(limiter->stats().admissionTimerArmed);
  }
  EXPECT_EQ(clock.now(), startedAt);
  EXPECT_EQ(limiter->available(), 0);
  EXPECT_TRUE(limiter->tryAcquireUpTo(20).empty());
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  EXPECT_FALSE(wait.wait.isReady());
  EXPECT_FALSE(limiter->stats().admissionTimerArmed);
  held.clear();
  EXPECT_TRUE(wait.wait.isReady());
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_EQ(limiter->stats().peakPending, 20);
  EXPECT_EQ(limiter->tryAcquireUpTo(3).size(), 3);
}

TEST_F(RPCRateLimiterTest, retryBackoffStopsAtFloorUnderBothPolicies) {
  for (const auto policy :
       {RPCFlowControlPolicy::kWindowAimd,
        RPCFlowControlPolicy::kSmartProbing}) {
    SCOPED_TRACE(rpcFlowControlPolicyName(policy));
    for (const bool floorCase : {false, true}) {
      SCOPED_TRACE(floorCase);
      ManualPacingClock clock;
      auto limiter = makePacingLimiter("test.floor-retry", clock);
      limiter->configure(
          {.adaptive = true,
           .ceiling = floorCase ? 200 : 20,
           .floor = floorCase ? 50.0 : 1.0,
           .flowControlPolicy = policy});
      if (floorCase) {
        EXPECT_EQ(limiter->stats().limitMilli, 200'000);
        EXPECT_TRUE(limiter->canHandleRetryBackoff());
        limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
        EXPECT_EQ(limiter->stats().limitMilli, 100'000);
        EXPECT_TRUE(limiter->canHandleRetryBackoff());
        limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
        EXPECT_EQ(limiter->stats().limitMilli, 50'000);
        EXPECT_FALSE(limiter->canHandleRetryBackoff());
      }
      const int64_t units = floorCase ? 50 : 20;
      const auto collect = [&] {
        std::vector<RPCRateLimiter::Token> held;
        held.reserve(static_cast<size_t>(units));
        while (held.size() < static_cast<size_t>(units)) {
          auto grant = limiter->tryAcquireUpTo(
              units - static_cast<int64_t>(held.size()));
          EXPECT_FALSE(grant.empty());
          if (grant.empty()) {
            break;
          }
          for (auto& token : grant) {
            held.push_back(std::move(token));
          }
        }
        return held;
      };
      auto first = collect();
      ASSERT_EQ(first.size(), units);
      if (!floorCase) {
        clock.advance(20ms);
      }
      const auto completed =
          limiter->makeLease(std::move(first))
              ->completeAndRelease(
                  floorCase ? RPCRateLimiter::Outcome::kOverload
                            : RPCRateLimiter::Outcome::kSuccess,
                  floorCase);
      EXPECT_EQ(limiter->stats().pending, 0);
      if (floorCase) {
        EXPECT_EQ(
            completed.disposition,
            RPCRateLimiter::FeedbackDisposition::kCanonical);
        EXPECT_TRUE(completed.sharedOverloadHandled);
        EXPECT_EQ(limiter->stats().limitMilli, 50'000);
        EXPECT_FALSE(limiter->canHandleRetryBackoff());
      }
      auto second = collect();
      ASSERT_EQ(second.size(), units);
      EXPECT_EQ(limiter->stats().pending, units);
    }
  }
}

TEST_F(
    RPCRateLimiterTest,
    smartProbingWindowColdStartCutUsesChargedPopulation) {
  for (const auto policy :
       {RPCFlowControlPolicy::kWindowAimd,
        RPCFlowControlPolicy::kSmartProbing}) {
    SCOPED_TRACE(rpcFlowControlPolicyName(policy));
    ManualPacingClock clock;
    auto limiter =
        makePacingLimiter("test.smart-probing-window.charged-cut", clock);
    limiter->configure(
        {.adaptive = true,
         .ceiling = 32,
         .floor = 1,
         .flowControlPolicy = policy});
    auto held = limiter->tryAcquireUpTo(4);
    ASSERT_EQ(held.size(), 4);
    EXPECT_EQ(limiter->stats().limitMilli, 32'000);
    auto lease = limiter->makeLease(std::move(held));
    EXPECT_TRUE(
        lease->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
            .sharedOverloadHandled);
    EXPECT_EQ(
        limiter->stats().limitMilli,
        policy == RPCFlowControlPolicy::kSmartProbing ? 2'000 : 16'000);
    EXPECT_EQ(limiter->stats().pending, 0);
    EXPECT_FALSE(limiter->stats().admissionTimerArmed);
  }
}

// A quota limits the rate, not the window: its rejection gets WINDOW's cut of
// the limit rather than of the charged startup peak, and does not start
// probing.
TEST_F(RPCRateLimiterTest, smartProbingQuotaRejectionCutsAsWindow) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.smart-probing.quota-cut", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 32,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  auto held = limiter->tryAcquireUpTo(4);
  ASSERT_EQ(held.size(), 4);
  auto lease = limiter->makeLease(std::move(held));
  lease->markNonCapacityRejection();
  EXPECT_TRUE(
      lease->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 16'000);
  EXPECT_FALSE(limiter->testingProbingSnapshot().cut);
  EXPECT_FALSE(limiter->usesAttributedAdmissionControl());
}

TEST_F(
    RPCRateLimiterTest,
    smartProbingWindowColdStartPeakSurvivesReleaseBeforeNegative) {
  ManualPacingClock clock;
  auto limiter =
      makePacingLimiter("test.smart-probing-window.release-before-cut", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 32,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  auto held = limiter->tryAcquireUpTo(4);
  ASSERT_EQ(held.size(), 4);
  auto rejected = limiter->makeLease(std::move(held.back()));
  held.pop_back();
  clock.advance(20ms);
  limiter->makeLease(std::move(held))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_EQ(limiter->stats().pending, 1);
  EXPECT_EQ(limiter->stats().peakPending, 4);
  EXPECT_EQ(limiter->stats().serviceHorizonNanos, 1'000'000'000);
  EXPECT_TRUE(
      rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 2'000);
  EXPECT_EQ(limiter->stats().pending, 0);
}

TEST_F(
    RPCRateLimiterTest,
    smartProbingWindowColdStartCutRetainsPeakAcrossChunks) {
  ManualPacingClock clock;
  auto limiter =
      makePacingLimiter("test.smart-probing-window.multi-chunk-cut", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 32,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  std::vector<RPCRateLimiter::Token> held;
  for (int i = 0; i < 3; ++i) {
    auto chunk = limiter->tryAcquireUpTo(32);
    ASSERT_EQ(chunk.size(), 4);
    for (auto& token : chunk) {
      held.push_back(std::move(token));
    }
  }
  EXPECT_EQ(limiter->stats().pending, 12);
  auto rejected = limiter->makeLease(std::move(held.back()));
  held.clear();
  EXPECT_EQ(limiter->stats().pending, 1);
  EXPECT_EQ(limiter->stats().peakPending, 12);
  EXPECT_TRUE(
      rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 6'000);
  EXPECT_EQ(limiter->stats().pending, 0);
}

TEST_F(
    RPCRateLimiterTest,
    smartProbingWindowSingleUnitLegacyIdentityAndCancellation) {
  ManualPacingClock clock;
  auto limiter =
      makePacingLimiter("test.smart-probing-window.physical-unit", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 4,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  auto held = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(held.size(), 1);
  EXPECT_EQ(held.front().admissionId(), 0);
  EXPECT_TRUE(held.front().belongsTo(limiter.get()));
  const auto epoch = held.front().overloadEpoch();
  RPCRateLimiter::Token moved(std::move(held.front()));
  EXPECT_FALSE(held.front().belongsTo(limiter.get()));
  RPCRateLimiter::Token assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.admissionId(), 0);
  EXPECT_TRUE(assigned.belongsTo(limiter.get()));
  held.clear();
  EXPECT_EQ(limiter->stats().pending, 1);
  auto cancelled = limiter->makeLease(std::move(assigned));
  cancelled->release();
  cancelled->release();
  cancelled->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 4'000);
  auto fresh = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(fresh.size(), 1);
  EXPECT_EQ(fresh.front().overloadEpoch(), epoch);
  EXPECT_EQ(fresh.front().admissionId(), 0);
  auto rejected = limiter->makeLease(std::move(fresh));
  EXPECT_TRUE(
      rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  auto next = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(next.size(), 1);
  EXPECT_EQ(next.front().overloadEpoch(), epoch + 1);
  EXPECT_NE(next.front().admissionId(), 0);
  cancelled->completeAndRelease(
      RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_EQ(limiter->stats().pending, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  next.clear();
  EXPECT_EQ(limiter->stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, retryBackoffOwnershipTracksCurrentWindow) {
  auto window = std::make_shared<RPCRateLimiter>("test.retry-backoff.window");
  window->configure({.adaptive = true, .ceiling = 200, .floor = 50});
  EXPECT_TRUE(window->canHandleRetryBackoff());
  window->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_TRUE(window->canHandleRetryBackoff());
  window->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_FALSE(window->canHandleRetryBackoff());

  auto precise = std::make_shared<RPCRateLimiter>("test.retry-backoff.precise");
  precise->configure(
      {.adaptive = true,
       .ceiling = 200,
       .floor = 50.0001,
       .decreaseFactor = 0.500001});
  precise->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  precise->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(precise->stats().limitMilli, 50'000);
  EXPECT_TRUE(precise->canHandleRetryBackoff());
  precise->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_FALSE(precise->canHandleRetryBackoff());

  auto paced = std::make_shared<RPCRateLimiter>("test.retry-backoff.paced");
  paced->configure({.adaptive = true, .ceiling = 1, .floor = 0.01});
  for (int32_t cut = 0; cut < 20; ++cut) {
    paced->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  EXPECT_TRUE(paced->canHandleRetryBackoff());

  for (const auto policy :
       {RPCFlowControlPolicy::kWindowAimd,
        RPCFlowControlPolicy::kSmartProbing}) {
    for (const bool adaptive : {false, true}) {
      auto fixed = std::make_shared<RPCRateLimiter>("test.retry-backoff.fixed");
      fixed->configure(
          {.adaptive = adaptive,
           .ceiling = 1,
           .floor = 1,
           .flowControlPolicy = policy});
      EXPECT_FALSE(fixed->canHandleRetryBackoff());
      fixed->amend([](auto& config) { config.floor = 0.01; });
      EXPECT_EQ(fixed->canHandleRetryBackoff(), adaptive);
    }
  }
}

TEST_F(RPCRateLimiterTest, serviceHorizonEwmaHandlesPositiveInt64Extremes) {
  constexpr int64_t kMaximum = std::numeric_limits<int64_t>::max();
  const std::vector<std::tuple<int64_t, int64_t, int64_t>> samples{
      {kMaximum, kMaximum, kMaximum},
      {kMaximum, 1, 8'070'450'532'247'928'831},
      {1, kMaximum, 1'152'921'504'606'846'976},
      {kMaximum, kMaximum - 1, kMaximum - 1},
      {8, 1, 7},
      {9, 2, 8},
      {8, 9, 8},
      {1, 8, 1},
      {1, 9, 2}};
  for (const auto& [initial, next, expected] : samples) {
    SCOPED_TRACE(initial);
    SCOPED_TRACE(next);
    auto limiter = std::make_shared<RPCRateLimiter>("test.horizon.int64");
    limiter->configure({.adaptive = false, .ceiling = 2});
    limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, initial);
    ASSERT_EQ(limiter->stats().serviceHorizonNanos, initial);
    limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, next);
    EXPECT_EQ(limiter->stats().serviceHorizonNanos, expected);
    limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, 0);
    limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, -1);
    EXPECT_EQ(limiter->stats().serviceHorizonNanos, expected);
  }
}

TEST_F(RPCRateLimiterTest, fixedCapacityPreservesLargeIntegerCeiling) {
  auto limiter = std::make_shared<RPCRateLimiter>("test.integer-ceiling");
  limiter->configure(
      {.adaptive = false, .ceiling = std::numeric_limits<int64_t>::max()});
  EXPECT_EQ(limiter->available(), std::numeric_limits<int64_t>::max());
  auto tokens = limiter->tryAcquireUpTo(2);
  EXPECT_EQ(tokens.size(), 2);
  EXPECT_EQ(limiter->available(), std::numeric_limits<int64_t>::max() - 2);
  const auto stats = limiter->stats();
  EXPECT_EQ(stats.capacity, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(stats.hardLimit, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(stats.lowWaterCapacity, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(stats.limitMilli, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(stats.lowWaterLimitMilli, std::numeric_limits<int64_t>::max());
}

TEST_F(RPCRateLimiterTest, registryKeepsSeparateLimiterPerPolicy) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config smartProbing{
      .adaptive = true,
      .ceiling = 100,
      .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing};
  auto* first =
      registry.getForPolicy("test.policy-limiters", smartProbing).limiter;
  auto windowAimd = smartProbing;
  windowAimd.flowControlPolicy = RPCFlowControlPolicy::kWindowAimd;
  auto* second =
      registry.getForPolicy("test.policy-limiters", windowAimd).limiter;
  EXPECT_NE(first, second);
  EXPECT_EQ(
      registry.getForPolicy("test.policy-limiters", smartProbing).limiter,
      first);
  EXPECT_EQ(
      first->stats().flowControlPolicy, RPCFlowControlPolicy::kSmartProbing);
  EXPECT_EQ(
      second->stats().flowControlPolicy, RPCFlowControlPolicy::kWindowAimd);
}

TEST_F(RPCRateLimiterTest, acquireAndRelease) {
  const std::string backend = "test.backend";
  EXPECT_EQ(limiterFor(backend).stats().pending, 0);

  {
    auto token = limiterFor(backend).acquire();
    EXPECT_EQ(limiterFor(backend).stats().pending, 1);
  }
  // Token destroyed — count should be back to 0.
  EXPECT_EQ(limiterFor(backend).stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, tokenReleaseSurvivesSynchronizationFailure) {
  auto& limiter = limiterFor("test.backend");

  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::rpc::RPCRateLimiter::release",
      std::function<void(RPCRateLimiter*)>([](RPCRateLimiter* limiter) {
        if (limiter != nullptr) {
          throw std::system_error(std::make_error_code(std::errc::owner_dead));
        }
      }));

  {
    auto token = limiter.acquire();
    EXPECT_EQ(limiter.stats().pending, 1);
  }
  EXPECT_EQ(limiter.stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, multipleAcquires) {
  const std::string backend = "test.backend";

  auto token1 = limiterFor(backend).acquire();
  auto token2 = limiterFor(backend).acquire();
  auto token3 = limiterFor(backend).acquire();
  EXPECT_EQ(limiterFor(backend).stats().pending, 3);
}

TEST_F(RPCRateLimiterTest, admissionLeaseCompletesExactlyOnce) {
  auto limiter = std::make_shared<RPCRateLimiter>("test.lease");
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 4,
          .floor = 1,
          .decreaseFactor = 0.5,
      });
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  auto lease = limiter->makeLease(std::move(tokens));

  EXPECT_TRUE(lease->completeAndRelease(true).sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().capacity, 2);
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_FALSE(lease->completeAndRelease(true).sharedOverloadHandled);
  lease->release();
  EXPECT_EQ(limiter->stats().capacity, 2);
  EXPECT_EQ(limiter->stats().pending, 0);
}

TEST_F(
    RPCRateLimiterTest,
    canonicalOverloadCoalescesWithinEpochAndShrinksNextEpoch) {
  auto limiter = std::make_shared<RPCRateLimiter>("test.overload-epochs");
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 1'000'000,
          .floor = 0.01,
          .decreaseFactor = 0.5,
      });

  auto tokens = limiter->tryAcquireUpTo(2);
  ASSERT_EQ(tokens.size(), 2);
  auto firstLease = limiter->makeLease(std::move(tokens.at(0)));
  auto secondLease = limiter->makeLease(std::move(tokens.at(1)));

  const auto firstCompletion = firstLease->completeAndRelease(true);
  EXPECT_TRUE(firstCompletion.sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 100'000);

  const auto secondCompletion = secondLease->completeAndRelease(true);
  EXPECT_TRUE(secondCompletion.sharedOverloadHandled);
  EXPECT_EQ(secondCompletion.overloadEpoch, firstCompletion.overloadEpoch);
  EXPECT_EQ(limiter->stats().limitMilli, 100'000);

  auto nextTokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(nextTokens.size(), 1);
  auto nextLease = limiter->makeLease(std::move(nextTokens.at(0)));
  const auto nextCompletion = nextLease->completeAndRelease(true);
  EXPECT_TRUE(nextCompletion.sharedOverloadHandled);
  EXPECT_NE(nextCompletion.overloadEpoch, firstCompletion.overloadEpoch);
  EXPECT_EQ(limiter->stats().limitMilli, 50'000);
  EXPECT_EQ(limiter->stats().lowWaterLimitMilli, 50'000);
}

TEST_F(RPCRateLimiterTest, dueWakeupWindowPreservesOutputLearning) {
  ManualPacingClock clock;
  auto window = makePacingLimiter("test.probing.window-reference", clock);
  auto probing = makePacingLimiter("test.probing.healthy", clock);
  window->configure({.adaptive = true, .ceiling = 400, .floor = 1});
  auto config = window->config();
  probing->configure(config);
  auto windowTokens = window->tryAcquireUpTo(400);
  auto probingTokens = probing->tryAcquireUpTo(400);
  ASSERT_EQ(windowTokens.size(), 200);
  ASSERT_EQ(probingTokens.size(), windowTokens.size());
  EXPECT_TRUE(probing->tryAcquireUpTo(1).empty());
  const auto windowEpoch = windowTokens.front().overloadEpoch();
  const auto probingEpoch = probingTokens.front().overloadEpoch();
  clock.advance(20ms);
  window->makeLease(std::move(windowTokens.front()))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  probing->makeLease(std::move(probingTokens.front()))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_EQ(probing->stats().serviceHorizonNanos, 1'000'000'000);
  EXPECT_EQ(probing->stats().limitMilli, window->stats().limitMilli);
  EXPECT_EQ(probing->tryAcquireUpTo(400).size(), 1);
  EXPECT_TRUE(window->onSuccessForEpoch(windowEpoch, 1, 20'000'000));
  EXPECT_TRUE(probing->onSuccessForEpoch(probingEpoch, 1, 20'000'000));
  EXPECT_EQ(
      probing->stats().serviceHorizonNanos,
      window->stats().serviceHorizonNanos);
  EXPECT_EQ(probing->stats().limitMilli, window->stats().limitMilli);
  window->makeLease(std::move(windowTokens.back()))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  probing->makeLease(std::move(probingTokens.back()))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_EQ(probing->stats().serviceHorizonNanos, 20'000'000);
  window->onSuccessForEpoch(windowEpoch, 1, 40'000'000);
  probing->onSuccessForEpoch(probingEpoch, 1, 40'000'000);
  EXPECT_EQ(probing->stats().serviceHorizonNanos, 22'500'000);
  EXPECT_EQ(probing->stats().limitMilli, window->stats().limitMilli);
}

TEST_F(RPCRateLimiterTest, windowCancellationAndUnsampledWorkDoNotProveGrowth) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.window.no-proof", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 20,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  for (const auto outcome :
       {RPCRateLimiter::Outcome::kNone,
        RPCRateLimiter::Outcome::kFailure,
        RPCRateLimiter::Outcome::kSuccess}) {
    auto tokens = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(tokens.size(), 1);
    auto lease = limiter->makeLease(std::move(tokens));
    lease->completeAndRelease(outcome, /*handedOff=*/false);
    lease->release();
    lease->completeAndRelease(
        RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
    EXPECT_EQ(limiter->stats().pending, 0);
  }
  auto compound = limiter->tryAcquireUpTo(2);
  auto second = limiter->tryAcquireUpTo(2);
  ASSERT_EQ(compound.size(), 1);
  ASSERT_EQ(second.size(), 1);
  compound.push_back(std::move(second.front()));
  limiter->makeLease(std::move(compound))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
}

TEST_F(RPCRateLimiterTest, windowResetKeepsOldPhysicalOwnershipAndUniqueIds) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.window.reset", clock);
  const RPCRateLimiter::Config config{
      .adaptive = true,
      .ceiling = 4,
      .floor = 1,
      .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing};
  limiter->configure(config);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto old = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(old.size(), 1);
  const auto oldId = old.front().admissionId();
  EXPECT_NE(oldId, 0);
  auto oldLease = limiter->makeLease(std::move(old));
  limiter->testingReset();
  limiter->configure(config);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto fresh = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(fresh.size(), 1);
  EXPECT_GT(fresh.front().admissionId(), oldId);
  EXPECT_EQ(limiter->stats().pending, 2);
  RPCRateLimiter::Token moved(std::move(fresh.front()));
  RPCRateLimiter::Token assigned;
  assigned = std::move(moved);
  fresh.clear();
  oldLease->completeAndRelease(
      RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_EQ(limiter->stats().pending, 1);
  auto freshLease = limiter->makeLease(std::move(assigned));
  freshLease->release();
  freshLease->release();
  EXPECT_EQ(limiter->stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, windowFeedbackPreservesFractionalProgress) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.window.active-fractional", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 2,
       .floor = 0.25,
       .flowControlPolicy = RPCFlowControlPolicy::kWindowAimd});
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, 100'000'000);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 500);
  clock.advance(200ms);
  auto fresh = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(fresh.size(), 1);
  const auto epoch = fresh.front().overloadEpoch();
  clock.advance(100ms);
  limiter->makeLease(std::move(fresh))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  ASSERT_EQ(limiter->stats().serviceHorizonNanos, 100'000'000);
  EXPECT_FALSE(limiter->stats().recovering);
  EXPECT_TRUE(limiter->onSuccessForEpoch(epoch, 1, 100'000'000));
  EXPECT_TRUE(limiter->stats().recovering);
  EXPECT_TRUE(limiter->tryAcquireUpTo(2).empty());
  clock.advance(100ms);
  auto progressed = limiter->tryAcquireUpTo(2);
  ASSERT_EQ(progressed.size(), 1);
  EXPECT_GE(limiter->stats().limitMilli, 1'000);
  EXPECT_TRUE(limiter->tryAcquireUpTo(2).empty());
}

TEST_F(RPCRateLimiterTest, windowAcceptedCutKeepsAdvancingClockMonotonic) {
  ManualPacingClock clock;
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.probing.advancing-clock",
      [&] {
        clock.advanceClockOnly(1ns);
        return clock.now();
      },
      clock.timekeeper(),
      [] { return 1.0; });
  limiter->configure(
      {.adaptive = true,
       .ceiling = 4,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kWindowAimd});
  auto cut = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(cut.size(), 1);
  const auto completion =
      limiter->makeLease(std::move(cut))
          ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
  EXPECT_TRUE(completion.sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 2'000);
  auto fresh = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(fresh.size(), 1);
  const auto epoch = fresh.front().overloadEpoch();
  clock.advanceClockOnly(20ms);
  limiter->makeLease(std::move(fresh))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_TRUE(limiter->onSuccessForEpoch(epoch, 1, 20'000'000));
  EXPECT_EQ(limiter->stats().serviceHorizonNanos, 20'000'000);
  EXPECT_EQ(limiter->stats().limitMilli, std::llround(std::sqrt(6.0) * 1'000));
  EXPECT_EQ(limiter->stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, windowDeadlineCrossingCannotStrandRunnableWaiter) {
  ManualPacingClock clock;
  bool crossDeadline{false};
  int clockReads{0};
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.window.deadline-crossing",
      [&] {
        if (crossDeadline && ++clockReads == 2) {
          crossDeadline = false;
          clock.advanceClockOnly(1ns);
        }
        return clock.now();
      },
      clock.timekeeper(),
      [] { return 1.0; });
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1,
       .floor = 0.01,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  auto demand = limiter->registerRunnableDemand();
  auto cut = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(cut.size(), 1);
  limiter->makeLease(std::move(cut))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
  clock.advance(
      std::chrono::ceil<std::chrono::milliseconds>(
          std::chrono::nanoseconds{limiter->stats().nextPermitDelayNanos}));
  auto fresh = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(fresh.size(), 1);
  clock.advance(20ms);
  limiter->makeLease(std::move(fresh))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  clock.advance(
      std::chrono::ceil<std::chrono::milliseconds>(
          std::chrono::nanoseconds{limiter->stats().nextPermitDelayNanos}));
  auto paced = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(paced.size(), 1);
  paced.clear();
  const auto delay = limiter->stats().nextPermitDelayNanos;
  ASSERT_GT(delay, 0);
  clock.advanceClockOnly(std::chrono::nanoseconds{delay - 1});
  crossDeadline = true;
  const auto wait = limiter->admitOrWait();
  EXPECT_FALSE(crossDeadline);
  clock.advance(1ms);
  EXPECT_TRUE(wait.admitted || wait.wait.isReady());
  EXPECT_EQ(limiter->stats().pending, 0);
}

// Bounds physical ownership independently of output feedback.
std::shared_ptr<RPCRateLimiter> makeSingleWindowLimiter(
    ManualPacingClock& clock) {
  auto limiter = makePacingLimiter("test.smart-probing-owned.moves", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1,
       .floor = 1,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  return limiter;
}

TEST_F(RPCRateLimiterTest, windowMovesAndDuplicateCancellationRetainOwnership) {
  ManualPacingClock clock;
  auto limiter = makeSingleWindowLimiter(clock);
  auto tokens = limiter->tryAcquireUpTo(32);
  ASSERT_EQ(tokens.size(), 1);
  RPCRateLimiter::Token moved(std::move(tokens.front()));
  RPCRateLimiter::Token assigned;
  assigned = std::move(moved);
  tokens.clear();
  EXPECT_EQ(limiter->stats().pending, 1);
  EXPECT_TRUE(limiter->tryAcquireUpTo(32).empty());
  auto lease = limiter->makeLease(std::move(assigned));
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  lease->release();
  lease->release();
  lease->completeAndRelease(
      RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  EXPECT_TRUE(wait.wait.isReady());
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_EQ(limiter->tryAcquireUpTo(32).size(), 1);
}

TEST_F(RPCRateLimiterTest, windowOwnedPolicyDoesNotPaceOrdinaryRetryPolicy) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config policy{.adaptive = true, .ceiling = 100, .floor = 1};
  auto* ordinary =
      registry.getForPolicy("test.rate-mode-isolation", policy).limiter;
  policy.flowControlPolicy = RPCFlowControlPolicy::kSmartProbing;
  auto* controlled =
      registry.getForPolicy("test.rate-mode-isolation", policy).limiter;
  ASSERT_NE(ordinary, controlled);
  auto rejected = controlled->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  controlled->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
  EXPECT_EQ(controlled->stats().limitMilli, 1'000);
  EXPECT_EQ(ordinary->tryAcquireUpTo(10).size(), 10);
}

TEST_F(
    RPCRateLimiterTest,
    fixedConcurrencyDoesNotMergeWindowOwnedAndOrdinaryPolicies) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config policy{.adaptive = true, .ceiling = 1, .floor = 1};
  auto* ordinary =
      registry.getForPolicy("test.fixed-rate-policy", policy).limiter;
  policy.flowControlPolicy = RPCFlowControlPolicy::kSmartProbing;
  auto* controlled =
      registry.getForPolicy("test.fixed-rate-policy", policy).limiter;
  ASSERT_NE(ordinary, controlled);
  auto rejected = controlled->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  controlled->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
  EXPECT_EQ(controlled->stats().capacity, 1);
  EXPECT_EQ(controlled->stats().limitMilli, 1'000);
  EXPECT_EQ(ordinary->tryAcquireUpTo(1).size(), 1);
}

TEST_F(
    RPCRateLimiterTest,
    fractionalAdmissionCrossingPermitDeadlineStillGetsWakeup) {
  ManualPacingClock clock;
  bool crossDeadline{false};
  int clockReads{0};
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.fractional-deadline-crossing",
      [&] {
        if (crossDeadline && ++clockReads == 2) {
          crossDeadline = false;
          clock.advanceClockOnly(2s);
        }
        return clock.now();
      },
      clock.timekeeper(),
      [] { return 1.0; });
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1,
       .floor = 0.01,
       .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 500);
  clock.advance(1s);
  crossDeadline = true;
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  EXPECT_TRUE(limiter->stats().admissionTimerArmed);
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, windowAimdPolicyRetainsOriginalDeadlineRace) {
  ManualPacingClock clock;
  bool crossDeadline{false};
  int clockReads{0};
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.legacy-deadline-rollback",
      [&] {
        if (crossDeadline && ++clockReads == 2) {
          crossDeadline = false;
          clock.advanceClockOnly(2s);
        }
        return clock.now();
      },
      clock.timekeeper(),
      [] { return 1.0; });
  limiter->configure({.adaptive = true, .ceiling = 1, .floor = 0.01});
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  clock.advance(1s);
  crossDeadline = true;
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  EXPECT_FALSE(limiter->stats().admissionTimerArmed);
  clock.advance(1ms);
  EXPECT_FALSE(wait.wait.isReady());
  // Release the deliberately preserved old wait without leaving a pending
  // promise at fixture teardown.
  limiter->amend([](auto& config) { config.adaptive = false; });
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, configAmendDoesNotSuppressCanonicalOverload) {
  auto limiter = std::make_shared<RPCRateLimiter>("test.amended-overload");
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 8,
          .floor = 1,
          .decreaseFactor = 0.5,
      });

  auto tokens = limiter->tryAcquireUpTo(2);
  ASSERT_EQ(tokens.size(), 2);
  auto firstLease = limiter->makeLease(std::move(tokens.at(0)));
  auto secondLease = limiter->makeLease(std::move(tokens.at(1)));

  limiter->amend([](RPCRateLimiter::Config& config) { config.ceiling = 16; });
  EXPECT_EQ(limiter->stats().limitMilli, 16'000);

  EXPECT_TRUE(firstLease->completeAndRelease(true).sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 8'000);

  EXPECT_TRUE(secondLease->completeAndRelease(true).sharedOverloadHandled);
  EXPECT_EQ(limiter->stats().limitMilli, 8'000);
}

// admitOrWait() answers "can you take work now, and if not, what do I wait on?"
// in one call. Deciding and enrolling under the same lock is what makes the
// answer usable: asked separately, a slot freeing between the two steps leaves
// the caller neither admitted nor enrolled, and falling through to some other
// wait is how a driver ends up parked on a future nothing can fulfil.
// A backend's configuration is fixed whole, by one query. This pins the
// invariant rather than the defect: the defect was two separate writes per
// query -- the function's ceiling, then the operator's session properties --
// with a window between them where a second query saw the backend still
// unconfigured and overwrote, fixing one query's ceiling with another's floor.
// A seal that checked and wrote under separate locks would let two callers
// both seal. One race rarely shows either defect, so the test runs many.
TEST_F(RPCRateLimiterTest, concurrentInitializersDoNotMixConfigurations) {
  constexpr int kThreads = 16;
  constexpr int kRounds = 50;

  // Each thread offers an internally consistent policy; the pairs are chosen
  // so a mixture is detectable.
  struct Policy {
    int64_t ceiling;
    double floor;
  };
  auto policyFor = [](int i) -> Policy {
    return {static_cast<int64_t>(100 * (i + 1)), static_cast<double>(i + 1)};
  };

  for (int round = 0; round < kRounds; ++round) {
    SCOPED_TRACE(round);
    auto limiter = std::make_shared<RPCRateLimiter>("test.backend");
    std::atomic<bool> go{false};
    std::atomic<int> numSealed{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
      threads.emplace_back([&, i]() {
        const auto policy = policyFor(i);
        while (!go.load(std::memory_order_relaxed)) {
        }
        if (limiter->initializeOrMatch(
                {.ceiling = policy.ceiling, .floor = policy.floor})) {
          ++numSealed;
        }
      });
    }
    go.store(true);
    for (auto& thread : threads) {
      thread.join();
    }
    // Every policy has its own ceiling, so only the thread that sealed
    // matches.
    EXPECT_EQ(numSealed.load(), 1);

    // Whatever landed must be exactly one thread's pair, not a blend.
    const auto config = limiter->config();
    bool matchesSomeThread = false;
    for (int i = 0; i < kThreads; ++i) {
      const auto policy = policyFor(i);
      if (config.ceiling == policy.ceiling && config.floor == policy.floor) {
        matchesSomeThread = true;
        break;
      }
    }
    EXPECT_TRUE(matchesSomeThread)
        << "configuration is a mixture: ceiling " << config.ceiling
        << " with floor " << config.floor;
  }
}

// initializeOrMatch() seals the first policy it sees and tells each later
// caller whether the sealed policy is the one it asked for.
TEST_F(RPCRateLimiterTest, initializeOrMatchReportsWhetherSealedPolicyMatches) {
  auto& limiter = limiterFor("test.backend");
  const RPCRateLimiter::Config small{.ceiling = 25};
  const RPCRateLimiter::Config large{.ceiling = 100};

  EXPECT_TRUE(limiter.initializeOrMatch(small));
  EXPECT_TRUE(limiter.initializeOrMatch(small));
  EXPECT_FALSE(limiter.initializeOrMatch(large));
  EXPECT_EQ(limiter.config(), small);
}

// A later policy matches the sealed one when a limiter running it would admit
// alike, and only then.
TEST_F(RPCRateLimiterTest, initializeOrMatchMatchesPoliciesThatAdmitAlike) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  auto sealedWith = [](const std::string& admissionKey,
                       const RPCRateLimiter::Config& policy) -> auto& {
    auto& limiter = limiterFor(admissionKey);
    EXPECT_TRUE(limiter.initializeOrMatch(policy));
    return limiter;
  };

  {
    // Out-of-range values compare as the values they clamp to.
    auto& limiter = sealedWith(
        "clamped",
        {.adaptive = true,
         .ceiling = 25,
         .floor = 0.001,
         .decreaseFactor = 0.99});
    EXPECT_TRUE(limiter.initializeOrMatch(
        {.adaptive = true, .ceiling = 25, .floor = 0, .decreaseFactor = 5}));
    EXPECT_FALSE(limiter.initializeOrMatch(
        {.adaptive = true,
         .ceiling = 25,
         .floor = 0.002,
         .decreaseFactor = 0.99}));
  }
  {
    // NaN compares as the default.
    auto& limiter = sealedWith("nan", {.adaptive = true, .ceiling = 25});
    EXPECT_TRUE(limiter.initializeOrMatch(
        {.adaptive = true,
         .ceiling = 25,
         .floor = kNaN,
         .decreaseFactor = kNaN}));
  }
  {
    // Fixed capacity reads neither the floor nor the decrease factor.
    auto& limiter = sealedWith(
        "fixed", {.ceiling = 25, .floor = 4, .decreaseFactor = 0.25});
    EXPECT_TRUE(limiter.initializeOrMatch({.ceiling = 25}));
    EXPECT_FALSE(limiter.initializeOrMatch({.ceiling = 26}));
    EXPECT_FALSE(limiter.initializeOrMatch(
        {.adaptive = true, .ceiling = 25, .floor = 4}));
  }
  {
    // A floor at or above a positive ceiling pins capacity at the ceiling.
    auto& limiter =
        sealedWith("pinned", {.adaptive = true, .ceiling = 25, .floor = 50});
    EXPECT_TRUE(limiter.initializeOrMatch(
        {.adaptive = true, .ceiling = 25, .floor = 25, .decreaseFactor = 0.9}));
    EXPECT_TRUE(limiter.initializeOrMatch({.ceiling = 25}));
    EXPECT_FALSE(limiter.initializeOrMatch(
        {.adaptive = true, .ceiling = 25, .floor = 24}));
  }
  {
    // A zero ceiling defers to the process default, which can change, so a
    // large floor does not pin it.
    auto& limiter =
        sealedWith("deferred", {.adaptive = true, .ceiling = 0, .floor = 50});
    EXPECT_FALSE(limiter.initializeOrMatch({.ceiling = 0}));
  }
}

// initializeOrMatch() seals the clamped policy, so the limiter never reads an
// out-of-range value: a factor above the range shrinks the window by 0.99, and
// a NaN factor by the default.
TEST_F(RPCRateLimiterTest, sealedPolicyIsClampedBeforeTheLimiterReadsIt) {
  auto& aboveRange = limiterFor("above.range");
  ASSERT_TRUE(aboveRange.initializeOrMatch(
      {.adaptive = true, .ceiling = 100, .floor = 0, .decreaseFactor = 5}));
  EXPECT_DOUBLE_EQ(aboveRange.config().floor, 0.001);
  EXPECT_DOUBLE_EQ(aboveRange.config().decreaseFactor, 0.99);
  aboveRange.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(aboveRange.stats().capacity, 99);

  auto& nan = limiterFor("nan.factor");
  ASSERT_TRUE(nan.initializeOrMatch(
      {.adaptive = true,
       .ceiling = 16,
       .floor = 1,
       .decreaseFactor = std::numeric_limits<double>::quiet_NaN()}));
  nan.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(nan.stats().capacity, 8);
}

// A zero ceiling defers to the process default on every read. A sealed zero
// ceiling follows a later change to the default, and it stays distinct from a
// policy that names today's default.
TEST_F(RPCRateLimiterTest, zeroCeilingStaysDeferredToProcessDefault) {
  auto& limiter = limiterFor("test.backend");
  ASSERT_TRUE(limiter.initializeOrMatch({.ceiling = 0}));
  EXPECT_FALSE(limiter.initializeOrMatch(
      {.ceiling = RPCRateLimiter::defaultCapacity()}));

  RPCRateLimiter::setDefaultCapacity(5);
  EXPECT_EQ(limiter.stats().hardLimit, 5);
  EXPECT_EQ(limiter.stats().capacity, 5);
  EXPECT_TRUE(limiter.initializeOrMatch({.ceiling = 0}));
}

// Policies that initializeOrMatch() treats as equal do admit alike: limiters
// sealed with either follow the same capacity through overload and recovery.
TEST_F(RPCRateLimiterTest, policiesThatMatchAdmitAlike) {
  auto trajectory = [](const RPCRateLimiter::Config& policy) {
    // A frozen clock: a recovering paced window grows with elapsed time, which
    // would make the trajectory depend on scheduling.
    auto limiter = std::make_shared<RPCRateLimiter>(
        "test.backend",
        [] { return std::chrono::steady_clock::time_point{}; },
        nullptr,
        [] { return 1.0; });
    EXPECT_TRUE(limiter->initializeOrMatch(policy));
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>> steps;
    auto record = [&] {
      const auto stats = limiter->stats();
      steps.emplace_back(
          stats.capacity,
          stats.hardLimit,
          stats.lowWaterCapacity,
          stats.limitMilli,
          stats.lowWaterLimitMilli);
    };
    record();
    for (int i = 0; i < 3; ++i) {
      limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
      record();
    }
    for (int i = 0; i < 3; ++i) {
      limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 10, 1'000'000);
      record();
    }
    return steps;
  };

  const auto fixed = trajectory({.ceiling = 25});
  EXPECT_EQ(
      trajectory({.ceiling = 25, .floor = 4, .decreaseFactor = 0.25}), fixed);
  EXPECT_EQ(trajectory({.adaptive = true, .ceiling = 25, .floor = 50}), fixed);
  EXPECT_EQ(trajectory({.adaptive = true, .ceiling = 25, .floor = 25}), fixed);
  // Above the initial adaptive window a pinned limiter holds its window at the
  // ceiling, where fixed capacity holds none, and still admits alike.
  EXPECT_EQ(
      trajectory({.adaptive = true, .ceiling = 300, .floor = 300}),
      trajectory({.ceiling = 300}));
  // A policy that does not match moves differently, so the comparisons above
  // can tell policies apart.
  EXPECT_NE(trajectory({.adaptive = true, .ceiling = 25, .floor = 1}), fixed);
  EXPECT_NE(
      trajectory({.adaptive = true, .ceiling = 300, .floor = 1}),
      trajectory({.ceiling = 300}));
}

// Over a grid of policies, every pair that initializeOrMatch() matches follows
// the same capacity trajectory through overload and recovery, so matching
// never merges policies that admit differently. The trajectory check cannot
// see a change that merges fewer policies, so the number of matching pairs is
// pinned as well.
TEST_F(RPCRateLimiterTest, maxProbeIntervalIsPartOfThePolicy) {
  auto& registry = RPCRateLimiterRegistry::global();
  const RPCRateLimiter::Config oneMinute{
      .adaptive = true, .ceiling = 25, .floor = 0.01};
  auto tenSeconds = oneMinute;
  tenSeconds.maxProbeInterval = std::chrono::seconds(10);

  const auto first = registry.getForPolicy("test.probe-interval", oneMinute);
  const auto second = registry.getForPolicy("test.probe-interval", tenSeconds);
  EXPECT_FALSE(first.policyScoped);
  EXPECT_TRUE(second.policyScoped);
  EXPECT_NE(second.limiter, first.limiter);
  EXPECT_EQ(
      second.limiter->config().maxProbeInterval, std::chrono::seconds(10));

  // A fixed limiter never paces, so the interval does not separate policies.
  const RPCRateLimiter::Config fixed{.adaptive = false, .ceiling = 25};
  auto fixedTenSeconds = fixed;
  fixedTenSeconds.maxProbeInterval = std::chrono::seconds(10);
  const auto fixedFirst =
      registry.getForPolicy("test.probe-interval-fixed", fixed);
  const auto fixedSecond =
      registry.getForPolicy("test.probe-interval-fixed", fixedTenSeconds);
  EXPECT_EQ(fixedSecond.limiter, fixedFirst.limiter);
  EXPECT_FALSE(fixedSecond.policyScoped);
}

TEST_F(RPCRateLimiterTest, matchingPoliciesAdmitAlikeAcrossPolicyGrid) {
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<RPCRateLimiter::Config> grid;
  for (const bool adaptive : {false, true}) {
    for (const int64_t ceiling : {0, 1, 25, 200, 250}) {
      for (const double floor : {0.0, 0.5, 1.0, 25.0, 50.0, 300.0}) {
        for (const double decrease : {0.25, 0.5, 0.99, kNaN}) {
          grid.push_back({adaptive, ceiling, floor, decrease});
        }
      }
    }
  }
  auto describe = [](const RPCRateLimiter::Config& policy) {
    return "{adaptive=" + std::to_string(policy.adaptive) +
        ", ceiling=" + std::to_string(policy.ceiling) +
        ", floor=" + std::to_string(policy.floor) +
        ", decrease=" + std::to_string(policy.decreaseFactor) + "}";
  };
  auto trajectory = [](const RPCRateLimiter::Config& policy) {
    // A frozen clock: a recovering paced window grows with elapsed time, which
    // would make the trajectory depend on scheduling.
    auto limiter = std::make_shared<RPCRateLimiter>(
        "test.backend",
        [] { return std::chrono::steady_clock::time_point{}; },
        nullptr,
        [] { return 1.0; });
    EXPECT_TRUE(limiter->initializeOrMatch(policy));
    std::vector<int64_t> steps;
    auto record = [&] {
      const auto stats = limiter->stats();
      steps.insert(
          steps.end(),
          {stats.capacity,
           stats.limitMilli,
           stats.lowWaterLimitMilli,
           stats.hardLimit});
    };
    record();
    for (int i = 0; i < 3; ++i) {
      limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
      record();
    }
    for (int i = 0; i < 3; ++i) {
      limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 10, 1'000'000);
      record();
    }
    return steps;
  };

  std::vector<std::vector<int64_t>> trajectories;
  std::vector<std::shared_ptr<RPCRateLimiter>> sealed;
  for (const auto& policy : grid) {
    trajectories.push_back(trajectory(policy));
    sealed.push_back(std::make_shared<RPCRateLimiter>("test.backend"));
    EXPECT_TRUE(sealed.back()->initializeOrMatch(policy));
  }
  int64_t matchingPairs{0};
  int64_t divergentPairs{0};
  std::string firstDivergent;
  for (size_t a = 0; a < grid.size(); ++a) {
    for (size_t b = 0; b < grid.size(); ++b) {
      if (!sealed[a]->initializeOrMatch(grid[b])) {
        continue;
      }
      ++matchingPairs;
      if (trajectories[a] != trajectories[b] && divergentPairs++ == 0) {
        firstDivergent = describe(grid[a]) + " and " + describe(grid[b]);
      }
    }
  }
  EXPECT_EQ(divergentPairs, 0)
      << "matching policies admit differently, first " << firstDivergent;
  // 240 policies fall into 68 classes; this is the sum of their squared sizes.
  EXPECT_EQ(matchingPairs, 5166);
}

// A query gets the backend's shared limiter when its policy admits like the
// sealed one, and otherwise the limiter for its own policy, which every later
// query running that policy shares.
TEST_F(RPCRateLimiterTest, getForPolicySharesLimitersByPolicy) {
  auto& registry = RPCRateLimiterRegistry::global();
  const RPCRateLimiter::Config small{
      .adaptive = true, .ceiling = 25, .floor = 1};
  const RPCRateLimiter::Config large{
      .adaptive = true, .ceiling = 100, .floor = 1};

  const auto first = registry.getForPolicy("test.backend", small);
  EXPECT_EQ(first.limiter, &registry.get("test.backend"));
  EXPECT_FALSE(first.policyScoped);

  const auto second = registry.getForPolicy("test.backend", large);
  EXPECT_NE(second.limiter, first.limiter);
  EXPECT_TRUE(second.policyScoped);
  EXPECT_EQ(second.limiter->config(), large);
  EXPECT_EQ(registry.get("test.backend").config(), small);

  {
    const auto again = registry.getForPolicy("test.backend", large);
    EXPECT_EQ(again.limiter, second.limiter);
    EXPECT_TRUE(again.policyScoped);
  }
  {
    // Admits like 'small', so it shares the shared limiter.
    auto clampsToSmall = small;
    clampsToSmall.decreaseFactor = std::numeric_limits<double>::quiet_NaN();
    const auto alike = registry.getForPolicy("test.backend", clampsToSmall);
    EXPECT_EQ(alike.limiter, first.limiter);
    EXPECT_FALSE(alike.policyScoped);
  }
  {
    // Policies one ULP apart get separate limiters, each sealed with a policy
    // that matches its own.
    auto oneUlpAway = large;
    oneUlpAway.floor = std::nextafter(large.floor, 2.0);
    const auto apart = registry.getForPolicy("test.backend", oneUlpAway);
    EXPECT_NE(apart.limiter, second.limiter);
    EXPECT_TRUE(apart.limiter->initializeOrMatch(oneUlpAway));
    EXPECT_FALSE(apart.limiter->initializeOrMatch(large));
  }
  {
    // A policy-scoped limiter is shared by every policy that matches the one
    // it was created for, including that policy itself when the registry
    // compares a reduced form of it.
    ASSERT_FALSE(registry.getForPolicy("pinned.backend", small).policyScoped);
    const RPCRateLimiter::Config pinned{
        .adaptive = true, .ceiling = 60, .floor = 60};
    const auto forPinned = registry.getForPolicy("pinned.backend", pinned);
    EXPECT_TRUE(forPinned.policyScoped);
    EXPECT_EQ(
        registry.getForPolicy("pinned.backend", pinned).limiter,
        forPinned.limiter);
    EXPECT_EQ(
        registry.getForPolicy("pinned.backend", {.ceiling = 60}).limiter,
        forPinned.limiter);
  }
  {
    // Another backend's limiters are its own, whatever the policy.
    ASSERT_FALSE(registry.getForPolicy("other.backend", small).policyScoped);
    const auto otherBackend = registry.getForPolicy("other.backend", large);
    EXPECT_TRUE(otherBackend.policyScoped);
    EXPECT_NE(otherBackend.limiter, second.limiter);
  }
}

// However queries with different policies interleave on a fresh backend, each
// runs under a limiter sealed with a policy that matches its own, and queries
// share a limiter exactly when their policies match.
TEST_F(RPCRateLimiterTest, concurrentQueriesGetTheLimiterForTheirPolicy) {
  constexpr int kThreads = 32;
  constexpr int kPolicies = 4;
  auto policyFor = [](int thread) {
    return RPCRateLimiter::Config{
        .adaptive = true, .ceiling = 25 * (thread % kPolicies + 1), .floor = 1};
  };
  auto& registry = RPCRateLimiterRegistry::global();
  std::vector<RPCRateLimiterRegistry::Lookup> lookups(kThreads);
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]() {
      while (!go.load(std::memory_order_relaxed)) {
      }
      lookups[i] = registry.getForPolicy("test.backend", policyFor(i));
    });
  }
  go.store(true);
  for (auto& thread : threads) {
    thread.join();
  }

  const auto* shared = &registry.get("test.backend");
  for (int i = 0; i < kThreads; ++i) {
    EXPECT_TRUE(lookups[i].limiter->initializeOrMatch(policyFor(i))) << i;
    EXPECT_EQ(lookups[i].policyScoped, lookups[i].limiter != shared) << i;
    for (int j = 0; j < kThreads; ++j) {
      EXPECT_EQ(
          lookups[i].limiter == lookups[j].limiter,
          i % kPolicies == j % kPolicies)
          << i << " " << j;
    }
  }
}

// A policy limiter is sealed before any query can reach it, so a query that
// gets it under contention already runs its own ceiling. One round rarely
// lands a query in that window, so the test runs many.
TEST_F(RPCRateLimiterTest, noQueryReachesAPolicyLimiterBeforeItIsSealed) {
  constexpr int kThreads = 32;
  constexpr int kPolicies = 4;
  constexpr int kRounds = 200;
  auto policyFor = [](int thread) {
    return RPCRateLimiter::Config{
        .adaptive = true, .ceiling = 25 * (thread % kPolicies + 1), .floor = 1};
  };
  auto& registry = RPCRateLimiterRegistry::global();
  int64_t numUnsealedReads{0};
  for (int round = 0; round < kRounds; ++round) {
    const std::string backend = "unsealed.backend." + std::to_string(round);
    ASSERT_FALSE(registry.getForPolicy(backend, {.ceiling = 7}).policyScoped);
    std::vector<int64_t> hardLimits(kThreads);
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
      threads.emplace_back([&, i]() {
        while (!go.load(std::memory_order_relaxed)) {
        }
        hardLimits[i] = registry.getForPolicy(backend, policyFor(i))
                            .limiter->stats()
                            .hardLimit;
      });
    }
    go.store(true);
    for (auto& thread : threads) {
      thread.join();
    }
    for (int i = 0; i < kThreads; ++i) {
      numUnsealedReads += hardLimits[i] != policyFor(i).ceiling;
    }
  }
  EXPECT_EQ(numUnsealedReads, 0) << "over " << kRounds * kThreads << " lookups";
}

// A query that finds no limiter for its policy builds one and publishes it,
// unless another query published one for that policy in the meantime. Two
// queries that interleave there share the limiter published first.
TEST_F(RPCRateLimiterTest, queriesRacingToCreateAPolicyLimiterShareOne) {
#ifdef NDEBUG
  GTEST_SKIP() << "Needs TestValue injection, which release builds omit.";
#endif
  // The registry outlives each run, so every run takes a backend no earlier
  // run has seen.
  static std::atomic<int> numRuns{0};
  const std::string backend = "race.backend." + std::to_string(numRuns++);
  auto& registry = RPCRateLimiterRegistry::global();
  ASSERT_FALSE(registry.getForPolicy(backend, {.ceiling = 25}).policyScoped);
  const RPCRateLimiter::Config policy{.ceiling = 100};

  std::promise<void> paused;
  std::promise<void> resume;
  const auto resumed = resume.get_future().share();
  std::atomic<bool> pauseNext{true};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::rpc::RPCRateLimiterRegistry::getForPolicy",
      std::function<void(RPCRateLimiterRegistry*)>(
          [&](RPCRateLimiterRegistry* /*registry*/) {
            if (pauseNext.exchange(false)) {
              paused.set_value();
              resumed.wait();
            }
          }));

  RPCRateLimiter* slowLimiter{nullptr};
  std::thread slow(
      [&] { slowLimiter = registry.getForPolicy(backend, policy).limiter; });
  const bool slowPaused =
      paused.get_future().wait_for(std::chrono::seconds(30)) ==
      std::future_status::ready;
  const auto fast = registry.getForPolicy(backend, policy);
  resume.set_value();
  slow.join();

  ASSERT_TRUE(slowPaused) << "the first query never reached the injection";
  EXPECT_TRUE(fast.policyScoped);
  EXPECT_EQ(slowLimiter, fast.limiter);
}

// Tests reset the registry between cases. A policy-scoped limiter stays listed
// under its policy across the reset, so it must come back sealed with it.
TEST_F(RPCRateLimiterTest, testingResetKeepsPolicyScopedLimitersSealed) {
  auto& registry = RPCRateLimiterRegistry::global();
  const RPCRateLimiter::Config pinned{
      .adaptive = true, .ceiling = 100, .floor = 100};
  ASSERT_FALSE(
      registry.getForPolicy("reset.backend", {.ceiling = 25}).policyScoped);
  const auto scoped = registry.getForPolicy("reset.backend", pinned);
  ASSERT_TRUE(scoped.policyScoped);

  registry.testingReset();
  ASSERT_FALSE(
      registry.getForPolicy("reset.backend", {.ceiling = 25}).policyScoped);
  const auto again = registry.getForPolicy("reset.backend", pinned);
  EXPECT_EQ(again.limiter, scoped.limiter);
  EXPECT_EQ(again.limiter->config(), pinned);
}

TEST_F(RPCRateLimiterTest, admitOrWaitEitherAdmitsOrEnrols) {
  const std::string backend = "test.backend";
  setCeiling(backend, 1);

  // Room: admitted, and nothing enrolled to be woken later.
  auto free = limiterFor(backend).admitOrWait();
  EXPECT_TRUE(free.admitted);

  auto token = limiterFor(backend).acquire();

  // Full: not admitted, and the wait is fulfilled by the release.
  auto parked = limiterFor(backend).admitOrWait();
  ASSERT_FALSE(parked.admitted);
  EXPECT_FALSE(parked.wait.isReady());
  token = RPCRateLimiter::Token();
  EXPECT_TRUE(parked.wait.isReady());
}

TEST_F(RPCRateLimiterTest, backpressureWhenAtLimit) {
  const std::string backend = "test.backend";
  setCeiling(backend, 2);

  auto token1 = limiterFor(backend).acquire();
  // 1 pending, limit 2 -- capacity is free, so the future comes back ready.
  EXPECT_TRUE(limiterFor(backend).admitOrWait().admitted);

  auto token2 = limiterFor(backend).acquire();
  // 2 pending, limit 2 -- at the limit, so the caller genuinely waits.
  auto admission = limiterFor(backend).admitOrWait();
  EXPECT_FALSE(admission.wait.isReady());
}

TEST_F(RPCRateLimiterTest, backpressureReliefOnRelease) {
  const std::string backend = "test.backend";
  setCeiling(backend, 1);

  auto token1 = limiterFor(backend).acquire();

  // At limit — should block.
  auto admission = limiterFor(backend).admitOrWait();
  EXPECT_FALSE(admission.wait.isReady());

  // Release the token — waiter should be notified.
  token1 = RPCRateLimiter::Token();
  EXPECT_TRUE(admission.wait.isReady());
  EXPECT_EQ(limiterFor(backend).stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, integerWaitersRetryAdmissionOnRelease) {
  const std::string backend = "test.backend";
  setCeiling(backend, 1);

  auto token = limiterFor(backend).acquire();

  // Two waiters enqueue while at limit.
  auto admission1 = limiterFor(backend).admitOrWait();
  auto admission2 = limiterFor(backend).admitOrWait();
  ASSERT_FALSE(admission1.wait.isReady());
  ASSERT_FALSE(admission2.wait.isReady());

  // Notifications carry no reservation. Wake every waiter so one abandoned
  // future cannot consume the only notification and strand live work.
  token = RPCRateLimiter::Token();
  EXPECT_TRUE(admission1.wait.isReady());
  EXPECT_TRUE(admission2.wait.isReady());

  auto firstGrant = limiterFor(backend).tryAcquireUpTo(1);
  auto secondGrant = limiterFor(backend).tryAcquireUpTo(1);
  EXPECT_EQ(firstGrant.size() + secondGrant.size(), 1);
}

TEST_F(RPCRateLimiterTest, abandonedIntegerWaiterDoesNotStrandLiveWaiter) {
  const std::string backend = "test.backend";
  setCeiling(backend, 1);
  auto token = limiterFor(backend).acquire();

  {
    auto abandoned = limiterFor(backend).admitOrWait();
    ASSERT_FALSE(abandoned.admitted);
  }
  auto live = limiterFor(backend).admitOrWait();
  ASSERT_FALSE(live.admitted);

  token = RPCRateLimiter::Token();
  EXPECT_TRUE(live.wait.isReady());
}

TEST_F(RPCRateLimiterTest, perBackendIsolation) {
  const std::string backend1 = "backend.one";
  const std::string backend2 = "backend.two";
  setCeiling(backend1, 1);
  setCeiling(backend2, 1);

  auto tokenA = limiterFor(backend1).acquire();
  EXPECT_EQ(limiterFor(backend1).stats().pending, 1);
  EXPECT_EQ(limiterFor(backend2).stats().pending, 0);

  // backend1 at limit, backend2 not.
  EXPECT_FALSE(limiterFor(backend1).admitOrWait().admitted);
  EXPECT_TRUE(limiterFor(backend2).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, perBackendMaxPending) {
  const std::string backend = "test.backend";
  RPCRateLimiter::setDefaultCapacity(10);
  setCeiling(backend, 2);

  auto token1 = limiterFor(backend).acquire();
  auto token2 = limiterFor(backend).acquire();

  // This backend's own limit of 2 applies, not the global default of 10.
  EXPECT_FALSE(limiterFor(backend).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, defaultMaxPending) {
  RPCRateLimiter::setDefaultCapacity(2);
  const std::string backend = "test.default.backend";

  auto token1 = limiterFor(backend).acquire();
  EXPECT_TRUE(limiterFor(backend).admitOrWait().admitted);

  auto token2 = limiterFor(backend).acquire();
  // Global default of 2 reached.
  EXPECT_FALSE(limiterFor(backend).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, tokenMoveConstructor) {
  const std::string backend = "test.backend";

  auto token1 = limiterFor(backend).acquire();
  EXPECT_EQ(limiterFor(backend).stats().pending, 1);

  // Move construct — ownership transfers, count stays 1.
  auto token2 = std::move(token1);
  EXPECT_EQ(limiterFor(backend).stats().pending, 1);

  // Destroy moved-from token — no effect.
  // (token1 is already moved-from, but let it go out of scope naturally)
}

TEST_F(RPCRateLimiterTest, tokenMoveAssignment) {
  const std::string backend1 = "backend.one";
  const std::string backend2 = "backend.two";

  auto token1 = limiterFor(backend1).acquire();
  auto token2 = limiterFor(backend2).acquire();
  EXPECT_EQ(limiterFor(backend1).stats().pending, 1);
  EXPECT_EQ(limiterFor(backend2).stats().pending, 1);

  // Move-assign token2 into token1 — old token1 (backend1) released.
  token1 = std::move(token2);
  EXPECT_EQ(limiterFor(backend1).stats().pending, 0);
  EXPECT_EQ(limiterFor(backend2).stats().pending, 1);
}

// A Token releases through a back-pointer to its issuing limiter, so the
// registry resets each backend in place rather than destroying it. Holding a
// live token across the reset is the case that would dangle if it did not.
// Two writers configure a backend in a fixed order: the function sets a
// ceiling from its SQL option during initialize(), then the operator applies
// the session properties. amend() exists so the second writer keeps the first
// one's ceiling. A read-modify-write through config()/configure() would drop
// it silently -- ceiling falls to 0, which resolves to the built-in default,
// and a backend provisioned for 200 quietly runs at 20.
// A backend's admission identity is its admission key, and each transport
// composes that key from whatever distinguishes one deployment from another --
// IPNext uses
// smcTier plus tenant plus group. Two deployments must therefore resolve to
// two limiters, so one saturating or backing off cannot admit or throttle the
// other. Collapsing the key would silently reunite them.
TEST_F(RPCRateLimiterTest, deploymentsResolveToSeparateLimiters) {
  const std::string groupA = "ipnext_prod/deployment#tm908223594#ggti_a";
  const std::string groupB = "ipnext_prod/deployment#tm908223594#ggti_b";
  const std::string otherTenant = "ipnext_prod/deployment#tm111111111#ggti_a";

  // Same key is the same backend; different keys are different backends.
  EXPECT_EQ(&limiterFor(groupA), &limiterFor(groupA));
  EXPECT_NE(&limiterFor(groupA), &limiterFor(groupB));
  EXPECT_NE(&limiterFor(groupA), &limiterFor(otherTenant));

  // Configuration does not leak across them.
  setCeiling(groupA, 8);
  setCeiling(groupB, 64);
  EXPECT_EQ(limiterFor(groupA).stats().capacity, 8);
  EXPECT_EQ(limiterFor(groupB).stats().capacity, 64);

  // Nor does occupancy: saturating one leaves the other's admission untouched.
  std::vector<RPCRateLimiter::Token> held;
  held.reserve(8);
  for (int i = 0; i < 8; ++i) {
    held.push_back(limiterFor(groupA).acquire());
  }
  EXPECT_EQ(limiterFor(groupA).available(), 0);
  EXPECT_EQ(limiterFor(groupB).available(), 64);
}

// Raising the ceiling grows capacity, but no release or adaptation need
// follow, so a driver parked under the old capacity would stay parked until
// some unrelated event woke it.
// available() followed by acquire() is two steps, so concurrent callers can
// each observe the same free slot and all take it -- the cap then bounds each
// caller's decision rather than the backend. tryAcquireUpTo() decides and
// claims in one atomic step, so pending can never exceed capacity however many
// race.
//
// One-sided by construction: the atomic form cannot overshoot, so this cannot
// flake; a check-then-claim form overshoots within a few thousand attempts.
// The bulk grant takes one lock for the whole chunk, so the capacity read and
// the claim are separated by a compare-exchange rather than a mutex. Concurrent
// callers must never be granted more than capacity between them.
//
// One-sided by construction: the atomic form cannot overshoot, so this cannot
// flake; a read-then-add form overshoots within a few thousand attempts.
TEST_F(RPCRateLimiterTest, bulkGrantNeverExceedsCapacity) {
  constexpr int64_t kCeiling = 6;
  constexpr int kThreads = 12;
  constexpr int kAttemptsPerThread = 20'000;
  auto& limiter = limiterFor("test.backend");
  limiter.amend(
      [](RPCRateLimiter::Config& config) { config.ceiling = kCeiling; });

  // Asking for more than the ceiling yields exactly the ceiling, not none.
  {
    auto all = limiter.tryAcquireUpTo(kCeiling * 4);
    EXPECT_EQ(static_cast<int64_t>(all.size()), kCeiling);
    EXPECT_EQ(limiter.stats().pending, kCeiling);
  }
  EXPECT_EQ(limiter.stats().pending, 0) << "tokens release on destruction";

  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&]() {
      while (!go.load(std::memory_order_relaxed)) {
      }
      for (int n = 0; n < kAttemptsPerThread; ++n) {
        // Each asks for the whole ceiling, so two concurrent callers reading
        // the same pending count would together claim twice it.
        auto grant = limiter.tryAcquireUpTo(kCeiling);
      }
    });
  }
  go.store(true);
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(limiter.stats().pending, 0) << "every slot should be released";
  EXPECT_LE(limiter.stats().peakPending, kCeiling)
      << kThreads << " racing callers drove pending to "
      << limiter.stats().peakPending << " against a capacity of " << kCeiling;
}

TEST_F(RPCRateLimiterTest, tryAcquireNeverExceedsCapacityUnderContention) {
  constexpr int64_t kCeiling = 4;
  constexpr int kThreads = 16;
  constexpr int kAttemptsPerThread = 20'000;
  auto& limiter = limiterFor("test.backend");
  limiter.amend(
      [](RPCRateLimiter::Config& config) { config.ceiling = kCeiling; });

  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&]() {
      while (!go.load(std::memory_order_relaxed)) {
      }
      for (int n = 0; n < kAttemptsPerThread; ++n) {
        // Take and immediately release, so slots churn and callers keep
        // racing for the same few.
        auto slot = limiter.tryAcquireUpTo(1);
      }
    });
  }
  go.store(true);
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(limiter.stats().pending, 0) << "every slot should be released";
  EXPECT_LE(limiter.stats().peakPending, kCeiling)
      << kThreads << " racing callers drove pending to "
      << limiter.stats().peakPending << " against a capacity of " << kCeiling;
}

TEST_F(RPCRateLimiterTest, amendWakesWaitersWhenCapacityGrows) {
  auto& limiter = limiterFor("test.backend");
  limiter.amend([](RPCRateLimiter::Config& config) { config.ceiling = 1; });

  // Take the only slot, then park a second caller behind it.
  auto token = limiter.acquire();
  auto parked = limiter.admitOrWait();
  EXPECT_FALSE(parked.wait.isReady());

  // Growing the ceiling alone must release the waiter; nothing else happens.
  limiter.amend([](RPCRateLimiter::Config& config) { config.ceiling = 4; });
  EXPECT_TRUE(parked.wait.isReady())
      << "a waiter stayed parked after capacity grew";
}

TEST_F(RPCRateLimiterTest, amendPreservesAnEarlierWritersCeiling) {
  const std::string backend = "test.backend";
  auto& limiter = limiterFor(backend);

  // Writer one: the function's SQL option.
  limiter.amend([](RPCRateLimiter::Config& config) { config.ceiling = 200; });

  // Writer two: session properties, with max_limit unset (0), so it must not
  // touch the ceiling.
  limiter.amend([](RPCRateLimiter::Config& config) {
    config.adaptive = true;
    config.floor = 4;
    config.decreaseFactor = 0.25;
  });

  const auto config = limiter.config();
  EXPECT_EQ(config.ceiling, 200);
  EXPECT_TRUE(config.adaptive);
  EXPECT_EQ(config.floor, 4);
  EXPECT_DOUBLE_EQ(config.decreaseFactor, 0.25);
  EXPECT_EQ(limiter.stats().capacity, 200);
}

TEST_F(RPCRateLimiterTest, testingResetClearsStateInPlace) {
  const std::string backend = "test.backend";
  RPCRateLimiter::setDefaultCapacity(5);
  setCeiling(backend, 3);
  auto token = limiterFor(backend).acquire();

  RPCRateLimiterRegistry::global().testingReset();

  EXPECT_EQ(RPCRateLimiter::defaultCapacity(), 20);
  EXPECT_EQ(limiterFor(backend).stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, adaptiveDisabledIsNoop) {
  const std::string backend = "test.backend";
  setCeiling(backend, 10);
  // Adaptive off (default): the overload signal must not shrink the cap.
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 10);
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 10);
}

TEST_F(RPCRateLimiterTest, adaptiveMultiplicativeDecrease) {
  const std::string backend = "test.backend";
  setCeiling(backend, 16);
  setAdaptive(backend, /*enabled*/ true, /*floor*/ 1, /*decreaseFactor*/ 0.5);

  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 8);
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 4);
}

TEST_F(RPCRateLimiterTest, adaptiveStartsBelowLargeHardLimit) {
  auto limiter = std::make_shared<RPCRateLimiter>("test.large-hard-limit");
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 1'000'000,
          .floor = 0.01,
          .decreaseFactor = 0.5,
      });

  const auto initial = limiter->stats();
  EXPECT_EQ(initial.capacity, 200);
  EXPECT_EQ(initial.lowWaterCapacity, 200);
  EXPECT_EQ(initial.limitMilli, 200'000);
  EXPECT_EQ(initial.lowWaterLimitMilli, 200'000);
  EXPECT_EQ(initial.hardLimit, 1'000'000);

  auto granted = limiter->tryAcquireUpTo(1'000'000);
  EXPECT_EQ(granted.size(), 200);
  granted.clear();

  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 400);
  EXPECT_EQ(limiter->stats().capacity, 201);
  EXPECT_EQ(limiter->stats().limitMilli, 201'990);
}

TEST_F(RPCRateLimiterTest, adaptiveFlooredAtMinLimit) {
  const std::string backend = "test.backend";
  setCeiling(backend, 16);
  setAdaptive(backend, true, /*floor*/ 4, 0.5);

  for (int i = 0; i < 10; ++i) {
    limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  // 16 -> 8 -> 4, then pinned at the floor.
  EXPECT_EQ(limiterFor(backend).stats().capacity, 4);
}

TEST_F(RPCRateLimiterTest, fractionalLimitUsesLearnedServiceHorizon) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.fractional.backend", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 0.25,
          .decreaseFactor = 0.5,
      });

  limiter->onOutcome(
      RPCRateLimiter::Outcome::kSuccess,
      1,
      std::chrono::duration_cast<std::chrono::nanoseconds>(100ms).count());
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 250);
  EXPECT_EQ(limiter->stats().serviceHorizonNanos, 100'000'000);

  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  clock.advance(399ms);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, staleSuccessUpdatesHorizonWithoutGrowingWindow) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.stale-horizon", clock);
  enableFractionalLimit(*limiter, 0.25);

  auto granted = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(granted.size(), 1);
  const auto staleEpoch = granted.front().overloadEpoch();
  granted.clear();
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 500);

  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  limiter->onSuccessForEpoch(
      staleEpoch,
      1,
      std::chrono::duration_cast<std::chrono::nanoseconds>(100ms).count());

  EXPECT_EQ(limiter->stats().limitMilli, 500);
  EXPECT_EQ(limiter->stats().serviceHorizonNanos, 100'000'000);
  clock.advance(199ms);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, staleSuccessIsReportedAsDiscarded) {
  const std::string backend = "test.backend";
  setCeiling(backend, 16);
  setAdaptive(backend, /*enabled=*/true, /*floor=*/1, /*decreaseFactor=*/0.5);
  auto& limiter = limiterFor(backend);
  const auto staleEpoch = limiter.acquire().overloadEpoch();
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  const auto currentEpoch = limiter.acquire().overloadEpoch();
  ASSERT_EQ(limiter.stats().limitMilli, 8'000);

  EXPECT_FALSE(limiter.onSuccessForEpoch(staleEpoch, 1, 0));
  EXPECT_EQ(limiter.stats().limitMilli, 8'000);

  EXPECT_TRUE(limiter.onSuccessForEpoch(currentEpoch, 1, 0));
}

TEST_F(RPCRateLimiterTest, staleSuccessWithNoCreditToLoseIsNotReported) {
  const std::string backend = "test.backend";
  setCeiling(backend, 16);
  auto& limiter = limiterFor(backend);

  // In fixed mode an overload still opens a new epoch, but success earns
  // nothing.
  const auto fixedEpoch = limiter.acquire().overloadEpoch();
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_TRUE(limiter.onSuccessForEpoch(fixedEpoch, 1, 0));

  // At the ceiling, recovery has nowhere to go.
  setAdaptive(backend, /*enabled=*/true, /*floor=*/1, /*decreaseFactor=*/0.5);
  const auto ceilingEpoch = limiter.acquire().overloadEpoch();
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000);
  ASSERT_EQ(limiter.stats().limitMilli, 16'000);
  EXPECT_TRUE(limiter.onSuccessForEpoch(ceilingEpoch, 1, 0));
}

TEST_F(RPCRateLimiterTest, staleSuccessAfterDefaultCapacityDropIsNotReported) {
  RPCRateLimiter::setDefaultCapacity(16);
  const std::string backend = "test.default.backend";
  setAdaptive(backend, /*enabled=*/true, /*floor=*/1, /*decreaseFactor=*/0.5);
  auto& limiter = limiterFor(backend);
  const auto staleEpoch = limiter.acquire().overloadEpoch();
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  // Lowering the default capacity below the stored window leaves it at the
  // new ceiling, where recovery has nowhere to go.
  RPCRateLimiter::setDefaultCapacity(4);
  ASSERT_EQ(limiter.stats().limitMilli, 4'000);
  EXPECT_TRUE(limiter.onSuccessForEpoch(staleEpoch, 1, 0));
}

TEST_F(RPCRateLimiterTest, timerSchedulingFailureFailsEveryWaiter) {
  ThrowingTimekeeper timekeeper;
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.throwing-timekeeper",
      [] { return std::chrono::steady_clock::time_point{}; },
      &timekeeper,
      [] { return 1.0; });
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  auto first = limiter->admitOrWait();
  auto second = limiter->admitOrWait();
  ASSERT_FALSE(first.admitted);
  ASSERT_FALSE(second.admitted);
  ASSERT_TRUE(first.wait.isReady());
  ASSERT_TRUE(second.wait.isReady());
  EXPECT_THROW(std::move(first.wait).get(), std::runtime_error);
  EXPECT_THROW(std::move(second.wait).get(), std::runtime_error);
}

TEST_F(RPCRateLimiterTest, timerFailureAllowsLaterWaiterToRearm) {
  FailOncePacingClock clock;
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.recovering-timekeeper",
      [&clock] { return clock.now(); },
      &clock,
      [] { return 1.0; });
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  auto first = limiter->admitOrWait();
  auto second = limiter->admitOrWait();
  ASSERT_FALSE(first.admitted);
  ASSERT_FALSE(second.admitted);
  EXPECT_FALSE(first.wait.isReady());
  EXPECT_FALSE(second.wait.isReady());

  clock.failFirstTimer();
  ASSERT_TRUE(first.wait.isReady());
  ASSERT_TRUE(second.wait.isReady());
  EXPECT_THROW(std::move(first.wait).get(), std::runtime_error);
  EXPECT_THROW(std::move(second.wait).get(), std::runtime_error);

  auto retry = limiter->admitOrWait();
  ASSERT_FALSE(retry.admitted);
  clock.advance(1'999ms);
  EXPECT_FALSE(retry.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(retry.wait.isReady());
  EXPECT_EQ(limiter->tryAcquireUpTo(1).size(), 1);
}

TEST_F(RPCRateLimiterTest, pacedAdmissionEnforcesDeadlineAndSingleFlight) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().capacity, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 500);
  EXPECT_EQ(limiter->available(), 0);

  auto timerWait = limiter->admitOrWait();
  ASSERT_FALSE(timerWait.admitted);
  EXPECT_FALSE(timerWait.wait.isReady());
  clock.advance(1'999ms);
  EXPECT_FALSE(timerWait.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(timerWait.wait.isReady());

  auto firstGrant = limiter->tryAcquireUpTo(10);
  ASSERT_EQ(firstGrant.size(), 1);
  EXPECT_EQ(limiter->stats().pending, 1);
  EXPECT_EQ(limiter->available(), 0);

  clock.advance(2s);
  auto concurrencyWait = limiter->admitOrWait();
  ASSERT_FALSE(concurrencyWait.admitted);
  EXPECT_FALSE(concurrencyWait.wait.isReady());

  firstGrant.clear();
  EXPECT_TRUE(concurrencyWait.wait.isReady());
  EXPECT_TRUE(limiter->admitOrWait().admitted);
  auto secondGrant = limiter->tryAcquireUpTo(10);
  EXPECT_EQ(secondGrant.size(), 1);
}

TEST_F(RPCRateLimiterTest, simultaneousPacingWakeupsGrantOneUnit) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  auto firstWait = limiter->admitOrWait();
  auto secondWait = limiter->admitOrWait();
  ASSERT_FALSE(firstWait.admitted);
  ASSERT_FALSE(secondWait.admitted);

  clock.advance(2s);
  EXPECT_TRUE(firstWait.wait.isReady());
  EXPECT_TRUE(secondWait.wait.isReady());

  auto firstGrant = limiter->tryAcquireUpTo(1);
  auto secondGrant = limiter->tryAcquireUpTo(1);
  EXPECT_EQ(firstGrant.size(), 1);
  EXPECT_TRUE(secondGrant.empty());
  EXPECT_EQ(limiter->stats().peakPending, 1);
}

TEST_F(RPCRateLimiterTest, losingPacedWaiterRearmsOnWinnerRelease) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.requeue", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  auto firstWait = limiter->admitOrWait();
  auto secondWait = limiter->admitOrWait();
  ASSERT_FALSE(firstWait.admitted);
  ASSERT_FALSE(secondWait.admitted);

  clock.advance(2s);
  ASSERT_TRUE(firstWait.wait.isReady());
  ASSERT_TRUE(secondWait.wait.isReady());

  auto winner = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(winner.size(), 1);
  EXPECT_TRUE(limiter->tryAcquireUpTo(1).empty());

  auto loser = limiter->admitOrWait();
  ASSERT_FALSE(loser.admitted);
  EXPECT_FALSE(loser.wait.isReady());
  winner.clear();
  clock.advance(1'999ms);
  EXPECT_FALSE(loser.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(loser.wait.isReady());
  EXPECT_EQ(limiter->tryAcquireUpTo(1).size(), 1);
}

TEST_F(RPCRateLimiterTest, fractionalLimitRecoversWithTimeAfterASuccess) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 500);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 250);
  EXPECT_EQ(limiter->stats().lowWaterLimitMilli, 250);

  // Without a success the window waits. The first success starts recovery:
  // the window then doubles every horizon, up to one, and the next paced
  // admission commits the growth.
  clock.advance(1s);
  EXPECT_EQ(limiter->stats().limitMilli, 250);
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 250);
  clock.advance(2s);
  EXPECT_FALSE(limiter->tryAcquireUpTo(1).empty());
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  EXPECT_EQ(limiter->stats().capacity, 1);
  EXPECT_EQ(limiter->stats().lowWaterLimitMilli, 250);
}

TEST_F(RPCRateLimiterTest, pacingTimeAccruesOnlyWhileTheWindowIsFractional) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.time", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 1'000);
  clock.advance(5s);
  EXPECT_EQ(limiter->stats().pacingNanos, 0);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 500);
  clock.advance(3s);
  // The stretch still in progress counts.
  EXPECT_EQ(limiter->stats().pacingNanos, std::chrono::nanoseconds{3s}.count());

  // A success starts recovery. The window grows back to one unit, and
  // admission stays paced until the next paced admission commits the growth.
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  clock.advance(2s);
  ASSERT_FALSE(limiter->tryAcquireUpTo(1).empty());
  ASSERT_EQ(limiter->stats().limitMilli, 1'000);
  clock.advance(7s);
  EXPECT_EQ(limiter->stats().pacingNanos, std::chrono::nanoseconds{5s}.count());
}

TEST_F(RPCRateLimiterTest, windowUnitSecondsIntegratesTheWindowOverTime) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.window.integral", clock);
  enableFractionalLimit(*limiter, 0.25);
  const auto start = limiter->stats();
  ASSERT_EQ(start.limitMilli, 2'000);

  // Two units for 3s; four for 1s once the ceiling grows; two for 2s once an
  // overload halves the window.
  clock.advance(3s);
  limiter->amend([](RPCRateLimiter::Config& config) { config.ceiling = 4; });
  ASSERT_EQ(limiter->stats().limitMilli, 4'000);
  clock.advance(1s);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 2'000);
  clock.advance(2s);

  const auto end = limiter->stats();
  EXPECT_DOUBLE_EQ(
      end.windowUnitSeconds - start.windowUnitSeconds, 2 * 3 + 4 * 1 + 2 * 2);
  EXPECT_EQ(
      end.clockNanos - start.clockNanos, std::chrono::nanoseconds{6s}.count());
}

TEST_F(RPCRateLimiterTest, windowUnitSecondsAccruesBeforeAHorizonSample) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.window.probe-floor", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 0.001,
          .decreaseFactor = 0.5,
          .maxProbeInterval = 10s,
      });
  // The 1s bootstrap horizon over the 10s probe interval sets a floor of 0.1.
  for (int i = 0; i < 5; ++i) {
    limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  ASSERT_EQ(limiter->stats().limitMilli, 100);
  const auto start = limiter->stats();
  clock.advance(10s);

  // A 2s horizon sample raises the probe floor, and the window with it.
  limiter->onOutcome(
      RPCRateLimiter::Outcome::kSuccess,
      1,
      std::chrono::nanoseconds{2s}.count());
  const auto end = limiter->stats();
  ASSERT_EQ(end.limitMilli, 200);
  EXPECT_DOUBLE_EQ(end.windowUnitSeconds - start.windowUnitSeconds, 0.1 * 10);
}

TEST_F(RPCRateLimiterTest, fractionalLimitRecoversWithCeilingOne) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.ceiling-one", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 1,
          .floor = 0.25,
          .decreaseFactor = 0.5,
      });

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 500);
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  clock.advance(2s);
  EXPECT_FALSE(limiter->tryAcquireUpTo(1).empty());
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  EXPECT_EQ(limiter->stats().capacity, 1);
}

TEST_F(RPCRateLimiterTest, pacingDisabledStopsAtIntegerFloor) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.unpaced.backend", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 1,
          .decreaseFactor = 0.5,
      });

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiter->stats().capacity, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  EXPECT_TRUE(limiter->admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, disablingAdaptiveModeExitsPacing) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 500);

  auto config = limiter->config();
  config.adaptive = false;
  limiter->configure(config);

  EXPECT_EQ(limiter->stats().limitMilli, 2'000);
  EXPECT_EQ(limiter->stats().capacity, 2);
  EXPECT_TRUE(limiter->admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, reenablingAdaptivePreservesLifetimeLowWater) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.toggle", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().lowWaterLimitMilli, 500);

  auto config = limiter->config();
  config.adaptive = false;
  limiter->configure(config);
  config.adaptive = true;
  limiter->configure(config);

  EXPECT_EQ(limiter->stats().limitMilli, 2'000);
  EXPECT_EQ(limiter->stats().lowWaterLimitMilli, 500);
}

TEST_F(RPCRateLimiterTest, canceledPacedWaiterDoesNotStrandNextWaiter) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  clock.advance(2s);
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);

  {
    auto abandoned = limiter->admitOrWait();
    ASSERT_FALSE(abandoned.admitted);
  }
  auto live = limiter->admitOrWait();
  ASSERT_FALSE(live.admitted);

  clock.advance(2s);
  tokens.clear();
  EXPECT_TRUE(live.wait.isReady());
}

TEST_F(RPCRateLimiterTest, pacedAdmissionRemainsSingleFlightAfterCeilingGrows) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.ceiling-growth", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 1,
          .floor = 0.25,
          .decreaseFactor = 0.5,
      });
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  clock.advance(2s);
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);

  auto config = limiter->config();
  config.ceiling = 4;
  limiter->configure(config);
  clock.advance(2s);

  auto blocked = limiter->admitOrWait();
  EXPECT_FALSE(blocked.admitted);
  EXPECT_FALSE(blocked.wait.isReady());
  EXPECT_TRUE(limiter->tryAcquireUpTo(1).empty());
  tokens.clear();
  EXPECT_TRUE(blocked.wait.isReady());
}

TEST_F(RPCRateLimiterTest, fractionalLimitIncreaseReschedulesEarlierWake) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.1);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);

  // A fixed window of 0.25 waits 4 horizons. Once a success starts recovery,
  // the window doubles every horizon and the wait is T(0.25) = log2(1 + ln2 /
  // 0.25) = 1.9156 horizons.
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  ASSERT_EQ(limiter->stats().limitMilli, 250);
  clock.advance(1'915ms);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, pacingUsesAdmissionStartForNextDeadline) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);

  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  clock.advance(10s);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  tokens.clear();

  EXPECT_TRUE(limiter->admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, pacingProbesAtLeastOncePerMaxInterval) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.01);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  for (int i = 0; i < 7; ++i) {
    limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  // The configured floor of 0.01 is raised to 1s / 60s, so probes are at most
  // 60s apart.
  ASSERT_EQ(limiter->stats().limitMilli, 17);

  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  clock.advance(59'999ms);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
}

// Admits one call whenever pacing allows, completes it with 'outcome', and
// returns the number of calls admitted before 'stop' returns true or 'limit'
// elapses.
int64_t runPacedCalls(
    RPCRateLimiter& limiter,
    ManualPacingClock& clock,
    RPCRateLimiter::Outcome outcome,
    std::chrono::seconds limit,
    const std::function<bool()>& stop) {
  const auto deadline = clock.now() + limit;
  // Bounds the loop when calls are admitted without the clock moving, as they
  // are once pacing stops.
  constexpr int64_t kMaxCalls{10'000};
  int64_t numCalls{0};
  while (!stop() && clock.now() < deadline && numCalls < kMaxCalls) {
    auto wait = limiter.admitOrWait();
    while (!wait.admitted && !wait.wait.isReady() && clock.now() < deadline) {
      clock.advance(100ms);
    }
    if (!limiter.tryAcquireUpTo(1).empty()) {
      ++numCalls;
      limiter.onOutcome(outcome, outcome == RPCRateLimiter::Outcome::kSuccess);
    }
  }
  return numCalls;
}

TEST_F(RPCRateLimiterTest, outageBacksOffToOneProbePerMaxInterval) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.outage", clock);
  enableFractionalLimit(*limiter, 0.001);

  // Every call is rejected for an hour. The window falls until the gap between
  // probes reaches the 60s maximum: a window of 1s / 60s.
  const auto numProbes = runPacedCalls(
      *limiter, clock, RPCRateLimiter::Outcome::kOverload, 3'600s, [] {
        return false;
      });

  EXPECT_EQ(limiter->stats().limitMilli, 17);
  EXPECT_GE(numProbes, 60);
  EXPECT_LE(numProbes, 75);
}

TEST_F(RPCRateLimiterTest, backOffProbesComeEarlyByAtMostTheJitter) {
  ManualPacingClock clock;
  // The earliest draw: every probe comes 20% before the interval.
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.paced.outage-jitter",
      [&clock] { return clock.now(); },
      clock.timekeeper(),
      [] { return 0.0; });
  enableFractionalLimit(*limiter, 0.001);

  // The window still backs off to 1s / 60s, but probes come every 48s rather
  // than every 60s: 75 an hour instead of 60.
  const auto numProbes = runPacedCalls(
      *limiter, clock, RPCRateLimiter::Outcome::kOverload, 3'600s, [] {
        return false;
      });

  EXPECT_EQ(limiter->stats().limitMilli, 17);
  EXPECT_GE(numProbes, 75);
  EXPECT_LE(numProbes, 90);
}

TEST_F(RPCRateLimiterTest, outageBackOffFollowsConfiguredMaxProbeInterval) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.outage-10s", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 0.001,
          .decreaseFactor = 0.5,
          .maxProbeInterval = 10s,
      });

  // An hour of rejections backs off to one probe per 10s: a window of
  // 1s / 10s.
  const auto numProbes = runPacedCalls(
      *limiter, clock, RPCRateLimiter::Outcome::kOverload, 3'600s, [] {
        return false;
      });

  EXPECT_EQ(limiter->stats().limitMilli, 100);
  EXPECT_GE(numProbes, 360);
  EXPECT_LE(numProbes, 375);
}

TEST_F(RPCRateLimiterTest, slowBackendStaysPacedDuringAnOutage) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.slow-outage", clock);
  enableFractionalLimit(*limiter, 0.001);
  // Calls take 120s, longer than the 60s probe interval.
  limiter->onOutcome(
      RPCRateLimiter::Outcome::kSuccess,
      1,
      std::chrono::duration_cast<std::chrono::nanoseconds>(120s).count());

  // The window still backs off below one, so probes stay paced, 240s apart at
  // a window of 0.5, rather than as fast as rejections return.
  const auto numProbes = runPacedCalls(
      *limiter, clock, RPCRateLimiter::Outcome::kOverload, 3'600s, [] {
        return false;
      });

  EXPECT_EQ(limiter->stats().limitMilli, 500);
  EXPECT_LE(numProbes, 20);
}

TEST_F(RPCRateLimiterTest, pacedWindowRecoversSoonAfterTheFirstSuccess) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.recovery", clock);
  enableFractionalLimit(*limiter, 0.001);
  runPacedCalls(
      *limiter, clock, RPCRateLimiter::Outcome::kOverload, 3'600s, [] {
        return false;
      });
  ASSERT_EQ(limiter->stats().limitMilli, 17);

  // The backend recovers. The next probe comes within 60s.
  const auto start = clock.now();
  int64_t numSuccesses{0};
  runPacedCalls(*limiter, clock, RPCRateLimiter::Outcome::kSuccess, 600s, [&] {
    return numSuccesses++ > 0;
  });
  EXPECT_LE(clock.now() - start, 60s);

  // Once it succeeds, the window doubles every horizon and reaches one about
  // 6.5 horizons later. Growth by successes alone would take about 2 / 0.017
  // horizons.
  const auto firstSuccess = clock.now();
  runPacedCalls(*limiter, clock, RPCRateLimiter::Outcome::kSuccess, 600s, [&] {
    return limiter->available() > 0;
  });
  EXPECT_GT(limiter->available(), 0);
  EXPECT_LE(clock.now() - firstSuccess, 8s);
}

namespace {
// Advances the clock in 'step' increments until a paced call is admitted, and
// returns the admission time.
std::chrono::steady_clock::time_point admitNextPacedCall(
    RPCRateLimiter& limiter,
    ManualPacingClock& clock,
    std::chrono::milliseconds step) {
  for (int i = 0; i < 1'000'000; ++i) {
    if (!limiter.tryAcquireUpTo(1).empty()) {
      return clock.now();
    }
    clock.advance(step);
  }
  VELOX_FAIL("No paced call was admitted");
}

// Backs 'limiter' off to the 60s probe interval, then reports one paced
// success, which starts time-based recovery.
void startPacedRecovery(RPCRateLimiter& limiter, ManualPacingClock& clock) {
  enableFractionalLimit(limiter, 0.001);
  runPacedCalls(limiter, clock, RPCRateLimiter::Outcome::kOverload, 3'600s, [] {
    return false;
  });
  admitNextPacedCall(limiter, clock, 100ms);
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
}
} // namespace

TEST_F(RPCRateLimiterTest, lateOverloadCutsTheWindowItsCallWasAdmittedUnder) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.late-overload", clock);
  startPacedRecovery(*limiter, clock);

  const auto admitted = admitNextPacedCall(*limiter, clock, 100ms);
  const auto admittedUnder = limiter->stats().limitMilli;

  // The rejection arrives 5 horizons later, as a timeout would. The cut halves
  // the window the call was admitted under, not the one projected since.
  clock.advance(5s);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  const auto cut = limiter->stats().limitMilli;
  EXPECT_LE(cut, admittedUnder / 2 + 1);

  // Recovery has ended: the next probe waits the back-off interval of the cut
  // window, and its admission commits no growth.
  const auto next = admitNextPacedCall(*limiter, clock, 10ms);
  EXPECT_GE(next - admitted, std::chrono::milliseconds(1'000'000 / cut - 10));
  EXPECT_EQ(limiter->stats().limitMilli, cut);
}

TEST_F(RPCRateLimiterTest, nonOverloadFailureStopsRecoveryUntilTheNextSuccess) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.failure", clock);
  startPacedRecovery(*limiter, clock);

  const auto admitted = admitNextPacedCall(*limiter, clock, 10ms);
  const auto window = limiter->stats().limitMilli;
  limiter->onOutcome(RPCRateLimiter::Outcome::kFailure, 0);

  // Back to the back-off interval of one horizon / window, measured from the
  // failed call's admission, rather than the shrinking recovery interval.
  const auto next = admitNextPacedCall(*limiter, clock, 10ms);
  EXPECT_GE(
      next - admitted, std::chrono::milliseconds(1'000'000 / window - 10));
  EXPECT_EQ(limiter->stats().limitMilli, window);

  // A success resumes recovery.
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  runPacedCalls(*limiter, clock, RPCRateLimiter::Outcome::kSuccess, 600s, [&] {
    return limiter->available() > 0;
  });
  EXPECT_GT(limiter->available(), 0);
}

TEST_F(RPCRateLimiterTest, pacedSuccessLearnsTheServiceHorizon) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.horizon", clock);
  enableFractionalLimit(*limiter, 0.001);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_LT(limiter->stats().limitMilli, 1'000);

  admitNextPacedCall(*limiter, clock, 100ms);
  limiter->onOutcome(
      RPCRateLimiter::Outcome::kSuccess,
      1,
      std::chrono::duration_cast<std::chrono::nanoseconds>(3s).count());
  EXPECT_EQ(limiter->stats().serviceHorizonNanos, 3'000'000'000);
}

TEST_F(RPCRateLimiterTest, recoveringWindowAboveOneMinusLn2WaitsForCredit) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.near-one", clock);
  enableFractionalLimit(*limiter, 0.001);
  // From the ceiling of 2, two overloads leave a window of 0.5.
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 500);

  const auto admitted = admitNextPacedCall(*limiter, clock, 10ms);
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);

  // Above 1 - ln2 the growing window reaches one before it accumulates a full
  // admission: log2(1 / 0.5) + 1 - (1 - 0.5) / ln2 = 1.279 horizons, against
  // 1.255 from the formula for smaller windows and 2 without recovery.
  const auto next = admitNextPacedCall(*limiter, clock, 10ms);
  EXPECT_GE(next - admitted, 1'270ms);
  EXPECT_LE(next - admitted, 1'290ms);
}

TEST_F(RPCRateLimiterTest, zeroMaxProbeIntervalLetsTheWindowReachTheFloor) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.no-probe-floor", clock);
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 0.001,
          .decreaseFactor = 0.5,
          .maxProbeInterval = 0ms,
      });

  runPacedCalls(
      *limiter, clock, RPCRateLimiter::Outcome::kOverload, 10'000s, [] {
        return false;
      });

  EXPECT_EQ(limiter->stats().limitMilli, 1);
}

TEST_F(RPCRateLimiterTest, fractionalLimitDecreaseInvalidatesEarlierWake) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter->stats().limitMilli, 250);
  clock.advance(2s);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(2s);
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, leavingPacingWakesWaitersImmediately) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);

  // A success starts recovery; one horizon later 0.5 has doubled to one, before
  // the paced wake at T(0.5) = 1.2787 horizons. The next outcome commits it.
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  clock.advance(1s);
  ASSERT_FALSE(wait.wait.isReady());
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);

  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  EXPECT_TRUE(wait.wait.isReady());
}

TEST_F(RPCRateLimiterTest, outstandingTimerDoesNotExtendLimiterLifetime) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.25);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  std::weak_ptr<RPCRateLimiter> weakLimiter = limiter;

  limiter.reset();
  EXPECT_TRUE(weakLimiter.expired());
  clock.advance(2s);
}

TEST_F(
    RPCRateLimiterTest,
    adaptiveRecoveryIsAdditiveAboveOneAndClampsAtHardLimit) {
  const std::string backend = "test.backend";
  setCeiling(backend, 16);
  setAdaptive(backend, true, 1, 0.5);

  limiterFor(backend).onOutcome(
      RPCRateLimiter::Outcome::kOverload, 0); // 16 -> 8
  ASSERT_EQ(limiterFor(backend).stats().capacity, 8);

  // One successful unit contributes one unit of squared-window credit.
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 8);
  EXPECT_EQ(limiterFor(backend).stats().limitMilli, 8'124);

  // A large drain reaches the ceiling and clears the adaptive state so the
  // static cap governs.
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 16);
  EXPECT_EQ(limiterFor(backend).stats().limitMilli, 16'000);

  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000);
  EXPECT_EQ(limiterFor(backend).stats().capacity, 16);
  EXPECT_EQ(limiterFor(backend).stats().limitMilli, 16'000);
}

TEST_F(RPCRateLimiterTest, adaptiveRecoveryIsInvariantToCompletionGrouping) {
  const RPCRateLimiter::Config config{
      .adaptive = true,
      .ceiling = 1'000,
      .floor = 1,
      .decreaseFactor = 0.5,
  };
  auto grouped = std::make_shared<RPCRateLimiter>("grouped-successes");
  auto individual = std::make_shared<RPCRateLimiter>("individual-successes");
  grouped->configure(config);
  individual->configure(config);
  grouped->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  individual->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  constexpr int64_t kSuccessfulUnits = 400;
  grouped->onOutcome(RPCRateLimiter::Outcome::kSuccess, kSuccessfulUnits);
  for (int64_t i = 0; i < kSuccessfulUnits; ++i) {
    individual->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  }

  EXPECT_NEAR(grouped->stats().limitMilli, individual->stats().limitMilli, 1);
  EXPECT_EQ(grouped->stats().capacity, individual->stats().capacity);
}

TEST_F(RPCRateLimiterTest, fractionalRecoveryIsInvariantToCompletionGrouping) {
  ManualPacingClock groupedClock;
  ManualPacingClock individualClock;
  auto grouped = makePacingLimiter("grouped-fractional", groupedClock);
  auto individual = makePacingLimiter("individual-fractional", individualClock);
  enableFractionalLimit(*grouped, 0.25);
  enableFractionalLimit(*individual, 0.25);
  for (int i = 0; i < 3; ++i) {
    grouped->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    individual->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }

  grouped->onOutcome(RPCRateLimiter::Outcome::kSuccess, 3);
  for (int i = 0; i < 3; ++i) {
    individual->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  }

  EXPECT_NEAR(grouped->stats().limitMilli, individual->stats().limitMilli, 1);
  EXPECT_EQ(grouped->stats().capacity, individual->stats().capacity);
}

TEST_F(RPCRateLimiterTest, adaptiveShrinkReducesAdmission) {
  const std::string backend = "test.backend";
  setCeiling(backend, 4);
  setAdaptive(backend, true, 1, 0.5);

  limiterFor(backend).onOutcome(
      RPCRateLimiter::Outcome::kOverload, 0); // cap 4 -> 2
  ASSERT_EQ(limiterFor(backend).stats().capacity, 2);

  auto token1 = limiterFor(backend).acquire();
  EXPECT_TRUE(limiterFor(backend).admitOrWait().admitted);
  auto token2 = limiterFor(backend).acquire();
  // At the shrunk cap of 2 (not the static 4), backpressure kicks in.
  EXPECT_FALSE(limiterFor(backend).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, noBackpressureBelowLimit) {
  const std::string backend = "test.backend";
  setCeiling(backend, 5);

  std::vector<RPCRateLimiter::Token> tokens;
  for (int i = 0; i < 4; ++i) {
    tokens.push_back(limiterFor(backend).acquire());
    EXPECT_TRUE(limiterFor(backend).admitOrWait().admitted);
  }
  EXPECT_EQ(limiterFor(backend).stats().pending, 4);
}

// A backend that never backed off has no low-water mark, and zero would be
// indistinguishable from "not recorded". Reporting the ceiling means every
// reading of the stat is a real capacity.
TEST_F(RPCRateLimiterTest, lowWaterReportsTheCeilingWhenNeverShrunk) {
  const std::string backend = "test.backend";
  setCeiling(backend, 64);
  setAdaptive(backend, /*enabled=*/true, /*floor=*/2, /*decreaseFactor=*/0.5);

  // Nothing has driven an overload, so capacity has never shrunk.
  EXPECT_EQ(limiterFor(backend).stats().lowWaterCapacity, 64);

  // Once it does shrink, the actual low-water value is reported instead.
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  const auto shrunk = limiterFor(backend).stats();
  EXPECT_LT(shrunk.lowWaterCapacity, 64);
  EXPECT_EQ(shrunk.lowWaterCapacity, shrunk.capacity);

  // Recovering above the low-water mark leaves the mark where it was.
  limiterFor(backend).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'024);
  EXPECT_EQ(
      limiterFor(backend).stats().lowWaterCapacity, shrunk.lowWaterCapacity);
}

// A SMART_PROBING limiter at ceiling 4 with registered demand, and helpers to
// drive it through the cut and probing phases.
struct OwnedWindow {
  OwnedWindow() {
    common::testutil::TestValue::enable();
    limiter = makePacingLimiter("test.owned-window", clock);
    limiter->configure(
        {.adaptive = true,
         .ceiling = 4,
         .floor = 1,
         .flowControlPolicy = RPCFlowControlPolicy::kSmartProbing});
    demand = limiter->registerRunnableDemand();
  }

  void cut() {
    auto tokens = limiter->tryAcquireUpTo(4);
    ASSERT_EQ(tokens.size(), 4);
    auto rejected = limiter->makeLease(std::move(tokens.front()));
    tokens.clear();
    auto result =
        rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
    EXPECT_EQ(
        result.disposition, RPCRateLimiter::FeedbackDisposition::kCanonical);
    EXPECT_EQ(limiter->stats().capacity, 2);
  }

  std::shared_ptr<RPCRateLimiter::AdmissionLease> admitOne() {
    auto tokens = limiter->tryAcquireUpTo(1);
    if (tokens.size() != 1) {
      ADD_FAILURE() << "Expected one owned admission";
      return nullptr;
    }
    return limiter->makeLease(std::move(tokens));
  }

  void succeed(const std::shared_ptr<RPCRateLimiter::AdmissionLease>& lease) {
    auto completion = lease->completeAndRelease(
        RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
    completion.credit.consume();
  }

  void baselineSuccesses(int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
      auto lease = admitOne();
      ASSERT_TRUE(lease);
      succeed(lease);
    }
  }

  // Raises the ceiling to 16, charges two startup acquisitions of four and
  // rejects one unit, which cuts the window to half the charged peak.
  void cutToFour() {
    limiter->amend([](auto& config) { config.ceiling = 16; });
    auto first = limiter->tryAcquireUpTo(4);
    auto second = limiter->tryAcquireUpTo(4);
    ASSERT_EQ(first.size() + second.size(), 8);
    auto rejected = limiter->makeLease(std::move(first.front()));
    first.clear();
    second.clear();
    EXPECT_EQ(
        rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
            .disposition,
        RPCRateLimiter::FeedbackDisposition::kCanonical);
    EXPECT_EQ(limiter->stats().capacity, 4);
  }

  void payRetryDebt() {
    baselineSuccesses(limiter->testingProbingSnapshot().retryDebt);
    EXPECT_EQ(limiter->testingProbingSnapshot().retryDebt, 0);
  }

  std::vector<std::shared_ptr<RPCRateLimiter::AdmissionLease>> fillBaseline() {
    std::vector<std::shared_ptr<RPCRateLimiter::AdmissionLease>> leases;
    while (limiter->stats().pending < limiter->stats().capacity) {
      auto lease = admitOne();
      if (!lease) {
        break;
      }
      leases.push_back(std::move(lease));
    }
    return leases;
  }

  ManualPacingClock clock;
  std::shared_ptr<RPCRateLimiter> limiter;
  std::shared_ptr<RPCRateLimiter::RunnableDemand> demand;
};

TEST_F(RPCRateLimiterTest, probeWaitsForOneCleanRoundThenPromotesOneUnit) {
  OwnedWindow window;
  window.cut();
  EXPECT_TRUE(window.limiter->testingProbingSnapshot().cut);
  EXPECT_TRUE(window.limiter->usesAttributedAdmissionControl());
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 2);
  EXPECT_EQ(window.limiter->tryAcquireUpTo(4).size(), 1);
  {
    auto held = window.fillBaseline();
    ASSERT_EQ(held.size(), 2);
    EXPECT_TRUE(window.limiter->tryAcquireUpTo(1).empty());
    for (auto& lease : held) {
      window.succeed(lease);
    }
  }
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 0);

  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 2);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().heldProbes, 1);
  EXPECT_TRUE(window.limiter->tryAcquireUpTo(1).empty());
  window.succeed(probe);
  EXPECT_EQ(window.limiter->stats().capacity, 3);
  // Level 4 failed once at the startup cut: 4 * 2^0 successes.
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 4);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 1);
}

TEST_F(RPCRateLimiterTest, failedProbeKeepsBaselineAndDoublesPenaltyUpToCap) {
  OwnedWindow window;
  window.cut();
  window.baselineSuccesses(2);
  // Level 3 penalties double from 3 until the 2 * 3^2 cap.
  for (const int64_t penalty : {3, 6, 12, 18}) {
    SCOPED_TRACE(penalty);
    auto held = window.fillBaseline();
    ASSERT_EQ(held.size(), 2);
    auto probe = window.admitOne();
    ASSERT_TRUE(probe);
    EXPECT_EQ(
        probe->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
            .disposition,
        RPCRateLimiter::FeedbackDisposition::kExclusivelySpeculative);
    EXPECT_EQ(window.limiter->stats().capacity, 2);
    EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, penalty);
    for (auto& lease : held) {
      lease->release();
    }
    window.baselineSuccesses(penalty);
  }
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 4);
}

TEST_F(RPCRateLimiterTest, probeAbsorbsOnlyOneOverloadBeforeCanonicalCut) {
  OwnedWindow window;
  window.cut();
  window.baselineSuccesses(2);
  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 2);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  EXPECT_EQ(
      held[0]
          ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kExclusivelySpeculative);
  EXPECT_EQ(window.limiter->stats().capacity, 2);
  EXPECT_EQ(
      held[1]
          ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
  window.succeed(probe);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
}

TEST_F(RPCRateLimiterTest, staleOverloadDuringProbeIsNotCharged) {
  OwnedWindow window;
  window.cutToFour();
  window.payRetryDebt();
  // Admitted before the next cut, and rejected only after a probe started.
  auto stale = window.admitOne();
  ASSERT_TRUE(stale);
  auto rejected = window.admitOne();
  ASSERT_TRUE(rejected);
  EXPECT_EQ(
      rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 2);
  window.payRetryDebt();
  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 1);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  ASSERT_EQ(window.limiter->testingProbingSnapshot().heldProbes, 1);

  EXPECT_EQ(
      stale->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kStale);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 0);
  window.succeed(probe);
  EXPECT_EQ(window.limiter->stats().capacity, 3);
}

TEST_F(RPCRateLimiterTest, overloadAfterProbePromotedCutsWindow) {
  OwnedWindow window;
  window.cut();
  window.baselineSuccesses(2);
  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 2);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  for (auto& lease : held) {
    window.succeed(lease);
  }
  // Admitted while the probe was in flight, rejected after it promoted.
  auto late = window.admitOne();
  ASSERT_TRUE(late);
  window.succeed(probe);
  ASSERT_EQ(window.limiter->stats().capacity, 3);

  EXPECT_EQ(
      late->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
}

TEST_F(RPCRateLimiterTest, probeFailedByQuotaCutAbsorbsNoOverload) {
  OwnedWindow window;
  window.cutToFour();
  window.payRetryDebt();
  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 4);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  held[0]->markNonCapacityRejection();
  EXPECT_EQ(
      held[0]
          ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 2);
  for (size_t i = 1; i < held.size(); ++i) {
    window.succeed(held[i]);
  }
  // Admitted after the quota cut, while the failed probe still holds a slot.
  auto fresh = window.admitOne();
  ASSERT_TRUE(fresh);

  EXPECT_EQ(
      fresh->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
}

TEST_F(RPCRateLimiterTest, unsettledProbeWithoutCapacityHintCutsWindow) {
  OwnedWindow window;
  window.cut();
  window.baselineSuccesses(2);
  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 2);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  // Completed before its raw attempt settles, as an operator timeout does.
  probe->bindRawAttempt();
  probe->markNonCapacityRejection();
  EXPECT_EQ(
      probe
          ->completeAndRelease(
              RPCRateLimiter::Outcome::kOverload, /*handedOff=*/false)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 0);
}

TEST_F(RPCRateLimiterTest, quotaRejectionDuringProbeCutsWithoutLevelFailure) {
  OwnedWindow window;
  window.cut();
  window.baselineSuccesses(2);
  auto held = window.fillBaseline();
  ASSERT_EQ(held.size(), 2);
  auto probe = window.admitOne();
  ASSERT_TRUE(probe);
  held[0]->markNonCapacityRejection();
  EXPECT_EQ(
      held[0]
          ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
          .disposition,
      RPCRateLimiter::FeedbackDisposition::kCanonical);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 0);
  window.succeed(probe);
  EXPECT_EQ(window.limiter->stats().capacity, 1);
}

TEST_F(RPCRateLimiterTest, sustainedLevelForgivesOneFailure) {
  OwnedWindow window;
  window.cut();
  window.baselineSuccesses(2);
  const auto canonicalCut = [&] {
    auto rejected = window.admitOne();
    ASSERT_TRUE(rejected);
    EXPECT_EQ(
        rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
            .disposition,
        RPCRateLimiter::FeedbackDisposition::kCanonical);
  };
  canonicalCut();
  EXPECT_EQ(window.limiter->stats().capacity, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 2);
  window.baselineSuccesses(2);
  {
    auto held = window.admitOne();
    auto probe = window.admitOne();
    ASSERT_TRUE(held);
    ASSERT_TRUE(probe);
    window.succeed(probe);
    window.succeed(held);
  }
  EXPECT_EQ(window.limiter->stats().capacity, 2);
  // The second clean success at level 2 sustains it for 2 * 2^0 successes.
  window.baselineSuccesses(1);
  canonicalCut();
  EXPECT_EQ(window.limiter->stats().capacity, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().nextLevelFailures, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 2);
}

TEST_F(RPCRateLimiterTest, startupSuccessesSettlingAfterCutPayRetryDebt) {
  OwnedWindow window;
  auto tokens = window.limiter->tryAcquireUpTo(4);
  ASSERT_EQ(tokens.size(), 4);
  const auto epoch = tokens.front().overloadEpoch();
  auto rejected = window.limiter->makeLease(std::move(tokens.back()));
  tokens.pop_back();
  rejected->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 2);
  window.limiter->makeLease(std::move(tokens))
      ->completeAndRelease(
          RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
  window.limiter->onSuccessForEpoch(epoch, 3, 1);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().retryDebt, 0);
  EXPECT_EQ(window.limiter->stats().capacity, 2);
}

TEST_F(RPCRateLimiterTest, rawSettlementKeepsOverloadVerdictAfterWakeFailure) {
  OwnedWindow window;
#ifdef NDEBUG
  GTEST_SKIP() << "Needs TestValue injection, which release builds omit.";
#endif
  window.limiter->amend([](auto& config) { config.ceiling = 1; });
  window.limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  for (const bool stale : {false, true}) {
    SCOPED_TRACE(stale ? "stale" : "fresh");
    auto tokens = window.limiter->tryAcquireUpTo(1);
    ASSERT_EQ(tokens.size(), 1);
    const auto admissionEpoch = tokens.front().overloadEpoch();
    auto lease = window.limiter->makeLease(std::move(tokens));
    lease->bindRawAttempt();
    auto waiter = window.limiter->admitOrWait();
    ASSERT_FALSE(waiter.admitted);
    ASSERT_TRUE(waiter.wait.valid());
    if (stale) {
      window.limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    }
    window.clock.advanceClockOnly(10ms);
    bool injected = false;
    {
      SCOPED_TESTVALUE_SET(
          "facebook::velox::exec::rpc::RPCRateLimiter::probingWakePreflight",
          std::function<void(RPCRateLimiter*)>([&](auto*) {
            injected = true;
            throw std::bad_alloc{};
          }));
      lease->settleRawAttempt(RPCRateLimiter::Outcome::kOverload, true);
    }
    EXPECT_TRUE(injected);
    EXPECT_EQ(window.limiter->stats().pending, 0);
    EXPECT_EQ(window.limiter->testingProbingSnapshot().held, 0);
    EXPECT_FALSE(waiter.wait.isReady());
    auto completion =
        lease->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true);
    EXPECT_EQ(completion.overloadEpoch, admissionEpoch);
    EXPECT_TRUE(completion.sharedOverloadHandled);
    EXPECT_EQ(
        completion.disposition,
        stale ? RPCRateLimiter::FeedbackDisposition::kStale
              : RPCRateLimiter::FeedbackDisposition::kCanonical);
    EXPECT_FALSE(completion.credit);
    EXPECT_FALSE(
        lease->completeAndRelease(RPCRateLimiter::Outcome::kOverload, true)
            .sharedOverloadHandled);
    // Mirror retry feedback without allowing an already handled or stale
    // overload to cut the current admission wave again.
    if (!completion.sharedOverloadHandled &&
        completion.disposition ==
            RPCRateLimiter::FeedbackDisposition::kCanonical) {
      window.limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    }
    auto retry = window.limiter->tryAcquireUpTo(1);
    ASSERT_EQ(retry.size(), 1);
    EXPECT_EQ(retry.front().overloadEpoch(), admissionEpoch + 1);
    retry.clear();
    EXPECT_TRUE(waiter.wait.isReady());
    EXPECT_EQ(window.limiter->stats().pending, 0);
  }
}

TEST_F(
    RPCRateLimiterTest,
    allocationPreflightRollsBackBeforePhysicalReservation) {
  OwnedWindow window;
#ifdef NDEBUG
  GTEST_SKIP() << "Needs TestValue injection, which release builds omit.";
#endif
  window.cut();
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::rpc::RPCRateLimiter::probingReservationPreflight",
      std::function<void(RPCRateLimiter*)>(
          [](auto*) __attribute__((noreturn)) { throw std::bad_alloc{}; }));
  EXPECT_THROW(window.limiter->tryAcquireUpTo(1), std::bad_alloc);
  EXPECT_EQ(window.limiter->stats().pending, 0);
  EXPECT_EQ(window.limiter->testingProbingSnapshot().held, 0);
}

TEST_F(
    RPCRateLimiterTest,
    actualChargeTimestampExcludesPreflightDelayAndReportedRtt) {
  OwnedWindow window;
  window.cut();
  {
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::rpc::RPCRateLimiter::probingReservationPreflight",
        std::function<void(RPCRateLimiter*)>(
            [&](auto*) { window.clock.advanceClockOnly(50ms); }));
    auto lease = window.admitOne();
    ASSERT_TRUE(lease);
    window.clock.advanceClockOnly(10ms);
    auto output = lease->completeAndRelease(
        RPCRateLimiter::Outcome::kSuccess, /*handedOff=*/false);
    output.credit.consume();
    EXPECT_EQ(window.limiter->stats().serviceHorizonNanos, 10'000'000);
  }
  EXPECT_EQ(window.limiter->testingProbingSnapshot().held, 0);
}

// Regression test for the defect that motivated one limiter per backend: the
// previous API configured the adaptive parameters process-globally, so a
// second backend's configuration silently reconfigured the first backend's
// limiter and both shrank together.
TEST_F(RPCRateLimiterTest, adaptiveIsPerBackend) {
  const std::string adaptiveBackend = "backend.adaptive";
  const std::string fixedBackend = "backend.fixed";
  setCeiling(adaptiveBackend, 16);
  setCeiling(fixedBackend, 16);
  setAdaptive(adaptiveBackend, true, /*floor*/ 1, /*decreaseFactor*/ 0.5);
  setAdaptive(fixedBackend, false, /*floor*/ 1, /*decreaseFactor*/ 0.5);

  limiterFor(adaptiveBackend)
      .onOutcome(RPCRateLimiter::Outcome::kOverload, /*units*/ 0);
  limiterFor(fixedBackend)
      .onOutcome(RPCRateLimiter::Outcome::kOverload, /*units*/ 0);

  EXPECT_EQ(limiterFor(adaptiveBackend).stats().capacity, 8);
  EXPECT_EQ(limiterFor(fixedBackend).stats().capacity, 16);
}

// Two adapting backends shrink by their own decrease factors rather than
// sharing one process-global value.
TEST_F(RPCRateLimiterTest, adaptiveFactorsAreIndependent) {
  const std::string halving = "backend.halving";
  const std::string quartering = "backend.quartering";
  setCeiling(halving, 16);
  setCeiling(quartering, 16);
  setAdaptive(halving, true, /*floor*/ 1, /*decreaseFactor*/ 0.5);
  setAdaptive(quartering, true, /*floor*/ 1, /*decreaseFactor*/ 0.25);

  limiterFor(halving).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiterFor(quartering).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);

  EXPECT_EQ(limiterFor(halving).stats().capacity, 8);
  EXPECT_EQ(limiterFor(quartering).stats().capacity, 4);
}

} // namespace
} // namespace facebook::velox::exec::rpc
