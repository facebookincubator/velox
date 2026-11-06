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

#include "velox/exec/rpc/RPCRateLimiter.h"

#include <folly/futures/ManualTimekeeper.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "velox/common/testutil/TestValue.h"

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

struct RateSimulationResult {
  int64_t elapsedMs{0};
  int64_t attempts{0};
  int64_t completed{0};
  int64_t peakPending{0};
  struct RateCut {
    int64_t atMs;
    int64_t admissionBucket;
    int64_t sendRateMilli;
  };
  std::vector<RateCut> rateCuts;
};

struct RateSimulationConfig {
  // An empty algorithm selects transport-owned retries.
  std::optional<RPCRateLimiter::RateAlgorithm> algorithm{};
  bool mixedCost{false};
  bool changingQuota{false};
  int64_t rows{1'000};
  int64_t responseMs{100};
  bool growingQuota{false};
  // Rejects the first attempt against a backend with spare capacity.
  bool rejectFirst{false};
  int64_t changeAtMs{10'000};
  int32_t workers{1};
  // Rejects every fifth attempt against a backend with spare capacity.
  bool independentRejections{false};
  int64_t concurrency{20};
  bool equalJitter{false};
  int64_t quotaPerSecond{50};
  std::optional<int64_t> quotaAfterChange{};
  // Zero keeps continuous refill; positive values reset a fixed-window budget.
  int64_t quotaWindowMs{0};
};

RateSimulationResult runFiniteRateSimulation(
    const RateSimulationConfig& config) {
  ManualPacingClock clock;
  std::vector<std::shared_ptr<RPCRateLimiter>> limiters;
  for (int32_t worker = 0; worker < config.workers; ++worker) {
    auto limiter = makePacingLimiter("test.finite-rate", clock);
    limiter->configure(
        {.adaptive = true,
         .ceiling = config.concurrency,
         .floor = 1,
         .ratePolicy =
             config.algorithm.value_or(RPCFlowControlPolicy::kWindowAimd),
         .scheduleFixes = config.algorithm.has_value() &&
             config.algorithm.value() != RPCFlowControlPolicy::kWindowAimd});
    limiters.push_back(std::move(limiter));
  }
  struct Work {
    int64_t id;
    int64_t cost;
    int32_t worker;
    int32_t retries{0};
    int64_t notBeforeMs{0};
  };
  struct Response {
    Work work;
    std::shared_ptr<RPCRateLimiter::AdmissionLease> lease;
    uint64_t epoch;
    int64_t dueMs;
    bool accepted;
  };
  std::deque<Work> queued;
  for (int64_t row = 0; row < config.rows; ++row) {
    queued.push_back(
        {.id = row,
         .cost = config.mixedCost && row % 2 == 0 ? 3 : 1,
         .worker = static_cast<int32_t>(row % config.workers)});
  }
  std::vector<Response> pending;
  std::vector<Response> inPlaceRetries;
  RateSimulationResult result;
  std::vector<int64_t> admissionWindowStarts(
      static_cast<size_t>(config.workers), -1);
  std::vector<int64_t> admissionBuckets(static_cast<size_t>(config.workers), 0);
  int64_t lastQuotaWindow{-1};
  const double burst =
      config.rejectFirst || config.independentRejections ? 20 : 5;
  double bucket{burst};
  const auto send = [&](Work work,
                        std::shared_ptr<RPCRateLimiter::AdmissionLease> lease,
                        uint64_t epoch,
                        int64_t nowMs) {
    ++result.attempts;
    const auto cost = static_cast<double>(work.cost);
    const bool accepted = bucket >= cost &&
        !(config.rejectFirst && result.attempts == 1) &&
        !(config.independentRejections && result.attempts % 5 == 0);
    if (accepted) {
      bucket -= cost;
    }
    pending.push_back(
        {work,
         std::move(lease),
         epoch,
         nowMs + (accepted ? config.responseMs : 1),
         accepted});
  };
  for (int64_t ms = 0; ms < 120'000; ++ms) {
    clock.advance(1ms);
    const auto quota = config.rejectFirst || config.independentRejections
        ? 10'000
        : config.changingQuota && ms >= config.changeAtMs
        ? config.quotaAfterChange.value_or(config.growingQuota ? 100 : 25)
        : config.quotaPerSecond;
    if (config.quotaWindowMs > 0) {
      const auto quotaWindow = ms / config.quotaWindowMs;
      if (quotaWindow != lastQuotaWindow) {
        bucket = static_cast<double>(quota) *
            static_cast<double>(config.quotaWindowMs) / 1'000.0;
        lastQuotaWindow = quotaWindow;
      }
    } else {
      bucket = std::min(burst, bucket + static_cast<double>(quota) / 1'000.0);
    }
    for (size_t i = 0; i < pending.size();) {
      if (pending[i].dueMs > ms) {
        ++i;
        continue;
      }
      auto response = std::move(pending[i]);
      pending.erase(
          pending.begin() +
          static_cast<std::vector<Response>::difference_type>(i));
      if (response.accepted) {
        response.lease->completeAndRelease(
            RPCRateLimiter::Outcome::kSuccess, config.responseMs * 1'000'000);
        limiters[response.work.worker]->onSuccessForEpoch(
            response.epoch, 1, config.responseMs * 1'000'000);
        ++result.completed;
      } else {
        const auto retry = response.work.retries++;
        const auto nominal = config.equalJitter
            ? std::min<int64_t>(30'000, 100LL << std::min(retry, 9))
            : 100LL << std::min(retry, 5);
        auto seed = static_cast<uint64_t>(response.work.id + 1) *
                0x9e3779b97f4a7c15ULL ^
            static_cast<uint64_t>(retry) * 0xbf58476d1ce4e5b9ULL;
        seed ^= seed >> 30;
        seed *= 0xbf58476d1ce4e5b9ULL;
        seed ^= seed >> 27;
        const auto waitMs = config.equalJitter
            ? nominal / 2 + static_cast<int64_t>(seed % (nominal / 2 + 1))
            : nominal;
        response.work.notBeforeMs = ms + waitMs;
        if (config.algorithm.has_value()) {
          const auto worker = static_cast<size_t>(response.work.worker);
          const auto previousCuts = config.quotaWindowMs > 0
              ? limiters[worker]->stats().rateDecreases
              : 0;
          response.lease->completeAndRelease(
              RPCRateLimiter::Outcome::kOverload, 1'000'000, true);
          if (config.quotaWindowMs > 0) {
            const auto stats = limiters[worker]->stats();
            if (stats.rateDecreases > previousCuts) {
              result.rateCuts.push_back(
                  {ms, admissionBuckets[worker], stats.sendRateMilli});
            }
          }
          queued.push_back(response.work);
        } else {
          response.dueMs = response.work.notBeforeMs;
          inPlaceRetries.push_back(std::move(response));
        }
      }
    }
    for (size_t i = 0; i < inPlaceRetries.size();) {
      if (inPlaceRetries[i].dueMs > ms) {
        ++i;
        continue;
      }
      auto retry = std::move(inPlaceRetries[i]);
      inPlaceRetries.erase(
          inPlaceRetries.begin() +
          static_cast<std::vector<Response>::difference_type>(i));
      send(retry.work, std::move(retry.lease), retry.epoch, ms);
    }
    for (int32_t worker = 0; worker < config.workers; ++worker) {
      const auto eligible = [ms, worker](const auto& work) {
        return work.worker == worker && work.notBeforeMs <= ms;
      };
      const auto ready = std::count_if(queued.begin(), queued.end(), eligible);
      auto granted = limiters[worker]->tryAcquireUpTo(ready);
      if (config.quotaWindowMs > 0 && !granted.empty()) {
        auto& start = admissionWindowStarts[worker];
        if (start < 0) {
          start = ms;
        }
        // Observe the same admission-triggered 500 ms measurement windows as
        // the limiter; this does not change the controller or backend model.
        if (ms - start >= 500) {
          ++admissionBuckets[worker];
          start = ms;
        }
      }
      for (auto& token : granted) {
        auto it =
            std::find_if(queued.begin(), queued.end(), [&](const auto& work) {
              return eligible(work) && work.retries > 0;
            });
        if (it == queued.end()) {
          it = std::find_if(queued.begin(), queued.end(), eligible);
        }
        VELOX_CHECK(it != queued.end());
        const auto work = *it;
        queued.erase(it);
        const auto epoch = token.overloadEpoch();
        send(work, limiters[worker]->makeLease(std::move(token)), epoch, ms);
      }
    }
    result.elapsedMs = ms + 1;
    if (result.completed == config.rows) {
      break;
    }
  }
  for (const auto& limiter : limiters) {
    result.peakPending += limiter->stats().peakPending;
  }
  return result;
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

TEST_F(
    RPCRateLimiterTest,
    flowPolicyDefaultsToWindowAimdAndParsesPublicChoices) {
  const RPCRateLimiter::Config defaults;
  EXPECT_EQ(defaults.ratePolicy, RPCFlowControlPolicy::kWindowAimd);
  EXPECT_FALSE(defaults.scheduleFixes);
  EXPECT_EQ(
      parseRPCFlowControlPolicy("WINDOW_AIMD"),
      RPCFlowControlPolicy::kWindowAimd);
  EXPECT_EQ(
      parseRPCFlowControlPolicy("window_AiMd"),
      RPCFlowControlPolicy::kWindowAimd);
  EXPECT_EQ(
      parseRPCFlowControlPolicy("rate_aimd"), RPCFlowControlPolicy::kRateAimd);
  EXPECT_EQ(parseRPCFlowControlPolicy("CuBiC"), RPCFlowControlPolicy::kCubic);
  EXPECT_EQ(
      rpcFlowControlPolicyName(RPCFlowControlPolicy::kWindowAimd),
      "WINDOW_AIMD");
  EXPECT_EQ(
      rpcFlowControlPolicyName(RPCFlowControlPolicy::kRateAimd), "RATE_AIMD");
  EXPECT_EQ(rpcFlowControlPolicyName(RPCFlowControlPolicy::kCubic), "CUBIC");
  EXPECT_THROW(parseRPCFlowControlPolicy(""), std::exception);
  EXPECT_THROW(parseRPCFlowControlPolicy("VEGAS"), std::exception);
  EXPECT_THROW(parseRPCFlowControlPolicy("AIMD"), std::exception);
  EXPECT_THROW(parseRPCFlowControlPolicy("LEGACY"), std::exception);
  EXPECT_THROW(parseRPCFlowControlPolicy("CUBIC extra"), std::exception);
}

TEST_F(
    RPCRateLimiterTest,
    ratePolicyImpliesRepairsAndDisabledRateRetainsRepairs) {
  auto limiter = std::make_shared<RPCRateLimiter>("test.policy-normalization");
  limiter->configure(
      {.adaptive = true, .ratePolicy = RPCFlowControlPolicy::kCubic});
  EXPECT_EQ(limiter->config().ratePolicy, RPCFlowControlPolicy::kCubic);
  EXPECT_TRUE(limiter->config().scheduleFixes);
  limiter->configure(
      {.adaptive = false, .ratePolicy = RPCFlowControlPolicy::kRateAimd});
  EXPECT_EQ(limiter->config().ratePolicy, RPCFlowControlPolicy::kWindowAimd);
  EXPECT_TRUE(limiter->config().scheduleFixes);
}

TEST_F(RPCRateLimiterTest, policyCacheDistinguishesRepairsWithInactiveRate) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config legacy{.ceiling = 1};
  auto* original =
      registry.getForPolicy("test.scheduler-policy", legacy).limiter;
  auto repaired = legacy;
  repaired.scheduleFixes = true;
  auto* modern =
      registry.getForPolicy("test.scheduler-policy", repaired).limiter;
  EXPECT_NE(original, modern);
  EXPECT_EQ(modern->config().ratePolicy, RPCFlowControlPolicy::kWindowAimd);
  EXPECT_TRUE(modern->config().scheduleFixes);
  EXPECT_FALSE(original->config().scheduleFixes);
}

TEST_F(RPCRateLimiterTest, policyCacheSeparatesSendingRateStrategies) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config cubic{
      .adaptive = true,
      .ceiling = 100,
      .ratePolicy = RPCFlowControlPolicy::kCubic,
      .scheduleFixes = true};
  auto* first =
      registry.getForPolicy("test.rate-strategy-policy", cubic).limiter;
  auto aimd = cubic;
  aimd.ratePolicy = RPCFlowControlPolicy::kRateAimd;
  auto* second =
      registry.getForPolicy("test.rate-strategy-policy", aimd).limiter;
  EXPECT_NE(first, second);
  EXPECT_EQ(
      registry.getForPolicy("test.rate-strategy-policy", cubic).limiter, first);
  EXPECT_EQ(first->stats().ratePolicy, RPCFlowControlPolicy::kCubic);
  EXPECT_EQ(second->stats().ratePolicy, RPCFlowControlPolicy::kRateAimd);
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

TEST_F(RPCRateLimiterTest, rateFeedbackIsRecordedOnceAtCompletion) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-feedback-completion", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 100,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto tokens = limiter->tryAcquireUpTo(2);
  ASSERT_EQ(tokens.size(), 2);
  const auto epoch = tokens.front().overloadEpoch();
  auto lease = limiter->makeLease(std::move(tokens));
  EXPECT_EQ(limiter->stats().completionAttempts, 0);
  clock.advance(1s);
  lease->completeAndRelease(RPCRateLimiter::Outcome::kSuccess, 1'000'000'000);
  auto completed = limiter->stats();
  EXPECT_EQ(completed.admittedAttempts, 2);
  EXPECT_EQ(completed.completionAttempts, 2);
  EXPECT_EQ(completed.successfulAttempts, 2);
  lease->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  limiter->onSuccessForEpoch(epoch, 2, 1'000'000'000);
  completed = limiter->stats();
  EXPECT_EQ(completed.completionAttempts, 2);
  EXPECT_EQ(completed.successfulAttempts, 2);
  EXPECT_EQ(completed.overloadAttempts, 0);
  EXPECT_EQ(completed.rateDecreases, 0);
}

TEST_F(RPCRateLimiterTest, unclassifiedCompletionMatchesRelease) {
  for (const bool rateSelected : {false, true}) {
    int64_t clockReads{0};
    auto limiter = std::make_shared<RPCRateLimiter>(
        "test.unclassified-completion",
        [&] {
          ++clockReads;
          return std::chrono::steady_clock::time_point{};
        },
        nullptr,
        [] { return 1.0; });
    limiter->configure(
        {.adaptive = true,
         .ceiling = 100,
         .floor = 1,
         .ratePolicy = rateSelected ? RPCFlowControlPolicy::kCubic
                                    : RPCFlowControlPolicy::kWindowAimd,
         .scheduleFixes = rateSelected});
    auto tokens = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(tokens.size(), 1);
    auto canceled = limiter->makeLease(std::move(tokens.front()));
    clockReads = 0;
    canceled->release();
    const auto releaseClockReads = clockReads;

    tokens = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(tokens.size(), 1);
    auto completed = limiter->makeLease(std::move(tokens.front()));
    clockReads = 0;
    completed->completeAndRelease(/*hardOverload=*/false);
    EXPECT_EQ(clockReads, releaseClockReads);
    EXPECT_EQ(limiter->stats().completionAttempts, 0);
    EXPECT_EQ(limiter->stats().successfulAttempts, 0);
    EXPECT_EQ(limiter->stats().pending, 0);
  }
}

TEST_F(RPCRateLimiterTest, firstHandoffBootstrapsAfterCanonicalOverload) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.stale-first-handoff", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 100,
       .floor = 100,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto tokens = limiter->tryAcquireUpTo(2);
  ASSERT_EQ(tokens.size(), 2);
  auto terminal = limiter->makeLease(std::move(tokens.front()));
  auto handedOff = limiter->makeLease(std::move(tokens.back()));
  terminal->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, false);
  EXPECT_EQ(limiter->stats().sendRateMilli, 0);
  EXPECT_EQ(limiter->stats().limitMilli, 100'000);

  const auto completion = handedOff->completeAndRelease(
      RPCRateLimiter::Outcome::kOverload, 0, true);
  EXPECT_TRUE(completion.sharedOverloadHandled);
  EXPECT_GT(limiter->stats().sendRateMilli, 0);
  EXPECT_EQ(limiter->stats().rateDecreases, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 100'000);
  clock.advance(20ms);
  EXPECT_EQ(limiter->tryAcquireUpTo(10).size(), 1);
}

TEST_F(RPCRateLimiterTest, rateProbeWithoutIntervalUsesBootstrapHorizon) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-bootstrap-probe", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 100,
       .floor = 0.01,
       .maxProbeInterval = 0ms,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  const auto initial = limiter->stats();
  EXPECT_EQ(initial.serviceHorizonNanos, 1'000'000'000);
  EXPECT_GT(initial.sendRateMilli, 0);
  EXPECT_GT(initial.nextPermitDelayNanos, 0);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  clock.advance(100ms);
  EXPECT_TRUE(wait.wait.isReady());
  auto admitted = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(admitted.size(), 1);
  limiter->makeLease(std::move(admitted.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kSuccess, 0);
  EXPECT_EQ(limiter->stats().serviceHorizonNanos, 1'000'000'000);
  EXPECT_GT(limiter->stats().sendRateMilli, 0);
}

TEST_F(RPCRateLimiterTest, rateTargetSurvivesIdleAndOutputDrain) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-idle", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 100,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  limiter->makeLease(std::move(tokens.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  const auto target = limiter->stats().sendRateMilli;
  EXPECT_GT(target, 0);
  clock.advance(1min);
  limiter->onSuccessForEpoch(0, 1, 0);
  EXPECT_EQ(limiter->stats().sendRateMilli, target);
  EXPECT_EQ(limiter->tryAcquireUpTo(10).size(), 1);
  EXPECT_TRUE(limiter->tryAcquireUpTo(10).empty());
}

TEST_F(RPCRateLimiterTest, abandonedRateWaitDoesNotTurnIdleIntoRecovery) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-abandoned-wait", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  clock.advance(20ms);
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  auto lease = limiter->makeLease(std::move(tokens.front()));
  {
    auto abandoned = limiter->admitOrWait();
    ASSERT_FALSE(abandoned.admitted);
    clock.advance(1ms);
  }
  clock.advance(20ms);
  const auto waiting = limiter->stats().rateWaitingNanos;
  EXPECT_GT(waiting, 0);
  const auto rate = limiter->stats().sendRateMilli;
  clock.advance(1min);
  EXPECT_EQ(limiter->stats().rateWaitingNanos, waiting);
  EXPECT_GT(limiter->stats().ratePacingNanos, waiting);
  lease->completeAndRelease(RPCRateLimiter::Outcome::kSuccess, 100'000'000);
  EXPECT_LT(limiter->stats().sendRateMilli, rate + 1'000);
}

TEST_F(RPCRateLimiterTest, zeroDemandStopsPollOnlyRecoveryClock) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-poll-idle", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  clock.advance(20ms);
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  auto lease = limiter->makeLease(std::move(tokens.front()));
  EXPECT_TRUE(limiter->tryAcquireUpTo(1).empty());
  clock.advance(1ms);
  EXPECT_TRUE(limiter->tryAcquireUpTo(0).empty());
  const auto waiting = limiter->stats().rateWaitingNanos;
  const auto rate = limiter->stats().sendRateMilli;
  clock.advance(1min);
  EXPECT_EQ(limiter->stats().rateWaitingNanos, waiting);
  lease->completeAndRelease(RPCRateLimiter::Outcome::kSuccess, 100'000'000);
  EXPECT_LT(limiter->stats().sendRateMilli, rate + 1'000);
}

TEST_F(RPCRateLimiterTest, canceledRateWaitDoesNotCancelAnotherDriver) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-wait-ownership", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  clock.advance(20ms);
  auto held = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(held.size(), 1);
  auto first = limiter->admitOrWait();
  auto second = limiter->admitOrWait();
  ASSERT_FALSE(first.admitted);
  ASSERT_FALSE(second.admitted);
  clock.advance(1ms);
  first.registration.reset();
  EXPECT_TRUE(first.wait.isReady());
  EXPECT_FALSE(second.wait.isReady());
  const auto waiting = limiter->stats().rateWaitingNanos;
  clock.advance(1ms);
  EXPECT_GT(limiter->stats().rateWaitingNanos, waiting);
  second.registration.reset();
  EXPECT_TRUE(second.wait.isReady());
  const auto afterCancel = limiter->stats().rateWaitingNanos;
  auto later = limiter->admitOrWait();
  ASSERT_FALSE(later.admitted);
  later.registration.reset();
  clock.advance(1min);
  EXPECT_EQ(limiter->stats().rateWaitingNanos, afterCancel);
}

TEST_F(RPCRateLimiterTest, ratePolicyDoesNotPaceOrdinaryRetryPolicy) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config policy{.adaptive = true, .ceiling = 100, .floor = 1};
  auto* ordinary =
      registry.getForPolicy("test.rate-mode-isolation", policy).limiter;
  policy.ratePolicy = RPCFlowControlPolicy::kCubic;
  policy.scheduleFixes = true;
  auto* controlled =
      registry.getForPolicy("test.rate-mode-isolation", policy).limiter;
  ASSERT_NE(ordinary, controlled);
  auto rejected = controlled->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  controlled->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  EXPECT_GT(controlled->stats().sendRateMilli, 0);
  EXPECT_EQ(ordinary->stats().sendRateMilli, 0);
  EXPECT_EQ(ordinary->tryAcquireUpTo(10).size(), 10);
}

TEST_F(RPCRateLimiterTest, windowRollbackDoesNotReuseActivatedModernPrimary) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config modern{
      .adaptive = true,
      .ceiling = 100,
      .floor = 1,
      .ratePolicy = RPCFlowControlPolicy::kCubic,
      .scheduleFixes = true};
  const auto primary =
      registry.getForPolicy("test.modern-primary-rollback", modern);
  EXPECT_FALSE(primary.policyScoped);
  auto rejected = primary.limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  primary.limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  EXPECT_GT(primary.limiter->stats().sendRateMilli, 0);

  auto window = modern;
  window.ratePolicy = RPCFlowControlPolicy::kWindowAimd;
  window.scheduleFixes = false;
  const auto rollback =
      registry.getForPolicy("test.modern-primary-rollback", window);
  ASSERT_NE(primary.limiter, rollback.limiter);
  EXPECT_TRUE(rollback.policyScoped);
  EXPECT_EQ(
      rollback.limiter->stats().ratePolicy, RPCFlowControlPolicy::kWindowAimd);
  EXPECT_FALSE(rollback.limiter->stats().scheduleFixes);
  EXPECT_EQ(rollback.limiter->stats().sendRateMilli, 0);
  EXPECT_EQ(rollback.limiter->tryAcquireUpTo(10).size(), 10);
}

TEST_F(RPCRateLimiterTest, ordinaryTemporaryAdmissionKeepsWaitRegistered) {
  auto limiter =
      std::make_shared<RPCRateLimiter>("test.ordinary-admission-future");
  limiter->configure({.ceiling = 1});
  auto held = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(held.size(), 1);
  auto wait = limiter->admitOrWait().wait;
  EXPECT_FALSE(wait.isReady());
  held.clear();
  EXPECT_TRUE(wait.isReady());
}

TEST_F(RPCRateLimiterTest, scheduleOnlyPolicyKeepsTemporaryWaitRegistered) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.scheduler-only-temporary", clock);
  limiter->configure(
      {.adaptive = true, .ceiling = 1, .floor = 0.01, .scheduleFixes = true});
  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  auto wait = limiter->admitOrWait().wait;
  EXPECT_FALSE(wait.isReady());
  clock.advance(2'100ms);
  EXPECT_TRUE(wait.isReady());
  EXPECT_EQ(limiter->stats().ratePolicy, RPCFlowControlPolicy::kWindowAimd);
  EXPECT_TRUE(limiter->stats().scheduleFixes);
}

TEST_F(
    RPCRateLimiterTest,
    fixedConcurrencyDoesNotMergeRateAndOrdinaryPolicies) {
  auto& registry = RPCRateLimiterRegistry::global();
  RPCRateLimiter::Config policy{.adaptive = true, .ceiling = 1, .floor = 1};
  auto* ordinary =
      registry.getForPolicy("test.fixed-rate-policy", policy).limiter;
  policy.ratePolicy = RPCFlowControlPolicy::kCubic;
  policy.scheduleFixes = true;
  auto* controlled =
      registry.getForPolicy("test.fixed-rate-policy", policy).limiter;
  ASSERT_NE(ordinary, controlled);
  auto rejected = controlled->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  controlled->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  EXPECT_EQ(controlled->stats().capacity, 1);
  EXPECT_GT(controlled->stats().sendRateMilli, 0);
  EXPECT_EQ(ordinary->stats().sendRateMilli, 0);
  EXPECT_EQ(ordinary->tryAcquireUpTo(1).size(), 1);
}

TEST_F(RPCRateLimiterTest, rateOverloadCoalescesOldAdmissionWave) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-wave", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 100,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto tokens = limiter->tryAcquireUpTo(20);
  ASSERT_EQ(tokens.size(), 20);
  for (auto& token : tokens) {
    limiter->makeLease(std::move(token))
        ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
    clock.advance(1s);
  }
  EXPECT_EQ(limiter->stats().overloadAttempts, 20);
  EXPECT_EQ(limiter->stats().rateDecreases, 1);
}

TEST_F(RPCRateLimiterTest, currentEpochOverloadInSameBucketPreservesPacing) {
  for (const auto policy :
       {RPCFlowControlPolicy::kRateAimd, RPCFlowControlPolicy::kCubic}) {
    ManualPacingClock clock;
    auto limiter = makePacingLimiter("test.current-epoch-coalescing", clock);
    limiter->configure(
        {.adaptive = true,
         .ceiling = 100,
         .floor = 1,
         .ratePolicy = policy,
         .scheduleFixes = true});
    auto first = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(first.size(), 1);
    const auto originalEpoch = first.front().overloadEpoch();
    limiter->makeLease(std::move(first.front()))
        ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
    const auto initial = limiter->stats();
    ASSERT_GT(initial.sendRateMilli, 0);
    ASSERT_GT(initial.nextPermitDelayNanos, 0);
    ASSERT_LT(initial.nextPermitDelayNanos, 500'000'000);
    clock.advanceClockOnly(
        std::chrono::nanoseconds(initial.nextPermitDelayNanos));

    auto current = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(current.size(), 1);
    ASSERT_NE(current.front().overloadEpoch(), originalEpoch);
    const auto before = limiter->stats();
    const auto completion =
        limiter->makeLease(std::move(current.front()))
            ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
    const auto after = limiter->stats();
    EXPECT_TRUE(completion.sharedOverloadHandled);
    EXPECT_EQ(after.rateDecreases, before.rateDecreases);
    EXPECT_EQ(after.sendRateMilli, before.sendRateMilli);
    EXPECT_GT(after.sendRateMilli, 0);
    EXPECT_EQ(after.nextPermitDelayNanos, before.nextPermitDelayNanos);
    EXPECT_EQ(limiter->available(), 0);
    EXPECT_TRUE(limiter->tryAcquireUpTo(10).empty());

    ASSERT_GT(after.nextPermitDelayNanos, 0);
    clock.advanceClockOnly(
        std::chrono::nanoseconds(after.nextPermitDelayNanos - 1));
    EXPECT_TRUE(limiter->tryAcquireUpTo(10).empty());
    clock.advanceClockOnly(std::chrono::nanoseconds(1));
    EXPECT_EQ(limiter->tryAcquireUpTo(10).size(), 1);
    EXPECT_TRUE(limiter->tryAcquireUpTo(10).empty());
  }
}

TEST_F(RPCRateLimiterTest, rateAndFractionalPacingUseStricterDeadline) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.combined-pacing", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 2,
       .floor = 0.01,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  for (int i = 0; i < 10; ++i) {
    clock.advance(1min);
    auto rejected = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(rejected.size(), 1);
    limiter->makeLease(std::move(rejected.front()))
        ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  }
  EXPECT_THROW(limiter->acquire(), std::exception);
  for (int i = 0; i < 2; ++i) {
    clock.advance(1min);
    auto terminal = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(terminal.size(), 1);
    limiter->makeLease(std::move(terminal.front()))
        ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, false);
  }
  ASSERT_EQ(limiter->stats().limitMilli, 500);
  clock.advance(3s);
  EXPECT_TRUE(limiter->tryAcquireUpTo(1).empty());
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  clock.advance(20s);
  EXPECT_TRUE(wait.wait.isReady());
  EXPECT_EQ(limiter->tryAcquireUpTo(1).size(), 1);
}

TEST_F(RPCRateLimiterTest, finiteRateRecoveryIncludesBootstrapAndTail) {
  for (const bool mixedCost : {false, true}) {
    for (const bool changingQuota : {false, true}) {
      const auto ordinary = runFiniteRateSimulation(
          {.mixedCost = mixedCost, .changingQuota = changingQuota});
      const auto cubic = runFiniteRateSimulation(
          {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
           .mixedCost = mixedCost,
           .changingQuota = changingQuota});
      const auto aimd = runFiniteRateSimulation(
          {.algorithm = RPCRateLimiter::RateAlgorithm::kRateAimd,
           .mixedCost = mixedCost,
           .changingQuota = changingQuota});
      LOG(INFO) << "RPC finite rate simulation mixedCost=" << mixedCost
                << " changingQuota=" << changingQuota
                << " ordinary=" << ordinary.elapsedMs << "/"
                << ordinary.attempts << " cubic=" << cubic.elapsedMs << "/"
                << cubic.attempts << " rateAimd=" << aimd.elapsedMs << "/"
                << aimd.attempts;
      EXPECT_EQ(ordinary.completed, 1'000);
      EXPECT_EQ(cubic.completed, 1'000);
      EXPECT_EQ(aimd.completed, 1'000);
      EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
      EXPECT_LE(
          cubic.attempts - cubic.completed,
          (ordinary.attempts - ordinary.completed) / 2);
    }
  }
}

TEST_F(RPCRateLimiterTest, finiteRateRecoveryWithSlowResponsesAndRisingQuota) {
  for (const int64_t responseMs : {100, 1'000}) {
    const auto rows = responseMs == 1'000 ? 200 : 1'000;
    const auto ordinary = runFiniteRateSimulation(
        {.changingQuota = true,
         .rows = rows,
         .responseMs = responseMs,
         .growingQuota = true,
         .changeAtMs = 3'000});
    const auto cubic = runFiniteRateSimulation(
        {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
         .changingQuota = true,
         .rows = rows,
         .responseMs = responseMs,
         .growingQuota = true,
         .changeAtMs = 3'000});
    const auto aimd = runFiniteRateSimulation(
        {.algorithm = RPCRateLimiter::RateAlgorithm::kRateAimd,
         .changingQuota = true,
         .rows = rows,
         .responseMs = responseMs,
         .growingQuota = true,
         .changeAtMs = 3'000});
    LOG(INFO) << "RPC rising quota simulation responseMs=" << responseMs
              << " ordinary=" << ordinary.elapsedMs << "/" << ordinary.attempts
              << " cubic=" << cubic.elapsedMs << "/" << cubic.attempts
              << " rateAimd=" << aimd.elapsedMs << "/" << aimd.attempts;
    EXPECT_EQ(ordinary.completed, rows);
    EXPECT_EQ(cubic.completed, rows);
    EXPECT_EQ(aimd.completed, rows);
    EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
    EXPECT_LE(
        cubic.attempts - cubic.completed,
        (ordinary.attempts - ordinary.completed) / 2);
  }
}

TEST_F(RPCRateLimiterTest, isolatedRejectionDoesNotSlowShortJobs) {
  for (const int64_t rows : {1, 20}) {
    const auto ordinary = runFiniteRateSimulation(
        {.rows = rows, .responseMs = 1'000, .rejectFirst = true});
    const auto cubic = runFiniteRateSimulation(
        {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
         .rows = rows,
         .responseMs = 1'000,
         .rejectFirst = true});
    LOG(INFO) << "RPC isolated rejection simulation rows=" << rows
              << " ordinary=" << ordinary.elapsedMs << "/" << ordinary.attempts
              << " cubic=" << cubic.elapsedMs << "/" << cubic.attempts;
    EXPECT_EQ(ordinary.completed, rows);
    EXPECT_EQ(cubic.completed, rows);
    EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
    EXPECT_EQ(cubic.attempts, rows + 1);
  }
  const auto ordinary = runFiniteRateSimulation(
      {.rows = 20, .responseMs = 10'000, .rejectFirst = true});
  const auto cubic = runFiniteRateSimulation(
      {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
       .rows = 20,
       .responseMs = 10'000,
       .rejectFirst = true});
  LOG(INFO) << "RPC ten second response simulation ordinary="
            << ordinary.elapsedMs << "/" << ordinary.attempts
            << " cubic=" << cubic.elapsedMs << "/" << cubic.attempts;
  EXPECT_EQ(ordinary.completed, 20);
  EXPECT_EQ(cubic.completed, 20);
  EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
}

TEST_F(RPCRateLimiterTest, longServiceCallsRetainFiniteJobThroughput) {
  const auto ordinary =
      runFiniteRateSimulation({.rows = 20, .responseMs = 10'000});
  const auto cubic = runFiniteRateSimulation(
      {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
       .rows = 20,
       .responseMs = 10'000});
  LOG(INFO) << "RPC long service quota simulation ordinary="
            << ordinary.elapsedMs << "/" << ordinary.attempts
            << " cubic=" << cubic.elapsedMs << "/" << cubic.attempts;
  EXPECT_EQ(ordinary.completed, 20);
  EXPECT_EQ(cubic.completed, 20);
  EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
  EXPECT_LE(
      cubic.attempts - cubic.completed,
      (ordinary.attempts - ordinary.completed) / 2);
}

TEST_F(RPCRateLimiterTest, independentWorkersShareOneRateQuota) {
  const auto ordinary = runFiniteRateSimulation({.workers = 2});
  const auto cubic = runFiniteRateSimulation(
      {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic, .workers = 2});
  LOG(INFO) << "RPC two worker rate simulation ordinary=" << ordinary.elapsedMs
            << "/" << ordinary.attempts << " cubic=" << cubic.elapsedMs << "/"
            << cubic.attempts;
  EXPECT_EQ(ordinary.completed, 1'000);
  EXPECT_EQ(cubic.completed, 1'000);
  EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
  EXPECT_LE(
      cubic.attempts - cubic.completed,
      (ordinary.attempts - ordinary.completed) / 2);
}

TEST_F(RPCRateLimiterTest, irreducibleRejectionFloorRetainsLiveness) {
  // Every fifth attempt fails independently of load. This is a robustness
  // model, not a claim about MetaGen's quota or an attainable rejection target.
  const auto ordinary =
      runFiniteRateSimulation({.rows = 100, .independentRejections = true});
  const auto cubic = runFiniteRateSimulation(
      {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
       .rows = 100,
       .independentRejections = true});
  LOG(INFO) << "RPC irreducible rejection simulation ordinary="
            << ordinary.elapsedMs << "/" << ordinary.attempts << "/"
            << ordinary.completed << " cubic=" << cubic.elapsedMs << "/"
            << cubic.attempts << "/" << cubic.completed;
  EXPECT_EQ(ordinary.completed, 100);
  EXPECT_EQ(cubic.completed, 100);
  EXPECT_GE(cubic.attempts, cubic.completed);
  EXPECT_LT(cubic.elapsedMs, 120'000);
}

TEST_F(RPCRateLimiterTest, finiteRateControlWithJitteredTransportRetries) {
  // The jitter draw depends on row and retry identities, so both modes use the
  // same retry schedule even when their dispatch orders differ.
  const auto ordinary = runFiniteRateSimulation(
      {.mixedCost = true,
       .rows = 2'000,
       .concurrency = 100,
       .equalJitter = true});
  const auto cubic = runFiniteRateSimulation(
      {.algorithm = RPCRateLimiter::RateAlgorithm::kCubic,
       .mixedCost = true,
       .rows = 2'000,
       .concurrency = 100,
       .equalJitter = true});
  const auto aimd = runFiniteRateSimulation(
      {.algorithm = RPCRateLimiter::RateAlgorithm::kRateAimd,
       .mixedCost = true,
       .rows = 2'000,
       .concurrency = 100,
       .equalJitter = true});
  LOG(INFO) << "RPC jittered retry simulation ordinary=" << ordinary.elapsedMs
            << "/" << ordinary.attempts << "/" << ordinary.completed
            << " cubic=" << cubic.elapsedMs << "/" << cubic.attempts << "/"
            << cubic.completed << " rateAimd=" << aimd.elapsedMs << "/"
            << aimd.attempts << "/" << aimd.completed;
  EXPECT_EQ(ordinary.completed, 2'000);
  EXPECT_EQ(cubic.completed, 2'000);
  EXPECT_EQ(aimd.completed, 2'000);
  EXPECT_LE(cubic.elapsedMs, ordinary.elapsedMs);
  EXPECT_LE(
      cubic.attempts - cubic.completed,
      (ordinary.attempts - ordinary.completed) / 2);
}

TEST_F(RPCRateLimiterTest, rateControlAdaptsToFixedWindowQuotaChanges) {
  for (const int64_t responseMs : {150, 3'000}) {
    const RateSimulationConfig ordinaryConfig{
        .changingQuota = true,
        .rows = 160,
        .responseMs = responseMs,
        .equalJitter = true,
        .quotaPerSecond = 8,
        .quotaAfterChange = 4,
        .quotaWindowMs = 1'000};
    auto controlledConfig = ordinaryConfig;
    controlledConfig.algorithm = RPCRateLimiter::RateAlgorithm::kCubic;
    const auto ordinary = runFiniteRateSimulation(ordinaryConfig);
    const auto controlled = runFiniteRateSimulation(controlledConfig);
    const auto ordinaryRejected = ordinary.attempts - ordinary.completed;
    const auto controlledRejected = controlled.attempts - controlled.completed;
    LOG(INFO) << "RPC fixed-window quota simulation responseMs=" << responseMs
              << " ordinary=" << ordinary.elapsedMs << "/" << ordinaryRejected
              << " controlled=" << controlled.elapsedMs << "/"
              << controlledRejected << " cuts=" << controlled.rateCuts.size();
    EXPECT_EQ(ordinary.completed, 160);
    EXPECT_EQ(controlled.completed, 160);
    EXPECT_LE(ordinary.peakPending, 20);
    EXPECT_LE(controlled.peakPending, 20);
    EXPECT_LE(controlled.elapsedMs, ordinary.elapsedMs);
    EXPECT_LE(2 * controlledRejected, ordinaryRejected);
    if (controlled.elapsedMs > ordinary.elapsedMs ||
        2 * controlledRejected > ordinaryRejected) {
      for (const auto& cut : controlled.rateCuts) {
        LOG(INFO) << "RPC fixed-window quota cut tMs=" << cut.atMs
                  << " admissionBucket=" << cut.admissionBucket
                  << " rateMilli=" << cut.sendRateMilli;
      }
    }
  }
}

TEST_F(RPCRateLimiterTest, ratePacingPreservesDeadlineOnSuccess) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-pacing-feedback", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  limiter->onSuccessForEpoch(0, 40, 0);
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(/*hardOverload=*/true, /*handedOff=*/true);

  clock.advance(20ms);
  auto admitted = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(admitted.size(), 1);
  const auto epoch = admitted.front().overloadEpoch();
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);

  clock.advance(1ms);
  limiter->onSuccessForEpoch(epoch, 1, 0);
  EXPECT_FALSE(wait.wait.isReady());
  EXPECT_EQ(limiter->available(), 0);
  EXPECT_TRUE(limiter->tryAcquireUpTo(1).empty());
}

TEST_F(RPCRateLimiterTest, ratePacingWakesWhileOtherCallsRemainInFlight) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.rate-pacing-delayed", clock);
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  limiter->onSuccessForEpoch(0, 40, 0);
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(/*hardOverload=*/true, /*handedOff=*/true);

  clock.advance(20ms);
  auto first = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(first.size(), 1);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  clock.advance(1ms);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(50ms);
  EXPECT_TRUE(wait.wait.isReady());
  auto second = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(second.size(), 1);
  EXPECT_EQ(limiter->stats().pending, 2);
}

TEST_F(RPCRateLimiterTest, rateGrantRearmsExistingWaiterAfterStaleTimer) {
  FailOncePacingClock clock;
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.rate-grant-existing-waiter",
      [&clock] { return clock.now(); },
      &clock,
      [] { return 1.0; });
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  clock.advance(20ms);
  auto first = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(first.size(), 1);
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  // The first timer is deliberately held while another driver takes its due
  // permit. The second grant must replace that now-stale timer.
  clock.advance(20ms);
  auto second = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(second.size(), 1);
  clock.advance(10ms);
  EXPECT_TRUE(wait.wait.isReady());
  EXPECT_EQ(limiter->stats().pending, 2);
  clock.failFirstTimer();
}

TEST_F(RPCRateLimiterTest, admissionCrossingPermitDeadlineStillGetsWakeup) {
  ManualPacingClock clock;
  bool crossDeadline{false};
  int clockReads{0};
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.pacing-deadline-crossing",
      [&] {
        if (crossDeadline && ++clockReads == 2) {
          crossDeadline = false;
          clock.advanceClockOnly(1s);
        }
        return clock.now();
      },
      clock.timekeeper(),
      [] { return 1.0; });
  limiter->configure(
      {.adaptive = true,
       .ceiling = 1'000,
       .floor = 1,
       .ratePolicy = RPCFlowControlPolicy::kCubic,
       .scheduleFixes = true});
  auto rejected = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(rejected.size(), 1);
  limiter->makeLease(std::move(rejected.front()))
      ->completeAndRelease(RPCRateLimiter::Outcome::kOverload, 0, true);
  clock.advance(20ms);
  auto first = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(first.size(), 1);
  crossDeadline = true;
  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  EXPECT_TRUE(limiter->stats().admissionTimerArmed);
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
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
      {.adaptive = true, .ceiling = 1, .floor = 0.01, .scheduleFixes = true});
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
