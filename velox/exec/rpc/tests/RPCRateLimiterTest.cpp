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
#include <stdexcept>
#include <system_error>
#include <thread>
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
  void advance(std::chrono::duration<Rep, Period> duration) {
    now_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        duration);
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
      clock.timekeeper());
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
// There is one call site now, so that shape cannot be written to fail this.
TEST_F(RPCRateLimiterTest, concurrentInitializersDoNotMixConfigurations) {
  constexpr int kThreads = 16;
  auto& limiter = limiterFor("test.backend");

  // Each thread offers an internally consistent policy; the pairs are chosen
  // so a mixture is detectable.
  struct Policy {
    int64_t ceiling;
    double floor;
  };
  auto policyFor = [](int i) -> Policy {
    return {static_cast<int64_t>(100 * (i + 1)), static_cast<double>(i + 1)};
  };

  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i]() {
      const auto policy = policyFor(i);
      while (!go.load(std::memory_order_relaxed)) {
      }
      limiter.initializeOnce([policy](RPCRateLimiter::Config& config) {
        config.ceiling = policy.ceiling;
        config.floor = policy.floor;
      });
    });
  }
  go.store(true);
  for (auto& t : threads) {
    t.join();
  }

  // Whatever landed must be exactly one thread's pair, not a blend.
  const auto config = limiter.config();
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

TEST_F(RPCRateLimiterTest, timerSchedulingFailureFailsEveryWaiter) {
  ThrowingTimekeeper timekeeper;
  auto limiter = std::make_shared<RPCRateLimiter>(
      "test.throwing-timekeeper",
      [] { return std::chrono::steady_clock::time_point{}; },
      &timekeeper);
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
      "test.recovering-timekeeper", [&clock] { return clock.now(); }, &clock);
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

TEST_F(RPCRateLimiterTest, fractionalLimitBacksOffAndRecoversToOne) {
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

  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 500);
  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  EXPECT_EQ(limiter->stats().limitMilli, 1'000);
  EXPECT_EQ(limiter->stats().capacity, 1);
  EXPECT_EQ(limiter->stats().lowWaterLimitMilli, 250);
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

  limiter->onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  ASSERT_EQ(limiter->stats().limitMilli, 500);
  clock.advance(2s);
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

TEST_F(RPCRateLimiterTest, pacingSupportsOneRequestPerHundredSeconds) {
  ManualPacingClock clock;
  auto limiter = makePacingLimiter("test.paced.backend", clock);
  enableFractionalLimit(*limiter, 0.01);

  limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  for (int i = 0; i < 7; ++i) {
    limiter->onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  ASSERT_EQ(limiter->stats().limitMilli, 10);

  auto wait = limiter->admitOrWait();
  ASSERT_FALSE(wait.admitted);
  clock.advance(99'999ms);
  EXPECT_FALSE(wait.wait.isReady());
  clock.advance(1ms);
  EXPECT_TRUE(wait.wait.isReady());
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
