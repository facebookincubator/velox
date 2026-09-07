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

/// RPCRateLimiterTest - Tests for per-tier RPC limiter control.
///
/// RPCRateLimiter provides per-tier concurrency limits with FIFO waiter
/// notification and RAII token-based slot management.
///
/// Tests cover:
/// - acquireAndRelease: Token acquire increments, destruction decrements.
/// - backpressureWhenAtLimit: admitOrWait waits once at capacity.
/// - backpressureReliefOnRelease: Waiter notified when token released.
/// - fifoWaiterNotification: Multiple waiters notified in FIFO order.
/// - perTierIsolation: Different tiers have independent limits.
/// - perTierMaxPending: a per-tier ceiling overrides the process default.
/// - defaultMaxPending: the process default applies to unconfigured tiers.
/// - tokenMoveSemantics: Move constructor/assignment transfer ownership.
/// - testingResetClearsStateInPlace: Reset restores defaults and zeroes
///   pending without destroying a backend an outstanding token points at.
/// - adaptiveIsPerTier: Two tiers adapt independently.
/// - adaptiveFactorsAreIndependent: Each tier uses its own decrease factor.

#include "velox/exec/rpc/RPCRateLimiter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace facebook::velox::exec::rpc {
namespace {

// RPCRateLimiter takes a whole Config per tier. These helpers amend one field
// group at a time so each test states only the setting it cares about.
RPCRateLimiter& limiterFor(const std::string& tierKey) {
  return RPCRateLimiterRegistry::global().get(tierKey);
}

void setCeiling(const std::string& tierKey, int64_t ceiling) {
  auto& limiter = limiterFor(tierKey);
  auto config = limiter.config();
  config.ceiling = ceiling;
  limiter.configure(config);
}

void setAdaptive(
    const std::string& tierKey,
    bool enabled,
    int64_t floor,
    double decreaseFactor) {
  auto& limiter = limiterFor(tierKey);
  auto config = limiter.config();
  config.adaptive = enabled;
  config.floor = floor;
  config.decreaseFactor = decreaseFactor;
  limiter.configure(config);
}

class RPCRateLimiterTest : public testing::Test {
 protected:
  void SetUp() override {
    RPCRateLimiterRegistry::global().testingReset();
  }

  void TearDown() override {
    RPCRateLimiterRegistry::global().testingReset();
  }
};

TEST_F(RPCRateLimiterTest, acquireAndRelease) {
  const std::string tier = "test.tier";
  EXPECT_EQ(limiterFor(tier).stats().pending, 0);

  {
    auto token = limiterFor(tier).acquire();
    EXPECT_EQ(limiterFor(tier).stats().pending, 1);
  }
  // Token destroyed — count should be back to 0.
  EXPECT_EQ(limiterFor(tier).stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, multipleAcquires) {
  const std::string tier = "test.tier";

  auto token1 = limiterFor(tier).acquire();
  auto token2 = limiterFor(tier).acquire();
  auto token3 = limiterFor(tier).acquire();
  EXPECT_EQ(limiterFor(tier).stats().pending, 3);
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
  auto& limiter = limiterFor("test.tier");

  // Each thread offers an internally consistent policy; the pairs are chosen
  // so a mixture is detectable.
  struct Policy {
    int64_t ceiling;
    int64_t floor;
  };
  auto policyFor = [](int i) -> Policy {
    return {static_cast<int64_t>(100 * (i + 1)), static_cast<int64_t>(i + 1)};
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
  const std::string tier = "test.tier";
  setCeiling(tier, 1);

  // Room: admitted, and nothing enrolled to be woken later.
  auto free = limiterFor(tier).admitOrWait();
  EXPECT_TRUE(free.admitted);

  auto token = limiterFor(tier).acquire();

  // Full: not admitted, and the wait is fulfilled by the release.
  auto parked = limiterFor(tier).admitOrWait();
  ASSERT_FALSE(parked.admitted);
  EXPECT_FALSE(parked.wait.isReady());
  token = RPCRateLimiter::Token();
  EXPECT_TRUE(parked.wait.isReady());
}

TEST_F(RPCRateLimiterTest, backpressureWhenAtLimit) {
  const std::string tier = "test.tier";
  setCeiling(tier, 2);

  auto token1 = limiterFor(tier).acquire();
  // 1 pending, limit 2 -- capacity is free, so the future comes back ready.
  EXPECT_TRUE(limiterFor(tier).admitOrWait().admitted);

  auto token2 = limiterFor(tier).acquire();
  // 2 pending, limit 2 -- at the limit, so the caller genuinely waits.
  auto admission = limiterFor(tier).admitOrWait();
  EXPECT_FALSE(admission.wait.isReady());
}

TEST_F(RPCRateLimiterTest, backpressureReliefOnRelease) {
  const std::string tier = "test.tier";
  setCeiling(tier, 1);

  auto token1 = limiterFor(tier).acquire();

  // At limit — should block.
  auto admission = limiterFor(tier).admitOrWait();
  EXPECT_FALSE(admission.wait.isReady());

  // Release the token — waiter should be notified.
  token1 = RPCRateLimiter::Token();
  EXPECT_TRUE(admission.wait.isReady());
  EXPECT_EQ(limiterFor(tier).stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, fifoWaiterNotification) {
  const std::string tier = "test.tier";
  setCeiling(tier, 1);

  auto token = limiterFor(tier).acquire();

  // Two waiters enqueue while at limit.
  auto admission1 = limiterFor(tier).admitOrWait();
  auto admission2 = limiterFor(tier).admitOrWait();
  ASSERT_FALSE(admission1.wait.isReady());
  ASSERT_FALSE(admission2.wait.isReady());

  // Release token — only first waiter should be notified (FIFO).
  token = RPCRateLimiter::Token();
  EXPECT_TRUE(admission1.wait.isReady());
  EXPECT_FALSE(admission2.wait.isReady());

  // Acquire and release again — second waiter should be notified.
  {
    auto token2 = limiterFor(tier).acquire();
  }
  EXPECT_TRUE(admission2.wait.isReady());
}

TEST_F(RPCRateLimiterTest, perTierIsolation) {
  const std::string tier1 = "tier.one";
  const std::string tier2 = "tier.two";
  setCeiling(tier1, 1);
  setCeiling(tier2, 1);

  auto tokenA = limiterFor(tier1).acquire();
  EXPECT_EQ(limiterFor(tier1).stats().pending, 1);
  EXPECT_EQ(limiterFor(tier2).stats().pending, 0);

  // tier1 at limit, tier2 not.
  EXPECT_FALSE(limiterFor(tier1).admitOrWait().admitted);
  EXPECT_TRUE(limiterFor(tier2).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, perTierMaxPending) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setDefaultCapacity(10);
  setCeiling(tier, 2);

  auto token1 = limiterFor(tier).acquire();
  auto token2 = limiterFor(tier).acquire();

  // This backend's own limit of 2 applies, not the global default of 10.
  EXPECT_FALSE(limiterFor(tier).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, defaultMaxPending) {
  RPCRateLimiter::setDefaultCapacity(2);
  const std::string tier = "test.default.tier";

  auto token1 = limiterFor(tier).acquire();
  EXPECT_TRUE(limiterFor(tier).admitOrWait().admitted);

  auto token2 = limiterFor(tier).acquire();
  // Global default of 2 reached.
  EXPECT_FALSE(limiterFor(tier).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, tokenMoveConstructor) {
  const std::string tier = "test.tier";

  auto token1 = limiterFor(tier).acquire();
  EXPECT_EQ(limiterFor(tier).stats().pending, 1);

  // Move construct — ownership transfers, count stays 1.
  auto token2 = std::move(token1);
  EXPECT_EQ(limiterFor(tier).stats().pending, 1);

  // Destroy moved-from token — no effect.
  // (token1 is already moved-from, but let it go out of scope naturally)
}

TEST_F(RPCRateLimiterTest, tokenMoveAssignment) {
  const std::string tier1 = "tier.one";
  const std::string tier2 = "tier.two";

  auto token1 = limiterFor(tier1).acquire();
  auto token2 = limiterFor(tier2).acquire();
  EXPECT_EQ(limiterFor(tier1).stats().pending, 1);
  EXPECT_EQ(limiterFor(tier2).stats().pending, 1);

  // Move-assign token2 into token1 — old token1 (tier1) released.
  token1 = std::move(token2);
  EXPECT_EQ(limiterFor(tier1).stats().pending, 0);
  EXPECT_EQ(limiterFor(tier2).stats().pending, 1);
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
// A backend's identity is its tier key, and each transport composes that key
// from whatever distinguishes one deployment from another -- IPNext uses
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
  auto& limiter = limiterFor("test.tier");
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
  auto& limiter = limiterFor("test.tier");
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
  auto& limiter = limiterFor("test.tier");
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
  const std::string tier = "test.tier";
  auto& limiter = limiterFor(tier);

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
  const std::string tier = "test.tier";
  RPCRateLimiter::setDefaultCapacity(5);
  setCeiling(tier, 3);
  auto token = limiterFor(tier).acquire();

  RPCRateLimiterRegistry::global().testingReset();

  EXPECT_EQ(RPCRateLimiter::defaultCapacity(), 20);
  EXPECT_EQ(limiterFor(tier).stats().pending, 0);
}

TEST_F(RPCRateLimiterTest, adaptiveDisabledIsNoop) {
  const std::string tier = "test.tier";
  setCeiling(tier, 10);
  // Adaptive off (default): the overload signal must not shrink the cap.
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiterFor(tier).stats().capacity, 10);
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000);
  EXPECT_EQ(limiterFor(tier).stats().capacity, 10);
}

TEST_F(RPCRateLimiterTest, adaptiveMultiplicativeDecrease) {
  const std::string tier = "test.tier";
  setCeiling(tier, 16);
  setAdaptive(tier, /*enabled*/ true, /*floor*/ 1, /*decreaseFactor*/ 0.5);

  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiterFor(tier).stats().capacity, 8);
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  EXPECT_EQ(limiterFor(tier).stats().capacity, 4);
}

TEST_F(RPCRateLimiterTest, adaptiveFlooredAtMinLimit) {
  const std::string tier = "test.tier";
  setCeiling(tier, 16);
  setAdaptive(tier, true, /*floor*/ 4, 0.5);

  for (int i = 0; i < 10; ++i) {
    limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  // 16 -> 8 -> 4, then pinned at the floor.
  EXPECT_EQ(limiterFor(tier).stats().capacity, 4);
}

TEST_F(RPCRateLimiterTest, adaptiveRecoveryScalesWithSuccesses) {
  const std::string tier = "test.tier";
  setCeiling(tier, 16);
  setAdaptive(tier, true, 1, 0.5);

  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kOverload, 0); // 16 -> 8
  ASSERT_EQ(limiterFor(tier).stats().capacity, 8);

  // A tiny drain recovers by the +1 floor (step = max(1, 1/8)).
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1);
  EXPECT_EQ(limiterFor(tier).stats().capacity, 9);

  // A large drain recovers proportionally (step = successes/cap) and, on
  // reaching the ceiling, clears the adaptive state so the static cap governs.
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000);
  EXPECT_EQ(limiterFor(tier).stats().capacity, 16);
}

TEST_F(RPCRateLimiterTest, adaptiveShrinkReducesAdmission) {
  const std::string tier = "test.tier";
  setCeiling(tier, 4);
  setAdaptive(tier, true, 1, 0.5);

  limiterFor(tier).onOutcome(
      RPCRateLimiter::Outcome::kOverload, 0); // cap 4 -> 2
  ASSERT_EQ(limiterFor(tier).stats().capacity, 2);

  auto token1 = limiterFor(tier).acquire();
  EXPECT_TRUE(limiterFor(tier).admitOrWait().admitted);
  auto token2 = limiterFor(tier).acquire();
  // At the shrunk cap of 2 (not the static 4), backpressure kicks in.
  EXPECT_FALSE(limiterFor(tier).admitOrWait().admitted);
}

TEST_F(RPCRateLimiterTest, noBackpressureBelowLimit) {
  const std::string tier = "test.tier";
  setCeiling(tier, 5);

  std::vector<RPCRateLimiter::Token> tokens;
  for (int i = 0; i < 4; ++i) {
    tokens.push_back(limiterFor(tier).acquire());
    EXPECT_TRUE(limiterFor(tier).admitOrWait().admitted);
  }
  EXPECT_EQ(limiterFor(tier).stats().pending, 4);
}

// A backend that never backed off has no low-water mark, and zero would be
// indistinguishable from "not recorded". Reporting the ceiling means every
// reading of the stat is a real capacity.
TEST_F(RPCRateLimiterTest, lowWaterReportsTheCeilingWhenNeverShrunk) {
  const std::string tier = "test.tier";
  setCeiling(tier, 64);
  setAdaptive(tier, /*enabled=*/true, /*floor=*/2, /*decreaseFactor=*/0.5);

  // Nothing has driven an overload, so capacity has never shrunk.
  EXPECT_EQ(limiterFor(tier).stats().lowWaterCapacity, 64);

  // Once it does shrink, the actual low-water value is reported instead.
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  const auto shrunk = limiterFor(tier).stats();
  EXPECT_LT(shrunk.lowWaterCapacity, 64);
  EXPECT_EQ(shrunk.lowWaterCapacity, shrunk.capacity);

  // Recovering above the low-water mark leaves the mark where it was.
  limiterFor(tier).onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'024);
  EXPECT_EQ(limiterFor(tier).stats().lowWaterCapacity, shrunk.lowWaterCapacity);
}

// Regression test for the defect that motivated one limiter per backend: the
// previous API configured the adaptive parameters process-globally, so a
// second backend's configuration silently reconfigured the first backend's
// limiter and both shrank together.
TEST_F(RPCRateLimiterTest, adaptiveIsPerTier) {
  const std::string adaptiveTier = "backend.adaptive";
  const std::string fixedTier = "backend.fixed";
  setCeiling(adaptiveTier, 16);
  setCeiling(fixedTier, 16);
  setAdaptive(adaptiveTier, true, /*floor*/ 1, /*decreaseFactor*/ 0.5);
  setAdaptive(fixedTier, false, /*floor*/ 1, /*decreaseFactor*/ 0.5);

  limiterFor(adaptiveTier)
      .onOutcome(RPCRateLimiter::Outcome::kOverload, /*units*/ 0);
  limiterFor(fixedTier).onOutcome(
      RPCRateLimiter::Outcome::kOverload, /*units*/ 0);

  EXPECT_EQ(limiterFor(adaptiveTier).stats().capacity, 8);
  EXPECT_EQ(limiterFor(fixedTier).stats().capacity, 16);
}

// Two adapting tiers shrink by their own decrease factors rather than sharing
// one process-global value.
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
