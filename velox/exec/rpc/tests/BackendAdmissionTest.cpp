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

/// BackendAdmissionTest - Tests for per-tier RPC admission control.
///
/// BackendAdmission provides per-tier concurrency limits with FIFO waiter
/// notification and RAII token-based slot management.
///
/// Tests cover:
/// - acquireAndRelease: Token acquire increments, destruction decrements.
/// - backpressureWhenAtLimit: waitForCapacity returns a future at capacity.
/// - backpressureReliefOnRelease: Waiter notified when token released.
/// - fifoWaiterNotification: Multiple waiters notified in FIFO order.
/// - perTierIsolation: Different tiers have independent limits.
/// - perTierMaxPending: a per-tier ceiling overrides the process default.
/// - defaultMaxPending: the process default applies to unconfigured tiers.
/// - tokenMoveSemantics: Move constructor/assignment transfer ownership.
/// - testingResetAllState: Reset clears tier state and restores defaults.
/// - adaptiveIsPerTier: Two tiers adapt independently.
/// - adaptiveFactorsAreIndependent: Each tier uses its own decrease factor.

#include "velox/exec/rpc/BackendAdmission.h"

#include <gtest/gtest.h>

#include <vector>

namespace facebook::velox::exec::rpc {
namespace {

// The old API set a tier's ceiling and the adaptive parameters through
// separate calls, the latter process-globally. BackendAdmission takes a whole
// Config per tier, so these helpers amend one field group at a time and keep
// each test's intent readable.
BackendAdmission& admissionFor(const std::string& backendKey) {
  return BackendRegistry::global().get(backendKey);
}

void setCeiling(const std::string& backendKey, int64_t ceiling) {
  auto& admission = admissionFor(backendKey);
  auto config = admission.config();
  config.ceiling = ceiling;
  admission.configure(config);
}

void setAdaptive(
    const std::string& backendKey,
    bool enabled,
    int64_t floor,
    double decreaseFactor) {
  auto& admission = admissionFor(backendKey);
  auto config = admission.config();
  config.adaptive = enabled;
  config.floor = floor;
  config.decreaseFactor = decreaseFactor;
  admission.configure(config);
}

class BackendAdmissionTest : public testing::Test {
 protected:
  void SetUp() override {
    BackendRegistry::global().testingReset();
  }

  void TearDown() override {
    BackendRegistry::global().testingReset();
  }
};

TEST_F(BackendAdmissionTest, acquireAndRelease) {
  const std::string tier = "test.tier";
  EXPECT_EQ(admissionFor(tier).stats().pending, 0);

  {
    auto token = admissionFor(tier).acquire();
    EXPECT_EQ(admissionFor(tier).stats().pending, 1);
  }
  // Token destroyed — count should be back to 0.
  EXPECT_EQ(admissionFor(tier).stats().pending, 0);
}

TEST_F(BackendAdmissionTest, multipleAcquires) {
  const std::string tier = "test.tier";

  auto token1 = admissionFor(tier).acquire();
  auto token2 = admissionFor(tier).acquire();
  auto token3 = admissionFor(tier).acquire();
  EXPECT_EQ(admissionFor(tier).stats().pending, 3);
}

TEST_F(BackendAdmissionTest, backpressureWhenAtLimit) {
  const std::string tier = "test.tier";
  setCeiling(tier, 2);

  auto token1 = admissionFor(tier).acquire();
  // 1 pending, limit 2 — no backpressure.
  EXPECT_FALSE(admissionFor(tier).waitForCapacity().has_value());

  auto token2 = admissionFor(tier).acquire();
  // 2 pending, limit 2 — at limit, should get backpressure.
  auto future = admissionFor(tier).waitForCapacity();
  EXPECT_TRUE(future.has_value());
}

TEST_F(BackendAdmissionTest, backpressureReliefOnRelease) {
  const std::string tier = "test.tier";
  setCeiling(tier, 1);

  auto token1 = admissionFor(tier).acquire();

  // At limit — should block.
  auto future = admissionFor(tier).waitForCapacity();
  ASSERT_TRUE(future.has_value());
  EXPECT_FALSE(future->isReady());

  // Release the token — waiter should be notified.
  token1 = BackendAdmission::Token();
  EXPECT_TRUE(future->isReady());
  EXPECT_EQ(admissionFor(tier).stats().pending, 0);
}

TEST_F(BackendAdmissionTest, fifoWaiterNotification) {
  const std::string tier = "test.tier";
  setCeiling(tier, 1);

  auto token = admissionFor(tier).acquire();

  // Two waiters enqueue while at limit.
  auto future1 = admissionFor(tier).waitForCapacity();
  auto future2 = admissionFor(tier).waitForCapacity();
  ASSERT_TRUE(future1.has_value());
  ASSERT_TRUE(future2.has_value());

  // Release token — only first waiter should be notified (FIFO).
  token = BackendAdmission::Token();
  EXPECT_TRUE(future1->isReady());
  EXPECT_FALSE(future2->isReady());

  // Acquire and release again — second waiter should be notified.
  {
    auto token2 = admissionFor(tier).acquire();
  }
  EXPECT_TRUE(future2->isReady());
}

TEST_F(BackendAdmissionTest, perTierIsolation) {
  const std::string tier1 = "tier.one";
  const std::string tier2 = "tier.two";
  setCeiling(tier1, 1);
  setCeiling(tier2, 1);

  auto tokenA = admissionFor(tier1).acquire();
  EXPECT_EQ(admissionFor(tier1).stats().pending, 1);
  EXPECT_EQ(admissionFor(tier2).stats().pending, 0);

  // tier1 at limit, tier2 not.
  EXPECT_TRUE(admissionFor(tier1).waitForCapacity().has_value());
  EXPECT_FALSE(admissionFor(tier2).waitForCapacity().has_value());
}

TEST_F(BackendAdmissionTest, perTierMaxPending) {
  const std::string tier = "test.tier";
  BackendAdmission::setDefaultCapacity(10);
  setCeiling(tier, 2);

  auto token1 = admissionFor(tier).acquire();
  auto token2 = admissionFor(tier).acquire();

  // Per-tier limit of 2 applies, not global default of 10.
  EXPECT_TRUE(admissionFor(tier).waitForCapacity().has_value());
}

TEST_F(BackendAdmissionTest, defaultMaxPending) {
  BackendAdmission::setDefaultCapacity(2);
  const std::string tier = "test.default.tier";

  auto token1 = admissionFor(tier).acquire();
  EXPECT_FALSE(admissionFor(tier).waitForCapacity().has_value());

  auto token2 = admissionFor(tier).acquire();
  // Global default of 2 reached.
  EXPECT_TRUE(admissionFor(tier).waitForCapacity().has_value());
}

TEST_F(BackendAdmissionTest, tokenMoveConstructor) {
  const std::string tier = "test.tier";

  auto token1 = admissionFor(tier).acquire();
  EXPECT_EQ(admissionFor(tier).stats().pending, 1);

  // Move construct — ownership transfers, count stays 1.
  auto token2 = std::move(token1);
  EXPECT_EQ(admissionFor(tier).stats().pending, 1);

  // Destroy moved-from token — no effect.
  // (token1 is already moved-from, but let it go out of scope naturally)
}

TEST_F(BackendAdmissionTest, tokenMoveAssignment) {
  const std::string tier1 = "tier.one";
  const std::string tier2 = "tier.two";

  auto token1 = admissionFor(tier1).acquire();
  auto token2 = admissionFor(tier2).acquire();
  EXPECT_EQ(admissionFor(tier1).stats().pending, 1);
  EXPECT_EQ(admissionFor(tier2).stats().pending, 1);

  // Move-assign token2 into token1 — old token1 (tier1) released.
  token1 = std::move(token2);
  EXPECT_EQ(admissionFor(tier1).stats().pending, 0);
  EXPECT_EQ(admissionFor(tier2).stats().pending, 1);
}

TEST_F(BackendAdmissionTest, testingResetAllState) {
  const std::string tier = "test.tier";
  BackendAdmission::setDefaultCapacity(5);
  setCeiling(tier, 3);
  auto token = admissionFor(tier).acquire();

  // Move the token out so it doesn't decrement during reset.
  // (In practice, testingResetAllState clears the tiers map,
  // so the token's destructor will create a new empty tier state.)

  BackendRegistry::global().testingReset();

  // Default restored to 20.
  EXPECT_EQ(BackendAdmission::defaultCapacity(), 20);
  // Tier state cleared — pending count is 0 for a fresh tier.
  EXPECT_EQ(admissionFor(tier).stats().pending, 0);
}

TEST_F(BackendAdmissionTest, adaptiveDisabledIsNoop) {
  const std::string tier = "test.tier";
  setCeiling(tier, 10);
  // Adaptive off (default): the overload signal must not shrink the cap.
  admissionFor(tier).onOutcome(BackendAdmission::Outcome::kOverload, 0);
  EXPECT_EQ(admissionFor(tier).stats().capacity, 10);
  admissionFor(tier).onOutcome(BackendAdmission::Outcome::kSuccess, 1000);
  EXPECT_EQ(admissionFor(tier).stats().capacity, 10);
}

TEST_F(BackendAdmissionTest, adaptiveMultiplicativeDecrease) {
  const std::string tier = "test.tier";
  setCeiling(tier, 16);
  setAdaptive(tier, /*enabled*/ true, /*floor*/ 1, /*decreaseFactor*/ 0.5);

  admissionFor(tier).onOutcome(BackendAdmission::Outcome::kOverload, 0);
  EXPECT_EQ(admissionFor(tier).stats().capacity, 8);
  admissionFor(tier).onOutcome(BackendAdmission::Outcome::kOverload, 0);
  EXPECT_EQ(admissionFor(tier).stats().capacity, 4);
}

TEST_F(BackendAdmissionTest, adaptiveFlooredAtMinLimit) {
  const std::string tier = "test.tier";
  setCeiling(tier, 16);
  setAdaptive(tier, true, /*floor*/ 4, 0.5);

  for (int i = 0; i < 10; ++i) {
    admissionFor(tier).onOutcome(BackendAdmission::Outcome::kOverload, 0);
  }
  // 16 -> 8 -> 4, then pinned at the floor.
  EXPECT_EQ(admissionFor(tier).stats().capacity, 4);
}

TEST_F(BackendAdmissionTest, adaptiveRecoveryScalesWithSuccesses) {
  const std::string tier = "test.tier";
  setCeiling(tier, 16);
  setAdaptive(tier, true, 1, 0.5);

  admissionFor(tier).onOutcome(
      BackendAdmission::Outcome::kOverload, 0); // 16 -> 8
  ASSERT_EQ(admissionFor(tier).stats().capacity, 8);

  // A tiny drain recovers by the +1 floor (step = max(1, 1/8)).
  admissionFor(tier).onOutcome(BackendAdmission::Outcome::kSuccess, 1);
  EXPECT_EQ(admissionFor(tier).stats().capacity, 9);

  // A large drain recovers proportionally (step = successes/cap) and, on
  // reaching the ceiling, clears the adaptive state so the static cap governs.
  admissionFor(tier).onOutcome(BackendAdmission::Outcome::kSuccess, 1000);
  EXPECT_EQ(admissionFor(tier).stats().capacity, 16);
}

TEST_F(BackendAdmissionTest, adaptiveShrinkReducesAdmission) {
  const std::string tier = "test.tier";
  setCeiling(tier, 4);
  setAdaptive(tier, true, 1, 0.5);

  admissionFor(tier).onOutcome(
      BackendAdmission::Outcome::kOverload, 0); // cap 4 -> 2
  ASSERT_EQ(admissionFor(tier).stats().capacity, 2);

  auto token1 = admissionFor(tier).acquire();
  EXPECT_FALSE(admissionFor(tier).waitForCapacity().has_value());
  auto token2 = admissionFor(tier).acquire();
  // At the shrunk cap of 2 (not the static 4), backpressure kicks in.
  EXPECT_TRUE(admissionFor(tier).waitForCapacity().has_value());
}

TEST_F(BackendAdmissionTest, noBackpressureBelowLimit) {
  const std::string tier = "test.tier";
  setCeiling(tier, 5);

  std::vector<BackendAdmission::Token> tokens;
  for (int i = 0; i < 4; ++i) {
    tokens.push_back(admissionFor(tier).acquire());
    EXPECT_FALSE(admissionFor(tier).waitForCapacity().has_value());
  }
  EXPECT_EQ(admissionFor(tier).stats().pending, 4);
}

// Regression test for the defect that motivated per-tier admission: the
// previous API configured the adaptive parameters process-globally, so a
// second backend's configuration silently reconfigured the first backend's
// limiter and both shrank together.
TEST_F(BackendAdmissionTest, adaptiveIsPerTier) {
  const std::string adaptiveTier = "backend.adaptive";
  const std::string fixedTier = "backend.fixed";
  setCeiling(adaptiveTier, 16);
  setCeiling(fixedTier, 16);
  setAdaptive(adaptiveTier, true, /*floor*/ 1, /*decreaseFactor*/ 0.5);
  setAdaptive(fixedTier, false, /*floor*/ 1, /*decreaseFactor*/ 0.5);

  admissionFor(adaptiveTier)
      .onOutcome(BackendAdmission::Outcome::kOverload, /*units*/ 0);
  admissionFor(fixedTier).onOutcome(
      BackendAdmission::Outcome::kOverload, /*units*/ 0);

  EXPECT_EQ(admissionFor(adaptiveTier).stats().capacity, 8);
  EXPECT_EQ(admissionFor(fixedTier).stats().capacity, 16);
}

// Two adapting tiers shrink by their own decrease factors rather than sharing
// one process-global value.
TEST_F(BackendAdmissionTest, adaptiveFactorsAreIndependent) {
  const std::string halving = "backend.halving";
  const std::string quartering = "backend.quartering";
  setCeiling(halving, 16);
  setCeiling(quartering, 16);
  setAdaptive(halving, true, /*floor*/ 1, /*decreaseFactor*/ 0.5);
  setAdaptive(quartering, true, /*floor*/ 1, /*decreaseFactor*/ 0.25);

  admissionFor(halving).onOutcome(BackendAdmission::Outcome::kOverload, 0);
  admissionFor(quartering).onOutcome(BackendAdmission::Outcome::kOverload, 0);

  EXPECT_EQ(admissionFor(halving).stats().capacity, 8);
  EXPECT_EQ(admissionFor(quartering).stats().capacity, 4);
}

} // namespace
} // namespace facebook::velox::exec::rpc
