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

/// RPCRateLimiterTest - Tests for the per-tier RPC rate limiter.
///
/// RPCRateLimiter provides per-tier concurrency limits with FIFO waiter
/// notification and RAII token-based slot management.
///
/// Tests cover:
/// - acquireAndRelease: Token acquire increments, destruction decrements.
/// - backpressureWhenAtLimit: checkBackpressure returns future at limit.
/// - backpressureReliefOnRelease: Waiter notified when token released.
/// - fifoWaiterNotification: Multiple waiters notified in FIFO order.
/// - perTierIsolation: Different tiers have independent limits.
/// - perTierMaxPending: setMaxPending overrides global default.
/// - defaultMaxPending: setDefaultMaxPending affects tiers without override.
/// - tokenMoveSemantics: Move constructor/assignment transfer ownership.
/// - testingResetAllState: Reset clears all tiers and restores defaults.

#include "velox/exec/rpc/RPCRateLimiter.h"

#include <gtest/gtest.h>

#include <vector>

namespace facebook::velox::exec::rpc {
namespace {

class RPCRateLimiterTest : public testing::Test {
 protected:
  void SetUp() override {
    RPCRateLimiter::testingResetAllState();
  }

  void TearDown() override {
    RPCRateLimiter::testingResetAllState();
  }
};

TEST_F(RPCRateLimiterTest, acquireAndRelease) {
  const std::string tier = "test.tier";
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 0);

  {
    auto token = RPCRateLimiter::acquire(tier);
    EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 1);
  }
  // Token destroyed — count should be back to 0.
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 0);
}

TEST_F(RPCRateLimiterTest, multipleAcquires) {
  const std::string tier = "test.tier";

  auto token1 = RPCRateLimiter::acquire(tier);
  auto token2 = RPCRateLimiter::acquire(tier);
  auto token3 = RPCRateLimiter::acquire(tier);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 3);
}

TEST_F(RPCRateLimiterTest, backpressureWhenAtLimit) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setMaxPending(tier, 2);

  auto token1 = RPCRateLimiter::acquire(tier);
  // 1 pending, limit 2 — no backpressure.
  EXPECT_FALSE(RPCRateLimiter::checkBackpressure(tier).has_value());

  auto token2 = RPCRateLimiter::acquire(tier);
  // 2 pending, limit 2 — at limit, should get backpressure.
  auto future = RPCRateLimiter::checkBackpressure(tier);
  EXPECT_TRUE(future.has_value());
}

TEST_F(RPCRateLimiterTest, backpressureReliefOnRelease) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setMaxPending(tier, 1);

  auto token1 = RPCRateLimiter::acquire(tier);

  // At limit — should block.
  auto future = RPCRateLimiter::checkBackpressure(tier);
  ASSERT_TRUE(future.has_value());
  EXPECT_FALSE(future->isReady());

  // Release the token — waiter should be notified.
  token1 = RPCRateLimiter::Token();
  EXPECT_TRUE(future->isReady());
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 0);
}

TEST_F(RPCRateLimiterTest, fifoWaiterNotification) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setMaxPending(tier, 1);

  auto token = RPCRateLimiter::acquire(tier);

  // Two waiters enqueue while at limit.
  auto future1 = RPCRateLimiter::checkBackpressure(tier);
  auto future2 = RPCRateLimiter::checkBackpressure(tier);
  ASSERT_TRUE(future1.has_value());
  ASSERT_TRUE(future2.has_value());

  // Release token — only first waiter should be notified (FIFO).
  token = RPCRateLimiter::Token();
  EXPECT_TRUE(future1->isReady());
  EXPECT_FALSE(future2->isReady());

  // Acquire and release again — second waiter should be notified.
  {
    auto token2 = RPCRateLimiter::acquire(tier);
  }
  EXPECT_TRUE(future2->isReady());
}

TEST_F(RPCRateLimiterTest, perTierIsolation) {
  const std::string tier1 = "tier.one";
  const std::string tier2 = "tier.two";
  RPCRateLimiter::setMaxPending(tier1, 1);
  RPCRateLimiter::setMaxPending(tier2, 1);

  auto tokenA = RPCRateLimiter::acquire(tier1);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier1), 1);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier2), 0);

  // tier1 at limit, tier2 not.
  EXPECT_TRUE(RPCRateLimiter::checkBackpressure(tier1).has_value());
  EXPECT_FALSE(RPCRateLimiter::checkBackpressure(tier2).has_value());
}

TEST_F(RPCRateLimiterTest, perTierMaxPending) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setDefaultMaxPending(10);
  RPCRateLimiter::setMaxPending(tier, 2);

  auto token1 = RPCRateLimiter::acquire(tier);
  auto token2 = RPCRateLimiter::acquire(tier);

  // Per-tier limit of 2 applies, not global default of 10.
  EXPECT_TRUE(RPCRateLimiter::checkBackpressure(tier).has_value());
}

TEST_F(RPCRateLimiterTest, defaultMaxPending) {
  RPCRateLimiter::setDefaultMaxPending(2);
  const std::string tier = "test.default.tier";

  auto token1 = RPCRateLimiter::acquire(tier);
  EXPECT_FALSE(RPCRateLimiter::checkBackpressure(tier).has_value());

  auto token2 = RPCRateLimiter::acquire(tier);
  // Global default of 2 reached.
  EXPECT_TRUE(RPCRateLimiter::checkBackpressure(tier).has_value());
}

TEST_F(RPCRateLimiterTest, tokenMoveConstructor) {
  const std::string tier = "test.tier";

  auto token1 = RPCRateLimiter::acquire(tier);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 1);

  // Move construct — ownership transfers, count stays 1.
  auto token2 = std::move(token1);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 1);

  // Destroy moved-from token — no effect.
  // (token1 is already moved-from, but let it go out of scope naturally)
}

TEST_F(RPCRateLimiterTest, tokenMoveAssignment) {
  const std::string tier1 = "tier.one";
  const std::string tier2 = "tier.two";

  auto token1 = RPCRateLimiter::acquire(tier1);
  auto token2 = RPCRateLimiter::acquire(tier2);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier1), 1);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier2), 1);

  // Move-assign token2 into token1 — old token1 (tier1) released.
  token1 = std::move(token2);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier1), 0);
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier2), 1);
}

TEST_F(RPCRateLimiterTest, testingResetAllState) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setDefaultMaxPending(5);
  RPCRateLimiter::setMaxPending(tier, 3);
  auto token = RPCRateLimiter::acquire(tier);

  // Move the token out so it doesn't decrement during reset.
  // (In practice, testingResetAllState clears the tiers map,
  // so the token's destructor will create a new empty tier state.)

  RPCRateLimiter::testingResetAllState();

  // Default restored to 20.
  EXPECT_EQ(RPCRateLimiter::defaultMaxPending(), 20);
  // Tier state cleared — pending count is 0 for a fresh tier.
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 0);
}

TEST_F(RPCRateLimiterTest, noBackpressureBelowLimit) {
  const std::string tier = "test.tier";
  RPCRateLimiter::setMaxPending(tier, 5);

  std::vector<RPCRateLimiter::Token> tokens;
  for (int i = 0; i < 4; ++i) {
    tokens.push_back(RPCRateLimiter::acquire(tier));
    EXPECT_FALSE(RPCRateLimiter::checkBackpressure(tier).has_value());
  }
  EXPECT_EQ(RPCRateLimiter::pendingCount(tier), 4);
}

} // namespace
} // namespace facebook::velox::exec::rpc
