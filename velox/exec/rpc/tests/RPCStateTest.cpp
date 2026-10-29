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

/// RPCStateTest - Tests for the RPCState shared state coordination.
///
/// RPCState coordinates between the operator's driver thread and async RPC
/// completion callbacks using a mutex-protected state object.
///
/// Tests cover both PER_ROW and BATCH streaming modes:
///
/// PER_ROW tests:
/// - basicAddAndClaim: Add pending row, fulfill, claim via tryClaimOrWait
/// - claimOrWaitReturnsFinished: After noMoreInput + all claimed
/// - claimOrWaitMustWait: When no rows ready, returns kMustWait with future
/// - pendingRowCount: Tracks pending count correctly
///
/// BATCH tests:
/// - basicAddAndPollBatch: Add pending batch, fulfill, poll
/// - pollBatchOrWaitFinished: After noMoreInput + all polled
/// - pollBatchOrWaitMustWait: When no batch ready, returns kMustWait
/// - batchErrorHandling: Exception in batch future handled gracefully
/// - batchRowLocationsCarriedThrough: Row locations carried from pending to
///   ready batch
///
/// Common tests:
/// - noMoreInputAndIsFinished: Lifecycle signals
/// - inputBatchStorageAndRelease: Batch-reference input storage and release
/// - backpressure: isUnderBackpressure when pending >= max
/// - drainReadyRows: Batched drain of multiple ready rows

#include "velox/exec/rpc/RPCState.h"
#include "velox/exec/rpc/RpcErrorClassification.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

#include <folly/futures/Promise.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"

#include "velox/common/memory/Memory.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <set>
#include <string>
#include <thread>

namespace facebook::velox::exec::rpc {
namespace {

class RPCStateTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::initialize({});
    }
  }

  static std::shared_ptr<memory::MemoryPool> testPool() {
    static auto root = memory::memoryManager()->addRootPool("rpcStateTest");
    static auto pool = root->addLeafChild("leaf");
    return pool;
  }

  void SetUp() override {
    state_ = std::make_shared<RPCState>(testPool());
  }

  /// Polls a condition with short waits until it becomes true or timeout.
  static void waitFor(
      const std::function<bool()>& condition,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition()) {
      ASSERT_LT(std::chrono::steady_clock::now(), deadline)
          << "Timed out waiting for condition";
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  std::shared_ptr<RPCState> state_;
};

TEST_F(RPCStateTest, configuresBatchCongestionStartWindow) {
  {
    state_->setStreamingMode(RPCStreamingMode::kBatch);
    EXPECT_EQ(state_->operatorSnapshot().windowLimit, 2);
  }
  {
    state_->setStreamingMode(RPCStreamingMode::kPerRow);
    EXPECT_EQ(state_->operatorSnapshot().windowLimit, 100);
  }
  {
    state_->setStreamingMode(
        RPCStreamingMode::kBatch,
        /*minWindow=*/1,
        /*stepCoef=*/1.0,
        /*maxWindow=*/64,
        /*batchStartWindow=*/8);
    EXPECT_EQ(state_->operatorSnapshot().windowLimit, 8);
  }
  {
    state_->setStreamingMode(
        RPCStreamingMode::kPerRow,
        /*minWindow=*/1,
        /*stepCoef=*/1.0,
        /*maxWindow=*/64,
        /*batchStartWindow=*/8);
    EXPECT_EQ(state_->operatorSnapshot().windowLimit, 64);
  }
  {
    state_->setStreamingMode(
        RPCStreamingMode::kBatch,
        /*minWindow=*/4,
        /*stepCoef=*/1.0,
        /*maxWindow=*/64,
        /*batchStartWindow=*/2);
    EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
  }
  {
    state_->setStreamingMode(
        RPCStreamingMode::kBatch,
        /*minWindow=*/1,
        /*stepCoef=*/1.0,
        /*maxWindow=*/4,
        /*batchStartWindow=*/8);
    EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
  }
  VELOX_ASSERT_THROW(
      state_->setStreamingMode(
          RPCStreamingMode::kBatch,
          /*minWindow=*/1,
          /*stepCoef=*/1.0,
          /*maxWindow=*/64,
          /*batchStartWindow=*/-1),
      "must be non-negative");
  VELOX_ASSERT_THROW(
      state_->setStreamingMode(
          RPCStreamingMode::kBatch,
          /*minWindow=*/1,
          /*stepCoef=*/1.0,
          /*maxWindow=*/64,
          /*batchStartWindow=*/8,
          /*asyncGrowthMaxWindow=*/-1),
      "async growth max window must be non-negative");
}

// ========== PER_ROW mode tests ==========

TEST_F(RPCStateTest, basicAddAndClaim) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();

  state_->addPendingRow(
      state_, 42, RPCState::RowLocation{0, 0}, std::move(future));
  EXPECT_EQ(state_->numInFlight(), 1);

  // Fulfill the promise
  RPCResponse response;
  response.rowId = 42;
  response.setPayload(makeTextPayload("test result"));
  promise.setValue(std::move(response));

  // Wait for async callback to move response into readyRows_
  waitFor([&]() { return state_->numInFlight() == 0; });

  // Now claim
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  auto result = state_->tryClaimOrWait(&waitFuture, &claimedRow);

  ASSERT_EQ(result, RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_EQ(claimedRow->rowId, 42);
  EXPECT_EQ(responseAs<TextPayload>(claimedRow->response).text, "test result");
}

TEST_F(RPCStateTest, claimReportsIdleWithoutCreatingAWaiter) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;

  EXPECT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kIdle);
  EXPECT_FALSE(waitFuture.valid());
  EXPECT_FALSE(claimedRow.has_value());
}

// A hinted rejection goes to the retry queue instead of to output. A due row
// is handed back for dispatch while its budget lasts; once the budget has run
// out it fails with its rejection instead.
TEST_F(RPCStateTest, capacityRetryDispatchesOnlyWithinBudget) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto addRejected = [&](int64_t rowId, std::chrono::milliseconds budget) {
    auto response = RPCResponse::failed(
        rowId, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
    response.setCapacityRetry(
        {.retryAfter = std::chrono::milliseconds(0), .budget = budget});
    state_->addPendingRow(
        state_,
        rowId,
        RPCState::RowLocation{0, 0},
        folly::makeSemiFuture(std::move(response)));
    waitFor([&]() { return state_->numInFlight() == 0; });
  };

  {
    addRejected(1, std::chrono::seconds(60));
    EXPECT_TRUE(state_->hasRetries());
    EXPECT_FALSE(state_->tryClaimReady().has_value());
    const auto retries = state_->takeDueRetries(1);
    ASSERT_EQ(retries.size(), 1);
    EXPECT_EQ(retries[0].rowId, 1);
    EXPECT_EQ(retries[0].numAttempts, 1);
  }

  {
    addRejected(2, std::chrono::milliseconds(200));
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_TRUE(state_->takeDueRetries(1).empty());
    EXPECT_FALSE(state_->hasRetries());
    const auto failed = state_->tryClaimReady();
    ASSERT_TRUE(failed.has_value());
    EXPECT_EQ(failed->rowId, 2);
    EXPECT_EQ(
        failed->response.errorKind(), velox::rpc::RPCErrorKind::kRateLimited);
    EXPECT_EQ(failed->response.error().message, "rejected");
    EXPECT_TRUE(failed->sharedOverloadHandled);
  }
}

// A row past its budget fails as soon as the driver claims, without waiting
// for an admission slot to dispatch it.
TEST_F(RPCStateTest, expiredCapacityRetryFailsOnClaim) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(0),
       .budget = std::chrono::milliseconds(200)});
  state_->addPendingRow(
      state_,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state_->hasRetries(); });
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kClaimed);
  EXPECT_EQ(
      claimedRow->response.errorKind(), velox::rpc::RPCErrorKind::kRateLimited);
  EXPECT_FALSE(state_->hasRetries());
}

// A row waiting for a retry keeps its input batch, and the arguments needed to
// dispatch it again, even after every other row of the batch was output.
TEST_F(RPCStateTest, capacityRetryKeepsBatchArgumentsAfterSiblingsOutput) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  const auto batchIndex = state_->storeInputBatch({}, 2);
  state_->setInputBatchArguments(
      batchIndex,
      {BaseVector::createConstant(
          BIGINT(), variant(int64_t{1}), 2, testPool().get())},
      2);

  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(0),
       .budget = std::chrono::seconds(60)});
  state_->addPendingRow(
      state_,
      1,
      RPCState::RowLocation{batchIndex, 0},
      folly::makeSemiFuture(std::move(rejected)));
  state_->addPendingRow(
      state_,
      2,
      RPCState::RowLocation{batchIndex, 1},
      folly::makeSemiFuture(RPCResponse::ok(2, makeTextPayload("served"))));
  waitFor([&]() { return state_->numInFlight() == 0; });

  // The served sibling is output and released.
  ASSERT_TRUE(state_->tryClaimReady().has_value());
  state_->releaseRows(batchIndex, 1);

  ASSERT_TRUE(state_->hasRetries());
  const auto [arguments, numRows] = state_->inputBatchArguments(batchIndex);
  EXPECT_EQ(arguments.size(), 1);
  EXPECT_EQ(numRows, 2);
}

// Closing with rows queued for a retry drops them, and a retry wake-up timer
// that fires afterwards finds nothing to do.
TEST_F(RPCStateTest, closeDropsQueuedCapacityRetries) {
  auto state = std::make_shared<RPCState>(testPool());
  state->setStreamingMode(RPCStreamingMode::kPerRow);
  state->setCapacityRetryEnabled(true);
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(200),
       .budget = std::chrono::seconds(60)});
  state->addPendingRow(
      state,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state->hasRetries(); });

  // Park on the state so a wake-up timer is armed for the queued retry.
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kMustWait);

  state->close();
  EXPECT_FALSE(state->hasRetries());
  EXPECT_TRUE(state->takeDueRetries(1).empty());
  EXPECT_TRUE(waitFuture.isReady());
  // The timer firing later finds nothing to do.
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

// A retry wake-up timer that fires after the state is gone does nothing.
TEST_F(RPCStateTest, retryTimerOutlivesState) {
  auto state = std::make_shared<RPCState>(testPool());
  state->setStreamingMode(RPCStreamingMode::kPerRow);
  state->setCapacityRetryEnabled(true);
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(200),
       .budget = std::chrono::seconds(60)});
  state->addPendingRow(
      state,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state->hasRetries(); });
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kMustWait);

  state->close();
  state.reset();
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

// The retry wake-up timer alone wakes a driver parked on a not-yet-due retry,
// and a due retry is then handed back for dispatch.
TEST_F(RPCStateTest, retryTimerWakesParkedDriver) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(100),
       .budget = std::chrono::seconds(60)});
  state_->addPendingRow(
      state_,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state_->hasRetries(); });

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kMustWait);
  EXPECT_FALSE(waitFuture.isReady());
  std::move(waitFuture).wait(std::chrono::seconds(5));
  EXPECT_EQ(state_->numDueRetries(), 1);
  EXPECT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kIdle);
}

// A retry that becomes the earliest while the driver is parked on a later one
// wakes the driver, and the driver's next park arms a timer for it: the wake
// comes when the new retry is due, not when the old one is.
TEST_F(RPCStateTest, earlierRetryRearmsWakeTimer) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto addRejected = [&](int64_t rowId, std::chrono::milliseconds retryAfter) {
    auto rejected = RPCResponse::failed(
        rowId, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
    rejected.setCapacityRetry(
        {.retryAfter = retryAfter, .budget = std::chrono::seconds(60)});
    state_->addPendingRow(
        state_,
        rowId,
        RPCState::RowLocation{0, 0},
        folly::makeSemiFuture(std::move(rejected)));
  };

  addRejected(1, std::chrono::seconds(30));
  waitFor([&]() { return state_->hasRetries(); });
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kMustWait);

  // Queueing the earlier retry wakes the parked driver.
  addRejected(2, std::chrono::milliseconds(100));
  ASSERT_TRUE(std::move(waitFuture).wait(std::chrono::seconds(5)));

  ASSERT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kMustWait);
  ASSERT_TRUE(std::move(waitFuture).wait(std::chrono::seconds(5)));
  EXPECT_EQ(state_->numDueRetries(), 1);
}

// With a maxRetryAfter, each further rejection of the same row doubles its
// wait; without one, every wait stays at retryAfter.
TEST_F(RPCStateTest, repeatedRejectionsGrowTheWaitUpToItsCap) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto rejected = [](int64_t rowId, std::chrono::milliseconds maxRetryAfter) {
    auto response = RPCResponse::failed(
        rowId, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
    response.setCapacityRetry(
        {.retryAfter = std::chrono::milliseconds(300),
         .budget = std::chrono::seconds(60),
         .maxRetryAfter = maxRetryAfter});
    return folly::makeSemiFuture(std::move(response));
  };
  // Rejected once, then again: whether the second wait lasted over 400 ms.
  auto secondWaitExceeds400Ms = [&](int64_t rowId,
                                    std::chrono::milliseconds maxRetryAfter) {
    state_->addPendingRow(
        state_,
        rowId,
        RPCState::RowLocation{0, 0},
        rejected(rowId, maxRetryAfter));
    waitFor([&]() { return state_->numDueRetries() == 1; });
    const auto retries = state_->takeDueRetries(1);
    VELOX_CHECK_EQ(retries.size(), 1);
    state_->addRetriedRow(
        state_, retries[0], rejected(rowId, maxRetryAfter), nullptr);
    waitFor([&]() { return state_->hasRetries(); });
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const bool notYetDue = state_->numDueRetries() == 0;
    waitFor([&]() { return state_->numDueRetries() == 1; });
    VELOX_CHECK_EQ(state_->takeDueRetries(1).size(), 1);
    return notYetDue;
  };

  // The second wait is 600 ms with a cap of 1.2 s, and 300 ms without a cap.
  EXPECT_TRUE(secondWaitExceeds400Ms(1, std::chrono::milliseconds(1'200)));
  EXPECT_FALSE(secondWaitExceeds400Ms(2, std::chrono::milliseconds(0)));
}

// With jitter, rows rejected together draw their waits from [0, retryAfter],
// so their retries spread over the window instead of coming due at once.
TEST_F(RPCStateTest, jitterSpreadsRetries) {
  constexpr int32_t kRows = 40;
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  for (int32_t rowId = 0; rowId < kRows; ++rowId) {
    auto response = RPCResponse::failed(
        rowId, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
    response.setCapacityRetry(
        {.retryAfter = std::chrono::milliseconds(800),
         .budget = std::chrono::seconds(60),
         .jitter = true});
    state_->addPendingRow(
        state_,
        rowId,
        RPCState::RowLocation{0, 0},
        folly::makeSemiFuture(std::move(response)));
  }
  waitFor([&]() { return state_->numInFlight() == 0; });
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  // About half are due at the midpoint; all are due by the end of the window.
  const auto dueAtMidpoint = state_->numDueRetries();
  EXPECT_GT(dueAtMidpoint, 0);
  EXPECT_LT(dueAtMidpoint, kRows);
  waitFor([&]() { return state_->numDueRetries() == kRows; });
}

// A rejection is counted as a capacity retry once it is admitted again, not
// when it is queued: a row that expires in the queue is not counted.
TEST_F(RPCStateTest, capacityRetriesCountReadmissions) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(0),
       .budget = std::chrono::seconds(60)});
  state_->addPendingRow(
      state_,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state_->hasRetries(); });
  EXPECT_EQ(state_->operatorSnapshot().numCapacityRetries, 0);

  const auto retries = state_->takeDueRetries(1);
  ASSERT_EQ(retries.size(), 1);
  state_->addRetriedRow(
      state_,
      retries[0],
      folly::makeSemiFuture(RPCResponse::ok(1, makeTextPayload("served"))),
      nullptr);
  waitFor([&]() { return state_->numInFlight() == 0; });
  EXPECT_EQ(state_->operatorSnapshot().numCapacityRetries, 1);
}

// A hint far beyond any real wait or budget queues the row rather than
// overflowing its deadline.
TEST_F(RPCStateTest, hugeCapacityRetryHintQueuesRow) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setCapacityRetryEnabled(true);
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::milliseconds(0),
       .budget = std::chrono::milliseconds::max()});
  state_->addPendingRow(
      state_,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state_->numInFlight() == 0; });
  EXPECT_TRUE(state_->hasRetries());
  EXPECT_EQ(state_->takeDueRetries(1).size(), 1);
}

// Rows waiting for a capacity retry hold back new input like rows in flight,
// since each keeps its input batch.
TEST_F(RPCStateTest, queuedCapacityRetriesApplyBackpressure) {
  state_->setStreamingMode(
      RPCStreamingMode::kPerRow,
      /*minWindow=*/1,
      /*stepCoef=*/0.5,
      /*maxWindow=*/1);
  state_->setCapacityRetryEnabled(true);
  EXPECT_FALSE(state_->isUnderBackpressure());
  auto rejected = RPCResponse::failed(
      1, velox::rpc::RPCErrorKind::kRateLimited, "rejected");
  rejected.setCapacityRetry(
      {.retryAfter = std::chrono::seconds(60),
       .budget = std::chrono::seconds(120)});
  state_->addPendingRow(
      state_,
      1,
      RPCState::RowLocation{0, 0},
      folly::makeSemiFuture(std::move(rejected)));
  waitFor([&]() { return state_->numInFlight() == 0; });
  ASSERT_TRUE(state_->hasRetries());
  EXPECT_TRUE(state_->isUnderBackpressure());
}

TEST_F(RPCStateTest, completedRowReleasesAdmissionBeforeDriverOutput) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  auto limiter = std::make_shared<RPCRateLimiter>("test.row.admission");
  limiter->configure(RPCRateLimiter::Config{.ceiling = 1});
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  auto admissionLease = limiter->makeLease(std::move(tokens.front()));

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_,
      42,
      RPCState::RowLocation{0, 0},
      std::move(future),
      std::move(admissionLease));
  promise.setValue(RPCResponse{});
  waitFor([&]() { return state_->numInFlight() == 0; });

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_FALSE(claimedRow->sharedOverloadHandled);
  EXPECT_TRUE(claimedRow->admissionEpoch.has_value());
}

TEST_F(RPCStateTest, completedOverloadBacksOffBeforeReleasingAdmission) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  auto limiter = std::make_shared<RPCRateLimiter>("test.row.overload");
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 1,
          .decreaseFactor = 0.5,
      });
  auto tokens = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_,
      42,
      RPCState::RowLocation{0, 0},
      std::move(future),
      limiter->makeLease(std::move(tokens.front())));
  promise.setValue(
      RPCResponse::failed(
          42, velox::rpc::RPCErrorKind::kRateLimited, "overloaded"));
  waitFor([&]() { return state_->numInFlight() == 0; });

  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_EQ(limiter->stats().capacity, 1);
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_TRUE(claimedRow->sharedOverloadHandled);
  EXPECT_TRUE(claimedRow->admissionEpoch.has_value());
}

TEST_F(RPCStateTest, completedRowMemoryIsAccountedUntilReleased) {
  auto root = memory::memoryManager()->addRootPool("rpcResponseAccounting");
  auto pool = root->addLeafChild("leaf");
  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kPerRow);
  const auto baseline = pool->stats();

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state->addPendingRow(
      state, 42, RPCState::RowLocation{0, 0}, std::move(future));

  RPCResponse response;
  response.rowId = 42;
  response.setPayload(makeTextPayload(std::string(4096, 'x')));
  const auto retainedBytes = response.retainedBytes();
  ASSERT_GT(retainedBytes, 0);
  promise.setValue(std::move(response));

  waitFor([&]() { return state->numInFlight() == 0; });
  EXPECT_EQ(pool->usedBytes(), baseline.usedBytes + retainedBytes);
  EXPECT_EQ(pool->stats().numExternalAllocs, baseline.numExternalAllocs + 1);

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  ASSERT_EQ(
      state->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_EQ(claimedRow->charge.bytes(), retainedBytes);
  EXPECT_EQ(pool->usedBytes(), baseline.usedBytes + retainedBytes);

  claimedRow.reset();
  EXPECT_EQ(pool->usedBytes(), baseline.usedBytes);
  EXPECT_EQ(pool->stats().numExternalFrees, baseline.numExternalFrees + 1);
}

TEST_F(RPCStateTest, completedBatchMemoryIsAccountedUntilReleased) {
  auto root = memory::memoryManager()->addRootPool("rpcBatchAccounting");
  auto pool = root->addLeafChild("leaf");
  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kBatch);
  const auto baseline = pool->stats();

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state->addPendingBatch(state, std::move(future), {}, /*admissionUnits=*/1);

  std::vector<RPCResponse> responses(2);
  responses[0].setPayload(makeTextPayload(std::string(2048, 'a')));
  responses[1].setPayload(makeTextPayload(std::string(4096, 'b')));
  const auto retainedBytes =
      responses[0].retainedBytes() + responses[1].retainedBytes();
  promise.setValue(std::move(responses));

  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    readyBatch = state->tryPollReady();
    return readyBatch.has_value();
  });
  ASSERT_TRUE(readyBatch.has_value());
  EXPECT_EQ(readyBatch->charge.bytes(), retainedBytes);
  EXPECT_EQ(pool->usedBytes(), baseline.usedBytes + retainedBytes);
  EXPECT_EQ(pool->stats().numExternalAllocs, baseline.numExternalAllocs + 1);

  readyBatch.reset();
  EXPECT_EQ(pool->usedBytes(), baseline.usedBytes);
  EXPECT_EQ(pool->stats().numExternalFrees, baseline.numExternalFrees + 1);
}

TEST_F(RPCStateTest, completedBatchReleasesAdmissionBeforeDriverOutput) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  auto limiter = std::make_shared<RPCRateLimiter>("test.batch.admission");
  limiter->configure(RPCRateLimiter::Config{.ceiling = 1});
  auto reserved = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(reserved.size(), 1);
  auto admissionLease = limiter->makeLease(std::move(reserved));

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(
      state_,
      std::move(future),
      {},
      /*admissionUnits=*/1,
      std::move(admissionLease));
  promise.setValue(std::vector<RPCResponse>(1));

  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    readyBatch = state_->tryPollReady();
    return readyBatch.has_value();
  });
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_FALSE(readyBatch->sharedOverloadHandled);
  EXPECT_TRUE(readyBatch->admissionEpoch.has_value());
}

TEST_F(RPCStateTest, completedBatchOverloadBacksOffBeforeDriverOutput) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  auto limiter = std::make_shared<RPCRateLimiter>("test.batch.overload");
  limiter->configure(
      RPCRateLimiter::Config{
          .adaptive = true,
          .ceiling = 2,
          .floor = 1,
          .decreaseFactor = 0.5,
      });
  auto reserved = limiter->tryAcquireUpTo(1);
  ASSERT_EQ(reserved.size(), 1);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(
      state_,
      std::move(future),
      {},
      /*admissionUnits=*/1,
      limiter->makeLease(std::move(reserved)));
  std::vector<RPCResponse> responses;
  responses.push_back(
      RPCResponse::failed(
          42, velox::rpc::RPCErrorKind::kRateLimited, "overloaded"));
  promise.setValue(std::move(responses));

  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    readyBatch = state_->tryPollReady();
    return readyBatch.has_value();
  });
  ASSERT_TRUE(readyBatch.has_value());
  EXPECT_EQ(limiter->stats().pending, 0);
  EXPECT_EQ(limiter->stats().capacity, 1);
  EXPECT_TRUE(readyBatch->sharedOverloadHandled);
  EXPECT_TRUE(readyBatch->admissionEpoch.has_value());
}

TEST_F(RPCStateTest, closeReleasesQueuedPayloadAndIgnoresLateCompletion) {
  auto root = memory::memoryManager()->addRootPool("rpcCloseAccounting");
  auto pool = root->addLeafChild("leaf");
  const auto baseline = pool->usedBytes();

  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kPerRow);
  auto [readyPromise, readyFuture] = folly::makePromiseContract<RPCResponse>();
  state->addPendingRow(
      state, 1, RPCState::RowLocation{0, 0}, std::move(readyFuture));
  RPCResponse readyResponse;
  readyResponse.setPayload(makeTextPayload(std::string(4096, 'x')));
  readyPromise.setValue(std::move(readyResponse));
  waitFor([&]() { return state->numInFlight() == 0; });
  EXPECT_GT(pool->usedBytes(), baseline);

  state->close();
  EXPECT_EQ(pool->usedBytes(), baseline);
  state->close();

  auto lateState = std::make_shared<RPCState>(pool);
  lateState->setStreamingMode(RPCStreamingMode::kPerRow);
  auto [latePromise, lateFuture] = folly::makePromiseContract<RPCResponse>();
  lateState->addPendingRow(
      lateState, 2, RPCState::RowLocation{0, 0}, std::move(lateFuture));
  lateState->close();
  RPCResponse lateResponse;
  lateResponse.setPayload(makeTextPayload(std::string(4096, 'y')));
  latePromise.setValue(std::move(lateResponse));
  waitFor([&]() { return lateState->numInFlight() == 0; });
  EXPECT_EQ(pool->usedBytes(), baseline);
}

TEST_F(RPCStateTest, closeReleasesPerRowAndBatchAdmission) {
  auto rowLimiter = std::make_shared<RPCRateLimiter>("test.close.row");
  rowLimiter->configure(RPCRateLimiter::Config{.ceiling = 1});
  auto rowTokens = rowLimiter->tryAcquireUpTo(1);
  ASSERT_EQ(rowTokens.size(), 1);
  auto rowLease = rowLimiter->makeLease(std::move(rowTokens.front()));

  auto rowState = std::make_shared<RPCState>(testPool());
  rowState->setStreamingMode(RPCStreamingMode::kPerRow);
  auto [rowPromise, rowFuture] = folly::makePromiseContract<RPCResponse>();
  rowState->addPendingRow(
      rowState,
      1,
      RPCState::RowLocation{0, 0},
      std::move(rowFuture),
      std::move(rowLease));
  ASSERT_EQ(rowLimiter->stats().pending, 1);

  rowState->close();
  EXPECT_EQ(rowLimiter->stats().pending, 0);
  rowPromise.setValue(RPCResponse{});
  waitFor([&]() { return rowState->numInFlight() == 0; });
  EXPECT_EQ(rowLimiter->stats().pending, 0);

  auto batchLimiter = std::make_shared<RPCRateLimiter>("test.close.batch");
  batchLimiter->configure(RPCRateLimiter::Config{.ceiling = 1});
  auto batchTokens = batchLimiter->tryAcquireUpTo(1);
  ASSERT_EQ(batchTokens.size(), 1);
  auto admissionLease = batchLimiter->makeLease(std::move(batchTokens));

  auto batchState = std::make_shared<RPCState>(testPool());
  batchState->setStreamingMode(RPCStreamingMode::kBatch);
  auto [batchPromise, batchFuture] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  batchState->addPendingBatch(
      batchState,
      std::move(batchFuture),
      {},
      /*admissionUnits=*/1,
      std::move(admissionLease));
  ASSERT_EQ(batchLimiter->stats().pending, 1);

  batchState->close();
  EXPECT_EQ(batchLimiter->stats().pending, 0);
  batchPromise.setValue(std::vector<RPCResponse>(1));
  EXPECT_EQ(batchLimiter->stats().pending, 0);
}

TEST_F(RPCStateTest, closeReleasesReadyBatchAndIgnoresLateBatch) {
  auto root = memory::memoryManager()->addRootPool("rpcBatchCloseAccounting");
  auto pool = root->addLeafChild("leaf");
  const auto baseline = pool->usedBytes();

  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kBatch);
  auto [readyPromise, readyFuture] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state->addPendingBatch(
      state, std::move(readyFuture), {}, /*admissionUnits=*/1);
  std::vector<RPCResponse> readyResponses(1);
  readyResponses[0].setPayload(makeTextPayload(std::string(4096, 'x')));
  readyPromise.setValue(std::move(readyResponses));
  waitFor(
      [&]() { return state->operatorSnapshot().numCompletionsSignaled == 1; });
  EXPECT_GT(pool->usedBytes(), baseline);

  state->close();
  EXPECT_EQ(pool->usedBytes(), baseline);

  auto lateState = std::make_shared<RPCState>(pool);
  lateState->setStreamingMode(RPCStreamingMode::kBatch);
  auto [latePromise, lateFuture] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  lateState->addPendingBatch(
      lateState, std::move(lateFuture), {}, /*admissionUnits=*/1);
  lateState->close();
  std::vector<RPCResponse> lateResponses(1);
  lateResponses[0].setPayload(makeTextPayload(std::string(4096, 'y')));
  latePromise.setValue(std::move(lateResponses));
  waitFor([&]() {
    return lateState->operatorSnapshot().numCompletionsSignaled == 1;
  });
  EXPECT_EQ(pool->usedBytes(), baseline);
}

TEST_F(RPCStateTest, payloadAccountingFailureFailsTheQuery) {
  auto root = memory::memoryManager()->addRootPool(
      "rpcResponseAccountingLimit", /*maxCapacity=*/1024);
  auto pool = root->addLeafChild("leaf");
  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state->addPendingRow(
      state, 1, RPCState::RowLocation{0, 0}, std::move(future));
  RPCResponse response;
  response.setPayload(makeTextPayload(std::string(4096, 'x')));
  promise.setValue(std::move(response));
  waitFor([&]() { return state->numInFlight() == 0; });

  state->setNoMoreInput();
  VELOX_ASSERT_THROW(state->isFinished(), "were dropped under memory");
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(RPCStateTest, batchAccountingFailureWakesWithAnError) {
  auto root = memory::memoryManager()->addRootPool(
      "rpcBatchAccountingLimit", /*maxCapacity=*/1024);
  auto pool = root->addLeafChild("leaf");
  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state->addPendingBatch(state, std::move(future), {}, /*admissionUnits=*/1);
  std::vector<RPCResponse> responses(1);
  responses[0].setPayload(makeTextPayload(std::string(4096, 'x')));
  promise.setValue(std::move(responses));
  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    readyBatch = state->tryPollReady();
    return readyBatch.has_value();
  });
  ASSERT_TRUE(readyBatch.has_value());
  EXPECT_TRUE(readyBatch->error.has_value());
  EXPECT_EQ(readyBatch->charge.bytes(), 0);
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(RPCStateTest, errorMessageMemoryIsAccounted) {
  auto root = memory::memoryManager()->addRootPool("rpcErrorAccounting");
  auto pool = root->addLeafChild("leaf");
  auto state = std::make_shared<RPCState>(pool);
  state->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state->addPendingRow(
      state, 1, RPCState::RowLocation{0, 0}, std::move(future));
  std::string message;
  message.reserve(4096);
  message.resize(101, 'e');
  const auto retainedBytes = static_cast<int64_t>(message.capacity());
  RPCResponse response;
  response.setError(
      velox::rpc::RPCErrorKind::kBackendError, std::move(message));
  promise.setValue(std::move(response));
  waitFor([&]() { return state->numInFlight() == 0; });
  EXPECT_EQ(pool->usedBytes(), retainedBytes);

  auto readyRow = state->tryClaimReady();
  ASSERT_TRUE(readyRow.has_value());
  readyRow.reset();
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(RPCStateTest, claimedChargeOutlivesStateCloseWithoutDoubleFree) {
  auto root = memory::memoryManager()->addRootPool("rpcClaimedAccounting");
  auto pool = root->addLeafChild("leaf");
  auto state = std::make_shared<RPCState>(pool);

  auto charge = state->chargePayloadBytes(4096);
  EXPECT_EQ(pool->usedBytes(), 4096);

  state->close();
  EXPECT_EQ(pool->usedBytes(), 4096);

  charge = RPCState::PayloadMemoryCharge{};
  EXPECT_EQ(pool->usedBytes(), 0);
}

TEST_F(RPCStateTest, closeRacingRowCompletionLeavesNoCharge) {
  auto root = memory::memoryManager()->addRootPool("rpcCloseRaceAccounting");
  auto pool = root->addLeafChild("leaf");
  const auto baseline = pool->usedBytes();

  for (int32_t i = 0; i < 100; ++i) {
    auto state = std::make_shared<RPCState>(pool);
    state->setStreamingMode(RPCStreamingMode::kPerRow);
    auto [promise, future] = folly::makePromiseContract<RPCResponse>();
    state->addPendingRow(
        state, i, RPCState::RowLocation{0, i}, std::move(future));

    std::atomic<bool> start{false};
    // Moved into an init-capture rather than captured by reference: a
    // structured binding cannot be captured before Clang 16.
    std::thread completer([&start, p = std::move(promise)]() mutable {
      while (!start.load(std::memory_order_acquire)) {
      }
      RPCResponse response;
      response.setPayload(makeTextPayload(std::string(4096, 'x')));
      p.setValue(std::move(response));
    });
    start.store(true, std::memory_order_release);
    state->close();
    completer.join();
    waitFor([&]() { return state->numInFlight() == 0; });
    EXPECT_EQ(pool->usedBytes(), baseline) << "iteration " << i;
  }
}

TEST_F(RPCStateTest, closeRacingBatchCompletionLeavesNoCharge) {
  auto root =
      memory::memoryManager()->addRootPool("rpcBatchCloseRaceAccounting");
  auto pool = root->addLeafChild("leaf");
  const auto baseline = pool->usedBytes();

  for (int32_t i = 0; i < 100; ++i) {
    auto state = std::make_shared<RPCState>(pool);
    state->setStreamingMode(RPCStreamingMode::kBatch);
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    state->addPendingBatch(state, std::move(future), {}, /*admissionUnits=*/1);

    std::atomic<bool> start{false};
    // Moved into an init-capture rather than captured by reference: a
    // structured binding cannot be captured before Clang 16.
    std::thread completer([&start, p = std::move(promise)]() mutable {
      while (!start.load(std::memory_order_acquire)) {
      }
      std::vector<RPCResponse> responses(1);
      responses[0].setPayload(makeTextPayload(std::string(4096, 'x')));
      p.setValue(std::move(responses));
    });
    start.store(true, std::memory_order_release);
    state->close();
    completer.join();
    waitFor([&]() {
      return state->operatorSnapshot().numCompletionsSignaled == 1;
    });
    EXPECT_EQ(pool->usedBytes(), baseline) << "iteration " << i;
  }
}

TEST_F(RPCStateTest, addAndClaimDirect) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();

  state_->addPendingRow(
      state_, 42, RPCState::RowLocation{0, 0}, std::move(future));

  // Fulfill the promise
  RPCResponse response;
  response.rowId = 42;
  response.setPayload(makeTextPayload("test result"));
  promise.setValue(std::move(response));

  // Wait for async callback to move response into readyRows_
  waitFor([&]() { return state_->numInFlight() == 0; });

  // Now claim
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  auto result = state_->tryClaimOrWait(&waitFuture, &claimedRow);

  ASSERT_EQ(result, RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_EQ(claimedRow->rowId, 42);
  EXPECT_EQ(responseAs<TextPayload>(claimedRow->response).text, "test result");
  EXPECT_FALSE(claimedRow->response.hasError());
}

TEST_F(RPCStateTest, claimOrWaitReturnsFinished) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  // No rows, signal noMoreInput
  state_->setNoMoreInput();

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  auto result = state_->tryClaimOrWait(&waitFuture, &claimedRow);

  EXPECT_EQ(result, RPCState::ClaimResult::kFinished);
}

TEST_F(RPCStateTest, claimOrWaitMustWait) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  // Add a pending row that hasn't completed yet
  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_, 1, RPCState::RowLocation{0, 0}, std::move(future));

  // Try to claim — should return kMustWait because row isn't ready
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  auto result = state_->tryClaimOrWait(&waitFuture, &claimedRow);

  EXPECT_EQ(result, RPCState::ClaimResult::kMustWait);
  EXPECT_FALSE(claimedRow.has_value());

  // Fulfill to clean up
  RPCResponse response;
  response.rowId = 1;
  response.setPayload(makeTextPayload("done"));
  promise.setValue(std::move(response));
}

TEST_F(RPCStateTest, pendingRowCount) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  EXPECT_EQ(state_->numInFlight(), 0);

  auto [promise1, future1] = folly::makePromiseContract<RPCResponse>();
  auto [promise2, future2] = folly::makePromiseContract<RPCResponse>();

  state_->addPendingRow(
      state_, 1, RPCState::RowLocation{0, 0}, std::move(future1));
  EXPECT_EQ(state_->numInFlight(), 1);

  state_->addPendingRow(
      state_, 2, RPCState::RowLocation{0, 1}, std::move(future2));
  EXPECT_EQ(state_->numInFlight(), 2);

  // Fulfill first
  RPCResponse r1;
  r1.rowId = 1;
  r1.setPayload(makeTextPayload("r1"));
  promise1.setValue(std::move(r1));

  waitFor([&]() { return state_->numInFlight() == 1; });
  EXPECT_EQ(state_->numInFlight(), 1);

  // Fulfill second
  RPCResponse r2;
  r2.rowId = 2;
  r2.setPayload(makeTextPayload("r2"));
  promise2.setValue(std::move(r2));

  waitFor([&]() { return state_->numInFlight() == 0; });
  EXPECT_EQ(state_->numInFlight(), 0);
}

// ========== BATCH mode tests ==========

TEST_F(RPCStateTest, basicAddAndPollBatch) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();

  state_->addPendingBatch(state_, std::move(future), {}, /*admissionUnits=*/7);

  // Fulfill the promise
  std::vector<RPCResponse> responses;
  RPCResponse batchResponse;
  batchResponse.rowId = 1;
  batchResponse.setPayload(makeTextPayload("test"));
  responses.push_back(std::move(batchResponse));
  promise.setValue(std::move(responses));

  // Wait for async callback and poll
  waitFor([&]() {
    ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
    std::optional<RPCState::ReadyBatch> readyBatch;
    auto result = state_->tryPollBatchOrWait(&waitFuture, &readyBatch);
    if (result == RPCState::BatchPollResult::kGotBatch) {
      // Verify row locations are empty (we passed empty).
      EXPECT_TRUE(readyBatch->rowLocations.empty());
      EXPECT_EQ(readyBatch->admissionUnits, 7);
      return true;
    }
    return false;
  });
}

TEST_F(RPCStateTest, pollBatchOrWaitFinished) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  // No batches, signal noMoreInput
  state_->setNoMoreInput();

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> readyBatch;
  auto result = state_->tryPollBatchOrWait(&waitFuture, &readyBatch);

  EXPECT_EQ(result, RPCState::BatchPollResult::kFinished);
}

TEST_F(RPCStateTest, pollBatchOrWaitMustWait) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  // Add a pending batch that hasn't completed yet
  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future), {}, /*admissionUnits=*/1);

  // Try to poll — should return kMustWait
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> readyBatch;
  auto result = state_->tryPollBatchOrWait(&waitFuture, &readyBatch);

  EXPECT_EQ(result, RPCState::BatchPollResult::kMustWait);
  EXPECT_FALSE(readyBatch.has_value());

  // Fulfill to clean up
  promise.setValue(std::vector<RPCResponse>{});
}

TEST_F(RPCStateTest, batchErrorHandling) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();

  state_->addPendingBatch(state_, std::move(future), {}, /*admissionUnits=*/1);

  // Set an exception
  promise.setException(std::runtime_error("RPC batch failed"));

  // Wait for async callback and poll
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> readyBatch;

  waitFor([&]() {
    auto result = state_->tryPollBatchOrWait(&waitFuture, &readyBatch);
    return result == RPCState::BatchPollResult::kGotBatch;
  });

  ASSERT_TRUE(readyBatch.has_value());
  ASSERT_TRUE(readyBatch->error.has_value());
  EXPECT_TRUE(readyBatch->error->find("RPC batch failed") != std::string::npos);
  EXPECT_TRUE(readyBatch->responses.empty());
}

// ========== Common tests ==========

TEST_F(RPCStateTest, noMoreInputAndIsFinished) {
  EXPECT_FALSE(state_->isFinished());

  state_->setNoMoreInput();
  EXPECT_TRUE(state_->isFinished());
}

TEST_F(RPCStateTest, isFinishedWithPendingRows) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_, 1, RPCState::RowLocation{0, 0}, std::move(future));

  state_->setNoMoreInput();

  // Not finished while rows are pending
  EXPECT_FALSE(state_->isFinished());

  // Fulfill and claim
  RPCResponse response;
  response.rowId = 1;
  response.setPayload(makeTextPayload("done"));
  promise.setValue(std::move(response));

  waitFor([&]() { return state_->numInFlight() == 0; });

  // Claim the ready row
  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  state_->tryClaimOrWait(&waitFuture, &claimedRow);

  // Now finished (noMoreInput + no pending + no ready)
  EXPECT_TRUE(state_->isFinished());
}

TEST_F(RPCStateTest, inputBatchStorageAndRelease) {
  // Store two input batches.
  std::vector<VectorPtr> columns1; // Empty but valid for testing
  std::vector<VectorPtr> columns2;

  auto batchIdx1 = state_->storeInputBatch(std::move(columns1), 3);
  auto batchIdx2 = state_->storeInputBatch(std::move(columns2), 2);

  EXPECT_EQ(batchIdx1, 0);
  EXPECT_EQ(batchIdx2, 1);

  // Verify we can retrieve columns (empty in this test).
  const auto& cols1 = state_->getInputBatchColumns(batchIdx1);
  EXPECT_TRUE(cols1.empty());

  const auto& cols2 = state_->getInputBatchColumns(batchIdx2);
  EXPECT_TRUE(cols2.empty());

  // Release rows incrementally.
  state_->releaseRows(batchIdx1, 2);
  // Batch 1 still has 1 active row, columns not released yet.
  const auto& cols1After = state_->getInputBatchColumns(batchIdx1);
  // Still accessible (columns empty in this test, but structure is still
  // valid).
  (void)cols1After;

  // Release remaining row — batch should be fully released.
  state_->releaseRows(batchIdx1, 1);

  // Release all rows from batch 2 at once.
  state_->releaseRows(batchIdx2, 2);
}

// A crashed worker's stack: an abandoned AI query tears down the Task while an
// RPC is still in flight; the callback holds a shared_ptr<RPCState>, so the
// retained input vectors outlive the pools that allocated them. Either the
// arbitrator's reservedBytes() == 0 check throws from ~MemoryPoolImpl() and
// terminates the worker, or the late callback frees into pools already gone.
// close() must therefore drop the vectors itself, on the driver thread, and
// not rely on the reference count reaching zero in time.
TEST_F(RPCStateTest, releaseAllInputBatchesDropsVectorsWhileStateIsStillHeld) {
  auto pool = memory::memoryManager()->addLeafPool("releaseAllInputBatches");
  VectorPtr vector = BaseVector::create(BIGINT(), 8, pool.get());

  // Two batches, one still holding "active" rows: the abandoned-query case,
  // where releaseRows() never runs because the rows are never output.
  state_->storeInputBatch(std::vector<VectorPtr>{vector}, /*rowCount=*/4);
  state_->storeInputBatch(std::vector<VectorPtr>{vector}, /*rowCount=*/4);
  EXPECT_GT(vector.use_count(), 1);

  // Stand in for an in-flight callback still holding the state alive.
  auto callbackRef = state_;

  state_->releaseAllInputBatches();

  // The state object survives, but owns none of the pool-backed memory: this
  // test holds the only remaining reference to the vector.
  EXPECT_EQ(vector.use_count(), 1);
  EXPECT_TRUE(state_->getInputBatchColumns(0).empty());
  EXPECT_TRUE(state_->getInputBatchColumns(1).empty());

  // Idempotent — close() may run after rows were already released.
  state_->releaseAllInputBatches();
  EXPECT_EQ(vector.use_count(), 1);
}

TEST_F(RPCStateTest, batchRowLocationsCarriedThrough) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();

  // Pass row locations with the batch.
  std::vector<RPCState::RowLocation> locations = {{0, 5}, {0, 10}, {1, 3}};
  state_->addPendingBatch(
      state_, std::move(future), locations, /*admissionUnits=*/1);

  // Fulfill the promise.
  std::vector<RPCResponse> responses;
  for (int i = 0; i < 3; ++i) {
    RPCResponse r;
    r.rowId = i;
    r.setPayload(makeTextPayload("result_" + std::to_string(i)));
    responses.push_back(std::move(r));
  }
  promise.setValue(std::move(responses));

  // Poll and verify row locations are carried through.
  waitFor([&]() {
    ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
    std::optional<RPCState::ReadyBatch> readyBatch;
    auto result = state_->tryPollBatchOrWait(&waitFuture, &readyBatch);
    if (result == RPCState::BatchPollResult::kGotBatch) {
      EXPECT_EQ(readyBatch->responses.size(), 3);
      EXPECT_EQ(readyBatch->rowLocations.size(), 3);
      EXPECT_EQ(readyBatch->rowLocations[0].batchIndex, 0);
      EXPECT_EQ(readyBatch->rowLocations[0].rowIndex, 5);
      EXPECT_EQ(readyBatch->rowLocations[1].batchIndex, 0);
      EXPECT_EQ(readyBatch->rowLocations[1].rowIndex, 10);
      EXPECT_EQ(readyBatch->rowLocations[2].batchIndex, 1);
      EXPECT_EQ(readyBatch->rowLocations[2].rowIndex, 3);
      return true;
    }
    return false;
  });
}

TEST_F(RPCStateTest, drainReadyRows) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  // Add 3 pending rows and fulfill them all.
  std::vector<folly::Promise<RPCResponse>> promises;
  for (int i = 0; i < 3; ++i) {
    auto [promise, future] = folly::makePromiseContract<RPCResponse>();
    state_->addPendingRow(
        state_,
        i,
        RPCState::RowLocation{0, static_cast<vector_size_t>(i)},
        std::move(future));
    promises.push_back(std::move(promise));
  }

  // Fulfill all 3.
  for (int i = 0; i < 3; ++i) {
    RPCResponse response;
    response.rowId = i;
    response.setPayload(makeTextPayload("result_" + std::to_string(i)));
    promises[i].setValue(std::move(response));
  }

  waitFor([&]() { return state_->numInFlight() == 0; });

  // Drain up to 10 rows (should get all 3).
  std::vector<RPCState::ReadyRow> out;
  state_->drainReadyRows(out, 10);

  ASSERT_EQ(out.size(), 3);
  // Verify all responses are present (order may vary due to async).
  std::set<int64_t> rowIds;
  for (const auto& row : out) {
    rowIds.insert(row.rowId);
  }
  EXPECT_EQ(rowIds.count(0), 1);
  EXPECT_EQ(rowIds.count(1), 1);
  EXPECT_EQ(rowIds.count(2), 1);

  // Drain again — should be empty.
  std::vector<RPCState::ReadyRow> out2;
  state_->drainReadyRows(out2, 10);
  EXPECT_TRUE(out2.empty());
}

TEST_F(RPCStateTest, backpressure) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setMaxWindow(2);

  EXPECT_FALSE(state_->isUnderBackpressure());

  auto [promise1, future1] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_, 1, RPCState::RowLocation{0, 0}, std::move(future1));
  EXPECT_FALSE(state_->isUnderBackpressure());

  auto [promise2, future2] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_, 2, RPCState::RowLocation{0, 1}, std::move(future2));
  EXPECT_TRUE(state_->isUnderBackpressure());

  // Fulfill one to relieve backpressure
  RPCResponse r1;
  r1.rowId = 1;
  r1.setPayload(makeTextPayload("r1"));
  promise1.setValue(std::move(r1));

  waitFor([&]() { return !state_->isUnderBackpressure(); });
  EXPECT_FALSE(state_->isUnderBackpressure());

  // Clean up
  RPCResponse r2;
  r2.rowId = 2;
  r2.setPayload(makeTextPayload("r2"));
  promise2.setValue(std::move(r2));
}

TEST_F(RPCStateTest, batchBackpressureUsesWindow) {
  // BATCH defaults to a window of 2 in-flight batches; the unified
  // isUnderBackpressure() gates on it the same way PER_ROW gates on rows.
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  EXPECT_FALSE(state_->isUnderBackpressure());

  auto [promise1, future1] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future1), {}, /*admissionUnits=*/1);
  EXPECT_FALSE(state_->isUnderBackpressure()); // 1 < 2

  auto [promise2, future2] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future2), {}, /*admissionUnits=*/1);
  EXPECT_TRUE(state_->isUnderBackpressure()); // 2 >= 2

  // Clean up the pending batch futures.
  promise1.setValue(std::vector<RPCResponse>{});
  promise2.setValue(std::vector<RPCResponse>{});
}

TEST_F(RPCStateTest, perRowWindowShrinksOnOverload) {
  // Overload (onUnitError) halves the gradient window, tightening backpressure;
  // user-data errors never call onUnitError so the window is unaffected.
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setMaxWindow(8); // window {8, 8}

  std::vector<folly::Promise<RPCResponse>> promises;
  for (int i = 0; i < 4; ++i) {
    auto [promise, future] = folly::makePromiseContract<RPCResponse>();
    state_->addPendingRow(
        state_,
        i,
        RPCState::RowLocation{0, static_cast<vector_size_t>(i)},
        std::move(future));
    promises.push_back(std::move(promise));
  }
  // 4 in-flight rows is below the window of 8.
  EXPECT_FALSE(state_->isUnderBackpressure());

  // One overload signal halves the window to 4 — now 4 in-flight is at limit.
  state_->onUnitError();
  EXPECT_TRUE(state_->isUnderBackpressure());

  // Clean up.
  for (int i = 0; i < 4; ++i) {
    RPCResponse response;
    response.rowId = i;
    response.setPayload(makeTextPayload("x"));
    promises[i].setValue(std::move(response));
  }
}

TEST_F(RPCStateTest, perRowWindowRecoversViaSamples) {
  // After an overload shrink, flat-latency samples grow the gradient window
  // back and relieve backpressure — the recovery path for the unified learner.
  state_->setStreamingMode(RPCStreamingMode::kPerRow);
  state_->setMaxWindow(8);
  state_->onUnitError(); // window 8 -> 4

  std::vector<folly::Promise<RPCResponse>> promises;
  for (int i = 0; i < 4; ++i) {
    auto [promise, future] = folly::makePromiseContract<RPCResponse>();
    state_->addPendingRow(
        state_,
        i,
        RPCState::RowLocation{0, static_cast<vector_size_t>(i)},
        std::move(future));
    promises.push_back(std::move(promise));
  }
  // 4 in-flight at window 4 -> backpressure.
  EXPECT_TRUE(state_->isUnderBackpressure());

  // Two flat-latency sample windows: the first sets the baseline, the second
  // grows the window (4 -> 6); 4 in-flight is now below it.
  for (int i = 0; i < 16; ++i) {
    state_->onUnitSample(1'000'000);
  }
  EXPECT_FALSE(state_->isUnderBackpressure());

  for (int i = 0; i < 4; ++i) {
    RPCResponse response;
    response.rowId = i;
    response.setPayload(makeTextPayload("x"));
    promises[i].setValue(std::move(response));
  }
}

TEST_F(RPCStateTest, batchInFlightTracksBatchCount) {
  // In BATCH mode inFlight_ counts batches and is decremented only at poll
  // time, not when the batch future completes.
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  state_->setMaxWindow(10); // high enough to avoid backpressure here

  std::vector<folly::Promise<std::vector<RPCResponse>>> promises;
  for (int i = 0; i < 3; ++i) {
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    state_->addPendingBatch(
        state_, std::move(future), {}, /*admissionUnits=*/1);
    promises.push_back(std::move(promise));
  }
  EXPECT_EQ(state_->numInFlight(), 3);

  // Completing a batch future does NOT decrement until it is polled.
  promises[0].setValue(std::vector<RPCResponse>{});
  EXPECT_EQ(state_->numInFlight(), 3);

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    return state_->tryPollBatchOrWait(&waitFuture, &readyBatch) ==
        RPCState::BatchPollResult::kGotBatch;
  });
  EXPECT_EQ(state_->numInFlight(), 2);

  promises[1].setValue(std::vector<RPCResponse>{});
  promises[2].setValue(std::vector<RPCResponse>{});
}

TEST_F(RPCStateTest, batchIsFinishedWithInFlightBatch) {
  // isFinished() keys on inFlight_ == 0; a BATCH in flight keeps it false until
  // the batch is polled (which is what drives inFlight_--).
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future), {}, /*admissionUnits=*/1);
  state_->setNoMoreInput();
  EXPECT_FALSE(state_->isFinished());

  // Completed but not yet polled -> still in flight.
  promise.setValue(std::vector<RPCResponse>{});
  EXPECT_FALSE(state_->isFinished());

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    return state_->tryPollBatchOrWait(&waitFuture, &readyBatch) ==
        RPCState::BatchPollResult::kGotBatch;
  });
  EXPECT_TRUE(state_->isFinished());
}

TEST_F(RPCStateTest, batchWindowGrowsViaGradientSamples) {
  // BATCH uses a latency-gradient window starting at 2. Flat-latency samples
  // let it learn upward past the start, relieving the backpressure that the
  // initial window of 2 imposes on 2 in-flight batches.
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  std::vector<folly::Promise<std::vector<RPCResponse>>> promises;
  for (int i = 0; i < 2; ++i) {
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    state_->addPendingBatch(
        state_, std::move(future), {}, /*admissionUnits=*/1);
    promises.push_back(std::move(promise));
  }
  // 2 in-flight at the starting window of 2 -> backpressure.
  EXPECT_TRUE(state_->isUnderBackpressure());

  // Two full sample windows (kSamplesPerWindow = 8) of flat latency: the first
  // sets the baseline, the second grows the window above 2.
  for (int i = 0; i < 16; ++i) {
    state_->onUnitSample(1'000'000);
  }
  EXPECT_FALSE(state_->isUnderBackpressure());

  for (auto& promise : promises) {
    promise.setValue(std::vector<RPCResponse>{});
  }
}

TEST_F(RPCStateTest, batchWindowGrowsOnCleanCompletions) {
  // BATCH async-job successes carry no usable latency, so they grow the
  // window additively on completion count: 8 clean units lift 2 -> 3.
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/0,
      /*batchStartWindow=*/0,
      /*asyncGrowthMaxWindow=*/64);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 2);
  state_->onUnitCleanCompletions(7);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 2);
  state_->onUnitCleanCompletions(1);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 3);
}

TEST_F(RPCStateTest, perRowWindowIgnoresCleanCompletions) {
  // PER_ROW windows start at their ceiling and must only shrink: clean
  // completions and the BATCH-only cap must not affect them.
  state_->setStreamingMode(
      RPCStreamingMode::kPerRow,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/0,
      /*batchStartWindow=*/0,
      /*asyncGrowthMaxWindow=*/-1);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 100);
  // Shrink first: at the ceiling the test cannot tell a missing BATCH-only
  // check from a working one, since the window cannot grow past it anyway.
  state_->onUnitError();
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 50);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 50);
}

TEST_F(RPCStateTest, cleanCompletionsRespectMaxWindow) {
  // Growth stops at maxWindow_ even with a higher clean cap configured.
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/8,
      /*batchStartWindow=*/0,
      /*asyncGrowthMaxWindow=*/64);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 8);
}

TEST_F(RPCStateTest, cleanGrowthDisabledByDefault) {
  // async_growth_max_window defaults to 0: clean completions are ignored and
  // the BATCH window does not recover after overload.
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/64,
      /*batchStartWindow=*/8);
  state_->onUnitError();
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
}

TEST_F(RPCStateTest, cleanGrowthRespectsAsyncCap) {
  // The async-growth cap binds below maxWindow_.
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/0,
      /*batchStartWindow=*/0,
      /*asyncGrowthMaxWindow=*/8);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 8);
}

TEST_F(RPCStateTest, cleanGrowthCapMayEqualEffectiveStart) {
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/64,
      /*batchStartWindow=*/8,
      /*asyncGrowthMaxWindow=*/8);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 8);
}

TEST_F(RPCStateTest, cleanGrowthCapUsesClampedEffectiveStart) {
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/4,
      /*batchStartWindow=*/8,
      /*asyncGrowthMaxWindow=*/4);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
  state_->onUnitError();
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 2);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
}

TEST_F(RPCStateTest, cleanGrowthCapBelowStartClampsToStart) {
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/64,
      /*batchStartWindow=*/8,
      /*asyncGrowthMaxWindow=*/7);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 8);
  state_->onUnitError();
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 8);
}

TEST_F(RPCStateTest, cleanGrowthCapAboveStartAllowsRecovery) {
  state_->setStreamingMode(
      RPCStreamingMode::kBatch,
      /*minWindow=*/1,
      /*stepCoef=*/1.0,
      /*maxWindow=*/64,
      /*batchStartWindow=*/8,
      /*asyncGrowthMaxWindow=*/16);
  state_->onUnitError();
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 4);
  state_->onUnitCleanCompletions(1'000);
  EXPECT_EQ(state_->operatorSnapshot().windowLimit, 16);
}

TEST_F(RPCStateTest, setMaxWindowOverridesModeDefault) {
  // setMaxWindow (called after setStreamingMode) overrides the per-mode default
  // ceiling: 4 in-flight batches would backpressure at the default 2, not at 5.
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  state_->setMaxWindow(5);

  std::vector<folly::Promise<std::vector<RPCResponse>>> promises;
  for (int i = 0; i < 4; ++i) {
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    state_->addPendingBatch(
        state_, std::move(future), {}, /*admissionUnits=*/1);
    promises.push_back(std::move(promise));
  }
  EXPECT_FALSE(state_->isUnderBackpressure());

  auto [promise5, future5] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future5), {}, /*admissionUnits=*/1);
  EXPECT_TRUE(state_->isUnderBackpressure()); // 5 >= 5

  for (auto& promise : promises) {
    promise.setValue(std::vector<RPCResponse>{});
  }
  promise5.setValue(std::vector<RPCResponse>{});
}

TEST_F(RPCStateTest, perRowErrorDecrementsInFlight) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_, 7, RPCState::RowLocation{0, 0}, std::move(future));
  EXPECT_EQ(state_->numInFlight(), 1);

  promise.setException(std::runtime_error("boom"));
  waitFor([&]() { return state_->numInFlight() == 0; });

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  auto result = state_->tryClaimOrWait(&waitFuture, &claimedRow);
  ASSERT_EQ(result, RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_EQ(
      claimedRow->response.errorKind(),
      velox::rpc::RPCErrorKind::kBackendError);
  EXPECT_NE(
      claimedRow->response.error().message.find("boom"), std::string::npos);
}

TEST_F(RPCStateTest, perRowClassifiedErrorWakesWaiter) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();
  state_->addPendingRow(
      state_, 7, RPCState::RowLocation{0, 0}, std::move(future));

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimedRow;
  EXPECT_EQ(
      state_->tryClaimOrWait(&waitFuture, &claimedRow),
      RPCState::ClaimResult::kMustWait);
  EXPECT_FALSE(waitFuture.isReady());

  promise.setException(
      RpcClassifiedError{
          velox::rpc::RPCErrorKind::kRateLimited, "429 from backend"});
  waitFor([&]() { return waitFuture.isReady(); });
  EXPECT_EQ(state_->numInFlight(), 0);

  ContinueFuture nextWaitFuture{ContinueFuture::makeEmpty()};
  EXPECT_EQ(
      state_->tryClaimOrWait(&nextWaitFuture, &claimedRow),
      RPCState::ClaimResult::kClaimed);
  ASSERT_TRUE(claimedRow.has_value());
  EXPECT_EQ(
      claimedRow->response.errorKind(), velox::rpc::RPCErrorKind::kRateLimited);
  EXPECT_NE(
      claimedRow->response.error().message.find("429 from backend"),
      std::string::npos);
}

TEST_F(RPCStateTest, batchRttExcludesPollDelay) {
  // rttNs is stamped when the batch future completes (in the callback), not at
  // poll time, so a long delay between completion and poll must NOT inflate it.
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future), {}, /*admissionUnits=*/1);

  // Complete immediately (dispatch->completion is ~microseconds here).
  std::vector<RPCResponse> responses(1);
  responses[0].setPayload(makeTextPayload("ok"));
  promise.setValue(std::move(responses));

  // Simulate a long poll delay AFTER completion; this is excluded from rttNs.
  constexpr int64_t kPollDelayMs = 100;
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(kPollDelayMs));

  ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyBatch> readyBatch;
  waitFor([&]() {
    return state_->tryPollBatchOrWait(&waitFuture, &readyBatch) ==
        RPCState::BatchPollResult::kGotBatch;
  });
  ASSERT_TRUE(readyBatch.has_value());
  EXPECT_GE(readyBatch->rttNs, 0);
  EXPECT_LT(
      readyBatch->rttNs,
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::milliseconds(kPollDelayMs))
          .count());
}

TEST_F(RPCStateTest, batchWaiterNotOrphanedByCompletionDrainRace) {
  // Regression guard for the completion-drain race in tryPollBatchOrWait. The
  // batch completion callback stamps completionTimeNs and drains the waiter
  // list under mutex_ BEFORE the batch future becomes ready (the future readies
  // only when the callback returns). A poller that registers a waiter in that
  // window
  // -- drain already passed, future not yet ready -- would be orphaned, and at
  // a window of 1 (the completing batch is the last in flight) the
  // single-consumer driver would park forever. Race a completing batch against
  // a poller many times and assert every park is fulfilled promptly (self-woken
  // via the completionTimeNs guard), never orphaned.
  constexpr int kIterations = 500;
  for (int iter = 0; iter < kIterations; ++iter) {
    auto state = std::make_shared<RPCState>(testPool());
    state->setStreamingMode(RPCStreamingMode::kBatch);
    state->setMaxWindow(1); // the completing batch is the last in flight

    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    state->addPendingBatch(state, std::move(future), {}, /*admissionUnits=*/1);

    // Complete on another thread so its inline drain races the poller below;
    // the atomic gate aligns the two so the interleaving is exercised.
    std::atomic<bool> go{false};
    std::thread completer([&go, p = std::move(promise)]() mutable {
      while (!go.load(std::memory_order_acquire)) {
      }
      p.setValue(std::vector<RPCResponse>{});
    });

    go.store(true, std::memory_order_release);
    bool claimed = false;
    while (!claimed) {
      ContinueFuture waitFuture{ContinueFuture::makeEmpty()};
      std::optional<RPCState::ReadyBatch> readyBatch;
      switch (state->tryPollBatchOrWait(&waitFuture, &readyBatch)) {
        case RPCState::BatchPollResult::kGotBatch:
          claimed = true;
          break;
        case RPCState::BatchPollResult::kMustWait: {
          // A legitimately-registered or self-woken waiter is fulfilled
          // promptly; an orphaned one never is. Bound the wait so a regression
          // fails cleanly instead of hanging the test.
          auto f = std::move(waitFuture);
          auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(10);
          while (!f.isReady()) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline)
                << "waiter orphaned by completion-drain race at iter " << iter;
            /* sleep override */
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          break;
        }
        case RPCState::BatchPollResult::kFinished:
          FAIL() << "unexpected kFinished while a batch is in flight";
      }
    }
    completer.join();
  }
}

// A row that completed but could not be queued is gone. Finishing normally
// would hand the query fewer rows than it was given and call that success,
// which nothing downstream can detect -- so the finish path must fail instead.
TEST_F(RPCStateTest, droppedRowFailsRatherThanShorteningTheResult) {
  auto state = std::make_shared<RPCState>(testPool());
  state->testingDropCompletedRow();
  state->setNoMoreInput();

  ContinueFuture future{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimed;
  VELOX_ASSERT_THROW(
      state->tryClaimOrWait(&future, &claimed), "were dropped under memory");
}

// Both out-params are required. Stating the precondition keeps a null from
// becoming a segfault deep inside the claim path.
TEST_F(RPCStateTest, claimRejectsNullOutParams) {
  auto state = std::make_shared<RPCState>(testPool());
  ContinueFuture future{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimed;

  VELOX_ASSERT_THROW(
      state->tryClaimOrWait(nullptr, &claimed), "requires a future out-param");
  VELOX_ASSERT_THROW(
      state->tryClaimOrWait(&future, nullptr),
      "requires a claimedRow out-param");
}

// The barrier drain asks isFinished() directly rather than going through
// tryClaimOrWait(), so it would otherwise report a clean finish over a short
// result. Both terminal paths have to answer the same way.
TEST_F(RPCStateTest, droppedRowAlsoFailsTheDrainFinishCheck) {
  auto state = std::make_shared<RPCState>(testPool());
  state->testingDropCompletedRow();
  state->setNoMoreInput();

  VELOX_ASSERT_THROW(state->isFinished(), "were dropped under memory");
}

// Without a drop the same finish path is an ordinary completion, so the check
// above cannot be satisfied by simply never finishing.
TEST_F(RPCStateTest, finishesNormallyWhenNoRowWasDropped) {
  auto state = std::make_shared<RPCState>(testPool());
  state->setNoMoreInput();

  ContinueFuture future{ContinueFuture::makeEmpty()};
  std::optional<RPCState::ReadyRow> claimed;
  EXPECT_EQ(
      state->tryClaimOrWait(&future, &claimed),
      RPCState::ClaimResult::kFinished);
  EXPECT_TRUE(state->isFinished());
}

} // namespace
} // namespace facebook::velox::exec::rpc
