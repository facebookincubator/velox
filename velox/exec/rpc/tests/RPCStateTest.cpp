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

#include <folly/futures/Promise.h>
#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <set>
#include <thread>

namespace facebook::velox::exec::rpc {
namespace {

class RPCStateTest : public testing::Test {
 protected:
  void SetUp() override {
    state_ = std::make_shared<RPCState>();
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
  response.result = "test result";
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
  EXPECT_EQ(claimedRow->response.result, "test result");
}

TEST_F(RPCStateTest, addAndClaimDirect) {
  state_->setStreamingMode(RPCStreamingMode::kPerRow);

  auto [promise, future] = folly::makePromiseContract<RPCResponse>();

  state_->addPendingRow(
      state_, 42, RPCState::RowLocation{0, 0}, std::move(future));

  // Fulfill the promise
  RPCResponse response;
  response.rowId = 42;
  response.result = "test result";
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
  EXPECT_EQ(claimedRow->response.result, "test result");
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
  response.result = "done";
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
  r1.result = "r1";
  promise1.setValue(std::move(r1));

  waitFor([&]() { return state_->numInFlight() == 1; });
  EXPECT_EQ(state_->numInFlight(), 1);

  // Fulfill second
  RPCResponse r2;
  r2.rowId = 2;
  r2.result = "r2";
  promise2.setValue(std::move(r2));

  waitFor([&]() { return state_->numInFlight() == 0; });
  EXPECT_EQ(state_->numInFlight(), 0);
}

// ========== BATCH mode tests ==========

TEST_F(RPCStateTest, basicAddAndPollBatch) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();

  state_->addPendingBatch(state_, std::move(future), {});

  // Fulfill the promise
  std::vector<RPCResponse> responses;
  RPCResponse batchResponse;
  batchResponse.rowId = 1;
  batchResponse.result = "test";
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
  state_->addPendingBatch(state_, std::move(future), {});

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

  state_->addPendingBatch(state_, std::move(future), {});

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
  response.result = "done";
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

TEST_F(RPCStateTest, batchRowLocationsCarriedThrough) {
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();

  // Pass row locations with the batch.
  std::vector<RPCState::RowLocation> locations = {{0, 5}, {0, 10}, {1, 3}};
  state_->addPendingBatch(state_, std::move(future), locations);

  // Fulfill the promise.
  std::vector<RPCResponse> responses;
  for (int i = 0; i < 3; ++i) {
    RPCResponse r;
    r.rowId = i;
    r.result = "result_" + std::to_string(i);
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
    response.result = "result_" + std::to_string(i);
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
  r1.result = "r1";
  promise1.setValue(std::move(r1));

  waitFor([&]() { return !state_->isUnderBackpressure(); });
  EXPECT_FALSE(state_->isUnderBackpressure());

  // Clean up
  RPCResponse r2;
  r2.rowId = 2;
  r2.result = "r2";
  promise2.setValue(std::move(r2));
}

TEST_F(RPCStateTest, batchBackpressureUsesWindow) {
  // BATCH defaults to a window of 2 in-flight batches; the unified
  // isUnderBackpressure() gates on it the same way PER_ROW gates on rows.
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  EXPECT_FALSE(state_->isUnderBackpressure());

  auto [promise1, future1] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future1), {});
  EXPECT_FALSE(state_->isUnderBackpressure()); // 1 < 2

  auto [promise2, future2] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future2), {});
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
    response.result = "x";
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
    response.result = "x";
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
    state_->addPendingBatch(state_, std::move(future), {});
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
  state_->addPendingBatch(state_, std::move(future), {});
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
    state_->addPendingBatch(state_, std::move(future), {});
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

TEST_F(RPCStateTest, setMaxWindowOverridesModeDefault) {
  // setMaxWindow (called after setStreamingMode) overrides the per-mode default
  // ceiling: 4 in-flight batches would backpressure at the default 2, not at 5.
  state_->setStreamingMode(RPCStreamingMode::kBatch);
  state_->setMaxWindow(5);

  std::vector<folly::Promise<std::vector<RPCResponse>>> promises;
  for (int i = 0; i < 4; ++i) {
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    state_->addPendingBatch(state_, std::move(future), {});
    promises.push_back(std::move(promise));
  }
  EXPECT_FALSE(state_->isUnderBackpressure());

  auto [promise5, future5] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future5), {});
  EXPECT_TRUE(state_->isUnderBackpressure()); // 5 >= 5

  for (auto& promise : promises) {
    promise.setValue(std::vector<RPCResponse>{});
  }
  promise5.setValue(std::vector<RPCResponse>{});
}

TEST_F(RPCStateTest, perRowErrorDecrementsInFlight) {
  // A row future that completes with an exception still routes through
  // completeRow (the deferError path), so inFlight_ returns to 0 and the row is
  // claimable with an error set — the PER_ROW counterpart of
  // batchErrorHandling.
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
  EXPECT_TRUE(claimedRow->response.hasError());
}

TEST_F(RPCStateTest, batchRttExcludesPollDelay) {
  // rttNs is stamped when the batch future completes (in the callback), not at
  // poll time, so a long delay between completion and poll must NOT inflate it.
  state_->setStreamingMode(RPCStreamingMode::kBatch);

  auto [promise, future] =
      folly::makePromiseContract<std::vector<RPCResponse>>();
  state_->addPendingBatch(state_, std::move(future), {});

  // Complete immediately (dispatch->completion is ~microseconds here).
  std::vector<RPCResponse> responses(1);
  responses[0].result = "ok";
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

} // namespace
} // namespace facebook::velox::exec::rpc
