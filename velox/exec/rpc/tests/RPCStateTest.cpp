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
  EXPECT_EQ(state_->numPendingRows(), 1);

  // Fulfill the promise
  RPCResponse response;
  response.rowId = 42;
  response.result = "test result";
  promise.setValue(std::move(response));

  // Wait for async callback to move response into readyRows_
  waitFor([&]() { return state_->numPendingRows() == 0; });

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
  waitFor([&]() { return state_->numPendingRows() == 0; });

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

  EXPECT_EQ(state_->numPendingRows(), 0);

  auto [promise1, future1] = folly::makePromiseContract<RPCResponse>();
  auto [promise2, future2] = folly::makePromiseContract<RPCResponse>();

  state_->addPendingRow(
      state_, 1, RPCState::RowLocation{0, 0}, std::move(future1));
  EXPECT_EQ(state_->numPendingRows(), 1);

  state_->addPendingRow(
      state_, 2, RPCState::RowLocation{0, 1}, std::move(future2));
  EXPECT_EQ(state_->numPendingRows(), 2);

  // Fulfill first
  RPCResponse r1;
  r1.rowId = 1;
  r1.result = "r1";
  promise1.setValue(std::move(r1));

  waitFor([&]() { return state_->numPendingRows() == 1; });
  EXPECT_EQ(state_->numPendingRows(), 1);

  // Fulfill second
  RPCResponse r2;
  r2.rowId = 2;
  r2.result = "r2";
  promise2.setValue(std::move(r2));

  waitFor([&]() { return state_->numPendingRows() == 0; });
  EXPECT_EQ(state_->numPendingRows(), 0);
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

  waitFor([&]() { return state_->numPendingRows() == 0; });

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

  waitFor([&]() { return state_->numPendingRows() == 0; });

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
  state_->setMaxPendingRows(2);

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

} // namespace
} // namespace facebook::velox::exec::rpc
