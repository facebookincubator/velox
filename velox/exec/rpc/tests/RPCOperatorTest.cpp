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

/// RPCOperatorTest - End-to-end task-level test for RPCOperator.
///
/// Runs a full Velox Task/Driver pipeline: Values → RPCNode → output.
/// Verifies that RPCPlanNodeTranslator, RPCOperator, RPCState, and
/// AsyncRPCFunction wire together correctly through the execution engine.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/rpc/DeferredRpcPhaseFunction.h"
#include "velox/exec/rpc/RPCOperator.h"
#include "velox/exec/rpc/RPCPlanNodeTranslator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
#include "velox/exec/rpc/tests/DemoBatchRPCFunction.h"
#include "velox/exec/rpc/tests/DemoRPCFunction.h"
#include "velox/exec/rpc/tests/TextOutput.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"
#include "velox/vector/DecodedVector.h"

#include <folly/container/F14Map.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include "velox/exec/Cursor.h"
#include "velox/exec/Task.h"
#include "velox/exec/rpc/RPCOperatorStatNames.h"

namespace facebook::velox::exec::rpc {

using namespace facebook::velox::exec::test;

namespace {

class OperatorReceipt final : public JobReceipt {
 public:
  OperatorReceipt(
      Outcome state,
      velox::rpc::RPCErrorKind kind,
      bool backend,
      int64_t inputId = 0)
      : state_(state), kind_(kind), backend_(backend), inputId_(inputId) {}
  Outcome outcome() const override {
    return state_;
  }
  velox::rpc::RPCErrorKind errorKind() const override {
    return kind_;
  }
  bool reportsBackendOutcome() const override {
    return backend_;
  }
  std::string_view error() const override {
    return "simulated receipt failure";
  }
  std::optional<int64_t> enqueueTime() const override {
    return std::nullopt;
  }
  std::optional<int64_t> absoluteDeadline() const override {
    return std::nullopt;
  }
  int64_t retainedBytes() const noexcept override {
    return sizeof(*this);
  }
  int64_t inputId() const {
    return inputId_;
  }

 private:
  const Outcome state_;
  const velox::rpc::RPCErrorKind kind_;
  const bool backend_;
  const int64_t inputId_;
};

class OperatorReceiptCodec final : public JobReceiptCodec {
 public:
  RowTypePtr receiptType() const override {
    return ROW({{"state", BIGINT()}, {"kind", BIGINT()}, {"input", BIGINT()}});
  }
  std::string signature() const override {
    return "row(state bigint,kind bigint,input bigint)";
  }
  JobReceiptPtr decode(const VectorPtr& input, vector_size_t row)
      const override {
    DecodedVector decoded(*input);
    const auto* values = decoded.base()->asChecked<RowVector>();
    const auto index = decoded.index(row);
    DecodedVector state(*values->childAt(0));
    DecodedVector kind(*values->childAt(1));
    DecodedVector inputId(*values->childAt(2));
    return std::make_shared<OperatorReceipt>(
        static_cast<JobReceipt::Outcome>(state.valueAt<int64_t>(index)),
        static_cast<velox::rpc::RPCErrorKind>(kind.valueAt<int64_t>(index)),
        false,
        inputId.valueAt<int64_t>(index));
  }
  VectorPtr encode(const ReceiptBatch& receipts, memory::MemoryPool* pool)
      const override {
    auto result =
        BaseVector::create<RowVector>(receiptType(), receipts.size(), pool);
    for (vector_size_t row = 0; row < receipts.size(); ++row) {
      result->childAt(0)->asFlatVector<int64_t>()->set(
          row, static_cast<int64_t>(receipts[row]->outcome()));
      result->childAt(1)->asFlatVector<int64_t>()->set(
          row, static_cast<int64_t>(receipts[row]->errorKind()));
      const auto* receipt =
          dynamic_cast<const OperatorReceipt*>(receipts[row].get());
      VELOX_CHECK_NOT_NULL(receipt);
      result->childAt(2)->asFlatVector<int64_t>()->set(row, receipt->inputId());
    }
    return result;
  }
  JobReceiptPtr fromError(velox::rpc::RPCErrorKind kind, std::string_view)
      const override {
    return std::make_shared<OperatorReceipt>(
        kind == velox::rpc::RPCErrorKind::kNullInput
            ? JobReceipt::Outcome::kNull
            : JobReceipt::Outcome::kFailed,
        kind,
        false);
  }
};

struct ReceiptOperatorState {
  velox::rpc::RPCErrorKind submitError{velox::rpc::RPCErrorKind::kNone};
  velox::rpc::RPCErrorKind pollError{velox::rpc::RPCErrorKind::kNone};
  double floor{1};
  bool holdPollPair{false};
  bool reverseCollection{false};
  int32_t pollRequests{0};
  std::vector<folly::Promise<ReceiptBatch>> promises;
  std::vector<ReceiptBatch> heldReceipts;
};

class ReceiptOperatorBackend final : public DeferredRpcBackend {
 public:
  ReceiptOperatorBackend(
      DeferredRpcPhase phase,
      std::shared_ptr<ReceiptOperatorState> state)
      : phase_(phase), state_(std::move(state)) {}
  std::string name() const override {
    return "operator_receipts";
  }
  TypePtr resultType() const override {
    return VARCHAR();
  }
  const JobReceiptCodec& receiptCodec() const override {
    return codec_;
  }
  std::string admissionKey() const override {
    return "";
  }
  double configuredAdaptiveFloor() const override {
    return state_->floor;
  }
  bool requiresRowInspectionBeforeAdmission() const override {
    return false;
  }
  std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>&) override {
    std::vector<vector_size_t> accepted;
    rows.applyToSelected([&](vector_size_t row) {
      ++pending_;
      accepted.push_back(row);
    });
    return accepted;
  }
  int32_t pendingBatchSize() const override {
    return pending_;
  }
  folly::SemiFuture<ReceiptBatch> submitPendingBatch(int32_t maxRows) override {
    const auto count = maxRows > 0 ? std::min(maxRows, pending_) : pending_;
    pending_ -= count;
    ReceiptBatch receipts;
    for (int32_t row = 0; row < count; ++row) {
      receipts.push_back(
          std::make_shared<OperatorReceipt>(
              state_->submitError == velox::rpc::RPCErrorKind::kNone
                  ? JobReceipt::Outcome::kSubmitted
                  : JobReceipt::Outcome::kFailed,
              state_->submitError,
              true,
              nextInput_++));
    }
    return folly::makeSemiFuture(std::move(receipts));
  }
  folly::SemiFuture<ReceiptBatch> poll(ReceiptBatch receipts) override {
    bool dispatched{false};
    for (auto& receipt : receipts) {
      if (receipt->outcome() == JobReceipt::Outcome::kSubmitted) {
        dispatched = true;
        const auto* input = dynamic_cast<const OperatorReceipt*>(receipt.get());
        VELOX_CHECK_NOT_NULL(input);
        receipt = std::make_shared<OperatorReceipt>(
            state_->pollError == velox::rpc::RPCErrorKind::kNone
                ? JobReceipt::Outcome::kReady
                : JobReceipt::Outcome::kFailed,
            state_->pollError,
            true,
            input->inputId());
      }
    }
    if (!dispatched) {
      return folly::makeSemiFuture(std::move(receipts));
    }
    ++state_->pollRequests;
    if (!state_->holdPollPair) {
      return folly::makeSemiFuture(std::move(receipts));
    }
    auto [promise, future] = folly::makePromiseContract<ReceiptBatch>();
    state_->promises.push_back(std::move(promise));
    state_->heldReceipts.push_back(std::move(receipts));
    if (state_->promises.size() == 2) {
      auto promises = std::move(state_->promises);
      auto values = std::move(state_->heldReceipts);
      for (size_t index = 0; index < promises.size(); ++index) {
        promises[index].setValue(std::move(values[index]));
      }
    }
    return std::move(future);
  }
  folly::SemiFuture<std::vector<RPCResponse>> collect(
      ReceiptBatch receipts) override {
    std::vector<RPCResponse> responses;
    for (size_t row = 0; row < receipts.size(); ++row) {
      if (receipts[row]->errorKind() != velox::rpc::RPCErrorKind::kNone) {
        auto response = RPCResponse::failed(
            row, receipts[row]->errorKind(), "historical receipt failure");
        response.setCompletionOutcome(receipts[row]->errorKind(), false);
        responses.push_back(std::move(response));
      } else {
        const auto* receipt =
            dynamic_cast<const OperatorReceipt*>(receipts[row].get());
        VELOX_CHECK_NOT_NULL(receipt);
        responses.push_back(
            RPCResponse::ok(
                row,
                makeTextPayload(fmt::format("value-{}", receipt->inputId()))));
      }
    }
    if (state_->reverseCollection) {
      std::reverse(responses.begin(), responses.end());
    }
    return folly::makeSemiFuture(std::move(responses));
  }
  int32_t admissionUnitsForReceipts(
      const ReceiptBatch& receipts) const override {
    return std::any_of(
               receipts.begin(),
               receipts.end(),
               [&](const auto& receipt) {
                 return receipt->outcome() == JobReceipt::Outcome::kSubmitted ||
                     (phase_ == DeferredRpcPhase::kCollect &&
                      receipt->outcome() == JobReceipt::Outcome::kReady);
               })
        ? 1
        : 0;
  }
  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    return buildTextOutput(responses, pool);
  }
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    bool hasBackendOutcome{false};
    bool hasFailure{false};
    for (const auto& response : responses) {
      if (!response.isBackendOutcome()) {
        continue;
      }
      hasBackendOutcome = true;
      const auto kind = response.completionErrorKind();
      if (velox::rpc::errorCategory(kind) ==
          velox::rpc::RPCErrorCategory::kOverload) {
        return CongestionSignal::kOverloaded;
      }
      hasFailure |= kind != velox::rpc::RPCErrorKind::kNone;
    }
    return !hasBackendOutcome ? CongestionSignal::kNone
        : hasFailure          ? CongestionSignal::kNonOverloadError
                              : CongestionSignal::kSuccessNoLatency;
  }
  void stop() noexcept override {
    pending_ = 0;
  }

 private:
  const DeferredRpcPhase phase_;
  const std::shared_ptr<ReceiptOperatorState> state_;
  OperatorReceiptCodec codec_;
  int32_t pending_{0};
  int64_t nextInput_{0};
};

memory::MemoryPool::Stats rpcOperatorMemoryStats(
    const std::shared_ptr<Task>& task,
    const core::PlanNodeId& rpcNodeId) {
  memory::MemoryPool* nodePool{nullptr};
  task->pool()->visitChildren([&](memory::MemoryPool* childPool) {
    if (childPool->name() == fmt::format("node.{}", rpcNodeId)) {
      VELOX_CHECK_NULL(nodePool, "Expected one RPC plan-node memory pool");
      nodePool = childPool;
    }
    return true;
  });
  VELOX_CHECK_NOT_NULL(nodePool, "RPC plan-node memory pool was not found");

  memory::MemoryPool* rpcPool{nullptr};
  nodePool->visitChildren([&](memory::MemoryPool* childPool) {
    if (childPool->isLeaf() && childPool->name().ends_with(".RPC")) {
      VELOX_CHECK_NULL(rpcPool, "Expected one RPC operator memory pool");
      rpcPool = childPool;
    }
    return true;
  });
  VELOX_CHECK_NOT_NULL(rpcPool, "RPC operator memory pool was not found");
  return rpcPool->stats();
}

} // namespace

class ShortTimeoutBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  std::chrono::milliseconds requestTimeout() const override {
    return std::chrono::milliseconds(10);
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    (void)std::move(DemoBatchRPCFunction::flushBatch(maxRows)).get();
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RPCResponse>>();
    heldResponse_ = std::move(promise);
    return std::move(future);
  }

 private:
  std::optional<folly::Promise<std::vector<RPCResponse>>> heldResponse_;
};

class FanOutBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  static std::vector<int32_t>& flushSizes() {
    static std::vector<int32_t> sizes;
    return sizes;
  }

  int32_t admissionUnitsForBatch(int32_t numRows) const override {
    return numRows;
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    flushSizes().push_back(maxRows);
    return DemoBatchRPCFunction::flushBatch(maxRows);
  }
};

class RecoveringFanOutBatchRPCFunction : public FanOutBatchRPCFunction {
 public:
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    auto& limiter = RPCRateLimiterRegistry::global().get("");
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
    return DemoBatchRPCFunction::evaluateCongestion(responses);
  }
};

// Applies the same overloads, but reports no congestion verdict for the batch.
class RecoveringNoVerdictBatchRPCFunction
    : public RecoveringFanOutBatchRPCFunction {
 public:
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    RecoveringFanOutBatchRPCFunction::evaluateCongestion(responses);
    return CongestionSignal::kNone;
  }
};

class MixedEpochRPCFunction : public DemoAsyncRPCFunction {
 public:
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      const auto prompt = prompts->valueAt(row).str();
      if (prompt == "slow") {
        auto [promise, future] = folly::makePromiseContract<RPCResponse>();
        slowResponse_ = std::move(promise);
        futures.emplace_back(row, std::move(future));
      } else if (prompt == "overload") {
        futures.emplace_back(
            row,
            folly::makeSemiFuture(
                RPCResponse::failed(
                    0,
                    velox::rpc::RPCErrorKind::kRateLimited,
                    "simulated overload")));
      } else {
        VELOX_CHECK(slowResponse_.has_value());
        RPCResponse slowResponse;
        slowResponse.setPayload(makeTextPayload("demo: slow"));
        slowResponse_->setValue(std::move(slowResponse));
        slowResponse_.reset();

        RPCResponse response;
        response.setPayload(makeTextPayload("demo: " + prompt));
        futures.emplace_back(row, folly::makeSemiFuture(std::move(response)));
      }
    });
    return futures;
  }

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    for (const auto& response : responses) {
      if (response.hasError() &&
          velox::rpc::errorCategory(response.errorKind()) ==
              velox::rpc::RPCErrorCategory::kOverload) {
        return CongestionSignal::kOverloaded;
      }
    }
    return CongestionSignal::kSuccess;
  }

 private:
  std::optional<folly::Promise<RPCResponse>> slowResponse_;
};

// Answers every row with a success whose transport reports
// kNumOverloadedAttempts overloaded attempts before it.
class OverloadedAttemptsRPCFunction : public DemoAsyncRPCFunction {
 public:
  static constexpr int32_t kNumOverloadedAttempts{2};

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      auto response = RPCResponse::ok(
          0, makeTextPayload("demo: " + prompts->valueAt(row).str()));
      response.numOverloadedAttempts = kNumOverloadedAttempts;
      futures.emplace_back(row, folly::makeSemiFuture(std::move(response)));
    });
    return futures;
  }
};

// Holds every response until releaseResponses(), then answers at once.
class HeldPacedRPCFunction : public DemoAsyncRPCFunction {
 public:
  static constexpr std::string_view kAdmissionKey = "held_paced_rpc";

  explicit HeldPacedRPCFunction(std::string key = std::string{kAdmissionKey})
      : key_(std::move(key)) {}

  std::string name() const override {
    return key_;
  }

  std::string admissionKey() const override {
    return key_;
  }

  bool mayReturnCapacityRetries() const override {
    return true;
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      auto response = RPCResponse::ok(
          0, makeTextPayload("demo: " + prompts->valueAt(row).str()));
      auto [promise, future] = folly::makePromiseContract<RPCResponse>();
      futures.emplace_back(row, std::move(future));
      std::unique_lock<std::mutex> lock(mutex_);
      ++numDispatches_;
      if (!released_) {
        held_.emplace_back(std::move(promise), std::move(response));
        lock.unlock();
        dispatchChanged_.notify_all();
        return;
      }
      lock.unlock();
      promise.setValue(std::move(response));
      dispatchChanged_.notify_all();
    });
    return futures;
  }

  bool waitForDispatches(int64_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return dispatchChanged_.wait_for(lock, std::chrono::seconds{5}, [&]() {
      return numDispatches_ >= count;
    });
  }

  void releaseResponses() {
    std::vector<std::pair<folly::Promise<RPCResponse>, RPCResponse>> responses;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
      responses.swap(held_);
    }
    for (auto& [promise, response] : responses) {
      promise.setValue(std::move(response));
    }
  }

 private:
  const std::string key_;
  std::mutex mutex_;
  std::condition_variable dispatchChanged_;
  int64_t numDispatches_{0};
  bool released_{false};
  std::vector<std::pair<folly::Promise<RPCResponse>, RPCResponse>> held_;
};

// Rejects the first attempt of the query with a retry hint naming 'cause',
// after 'numInPlaceRetries' retries in place, and answers every other attempt.
class FirstCapacityRetryRPCFunction : public DemoAsyncRPCFunction {
 public:
  FirstCapacityRetryRPCFunction(std::string key, int32_t numInPlaceRetries)
      : key_(std::move(key)), numInPlaceRetries_(numInPlaceRetries) {}

  std::string name() const override {
    return key_;
  }

  std::string admissionKey() const override {
    return key_;
  }

  bool mayReturnCapacityRetries() const override {
    return true;
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      auto response = RPCResponse::ok(
          0, makeTextPayload("demo: " + prompts->valueAt(row).str()));
      if (numAttempts.fetch_add(1) == 0) {
        response = RPCResponse::failed(
            0, velox::rpc::RPCErrorKind::kRateLimited, "first capacity retry");
        response.setCapacityRetry(
            {.cause = cause,
             .retryAfter = std::chrono::milliseconds{1},
             .maxRetries = 1,
             .numInPlaceRetries = numInPlaceRetries_});
      }
      futures.emplace_back(row, folly::makeSemiFuture(std::move(response)));
    });
    return futures;
  }

  std::atomic<int64_t> numAttempts{0};
  // Set before the query runs.
  velox::rpc::CapacityRetryCause cause{
      velox::rpc::CapacityRetryCause::kCapacity};

 private:
  const std::string key_;
  const int32_t numInPlaceRetries_;
};

// Rejects the first attempt of each of "a", "b" and "c" with a named capacity
// hint. "b" and "c" are both admitted before either is rejected: "b" is
// rejected while "c" is dispatched, and "c" only when "b" is retried. So "c"
// comes back after "b" advanced the overload epoch, stale and alone in its
// retry feedback.
class StaleCapacityRejectionRPCFunction : public DemoAsyncRPCFunction {
 public:
  explicit StaleCapacityRejectionRPCFunction(std::string key)
      : key_(std::move(key)) {}

  std::string name() const override {
    return key_;
  }

  std::string admissionKey() const override {
    return key_;
  }

  bool mayReturnCapacityRetries() const override {
    return true;
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      const auto prompt = prompts->valueAt(row).str();
      const bool firstAttempt = ++attempts_[prompt] == 1;
      if (firstAttempt && (prompt == "b" || prompt == "c")) {
        auto [promise, future] = folly::makePromiseContract<RPCResponse>();
        futures.emplace_back(row, std::move(future));
        held_.emplace(prompt, std::move(promise));
        if (prompt == "c") {
          release("b");
        }
        return;
      }
      if (firstAttempt) {
        futures.emplace_back(row, folly::makeSemiFuture(rejection()));
        return;
      }
      if (prompt == "b") {
        release("c");
      }
      futures.emplace_back(
          row,
          folly::makeSemiFuture(
              RPCResponse::ok(0, makeTextPayload("demo: " + prompt))));
    });
    return futures;
  }

 private:
  static RPCResponse rejection() {
    auto response = RPCResponse::failed(
        0, velox::rpc::RPCErrorKind::kRateLimited, "capacity rejection");
    response.setCapacityRetry(
        {.cause = velox::rpc::CapacityRetryCause::kCapacity,
         .retryAfter = std::chrono::milliseconds{1},
         .maxRetries = 1});
    return response;
  }

  void release(const std::string& prompt) {
    auto held = held_.find(prompt);
    VELOX_CHECK(held != held_.end(), "No held response for {}", prompt);
    auto promise = std::move(held->second);
    held_.erase(held);
    promise.setValue(rejection());
  }

  const std::string key_;
  // Touched only from the single driver's dispatch calls.
  folly::F14FastMap<std::string, int32_t> attempts_;
  folly::F14FastMap<std::string, folly::Promise<RPCResponse>> held_;
};

// Answers every row, each after one overloaded attempt the transport retried
// in place.
class OverloadRetriedRPCFunction : public DemoAsyncRPCFunction {
 public:
  explicit OverloadRetriedRPCFunction(std::string key) : key_(std::move(key)) {}

  std::string name() const override {
    return key_;
  }

  std::string admissionKey() const override {
    return key_;
  }

  bool mayReturnCapacityRetries() const override {
    return true;
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      auto response = RPCResponse::ok(
          0, makeTextPayload("demo: " + prompts->valueAt(row).str()));
      response.numOverloadedAttempts = 1;
      futures.emplace_back(row, folly::makeSemiFuture(std::move(response)));
    });
    return futures;
  }

 private:
  const std::string key_;
};

class ConservativeRetryRPCFunction : public HeldPacedRPCFunction {
 public:
  std::string name() const override {
    return "conservative_retry_rpc";
  }
};

class InvalidAdmissionBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  int32_t admissionUnitsForBatch(int32_t numRows) const override {
    return numRows + 1;
  }
};

enum class BatchContractViolation {
  kDuplicateRowId,
  kOutOfRangeRowId,
  kUnsetResponse,
};

class MalformedBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  explicit MalformedBatchRPCFunction(
      BatchContractViolation violation,
      bool failOnError = false)
      : DemoBatchRPCFunction(
            ResponseOrder::kInOrder,
            {},
            /*failWholeBatch=*/false,
            failOnError),
        violation_(violation) {}

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    return std::move(DemoBatchRPCFunction::flushBatch(maxRows))
        .deferValue([violation =
                         violation_](std::vector<RPCResponse> responses) {
          VELOX_CHECK_GE(responses.size(), 2);
          switch (violation) {
            case BatchContractViolation::kDuplicateRowId:
              responses.at(1).rowId = responses.at(0).rowId;
              break;
            case BatchContractViolation::kOutOfRangeRowId:
              responses.at(0).rowId = static_cast<int64_t>(responses.size());
              break;
            case BatchContractViolation::kUnsetResponse: {
              const auto rowId = responses.at(0).rowId;
              responses.at(0) = RPCResponse{};
              responses.at(0).rowId = rowId;
              break;
            }
          }
          return responses;
        });
  }

 private:
  const BatchContractViolation violation_;
};

class UnsetPerRowRPCFunction : public DemoAsyncRPCFunction {
 public:
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& /*args*/) override {
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        futures;
    rows.applyToSelected([&](vector_size_t row) {
      futures.emplace_back(row, folly::makeSemiFuture(RPCResponse{}));
    });
    return futures;
  }
};

class PreAdmissionKeyRPCFunction : public DemoAsyncRPCFunction {
 public:
  static void reset() {
    prepared_ = false;
    dispatchedAfterPrepare_ = false;
  }

  static bool dispatchedAfterPrepare() {
    return dispatchedAfterPrepare_;
  }

  AdmissionPreparationResult prepareInputForAdmissionAndDispatch(
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/) override {
    admissionKey_ = "resolved-tier";
    prepared_ = true;
    return AdmissionPreparationResult::kRequiresAdmission;
  }

  std::string admissionKey() const override {
    return admissionKey_;
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    dispatchedAfterPrepare_ = prepared_;
    return DemoAsyncRPCFunction::dispatchPerRow(rows, args);
  }

 private:
  std::string admissionKey_{"provisional-tier"};
  static inline bool prepared_{false};
  static inline bool dispatchedAfterPrepare_{false};
};

class CappedRPCFunction : public DemoAsyncRPCFunction {
 public:
  int64_t configuredCeiling() const override {
    return 64;
  }

  double configuredAdaptiveFloor() const override {
    return 0.5;
  }

  std::string admissionKey() const override {
    return "capped-rpc";
  }
};

class BackendErrorRPCFunction : public DemoAsyncRPCFunction {
 public:
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& /*args*/) override {
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        results;
    rows.applyToSelected([&](vector_size_t row) {
      results.emplace_back(
          row,
          folly::makeSemiFuture(
              RPCResponse::failed(
                  0,
                  velox::rpc::RPCErrorKind::kBackendError,
                  "backend_error")));
    });
    return results;
  }

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    return responses.empty() ? CongestionSignal::kNone
                             : CongestionSignal::kNonOverloadError;
  }

  std::string admissionKey() const override {
    return "backend-error-rpc";
  }
};

class UndeclaredBoundsRPCFunction : public DemoAsyncRPCFunction {
 public:
  double configuredAdaptiveFloor() const override {
    return 0.0;
  }

  std::string admissionKey() const override {
    return "undeclared-bounds-rpc";
  }
};

class DifferentFloorSameKeyRPCFunction : public CappedRPCFunction {
 public:
  double configuredAdaptiveFloor() const override {
    return 1.0;
  }
};

class ConstantSetupRPCFunction : public DemoAsyncRPCFunction {
 public:
  static void reset() {
    numPreparations_ = 0;
  }

  static int32_t numPreparations() {
    return numPreparations_;
  }

  bool requiresRowInspectionBeforeAdmission() const override {
    return false;
  }

  AdmissionPreparationResult prepareInputForAdmissionAndDispatch(
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/) override {
    ++numPreparations_;
    return AdmissionPreparationResult::kRequiresAdmission;
  }

 private:
  static inline int32_t numPreparations_{0};
};

class DeferredAdmissionRPCFunction : public DemoAsyncRPCFunction {
 public:
  static void reset() {
    numCongestionEvaluations_ = 0;
  }

  static int32_t numCongestionEvaluations() {
    return numCongestionEvaluations_;
  }

  AdmissionPreparationResult prepareInputForAdmissionAndDispatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    bool hasDispatchableRow = false;
    rows.applyToSelected([&](vector_size_t row) {
      hasDispatchableRow = hasDispatchableRow ||
          (!prompts->isNullAt(row) &&
           prompts->valueAt(row) != StringView("local error"));
    });
    if (!hasDispatchableRow) {
      return AdmissionPreparationResult::kLocalOnly;
    }
    admissionKey_ = "resolved-after-live-row";
    return AdmissionPreparationResult::kRequiresAdmission;
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    SelectivityVector backendRows(rows.size(), false);
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        results;
    rows.applyToSelected([&](vector_size_t row) {
      if (prompts->isNullAt(row)) {
        results.emplace_back(
            row,
            folly::makeSemiFuture(
                RPCResponse::failed(
                    0, velox::rpc::RPCErrorKind::kNullInput, "null_input")));
      } else if (prompts->valueAt(row) == StringView("local error")) {
        results.emplace_back(
            row,
            folly::makeSemiFuture(
                RPCResponse::failed(
                    0,
                    velox::rpc::RPCErrorKind::kInvalidRequest,
                    "local_error")));
      } else {
        backendRows.setValid(row, true);
      }
    });
    backendRows.updateBounds();
    auto backendResults =
        DemoAsyncRPCFunction::dispatchPerRow(backendRows, args);
    results.insert(
        results.end(),
        std::make_move_iterator(backendResults.begin()),
        std::make_move_iterator(backendResults.end()));
    return results;
  }

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    ++numCongestionEvaluations_;
    for (const auto& response : responses) {
      if (response.hasError()) {
        return CongestionSignal::kNonOverloadError;
      }
    }
    return DemoAsyncRPCFunction::evaluateCongestion(responses);
  }

  std::string admissionKey() const override {
    return admissionKey_;
  }

 private:
  std::string admissionKey_{"local-only-provisional-tier"};
  static inline int32_t numCongestionEvaluations_{0};
};

class DeferredAdmissionBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  static void reset() {
    numCongestionEvaluations_ = 0;
  }

  static int32_t numCongestionEvaluations() {
    return numCongestionEvaluations_;
  }

  AdmissionPreparationResult prepareInputForAdmissionAndDispatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    const auto* prompts = args.at(0)->asChecked<SimpleVector<StringView>>();
    bool hasDispatchableRow = false;
    rows.applyToSelected([&](vector_size_t row) {
      hasDispatchableRow = hasDispatchableRow || !prompts->isNullAt(row);
    });
    if (!hasDispatchableRow) {
      return AdmissionPreparationResult::kLocalOnly;
    }
    admissionKey_ = "resolved-batch-after-live-row";
    return AdmissionPreparationResult::kRequiresAdmission;
  }

  std::string admissionKey() const override {
    return admissionKey_;
  }

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    ++numCongestionEvaluations_;
    return DemoBatchRPCFunction::evaluateCongestion(responses);
  }

 private:
  std::string admissionKey_{"batch-local-only-provisional-tier"};
  static inline int32_t numCongestionEvaluations_{0};
};

class NoLatencySuccessBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& /*responses*/) const override {
    return CongestionSignal::kSuccessNoLatency;
  }
};

// Fails the only row of every batch with a backend error.
class BackendErrorBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  BackendErrorBatchRPCFunction()
      : DemoBatchRPCFunction(ResponseOrder::kInOrder, {0}) {}

  std::string admissionKey() const override {
    return "backend-error-batch-rpc";
  }
};

// Replays historical outcomes alongside a fresh backend error.
class MixedHistoricalBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  std::string admissionKey() const override {
    return "mixed-historical-batch-rpc";
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    return DemoBatchRPCFunction::flushBatch(maxRows).deferValue(
        [](std::vector<RPCResponse> responses) {
          VELOX_CHECK_EQ(responses.size(), 3);
          responses[0].setCompletionOutcome(
              velox::rpc::RPCErrorKind::kNone, false);
          responses[1].setError(
              velox::rpc::RPCErrorKind::kRateLimited, "historical overload");
          responses[1].setCompletionOutcome(
              velox::rpc::RPCErrorKind::kRateLimited, false);
          responses[2].setError(
              velox::rpc::RPCErrorKind::kBackendError, "current failure");
          return responses;
        });
  }

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>&) const override {
    // Supplies the production policy's verdict for this response combination.
    return CongestionSignal::kNonOverloadError;
  }
};

// Rate-limits every row of every flush, and paces down to half a unit.
class RateLimitedBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  double configuredAdaptiveFloor() const override {
    return 0.5;
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    return DemoBatchRPCFunction::flushBatch(maxRows).deferValue(
        [](std::vector<RPCResponse> responses) {
          for (auto& response : responses) {
            response.setError(
                velox::rpc::RPCErrorKind::kRateLimited, "simulated overload");
          }
          return responses;
        });
  }
};

// Rate-limits the first row of every flush and serves the rest.
class PartlyRateLimitedBatchRPCFunction : public RateLimitedBatchRPCFunction {
 public:
  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    return DemoBatchRPCFunction::flushBatch(maxRows).deferValue(
        [](std::vector<RPCResponse> responses) {
          if (!responses.empty()) {
            responses.front().setError(
                velox::rpc::RPCErrorKind::kRateLimited, "simulated overload");
          }
          return responses;
        });
  }
};

// Declares a per-request byte budget and reports a fixed number of rows that
// fit it, so the operator must chunk a large backlog to keep each request
// under the cap. Records every flush's row count so the test can assert no
// flush exceeds the budget-derived limit and no rows are lost.
class ByteBudgetedBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  using DemoBatchRPCFunction::DemoBatchRPCFunction;

  static constexpr int32_t kRowsPerBudget = 3;

  // Flush chunk sizes observed across a query. Cleared at the start of the
  // test.
  static std::vector<int32_t>& flushChunks() {
    static std::vector<int32_t> chunks;
    return chunks;
  }

  // Report a fixed cap (>= 1), independent of any byte math — this test
  // exercises the operator's clamp, not a function's estimator.
  int32_t maxRowsPerFlush(int32_t desired) const override {
    return std::min(desired, kRowsPerBudget);
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    flushChunks().push_back(maxRows);
    return DemoBatchRPCFunction::flushBatch(maxRows);
  }
};

class InvalidMaxRowsBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  explicit InvalidMaxRowsBatchRPCFunction(bool returnZero)
      : returnZero_(returnZero) {}

  int32_t maxRowsPerFlush(int32_t desired) const override {
    return returnZero_ ? 0 : desired + 1;
  }

 private:
  const bool returnZero_;
};

class RPCOperatorTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    AsyncRPCFunctionRegistry::testingClear();
    OperatorTestBase::SetUpTestCase();
    registerRPCPlanNodeTranslator();
    registerFunction<DemoAsyncRPCFunction>("demo_rpc");
    registerFunction<ShortTimeoutBatchRPCFunction>("short_timeout_batch_rpc");
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_held",
        []() {
          auto fn = std::make_shared<DemoBatchRPCFunction>();
          fn->testingHoldFlushes();
          return fn;
        },
        DemoBatchRPCFunction::signatures());
    registerFunction<FanOutBatchRPCFunction>("fan_out_batch_rpc");
    registerFunction<RecoveringFanOutBatchRPCFunction>(
        "recovering_fan_out_batch_rpc");
    registerFunction<RecoveringNoVerdictBatchRPCFunction>(
        "recovering_no_verdict_batch_rpc");
    registerFunction<MixedEpochRPCFunction>("mixed_epoch_rpc");
    registerFunction<OverloadedAttemptsRPCFunction>("overloaded_attempts_rpc");
    registerFunction<InvalidAdmissionBatchRPCFunction>(
        "invalid_admission_batch_rpc");
    registerFunction<NoLatencySuccessBatchRPCFunction>(
        "no_latency_success_batch_rpc");
    registerFunction<BackendErrorBatchRPCFunction>("backend_error_batch_rpc");
    registerFunction<MixedHistoricalBatchRPCFunction>(
        "mixed_historical_batch_rpc");
    registerFunction<RateLimitedBatchRPCFunction>("rate_limited_batch_rpc");
    AsyncRPCFunctionRegistry::registerFunction(
        "rate_limited_batch_rpc_held",
        []() {
          auto function = std::make_shared<RateLimitedBatchRPCFunction>();
          function->testingHoldFlushes();
          return function;
        },
        DemoBatchRPCFunction::signatures());
    registerFunction<PartlyRateLimitedBatchRPCFunction>(
        "partly_rate_limited_batch_rpc");
    registerFunction<DemoBatchRPCFunction>("demo_batch_rpc");
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_reversed",
        DemoBatchRPCFunction::ResponseOrder::kReversed);
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_partial_fail",
        DemoBatchRPCFunction::ResponseOrder::kInOrder,
        std::unordered_set<int32_t>{1, 3});
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_whole_fail",
        DemoBatchRPCFunction::ResponseOrder::kInOrder,
        std::unordered_set<int32_t>{},
        /*failWholeBatch=*/true);
    // Whole-batch failure AND a fail-on-error policy (mimics
    // meta_ai_on_error='fail'): the query must still hard-fail, not degrade.
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_whole_fail_strict",
        DemoBatchRPCFunction::ResponseOrder::kInOrder,
        std::unordered_set<int32_t>{},
        /*failWholeBatch=*/true,
        /*failOnError=*/true);
    // Returns fewer responses than rows. The operator must fail the query
    // rather than guess response placement.
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_wrong_count",
        DemoBatchRPCFunction::ResponseOrder::kInOrder,
        std::unordered_set<int32_t>{},
        /*failWholeBatch=*/false,
        /*failOnError=*/false,
        /*dropOneResponse=*/true);
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_wrong_count_strict",
        DemoBatchRPCFunction::ResponseOrder::kInOrder,
        std::unordered_set<int32_t>{},
        /*failWholeBatch=*/false,
        /*failOnError=*/true,
        /*dropOneResponse=*/true);
    registerFunction<MalformedBatchRPCFunction>(
        "demo_batch_rpc_duplicate_row_id",
        BatchContractViolation::kDuplicateRowId);
    registerFunction<MalformedBatchRPCFunction>(
        "demo_batch_rpc_duplicate_row_id_strict",
        BatchContractViolation::kDuplicateRowId,
        /*failOnError=*/true);
    registerFunction<MalformedBatchRPCFunction>(
        "demo_batch_rpc_out_of_range_row_id",
        BatchContractViolation::kOutOfRangeRowId);
    registerFunction<MalformedBatchRPCFunction>(
        "demo_batch_rpc_out_of_range_row_id_strict",
        BatchContractViolation::kOutOfRangeRowId,
        /*failOnError=*/true);
    registerFunction<MalformedBatchRPCFunction>(
        "demo_batch_rpc_unset_response",
        BatchContractViolation::kUnsetResponse);
    registerFunction<MalformedBatchRPCFunction>(
        "demo_batch_rpc_unset_response_strict",
        BatchContractViolation::kUnsetResponse,
        /*failOnError=*/true);
    registerFunction<UnsetPerRowRPCFunction>("demo_rpc_unset_response");
    registerFunction<PreAdmissionKeyRPCFunction>("pre_admission_tier_rpc");
    registerFunction<CappedRPCFunction>("capped_rpc");
    registerFunction<BackendErrorRPCFunction>("backend_error_rpc");
    registerFunction<UndeclaredBoundsRPCFunction>("undeclared_bounds_rpc");
    registerFunction<DifferentFloorSameKeyRPCFunction>(
        "different_floor_same_key_rpc");
    registerFunction<ConstantSetupRPCFunction>("constant_setup_rpc");
    registerFunction<DeferredAdmissionRPCFunction>("deferred_admission_rpc");
    registerFunction<DeferredAdmissionBatchRPCFunction>(
        "deferred_admission_batch_rpc");
    // Fails asynchronous batch work with an internal error.
    registerFunction<DemoBatchRPCFunction>(
        "demo_batch_rpc_whole_fail_fatal",
        DemoBatchRPCFunction::ResponseOrder::kInOrder,
        std::unordered_set<int32_t>{},
        /*failWholeBatch=*/false,
        /*failOnError=*/false,
        /*dropOneResponse=*/false,
        /*failWholeBatchFatal=*/true);
    // Same on the per-row path, which completes through RPCState.
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_rpc_fatal",
        []() {
          auto function = std::make_shared<DemoAsyncRPCFunction>();
          function->testingFailFatally();
          return function;
        },
        DemoAsyncRPCFunction::signatures());
    // Declares maxBatchBytes so the operator clamps each flush by the byte
    // budget (see batchClampedByByteBudget).
    registerFunction<ByteBudgetedBatchRPCFunction>(
        "demo_batch_rpc_byte_budgeted");
    registerFunction<InvalidMaxRowsBatchRPCFunction>(
        "demo_batch_rpc_zero_flush_limit", true);
    registerFunction<InvalidMaxRowsBatchRPCFunction>(
        "demo_batch_rpc_excessive_flush_limit", false);
  }

  static void TearDownTestCase() {
    AsyncRPCFunctionRegistry::testingClear();
    OperatorTestBase::TearDownTestCase();
    // Reset MemoryManager to shut down SharedArbitrator executor threads.
    // Without this, TSAN reports a non-zero exit because the executor
    // threads are still running at process exit.
    memory::MemoryManager::testingSetInstance({});
  }

  void TearDown() override {
    RPCRateLimiterRegistry::global().testingReset();
    OperatorTestBase::TearDown();
  }

  // A rate-limiter policy a test seals on the default backend before its
  // query runs. A query shares that limiter only when it resolves to the same
  // policy, so the test also runs the query under sessionFor(policy).
  struct DefaultBackendPolicy {
    bool adaptive{false};
    int64_t ceiling{0};
    int64_t floor{1};
  };

  static RPCRateLimiter& sealDefaultBackend(
      const DefaultBackendPolicy& policy) {
    auto& limiter = RPCRateLimiterRegistry::global().get("");
    RPCRateLimiter::Config config;
    config.adaptive = policy.adaptive;
    config.ceiling = policy.ceiling;
    config.floor = static_cast<double>(policy.floor);
    EXPECT_TRUE(limiter.initializeOrMatch(config));
    return limiter;
  }

  // Session properties under which a query resolves to 'policy'.
  static std::unordered_map<std::string, std::string> sessionFor(
      const DefaultBackendPolicy& policy) {
    return {
        {core::QueryConfig::kRpcRateLimiterAdaptiveEnabled,
         policy.adaptive ? "true" : "false"},
        {core::QueryConfig::kRpcRateLimiterMaxLimit,
         std::to_string(policy.ceiling)},
        {core::QueryConfig::kRpcRateLimiterMinLimit,
         std::to_string(policy.floor)},
    };
  }

  // Registers 'name' as a function that creates a new T(args...) for each
  // operator, with the signature of demo_rpc.
  template <typename T, typename... Args>
  static void registerFunction(const std::string& name, Args... args) {
    ASSERT_TRUE(
        AsyncRPCFunctionRegistry::registerFunction(
            name,
            [args...] { return std::make_shared<T>(args...); },
            DemoAsyncRPCFunction::signatures()));
  }

  // Registers 'function' under its name with the signature of demo_rpc. Every
  // query of the test runs this instance.
  static void registerInstance(
      const std::shared_ptr<AsyncRPCFunction>& function) {
    ASSERT_TRUE(
        AsyncRPCFunctionRegistry::registerFunction(
            function->name(),
            [function] { return function; },
            DemoAsyncRPCFunction::signatures()));
  }

  void assertCapacityRetryBackoff(
      const std::string& key,
      const std::unordered_map<std::string, std::string>& session,
      bool sharedAdaptive,
      RPCFlowControlPolicy flowControlPolicy,
      velox::rpc::CapacityRetryCause cause,
      std::optional<int64_t> minimumSharedCapacity) {
    SCOPED_TRACE(key);
    auto function = std::make_shared<FirstCapacityRetryRPCFunction>(
        key, /*numInPlaceRetries=*/0);
    function->cause = cause;
    registerInstance(function);
    auto input = makeNamedRowVector(
        "prompt", makeFlatVector<StringView>({"first", "second", "third"}));
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(fmt::format("{}(prompt)", function->name()))
                    .planNode();
    const auto task =
        AssertQueryBuilder(plan).configs(session).assertResults(makeRowVector({
            makeFlatVector<std::string>({"first", "second", "third"}),
            makeFlatVector<std::string>(
                {"demo: first", "demo: second", "demo: third"}),
        }));
    EXPECT_EQ(function->numAttempts.load(), 4);
    const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
    EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcRequestsDispatched), 3);
    EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcResponsesReceived), 3);
    EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcErrorCount), 0);
    EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcCapacityRetries), 1);
    const auto rejections = [&](const std::string& name) {
      return stats.hasCustomStat(name) ? stats.customSum(name) : 0;
    };
    EXPECT_EQ(
        rejections(RPCOperatorStatNames::kRpcQuotaRejections),
        cause == velox::rpc::CapacityRetryCause::kQuota ? 1 : 0);
    EXPECT_EQ(
        rejections(RPCOperatorStatNames::kRpcCapacityRejections),
        cause == velox::rpc::CapacityRetryCause::kCapacity ? 1 : 0);
    // Only CAPACITY_PROBING leaves the backoff of a named capacity rejection to
    // an adaptive shared limiter. WINDOW_AIMD, and CAPACITY_PROBING for any
    // other rejection, halve the driver window as well.
    const bool capacityProbing =
        flowControlPolicy == RPCFlowControlPolicy::kCapacityProbing;
    const bool namedCapacity =
        cause == velox::rpc::CapacityRetryCause::kCapacity;
    EXPECT_EQ(
        stats.hasCustomStat(
            RPCOperatorStatNames::kRpcCongestionOverloadShrinks),
        !(sharedAdaptive && capacityProbing && namedCapacity));
    EXPECT_EQ(
        stats.customStat(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy)
            .max,
        static_cast<int64_t>(flowControlPolicy));
    if (minimumSharedCapacity.has_value()) {
      EXPECT_EQ(
          stats.customStat(RPCOperatorStatNames::kRpcRateLimiterMinCap).min,
          minimumSharedCapacity.value());
    } else if (!sharedAdaptive) {
      EXPECT_EQ(
          stats.customStat(RPCOperatorStatNames::kRpcRateLimiterMinCap).min,
          64);
    } else if (!capacityProbing) {
      EXPECT_LT(
          stats.customStat(RPCOperatorStatNames::kRpcRateLimiterMinCap).min,
          64);
    } else {
      EXPECT_LT(
          stats.customStat(RPCOperatorStatNames::kRpcRateLimiterMinLimitMilli)
              .min,
          stats.customStat(RPCOperatorStatNames::kRpcRateLimiterHardLimit).max *
              1'000);
    }
  }

  // Returns whether the query's RPC operator, the root of its plan, ran under a
  // limiter scoped to its own policy rather than the backend's shared limiter.
  static bool ranOnPolicyScopedLimiter(
      const std::shared_ptr<exec::Task>& task) {
    return PlanNodeStats::extract(
               task->taskStats(), task->planFragment().planNode->id())
        .hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterPolicyScoped);
  }

  static void expectRanOnSharedLimiter(
      const std::shared_ptr<exec::Task>& task) {
    EXPECT_FALSE(ranOnPolicyScopedLimiter(task))
        << "the query did not run on the limiter the test sealed";
  }

  // Drives a query whose backend is fully held by the test body until a
  // background thread releases it. Defined below the fixture.
  void runContendedDrain(bool batchMode);

  std::array<std::string, 3> registerReceiptFunctions(
      const std::string& prefix,
      const std::shared_ptr<ReceiptOperatorState>& state) {
    std::array<std::string, 3> names{
        prefix + "_submit", prefix + "_poll", prefix + "_collect"};
    const std::array<DeferredRpcPhase, 3> phases{
        DeferredRpcPhase::kSubmit,
        DeferredRpcPhase::kPoll,
        DeferredRpcPhase::kCollect};
    for (size_t index = 0; index < names.size(); ++index) {
      const auto name = names[index];
      const auto phase = phases[index];
      EXPECT_TRUE(
          AsyncRPCFunctionRegistry::registerFunction(
              name,
              [state, phase, name] {
                return std::make_shared<DeferredRpcPhaseFunction>(
                    name, phase, [state](DeferredRpcPhase bindingPhase) {
                      return std::make_shared<ReceiptOperatorBackend>(
                          bindingPhase, state);
                    });
              },
              {FunctionSignatureBuilder()
                   .argumentType(
                       phase == DeferredRpcPhase::kSubmit
                           ? "varchar"
                           : OperatorReceiptCodec{}.signature())
                   .returnType(
                       phase == DeferredRpcPhase::kCollect
                           ? "varchar"
                           : OperatorReceiptCodec{}.signature())
                   .build()}));
    }
    return names;
  }
};

TEST_F(RPCOperatorTest, capacityRetryBackoffAtDefaultWindowFloor) {
  const std::string key{"capacity_retry_default_window_floor"};
  RPCRateLimiter::Config policy{
      .adaptive = true,
      .ceiling = 200,
      .floor = 50,
  };
  auto* limiter =
      RPCRateLimiterRegistry::global().getForPolicy(key, policy).limiter;
  for (int32_t cut = 0; cut < 2; ++cut) {
    auto reserved = limiter->tryAcquireUpTo(1);
    ASSERT_EQ(reserved.size(), 1);
    auto lease = limiter->makeLease(std::move(reserved));
    const auto completion = lease->completeAndRelease(
        /*hardOverload=*/true, /*capacityRejection=*/false);
    EXPECT_TRUE(completion.sharedOverloadHandled);
  }
  ASSERT_EQ(limiter->stats().capacity, 50);

  auto session = sessionFor({.adaptive = true, .ceiling = 200, .floor = 50});
  session[core::QueryConfig::kRpcCongestionMaxWindow] = "200";
  // Shared floor and driver fallback are independent backoff observations.
  assertCapacityRetryBackoff(
      key,
      session,
      /*sharedAdaptive=*/false,
      RPCFlowControlPolicy::kWindowAimd,
      velox::rpc::CapacityRetryCause::kCapacity,
      /*minimumSharedCapacity=*/50);
  EXPECT_EQ(limiter->stats().pending, 0);
}

// A row the transport answered only after retrying an overload in place earns
// a CAPACITY_PROBING limiter no growth; WINDOW_AIMD still grows on it.
TEST_F(RPCOperatorTest, capacityProbingGivesNoGrowthForOverloadRetriedRows) {
  constexpr int32_t kRows = 300;
  std::vector<std::string> prompts;
  prompts.reserve(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    prompts.push_back(fmt::format("prompt{}", i));
  }
  for (const auto& [policyName, grows] :
       std::vector<std::pair<std::string, bool>>{
           {"WINDOW_AIMD", true}, {"CAPACITY_PROBING", false}}) {
    SCOPED_TRACE(policyName);
    auto function = std::make_shared<OverloadRetriedRPCFunction>(
        fmt::format("overload_retried_{}", grows));
    registerInstance(function);
    auto plan = PlanBuilder()
                    .values(makeNamedRowVector(
                        "prompt", makeFlatVector<std::string>(prompts)))
                    .rpc(fmt::format("{}(prompt)", function->name()))
                    .planNode();
    std::shared_ptr<Task> task;
    ASSERT_EQ(
        AssertQueryBuilder(plan)
            .configs(
                {{core::QueryConfig::kRpcFlowControlPolicy, policyName},
                 {core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, "true"},
                 {core::QueryConfig::kRpcRateLimiterMaxLimit, "1000"},
                 {core::QueryConfig::kRpcRateLimiterMinLimit, "1"}})
            .countResults(task),
        kRows);
    const auto limitMilli =
        PlanNodeStats::extract(task->taskStats(), plan->id())
            .customStat(RPCOperatorStatNames::kRpcRateLimiterLimitMilli)
            .max;
    if (grows) {
      EXPECT_GT(limitMilli, 200'000);
    } else {
      EXPECT_EQ(limitMilli, 200'000);
    }
  }
}

TEST_F(RPCOperatorTest, capacityRetryDriverBackoffFollowsPolicy) {
  struct Selection {
    std::optional<std::string> policyName;
    bool adaptive;
    int64_t floor{1};
  };
  const std::vector<Selection> selections{
      {std::nullopt, true},
      {"WINDOW_AIMD", true},
      {"CAPACITY_PROBING", true},
      {std::nullopt, false},
      {"WINDOW_AIMD", false},
      {"CAPACITY_PROBING", false},
      {std::nullopt, true, 64},
      {"WINDOW_AIMD", true, 64},
      {"CAPACITY_PROBING", true, 64},
      {std::nullopt, true, 128},
      {"WINDOW_AIMD", true, 128},
      {"CAPACITY_PROBING", true, 128},
  };
  const std::vector<velox::rpc::CapacityRetryCause> causes{
      velox::rpc::CapacityRetryCause::kCapacity,
      velox::rpc::CapacityRetryCause::kQuota,
      velox::rpc::CapacityRetryCause::kUnknown,
  };
  int32_t caseIndex = 0;
  for (const auto cause : causes) {
    for (const auto& [policyName, adaptive, floor] : selections) {
      SCOPED_TRACE(
          fmt::format(
              "cause={} policy={} adaptive={} floor={}",
              static_cast<int>(cause),
              policyName.value_or("default"),
              adaptive,
              floor));
      auto session =
          sessionFor({.adaptive = adaptive, .ceiling = 64, .floor = floor});
      session[core::QueryConfig::kRpcCongestionMaxWindow] = "64";
      if (policyName.has_value()) {
        session[core::QueryConfig::kRpcFlowControlPolicy] = policyName.value();
      }
      const bool window =
          !policyName.has_value() || policyName.value() == "WINDOW_AIMD";
      const auto flowControlPolicy = !adaptive || window
          ? RPCFlowControlPolicy::kWindowAimd
          : RPCFlowControlPolicy::kCapacityProbing;
      // A floor at or above the ceiling pins the window, except that
      // CAPACITY_PROBING follows a capacity rejection below the floor.
      const bool pinned = floor >= 64 &&
          !(flowControlPolicy == RPCFlowControlPolicy::kCapacityProbing &&
            cause == velox::rpc::CapacityRetryCause::kCapacity);
      const bool sharedAdaptive = adaptive && !pinned;
      assertCapacityRetryBackoff(
          fmt::format("capacity_retry_backoff_{}", caseIndex++),
          session,
          sharedAdaptive,
          flowControlPolicy,
          cause,
          pinned ? std::optional<int64_t>{64} : std::nullopt);
    }
  }
}

// WINDOW_AIMD halves the driver window on every overloaded retry, not only the
// first, including one that arrives after another rejection cut the limiter.
// The floor of 2 keeps a slot for "b"'s retry while "c" is held.
TEST_F(RPCOperatorTest, windowAimdCutsDriverWindowOnEveryOverloadedRetry) {
  const std::string key = "window_aimd_every_overloaded_retry";
  auto function = std::make_shared<StaleCapacityRejectionRPCFunction>(key);
  registerInstance(function);
  auto session = sessionFor({.adaptive = true, .ceiling = 4, .floor = 2});
  session[core::QueryConfig::kRpcCongestionMaxWindow] = "64";
  const auto first =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a"}));
  const auto second =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"b", "c"}));
  auto plan = PlanBuilder()
                  .values({first, second})
                  .rpc(fmt::format("{}(prompt)", function->name()))
                  .planNode();

  std::shared_ptr<Task> task;
  ASSERT_EQ(AssertQueryBuilder(plan).configs(session).countResults(task), 3);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(
      stats.customStat(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy)
          .max,
      0);
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcCapacityRetries), 3);
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcCongestionOverloadShrinks), 3);
}

// A function's aggregate congestion verdict carries no capacity hint, so it
// gets WINDOW_AIMD's cut without starting CAPACITY_PROBING's probing.
TEST_F(RPCOperatorTest, capacityProbingAnswersAggregateOverloadAsWindowAimd) {
  class OverloadVerdictFunction : public DemoAsyncRPCFunction {
   public:
    std::string name() const override {
      return "capacity_aggregate_overload_verdict";
    }
    std::string admissionKey() const override {
      return name();
    }
    bool mayReturnCapacityRetries() const override {
      return true;
    }
    CongestionSignal evaluateCongestion(
        const std::vector<RPCResponse>& /*responses*/) const override {
      return CongestionSignal::kOverloaded;
    }
  };
  auto function = std::make_shared<OverloadVerdictFunction>();
  registerInstance(function);
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"first"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(fmt::format("{}(prompt)", function->name()))
                  .planNode();
  auto session = sessionFor({.adaptive = true, .ceiling = 10, .floor = 9});
  session[core::QueryConfig::kRpcFlowControlPolicy] = "CAPACITY_PROBING";
  AssertQueryBuilder(plan).configs(session).countResults();
  // The verdict names no lack of capacity, so the cut stops at the floor of 9
  // as under WINDOW_AIMD; a capacity cut would keep 8 of the window of 10.
  EXPECT_EQ(
      RPCRateLimiterRegistry::global().get(function->name()).stats().capacity,
      9);
}

TEST_F(RPCOperatorTest, terminalCapacityRejectionStillCutsDriverWindow) {
  // The transport has spent the positive count budget before handing off.
  auto function = std::make_shared<FirstCapacityRetryRPCFunction>(
      "terminal_capacity_rejection", /*numInPlaceRetries=*/1);
  registerInstance(function);
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"first"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(fmt::format("{}(prompt)", function->name()))
                  .planNode();
  auto session = sessionFor({.adaptive = true, .ceiling = 64, .floor = 1});
  session[core::QueryConfig::kRpcCongestionMaxWindow] = "64";
  const auto task =
      AssertQueryBuilder(plan).configs(session).assertResults(makeRowVector(
          {input->childAt(0), makeNullConstant(TypeKind::VARCHAR, 1)}));
  EXPECT_EQ(function->numAttempts.load(), 1);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcRequestsDispatched), 1);
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcResponsesReceived), 1);
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcErrorCount), 1);
  EXPECT_FALSE(stats.hasCustomStat(RPCOperatorStatNames::kRpcCapacityRetries));
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcCongestionOverloadShrinks), 1);
  EXPECT_LT(
      stats.customStat(RPCOperatorStatNames::kRpcRateLimiterMinCap).min, 64);
}

/// Runs Values(3 rows) → RPCNode → verifies passthrough + RPC result.
TEST_F(RPCOperatorTest, basicPerRow) {
  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<std::string>({"hello world", "test prompt", "third row"}));

  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeFlatVector<std::string>(
          {"demo: hello world", "demo: test prompt", "demo: third row"}),
  }));
}

// kPerRow output is sized from QueryConfig::preferredOutputBatchRows: 50 rows
// with a cap of 10 emit at least 5 output vectors.
TEST_F(RPCOperatorTest, outputBatchSizeFromConfig) {
  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<std::string>(50, [](auto row) {
        return fmt::format("row {}", row);
      }));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();

  std::shared_ptr<exec::Task> task;
  EXPECT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kPreferredOutputBatchRows, 10)
          .countResults(task),
      50);

  // 50 rows capped at 10 per output vector => at least 5 vectors.
  EXPECT_GE(
      PlanNodeStats::extract(task->taskStats(), plan->id()).outputVectors, 5);
}

TEST_F(RPCOperatorTest, usesFunctionBatchTimeout) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"hello"}));
  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("short_timeout_batch_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();
  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan).copyResults(pool(), task);
  ASSERT_EQ(result->size(), 1);
  ASSERT_TRUE(result->childAt(1)->isNullAt(0));
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcErrorKindTimeout));
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcErrorKindTimeout), 1);
}

/// Null input rows should produce null in the RPC result column.
TEST_F(RPCOperatorTest, nullInput) {
  auto input = makeNamedRowVector(
      "prompt",
      makeNullableFlatVector<std::string>({"valid prompt", std::nullopt}));

  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeNullableFlatVector<std::string>({"demo: valid prompt", std::nullopt}),
  }));
}

/// Multiple source columns — verifies all passthrough columns are preserved.
TEST_F(RPCOperatorTest, multipleColumns) {
  auto input = makeRowVector(
      {"id", "prompt", "extra"},
      {makeFlatVector<int64_t>({100, 200}),
       makeFlatVector<std::string>({"question one", "question two"}),
       makeFlatVector<double>({1.5, 2.5})});

  // Only "prompt" is an RPC argument; "id" and "extra" are passthrough.
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      input->childAt(1),
      input->childAt(2),
      makeFlatVector<std::string>({"demo: question one", "demo: question two"}),
  }));
}

// ============================================================
// BATCH mode tests — exercise accumulateBatch/flushBatch path
// ============================================================

// Basic batch mode: responses in order, verify passthrough + result.
TEST_F(RPCOperatorTest, batchBasic) {
  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<std::string>({"hello", "world", "batch"}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("demo_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeFlatVector<std::string>(
          {"Batch response for: hello",
           "Batch response for: world",
           "Batch response for: batch"}),
  }));
}

// The first policy to reach a backend seals its shared limiter. A later query
// with another policy runs under a limiter of its own and leaves the shared
// policy as it was, so a pipeline that lowered the floor to fit a small
// entitlement keeps that floor.
TEST_F(
    RPCOperatorTest,
    laterQueryWithAnotherPolicyLeavesSharedPolicyUnchanged) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b"}));
  auto& limiter = RPCRateLimiterRegistry::global().get("");

  // First query: a deliberate policy, as a small-entitlement pipeline sets.
  ASSERT_TRUE(limiter.initializeOrMatch(
      {.adaptive = true, .floor = 4, .decreaseFactor = 0.25}));

  // A later query on the same backend, carrying the defaults.
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("demo_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(task), 2);
  EXPECT_TRUE(ranOnPolicyScopedLimiter(task));
  const core::QueryConfig defaults{
      std::unordered_map<std::string, std::string>{}};
  EXPECT_EQ(
      PlanNodeStats::extract(task->taskStats(), plan->id())
          .customStat(RPCOperatorStatNames::kRpcRateLimiterHardLimit)
          .max,
      defaults.rpcRateLimiterMaxLimit())
      << "the later query ran under the shared limiter's ceiling";

  const auto config = limiter.config();
  EXPECT_TRUE(config.adaptive);
  EXPECT_EQ(config.floor, 4) << "a later query reconfigured the backend";
  EXPECT_DOUBLE_EQ(config.decreaseFactor, 0.25);
}

// A query that sets its own rate-limiter limits runs under them, even after
// another query with different limits sealed the backend's shared limiter. The
// first query's limits still hold for later queries that ask for them.
TEST_F(RPCOperatorTest, explicitSessionLimitsAreHonoredOnSealedBackend) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b"}));
  struct Run {
    int64_t lowestCap;
    bool policyScoped;
  };
  auto runWithLimit = [&](const std::string& limit) {
    auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
    std::shared_ptr<exec::Task> task;
    AssertQueryBuilder(plan)
        .config(core::QueryConfig::kRpcRateLimiterMinLimit, limit)
        .config(core::QueryConfig::kRpcRateLimiterMaxLimit, limit)
        .countResults(task);
    return Run{
        .lowestCap = PlanNodeStats::extract(task->taskStats(), plan->id())
                         .customStat(RPCOperatorStatNames::kRpcRateLimiterCap)
                         .min,
        .policyScoped = ranOnPolicyScopedLimiter(task)};
  };

  const auto first = runWithLimit("25");
  EXPECT_EQ(first.lowestCap, 25);
  EXPECT_FALSE(first.policyScoped);

  const auto larger = runWithLimit("100");
  EXPECT_EQ(larger.lowestCap, 100);
  EXPECT_TRUE(larger.policyScoped);

  const auto sameAsFirst = runWithLimit("25");
  EXPECT_EQ(sameAsFirst.lowestCap, 25);
  EXPECT_FALSE(sameAsFirst.policyScoped);
}

// A query on the default settings runs the default policy, even after a query
// that set its own limits sealed the backend's shared limiter first.
TEST_F(RPCOperatorTest, defaultQueryDoesNotInheritAnotherQuerysPolicy) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  AssertQueryBuilder(plan)
      .config(core::QueryConfig::kRpcRateLimiterMinLimit, 25)
      .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 25)
      .countResults();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).countResults(task);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  const core::QueryConfig defaults{
      std::unordered_map<std::string, std::string>{}};
  EXPECT_EQ(
      stats.customStat(RPCOperatorStatNames::kRpcRateLimiterHardLimit).max,
      defaults.rpcRateLimiterMaxLimit());
  EXPECT_TRUE(ranOnPolicyScopedLimiter(task));
}

// A function that admits without inspecting rows resolves its limiter in
// initialize(), before any input arrives. Its query reports whether that
// limiter is the backend's shared one even when no row reaches the operator.
TEST_F(RPCOperatorTest, eagerAdmissionReportsPolicyScopedLimiter) {
  const DefaultBackendPolicy policy{.ceiling = 25};
  sealDefaultBackend(policy);
  auto noRows = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>(std::vector<StringView>{}));
  auto runtimeStats =
      [&](const std::string& functionName,
          const std::unordered_map<std::string, std::string>& session) {
        auto plan = PlanBuilder()
                        .values(noRows)
                        .rpc(fmt::format("{}(prompt)", functionName))
                        .planNode();
        std::shared_ptr<exec::Task> task;
        AssertQueryBuilder(plan).configs(session).countResults(task);
        return PlanNodeStats::extract(task->taskStats(), plan->id());
      };

  {
    // The default policy differs from the sealed one, so the query runs under
    // the default ceiling on a limiter of its own.
    const auto stats = runtimeStats("constant_setup_rpc", {});
    EXPECT_TRUE(
        stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterPolicyScoped));
    ASSERT_TRUE(
        stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterHardLimit));
    const core::QueryConfig defaults{
        std::unordered_map<std::string, std::string>{}};
    EXPECT_EQ(
        stats.customStat(RPCOperatorStatNames::kRpcRateLimiterHardLimit).max,
        defaults.rpcRateLimiterMaxLimit());
  }
  {
    const auto stats = runtimeStats("constant_setup_rpc", sessionFor(policy));
    EXPECT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterCap));
    EXPECT_FALSE(
        stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterPolicyScoped));
  }
  {
    // A function that inspects rows first resolves no limiter without input.
    const auto stats = runtimeStats("demo_rpc", {});
    EXPECT_FALSE(stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterCap));
  }
}

// Every driver of a query shares the limiter for its policy, also when the
// registry compares a reduced form of that policy: min_limit equal to
// max_limit pins capacity at the ceiling, as fixed capacity does.
TEST_F(RPCOperatorTest, driversShareTheLimiterForTheirPolicy) {
  sealDefaultBackend({.ceiling = 7});
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(8);
  for (int i = 0; i < 8; ++i) {
    inputs.push_back(makeNamedRowVector(
        "prompt", makeFlatVector<StringView>({"a", "b", "c", "d"})));
  }
  auto plan = PlanBuilder()
                  .values(inputs, /*parallelizable=*/true)
                  .rpc("demo_rpc(prompt)")
                  .planNode();
  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan)
      .maxDrivers(4)
      .config(core::QueryConfig::kRpcRateLimiterMinLimit, 25)
      .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 25)
      .countResults(task);
  EXPECT_TRUE(ranOnPolicyScopedLimiter(task));

  const core::QueryConfig defaults{
      std::unordered_map<std::string, std::string>{}};
  const auto lookup = RPCRateLimiterRegistry::global().getForPolicy(
      "",
      {.adaptive = defaults.rpcRateLimiterAdaptiveEnabled(),
       .ceiling = 25,
       .floor = 25,
       .decreaseFactor = defaults.rpcRateLimiterDecreaseFactor()});
  EXPECT_TRUE(lookup.policyScoped);
  EXPECT_GT(lookup.limiter->stats().peakPending, 0)
      << "the query's drivers ran under another limiter";
  EXPECT_LE(lookup.limiter->stats().peakPending, 25);
}

TEST_F(RPCOperatorTest, resolvesAdmissionKeyBeforeDispatch) {
  PreAdmissionKeyRPCFunction::reset();
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"hello"}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("pre_admission_tier_rpc(prompt)")
                  .planNode();
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 1);
  EXPECT_TRUE(PreAdmissionKeyRPCFunction::dispatchedAfterPrepare());
  EXPECT_EQ(
      RPCRateLimiterRegistry::global()
          .get("provisional-tier")
          .stats()
          .peakPending,
      0);
  EXPECT_EQ(
      RPCRateLimiterRegistry::global().get("resolved-tier").stats().peakPending,
      1);
}

TEST_F(RPCOperatorTest, constantSetupSkipsInputPreparation) {
  ConstantSetupRPCFunction::reset();
  std::vector<RowVectorPtr> inputs{
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"first input"})),
      makeNamedRowVector(
          "prompt", makeFlatVector<StringView>({"second input"})),
  };

  auto plan =
      PlanBuilder().values(inputs).rpc("constant_setup_rpc(prompt)").planNode();
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 2);
  EXPECT_EQ(ConstantSetupRPCFunction::numPreparations(), 0);
}

TEST_F(RPCOperatorTest, localOnlyInputsBypassAdmissionAndCongestion) {
  DeferredAdmissionRPCFunction::reset();
  std::vector<RowVectorPtr> inputs{
      makeNamedRowVector(
          "prompt", makeNullableFlatVector<StringView>({std::nullopt})),
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"live prompt"})),
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"local error"})),
  };

  auto plan = PlanBuilder()
                  .values(inputs)
                  .rpc("deferred_admission_rpc(prompt)")
                  .planNode();
  const auto task = AssertQueryBuilder(plan).assertResults(makeRowVector({
      makeNullableFlatVector<std::string>(
          {std::nullopt, "live prompt", "local error"}),
      makeNullableFlatVector<std::string>(
          {std::nullopt, "demo: live prompt", std::nullopt}),
  }));
  EXPECT_EQ(
      RPCRateLimiterRegistry::global()
          .get("local-only-provisional-tier")
          .stats()
          .peakPending,
      0);
  EXPECT_EQ(
      RPCRateLimiterRegistry::global()
          .get("resolved-after-live-row")
          .stats()
          .peakPending,
      1);
  EXPECT_EQ(DeferredAdmissionRPCFunction::numCongestionEvaluations(), 1);
  const auto memoryStats = rpcOperatorMemoryStats(task, plan->id());
  EXPECT_GT(memoryStats.numExternalAllocs, 0);
  EXPECT_EQ(memoryStats.numExternalAllocs, memoryStats.numExternalFrees);
  EXPECT_GT(memoryStats.cumulativeExternalBytes, 0);
}

TEST_F(RPCOperatorTest, localOnlyPerRowPayloadMemoryIsAccounted) {
  DeferredAdmissionRPCFunction::reset();
  auto input = makeNamedRowVector(
      "prompt",
      makeNullableFlatVector<std::string>({std::nullopt, "local error"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("deferred_admission_rpc(prompt)")
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan).copyResults(pool(), task);

  ASSERT_EQ(result->size(), 2);
  const auto& responses = result->childAt(1);
  EXPECT_EQ(BaseVector::countNulls(responses->nulls(), responses->size()), 2);
  const auto memoryStats = rpcOperatorMemoryStats(task, plan->id());
  EXPECT_GT(memoryStats.numExternalAllocs, 0);
  EXPECT_EQ(memoryStats.numExternalAllocs, memoryStats.numExternalFrees);
  EXPECT_GT(memoryStats.cumulativeExternalBytes, 0);
}

TEST_F(RPCOperatorTest, invalidRequestRowsAreCountedByErrorKind) {
  DeferredAdmissionRPCFunction::reset();
  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"live prompt", "local error"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("deferred_admission_rpc(prompt)")
                  .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).countResults(task);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(
      stats.hasCustomStat(RPCOperatorStatNames::kRpcErrorKindInvalidRequest));
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcErrorKindInvalidRequest), 1);
}

TEST_F(RPCOperatorTest, localOnlyBatchPreservesPendingAdmittedRows) {
  DeferredAdmissionBatchRPCFunction::reset();
  std::vector<RowVectorPtr> inputs{
      makeNamedRowVector(
          "prompt", makeNullableFlatVector<StringView>({std::nullopt})),
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"live prompt"})),
      makeNamedRowVector(
          "prompt", makeNullableFlatVector<StringView>({std::nullopt})),
  };

  auto plan =
      PlanBuilder()
          .values(inputs)
          .rpc("deferred_admission_batch_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();
  const auto task = AssertQueryBuilder(plan).assertResults(makeRowVector({
      makeNullableFlatVector<std::string>(
          {std::nullopt, "live prompt", std::nullopt}),
      makeNullableFlatVector<std::string>(
          {std::nullopt, "Batch response for: live prompt", std::nullopt}),
  }));
  EXPECT_EQ(
      RPCRateLimiterRegistry::global()
          .get("batch-local-only-provisional-tier")
          .stats()
          .peakPending,
      0);
  EXPECT_EQ(
      RPCRateLimiterRegistry::global()
          .get("resolved-batch-after-live-row")
          .stats()
          .peakPending,
      1);
  EXPECT_EQ(DeferredAdmissionBatchRPCFunction::numCongestionEvaluations(), 1);
  const auto memoryStats = rpcOperatorMemoryStats(task, plan->id());
  EXPECT_GT(memoryStats.numExternalAllocs, 0);
  EXPECT_EQ(memoryStats.numExternalAllocs, memoryStats.numExternalFrees);
  EXPECT_GT(memoryStats.cumulativeExternalBytes, 0);
}

TEST_F(RPCOperatorTest, localOnlyBatchPayloadMemoryIsAccounted) {
  DeferredAdmissionBatchRPCFunction::reset();
  auto input = makeNamedRowVector(
      "prompt",
      makeNullableFlatVector<StringView>({std::nullopt, std::nullopt}));
  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("deferred_admission_batch_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan).copyResults(pool(), task);

  ASSERT_EQ(result->size(), 2);
  const auto& responses = result->childAt(1);
  EXPECT_EQ(BaseVector::countNulls(responses->nulls(), responses->size()), 2);
  const auto memoryStats = rpcOperatorMemoryStats(task, plan->id());
  EXPECT_GT(memoryStats.numExternalAllocs, 0);
  EXPECT_EQ(memoryStats.numExternalAllocs, memoryStats.numExternalFrees);
  EXPECT_GT(memoryStats.cumulativeExternalBytes, 0);
}

// Dispatch must respect the backend's admission cap, not only the per-driver
// window. Other drivers can exhaust the backend while this driver's window is
// still open, and an ungated flush loop then pushes pending past the cap.
// The mock holds each flush open so several are genuinely in flight; the
// backend ceiling is set below the BATCH window's starting value so the backend
// is the binding constraint and the two gates are distinguishable. Intake is
// bounded by accumulator depth, and BATCH flushes from isBlocked() as well as
// addInput(). Both halves are needed: bounding intake without the flush from
// isBlocked() lets a full accumulator sit forever once needsInput() stops
// taking input, because BATCH has no other place to drain from -- the query
// then hangs rather than fails. Feeds many input vectors so needsInput() is
// actually consulted mid-stream, against a backend admitting one flush.
TEST_F(RPCOperatorTest, batchMakesProgressWhenIntakeIsThrottled) {
  constexpr int64_t kCeiling = 1;
  constexpr int32_t kVectors = 8;
  constexpr int32_t kRowsPerVector = 16;

  std::vector<std::string> storage;
  storage.reserve(kVectors * kRowsPerVector);
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kVectors);
  for (int v = 0; v < kVectors; ++v) {
    std::vector<StringView> prompts;
    prompts.reserve(kRowsPerVector);
    for (int i = 0; i < kRowsPerVector; ++i) {
      storage.push_back(fmt::format("throttled-{}-{}", v, i));
      prompts.emplace_back(storage.back());
    }
    inputs.push_back(
        makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts)));
  }

  const DefaultBackendPolicy policy{.ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto plan = PlanBuilder()
                  .values(inputs)
                  .rpc(
                      "demo_batch_rpc_held(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/4)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kVectors * kRowsPerVector);
  expectRanOnSharedLimiter(task);

  // Every row comes out: throttling intake never stranded the accumulator.
  EXPECT_LE(limiter.stats().peakPending, kCeiling);
}

TEST_F(RPCOperatorTest, perRowReadyResponseReleasesAdmissionBeforeDrain) {
  constexpr int64_t kRows = 8;
  std::vector<std::string> storage;
  std::vector<StringView> prompts;
  storage.reserve(kRows);
  prompts.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    storage.push_back(fmt::format("single-flight-{}", i));
    prompts.emplace_back(storage.back());
  }
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.ceiling = 1};
  auto& limiter = sealDefaultBackend(policy);

  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  std::shared_ptr<exec::Task> task;
  EXPECT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);

  EXPECT_LE(limiter.stats().peakPending, 1);
}

// Contended admission: the backend's slots are held by someone else, so a
// refused dispatch meets numInFlight() == 0 -- the state the other operator
// tests never reach, since they run one driver against a backend nothing else
// holds. The test body holds the ceiling itself and releases from a background
// thread, which is the only way to produce a refusal this operator cannot
// resolve alone.
//
// Coverage, not a regression guard for one defect: it exercises a state nothing
// else does, and asserts the query neither hangs nor comes up short. Reverting
// the end-of-input guards does not fail it, because the Driver only calls
// noMoreInput() when needsInput() is true and a non-empty buffer already makes
// that false.
void RPCOperatorTest::runContendedDrain(bool batchMode) {
  constexpr int64_t kCeiling = 2;
  constexpr int32_t kRows = 64;

  std::vector<std::string> storage;
  std::vector<StringView> prompts;
  storage.reserve(kRows);
  prompts.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    storage.push_back(fmt::format("contended-{}", i));
    prompts.emplace_back(storage.back());
  }
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  // Occupy every slot before the query starts, so the operator's first
  // dispatch is refused with nothing of its own in flight.
  std::vector<RPCRateLimiter::Token> held;
  held.reserve(kCeiling);
  for (int64_t i = 0; i < kCeiling; ++i) {
    held.push_back(limiter.acquire());
  }

  std::thread releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    held.clear();
  });

  auto plan = batchMode
      ? PlanBuilder()
            .values(input)
            .rpc(
                "demo_batch_rpc(prompt)",
                RPCStreamingMode::kBatch,
                /*dispatchBatchSize=*/8)
            .planNode()
      : PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);
  releaser.join();

  // Nothing dropped while admission held the rows, and nothing hung.
}

TEST_F(RPCOperatorTest, perRowDrainCompletesWhileTheBackendIsHeld) {
  runContendedDrain(/*batchMode=*/false);
}

TEST_F(RPCOperatorTest, batchDrainCompletesWhileTheBackendIsHeld) {
  runContendedDrain(/*batchMode=*/true);
}

// close() runs on operators that never initialized. Driver::closeOperators()
// walks every operator regardless of whether Driver::initializeOperators() ran,
// so a task that terminates during setup reaches close() with limiter_ still
// null -- and recordRuntimeStats() dereferenced it unconditionally, taking the
// worker down with SIGSEGV rather than failing the query.
//
// An unregistered function name reaches that state deterministically:
// initialize() throws at the "Unknown RPC function" check, which runs long
// before limiter_ is assigned. The query must surface that error; without the
// guard the process dies instead.
TEST_F(RPCOperatorTest, closeWithoutInitializeDoesNotCrash) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  // PlanBuilder::rpc() rejects an unregistered function, so the node is built
  // directly.
  auto plan = std::make_shared<core::RPCNode>(
      "rpc",
      PlanBuilder().values(input).planNode(),
      std::make_shared<core::CallTypedExpr>(
          VARCHAR(),
          "no_such_rpc_function_registered",
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "prompt")),
      "result",
      ROW({"prompt", "result"}, VARCHAR()),
      RPCStreamingMode::kBatch);

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(), "Unknown RPC function");
}

// A standalone operator on a live DriverCtx reproduces the teardown that
// skips close() without the race: the second RPCState reference stands in
// for an in-flight callback, and the input vector's use count is what the
// operator would otherwise still be holding at ~Task.
TEST_F(RPCOperatorTest, retainedInputIsReleasedWhenCloseNeverRuns) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  auto rpcNode = std::dynamic_pointer_cast<const core::RPCNode>(plan);
  ASSERT_TRUE(rpcNode != nullptr);

  // Parallel mode so Task::init() compiles no drivers: a Driver would hold
  // the Task alive through its DriverCtx and this Task would never be
  // destroyed.
  auto task = exec::Task::create(
      "retained-input-release",
      core::PlanFragment{plan},
      /*destination=*/0,
      core::QueryCtx::create(driverExecutor_.get()),
      exec::Task::ExecutionMode::kParallel);
  exec::DriverCtx driverCtx{
      task,
      /*driverId=*/0,
      /*pipelineId=*/0,
      /*splitGroupId=*/0,
      /*partitionId=*/0};

  auto rpcOperator =
      std::make_unique<RPCOperator>(/*operatorId=*/0, &driverCtx, rpcNode);
  auto retained = BaseVector::create(VARCHAR(), 3, pool());
  rpcOperator->testingState()->storeInputBatch(
      std::vector<VectorPtr>{retained}, /*rowCount=*/3);
  ASSERT_GT(retained.use_count(), 1);
  auto callbackRef = rpcOperator->testingState();

  rpcOperator.reset();

  EXPECT_EQ(retained.use_count(), 1)
      << "the operator was destroyed without close() and kept its input "
         "vectors, which upstream pools would still be reserving at ~Task";
  EXPECT_TRUE(callbackRef->getInputBatchColumns(0).empty());
}

// The reservation the arbitrator actually checks. The test above retains a
// vector from VectorTestBase::pool(), which is rooted outside the Task, so
// the Task's pools read zero whether or not the operator released -- it can
// only observe a use count. The driver allocates input from an upstream
// operator's pool, a leaf of the Task's tree, and those bytes reach the query
// root: left there, ~Task drops the root with a live reservation and
// SharedArbitrator::removePool fails pool->reservedBytes() == 0, taking the
// worker down with SIGABRT.
TEST_F(RPCOperatorTest, retainedInputReleasesTheTaskPoolReservation) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  auto rpcNode = std::dynamic_pointer_cast<const core::RPCNode>(plan);
  ASSERT_TRUE(rpcNode != nullptr);

  // Parallel mode so Task::init() compiles no drivers: a Driver would hold
  // the Task alive through its DriverCtx and this Task would never be
  // destroyed.
  auto task = exec::Task::create(
      "retained-input-reservation",
      core::PlanFragment{plan},
      /*destination=*/0,
      core::QueryCtx::create(driverExecutor_.get()),
      exec::Task::ExecutionMode::kParallel);
  exec::DriverCtx driverCtx{
      task,
      /*driverId=*/0,
      /*pipelineId=*/0,
      /*splitGroupId=*/0,
      /*partitionId=*/0};

  auto rpcOperator =
      std::make_unique<RPCOperator>(/*operatorId=*/0, &driverCtx, rpcNode);
  auto* upstreamPool = driverCtx.addOperatorPool("values-0", "Values");
  constexpr vector_size_t kRetainedRows{1 << 16};
  rpcOperator->testingState()->storeInputBatch(
      std::vector<VectorPtr>{
          BaseVector::create(VARCHAR(), kRetainedRows, upstreamPool)},
      kRetainedRows);
  auto callbackRef = rpcOperator->testingState();
  ASSERT_GT(task->pool()->reservedBytes(), 0)
      << "the retained vector must reserve on the Task's tree, otherwise this "
         "test cannot see the invariant it is about";

  rpcOperator.reset();

  EXPECT_EQ(task->pool()->reservedBytes(), 0)
      << "the operator was destroyed without close() and left its input "
         "vectors reserving on the Task's pools; ~Task would abort in "
         "SharedArbitrator::removePool";
  // Keep a regression from taking the whole binary down in ~Task.
  callbackRef->releaseAllInputBatches();
}

// Claims VARCHAR at the plan level -- so RPCNode's own checks pass -- while
// the function actually returns BIGINT. This is the disagreement the plan node
// cannot see: it compares the CALL expression against the declared column, not
// against the registered function.
class MismatchedResultTypeRPCFunction : public AsyncRPCFunction {
 public:
  void initialize(
      const core::QueryConfig&,
      const std::vector<TypePtr>&,
      const std::vector<VectorPtr>&,
      RPCStreamingMode) override {}

  std::string name() const override {
    return "mismatched_result_type_rpc";
  }

  TypePtr resultType() const override {
    return BIGINT();
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(const SelectivityVector&, const std::vector<VectorPtr>&)
      override {
    VELOX_UNREACHABLE("initialize() rejects this function before dispatch");
  }

  VectorPtr buildOutput(const std::vector<RPCResponse>&, memory::MemoryPool*)
      const override {
    VELOX_UNREACHABLE("initialize() rejects this function before output");
  }
};

// The framework owns the destination type and the function owns the mapping
// onto it. RPCNode checks the CALL expression against the declared column, but
// nothing checked the registered function agrees with either -- so a function
// returning a different type produced a RowVector whose child disagreed with
// its own declared type, and the failure surfaced downstream.
TEST_F(RPCOperatorTest, resultColumnTypeMustMatchTheFunction) {
  registerFunction<MismatchedResultTypeRPCFunction>(
      "mismatched_result_type_rpc");

  auto input = makeNamedRowVector("prompt", makeFlatVector<StringView>({"a"}));
  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("mismatched_result_type_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(),
      "RPC function 'mismatched_result_type_rpc' returns BIGINT but the plan "
      "declares column '__rpc_result' as VARCHAR");
}

TEST_F(RPCOperatorTest, batchDispatchRespectsTheBackendCap) {
  constexpr int64_t kCeiling = 1;
  std::vector<std::string> storage;
  std::vector<StringView> prompts;
  storage.reserve(64);
  prompts.reserve(64);
  for (int i = 0; i < 64; ++i) {
    storage.push_back(fmt::format("row-{}", i));
    prompts.emplace_back(storage.back());
  }
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "demo_batch_rpc_held(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/4)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      64);
  expectRanOnSharedLimiter(task);

  EXPECT_LE(limiter.stats().peakPending, kCeiling)
      << "dispatch admitted more than the backend cap";
}

TEST_F(RPCOperatorTest, batchFanOutIsClippedToGrantedAdmissionUnits) {
  constexpr int64_t kCeiling = 2;
  constexpr int32_t kRows = 8;
  FanOutBatchRPCFunction::flushSizes().clear();

  std::vector<StringView> prompts(kRows, StringView("prompt"));
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.ceiling = kCeiling};
  sealDefaultBackend(policy);

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "fan_out_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/kRows)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);

  for (const auto flushSize : FanOutBatchRPCFunction::flushSizes()) {
    EXPECT_GE(flushSize, 1);
    EXPECT_LE(flushSize, kCeiling);
  }
}

TEST_F(RPCOperatorTest, batchSuccessFromStaleEpochDoesNotUndoOverload) {
  constexpr int64_t kCeiling = 8;
  constexpr int32_t kRows = 8;

  std::vector<StringView> prompts(kRows, StringView("prompt"));
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "recovering_fan_out_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/kRows)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);

  const auto limiterStats = limiter.stats();
  ASSERT_EQ(limiterStats.peakPending, kRows);
  EXPECT_EQ(limiterStats.capacity, 2)
      << "success admitted before overload must not recover the new epoch";
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcStaleSuccessUnits));
  // The fan-out batch reserves one admission unit per row.
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcStaleSuccessUnits), kRows);
}

// No row of a rate-limited batch succeeds, so the batch earns no credit and has
// no stale success to report, although its own overload cut the window.
TEST_F(RPCOperatorTest, rateLimitedBatchReportsNoStaleSuccess) {
  constexpr int64_t kCeiling = 8;
  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"a", "b", "c", "d"}));
  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("rate_limited_batch_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      4);
  expectRanOnSharedLimiter(task);
  ASSERT_LT(limiter.stats().limitMilli, kCeiling * 1'000);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcStaleSuccessUnits));
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcStaleSuccessUnits), 0);
}

// A flush whose own overload cuts the window still reports its credit as stale
// success: some of its rows succeeded, and the overload arrived after their
// admission.
TEST_F(RPCOperatorTest, partlyRateLimitedBatchReportsStaleSuccess) {
  constexpr int64_t kCeiling = 8;
  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"a", "b", "c", "d"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "partly_rate_limited_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/4)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      4);
  expectRanOnSharedLimiter(task);
  ASSERT_LT(limiter.stats().limitMilli, kCeiling * 1'000);

  // One native flush holds one admission unit.
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcStaleSuccessUnits));
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcStaleSuccessUnits), 1);
}

// Credit follows the rows, not the verdict: a batch whose rows succeed reports
// its discarded units as stale success even without a success verdict.
TEST_F(RPCOperatorTest, staleBatchWithoutSuccessVerdictReportsStaleSuccess) {
  constexpr int64_t kCeiling = 8;
  constexpr int32_t kRows = 8;
  std::vector<StringView> prompts(kRows, StringView("prompt"));
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "recovering_no_verdict_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/kRows)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);
  // Crediting the batch would have grown the window back above 2.
  ASSERT_EQ(limiter.stats().capacity, 2);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcStaleSuccessUnits));
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcStaleSuccessUnits), kRows);
}

TEST_F(RPCOperatorTest, currentEpochSuccessRecoversAlongsideOverload) {
  constexpr int64_t kCeiling = 8;
  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"slow", "overload", "recover"}));
  auto plan =
      PlanBuilder().values(input).rpc("mixed_epoch_rpc(prompt)").planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .configs(sessionFor(policy))
          .config(core::QueryConfig::kRpcCongestionMaxWindow, 2)
          .countResults(task),
      3);
  expectRanOnSharedLimiter(task);

  const auto expectedLimitMilli =
      static_cast<int64_t>(std::llround(std::sqrt(4.0 * 4.0 + 2.0) * 1'000));
  EXPECT_EQ(limiter.stats().limitMilli, expectedLimitMilli)
      << "a current-epoch success must recover after the overload marker";
  // The slow row was admitted before the overload.
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcStaleSuccessUnits));
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcStaleSuccessUnits), 1);
}

// Successes after overloaded attempts are counted, but neither the driver
// window nor the shared limiter treats them as an overload.
TEST_F(RPCOperatorTest, overloadedAttemptsAreCounted) {
  constexpr int64_t kCeiling = 8;
  const DefaultBackendPolicy policy{
      .adaptive = true, .ceiling = kCeiling, .floor = 1};
  auto& limiter = sealDefaultBackend(policy);

  std::vector<StringView> prompts(kCeiling, StringView("retried"));
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("overloaded_attempts_rpc(prompt)")
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kCeiling);
  expectRanOnSharedLimiter(task);

  EXPECT_EQ(limiter.stats().limitMilli, kCeiling * 1'000);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  auto statSum = [&](const std::string& key) -> int64_t {
    return stats.hasCustomStat(key) ? stats.customSum(key) : -1;
  };
  EXPECT_EQ(statSum(RPCOperatorStatNames::kRpcCongestionOverloadShrinks), -1);
  EXPECT_EQ(
      statSum(RPCOperatorStatNames::kRpcSuccessesAfterOverload), kCeiling);
  EXPECT_EQ(
      statSum(RPCOperatorStatNames::kRpcOverloadedAttempts),
      OverloadedAttemptsRPCFunction::kNumOverloadedAttempts * kCeiling);
}

TEST_F(RPCOperatorTest, batchAdmissionRequiresOneUnitForOneRow) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"prompt"}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "invalid_admission_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/1)
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(), "one admission unit");
}

// BATCH reserves one slot per flushBatch() regardless of row count, so its
// AIMD recovery has to be credited in batches too. onSuccess() steps capacity
// by units/capacity, so crediting the row count against a capacity counted in
// batches makes the step grow as capacity shrinks -- recovery accelerates
// exactly when it should be most cautious, undoing the multiplicative
// decrease.
//
// Four batches of 16 rows drain successfully from a capacity of 2. Canonical
// additive credit uses four backend completions, independent of the 64 rows.
TEST_F(RPCOperatorTest, batchAimdRecoversPerBatchNotPerRow) {
  constexpr int64_t kCeiling = 64;
  constexpr int64_t kRows = 64;
  constexpr int32_t kDispatchBatchSize = 16;
  constexpr int64_t kNumBatches = kRows / kDispatchBatchSize;

  std::vector<std::string> storage;
  std::vector<StringView> prompts;
  storage.reserve(kRows);
  prompts.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    storage.push_back(fmt::format("row-{}", i));
    prompts.emplace_back(storage.back());
  }
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  // Below the built-in floor of 50, so capacity can be driven low enough
  // for the two crediting schemes to separate.
  const DefaultBackendPolicy policy{
      .adaptive = true, .ceiling = kCeiling, .floor = 1};
  auto& limiter = sealDefaultBackend(policy);

  // 64 -> 32 -> 16 -> 8 -> 4 -> 2.
  for (int i = 0; i < 5; ++i) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, /*units=*/0);
  }
  const int64_t shrunk = limiter.stats().capacity;
  ASSERT_EQ(shrunk, 2) << "precondition: capacity must start well below the "
                          "rows per batch for the two schemes to differ";

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "demo_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/kDispatchBatchSize)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);

  const auto expectedLimitMilli = static_cast<int64_t>(std::llround(
      std::sqrt(
          static_cast<double>(shrunk * shrunk) +
          2.0 * static_cast<double>(kNumBatches)) *
      1'000));
  EXPECT_EQ(limiter.stats().limitMilli, expectedLimitMilli)
      << "recovery must credit backend batches rather than their rows";
}

TEST_F(
    RPCOperatorTest,
    batchSuccessWithoutLatencyRecoversLimiterWithoutRttSample) {
  constexpr int64_t kCeiling = 8;
  constexpr int64_t kRows = 16;

  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, /*units=*/0);
  ASSERT_EQ(limiter.stats().capacity, 4);

  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>({
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
      }));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "no_latency_success_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/2)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .configs(sessionFor(policy))
          .config(core::QueryConfig::kRpcCongestionAsyncGrowthMaxWindow, 64)
          .countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);

  EXPECT_EQ(limiter.stats().capacity, 5);
  EXPECT_EQ(limiter.stats().limitMilli, 5'657);
  // Driver-local growth and shared pacing learn from separate evidence: clean
  // completion count grows the BATCH window, while elapsed service duration
  // still trains the shared limiter's horizon.
  EXPECT_NE(limiter.stats().serviceHorizonNanos, 1'000'000'000);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_FALSE(stats.hasCustomStat(RPCOperatorStatNames::kRpcBaselineRttNanos));
  auto statSum = [&](const std::string& key) -> int64_t {
    return stats.hasCustomStat(key) ? stats.customSum(key) : -1;
  };
  // 16 rows at 2 per batch complete as 8 flushes: 8 clean units lift the
  // BATCH window 2 -> 3. Without the NoLatency branch it would stay at 2;
  // counting rows instead of flushes would reach 4.
  EXPECT_EQ(statSum(RPCOperatorStatNames::kRpcCongestionWindowFinal), 3);
}

TEST_F(
    RPCOperatorTest,
    batchSuccessWithoutLatencyTrainsHorizonWhenGrowthDisabled) {
  constexpr int64_t kCeiling = 8;
  constexpr int64_t kRows = 16;

  const DefaultBackendPolicy policy{.adaptive = true, .ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, /*units=*/0);
  ASSERT_EQ(limiter.stats().capacity, 4);

  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>({
          "0",
          "1",
          "2",
          "3",
          "4",
          "5",
          "6",
          "7",
          "8",
          "9",
          "10",
          "11",
          "12",
          "13",
          "14",
          "15",
      }));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "no_latency_success_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/2)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  // No async-growth cap: the default 0 disables clean growth.
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      kRows);
  expectRanOnSharedLimiter(task);

  EXPECT_EQ(limiter.stats().capacity, 5);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  auto statSum = [&](const std::string& key) -> int64_t {
    return stats.hasCustomStat(key) ? stats.customSum(key) : -1;
  };
  // Growth disabled: legacy behavior end to end — the BATCH window stays
  // pinned at its start, and the no-latency round trips still train the
  // shared service horizon instead of leaving it at the bootstrap.
  EXPECT_EQ(statSum(RPCOperatorStatNames::kRpcCongestionWindowFinal), 2);
  EXPECT_NE(limiter.stats().serviceHorizonNanos, 1'000'000'000);
}

// The low-water capacity lands on every query, not only the ones where the
// backend shrank. A query that never backed off reports the ceiling, which is
// a real capacity; the old sentinel zero was indistinguishable from "the stat
// was never recorded".
TEST_F(RPCOperatorTest, lowWaterCapacityIsReportedWhenTheBackendStaysHealthy) {
  constexpr int64_t kCeiling = 16;
  std::vector<std::string> storage;
  std::vector<StringView> prompts;
  storage.reserve(8);
  prompts.reserve(8);
  for (int i = 0; i < 8; ++i) {
    storage.push_back(fmt::format("healthy-{}", i));
    prompts.emplace_back(storage.back());
  }
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>(prompts));

  const DefaultBackendPolicy policy{.ceiling = kCeiling};
  auto& limiter = sealDefaultBackend(policy);

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("demo_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan).configs(sessionFor(policy)).countResults(task),
      8);
  expectRanOnSharedLimiter(task);

  // Nothing drove an overload, so the cap never shrank.
  ASSERT_EQ(limiter.stats().capacity, kCeiling);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterMinCap))
      << "the low-water stat must land even when the backend never shrank";
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcRateLimiterMinCap), kCeiling);
}

// The shared limiter outlives the query and paces before the driver's first
// admission request. The span stats cover none of that pacing, and the limiter
// holds its ceiling over the whole span; nothing overloaded, so no success
// went stale.
TEST_F(RPCOperatorTest, limiterStatsCoverOnlyTheOperatorsSpan) {
  constexpr int64_t kCeiling = 2;
  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"a", "b", "c", "d"}));

  // The policy the query resolves with demo_batch_rpc's floor of 0.25.
  auto& limiter = RPCRateLimiterRegistry::global().get("");
  ASSERT_TRUE(limiter.initializeOrMatch({
      .adaptive = true,
      .ceiling = kCeiling,
      .floor = 0.25,
      .decreaseFactor = 0.5,
  }));
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  ASSERT_EQ(limiter.stats().limitMilli, 500);
  // Pace until the limiter has measured some pacing time, which the span must
  // exclude.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (limiter.stats().pacingNanos == 0) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline);
  }
  // Successes with a 1ms service horizon grow the window back to one unit
  // within a horizon, and then to the ceiling.
  constexpr int64_t kHorizonNanos{1'000'000};
  while (limiter.stats().limitMilli < kCeiling * 1'000) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline);
    limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1'000, kHorizonNanos);
  }

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("demo_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, kCeiling)
          .countResults(task),
      4);
  expectRanOnSharedLimiter(task);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(
      stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterAttachedNanos));
  const auto& attached =
      stats.customStat(RPCOperatorStatNames::kRpcRateLimiterAttachedNanos);
  EXPECT_EQ(attached.unit, RuntimeCounter::Unit::kNanos);
  // One driver published it once, whatever the live samples did.
  EXPECT_EQ(attached.count, 1);
  EXPECT_GT(attached.sum, 0);
  ASSERT_TRUE(
      stats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterPacingNanos));
  const auto& pacing =
      stats.customStat(RPCOperatorStatNames::kRpcRateLimiterPacingNanos);
  EXPECT_EQ(pacing.unit, RuntimeCounter::Unit::kNanos);
  EXPECT_EQ(pacing.sum, 0);
  // A constant window integrates to the window times the span.
  ASSERT_TRUE(stats.hasCustomStat(
      RPCOperatorStatNames::kRpcRateLimiterLimitMilliUnitSeconds));
  EXPECT_EQ(
      stats.customSum(
          RPCOperatorStatNames::kRpcRateLimiterLimitMilliUnitSeconds),
      std::llround(
          kCeiling * 1'000 * (static_cast<double>(attached.sum) / 1e9)));
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcStaleSuccessUnits));
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcStaleSuccessUnits), 0);
}

// constant_setup_rpc binds the limiter in initialize(), before any input. The
// span stats start at a driver's first admission request instead, so a driver
// that gets no input reports none of them.
TEST_F(RPCOperatorTest, limiterSpanStartsAtTheFirstAdmissionRequest) {
  const auto runQuery = [&](const std::vector<StringView>& prompts) {
    auto plan = PlanBuilder()
                    .values(makeNamedRowVector(
                        "prompt", makeFlatVector<StringView>(prompts)))
                    .rpc("constant_setup_rpc(prompt)")
                    .planNode();
    std::shared_ptr<exec::Task> task;
    EXPECT_EQ(AssertQueryBuilder(plan).countResults(task), prompts.size());
    return PlanNodeStats::extract(task->taskStats(), plan->id());
  };

  {
    const auto customStats = runQuery({});
    ASSERT_TRUE(
        customStats.hasCustomStat(RPCOperatorStatNames::kRpcRateLimiterCap));
    EXPECT_FALSE(customStats.hasCustomStat(
        RPCOperatorStatNames::kRpcRateLimiterAttachedNanos));
    EXPECT_FALSE(customStats.hasCustomStat(
        RPCOperatorStatNames::kRpcRateLimiterPacingNanos));
    EXPECT_FALSE(customStats.hasCustomStat(
        RPCOperatorStatNames::kRpcRateLimiterLimitMilliUnitSeconds));
  }
  {
    const auto customStats = runQuery({"a", "b"});
    ASSERT_TRUE(customStats.hasCustomStat(
        RPCOperatorStatNames::kRpcRateLimiterAttachedNanos));
    EXPECT_GT(
        customStats.customSum(
            RPCOperatorStatNames::kRpcRateLimiterAttachedNanos),
        0);
    EXPECT_TRUE(customStats.hasCustomStat(
        RPCOperatorStatNames::kRpcRateLimiterPacingNanos));
    EXPECT_TRUE(customStats.hasCustomStat(
        RPCOperatorStatNames::kRpcRateLimiterLimitMilliUnitSeconds));
  }
}

// Each flush is held for 50ms and rate limited. The first overload halves the
// window from one unit to the 0.5 floor, where the second flush waits for its
// paced admission. Over the span, the window integral lies strictly between
// those of the two windows, and part of the span was spent pacing.
TEST_F(RPCOperatorTest, limiterSpanStatsFollowAMidQueryWindowChange) {
  // The policy the query resolves with rate_limited_batch_rpc_held's floor.
  auto& limiter = RPCRateLimiterRegistry::global().get("");
  ASSERT_TRUE(limiter.initializeOrMatch(
      {.adaptive = true, .ceiling = 1, .floor = 0.5, .decreaseFactor = 0.5}));

  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "rate_limited_batch_rpc_held(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/1)
                  .planNode();
  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 1)
          .countResults(task),
      2);
  expectRanOnSharedLimiter(task);
  ASSERT_EQ(limiter.stats().limitMilli, 500);

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  const auto attachedNanos =
      stats.customSum(RPCOperatorStatNames::kRpcRateLimiterAttachedNanos);
  const auto pacingNanos =
      stats.customSum(RPCOperatorStatNames::kRpcRateLimiterPacingNanos);
  const auto limitMilliSeconds = stats.customSum(
      RPCOperatorStatNames::kRpcRateLimiterLimitMilliUnitSeconds);
  // Each window held over the whole span bounds the integral, which is rounded
  // to the nearest milli-unit-second.
  const double attachedSeconds = static_cast<double>(attachedNanos) / 1e9;
  EXPECT_GT(limitMilliSeconds, 500 * attachedSeconds + 0.5);
  EXPECT_LT(limitMilliSeconds, 1'000 * attachedSeconds - 0.5);
  EXPECT_GT(pacingNanos, 0);
  EXPECT_LT(pacingNanos, attachedNanos);
}

TEST_F(RPCOperatorTest, sessionZeroDelegatesToPerLimiterFloor) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto demoPlan =
      PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(demoPlan)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
          .config(core::QueryConfig::kRpcRateLimiterHardLimit, 2)
          .countResults(),
      1);

  const auto& demoLimiter = RPCRateLimiterRegistry::global().get("");
  EXPECT_DOUBLE_EQ(demoLimiter.config().floor, 0.25);
  EXPECT_EQ(demoLimiter.stats().hardLimit, 2);

  auto cappedPlan =
      PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(cappedPlan)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
          .config(core::QueryConfig::kRpcRateLimiterHardLimit, 2)
          .countResults(),
      1);

  const auto& cappedLimiter =
      RPCRateLimiterRegistry::global().get("capped-rpc");
  EXPECT_DOUBLE_EQ(cappedLimiter.config().floor, 0.5);
  EXPECT_EQ(cappedLimiter.stats().hardLimit, 2);
}

TEST_F(RPCOperatorTest, functionFloorOverridesSessionFloor) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 4)
          .config(core::QueryConfig::kRpcRateLimiterHardLimit, 8)
          .countResults(),
      1);

  // The function's floor (0.5) wins; a positive hard limit still caps its
  // ceiling (64).
  const auto& limiter = RPCRateLimiterRegistry::global().get("capped-rpc");
  EXPECT_DOUBLE_EQ(limiter.config().floor, 0.5);
  EXPECT_EQ(limiter.stats().hardLimit, 8);
}

TEST_F(RPCOperatorTest, defaultHardLimitKeepsSessionPrecedence) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 1);

  // With hard_limit -1, the session defaults (50 and 200) still replace the
  // function's floor (0.5) and ceiling (64).
  const auto& limiter = RPCRateLimiterRegistry::global().get("capped-rpc");
  EXPECT_DOUBLE_EQ(limiter.config().floor, 50.0);
  EXPECT_EQ(limiter.stats().hardLimit, 200);
}

TEST_F(RPCOperatorTest, zeroHardLimitSelectsFunctionBounds) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterHardLimit, 0)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 4)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 128)
          .countResults(),
      1);

  const auto& limiter = RPCRateLimiterRegistry::global().get("capped-rpc");
  EXPECT_DOUBLE_EQ(limiter.config().floor, 0.5);
  EXPECT_EQ(limiter.stats().hardLimit, 64);
}

TEST_F(RPCOperatorTest, sessionBoundsApplyWhenFunctionDeclaresNone) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("undeclared_bounds_rpc(prompt)")
                  .planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 4)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 128)
          .countResults(),
      1);

  const auto& limiter =
      RPCRateLimiterRegistry::global().get("undeclared-bounds-rpc");
  EXPECT_DOUBLE_EQ(limiter.config().floor, 4.0);
  EXPECT_EQ(limiter.stats().hardLimit, 128);
}

TEST_F(RPCOperatorTest, defaultBoundsApplyWhenNeitherDeclares) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("undeclared_bounds_rpc(prompt)")
                  .planNode();
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 1);

  const auto& limiter =
      RPCRateLimiterRegistry::global().get("undeclared-bounds-rpc");
  EXPECT_DOUBLE_EQ(limiter.config().floor, 50.0);
  EXPECT_EQ(limiter.stats().hardLimit, 200);
}

TEST_F(RPCOperatorTest, optedInQueryGetsItsOwnLimiterOnASharedKey) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  // The default query seals the shared limiter with the session's bounds.
  AssertQueryBuilder(plan).countResults();
  // A query that opts in resolves to the function's floor and ceiling.
  AssertQueryBuilder(plan)
      .config(core::QueryConfig::kRpcRateLimiterHardLimit, 0)
      .countResults();

  auto& registry = RPCRateLimiterRegistry::global();
  EXPECT_DOUBLE_EQ(registry.get("capped-rpc").config().floor, 50.0);
  RPCRateLimiter::Config optedIn{
      .adaptive = true, .ceiling = 64, .floor = 0.5, .decreaseFactor = 0.5};
  const auto lookup = registry.getForPolicy("capped-rpc", optedIn);
  EXPECT_TRUE(lookup.policyScoped);
  EXPECT_GE(lookup.limiter->stats().peakPending, 1);
}

// Two functions share an admission key but resolve to different floors. The
// first seals the shared limiter; the second runs under a limiter of its own
// and leaves the shared policy as the first set it.
TEST_F(RPCOperatorTest, laterFunctionLeavesSharedLimiterPolicyUnchanged) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto runFunction = [&](const std::string& functionName) {
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(fmt::format("{}(prompt)", functionName))
                    .planNode();
    std::shared_ptr<exec::Task> task;
    EXPECT_EQ(
        AssertQueryBuilder(plan)
            .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
            .countResults(task),
        1);
    return ranOnPolicyScopedLimiter(task);
  };

  EXPECT_FALSE(runFunction("capped_rpc"));
  EXPECT_TRUE(runFunction("different_floor_same_key_rpc"));

  auto& registry = RPCRateLimiterRegistry::global();
  EXPECT_DOUBLE_EQ(registry.get("capped-rpc").config().floor, 0.5);

  // The second function's own limiter runs its floor. It is the limiter that
  // function's query used: the one row it dispatched went through it.
  const core::QueryConfig defaults{
      std::unordered_map<std::string, std::string>{}};
  const auto scoped = registry.getForPolicy(
      "capped-rpc",
      {.adaptive = defaults.rpcRateLimiterAdaptiveEnabled(),
       .ceiling = defaults.rpcRateLimiterMaxLimit(),
       .floor = 1.0,
       .decreaseFactor = defaults.rpcRateLimiterDecreaseFactor()});
  EXPECT_TRUE(scoped.policyScoped);
  EXPECT_EQ(scoped.limiter->stats().peakPending, 1);
  EXPECT_DOUBLE_EQ(scoped.limiter->config().floor, 1.0);
}

TEST_F(RPCOperatorTest, explicitUnboundedAdaptiveLimitUsesSafetyCeiling) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterHardLimit, 0)
          .countResults(),
      1);

  const auto limiterStats = RPCRateLimiterRegistry::global().get("").stats();
  EXPECT_EQ(limiterStats.hardLimit, 1'000'000);
  EXPECT_EQ(limiterStats.capacity, 200);
}

TEST_F(RPCOperatorTest, defaultFixedLimitPreservesLegacyCapacity) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, false)
          .countResults(),
      1);

  const auto limiterStats = RPCRateLimiterRegistry::global().get("").stats();
  EXPECT_EQ(limiterStats.hardLimit, 200);
  EXPECT_EQ(limiterStats.capacity, 200);
}

TEST_F(RPCOperatorTest, defaultAdaptiveLimitPreservesLegacyCeiling) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 1);

  const auto limiterStats = RPCRateLimiterRegistry::global().get("").stats();
  EXPECT_EQ(limiterStats.hardLimit, 200);
}

TEST_F(RPCOperatorTest, backendErrorsStopPacedRecovery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"fails"}));
  auto plan =
      PlanBuilder().values(input).rpc("backend_error_rpc(prompt)").planNode();
  auto runQuery = [&]() {
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
        .countResults();
  };
  // The first query seals the limiter with the function's fractional floor.
  runQuery();
  auto& limiter = RPCRateLimiterRegistry::global().get("backend-error-rpc");

  // A 10ms service horizon keeps paced waits short. Overloads drive the
  // window to the function's floor of 0.25, and a success starts paced
  // recovery. The next paced admission comes about 1.9 horizons later, when
  // the window has grown to about 0.94, so the limiter is still pacing.
  constexpr int64_t kHorizonNanos{10'000'000};
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, kHorizonNanos);
  while (limiter.stats().limitMilli > 250) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, kHorizonNanos);
  ASSERT_TRUE(limiter.stats().recovering);

  // A call that fails without overload ends the recovery.
  runQuery();
  EXPECT_FALSE(limiter.stats().recovering);
}

TEST_F(RPCOperatorTest, backendErrorBatchesDoNotStartPacedRecovery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"fails"}));
  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("backend_error_batch_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();
  auto runQuery = [&]() {
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
        .countResults();
  };
  // The first query seals the limiter with the function's fractional floor.
  runQuery();
  auto& limiter =
      RPCRateLimiterRegistry::global().get("backend-error-batch-rpc");

  // A 10ms service horizon keeps paced waits short. Overloads drive the
  // window to the function's floor of 0.25 and leave it paced, not
  // recovering.
  constexpr int64_t kHorizonNanos{10'000'000};
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, kHorizonNanos);
  while (limiter.stats().limitMilli > 250) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  ASSERT_FALSE(limiter.stats().recovering);

  // Batches that fail without overload prove no capacity: the window neither
  // starts recovering nor grows.
  runQuery();
  runQuery();
  const auto stats = limiter.stats();
  EXPECT_FALSE(stats.recovering);
  EXPECT_EQ(stats.limitMilli, 250);
}

TEST_F(RPCOperatorTest, historicalBatchOutcomesDoNotBackOffOrEarnRecovery) {
  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>(
          {"historical-success", "historical-rate", "error"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "mixed_historical_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/3)
                  .planNode();
  auto runQuery = [&]() {
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, true)
        .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 20)
        .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
        .countResults();
  };
  ASSERT_EQ(runQuery(), 3);
  auto& limiter =
      RPCRateLimiterRegistry::global().get("mixed-historical-batch-rpc");
  const auto before = limiter.stats();
  ASSERT_GT(before.limitMilli, 250);
  ASSERT_EQ(runQuery(), 3);
  EXPECT_EQ(limiter.stats().limitMilli, before.limitMilli);

  constexpr int64_t kHorizonNanos{10'000'000};
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, kHorizonNanos);
  while (limiter.stats().limitMilli > 250) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  ASSERT_FALSE(limiter.stats().recovering);
  ASSERT_EQ(runQuery(), 3);
  const auto stats = limiter.stats();
  EXPECT_EQ(stats.limitMilli, 250);
  EXPECT_FALSE(stats.recovering);
  EXPECT_EQ(stats.pending, 0);
}

TEST_F(RPCOperatorTest, batchWithASuccessfulRowStartsPacedRecovery) {
  // Both rows go in one flush; the function fails the first and serves the
  // second.
  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"fails", "served"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "backend_error_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/2)
                  .planNode();
  auto runQuery = [&]() {
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
        .countResults();
  };
  runQuery();
  auto& limiter =
      RPCRateLimiterRegistry::global().get("backend-error-batch-rpc");

  constexpr int64_t kHorizonNanos{10'000'000};
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, kHorizonNanos);
  while (limiter.stats().limitMilli > 250) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  ASSERT_FALSE(limiter.stats().recovering);

  // The served row proves capacity, as it would in PER_ROW, although the
  // flush's verdict is kNonOverloadError.
  runQuery();
  EXPECT_TRUE(limiter.stats().recovering);
}

TEST_F(RPCOperatorTest, deferredReceiptFailuresDoNotStartRecovery) {
  auto state = std::make_shared<ReceiptOperatorState>();
  state->submitError = velox::rpc::RPCErrorKind::kBackendError;
  state->floor = 0.25;
  const auto names = registerReceiptFunctions("failed_receipt", state);
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"input"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      fmt::format("{}(prompt) as receipt", names[0]),
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/0,
                      "SUBMIT")
                  .planNode();
  auto run = [&](std::shared_ptr<Task>& task) {
    return AssertQueryBuilder(plan)
        .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, true)
        .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 20)
        .config(core::QueryConfig::kRpcRateLimiterMinLimit, 0)
        .copyResults(pool(), task);
  };
  std::shared_ptr<Task> task;
  run(task);
  auto& limiter = RPCRateLimiterRegistry::global().get("");
  limiter.onOutcome(RPCRateLimiter::Outcome::kSuccess, 1, 10'000'000);
  while (limiter.stats().limitMilli > 250) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, 0);
  }
  ASSERT_FALSE(limiter.stats().recovering);
  auto result = run(task);
  ASSERT_EQ(result->size(), 1);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  ASSERT_TRUE(stats.hasCustomStat(RPCOperatorStatNames::kRpcErrorCount));
  ASSERT_TRUE(
      stats.hasCustomStat(RPCOperatorStatNames::kRpcErrorKindBackendError));
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcErrorCount), 1);
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcErrorKindBackendError), 1);
  EXPECT_FALSE(limiter.stats().recovering);
  EXPECT_EQ(limiter.stats().limitMilli, 250);
  const auto* receipt = result->childAt(1)->asChecked<RowVector>();
  EXPECT_EQ(
      receipt->childAt(1)->asFlatVector<int64_t>()->valueAt(0),
      static_cast<int64_t>(velox::rpc::RPCErrorKind::kBackendError));
}

TEST_F(RPCOperatorTest, concurrentDeferredPollOverloadsCutOneEpoch) {
  auto state = std::make_shared<ReceiptOperatorState>();
  state->pollError = velox::rpc::RPCErrorKind::kRateLimited;
  state->holdPollPair = true;
  const auto names = registerReceiptFunctions("overloaded_receipt", state);
  const OperatorReceiptCodec codec;
  ReceiptBatch receipts{
      std::make_shared<OperatorReceipt>(
          JobReceipt::Outcome::kSubmitted,
          velox::rpc::RPCErrorKind::kNone,
          false,
          0),
      std::make_shared<OperatorReceipt>(
          JobReceipt::Outcome::kSubmitted,
          velox::rpc::RPCErrorKind::kNone,
          false,
          1)};
  auto input = makeNamedRowVector("receipt", codec.encode(receipts, pool()));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      fmt::format("{}(receipt) as ready", names[1]),
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/1,
                      "POLL")
                  .planNode();
  std::shared_ptr<Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, true)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 20)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 1)
          .countResults(task),
      2);
  EXPECT_EQ(state->pollRequests, 2);
  EXPECT_EQ(
      RPCRateLimiterRegistry::global().get("").stats().limitMilli, 10'000);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcErrorCount), 2);
  EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcErrorKindRateLimited), 2);
}

TEST_F(RPCOperatorTest, replayedReceiptFailuresDoNotBackOffAgain) {
  auto state = std::make_shared<ReceiptOperatorState>();
  state->submitError = velox::rpc::RPCErrorKind::kTimeout;
  const auto names = registerReceiptFunctions("historical_receipt", state);
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"input"}));
  core::PlanNodeId submitId;
  core::PlanNodeId pollId;
  core::PlanNodeId collectId;
  auto collect = PlanBuilder()
                     .values(input)
                     .rpc(
                         fmt::format("{}(prompt) as receipt", names[0]),
                         RPCStreamingMode::kBatch,
                         /*dispatchBatchSize=*/0,
                         "SUBMIT")
                     .capturePlanNodeId(submitId)
                     .rpc(
                         fmt::format("{}(receipt) as ready", names[1]),
                         RPCStreamingMode::kBatch,
                         /*dispatchBatchSize=*/0,
                         "POLL")
                     .capturePlanNodeId(pollId)
                     .rpc(
                         fmt::format("{}(ready) as value", names[2]),
                         RPCStreamingMode::kBatch,
                         /*dispatchBatchSize=*/0,
                         "COLLECT")
                     .capturePlanNodeId(collectId)
                     .planNode();
  std::shared_ptr<Task> task;
  auto result =
      AssertQueryBuilder(collect)
          .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, true)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 20)
          .config(core::QueryConfig::kRpcRateLimiterMinLimit, 1)
          .copyResults(pool(), task);
  ASSERT_EQ(result->size(), 1);
  EXPECT_TRUE(result->childAt(3)->isNullAt(0));
  EXPECT_EQ(state->pollRequests, 0);
  EXPECT_EQ(
      RPCRateLimiterRegistry::global().get("").stats().limitMilli, 10'000);
  for (const auto& id : {submitId, pollId, collectId}) {
    EXPECT_EQ(
        PlanNodeStats::extract(task->taskStats(), id)
            .customSum(RPCOperatorStatNames::kRpcErrorCount),
        1);
  }
}

TEST_F(RPCOperatorTest, reorderedDeferredCollectionPreservesRowValues) {
  auto state = std::make_shared<ReceiptOperatorState>();
  state->reverseCollection = true;
  const auto names = registerReceiptFunctions("reordered_receipt", state);
  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<StringView>({"first", "second"}));
  auto collect = PlanBuilder()
                     .values(input)
                     .rpc(
                         fmt::format("{}(prompt) as receipt", names[0]),
                         RPCStreamingMode::kBatch,
                         /*dispatchBatchSize=*/0,
                         "SUBMIT")
                     .rpc(
                         fmt::format("{}(receipt) as ready", names[1]),
                         RPCStreamingMode::kBatch,
                         /*dispatchBatchSize=*/0,
                         "POLL")
                     .rpc(
                         fmt::format("{}(ready) as value", names[2]),
                         RPCStreamingMode::kBatch,
                         /*dispatchBatchSize=*/0,
                         "COLLECT")
                     .project({"prompt", "value"})
                     .planNode();
  AssertQueryBuilder(collect).assertResults(makeRowVector({
      input->childAt(0),
      makeFlatVector<std::string>({"value-0", "value-1"}),
  }));
}

TEST_F(RPCOperatorTest, completionMetadataPreservesPayloadAndActualErrors) {
  auto response = RPCResponse::ok(0, makeTextPayload("failure receipt"));
  response.setCompletionOutcome(velox::rpc::RPCErrorKind::kTimeout, false);
  EXPECT_FALSE(response.hasError());
  EXPECT_EQ(responseAs<TextPayload>(response).text, "failure receipt");
  EXPECT_EQ(response.completionErrorKind(), velox::rpc::RPCErrorKind::kTimeout);
  EXPECT_FALSE(response.isBackendOutcome());
  response.setPayload(makeTextPayload("ready receipt"));
  EXPECT_EQ(response.completionErrorKind(), velox::rpc::RPCErrorKind::kNone);
  EXPECT_TRUE(response.isBackendOutcome());
  response.setError(velox::rpc::RPCErrorKind::kBackendError, "backend failure");
  EXPECT_EQ(
      response.completionErrorKind(), velox::rpc::RPCErrorKind::kBackendError);
  VELOX_ASSERT_THROW(
      response.setCompletionOutcome(velox::rpc::RPCErrorKind::kNone, false),
      "Completion metadata must preserve the response error kind");
  VELOX_ASSERT_THROW(
      response.setCompletionOutcome(velox::rpc::RPCErrorKind::kTimeout, true),
      "Completion metadata must preserve the response error kind");
  response.setCompletionOutcome(velox::rpc::RPCErrorKind::kBackendError, false);
  EXPECT_TRUE(response.hasError());
  EXPECT_EQ(response.error().message, "backend failure");
  EXPECT_FALSE(response.isBackendOutcome());
}

TEST_F(RPCOperatorTest, maxProbeIntervalDefaultsToOneMinute) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 1);

  const auto& limiter = RPCRateLimiterRegistry::global().get("capped-rpc");
  EXPECT_EQ(limiter.config().maxProbeInterval, std::chrono::seconds(60));
}

TEST_F(RPCOperatorTest, maxProbeIntervalReachesLimiter) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMaxProbeIntervalMs, 10'000)
          .countResults(),
      1);

  const auto& limiter = RPCRateLimiterRegistry::global().get("capped-rpc");
  EXPECT_EQ(limiter.config().maxProbeInterval, std::chrono::seconds(10));
}

TEST_F(RPCOperatorTest, legacySessionLimitPreservesOverrideSemantics) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"bounded"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 128)
          .countResults(),
      1);

  const auto limiterStats =
      RPCRateLimiterRegistry::global().get("capped-rpc").stats();
  EXPECT_EQ(limiterStats.hardLimit, 128);
}

TEST_F(RPCOperatorTest, negativeRateLimiterBoundsAreRejected) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"healthy"}));

  for (const auto& [property, message] :
       std::vector<std::pair<std::string, std::string>>{
           {core::QueryConfig::kRpcRateLimiterMinLimit,
            "rpc.ratelimiter.min_limit must be non-negative"},
           {core::QueryConfig::kRpcRateLimiterMaxLimit,
            "rpc.ratelimiter.max_limit must be non-negative"},
           {core::QueryConfig::kRpcRateLimiterHardLimit,
            "rpc.ratelimiter.hard_limit must be at least -1"},
           {core::QueryConfig::kRpcRateLimiterMaxProbeIntervalMs,
            "rpc.ratelimiter.max_probe_interval_ms must be non-negative"}}) {
    auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan)
            .config(
                property,
                property == core::QueryConfig::kRpcRateLimiterHardLimit ? "-2"
                                                                        : "-1")
            .countResults(),
        message);
  }
}

TEST_F(RPCOperatorTest, functionAndSessionHardLimitsUseSmallerValue) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"bounded"}));
  auto plan = PlanBuilder().values(input).rpc("capped_rpc(prompt)").planNode();
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 32)
          .config(core::QueryConfig::kRpcRateLimiterHardLimit, 32)
          .countResults(),
      1);

  const auto limiterStats =
      RPCRateLimiterRegistry::global().get("capped-rpc").stats();
  EXPECT_EQ(limiterStats.hardLimit, 32);
  EXPECT_EQ(limiterStats.capacity, 32);
}

// Reversed responses: the mock returns results in reverse order.
// Before the fix in RPCOperator::flushBatchRequests (scatter by rowId instead
// of positional stamping), this test would fail — each row would receive
// another row's result because the operator stamped rowIds positionally
// onto the reversed response vector.
TEST_F(RPCOperatorTest, batchReversedResponseOrder) {
  auto input = makeRowVector(
      {"id", "prompt"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<StringView>(
           {"alpha", "bravo", "charlie", "delta", "echo"})});

  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("demo_batch_rpc_reversed(prompt)", RPCStreamingMode::kBatch)
          .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      input->childAt(1),
      makeFlatVector<std::string>({
          "Batch response for: alpha",
          "Batch response for: bravo",
          "Batch response for: charlie",
          "Batch response for: delta",
          "Batch response for: echo",
      }),
  }));
}

// Partial batch failure: rows at indices 1 and 3 fail, others succeed.
// Verifies that failed rows produce NULL results while successful rows
// are correctly mapped to their prompts.
TEST_F(RPCOperatorTest, batchPartialFailure) {
  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>(
          {"row0", "row1_fail", "row2", "row3_fail", "row4"}));

  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("demo_batch_rpc_partial_fail(prompt)", RPCStreamingMode::kBatch)
          .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeNullableFlatVector<std::string>({
          "Batch response for: row0",
          std::nullopt,
          "Batch response for: row2",
          std::nullopt,
          "Batch response for: row4",
      }),
  }));
}

// Whole-batch failure (e.g. an operator-level batch/RPC timeout) should DEGRADE
// to per-row errored responses (-> NULL under the return-null policy), NOT
// hard-fail the entire query. This is the repro for the batch-timeout bug:
// today RPCOperator::getOutput VELOX_FAILs on claimedBatch_->error, bypassing
// the per-row error policy, so this test currently fails with "RPC batch
// failed: simulated batch timeout". After routing the operator's deferError
// through a per-row fan-out, all rows should come back NULL and the query
// should complete.
TEST_F(RPCOperatorTest, batchWholeBatchFailureDegradesToNull) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("demo_batch_rpc_whole_fail(prompt)", RPCStreamingMode::kBatch)
          .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeNullConstant(TypeKind::VARCHAR, 3),
  }));
}

// With a fail-on-error policy (meta_ai_on_error='fail'), a whole-batch failure
// must still HARD-FAIL the query — the degrade-to-per-row change must not
// silently turn a 'fail' request into all-NULL. The per-row errors produced by
// the operator flow to the function's buildOutput, which fails the query.
TEST_F(RPCOperatorTest, batchWholeBatchFailureWithFailPolicyThrows) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "demo_batch_rpc_whole_fail_strict(prompt)",
                      RPCStreamingMode::kBatch)
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(), "RPC call failed for row");
}

// A function that returns fewer responses than rows leaves the scatter no way
// to say which response belongs to which row. The framework fails the query
// before the function's row-error policy can run.
TEST_F(RPCOperatorTest, batchWrongResponseCountFailsQuery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  for (const auto& functionName :
       {"demo_batch_rpc_wrong_count", "demo_batch_rpc_wrong_count_strict"}) {
    SCOPED_TRACE(functionName);
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(
                        fmt::format("{}(prompt)", functionName),
                        RPCStreamingMode::kBatch)
                    .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).countResults(),
        "RPC batch response count (2) does not match row count (3)");
  }
}

TEST_F(RPCOperatorTest, batchDuplicateRowIdFailsQuery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  for (const auto& functionName :
       {"demo_batch_rpc_duplicate_row_id",
        "demo_batch_rpc_duplicate_row_id_strict"}) {
    SCOPED_TRACE(functionName);
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(
                        fmt::format("{}(prompt)", functionName),
                        RPCStreamingMode::kBatch)
                    .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).countResults(),
        "RPC duplicate batch response rowId (0)");
  }
}

TEST_F(RPCOperatorTest, batchOutOfRangeRowIdFailsQuery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  for (const auto& functionName :
       {"demo_batch_rpc_out_of_range_row_id",
        "demo_batch_rpc_out_of_range_row_id_strict"}) {
    SCOPED_TRACE(functionName);
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(
                        fmt::format("{}(prompt)", functionName),
                        RPCStreamingMode::kBatch)
                    .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).countResults(),
        "RPC batch response rowId (3) out of range [0, 3)");
  }
}

TEST_F(RPCOperatorTest, batchUnsetResponseFailsQuery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  for (const auto& functionName :
       {"demo_batch_rpc_unset_response",
        "demo_batch_rpc_unset_response_strict"}) {
    SCOPED_TRACE(functionName);
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(
                        fmt::format("{}(prompt)", functionName),
                        RPCStreamingMode::kBatch)
                    .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).countResults(),
        "RPC function 'demo_batch_rpc' returned an unset response for row 0");
  }
}

TEST_F(RPCOperatorTest, perRowUnsetResponseFailsQuery) {
  auto input = makeNamedRowVector("prompt", makeFlatVector<StringView>({"a"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("demo_rpc_unset_response(prompt)")
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(),
      "RPC function 'demo_rpc' returned an unset response for row 0");
}

TEST_F(RPCOperatorTest, perRowAsyncInternalFailureFailsQuery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b"}));

  auto plan =
      PlanBuilder().values(input).rpc("demo_rpc_fatal(prompt)").planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(), "simulated invariant failure");
}

TEST_F(RPCOperatorTest, batchAsyncInternalFailureFailsQuery) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "demo_batch_rpc_whole_fail_fatal(prompt)",
                      RPCStreamingMode::kBatch)
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).countResults(), "simulated invariant failure");
}

// Null inputs in batch mode produce null results.
TEST_F(RPCOperatorTest, batchNullInput) {
  auto input = makeNamedRowVector(
      "prompt",
      makeNullableFlatVector<StringView>(
          {"valid1"_sv, std::nullopt, "valid2"_sv, std::nullopt}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("demo_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeNullableFlatVector<std::string>({
          "Batch response for: valid1",
          std::nullopt,
          "Batch response for: valid2",
          std::nullopt,
      }),
  }));
}

// Multiple input batches: two separate addInput() calls, both processed
// correctly in batch mode.
TEST_F(RPCOperatorTest, batchMultipleInputBatches) {
  auto batch1 =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"a", "b", "c"}));
  auto batch2 =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"d", "e"}));

  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .rpc("demo_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      makeFlatVector<std::string>({
          "Batch response for: a",
          "Batch response for: b",
          "Batch response for: c",
          "Batch response for: d",
          "Batch response for: e",
      }),
  }));
}

// Pipelined batch dispatch: with dispatchBatchSize=2, the operator flushes
// mid-addInput() instead of waiting for noMoreInput(). Verifies all rows
// are still accounted for.
TEST_F(RPCOperatorTest, batchPipelinedDispatch) {
  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>({"p1", "p2", "p3", "p4", "p5", "p6", "p7"}));

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "demo_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/2)
                  .planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeFlatVector<std::string>(
          7,
          [](auto row) {
            return fmt::format("Batch response for: p{}", row + 1);
          }),
  }));
}

namespace {

// BATCH RPC function whose flushBatch completes after a fixed latency on a
// SEPARATE executor (mirroring a real transport). Lets a test create sustained
// mid-stream BATCH back-pressure with in-flight batches to wait on.
class SlowBatchRPCFunction : public AsyncRPCFunction {
 public:
  SlowBatchRPCFunction(
      std::chrono::milliseconds latency,
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor)
      : latency_(latency), executor_(std::move(executor)) {}

  std::string name() const override {
    return "slow_batch_rpc";
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(const SelectivityVector&, const std::vector<VectorPtr>&)
      override {
    VELOX_UNSUPPORTED("slow_batch_rpc is batch-only");
  }

  std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& /*args*/) override {
    std::vector<vector_size_t> indices;
    rows.applyToSelected([&](vector_size_t row) {
      indices.push_back(row);
      ++pending_;
    });
    return indices;
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    const int32_t n =
        maxRows > 0 ? std::min<int32_t>(maxRows, pending_) : pending_;
    pending_ -= n;
    std::vector<RPCResponse> responses;
    responses.reserve(n);
    for (int32_t i = 0; i < n; ++i) {
      RPCResponse response;
      response.rowId = i;
      response.setPayload(makeTextPayload("ok"));
      responses.push_back(std::move(response));
    }
    // Complete after `latency_` on the transport executor (NOT the driver
    // thread), so the operator has a genuinely in-flight batch to park on. Use
    // a futures-based delay rather than a blocking sleep so no executor thread
    // is parked for the latency.
    return folly::futures::sleep(latency_)
        .via(executor_.get())
        .thenValue([responses = std::move(responses)](auto&&) mutable {
          return std::move(responses);
        })
        .semi();
  }

  int32_t pendingBatchSize() const override {
    return pending_;
  }

 private:
  const std::chrono::milliseconds latency_;
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
  int32_t pending_{0};
  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    return buildTextOutput(responses, pool);
  }
};

class ExecutionModeRecordingRPCFunction : public SlowBatchRPCFunction {
 public:
  using SlowBatchRPCFunction::SlowBatchRPCFunction;

  std::string name() const override {
    return "execution_mode_recording_rpc";
  }

  void initialize(
      const core::QueryConfig&,
      const std::vector<TypePtr>&,
      const std::vector<VectorPtr>&,
      RPCStreamingMode instruction) override {
    receivedExecutionMode_ = instruction;
  }

  RPCStreamingMode receivedExecutionMode() const {
    return receivedExecutionMode_;
  }

 private:
  RPCStreamingMode receivedExecutionMode_{RPCStreamingMode::kPerRow};
};

// Holds one batch open while the test advances the transport's retry counter.
class RetryingBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  static constexpr int32_t kNumRetries{3};

  std::string name() const override {
    return "retrying_batch_rpc";
  }

  int64_t numRetriesAttempted() const override {
    return numRetriesAttempted_.load(std::memory_order_relaxed);
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    auto responses = std::move(DemoBatchRPCFunction::flushBatch(maxRows)).get();
    return batchCompletion_.getSemiFuture().deferValue(
        [responses = std::move(responses)](folly::Unit) mutable {
          return std::move(responses);
        });
  }

  void testingAttemptRetry() {
    numRetriesAttempted_.fetch_add(1, std::memory_order_relaxed);
  }

  void testingCompleteBatch() {
    batchCompletion_.setValue();
  }

 private:
  std::atomic<int64_t> numRetriesAttempted_{0};
  folly::Promise<folly::Unit> batchCompletion_;
};

} // namespace

// Regression proof for the BATCH mid-stream back-pressure yield.
//
// Before the fix, RPCOperator::isBlocked returned kNotBlocked while a batch was
// in flight under mid-stream back-pressure, so the driver busy-spun on its
// thread (never reporting itself blocked) until the batch completed. After the
// fix it parks (kWaitForRPC) on an in-flight batch, so the driver's
// blockedWallNanos reflects the batch waits.
//
// The plan feeds many single-row Values vectors (input keeps arriving, so
// noMoreInput_ stays false) into a BATCH RPCNode with dispatchBatchSize=1 and a
// slow (200ms) batch. With the BATCH window of 2 this produces repeated
// mid-stream back-pressure. We assert the RPC operator's blockedWallNanos is
// several multiples of the batch latency (parked). On the pre-fix code it is at
// most ~one latency (the end-of-input wait only), which fails this assertion.
TEST_F(RPCOperatorTest, batchMidStreamBackpressureParksNotSpins) {
  constexpr std::chrono::milliseconds kLatency{200};
  auto rpcExecutor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  registerFunction<SlowBatchRPCFunction>(
      "slow_batch_rpc", kLatency, rpcExecutor);

  constexpr int kRows = 8;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    inputs.push_back(
        makeNamedRowVector("prompt", makeFlatVector<std::string>({"hi"})));
  }

  auto plan = PlanBuilder()
                  .values(inputs)
                  .rpc(
                      "slow_batch_rpc(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/1)
                  .planNode();

  std::shared_ptr<exec::Task> task;
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(task), kRows);

  uint64_t blockedNs = 0;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (op.operatorType == "RPC") {
        blockedNs += op.blockedWallNanos;
      }
    }
  }

  const uint64_t latencyNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(kLatency).count();
  EXPECT_GT(blockedNs, 2 * latencyNs)
      << "RPC operator did not park under mid-stream BATCH back-pressure: "
      << "blockedWallNanos=" << blockedNs << " latencyNs=" << latencyNs;
}

// RPC progress must be observable from Task::taskStats() while the driver is
// blocked, not only after the operator closes.
//
// While a driver is blocked on kWaitForRPC no operator counter advances:
// inputVectors and outputVectors are frozen and the memory reservation is
// already taken. Previously RPCOperator published its counters from close()
// only, so a caller sampling Task::taskStats() saw identical stats for the
// whole backend round trip. stats() now refreshes them on every sample.
//
// The plan runs a slow BATCH RPC and drives the task through a non-blocking
// TaskCursor, sampling Task::taskStats() while the cursor reports itself
// blocked. Without the stats() override rpcInFlight is absent from
// runtimeStats for every sample and both assertions below fail.
TEST_F(RPCOperatorTest, rpcLivenessStatsVisibleWhileDriverIsParked) {
  constexpr std::chrono::milliseconds kLatency{500};
  auto rpcExecutor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  registerFunction<SlowBatchRPCFunction>(
      "slow_batch_rpc_liveness", kLatency, rpcExecutor);

  constexpr int kRows = 4;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    inputs.push_back(
        makeNamedRowVector("prompt", makeFlatVector<std::string>({"hi"})));
  }

  CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(inputs)
                        .rpc(
                            "slow_batch_rpc_liveness(prompt)",
                            RPCStreamingMode::kBatch,
                            /*dispatchBatchSize=*/1)
                        .planNode();
  params.maxDrivers = 1;

  auto cursor = TaskCursor::create(params);
  const auto task = cursor->task();

  // Highest rpcInFlight and rpcCompletionsSignaled seen mid-run, i.e. strictly
  // before the operator closes and publishes its final stats.
  int64_t peakInFlightWhileRunning{0};
  int64_t completionsWhileRunning{0};
  const auto sampleLiveStats = [&]() {
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType != "RPC") {
          continue;
        }
        const auto inFlight =
            op.runtimeStats.find(RPCOperatorStatNames::kRpcInFlight);
        if (inFlight != op.runtimeStats.end()) {
          peakInFlightWhileRunning =
              std::max(peakInFlightWhileRunning, inFlight->second.sum);
        }
        const auto completions =
            op.runtimeStats.find(RPCOperatorStatNames::kRpcCompletionsSignaled);
        if (completions != op.runtimeStats.end()) {
          completionsWhileRunning =
              std::max(completionsWhileRunning, completions->second.sum);
        }
      }
    }
  };

  int32_t numOutputRows{0};
  while (true) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    if (cursor->moveNext(&future)) {
      numOutputRows += cursor->current()->size();
      continue;
    }
    if (!future.valid()) {
      break;
    }
    // The cursor is parked on the in-flight batch. Sample repeatedly so the
    // assertion does not depend on catching a single instant.
    while (!future.isReady()) {
      sampleLiveStats();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::move(future).wait();
  }

  EXPECT_EQ(numOutputRows, kRows);
  EXPECT_GT(peakInFlightWhileRunning, 0)
      << "rpcInFlight was never observed while the driver was blocked";
  EXPECT_GT(completionsWhileRunning, 0)
      << "rpcCompletionsSignaled did not advance mid-run";

  // The in-flight gauge must not survive into the finished task's stats. Task
  // accumulates a closed driver's stats permanently, so a leftover positive
  // reading would report RPCs still in flight long after they completed. The
  // monotonic counter, by contrast, is meaningful after the fact and is still
  // published at close().
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (op.operatorType != "RPC") {
        continue;
      }
      EXPECT_EQ(op.runtimeStats.count(RPCOperatorStatNames::kRpcInFlight), 0)
          << "in-flight gauge was frozen into the finished task's stats";
      EXPECT_GT(
          op.runtimeStats.at(RPCOperatorStatNames::kRpcCompletionsSignaled).sum,
          0);
    }
  }
}

// A transport retrying a request keeps the row open without resolving it, so
// rpcRequestsDispatched and rpcCompletionsSignaled both stand still for as long
// as the ladder runs. rpcRetriesAttempted is what separates that from a backend
// that has stopped answering.
//
// The serial cursor parks on one held batch. Each retry advances while its
// wait future remains unready, with dispatches and completions standing still.
TEST_F(RPCOperatorTest, rpcRetriesAttemptedAdvancesWhileDriverIsParked) {
  auto function = std::make_shared<RetryingBatchRPCFunction>();
  AsyncRPCFunctionRegistry::registerFunction(
      "retrying_batch_rpc_liveness",
      [function]() { return function; },
      DemoBatchRPCFunction::signatures());

  constexpr int kRows = 2;
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"hi", "hello"}));

  CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(input)
                        .rpc(
                            "retrying_batch_rpc_liveness(prompt)",
                            RPCStreamingMode::kBatch,
                            /*dispatchBatchSize=*/kRows)
                        .planNode();
  params.maxDrivers = 1;
  params.serialExecution = true;

  auto cursor = TaskCursor::create(params);
  const auto task = cursor->task();

  struct Sample {
    int64_t retries{0};
    int64_t dispatched{0};
    int64_t completions{0};
    int64_t inFlight{0};
  };

  const auto readStat = [](const auto& op, const std::string& name) {
    const auto it = op.runtimeStats.find(name);
    return it == op.runtimeStats.end() ? 0 : it->second.sum;
  };
  const auto sampleStats = [&]() -> std::optional<Sample> {
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType != "RPC") {
          continue;
        }
        return Sample{
            .retries = readStat(op, RPCOperatorStatNames::kRpcRetriesAttempted),
            .dispatched =
                readStat(op, RPCOperatorStatNames::kRpcRequestsDispatched),
            .completions =
                readStat(op, RPCOperatorStatNames::kRpcCompletionsSignaled),
            .inFlight = readStat(op, RPCOperatorStatNames::kRpcInFlight),
        };
      }
    }
    return std::nullopt;
  };

  ContinueFuture future = ContinueFuture::makeEmpty();
  ASSERT_FALSE(cursor->moveNext(&future));
  ASSERT_TRUE(future.valid());
  ASSERT_FALSE(future.isReady());

  auto previous = sampleStats();
  ASSERT_TRUE(previous.has_value());
  EXPECT_EQ(previous->retries, 0);
  EXPECT_EQ(previous->dispatched, kRows);
  EXPECT_EQ(previous->completions, 0);
  EXPECT_EQ(previous->inFlight, 1);

  for (int32_t retry = 1; retry <= RetryingBatchRPCFunction::kNumRetries;
       ++retry) {
    function->testingAttemptRetry();
    ASSERT_FALSE(future.isReady());
    const auto sample = sampleStats();
    ASSERT_TRUE(sample.has_value());
    EXPECT_EQ(sample->retries, retry);
    EXPECT_EQ(sample->dispatched, previous->dispatched);
    EXPECT_EQ(sample->completions, previous->completions);
    EXPECT_EQ(sample->inFlight, 1);
    previous = sample;
  }

  function->testingCompleteBatch();
  std::move(future).wait();
  EXPECT_EQ(cursor->countRows(), kRows);

  const auto finalStats = sampleStats();
  ASSERT_TRUE(finalStats.has_value());
  EXPECT_EQ(finalStats->retries, RetryingBatchRPCFunction::kNumRetries);
  EXPECT_EQ(finalStats->dispatched, kRows);
  EXPECT_EQ(finalStats->completions, 1);
}

TEST_F(RPCOperatorTest, demoBatchErrorSignalsOverload) {
  DemoBatchRPCFunction function;
  std::vector<RPCResponse> responses;
  responses.push_back(
      RPCResponse::failed(
          0, velox::rpc::RPCErrorKind::kRateLimited, "simulated_overload"));

  EXPECT_EQ(
      function.evaluateCongestion(responses),
      AsyncRPCFunction::CongestionSignal::kOverloaded);
}

TEST_F(RPCOperatorTest, demoBatchBackendErrorIsNotOverload) {
  DemoBatchRPCFunction function;
  std::vector<RPCResponse> responses;
  responses.push_back(
      RPCResponse::failed(
          0, velox::rpc::RPCErrorKind::kBackendError, "simulated_failure"));

  EXPECT_EQ(
      function.evaluateCongestion(responses),
      AsyncRPCFunction::CongestionSignal::kNonOverloadError);
}

// Per-flush row cap: the operator asks the function how many rows it will take
// and chunks a backlog accordingly, so no single request exceeds what the
// backend accepts while every row is still emitted. Uses the flush-all path
// (dispatchBatchSize=0), which noMoreInput() must also chunk.
TEST_F(RPCOperatorTest, batchClampedByByteBudget) {
  ByteBudgetedBatchRPCFunction::flushChunks().clear();

  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>(
          {"r0", "r1", "r2", "r3", "r4", "r5", "r6"})); // 7 rows

  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(
                      "demo_batch_rpc_byte_budgeted(prompt)",
                      RPCStreamingMode::kBatch,
                      /*dispatchBatchSize=*/0)
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  // No rows lost.
  ASSERT_EQ(result->size(), 7);
  const auto& results = result->childAt(1);
  EXPECT_EQ(BaseVector::countNulls(results->nulls(), results->size()), 0);

  const std::vector<int32_t> expectedFlushChunks{3, 3, 1};
  EXPECT_EQ(ByteBudgetedBatchRPCFunction::flushChunks(), expectedFlushChunks);
}

TEST_F(RPCOperatorTest, rejectsInvalidMaxRowsPerFlush) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"r0", "r1"}));

  for (const auto& functionName : {
           "demo_batch_rpc_zero_flush_limit",
           "demo_batch_rpc_excessive_flush_limit",
       }) {
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(
                        fmt::format("{}(prompt)", functionName),
                        RPCStreamingMode::kBatch,
                        /*dispatchBatchSize=*/2)
                    .planNode();

    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).countResults(), "expected a value in [1, 2]");
  }
}

/// PER_ROW congestion path. On the function's overload verdict
/// (evaluateCongestion -> kOverloaded) both AIMD controllers back off: the
/// per-driver window (onUnitError) and the process-global rate limiter
/// (onRateLimited); on kSuccess the window's latency gradient is fed. Verifies
/// the query still completes correctly through that path. The controllers'
/// adjustments are unit-tested in RPCStateTest / RPCRateLimiterTest; here we
/// guard the operator-level materialization + signal plumbing against
/// crashes/regressions.
TEST_F(RPCOperatorTest, perRowCongestionPath) {
  // DemoAsyncRPCFunction::evaluateCongestion returns kOverloaded when a
  // response result contains "OVERLOAD" (the mock echoes the prompt into the
  // result).
  auto input = makeNamedRowVector(
      "prompt",
      makeFlatVector<StringView>(
          {"OVERLOAD one", "OVERLOAD two", "normal three"}));

  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();

  AssertQueryBuilder(plan).assertResults(makeRowVector({
      input->childAt(0),
      makeFlatVector<std::string>(
          {"demo: OVERLOAD one", "demo: OVERLOAD two", "demo: normal three"}),
  }));
}

TEST_F(RPCOperatorTest, batchExecutionModeReachesTheFunction) {
  auto rpcExecutor = std::make_shared<folly::CPUThreadPoolExecutor>(4);
  std::shared_ptr<ExecutionModeRecordingRPCFunction> function;
  AsyncRPCFunctionRegistry::registerFunction(
      "execution_mode_recording_rpc",
      [&function, rpcExecutor]() {
        function = std::make_shared<ExecutionModeRecordingRPCFunction>(
            std::chrono::milliseconds{0}, rpcExecutor);
        return function;
      },
      DemoBatchRPCFunction::signatures());

  auto input =
      makeNamedRowVector("prompt", makeFlatVector<std::string>({"hi"}));
  auto plan =
      PlanBuilder()
          .values(input)
          .rpc("execution_mode_recording_rpc(prompt)", RPCStreamingMode::kBatch)
          .planNode();

  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 1);

  ASSERT_NE(function, nullptr);
  EXPECT_EQ(function->receivedExecutionMode(), RPCStreamingMode::kBatch);
}

class HeldPacedCeilingRPCFunction : public HeldPacedRPCFunction {
 public:
  HeldPacedCeilingRPCFunction(std::string key, int64_t ceiling)
      : HeldPacedRPCFunction(std::move(key)), ceiling_(ceiling) {}

  int64_t configuredCeiling() const override {
    return ceiling_;
  }

 private:
  const int64_t ceiling_;
};

// A PER_ROW driver window starts at 100. Under CAPACITY_PROBING it rises to
// the limiter's ceiling, here 300 from the session, but not past a ceiling the
// function declares; it never falls below 100. A driver window ceiling the
// session sets holds under either policy. A request with adaptation off is not
// eligible for CAPACITY_PROBING, so its window stays at 100.
TEST_F(RPCOperatorTest, capacityProbingDriverWindowRisesToLimiterCeiling) {
  constexpr int32_t kRows = 150;
  struct Case {
    const char* policyName;
    int64_t functionCeiling;
    int64_t sessionMaxWindow;
    bool adaptive;
    int64_t expectedInFlight;
  };
  for (const auto& testCase : std::vector<Case>{
           {"WINDOW_AIMD", 0, 0, true, 100},
           {"CAPACITY_PROBING", 0, 0, true, kRows},
           {"CAPACITY_PROBING", 120, 0, true, 120},
           {"CAPACITY_PROBING", 64, 0, true, 100},
           {"CAPACITY_PROBING", 0, 120, true, 120},
           {"CAPACITY_PROBING", 0, 0, false, 100},
       }) {
    const auto name = fmt::format(
        "driver_window_ceiling_{}_{}_{}_{}",
        testCase.policyName,
        testCase.functionCeiling,
        testCase.sessionMaxWindow,
        testCase.adaptive);
    SCOPED_TRACE(name);
    auto function = std::make_shared<HeldPacedCeilingRPCFunction>(
        name, testCase.functionCeiling);
    registerInstance(function);
    std::vector<std::string> prompts;
    prompts.reserve(kRows);
    for (int32_t i = 0; i < kRows; ++i) {
      prompts.push_back(fmt::format("prompt{}", i));
    }
    CursorParameters params;
    params.planNode = PlanBuilder()
                          .values(makeNamedRowVector(
                              "prompt", makeFlatVector<std::string>(prompts)))
                          .rpc(fmt::format("{}(prompt)", function->name()))
                          .planNode();
    params.maxDrivers = 1;
    params.queryConfigs = {
        {core::QueryConfig::kRpcFlowControlPolicy, testCase.policyName},
        {core::QueryConfig::kRpcRateLimiterAdaptiveEnabled,
         testCase.adaptive ? "true" : "false"},
        {core::QueryConfig::kRpcRateLimiterMaxLimit, "300"},
        {core::QueryConfig::kRpcRateLimiterMinLimit, "1"},
    };
    if (testCase.sessionMaxWindow > 0) {
      params.queryConfigs.emplace(
          core::QueryConfig::kRpcCongestionMaxWindow,
          std::to_string(testCase.sessionMaxWindow));
    }
    auto cursor = TaskCursor::create(params);
    cursor->start();

    ASSERT_TRUE(function->waitForDispatches(testCase.expectedInFlight));
    function->releaseResponses();
    EXPECT_EQ(cursor->countRows(), kRows);
    ASSERT_TRUE(
        cursor->task()->allDriversClosedFuture().wait(std::chrono::seconds{5}));
    EXPECT_EQ(
        PlanNodeStats::extract(
            cursor->task()->taskStats(), params.planNode->id())
            .customStat(RPCOperatorStatNames::kRpcRateLimiterPeakPending)
            .max,
        testCase.expectedInFlight);
  }
}

// With rpc.ratelimiter.hard_limit=0 and no function ceiling, only the
// 1,000,000 safety ceiling bounds the limiter, so a CAPACITY_PROBING driver
// window rises only to where an adaptive limiter starts, 200. A WINDOW_AIMD
// driver window stays at its default of 100. Responses are held until the
// first window's worth has been dispatched, so no latency sample can move the
// window before the peak is reached.
TEST_F(RPCOperatorTest, driverWindowUnderSafetyCeilingFollowsPolicy) {
  constexpr int32_t kRows = 1'000;
  std::vector<std::string> prompts;
  prompts.reserve(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    prompts.push_back(fmt::format("prompt{}", i));
  }
  for (const auto& [policy, expectedPeak] :
       std::vector<std::pair<std::string, int64_t>>{
           {"WINDOW_AIMD", 100}, {"CAPACITY_PROBING", 200}}) {
    SCOPED_TRACE(policy);
    auto function = std::make_shared<HeldPacedCeilingRPCFunction>(
        fmt::format("driver_window_safety_ceiling_{}", expectedPeak), 0);
    registerInstance(function);
    CursorParameters params;
    params.planNode = PlanBuilder()
                          .values(makeNamedRowVector(
                              "prompt", makeFlatVector<std::string>(prompts)))
                          .rpc(fmt::format("{}(prompt)", function->name()))
                          .planNode();
    params.maxDrivers = 1;
    params.queryConfigs = {
        {core::QueryConfig::kRpcFlowControlPolicy, policy},
        {core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, "true"},
        {core::QueryConfig::kRpcRateLimiterHardLimit, "0"},
        {core::QueryConfig::kRpcRateLimiterMinLimit, "1"},
    };
    auto cursor = TaskCursor::create(params);
    cursor->start();
    // Reads output as it comes, so output backpressure cannot stall dispatch.
    uint64_t outputRows = 0;
    std::thread reader([&] { outputRows = cursor->countRows(); });
    const bool firstWindow = function->waitForDispatches(expectedPeak);
    function->releaseResponses();
    reader.join();

    ASSERT_TRUE(firstWindow);
    EXPECT_EQ(outputRows, kRows);
    ASSERT_TRUE(
        cursor->task()->allDriversClosedFuture().wait(std::chrono::seconds{5}));
    EXPECT_EQ(
        PlanNodeStats::extract(
            cursor->task()->taskStats(), params.planNode->id())
            .customStat(RPCOperatorStatNames::kRpcRateLimiterPeakPending)
            .max,
        expectedPeak);
  }
}

TEST_F(RPCOperatorTest, nonAdaptiveCapacityProbingRequestSharesFixedLimiter) {
  const std::string key = "fixed_shared_policy_gauges";
  auto function = std::make_shared<HeldPacedRPCFunction>(key);
  function->releaseResponses();
  registerInstance(function);
  const RPCRateLimiter::Config base{.adaptive = false, .ceiling = 2};
  auto& registry = RPCRateLimiterRegistry::global();
  auto* original = registry.getForPolicy(key, base).limiter;
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<std::string>({"backend"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc(fmt::format("{}(prompt)", key))
                  .planNode();
  for (const auto requested :
       {RPCFlowControlPolicy::kWindowAimd,
        RPCFlowControlPolicy::kCapacityProbing}) {
    SCOPED_TRACE(RPCFlowControlPolicyName::toName(requested));
    const auto task =
        AssertQueryBuilder(plan)
            .config(
                core::QueryConfig::kRpcFlowControlPolicy,
                std::string{RPCFlowControlPolicyName::toName(requested)})
            .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, false)
            .config(core::QueryConfig::kRpcRateLimiterMaxLimit, 2)
            .assertResults(makeRowVector({
                input->childAt(0),
                makeFlatVector<std::string>({"demo: backend"}),
            }));
    const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
    EXPECT_EQ(
        stats.customSum(RPCOperatorStatNames::kRpcFlowControlRequestedPolicy),
        static_cast<int64_t>(requested));
    EXPECT_EQ(
        stats.customSum(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy),
        0);
    auto policy = base;
    policy.flowControlPolicy = requested;
    EXPECT_EQ(registry.getForPolicy(key, policy).limiter, original);
  }
}

TEST_F(RPCOperatorTest, retryCapabilityAloneDoesNotSelectCapacityProbing) {
  RPCRateLimiter::Config policy{
      .adaptive = true,
      .ceiling = 100,
      .floor = 1,
      .decreaseFactor = 0.5,
      .flowControlPolicy = RPCFlowControlPolicy::kCapacityProbing,
  };
  auto function = std::make_shared<ConservativeRetryRPCFunction>();
  registerInstance(function);
  auto* active = RPCRateLimiterRegistry::global()
                     .getForPolicy(function->admissionKey(), policy)
                     .limiter;
  auto tokens = active->tryAcquireUpTo(1);
  ASSERT_EQ(tokens.size(), 1);
  active->makeLease(std::move(tokens.front()))
      ->completeAndRelease(/*hardOverload=*/true, /*capacityRejection=*/false);
  ASSERT_EQ(active->stats().limitMilli, 50'000);

  auto input = makeNamedRowVector(
      "prompt", makeFlatVector<std::string>({"one", "two", "three", "four"}));
  CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(input)
                        .rpc(fmt::format("{}(prompt)", function->name()))
                        .planNode();
  params.maxDrivers = 1;
  params.queryConfigs = {
      {core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, "true"},
      {core::QueryConfig::kRpcRateLimiterMaxLimit, "100"},
      {core::QueryConfig::kRpcRateLimiterMinLimit, "1"},
  };
  auto cursor = TaskCursor::create(params);
  cursor->start();
  EXPECT_TRUE(function->waitForDispatches(4));
  function->releaseResponses();
  EXPECT_EQ(cursor->countRows(), 4);
  ASSERT_TRUE(
      cursor->task()->allDriversClosedFuture().wait(std::chrono::seconds{5}));
  const auto stats = PlanNodeStats::extract(
      cursor->task()->taskStats(), params.planNode->id());
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcRateLimiterPolicyScoped), 1);
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcFlowControlRequestedPolicy), 0);
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy), 0);
}

TEST_F(RPCOperatorTest, flowControlPolicySelectionIsExplicitAndReported) {
  for (const auto& [policy, code] :
       std::vector<std::pair<std::string, int64_t>>{
           {"WINDOW_AIMD", 0}, {"cApAcItY_pRoBiNg", 1}}) {
    SCOPED_TRACE(policy);
    auto function = std::make_shared<HeldPacedRPCFunction>();
    function->releaseResponses();
    const auto name = fmt::format("reported_policy_{}", code);
    AsyncRPCFunctionRegistry::registerFunction(
        name,
        [function]() { return function; },
        DemoAsyncRPCFunction::signatures());
    auto input = makeNamedRowVector(
        "prompt", makeFlatVector<StringView>({"one", "two"}));
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(fmt::format("{}(prompt)", name))
                    .planNode();
    std::shared_ptr<Task> task;
    ASSERT_EQ(
        AssertQueryBuilder(plan)
            .config(core::QueryConfig::kRpcFlowControlPolicy, policy)
            .countResults(task),
        2);
    const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
    EXPECT_EQ(
        stats.customSum(RPCOperatorStatNames::kRpcFlowControlRequestedPolicy),
        code);
    EXPECT_EQ(
        stats.customSum(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy),
        code);
    EXPECT_TRUE(
        !stats.hasCustomStat(RPCOperatorStatNames::kRpcCapacityRetries) ||
        stats.customSum(RPCOperatorStatNames::kRpcCapacityRetries) == 0);
  }
}

TEST_F(RPCOperatorTest, batchCapacityProbingRequestRunsWindowPolicy) {
  registerFunction<DemoBatchRPCFunction>("policy_batch_rpc");
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"one", "two"}));
  auto plan = PlanBuilder()
                  .values(input)
                  .rpc("policy_batch_rpc(prompt)", RPCStreamingMode::kBatch)
                  .planNode();
  // A default query seals the backend's shared limiter first.
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 2);
  std::shared_ptr<Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcFlowControlPolicy, "CAPACITY_PROBING")
          .countResults(task),
      2);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcFlowControlRequestedPolicy), 1);
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy), 0);
  // An ineligible request runs on the default query's limiter, not its own.
  expectRanOnSharedLimiter(task);
}

TEST_F(RPCOperatorTest, perRowCapacityProbingRequestRunsWindowPolicy) {
  // A PER_ROW function that never returns capacity retries is not eligible.
  class NoCapacityRetriesRPCFunction : public HeldPacedRPCFunction {
   public:
    using HeldPacedRPCFunction::HeldPacedRPCFunction;

    bool mayReturnCapacityRetries() const override {
      return false;
    }
  };
  auto function =
      std::make_shared<NoCapacityRetriesRPCFunction>("policy_per_row_rpc");
  function->releaseResponses();
  registerInstance(function);
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"one", "two"}));
  auto plan =
      PlanBuilder().values(input).rpc("policy_per_row_rpc(prompt)").planNode();
  // A default query seals the backend's shared limiter first.
  ASSERT_EQ(AssertQueryBuilder(plan).countResults(), 2);
  std::shared_ptr<Task> task;
  ASSERT_EQ(
      AssertQueryBuilder(plan)
          .config(core::QueryConfig::kRpcFlowControlPolicy, "CAPACITY_PROBING")
          .countResults(task),
      2);
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcFlowControlRequestedPolicy), 1);
  EXPECT_EQ(
      stats.customSum(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy), 0);
  expectRanOnSharedLimiter(task);
}

TEST_F(
    RPCOperatorTest,
    probingUnsupportedModesReportRequestedAndEffectivePolicy) {
  // Requests CAPACITY_PROBING for 'functionName' and verifies that the query
  // reports the request and runs under WINDOW_AIMD.
  auto testFallback = [&](const std::string& functionName,
                          RPCStreamingMode streamingMode,
                          bool adaptive) {
    SCOPED_TRACE(functionName);
    auto input = makeNamedRowVector(
        "prompt", makeFlatVector<std::string>({"one", "two", "three"}));
    auto plan = PlanBuilder()
                    .values(input)
                    .rpc(fmt::format("{}(prompt)", functionName), streamingMode)
                    .planNode();
    std::shared_ptr<Task> task;
    ASSERT_EQ(
        AssertQueryBuilder(plan)
            .config(
                core::QueryConfig::kRpcFlowControlPolicy, "CAPACITY_PROBING")
            .config(core::QueryConfig::kRpcRateLimiterAdaptiveEnabled, adaptive)
            .countResults(task),
        3);
    const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
    EXPECT_EQ(
        stats.customSum(RPCOperatorStatNames::kRpcFlowControlRequestedPolicy),
        1);
    EXPECT_EQ(
        stats.customSum(RPCOperatorStatNames::kRpcFlowControlEffectivePolicy),
        0);
    EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcRequestsDispatched), 3);
    EXPECT_EQ(stats.customSum(RPCOperatorStatNames::kRpcResponsesReceived), 3);
  };

  // Batch mode.
  registerFunction<DemoBatchRPCFunction>("probing_fallback_batch");
  testFallback(
      "probing_fallback_batch", RPCStreamingMode::kBatch, /*adaptive=*/true);

  // A per-row function that never returns capacity retries.
  registerFunction<DemoAsyncRPCFunction>("probing_fallback_per_row");
  testFallback(
      "probing_fallback_per_row", RPCStreamingMode::kPerRow, /*adaptive=*/true);

  // A fixed limiter.
  auto function =
      std::make_shared<HeldPacedRPCFunction>("probing_fallback_fixed_limiter");
  function->releaseResponses();
  registerInstance(function);
  testFallback(function->name(), RPCStreamingMode::kPerRow, /*adaptive=*/false);
  // Only this function has its own admission key, so its limiter is fresh and
  // stores exactly the WINDOW_AIMD policy.
  EXPECT_EQ(
      RPCRateLimiterRegistry::global()
          .get(function->name())
          .config()
          .flowControlPolicy,
      RPCFlowControlPolicy::kWindowAimd);
}

TEST_F(RPCOperatorTest, invalidFlowControlPolicyFailsBeforeDispatch) {
  auto input =
      makeNamedRowVector("prompt", makeFlatVector<StringView>({"one"}));
  auto plan = PlanBuilder().values(input).rpc("demo_rpc(prompt)").planNode();
  for (const auto* unsupported : {"UNKNOWN", "1", " CAPACITY_PROBING"}) {
    SCOPED_TRACE(unsupported);
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan)
            .config(core::QueryConfig::kRpcFlowControlPolicy, unsupported)
            .countResults(),
        "RPC flow control policy must be WINDOW_AIMD or CAPACITY_PROBING");
  }
}

} // namespace facebook::velox::exec::rpc
