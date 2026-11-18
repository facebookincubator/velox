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

#include "velox/exec/rpc/DeferredRpcPhaseFunction.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec::rpc {
namespace {

// Uses opaque string identities and a non-text collected payload.
struct NumberPayload {
  int64_t value;
};

class StringReceipt final : public JobReceipt {
 public:
  StringReceipt(
      std::string job,
      std::string input,
      Outcome state,
      velox::rpc::RPCErrorKind kind = velox::rpc::RPCErrorKind::kNone,
      bool backendOutcome = true)
      : job(std::move(job)),
        input(std::move(input)),
        state(state),
        kind(kind),
        backendOutcome(backendOutcome) {}
  Outcome outcome() const override {
    return state;
  }
  velox::rpc::RPCErrorKind errorKind() const override {
    return kind;
  }
  bool reportsBackendOutcome() const override {
    return backendOutcome;
  }
  std::string_view error() const override {
    return "fake failure";
  }
  std::optional<int64_t> enqueueTime() const override {
    return 10;
  }
  std::optional<int64_t> absoluteDeadline() const override {
    return 100;
  }
  int64_t retainedBytes() const noexcept override {
    return static_cast<int64_t>(sizeof(*this)) +
        static_cast<int64_t>(job.capacity()) +
        static_cast<int64_t>(input.capacity());
  }
  // Keeps actual backend identities distinct from result positions.
  const std::string job;
  const std::string input;
  const Outcome state;
  const velox::rpc::RPCErrorKind kind;
  const bool backendOutcome;
};

class StringReceiptCodec final : public JobReceiptCodec {
 public:
  RowTypePtr receiptType() const override {
    return ROW(
        {{"job", VARCHAR()},
         {"input", VARCHAR()},
         {"state", BIGINT()},
         {"kind", BIGINT()}});
  }
  std::string signature() const override {
    return "row(job varchar,input varchar,state bigint,kind bigint)";
  }
  JobReceiptPtr decode(const VectorPtr& rows, vector_size_t row)
      const override {
    VELOX_CHECK(rows->type()->equivalent(*receiptType()));
    DecodedVector decoded(*rows);
    VELOX_CHECK(!decoded.isNullAt(row));
    const auto* values = decoded.base()->as<RowVector>();
    const auto index = decoded.index(row);
    DecodedVector job(*values->childAt(0));
    DecodedVector input(*values->childAt(1));
    DecodedVector state(*values->childAt(2));
    DecodedVector kind(*values->childAt(3));
    return std::make_shared<StringReceipt>(
        job.valueAt<StringView>(index).str(),
        input.valueAt<StringView>(index).str(),
        static_cast<JobReceipt::Outcome>(state.valueAt<int64_t>(index)),
        static_cast<velox::rpc::RPCErrorKind>(kind.valueAt<int64_t>(index)),
        false);
  }
  VectorPtr encode(const ReceiptBatch& receipts, memory::MemoryPool* pool)
      const override {
    VELOX_CHECK_LE(
        receipts.size(),
        static_cast<size_t>(std::numeric_limits<vector_size_t>::max()));
    const auto numRows = static_cast<vector_size_t>(receipts.size());
    auto result = BaseVector::create<RowVector>(receiptType(), numRows, pool);
    for (vector_size_t row = 0; row < numRows; ++row) {
      const auto* receipt =
          dynamic_cast<const StringReceipt*>(receipts[row].get());
      VELOX_CHECK_NOT_NULL(receipt);
      result->childAt(0)->asFlatVector<StringView>()->set(
          row, StringView(receipt->job));
      result->childAt(1)->asFlatVector<StringView>()->set(
          row, StringView(receipt->input));
      result->childAt(2)->asFlatVector<int64_t>()->set(
          row, static_cast<int64_t>(receipt->state));
      result->childAt(3)->asFlatVector<int64_t>()->set(
          row, static_cast<int64_t>(receipt->kind));
    }
    return result;
  }
  JobReceiptPtr fromError(velox::rpc::RPCErrorKind kind, std::string_view)
      const override {
    return std::make_shared<StringReceipt>(
        "",
        "",
        kind == velox::rpc::RPCErrorKind::kNullInput
            ? JobReceipt::Outcome::kNull
            : JobReceipt::Outcome::kFailed,
        kind,
        false);
  }
};

enum class PositionIssue { kNone, kDuplicate, kOutOfRange, kNegative };

struct BackendState {
  int64_t nextInput{0};
  int32_t instances{0};
  int32_t submissions{0};
  bool failSubmission{false};
  // Exercises framework failures as well as backend rejections.
  velox::rpc::RPCErrorKind submissionErrorKind{
      velox::rpc::RPCErrorKind::kBackendError};
  bool failPolling{false};
  PositionIssue positionIssue{PositionIssue::kNone};
  bool reverseResponses{false};
  int32_t responseCountDelta{0};
  int32_t destructions{0};
  int32_t stops{0};
  int32_t pendingRowsAtStop{0};
  int32_t polls{0};
  int32_t collections{0};
  std::optional<DeferredRpcPhase> heldPhase;
  std::optional<folly::Promise<ReceiptBatch>> receiptPromise;
  std::optional<folly::Promise<std::vector<RPCResponse>>> responsePromise;
  ReceiptBatch heldReceipts;
  std::vector<RPCResponse> heldResponses;
  std::vector<std::string> credentials;
  std::vector<bool> congestionOrigins;
  std::vector<bool> congestionAccountingOwners;
  std::map<std::string, std::map<std::string, int64_t>> jobs;
};

class StringBackend final : public DeferredRpcBackend {
 public:
  StringBackend(DeferredRpcPhase phase, std::shared_ptr<BackendState> state)
      : phase_(phase), state_(std::move(state)) {
    ++state_->instances;
  }
  ~StringBackend() override {
    ++state_->destructions;
  }
  void initialize(
      const core::QueryConfig& config,
      const std::vector<TypePtr>&,
      const std::vector<VectorPtr>&,
      RPCStreamingMode) override {
    initialized_ = true;
    credential_ = config.get<std::string>("credential", "");
    state_->credentials.push_back(credential_);
    failResults_ = config.get<std::string>("on_error", "null") == "fail";
  }
  std::string name() const override {
    return "string_jobs";
  }
  TypePtr resultType() const override {
    return BIGINT();
  }
  const JobReceiptCodec& receiptCodec() const override {
    VELOX_CHECK(initialized_, "Receipt codec requires initialization");
    return codec_;
  }
  std::string admissionKey() const override {
    return "string_jobs";
  }
  bool requiresRowInspectionBeforeAdmission() const override {
    return false;
  }
  std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    DecodedVector values(*args.at(0));
    std::vector<vector_size_t> accepted;
    rows.applyToSelected([&](vector_size_t row) {
      pending_.push_back(
          values.isNullAt(row)
              ? std::nullopt
              : std::optional<int64_t>(values.valueAt<int64_t>(row)));
      accepted.push_back(row);
    });
    return accepted;
  }
  int32_t pendingBatchSize() const override {
    return static_cast<int32_t>(pending_.size());
  }
  folly::SemiFuture<ReceiptBatch> submitPendingBatch(int32_t maxRows) override {
    ++state_->submissions;
    const auto pending = pendingBatchSize();
    const auto count = maxRows > 0 ? std::min(maxRows, pending) : pending;
    ReceiptBatch receipts;
    for (size_t row = 0; row < count; ++row) {
      if (!pending_[row]) {
        receipts.push_back(
            codec_.fromError(velox::rpc::RPCErrorKind::kNullInput, ""));
      } else if (state_->failSubmission) {
        receipts.push_back(
            std::make_shared<StringReceipt>(
                "",
                "",
                JobReceipt::Outcome::kFailed,
                state_->submissionErrorKind));
      } else {
        const auto input = "input-" + std::to_string(state_->nextInput++);
        state_->jobs["opaque-job"][input] = *pending_[row] * 10;
        receipts.push_back(
            std::make_shared<StringReceipt>(
                "opaque-job", input, JobReceipt::Outcome::kSubmitted));
      }
    }
    pending_.erase(pending_.begin(), pending_.begin() + count);
    return completeReceipts(std::move(receipts));
  }
  folly::SemiFuture<ReceiptBatch> poll(ReceiptBatch receipts) override {
    ++state_->polls;
    for (auto& value : receipts) {
      const auto* receipt = dynamic_cast<const StringReceipt*>(value.get());
      VELOX_CHECK_NOT_NULL(receipt);
      if (receipt->state == JobReceipt::Outcome::kSubmitted) {
        value = std::make_shared<StringReceipt>(
            receipt->job,
            receipt->input,
            state_->failPolling ? JobReceipt::Outcome::kFailed
                                : JobReceipt::Outcome::kReady,
            state_->failPolling ? velox::rpc::RPCErrorKind::kBackendError
                                : velox::rpc::RPCErrorKind::kNone);
      }
    }
    return completeReceipts(std::move(receipts));
  }
  folly::SemiFuture<std::vector<RPCResponse>> collect(
      ReceiptBatch receipts) override {
    ++state_->collections;
    std::vector<RPCResponse> responses;
    const auto numRows = static_cast<int64_t>(receipts.size());
    for (int64_t row = 0; row < numRows; ++row) {
      const auto* receipt =
          dynamic_cast<const StringReceipt*>(receipts[row].get());
      VELOX_CHECK_NOT_NULL(receipt);
      if (receipt->state == JobReceipt::Outcome::kNull ||
          receipt->state == JobReceipt::Outcome::kFailed) {
        responses.push_back(
            RPCResponse::failed(row, receipt->kind, "fake failure"));
        responses.back().setCompletionOutcome(receipt->kind, false);
      } else {
        VELOX_CHECK(receipt->state == JobReceipt::Outcome::kReady);
        responses.push_back(
            RPCResponse::ok(
                row,
                NumberPayload{
                    state_->jobs.at(receipt->job).at(receipt->input)}));
      }
    }
    if (!responses.empty()) {
      if (state_->positionIssue == PositionIssue::kOutOfRange) {
        responses.front().rowId = static_cast<int64_t>(responses.size());
      } else if (state_->positionIssue == PositionIssue::kNegative) {
        responses.front().rowId = -1;
      } else if (
          state_->positionIssue == PositionIssue::kDuplicate &&
          responses.size() > 1) {
        responses.back().rowId = responses.front().rowId;
      }
    }
    if (state_->reverseResponses) {
      std::reverse(responses.begin(), responses.end());
    }
    if (state_->responseCountDelta < 0 && !responses.empty()) {
      responses.pop_back();
    } else if (state_->responseCountDelta > 0) {
      responses.push_back(
          RPCResponse::ok(
              static_cast<int64_t>(responses.size()), NumberPayload{0}));
    }
    if (state_->heldPhase == phase_) {
      auto [promise, future] =
          folly::makePromiseContract<std::vector<RPCResponse>>();
      state_->responsePromise = std::move(promise);
      state_->heldResponses = std::move(responses);
      return std::move(future);
    }
    return folly::makeSemiFuture(std::move(responses));
  }
  int32_t admissionUnitsForReceipts(
      const ReceiptBatch& receipts) const override {
    std::set<std::string> jobs;
    for (const auto& value : receipts) {
      const auto* receipt = dynamic_cast<const StringReceipt*>(value.get());
      VELOX_CHECK_NOT_NULL(receipt);
      if (!receipt->job.empty() &&
          receipt->state != JobReceipt::Outcome::kFailed &&
          receipt->state != JobReceipt::Outcome::kNull &&
          !(phase_ == DeferredRpcPhase::kPoll &&
            receipt->state == JobReceipt::Outcome::kReady)) {
        jobs.insert(receipt->job);
      }
    }
    return static_cast<int32_t>(jobs.size());
  }
  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    VELOX_CHECK_LE(
        responses.size(),
        static_cast<size_t>(std::numeric_limits<vector_size_t>::max()));
    const auto numRows = static_cast<vector_size_t>(responses.size());
    auto values =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), numRows, pool);
    for (vector_size_t row = 0; row < numRows; ++row) {
      if (responses[row].hasError()) {
        VELOX_USER_CHECK(
            !failResults_ ||
                responses[row].errorKind() ==
                    velox::rpc::RPCErrorKind::kNullInput,
            "fake collect failure");
        values->setNull(row, true);
      } else {
        values->set(row, responseAs<NumberPayload>(responses[row]).value);
      }
    }
    return values;
  }
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    state_->congestionOrigins.clear();
    state_->congestionAccountingOwners.clear();
    bool hasBackendOutcome{false};
    bool hasFailure{false};
    for (const auto& response : responses) {
      state_->congestionOrigins.push_back(response.isBackendOutcome());
      state_->congestionAccountingOwners.push_back(
          phase_ != DeferredRpcPhase::kCollect && !response.hasError() &&
          responseAs<JobReceiptPayload>(response).accountingOwner != nullptr);
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
      if (!response.hasError()) {
        if (phase_ == DeferredRpcPhase::kCollect) {
          (void)responseAs<NumberPayload>(response);
        } else {
          (void)responseAs<JobReceiptPayload>(response);
        }
      }
    }
    return !hasBackendOutcome ? CongestionSignal::kNone
        : hasFailure          ? CongestionSignal::kNonOverloadError
                              : CongestionSignal::kSuccessNoLatency;
  }
  void stop() noexcept override {
    if (stopped_) {
      return;
    }
    stopped_ = true;
    ++state_->stops;
    state_->pendingRowsAtStop = static_cast<int32_t>(pending_.size());
    pending_.clear();
  }

 private:
  folly::SemiFuture<ReceiptBatch> completeReceipts(ReceiptBatch receipts) {
    if (state_->responseCountDelta < 0 && !receipts.empty()) {
      receipts.pop_back();
    } else if (state_->responseCountDelta > 0) {
      receipts.push_back(
          std::make_shared<StringReceipt>(
              "extra-job", "extra-input", JobReceipt::Outcome::kReady));
    }
    if (state_->heldPhase == phase_) {
      auto [promise, future] = folly::makePromiseContract<ReceiptBatch>();
      state_->receiptPromise = std::move(promise);
      state_->heldReceipts = std::move(receipts);
      return std::move(future);
    }
    return folly::makeSemiFuture(std::move(receipts));
  }

  const DeferredRpcPhase phase_;
  const std::shared_ptr<BackendState> state_;
  StringReceiptCodec codec_;
  std::string credential_;
  bool failResults_{false};
  bool initialized_{false};
  bool stopped_{false};
  std::vector<std::optional<int64_t>> pending_;
};

class DeferredRpcPhaseFunctionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    AsyncRPCFunctionRegistry::testingClear();
    memory::MemoryManager::testingSetInstance({});
  }
  static void TearDownTestSuite() {
    AsyncRPCFunctionRegistry::testingClear();
  }
  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }
  // Uses the same tracked root/leaf hierarchy as a query operator.
  void useTrackedQueryPool() {
    queryPool_ = memory::memoryManager()->addRootPool();
    pool_ = queryPool_->addLeafChild("receipts");
  }
  DeferredRpcBackendFactory factory() {
    return [state = state_](DeferredRpcPhase phase) {
      return std::make_shared<StringBackend>(phase, state);
    };
  }
  // Outlives the accounting leaf used by the memory tests.
  std::shared_ptr<memory::MemoryPool> queryPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<BackendState> state_{std::make_shared<BackendState>()};
};

TEST_F(DeferredRpcPhaseFunctionTest, stringReceiptsCollectNonTextResults) {
  const auto providerFactory = factory();
  EXPECT_TRUE(
      AsyncRPCFunctionRegistry::registerFunction(
          "string_receipt_submit",
          [providerFactory] {
            return std::make_shared<DeferredRpcPhaseFunction>(
                "string_receipt_submit",
                DeferredRpcPhase::kSubmit,
                providerFactory);
          },
          {FunctionSignatureBuilder()
               .returnType(StringReceiptCodec{}.signature())
               .argumentType("bigint")
               .build()}));
  auto registered = std::dynamic_pointer_cast<DeferredRpcPhaseFunction>(
      AsyncRPCFunctionRegistry::create("string_receipt_submit"));
  ASSERT_NE(registered, nullptr);
  auto& submit = *registered;
  DeferredRpcPhaseFunction poll("poll", DeferredRpcPhase::kPoll, factory());
  DeferredRpcPhaseFunction collect(
      "collect", DeferredRpcPhase::kCollect, factory());
  auto config = core::QueryConfig{{{"credential", "test-auth"}}};
  submit.initialize(config, {BIGINT()}, {nullptr}, RPCStreamingMode::kBatch);
  const auto type = StringReceiptCodec{}.receiptType();
  poll.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
  collect.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
  EXPECT_EQ(state_->credentials.size(), 3);
  for (const auto& credential : state_->credentials) {
    EXPECT_EQ(credential, "test-auth");
  }
  auto input =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), 3, pool_.get());
  input->set(0, 1);
  input->set(1, 2);
  input->setNull(2, true);
  submit.accumulateBatch(SelectivityVector(3), {input});
  auto acknowledged = submit.flushBatch(0).get();
  EXPECT_EQ(
      submit.evaluateCongestion(acknowledged),
      AsyncRPCFunction::CongestionSignal::kSuccessNoLatency);
  auto receipts = submit.buildOutput(acknowledged, pool_.get());
  EXPECT_TRUE(receipts->type()->equivalent(*type));
  const auto* rows = receipts->as<RowVector>();
  poll.accumulateBatch(SelectivityVector(3), {receipts});
  auto ready = poll.buildOutput(poll.flushBatch(0).get(), pool_.get());
  auto order = AlignedBuffer::allocate<vector_size_t>(3, pool_.get());
  order->asMutable<vector_size_t>()[0] = 1;
  order->asMutable<vector_size_t>()[1] = 0;
  order->asMutable<vector_size_t>()[2] = 2;
  collect.accumulateBatch(
      SelectivityVector(3),
      {BaseVector::wrapInDictionary(nullptr, order, 3, ready)});
  auto responses = collect.flushBatch(2).get();
  EXPECT_EQ(
      collect.evaluateCongestion(responses),
      AsyncRPCFunction::CongestionSignal::kSuccessNoLatency);
  auto result = collect.buildOutput(responses, pool_.get());
  EXPECT_EQ(result->asFlatVector<int64_t>()->valueAt(0), 20);
  EXPECT_EQ(result->asFlatVector<int64_t>()->valueAt(1), 10);
  auto nullResult =
      collect.buildOutput(collect.flushBatch(0).get(), pool_.get());
  EXPECT_TRUE(nullResult->isNullAt(0));
  EXPECT_EQ(state_->submissions, 1);
  EXPECT_EQ(state_->instances, 3);
  EXPECT_EQ(
      rows->childAt(0)->as<SimpleVector<StringView>>()->valueAt(0).str(),
      "opaque-job");
}

TEST_F(DeferredRpcPhaseFunctionTest, perRowDispatchIsUnsupported) {
  auto backend = factory()(DeferredRpcPhase::kSubmit);
  VELOX_ASSERT_THROW(
      backend->dispatchPerRow(SelectivityVector(0), {}),
      "require BATCH dispatch");
  DeferredRpcPhaseFunction submit(
      "submit", DeferredRpcPhase::kSubmit, factory());
  VELOX_ASSERT_THROW(
      submit.dispatchPerRow(SelectivityVector(0), {}),
      "require BATCH dispatch");
}

TEST_F(DeferredRpcPhaseFunctionTest, admissionCountsDistinctJobsInEachPrefix) {
  for (const auto phase :
       {DeferredRpcPhase::kPoll, DeferredRpcPhase::kCollect}) {
    SCOPED_TRACE(static_cast<int>(phase));
    state_ = std::make_shared<BackendState>();
    const StringReceiptCodec codec;
    const auto outcome = phase == DeferredRpcPhase::kPoll
        ? JobReceipt::Outcome::kSubmitted
        : JobReceipt::Outcome::kReady;
    ReceiptBatch receipts{
        std::make_shared<StringReceipt>("first-job", "first", outcome),
        std::make_shared<StringReceipt>("first-job", "second", outcome),
        std::make_shared<StringReceipt>("second-job", "first", outcome),
        std::make_shared<StringReceipt>(
            "failed-job",
            "first",
            JobReceipt::Outcome::kFailed,
            velox::rpc::RPCErrorKind::kBackendError)};
    DeferredRpcPhaseFunction function("admission-prefix", phase, factory());
    function.initialize(
        core::QueryConfig{{}},
        {codec.receiptType()},
        {nullptr},
        RPCStreamingMode::kBatch);
    function.accumulateBatch(
        SelectivityVector(4), {codec.encode(receipts, pool_.get())});
    EXPECT_EQ(function.admissionUnitsForBatch(1), 1);
    EXPECT_EQ(function.admissionUnitsForBatch(2), 1);
    EXPECT_EQ(function.admissionUnitsForBatch(3), 2);
    EXPECT_EQ(function.admissionUnitsForBatch(4), 2);
  }
}

TEST_F(DeferredRpcPhaseFunctionTest, intermediateFailuresRemainReceipts) {
  state_->failSubmission = true;
  DeferredRpcPhaseFunction submit(
      "submit", DeferredRpcPhase::kSubmit, factory());
  DeferredRpcPhaseFunction poll("poll", DeferredRpcPhase::kPoll, factory());
  DeferredRpcPhaseFunction collect(
      "collect", DeferredRpcPhase::kCollect, factory());
  const core::QueryConfig config{{{"on_error", "fail"}}};
  const auto type = StringReceiptCodec{}.receiptType();
  submit.initialize(config, {BIGINT()}, {nullptr}, RPCStreamingMode::kBatch);
  poll.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
  collect.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
  auto input =
      BaseVector::createConstant(BIGINT(), variant(int64_t{1}), 1, pool_.get());
  submit.accumulateBatch(SelectivityVector(1), {input});
  auto submitted = submit.flushBatch(0).get();
  ASSERT_EQ(submitted.size(), 1);
  EXPECT_FALSE(submitted[0].hasError());
  EXPECT_EQ(
      submitted[0].completionErrorKind(),
      velox::rpc::RPCErrorKind::kBackendError);
  EXPECT_TRUE(submitted[0].isBackendOutcome());
  auto receipts = submit.buildOutput(submitted, pool_.get());
  EXPECT_FALSE(receipts->isNullAt(0));
  poll.accumulateBatch(SelectivityVector(1), {receipts});
  auto failed = poll.buildOutput(poll.flushBatch(0).get(), pool_.get());
  EXPECT_FALSE(failed->isNullAt(0));
  collect.accumulateBatch(SelectivityVector(1), {failed});
  auto responses = collect.flushBatch(0).get();
  VELOX_ASSERT_THROW(
      collect.buildOutput(responses, pool_.get()), "fake collect failure");
}

TEST_F(DeferredRpcPhaseFunctionTest, frameworkReceiptsKeepFatalDiagnostics) {
  for (const auto phase :
       {DeferredRpcPhase::kSubmit, DeferredRpcPhase::kPoll}) {
    SCOPED_TRACE(static_cast<int>(phase));
    state_ = std::make_shared<BackendState>();
    state_->failSubmission = true;
    state_->submissionErrorKind = velox::rpc::RPCErrorKind::kInternalError;
    const StringReceiptCodec codec;
    DeferredRpcPhaseFunction function("framework-error", phase, factory());
    const core::QueryConfig config{{{"on_error", "null"}}};
    const TypePtr inputType = phase == DeferredRpcPhase::kSubmit
        ? TypePtr(BIGINT())
        : TypePtr(codec.receiptType());
    function.initialize(
        config, {inputType}, {nullptr}, RPCStreamingMode::kBatch);
    if (phase == DeferredRpcPhase::kSubmit) {
      auto input = BaseVector::createConstant(
          BIGINT(), variant(int64_t{1}), 1, pool_.get());
      function.accumulateBatch(SelectivityVector(1), {input});
    } else {
      auto input = codec.encode(
          {codec.fromError(velox::rpc::RPCErrorKind::kInternalError, "")},
          pool_.get());
      function.accumulateBatch(SelectivityVector(1), {input});
    }
    auto responses = function.flushBatch(0).get();
    ASSERT_EQ(responses.size(), 1);
    ASSERT_TRUE(responses[0].hasError());
    EXPECT_EQ(
        responses[0].errorKind(), velox::rpc::RPCErrorKind::kInternalError);
    EXPECT_EQ(responses[0].error().message, "fake failure");
    EXPECT_EQ(
        responses[0].isBackendOutcome(), phase == DeferredRpcPhase::kSubmit);
    VELOX_ASSERT_THROW(
        function.buildOutput(responses, pool_.get()),
        "Deferred RPC framework failure: fake failure");
  }
}

TEST_F(DeferredRpcPhaseFunctionTest, collectionRejectsInvalidPositions) {
  for (const auto issue :
       {PositionIssue::kDuplicate,
        PositionIssue::kOutOfRange,
        PositionIssue::kNegative}) {
    state_->positionIssue = issue;
    DeferredRpcPhaseFunction submit(
        "submit", DeferredRpcPhase::kSubmit, factory());
    DeferredRpcPhaseFunction poll("poll", DeferredRpcPhase::kPoll, factory());
    DeferredRpcPhaseFunction collect(
        "collect", DeferredRpcPhase::kCollect, factory());
    const core::QueryConfig config{{}};
    const auto type = StringReceiptCodec{}.receiptType();
    submit.initialize(config, {BIGINT()}, {nullptr}, RPCStreamingMode::kBatch);
    poll.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
    collect.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
    auto input =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), 2, pool_.get());
    input->set(0, 1);
    input->set(1, 2);
    submit.accumulateBatch(SelectivityVector(2), {input});
    auto receipts = submit.buildOutput(submit.flushBatch(0).get(), pool_.get());
    poll.accumulateBatch(SelectivityVector(2), {receipts});
    auto ready = poll.buildOutput(poll.flushBatch(0).get(), pool_.get());
    collect.accumulateBatch(SelectivityVector(2), {ready});
    if (issue == PositionIssue::kDuplicate) {
      VELOX_ASSERT_THROW(
          collect.flushBatch(0).get(), "duplicate response position");
    } else {
      VELOX_ASSERT_THROW(collect.flushBatch(0).get(), "outside input range");
    }
  }
}

TEST_F(DeferredRpcPhaseFunctionTest, collectionAcceptsReorderedPositions) {
  state_->reverseResponses = true;
  DeferredRpcPhaseFunction collect(
      "collect", DeferredRpcPhase::kCollect, factory());
  const core::QueryConfig config{{}};
  const auto type = StringReceiptCodec{}.receiptType();
  collect.initialize(config, {type}, {nullptr}, RPCStreamingMode::kBatch);
  state_->jobs["opaque-job"]["first"] = 10;
  state_->jobs["opaque-job"]["second"] = 20;
  ReceiptBatch receipts{
      std::make_shared<StringReceipt>(
          "opaque-job", "first", JobReceipt::Outcome::kReady),
      std::make_shared<StringReceipt>(
          "opaque-job", "second", JobReceipt::Outcome::kReady)};
  collect.accumulateBatch(
      SelectivityVector(2),
      {StringReceiptCodec{}.encode(receipts, pool_.get())});
  auto responses = collect.flushBatch(0).get();
  EXPECT_EQ(responses.at(0).rowId, 1);
  EXPECT_EQ(responses.at(1).rowId, 0);
}

TEST_F(
    DeferredRpcPhaseFunctionTest,
    congestionNormalizationPreservesHistoricalOriginAndAccounting) {
  state_->failPolling = true;
  const StringReceiptCodec codec;
  auto input = codec.encode(
      {std::make_shared<StringReceipt>(
           "cached-job", "cached-input", JobReceipt::Outcome::kReady),
       std::make_shared<StringReceipt>(
           "active-job", "active-input", JobReceipt::Outcome::kSubmitted)},
      pool_.get());
  DeferredRpcPhaseFunction function(
      "mixed-poll", DeferredRpcPhase::kPoll, factory());
  function.setMemoryPool(pool_);
  function.initialize(
      core::QueryConfig{{}},
      {codec.receiptType()},
      {nullptr},
      RPCStreamingMode::kBatch);
  function.accumulateBatch(SelectivityVector(2), {input});
  auto responses = function.flushBatch(0).get();
  ASSERT_EQ(responses.size(), 2);
  ASSERT_FALSE(responses[0].isBackendOutcome());
  ASSERT_TRUE(responses[1].isBackendOutcome());
  ASSERT_NE(
      responseAs<JobReceiptPayload>(responses[0]).accountingOwner, nullptr);
  EXPECT_EQ(
      function.evaluateCongestion(responses),
      AsyncRPCFunction::CongestionSignal::kNonOverloadError);
  EXPECT_EQ(state_->congestionOrigins, (std::vector<bool>{false, true}));
  EXPECT_EQ(
      state_->congestionAccountingOwners, (std::vector<bool>{true, false}));
}

TEST_F(DeferredRpcPhaseFunctionTest, allPhasesRejectMalformedCardinality) {
  for (const auto phase :
       {DeferredRpcPhase::kSubmit,
        DeferredRpcPhase::kPoll,
        DeferredRpcPhase::kCollect}) {
    for (const auto& [numRows, delta] :
         {std::pair<int32_t, int32_t>{2, -1}, {2, 1}, {0, 1}}) {
      SCOPED_TRACE(
          fmt::format(
              "phase={} numRows={} delta={}",
              static_cast<int>(phase),
              numRows,
              delta));
      state_ = std::make_shared<BackendState>();
      state_->responseCountDelta = delta;
      DeferredRpcPhaseFunction function("cardinality", phase, factory());
      const auto receiptType = StringReceiptCodec{}.receiptType();
      const TypePtr inputType = phase == DeferredRpcPhase::kSubmit
          ? TypePtr(BIGINT())
          : TypePtr(receiptType);
      function.initialize(
          core::QueryConfig{{}},
          {inputType},
          {nullptr},
          RPCStreamingMode::kBatch);
      if (phase == DeferredRpcPhase::kSubmit) {
        auto input = BaseVector::createConstant(
            BIGINT(), variant(int64_t{1}), numRows, pool_.get());
        function.accumulateBatch(SelectivityVector(numRows), {input});
      } else {
        ReceiptBatch receipts;
        for (int32_t row = 0; row < numRows; ++row) {
          const auto input = std::to_string(row);
          state_->jobs["opaque-job"][input] = row;
          receipts.push_back(
              std::make_shared<StringReceipt>(
                  "opaque-job",
                  input,
                  phase == DeferredRpcPhase::kPoll
                      ? JobReceipt::Outcome::kSubmitted
                      : JobReceipt::Outcome::kReady));
        }
        function.accumulateBatch(
            SelectivityVector(numRows),
            {StringReceiptCodec{}.encode(receipts, pool_.get())});
      }
      VELOX_ASSERT_THROW(
          function.flushBatch(0).get(), "changed input cardinality");
    }
  }
}

TEST_F(
    DeferredRpcPhaseFunctionTest,
    everyPhaseRetainsBackendUntilLateCompletion) {
  for (const auto phase :
       {DeferredRpcPhase::kSubmit,
        DeferredRpcPhase::kPoll,
        DeferredRpcPhase::kCollect}) {
    SCOPED_TRACE(static_cast<int>(phase));
    state_ = std::make_shared<BackendState>();
    state_->heldPhase = phase;
    std::weak_ptr<StringBackend> weakBackend;
    auto function = std::make_shared<DeferredRpcPhaseFunction>(
        "late-completion",
        phase,
        [state = state_, &weakBackend](DeferredRpcPhase bindingPhase) {
          auto backend = std::make_shared<StringBackend>(bindingPhase, state);
          weakBackend = backend;
          return backend;
        });
    const auto receiptType = StringReceiptCodec{}.receiptType();
    const TypePtr inputType = phase == DeferredRpcPhase::kSubmit
        ? TypePtr(BIGINT())
        : TypePtr(receiptType);
    function->initialize(
        core::QueryConfig{{}},
        {inputType},
        {nullptr},
        RPCStreamingMode::kBatch);
    if (phase == DeferredRpcPhase::kSubmit) {
      auto input = BaseVector::createConstant(
          BIGINT(), variant(int64_t{1}), 2, pool_.get());
      function->accumulateBatch(SelectivityVector(2), {input});
    } else {
      state_->jobs["opaque-job"]["first"] = 10;
      state_->jobs["opaque-job"]["second"] = 20;
      ReceiptBatch receipts{
          std::make_shared<StringReceipt>(
              "opaque-job",
              "first",
              phase == DeferredRpcPhase::kPoll ? JobReceipt::Outcome::kSubmitted
                                               : JobReceipt::Outcome::kReady),
          std::make_shared<StringReceipt>(
              "opaque-job",
              "second",
              phase == DeferredRpcPhase::kPoll ? JobReceipt::Outcome::kSubmitted
                                               : JobReceipt::Outcome::kReady)};
      function->accumulateBatch(
          SelectivityVector(2),
          {StringReceiptCodec{}.encode(receipts, pool_.get())});
    }
    auto future = function->flushBatch(1);
    ASSERT_FALSE(future.isReady());
    EXPECT_EQ(function->pendingBatchSize(), 1);
    function->close();
    EXPECT_EQ(function->pendingBatchSize(), 0);
    EXPECT_EQ(state_->stops, 1);
    if (phase == DeferredRpcPhase::kSubmit) {
      EXPECT_EQ(state_->pendingRowsAtStop, 1);
    }
    function.reset();
    EXPECT_FALSE(weakBackend.expired());
    EXPECT_EQ(state_->destructions, 0);
    if (phase == DeferredRpcPhase::kCollect) {
      state_->responsePromise->setValue(std::move(state_->heldResponses));
    } else {
      state_->receiptPromise->setValue(std::move(state_->heldReceipts));
    }
    auto responses = std::move(future).get();
    ASSERT_EQ(responses.size(), 1);
    EXPECT_FALSE(responses.front().hasError());
    EXPECT_TRUE(weakBackend.expired());
    EXPECT_EQ(state_->destructions, 1);
    EXPECT_EQ(state_->stops, 1);
  }
}

TEST_F(
    DeferredRpcPhaseFunctionTest,
    receiptCopiesAreChargedUntilTheirOwnerReleases) {
  useTrackedQueryPool();
  constexpr vector_size_t kRows = 16;
  constexpr size_t kIdentityBytes = 32 * 1024;
  const StringReceiptCodec codec;
  auto base = codec.encode(
      {std::make_shared<StringReceipt>(
          std::string(kIdentityBytes, 'j'),
          "input",
          JobReceipt::Outcome::kSubmitted)},
      pool_.get());
  auto indices = AlignedBuffer::allocate<vector_size_t>(kRows, pool_.get());
  std::fill_n(indices->asMutable<vector_size_t>(), kRows, 0);
  auto input = BaseVector::wrapInDictionary(nullptr, indices, kRows, base);
  state_->heldPhase = DeferredRpcPhase::kPoll;
  DeferredRpcPhaseFunction function(
      "memory", DeferredRpcPhase::kPoll, factory());
  function.setMemoryPool(pool_);
  function.initialize(
      core::QueryConfig{{}},
      {codec.receiptType()},
      {nullptr},
      RPCStreamingMode::kBatch);
  const auto before = pool_->stats();
  ASSERT_EQ(input->pool(), pool_.get());
  ASSERT_GT(before.usedBytes, 0);
  function.accumulateBatch(SelectivityVector(kRows), {input});
  const auto buffered = pool_->stats();
  EXPECT_GE(buffered.usedBytes, before.usedBytes + kRows * kIdentityBytes);
  EXPECT_GT(buffered.numExternalAllocs, before.numExternalAllocs);
  auto future = function.flushBatch(0);
  ASSERT_FALSE(future.isReady());
  EXPECT_GE(pool_->usedBytes(), before.usedBytes + kRows * kIdentityBytes);
  state_->receiptPromise->setValue(std::move(state_->heldReceipts));
  auto responses = std::move(future).get();
  EXPECT_EQ(velox::rpc::totalResponseRetainedBytes(responses), 0);
  EXPECT_GE(pool_->usedBytes(), before.usedBytes + kRows * kIdentityBytes);
  function.close();
  responses.clear();
  EXPECT_EQ(pool_->usedBytes(), before.usedBytes);
  EXPECT_EQ(pool_->stats().numExternalAllocs, pool_->stats().numExternalFrees);
}

TEST_F(DeferredRpcPhaseFunctionTest, closeReleasesUndispatchedReceiptCopies) {
  useTrackedQueryPool();
  const StringReceiptCodec codec;
  auto input = codec.encode(
      {std::make_shared<StringReceipt>(
          std::string(32 * 1024, 'j'),
          "input",
          JobReceipt::Outcome::kSubmitted)},
      pool_.get());
  DeferredRpcPhaseFunction function(
      "close", DeferredRpcPhase::kPoll, factory());
  function.setMemoryPool(pool_);
  function.initialize(
      core::QueryConfig{{}},
      {codec.receiptType()},
      {nullptr},
      RPCStreamingMode::kBatch);
  const auto before = pool_->usedBytes();
  ASSERT_EQ(input->pool(), pool_.get());
  ASSERT_GT(before, 0);
  function.accumulateBatch(SelectivityVector(1), {input});
  EXPECT_GT(pool_->usedBytes(), before);
  function.close();
  EXPECT_EQ(function.pendingBatchSize(), 0);
  EXPECT_EQ(pool_->usedBytes(), before);
  EXPECT_EQ(state_->polls, 0);
}

TEST_F(DeferredRpcPhaseFunctionTest, preparedReceiptsRespectChangedSelection) {
  for (const auto phase :
       {DeferredRpcPhase::kPoll, DeferredRpcPhase::kCollect}) {
    SCOPED_TRACE(static_cast<int>(phase));
    state_ = std::make_shared<BackendState>();
    const StringReceiptCodec codec;
    ReceiptBatch receipts;
    for (vector_size_t row = 0; row < 3; ++row) {
      const auto input = std::to_string(row);
      state_->jobs["job"][input] = row + 10;
      receipts.push_back(
          std::make_shared<StringReceipt>(
              "job",
              input,
              phase == DeferredRpcPhase::kPoll ? JobReceipt::Outcome::kSubmitted
                                               : JobReceipt::Outcome::kReady));
    }
    auto input = codec.encode(receipts, pool_.get());
    DeferredRpcPhaseFunction function("selection", phase, factory());
    function.initialize(
        core::QueryConfig{{}},
        {codec.receiptType()},
        {nullptr},
        RPCStreamingMode::kBatch);
    SelectivityVector preparedRows(3, false);
    preparedRows.setValid(0, true);
    preparedRows.setValid(1, true);
    preparedRows.updateBounds();
    function.prepareInputForAdmissionAndDispatch(preparedRows, {input});

    SelectivityVector selectedRows(3, false);
    selectedRows.setValid(0, true);
    selectedRows.setValid(2, true);
    selectedRows.updateBounds();
    EXPECT_THAT(
        function.accumulateBatch(selectedRows, {input}),
        testing::ElementsAre(0, 2));
    auto responses = function.flushBatch(0).get();
    ASSERT_EQ(responses.size(), 2);
    if (phase == DeferredRpcPhase::kPoll) {
      for (size_t row = 0; row < responses.size(); ++row) {
        const auto* receipt = dynamic_cast<const StringReceipt*>(
            responseAs<JobReceiptPayload>(responses[row]).receipt.get());
        ASSERT_NE(receipt, nullptr);
        EXPECT_EQ(receipt->input, std::to_string(row * 2));
        EXPECT_EQ(receipt->outcome(), JobReceipt::Outcome::kReady);
      }
    } else {
      EXPECT_EQ(responseAs<NumberPayload>(responses[0]).value, 10);
      EXPECT_EQ(responseAs<NumberPayload>(responses[1]).value, 12);
    }
  }
}

TEST_F(DeferredRpcPhaseFunctionTest, terminalReceiptsCompleteWithoutAdmission) {
  for (const auto phase :
       {DeferredRpcPhase::kPoll, DeferredRpcPhase::kCollect}) {
    state_ = std::make_shared<BackendState>();
    const StringReceiptCodec codec;
    auto input = codec.encode(
        {codec.fromError(velox::rpc::RPCErrorKind::kTimeout, "timeout"),
         codec.fromError(velox::rpc::RPCErrorKind::kNullInput, "")},
        pool_.get());
    DeferredRpcPhaseFunction function("local", phase, factory());
    function.initialize(
        core::QueryConfig{{}},
        {codec.receiptType()},
        {nullptr},
        RPCStreamingMode::kBatch);
    EXPECT_TRUE(function.requiresRowInspectionBeforeAdmission());
    EXPECT_EQ(
        function.prepareInputForAdmissionAndDispatch(
            SelectivityVector(2), {input}),
        AsyncRPCFunction::AdmissionPreparationResult::kLocalOnly);
    function.accumulateBatch(SelectivityVector(2), {input});
    auto responses = function.flushBatch(0).get();
    ASSERT_EQ(responses.size(), 2);
    EXPECT_FALSE(responses[0].isBackendOutcome());
    EXPECT_FALSE(responses[1].isBackendOutcome());
    EXPECT_EQ(
        responses[0].completionErrorKind(), velox::rpc::RPCErrorKind::kTimeout);
    EXPECT_EQ(
        responses[1].completionErrorKind(),
        velox::rpc::RPCErrorKind::kNullInput);
  }
}

} // namespace
} // namespace facebook::velox::exec::rpc
