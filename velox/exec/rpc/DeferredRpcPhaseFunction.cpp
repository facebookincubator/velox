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

#include <algorithm>
#include <limits>

#include <folly/Synchronized.h>
#include <folly/executors/InlineExecutor.h>

#include "velox/exec/rpc/RPCState.h"

namespace facebook::velox::exec::rpc {
namespace {

// Runs deferred CPU continuations so a local-only flush is immediately ready.
folly::SemiFuture<std::vector<RPCResponse>> startPhaseFuture(
    folly::SemiFuture<std::vector<RPCResponse>> future) {
  return std::move(future)
      .via(folly::getKeepAliveToken(folly::InlineExecutor::instance()))
      .semi();
}

// Pairs opaque receipt storage with its query charge without changing its type.
struct AccountedReceiptDeleter {
  // Keeps accounting alive until storage has been released.
  std::shared_ptr<RPCState::PayloadMemoryCharge> charge;
  // Owns the concrete receipt behind the alias exposed to the backend.
  JobReceiptPtr storage;

  void operator()(const JobReceipt*) noexcept {
    storage.reset();
    charge.reset();
  }
};

JobReceiptPayload accountReceipt(
    JobReceiptPtr receipt,
    const std::shared_ptr<memory::MemoryPool>& pool) {
  VELOX_CHECK_NOT_NULL(receipt);
  if (auto* owner = std::get_deleter<AccountedReceiptDeleter>(receipt)) {
    return JobReceiptPayload{std::move(receipt), owner->charge};
  }
  if (!pool) {
    return JobReceiptPayload{std::move(receipt), {}};
  }
  const auto retainedBytes = receipt->retainedBytes();
  VELOX_CHECK_GE(retainedBytes, 0);
  const auto accountingBytes = static_cast<int64_t>(
      sizeof(AccountedReceiptDeleter) + sizeof(RPCState::PayloadMemoryCharge));
  VELOX_CHECK_LE(
      retainedBytes, std::numeric_limits<int64_t>::max() - accountingBytes);
  const auto chargeBytes = retainedBytes + accountingBytes;
  auto charge =
      std::make_shared<RPCState::PayloadMemoryCharge>(pool, chargeBytes);
  const auto* value = receipt.get();
  auto accounted =
      JobReceiptPtr(value, AccountedReceiptDeleter{charge, std::move(receipt)});
  return JobReceiptPayload{std::move(accounted), std::move(charge)};
}

bool isTerminalForPhase(const JobReceipt& receipt, DeferredRpcPhase phase) {
  return receipt.outcome() == JobReceipt::Outcome::kFailed ||
      receipt.outcome() == JobReceipt::Outcome::kNull ||
      (phase == DeferredRpcPhase::kPoll &&
       receipt.outcome() == JobReceipt::Outcome::kReady);
}

} // namespace

struct DeferredRpcPhaseFunction::ReceiptAccounting {
  // Detaches at close; charges themselves retain their allocator independently.
  folly::Synchronized<std::shared_ptr<memory::MemoryPool>> pool;
};

struct DeferredRpcPhaseFunction::PendingReceipts {
  using PayloadQueue =
      std::vector<JobReceiptPayload, memory::StlAllocator<JobReceiptPayload>>;
  using PreparedInput = std::pair<vector_size_t, JobReceiptPayload>;
  using PreparedQueue =
      std::vector<PreparedInput, memory::StlAllocator<PreparedInput>>;

  explicit PendingReceipts(std::shared_ptr<memory::MemoryPool> allocator)
      : pool(std::move(allocator)),
        pending(memory::StlAllocator<JobReceiptPayload>(pool.get())),
        prepared(memory::StlAllocator<PreparedInput>(pool.get())) {}

  // Outlives the containers that use its raw allocator pointer.
  std::shared_ptr<memory::MemoryPool> pool;
  // Stores owned, accounted receipts until they are dispatched.
  PayloadQueue pending;
  // Avoids a second decode after admission preparation.
  PreparedQueue prepared;
  // Identifies the vector inspected by the current preparation call.
  const BaseVector* preparedInput{nullptr};
};

struct DeferredRpcPhaseFunction::ReceiptOwners {
  explicit ReceiptOwners(std::shared_ptr<memory::MemoryPool> allocator)
      : pool(std::move(allocator)),
        payloads(memory::StlAllocator<JobReceiptPayload>(pool.get())) {}

  // Outlives the container allocator through asynchronous completion.
  std::shared_ptr<memory::MemoryPool> pool;
  // Accounts the transferred ReceiptBatch's pointer storage.
  RPCState::PayloadMemoryCharge dispatchCharge;
  // Keeps decoded storage and its charges alive during backend work.
  PendingReceipts::PayloadQueue payloads;
};

std::vector<RPCResponse> DeferredRpcPhaseFunction::receiptResponses(
    ReceiptBatch receipts,
    const std::shared_ptr<ReceiptAccounting>& accounting) {
  const auto pool = accounting->pool.copy();
  VELOX_CHECK_LE(
      receipts.size(),
      static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  const auto numRows = static_cast<int64_t>(receipts.size());
  std::vector<RPCResponse> responses;
  responses.reserve(receipts.size());
  for (int64_t row = 0; row < numRows; ++row) {
    VELOX_CHECK_NOT_NULL(receipts[row]);
    const auto kind = receipts[row]->errorKind();
    const auto backendOutcome = receipts[row]->reportsBackendOutcome();
    // Preserve the diagnostic before the operator rejects a framework failure.
    auto response = velox::rpc::errorCategory(kind) ==
            velox::rpc::RPCErrorCategory::kFramework
        ? RPCResponse::failed(row, kind, std::string(receipts[row]->error()))
        : RPCResponse::ok(row, accountReceipt(std::move(receipts[row]), pool));
    response.setCompletionOutcome(kind, backendOutcome);
    responses.push_back(std::move(response));
  }
  return responses;
}

DeferredRpcPhaseFunction::DeferredRpcPhaseFunction(
    std::string name,
    DeferredRpcPhase phase,
    const DeferredRpcBackendFactory& factory)
    : name_(std::move(name)),
      phase_(phase),
      backend_(factory ? factory(phase) : nullptr),
      accounting_(std::make_shared<ReceiptAccounting>()) {
  VELOX_CHECK_NOT_NULL(backend_);
}

DeferredRpcPhaseFunction::~DeferredRpcPhaseFunction() {
  close();
}

void DeferredRpcPhaseFunction::setMemoryPool(
    std::shared_ptr<memory::MemoryPool> pool) {
  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK(!closed_, "Deferred RPC function is closed");
  if (receipts_) {
    VELOX_CHECK(
        receipts_->pool.get() == pool.get(),
        "Deferred RPC allocator must remain unchanged after decoding input");
  } else if (phase_ != DeferredRpcPhase::kSubmit) {
    receipts_ = std::make_unique<PendingReceipts>(pool);
  }
  *accounting_->pool.wlock() = pool;
  backend_->setMemoryPool(std::move(pool));
}

void DeferredRpcPhaseFunction::ensureMemoryPool(
    const std::vector<VectorPtr>& args) {
  if (!accounting_->pool.copy()) {
    VELOX_CHECK(!args.empty());
    VELOX_CHECK_NOT_NULL(args[0]);
    setMemoryPool(args[0]->pool()->shared_from_this());
  }
}

void DeferredRpcPhaseFunction::close() noexcept {
  if (closed_) {
    return;
  }
  closed_ = true;
  backend_->stop();
  receipts_.reset();
  accounting_->pool.wlock()->reset();
}

void DeferredRpcPhaseFunction::initialize(
    const core::QueryConfig& queryConfig,
    const std::vector<TypePtr>& inputTypes,
    const std::vector<VectorPtr>& constantInputs,
    RPCStreamingMode instruction) {
  VELOX_USER_CHECK(
      instruction == RPCStreamingMode::kBatch,
      "Deferred RPC requires BATCH dispatch");
  backend_->initialize(queryConfig, inputTypes, constantInputs, instruction);
  if (phase_ != DeferredRpcPhase::kSubmit) {
    VELOX_USER_CHECK(!inputTypes.empty());
    VELOX_USER_CHECK(
        inputTypes[0]->equivalent(*backend_->receiptCodec().receiptType()));
  }
}

std::string DeferredRpcPhaseFunction::name() const {
  return name_;
}
TypePtr DeferredRpcPhaseFunction::resultType() const {
  return phase_ == DeferredRpcPhase::kCollect
      ? backend_->resultType()
      : backend_->receiptCodec().receiptType();
}
std::string DeferredRpcPhaseFunction::admissionKey() const {
  return backend_->admissionKey();
}
int64_t DeferredRpcPhaseFunction::numRetriesAttempted() const {
  return backend_->numRetriesAttempted();
}
int64_t DeferredRpcPhaseFunction::configuredCeiling() const {
  return backend_->configuredCeiling();
}
double DeferredRpcPhaseFunction::configuredAdaptiveFloor() const {
  return backend_->configuredAdaptiveFloor();
}
std::chrono::milliseconds DeferredRpcPhaseFunction::requestTimeout() const {
  return backend_->requestTimeout();
}
bool DeferredRpcPhaseFunction::requiresRowInspectionBeforeAdmission() const {
  return phase_ != DeferredRpcPhase::kSubmit ||
      backend_->requiresRowInspectionBeforeAdmission();
}

AsyncRPCFunction::AdmissionPreparationResult
DeferredRpcPhaseFunction::prepareInputForAdmissionAndDispatch(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args) {
  VELOX_CHECK(!closed_, "Deferred RPC function is closed");
  ensureMemoryPool(args);
  if (phase_ == DeferredRpcPhase::kSubmit) {
    return backend_->prepareInputForAdmissionAndDispatch(rows, args);
  }
  receipts_->prepared.clear();
  receipts_->preparedInput = args[0].get();
  ReceiptBatch admitted;
  admitted.reserve(rows.countSelected());
  bool localOnly{true};
  rows.applyToSelected([&](vector_size_t row) {
    auto payload = accountReceipt(
        backend_->receiptCodec().decode(args[0], row), receipts_->pool);
    localOnly = localOnly && isTerminalForPhase(*payload.receipt, phase_);
    admitted.push_back(payload.receipt);
    receipts_->prepared.emplace_back(row, std::move(payload));
  });
  return localOnly || backend_->admissionUnitsForReceipts(admitted) == 0
      ? AdmissionPreparationResult::kLocalOnly
      : AdmissionPreparationResult::kRequiresAdmission;
}

std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
DeferredRpcPhaseFunction::dispatchPerRow(
    const SelectivityVector&,
    const std::vector<VectorPtr>&) {
  VELOX_UNSUPPORTED("Deferred RPC phases require BATCH dispatch");
}

std::vector<vector_size_t> DeferredRpcPhaseFunction::accumulateBatch(
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args) {
  VELOX_CHECK(!closed_, "Deferred RPC function is closed");
  ensureMemoryPool(args);
  if (phase_ == DeferredRpcPhase::kSubmit) {
    return backend_->accumulateBatch(rows, args);
  }
  VELOX_CHECK(!args.empty());
  std::vector<vector_size_t> accepted;
  if (receipts_->preparedInput == args[0].get() &&
      receipts_->prepared.size() == rows.countSelected() &&
      std::all_of(
          receipts_->prepared.begin(),
          receipts_->prepared.end(),
          [&rows](const auto& prepared) {
            return prepared.first < rows.size() && rows.isValid(prepared.first);
          })) {
    for (auto& [row, payload] : receipts_->prepared) {
      receipts_->pending.push_back(std::move(payload));
      accepted.push_back(row);
    }
  } else {
    rows.applyToSelected([&](vector_size_t row) {
      receipts_->pending.push_back(accountReceipt(
          backend_->receiptCodec().decode(args[0], row), receipts_->pool));
      accepted.push_back(row);
    });
  }
  receipts_->prepared.clear();
  receipts_->prepared.shrink_to_fit();
  receipts_->preparedInput = nullptr;
  return accepted;
}

folly::SemiFuture<std::vector<RPCResponse>>
DeferredRpcPhaseFunction::flushBatch(int32_t maxRows) {
  VELOX_CHECK(!closed_, "Deferred RPC function is closed");
  if (phase_ == DeferredRpcPhase::kSubmit) {
    const auto pending = backend_->pendingBatchSize();
    const auto count = maxRows > 0 ? std::min(maxRows, pending) : pending;
    return startPhaseFuture(backend_->submitPendingBatch(maxRows).deferValue(
        [backend = backend_, accounting = accounting_, count](
            ReceiptBatch receipts) {
          (void)backend;
          VELOX_CHECK_EQ(
              receipts.size(),
              count,
              "Deferred submission changed input cardinality");
          return receiptResponses(std::move(receipts), accounting);
        }));
  }
  const auto pending = pendingBatchSize();
  const auto count = maxRows > 0 ? std::min(maxRows, pending) : pending;
  std::shared_ptr<ReceiptOwners> owners;
  ReceiptBatch receipts;
  receipts.reserve(count);
  if (count > 0) {
    VELOX_CHECK_NOT_NULL(receipts_);
    owners = std::make_shared<ReceiptOwners>(receipts_->pool);
    owners->payloads.insert(
        owners->payloads.end(),
        receipts_->pending.begin(),
        receipts_->pending.begin() + count);
    VELOX_CHECK_LE(
        receipts.capacity(),
        static_cast<size_t>(std::numeric_limits<int64_t>::max()) /
            sizeof(JobReceiptPtr));
    const auto dispatchBytes = static_cast<int64_t>(receipts.capacity()) *
        static_cast<int64_t>(sizeof(JobReceiptPtr));
    owners->dispatchCharge =
        RPCState::PayloadMemoryCharge(owners->pool, dispatchBytes);
    for (const auto& payload : owners->payloads) {
      receipts.push_back(payload.receipt);
    }
    receipts_->pending.erase(
        receipts_->pending.begin(), receipts_->pending.begin() + count);
    if (receipts_->pending.empty()) {
      receipts_->pending.shrink_to_fit();
    }
  }
  const bool localOnly = !receipts.empty() &&
      std::all_of(
          receipts.begin(),
          receipts.end(),
          [phase = phase_](const auto& receipt) {
            return isTerminalForPhase(*receipt, phase);
          });
  if (phase_ == DeferredRpcPhase::kPoll) {
    auto future = localOnly ? folly::makeSemiFuture(std::move(receipts))
                            : backend_->poll(std::move(receipts));
    return startPhaseFuture(
        std::move(future).deferValue(
            [backend = backend_, owners, accounting = accounting_, count](
                ReceiptBatch ready) {
              (void)backend;
              (void)owners;
              VELOX_CHECK_EQ(
                  ready.size(),
                  count,
                  "Deferred polling changed input cardinality");
              return receiptResponses(std::move(ready), accounting);
            }));
  }
  auto future = [&]() -> folly::SemiFuture<std::vector<RPCResponse>> {
    if (!localOnly) {
      return backend_->collect(std::move(receipts));
    }
    std::vector<RPCResponse> responses;
    responses.reserve(receipts.size());
    for (int32_t row = 0; row < count; ++row) {
      auto response = RPCResponse::failed(
          row, receipts[row]->errorKind(), std::string(receipts[row]->error()));
      response.setCompletionOutcome(response.errorKind(), false);
      responses.push_back(std::move(response));
    }
    return folly::makeSemiFuture(std::move(responses));
  }();
  return startPhaseFuture(
      std::move(future).deferValue([backend = backend_, owners, count](
                                       std::vector<RPCResponse> responses) {
        (void)backend;
        (void)owners;
        VELOX_CHECK_EQ(
            responses.size(),
            count,
            "Deferred collection changed input cardinality");
        std::vector<bool> seen(count, false);
        for (const auto& response : responses) {
          VELOX_CHECK_GE(
              response.rowId,
              0,
              "Deferred collection response position is outside input range: {}",
              response.rowId);
          VELOX_CHECK_LT(
              response.rowId,
              static_cast<int64_t>(count),
              "Deferred collection response position is outside input range: {}",
              response.rowId);
          const auto position = static_cast<size_t>(response.rowId);
          VELOX_CHECK(
              !seen[position],
              "Deferred collection returned duplicate response position: {}",
              response.rowId);
          seen[position] = true;
        }
        return responses;
      }));
}

int32_t DeferredRpcPhaseFunction::pendingBatchSize() const {
  if (closed_) {
    return 0;
  }
  if (phase_ == DeferredRpcPhase::kSubmit) {
    return backend_->pendingBatchSize();
  }
  if (!receipts_) {
    return 0;
  }
  VELOX_CHECK_LE(
      receipts_->pending.size(),
      static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "Deferred RPC pending receipts exceed the supported row count");
  return static_cast<int32_t>(receipts_->pending.size());
}
int32_t DeferredRpcPhaseFunction::maxRowsPerFlush(int32_t desired) const {
  return phase_ == DeferredRpcPhase::kSubmit
      ? backend_->maxRowsPerFlush(desired)
      : desired;
}
int32_t DeferredRpcPhaseFunction::admissionUnitsForBatch(
    int32_t numRows) const {
  if (phase_ == DeferredRpcPhase::kSubmit) {
    return backend_->admissionUnitsForBatch(numRows);
  }
  if (!receipts_) {
    return 1;
  }
  const auto count =
      std::min<size_t>(std::max(0, numRows), receipts_->pending.size());
  ReceiptBatch receipts;
  receipts.reserve(count);
  for (size_t row = 0; row < count; ++row) {
    receipts.push_back(receipts_->pending[row].receipt);
  }
  return std::max(1, backend_->admissionUnitsForReceipts(receipts));
}

VectorPtr DeferredRpcPhaseFunction::buildOutput(
    const std::vector<RPCResponse>& responses,
    memory::MemoryPool* pool) const {
  if (phase_ == DeferredRpcPhase::kCollect) {
    return backend_->buildOutput(responses, pool);
  }
  ReceiptBatch receipts;
  receipts.reserve(responses.size());
  for (const auto& response : responses) {
    if (response.hasError()) {
      VELOX_CHECK(
          velox::rpc::errorCategory(response.errorKind()) !=
              velox::rpc::RPCErrorCategory::kFramework,
          "Deferred RPC framework failure: {}",
          response.error().message);
      receipts.push_back(backend_->receiptCodec().fromError(
          response.errorKind(), response.error().message));
    } else {
      const auto& receipt = responseAs<JobReceiptPayload>(response).receipt;
      VELOX_CHECK(
          velox::rpc::errorCategory(receipt->errorKind()) !=
              velox::rpc::RPCErrorCategory::kFramework,
          "Deferred RPC framework failure: {}",
          receipt->error());
      receipts.push_back(receipt);
    }
    VELOX_CHECK_NOT_NULL(receipts.back());
  }
  return backend_->receiptCodec().encode(receipts, pool);
}

AsyncRPCFunction::CongestionSignal DeferredRpcPhaseFunction::evaluateCongestion(
    const std::vector<RPCResponse>& responses) const {
  if (phase_ == DeferredRpcPhase::kCollect) {
    return backend_->evaluateCongestion(responses);
  }
  std::vector<RPCResponse> normalized;
  normalized.reserve(responses.size());
  for (const auto& response : responses) {
    const auto kind = response.completionErrorKind();
    if (kind != velox::rpc::RPCErrorKind::kNone) {
      auto value = RPCResponse::failed(response.rowId, kind, "");
      value.setCompletionOutcome(kind, response.isBackendOutcome());
      normalized.push_back(std::move(value));
    } else {
      const auto& receipt = responseAs<JobReceiptPayload>(response).receipt;
      VELOX_CHECK_NOT_NULL(receipt);
      if (receipt->outcome() == JobReceipt::Outcome::kFailed ||
          receipt->outcome() == JobReceipt::Outcome::kNull) {
        auto value =
            RPCResponse::failed(response.rowId, receipt->errorKind(), "");
        value.setCompletionOutcome(
            receipt->errorKind(), response.isBackendOutcome());
        normalized.push_back(std::move(value));
      } else {
        auto value = RPCResponse::ok(
            response.rowId, responseAs<JobReceiptPayload>(response));
        value.setCompletionOutcome(
            velox::rpc::RPCErrorKind::kNone, response.isBackendOutcome());
        normalized.push_back(std::move(value));
      }
    }
  }
  return backend_->evaluateCongestion(normalized);
}

} // namespace facebook::velox::exec::rpc
