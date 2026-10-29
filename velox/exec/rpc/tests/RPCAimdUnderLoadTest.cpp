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

/// Layer 2 — AIMD-under-load behavioral test for the async RPC framework.
///
/// Layer 1 (the folly microbenchmark) proves the framework's per-row overhead
/// is stable. This layer proves the framework's *closed-loop behavior* under a
/// backend overload: it drives a real Task/Driver pipeline through three phases
///   warm-up  → overload burst → recovery
/// and asserts, from the operator's own runtime stats, that BOTH AIMD control
/// loops reacted correctly — the per-driver congestion window shrank, the
/// process-global rate-limiter cap backed off below its ceiling, and the cap
/// then recovered upward once the burst cleared.
///
/// The burst is injected deterministically: ResponseSimulator tags a fixed
/// window of call ordinals with errorKind=kRateLimited (see
/// ResponseSimulator::ErrorBurst). Call ordinals are reserved in input
/// traversal order; completion order is deliberately not assumed. The query
/// evaluates one completed response at a time so an overload and later
/// successes cannot be coalesced into one congestion verdict. The whole
/// trajectory is captured in a single close()-time stats snapshot:
/// numShrinks>0 proves the window dipped; rpcRateLimiterMinCap lands on every
/// query, so it is the value, not its presence, that proves the limiter backed
/// off -- a low-water mark below the ceiling; the final cap sitting above that
/// low-water mark proves recovery.

#include <mutex>
#include <optional>

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/rpc/RPCOperator.h"
#include "velox/exec/rpc/RPCPlanNodeTranslator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
#include "velox/exec/rpc/tests/ResponseSimulator.h"
#include "velox/exec/rpc/tests/TextOutput.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

namespace facebook::velox::exec::rpc {
namespace {

using namespace facebook::velox::exec::test;
using exec::rpc::test::ResponseSimulator;
using velox::rpc::RPCErrorKind;
using velox::rpc::RPCResponse;

// The registry requires every function to declare a signature. These
// functions are reached through a hand-built plan rather than by name
// resolution, so one varchar overload is all they need.
AsyncRPCFunctionRegistry::Signatures varcharSignature() {
  return {exec::FunctionSignatureBuilder()
              .returnType("varchar")
              .argumentType("varchar")
              .build()};
}

// What the two burst harnesses share: a ResponseSimulator that fails a
// deterministic window of call ordinals, the admission key, and the overload
// classifier. Only the dispatch shape differs, so only that is left to the
// subclasses -- a contract change on AsyncRPCFunction is then absorbed here
// once instead of in both.
//
// The classifier treats rate-limit/timeout as backend overload (kOverloaded ->
// both controllers back off) and a clean drain as kSuccess (feeds the RTT
// gradient and drives rate-limiter recovery), exactly as a production
// congestion policy would.
class BurstFunctionBase : public AsyncRPCFunction {
 public:
  struct Config {
    // Requests with call ordinal in [burstFirstCall, burstLastCall) are failed.
    int64_t burstFirstCall{0};
    int64_t burstLastCall{0};
    RPCErrorKind burstKind{RPCErrorKind::kRateLimited};
    std::chrono::milliseconds latency{std::chrono::milliseconds(1)};
    // Non-empty so the process-global rate limiter is keyed on a real backend.
    std::string backend{"layer2.test.backend"};
    // Attach a retry hint to burst errors and declare it in
    // mayReturnCapacityRetries().
    bool hintBurstErrors{false};
  };

  explicit BurstFunctionBase(Config config) : config_{std::move(config)} {}

  void initialize(
      const core::QueryConfig& /*queryConfig*/,
      const std::vector<TypePtr>& /*inputTypes*/,
      const std::vector<VectorPtr>& /*constantInputs*/,
      RPCStreamingMode /*instruction*/) override {
    // Call ordinals are what pin the burst to rows [kWarmupRows,
    // kWarmupRows + kBurstRows). A second initialize() would install a fresh
    // client whose ordinals restart at 0 and silently move the burst, so
    // require exactly one.
    VELOX_CHECK_NULL(simulator_, "initialize() must be called at most once");
    simulator_ = std::make_shared<ResponseSimulator>(config_.latency);
    simulator_->setErrorBurst(
        {config_.burstFirstCall, config_.burstLastCall, config_.burstKind});
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  std::string admissionKey() const override {
    return config_.backend;
  }

  bool mayReturnCapacityRetries() const override {
    return config_.hintBurstErrors;
  }

  // Overload failures produce kOverloaded. A null-input error is a user error
  // and must not move the window (folded into kSuccess/kNone below). A clean
  // drain feeds its RTT to the gradient and drives rate-limiter recovery.
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    bool hasNonOverloadError{false};
    for (const auto& response : responses) {
      if (!response.hasError()) {
        continue;
      }
      if (velox::rpc::errorCategory(response.errorKind()) ==
          velox::rpc::RPCErrorCategory::kOverload) {
        return CongestionSignal::kOverloaded;
      }
      hasNonOverloadError = true;
    }
    if (responses.empty()) {
      return CongestionSignal::kNone;
    }
    return hasNonOverloadError ? CongestionSignal::kNonOverloadError
                               : CongestionSignal::kSuccess;
  }

  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    return buildTextOutput(responses, pool);
  }

 protected:
  Config config_;
  std::shared_ptr<ResponseSimulator> simulator_;
};

// PER_ROW: one call per row, so the burst window maps 1:1 onto row indices.
class BurstRPCFunction : public BurstFunctionBase {
 public:
  using BurstFunctionBase::BurstFunctionBase;

  std::string name() const override {
    return "burst_rpc";
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        results;
    VELOX_CHECK(!args.empty(), "burst_rpc expects one argument");
    // Returning no futures for selected rows would strand them and surface far
    // downstream, so fail here instead of degrading silently.
    auto* promptVector = args[0]->as<SimpleVector<StringView>>();
    VELOX_CHECK_NOT_NULL(
        promptVector, "burst_rpc expects a VARCHAR prompt argument");

    rows.applyToSelected([&](vector_size_t row) {
      // One driver, one batch, so the row index doubles as the request id.
      // Carrying it through makes the simulated error messages name the row
      // they came from instead of reporting every failure as row 0.
      if (promptVector->isNullAt(row)) {
        results.emplace_back(
            row,
            folly::makeSemiFuture<RPCResponse>(RPCResponse::failed(
                row, RPCErrorKind::kNullInput, "null_input")));
        return;
      }
      std::string prompt = promptVector->valueAt(row).str();
      results.emplace_back(
          row,
          simulator_->nextCall().deferValue(
              [prompt = std::move(prompt)](RPCErrorKind kind) {
                RPCResponse response;
                if (kind != RPCErrorKind::kNone) {
                  response.setError(kind, "simulated backend overload");
                  return response;
                }
                response.setPayload(makeTextPayload("burst: " + prompt));
                return response;
              }));
    });

    return results;
  }
};

// The BATCH counterpart of BurstRPCFunction. BATCH reserves one rate-limiter
// slot per flushBatch() and drains one batch at a time, so the recovery it
// drives is per batch, not per row -- the reason this needs its own harness
// rather than reusing the PER_ROW burst above.
//
// ResponseSimulator::setErrorBurst fails a contiguous run of call ordinals, and
// nextBatch() consumes one ordinal per row, so the burst still lands on a
// deterministic row range.
class BurstBatchRPCFunction : public BurstFunctionBase {
 public:
  using BurstFunctionBase::BurstFunctionBase;

  std::string name() const override {
    return "burst_batch_rpc";
  }

  // Pure virtual on the base. BATCH never routes here; failing loudly beats
  // returning nothing, which would strand rows and surface far downstream.
  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& /*rows*/,
      const std::vector<VectorPtr>& /*args*/) override {
    VELOX_FAIL("burst_batch_rpc is BATCH-only; dispatchPerRow is unreachable");
  }

  std::vector<vector_size_t> accumulateBatch(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    std::vector<vector_size_t> indices;
    VELOX_CHECK(!args.empty(), "burst_batch_rpc expects one argument");
    auto* promptVector = args[0]->as<SimpleVector<StringView>>();
    VELOX_CHECK_NOT_NULL(
        promptVector, "burst_batch_rpc expects a VARCHAR prompt argument");
    rows.applyToSelected([&](vector_size_t row) {
      indices.push_back(row);
      pending_.push_back(
          promptVector->isNullAt(row) ? "" : promptVector->valueAt(row).str());
    });
    return indices;
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    const auto count = maxRows > 0
        ? std::min<int32_t>(maxRows, static_cast<int32_t>(pending_.size()))
        : static_cast<int32_t>(pending_.size());
    std::vector<std::string> prompts(
        pending_.begin(), pending_.begin() + count);
    pending_.erase(pending_.begin(), pending_.begin() + count);
    // The simulator decides only whether each call fails; the function owns
    // the response it builds from that verdict.
    return simulator_->nextBatch(count).deferValue(
        [prompts = std::move(prompts),
         hintBurstErrors =
             config_.hintBurstErrors](std::vector<RPCErrorKind> kinds) {
          std::vector<RPCResponse> responses;
          responses.reserve(kinds.size());
          for (size_t i = 0; i < kinds.size(); ++i) {
            RPCResponse response;
            // Batch-position rowId: the operator scatters responses by it.
            response.rowId = static_cast<int64_t>(i);
            if (kinds[i] != RPCErrorKind::kNone) {
              response.setError(kinds[i], "simulated backend overload");
              if (hintBurstErrors) {
                response.setCapacityRetry(
                    {.retryAfter = std::chrono::milliseconds(1),
                     .budget = std::chrono::seconds(60)});
              }
            } else {
              response.setPayload(makeTextPayload("burst: " + prompts[i]));
            }
            responses.push_back(std::move(response));
          }
          return responses;
        });
  }

  int32_t pendingBatchSize() const override {
    return static_cast<int32_t>(pending_.size());
  }

 private:
  std::vector<std::string> pending_;
};

// The backend that capacity_step_rpc and capacity_step_hint_rpc call.
constexpr const char* kCapacityStepBackend =
    "layer2.test.capacity.step.backend";

// A backend that serves at most 'capacity' calls at once and rejects the rest
// with kRateLimited. Capacity steps from 'lowCapacity' to 'highCapacity' after
// 'stepAfterSuccesses' calls succeed, like a service whose new copies come up
// during the query.
class CapacityBackend {
 public:
  struct Config {
    // Calls served at once before the step.
    int32_t lowCapacity{3};
    // Calls served at once after the step.
    int32_t highCapacity{3};
    // Successful calls after which capacity steps up. 0 never steps.
    int64_t stepAfterSuccesses{0};
  };

  explicit CapacityBackend(Config config) : config_{config} {}

  // Admits a call for 'prompt' when capacity allows. Counts every attempt,
  // and the rows rejected and not yet served.
  bool tryEnter(const std::string& prompt) {
    std::lock_guard<std::mutex> l(mutex_);
    ++numAttempts_;
    const auto capacity = numSuccesses_ >= config_.stepAfterSuccesses &&
            config_.stepAfterSuccesses > 0
        ? config_.highCapacity
        : config_.lowCapacity;
    if (numInFlight_ >= capacity) {
      rejected_.insert(prompt);
      peakRejected_ =
          std::max(peakRejected_, static_cast<int64_t>(rejected_.size()));
      return false;
    }
    rejected_.erase(prompt);
    ++numInFlight_;
    return true;
  }

  void exit() {
    std::lock_guard<std::mutex> l(mutex_);
    --numInFlight_;
    ++numSuccesses_;
  }

  int64_t numAttempts() const {
    std::lock_guard<std::mutex> l(mutex_);
    return numAttempts_;
  }

  // The most rows rejected and not yet served at once.
  int64_t peakRejected() const {
    std::lock_guard<std::mutex> l(mutex_);
    return peakRejected_;
  }

 private:
  const Config config_;
  mutable std::mutex mutex_;
  int32_t numInFlight_{0};
  int64_t numSuccesses_{0};
  int64_t numAttempts_{0};
  folly::F14FastSet<std::string> rejected_;
  int64_t peakRejected_{0};
};

// PER_ROW function over a CapacityBackend. With 'returnHints' it returns each
// rejection with a retry hint, as a transport does when the operator owns
// capacity retries; otherwise it retries internally while the row holds its
// admission slot.
class CapacityRPCFunction : public AsyncRPCFunction {
 public:
  struct Config {
    // Registered function name.
    std::string name;
    // Name of the shared CapacityBackend; functions with the same name share
    // one backend.
    std::string backend;
    // Capacity of the backend, taken from the function that creates it.
    CapacityBackend::Config capacity;
    // Time each admitted call takes.
    std::chrono::milliseconds latency{20};
    // Wait before retrying a rejected call, in the hint or in place.
    std::chrono::milliseconds retryAfter{5};
    // Budget in the hint: how long the operator retries a row before failing
    // it. In-place retries continue until the backend admits the call.
    std::chrono::milliseconds retryBudget{60'000};
    // Whether rejections are returned with a retry hint instead of retried
    // in place.
    bool returnHints{false};
    // Returned from mayReturnCapacityRetries(). Unset means 'returnHints'.
    std::optional<bool> declareHints{};
  };

  explicit CapacityRPCFunction(Config config) : config_{std::move(config)} {}

  void initialize(
      const core::QueryConfig& /*queryConfig*/,
      const std::vector<TypePtr>& /*inputTypes*/,
      const std::vector<VectorPtr>& /*constantInputs*/,
      RPCStreamingMode /*instruction*/) override {
    backend_ = sharedBackend(config_.backend, config_.capacity);
  }

  std::string name() const override {
    return config_.name;
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  std::string admissionKey() const override {
    return config_.backend;
  }

  bool mayReturnCapacityRetries() const override {
    return config_.declareHints.value_or(config_.returnHints);
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        results;
    DecodedVector prompts(*args[0], rows);
    rows.applyToSelected([&](vector_size_t row) {
      results.emplace_back(
          row,
          attempt(backend_, config_, prompts.valueAt<StringView>(row).str()));
    });
    return results;
  }

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    for (const auto& response : responses) {
      if (response.hasError()) {
        return CongestionSignal::kOverloaded;
      }
    }
    return responses.empty() ? CongestionSignal::kNone
                             : CongestionSignal::kSuccess;
  }

  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    return buildTextOutput(responses, pool);
  }

  // The backend that every function on 'backend' shares, created on first
  // use with 'capacity'.
  static std::shared_ptr<CapacityBackend> sharedBackend(
      const std::string& backend,
      const CapacityBackend::Config& capacity) {
    auto locked = backends().wlock();
    auto& entry = (*locked)[backend];
    if (entry == nullptr) {
      entry = std::make_shared<CapacityBackend>(capacity);
    }
    return entry;
  }

  // The backend shared on 'backend', for assertions.
  static std::shared_ptr<CapacityBackend> backendFor(
      const std::string& backend) {
    return backends().rlock()->at(backend);
  }

  // Forgets every backend, so the next query starts on fresh ones.
  static void resetBackends() {
    backends().wlock()->clear();
  }

 private:
  static folly::Synchronized<
      folly::F14FastMap<std::string, std::shared_ptr<CapacityBackend>>>&
  backends() {
    static folly::Synchronized<
        folly::F14FastMap<std::string, std::shared_ptr<CapacityBackend>>>
        instance;
    return instance;
  }

  static folly::SemiFuture<RPCResponse> attempt(
      std::shared_ptr<CapacityBackend> backend,
      Config config,
      std::string prompt) {
    if (backend->tryEnter(prompt)) {
      return folly::futures::sleep(config.latency)
          .deferValue([backend, prompt](folly::Unit) {
            backend->exit();
            RPCResponse response;
            response.setPayload(makeTextPayload("served:" + prompt));
            return response;
          });
    }
    if (config.returnHints) {
      auto response = RPCResponse::failed(
          0, RPCErrorKind::kRateLimited, "simulated capacity rejection");
      response.setCapacityRetry(
          {.retryAfter = config.retryAfter, .budget = config.retryBudget});
      return folly::makeSemiFuture(std::move(response));
    }
    const auto retryAfter = config.retryAfter;
    return folly::futures::sleep(retryAfter)
        .deferValue([backend = std::move(backend),
                     config = std::move(config),
                     prompt = std::move(prompt)](folly::Unit) mutable {
          return attempt(
              std::move(backend), std::move(config), std::move(prompt));
        });
  }

  const Config config_;
  std::shared_ptr<CapacityBackend> backend_;
};

class RPCAimdUnderLoadTest : public OperatorTestBase {
 protected:
  // Phase sizes (rows). PER_ROW issues one call per row in row order, so call
  // ordinals map 1:1 to row indices: warm-up = [0, kWarmupRows), burst =
  // [kWarmupRows, kWarmupRows + kBurstRows), recovery = the rest.
  static constexpr int32_t kWarmupRows = 64;
  static constexpr int32_t kBurstRows = 96;
  static constexpr int32_t kRecoveryRows = 320;
  static constexpr int32_t kTotalRows =
      kWarmupRows + kBurstRows + kRecoveryRows;

  // Rate-limiter AIMD bounds for the test backend. Keeping the in-flight
  // ceiling (window 32, cap 64) below the burst size guarantees the burst
  // drains in several waves, so the cap shrinks multiple times (64 -> 32 -> 16
  // -> ...) well below its ceiling before recovery.
  static constexpr int64_t kMaxLimit = 64;
  static constexpr int64_t kMinLimit = 4;
  static constexpr int64_t kMaxWindow = 32;
  // Rows sent to a backend whose capacity steps up halfway through.
  static constexpr int32_t kCapacityStepRows = 200;
  // Rows sent to a backend that rejects every call.
  static constexpr int32_t kCapacityExhaustedRows = 8;

  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    registerRPCPlanNodeTranslator();
    AsyncRPCFunctionRegistry::registerFunction(
        "burst_rpc",
        [] {
          return std::make_shared<BurstRPCFunction>(BurstRPCFunction::Config{
              /*burstFirstCall=*/kWarmupRows,
              /*burstLastCall=*/kWarmupRows + kBurstRows,
              /*burstKind=*/RPCErrorKind::kRateLimited});
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "burst_batch_rpc",
        [] {
          return std::make_shared<BurstBatchRPCFunction>(
              BurstRPCFunction::Config{
                  /*burstFirstCall=*/kWarmupRows,
                  /*burstLastCall=*/kWarmupRows + kBurstRows,
                  /*burstKind=*/RPCErrorKind::kRateLimited,
                  /*latency=*/std::chrono::milliseconds(1),
                  /*backend=*/"layer2.test.batch.backend"});
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "backend_error_rpc",
        [] {
          return std::make_shared<BurstRPCFunction>(BurstRPCFunction::Config{
              /*burstFirstCall=*/kWarmupRows,
              /*burstLastCall=*/kWarmupRows + kBurstRows,
              /*burstKind=*/RPCErrorKind::kBackendError,
              /*latency=*/std::chrono::milliseconds(1),
              /*backend=*/"layer2.test.backend.error"});
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "capacity_step_rpc",
        [] {
          return std::make_shared<CapacityRPCFunction>(
              CapacityRPCFunction::Config{
                  .name = "capacity_step_rpc",
                  .backend = kCapacityStepBackend,
                  .capacity =
                      {
                          .lowCapacity = 3,
                          .highCapacity = 10,
                          .stepAfterSuccesses = kCapacityStepRows / 2,
                      },
              });
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "capacity_step_hint_rpc",
        [] {
          return std::make_shared<CapacityRPCFunction>(
              CapacityRPCFunction::Config{
                  .name = "capacity_step_hint_rpc",
                  .backend = kCapacityStepBackend,
                  .capacity =
                      {
                          .lowCapacity = 3,
                          .highCapacity = 10,
                          .stepAfterSuccesses = kCapacityStepRows / 2,
                      },
                  .returnHints = true,
              });
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "capacity_tight_hint_rpc",
        [] {
          return std::make_shared<CapacityRPCFunction>(
              CapacityRPCFunction::Config{
                  .name = "capacity_tight_hint_rpc",
                  .backend = "layer2.test.capacity.tight.backend",
                  .capacity = {.lowCapacity = 1, .highCapacity = 1},
                  .latency = std::chrono::milliseconds(5),
                  .retryAfter = std::chrono::milliseconds(20),
                  .returnHints = true,
              });
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "capacity_undeclared_hint_rpc",
        [] {
          return std::make_shared<CapacityRPCFunction>(
              CapacityRPCFunction::Config{
                  .name = "capacity_undeclared_hint_rpc",
                  .backend = "layer2.test.capacity.undeclared.backend",
                  .capacity = {.lowCapacity = 0, .highCapacity = 0},
                  .returnHints = true,
                  .declareHints = false,
              });
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "burst_batch_hint_rpc",
        [] {
          return std::make_shared<BurstBatchRPCFunction>(
              BurstRPCFunction::Config{
                  .burstFirstCall = kWarmupRows,
                  .burstLastCall = kWarmupRows + kBurstRows,
                  .burstKind = RPCErrorKind::kRateLimited,
                  .backend = "layer2.test.batch.hint.backend",
                  .hintBurstErrors = true,
              });
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "capacity_exhausted_rpc",
        [] {
          return std::make_shared<CapacityRPCFunction>(
              CapacityRPCFunction::Config{
                  .name = "capacity_exhausted_rpc",
                  .backend = "layer2.test.capacity.exhausted.backend",
                  .capacity = {.lowCapacity = 0, .highCapacity = 0},
                  .retryAfter = std::chrono::milliseconds(10),
                  .retryBudget = std::chrono::milliseconds(50),
                  .returnHints = true,
              });
        },
        varcharSignature());
    AsyncRPCFunctionRegistry::registerFunction(
        "backend_error_batch_rpc",
        [] {
          return std::make_shared<BurstBatchRPCFunction>(
              BurstRPCFunction::Config{
                  /*burstFirstCall=*/kWarmupRows,
                  /*burstLastCall=*/kWarmupRows + kBurstRows,
                  /*burstKind=*/RPCErrorKind::kBackendError,
                  /*latency=*/std::chrono::milliseconds(1),
                  /*backend=*/"layer2.test.batch.backend.error"});
        },
        varcharSignature());
  }

  static void TearDownTestCase() {
    OperatorTestBase::TearDownTestCase();
    // Reset MemoryManager to shut down SharedArbitrator executor threads so
    // TSAN does not report threads still running at process exit.
    memory::MemoryManager::testingSetInstance({});
  }

  void TearDown() override {
    // The rate limiter is process-global; reset it so its adaptive state does
    // not leak into other tests.
    RPCRateLimiterRegistry::global().testingReset();
    CapacityRPCFunction::resetBackends();
    OperatorTestBase::TearDown();
  }

  core::PlanNodePtr makeRPCNode(
      const core::PlanNodePtr& source,
      const std::string& functionName) {
    const auto& sourceType = source->outputType();

    // The sole argument is the prompt column, read from the source at runtime.
    std::vector<core::TypedExprPtr> callInputs;
    callInputs.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(
            sourceType->findChild("prompt"), "prompt"));
    auto call = std::make_shared<core::CallTypedExpr>(
        VARCHAR(), std::move(callInputs), functionName);

    auto outputNames = sourceType->names();
    auto outputTypes = sourceType->children();
    outputNames.emplace_back("__rpc_result");
    outputTypes.push_back(VARCHAR());
    auto outputType = ROW(std::move(outputNames), std::move(outputTypes));

    // Default streaming mode is PER_ROW.
    return std::make_shared<core::RPCNode>(
        "rpc-0", source, std::move(call), "__rpc_result", outputType);
  }

  core::PlanNodePtr makeBatchRPCNode(
      const core::PlanNodePtr& source,
      int32_t dispatchBatchSize,
      const std::string& functionName) {
    const auto& sourceType = source->outputType();
    std::vector<core::TypedExprPtr> callInputs;
    callInputs.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(
            sourceType->findChild("prompt"), "prompt"));
    auto call = std::make_shared<core::CallTypedExpr>(
        VARCHAR(), std::move(callInputs), functionName);

    auto outputNames = sourceType->names();
    auto outputTypes = sourceType->children();
    outputNames.emplace_back("__rpc_result");
    outputTypes.push_back(VARCHAR());
    auto outputType = ROW(std::move(outputNames), std::move(outputTypes));

    return std::make_shared<core::RPCNode>(
        "rpc-0",
        source,
        std::move(call),
        "__rpc_result",
        outputType,
        RPCStreamingMode::kBatch,
        dispatchBatchSize);
  }
};

TEST_F(RPCAimdUnderLoadTest, windowAndRateLimiterBackOffThenRecover) {
  // One input vector of distinct prompts reserves the simulator's call
  // ordinals in traversal order, so the burst covers exactly kBurstRows calls.
  std::vector<std::string> storage;
  storage.reserve(kTotalRows);
  for (int32_t i = 0; i < kTotalRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts;
  prompts.reserve(kTotalRows);
  for (const auto& prompt : storage) {
    prompts.emplace_back(prompt);
  }
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  auto plan =
      makeRPCNode(PlanBuilder().values({input}).planNode(), "burst_rpc");

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxLimit)
                    .config("rpc.ratelimiter.min_limit", kMinLimit)
                    .config("rpc.ratelimiter.decrease_factor", "0.5")
                    .config("rpc.congestion.max_window", kMaxWindow)
                    .config("rpc.congestion.min_window", 1)
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
                    .copyResults(pool(), task);

  // The query completes end-to-end: every row yields an output row, and exactly
  // the burst window errors out to NULL (proving the deterministic burst hit
  // the rows we expect and nothing else).
  ASSERT_EQ(result->size(), kTotalRows);
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  int64_t numNulls{0};
  for (vector_size_t i = 0; i < result->size(); ++i) {
    if (results->isNullAt(i)) {
      ++numNulls;
    }
  }
  EXPECT_EQ(numNulls, kBurstRows)
      << "exactly the burst window should error out to NULL";

  auto planStats = toPlanStats(task->taskStats());
  ASSERT_EQ(planStats.count("rpc-0"), 1);
  const auto& customStats = planStats.at("rpc-0").customStats;

  // Returns the stat's summed value, or -1 when the stat was never emitted
  // (several are only emitted when the corresponding event actually happened).
  auto statSum = [&](const std::string& key) -> int64_t {
    auto it = customStats.find(key);
    return it == customStats.end() ? -1 : it->second.sum;
  };

  // (1) The mock injected exactly the burst window of rate-limit errors, and
  // the
  //     operator classified them by typed cause.
  EXPECT_EQ(statSum(RPCOperator::kRpcErrorKindRateLimited), kBurstRows);
  EXPECT_GT(statSum(RPCOperator::kRpcCongestionOverloadShrinks), 0);

  // (2) Overload -> the per-driver congestion window shrank at least once
  //     (this stat is only emitted when numShrinks > 0).
  EXPECT_GT(statSum(RPCOperator::kRpcCongestionShrinks), 0);

  // (3) Overload -> this backend's rate-limiter cap backed off below its
  //     ceiling. rpcRateLimiterMinCap lands on every query, so presence alone
  //     says nothing about backoff; the value below the ceiling is the claim.
  const int64_t minCap = statSum(RPCOperator::kRpcRateLimiterMinCap);
  ASSERT_GT(minCap, 0)
      << "the low-water stat must be emitted, and a zero cap would mean the "
         "backend stalled outright rather than backed off";
  EXPECT_LT(minCap, kMaxLimit) << "cap should have dipped below its ceiling";

  // (4) After the burst, the clean success stream drove AIMD additive recovery:
  //     the final cap climbed back above the low-water mark.
  const int64_t finalCap = statSum(RPCOperator::kRpcRateLimiterCap);
  EXPECT_GT(finalCap, minCap) << "cap should recover above its low-water mark";
}

// A backend serves 3 calls at once, then 10. When the function returns its
// rejections with retry hints, each rejected attempt gives back its slot and
// the shared limit falls toward the backend's capacity, so far fewer attempts
// are rejected than when the function retries in place while the row holds
// its slot.
TEST_F(RPCAimdUnderLoadTest, capacityRetriesBoundAttemptsToCapacity) {
  std::vector<std::string> storage;
  storage.reserve(kCapacityStepRows);
  for (int32_t i = 0; i < kCapacityStepRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts(storage.begin(), storage.end());
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  struct Run {
    int64_t numAttempts{0};
    int64_t numCapacityRetries{0};
    int64_t numBlockedCalls{0};
  };
  auto run = [&](const std::string& functionName) {
    RPCRateLimiterRegistry::global().testingReset();
    CapacityRPCFunction::resetBackends();
    auto plan =
        makeRPCNode(PlanBuilder().values({input}).planNode(), functionName);
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .maxDrivers(1)
                      .config("rpc.ratelimiter.adaptive_enabled", "true")
                      .config("rpc.ratelimiter.max_limit", kMaxWindow)
                      .config("rpc.ratelimiter.min_limit", 1)
                      .config("rpc.ratelimiter.decrease_factor", "0.5")
                      .config("rpc.congestion.max_window", kMaxWindow)
                      .config("rpc.congestion.min_window", 1)
                      .copyResults(pool(), task);
    EXPECT_EQ(result->size(), kCapacityStepRows);
    auto* results = result->childAt(1)->asFlatVector<StringView>();
    for (vector_size_t i = 0; i < result->size(); ++i) {
      EXPECT_FALSE(results->isNullAt(i)) << "row " << i << " failed";
    }
    const auto planStats = toPlanStats(task->taskStats());
    const auto& customStats = planStats.at("rpc-0").customStats;
    const auto it = customStats.find(RPCOperator::kRpcCapacityRetries);
    return Run{
        .numAttempts = CapacityRPCFunction::backendFor(kCapacityStepBackend)
                           ->numAttempts(),
        .numCapacityRetries = it == customStats.end() ? 0 : it->second.sum,
        .numBlockedCalls =
            static_cast<int64_t>(planStats.at("rpc-0").isBlockedTiming.count),
    };
  };

  const auto inPlace = run("capacity_step_rpc");
  const auto readmitted = run("capacity_step_hint_rpc");

  EXPECT_EQ(inPlace.numCapacityRetries, 0);
  EXPECT_GT(readmitted.numCapacityRetries, 0);
  // Every rejected attempt was readmitted through the operator.
  EXPECT_EQ(
      readmitted.numAttempts,
      kCapacityStepRows + readmitted.numCapacityRetries);
  EXPECT_LT(readmitted.numAttempts * 2, inPlace.numAttempts)
      << "in place: " << inPlace.numAttempts
      << ", readmitted: " << readmitted.numAttempts;
  // Rows due for a retry while no slot is free wait for one; they do not
  // wake the driver over and over until a slot frees.
  EXPECT_LE(readmitted.numBlockedCalls, 2 * readmitted.numAttempts)
      << "isBlocked() calls: " << readmitted.numBlockedCalls;
}

// A row whose retries would outlast its budget fails with the rejection it
// last got, kRateLimited, as it does when the function retries in place.
TEST_F(RPCAimdUnderLoadTest, capacityRetriesStopAtBudget) {
  std::vector<std::string> storage;
  storage.reserve(kCapacityExhaustedRows);
  for (int32_t i = 0; i < kCapacityExhaustedRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts(storage.begin(), storage.end());
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});
  auto plan = makeRPCNode(
      PlanBuilder().values({input}).planNode(), "capacity_exhausted_rpc");

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.min_limit", 1)
                    .copyResults(pool(), task);
  ASSERT_EQ(result->size(), kCapacityExhaustedRows);
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < result->size(); ++i) {
    EXPECT_TRUE(results->isNullAt(i));
  }
  const auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at("rpc-0").customStats;
  EXPECT_EQ(
      customStats.at(RPCOperator::kRpcErrorKindRateLimited).sum,
      kCapacityExhaustedRows);
  EXPECT_GT(customStats.at(RPCOperator::kRpcCapacityRetries).sum, 0);
}

// A function that returns hints without declaring them in
// mayReturnCapacityRetries() gets today's behavior: each hinted row fails on
// its first rejection.
TEST_F(RPCAimdUnderLoadTest, undeclaredCapacityRetriesAreIgnored) {
  auto input = makeRowVector(
      {"prompt"}, {makeFlatVector<StringView>({"a", "b", "c", "d"})});
  auto plan = makeRPCNode(
      PlanBuilder().values({input}).planNode(), "capacity_undeclared_hint_rpc");
  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config("rpc.ratelimiter.min_limit", 1)
                    .copyResults(pool(), task);
  ASSERT_EQ(result->size(), 4);
  const auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at("rpc-0").customStats;
  EXPECT_EQ(customStats.at(RPCOperator::kRpcErrorKindRateLimited).sum, 4);
  EXPECT_EQ(customStats.count(RPCOperator::kRpcCapacityRetries), 0);
  EXPECT_EQ(
      CapacityRPCFunction::backendFor("layer2.test.capacity.undeclared.backend")
          ->numAttempts(),
      4);
}

// BATCH keeps the function's own retries: hinted rows in a batch fail with
// their rejection instead of being re-admitted.
TEST_F(RPCAimdUnderLoadTest, batchIgnoresCapacityRetries) {
  std::vector<std::string> storage;
  storage.reserve(kTotalRows);
  for (int32_t i = 0; i < kTotalRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts(storage.begin(), storage.end());
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});
  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(), 16, "burst_batch_hint_rpc");
  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxLimit)
                    .config("rpc.ratelimiter.min_limit", kMinLimit)
                    .copyResults(pool(), task);
  ASSERT_EQ(result->size(), kTotalRows);
  const auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at("rpc-0").customStats;
  EXPECT_EQ(
      customStats.at(RPCOperator::kRpcErrorKindRateLimited).sum, kBurstRows);
  EXPECT_EQ(customStats.count(RPCOperator::kRpcCapacityRetries), 0);
}

// Several drivers share one backend and one limiter; every row is output
// exactly once and succeeds.
TEST_F(RPCAimdUnderLoadTest, capacityRetriesWithSeveralDrivers) {
  constexpr int32_t kRows = 50;
  constexpr int32_t kDrivers = 4;
  std::vector<std::string> storage;
  storage.reserve(kRows);
  for (int32_t i = 0; i < kRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts(storage.begin(), storage.end());
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});
  auto plan = makeRPCNode(
      PlanBuilder().values({input}, /*parallelizable=*/true).planNode(),
      "capacity_step_hint_rpc");
  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(kDrivers)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxWindow)
                    .config("rpc.ratelimiter.min_limit", 1)
                    .copyResults(pool(), task);
  ASSERT_EQ(result->size(), kRows * kDrivers);
  auto* inputs = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < result->size(); ++i) {
    ASSERT_FALSE(results->isNullAt(i)) << "row " << i << " failed";
    EXPECT_EQ(results->valueAt(i).str(), "served:" + inputs->valueAt(i).str());
  }
  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(
      planStats.at("rpc-0")
          .customStats.at(RPCOperator::kRpcCapacityRetries)
          .sum,
      0);
}

// Rows retried after later input batches arrived are dispatched with their own
// batch's arguments: every output row carries its own prompt's result, and
// every prompt comes back exactly once.
TEST_F(RPCAimdUnderLoadTest, capacityRetriesKeepRowIdentityAcrossBatches) {
  constexpr int32_t kBatches = 5;
  constexpr int32_t kRowsPerBatch = 40;
  std::vector<RowVectorPtr> batches;
  std::vector<std::string> storage;
  storage.reserve(kBatches * kRowsPerBatch);
  for (int32_t b = 0; b < kBatches; ++b) {
    std::vector<StringView> prompts;
    for (int32_t i = 0; i < kRowsPerBatch; ++i) {
      storage.push_back(fmt::format("batch_{}_row_{}", b, i));
      prompts.emplace_back(storage.back());
    }
    batches.push_back(
        makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)}));
  }
  auto plan = makeRPCNode(
      PlanBuilder().values(batches).planNode(), "capacity_step_hint_rpc");
  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxWindow)
                    .config("rpc.ratelimiter.min_limit", 1)
                    .copyResults(pool(), task);
  ASSERT_EQ(result->size(), kBatches * kRowsPerBatch);
  auto* inputs = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  folly::F14FastSet<std::string> seen;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    ASSERT_FALSE(results->isNullAt(i)) << "row " << i << " failed";
    EXPECT_EQ(results->valueAt(i).str(), "served:" + inputs->valueAt(i).str());
    EXPECT_TRUE(seen.insert(inputs->valueAt(i).str()).second);
  }
  EXPECT_GT(
      toPlanStats(task->taskStats())
          .at("rpc-0")
          .customStats.at(RPCOperator::kRpcCapacityRetries)
          .sum,
      0);
}

// Rows waiting for a retry keep their input batches. Against a backend serving
// one call at a time, the rows rejected and waiting at once stay within about
// one input batch instead of growing with the input.
TEST_F(RPCAimdUnderLoadTest, capacityRetriesBoundRetainedInput) {
  constexpr int32_t kBatches = 10;
  constexpr int32_t kRowsPerBatch = 20;
  constexpr int32_t kWindow = 4;
  std::vector<RowVectorPtr> batches;
  std::vector<std::string> storage;
  storage.reserve(kBatches * kRowsPerBatch);
  for (int32_t b = 0; b < kBatches; ++b) {
    std::vector<StringView> prompts;
    for (int32_t i = 0; i < kRowsPerBatch; ++i) {
      storage.push_back(fmt::format("batch_{}_row_{}", b, i));
      prompts.emplace_back(storage.back());
    }
    batches.push_back(
        makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)}));
  }
  auto plan = makeRPCNode(
      PlanBuilder().values(batches).planNode(), "capacity_tight_hint_rpc");
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.min_limit", 1)
                    .config("rpc.congestion.max_window", kWindow)
                    .config("rpc.congestion.min_window", 1)
                    .copyResults(pool());
  ASSERT_EQ(result->size(), kBatches * kRowsPerBatch);
  EXPECT_LE(
      CapacityRPCFunction::backendFor("layer2.test.capacity.tight.backend")
          ->peakRejected(),
      kRowsPerBatch + kWindow);
}

TEST_F(RPCAimdUnderLoadTest, perRowBackendErrorsDoNotBackOff) {
  std::vector<std::string> storage;
  storage.reserve(kTotalRows);
  for (int32_t i = 0; i < kTotalRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts;
  prompts.reserve(kTotalRows);
  for (const auto& prompt : storage) {
    prompts.emplace_back(prompt);
  }
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});
  auto plan = makeRPCNode(
      PlanBuilder().values({input}).planNode(), "backend_error_rpc");

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxLimit)
                    .config("rpc.ratelimiter.min_limit", kMinLimit)
                    .config("rpc.ratelimiter.decrease_factor", "0.5")
                    .config("rpc.congestion.max_window", kMaxWindow)
                    .config("rpc.congestion.min_window", 1)
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "1")
                    .copyResults(pool(), task);

  ASSERT_EQ(result->size(), kTotalRows);
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  int64_t numNulls{0};
  for (vector_size_t i = 0; i < result->size(); ++i) {
    numNulls += results->isNullAt(i) ? 1 : 0;
  }
  EXPECT_EQ(numNulls, kBurstRows);

  auto planStats = toPlanStats(task->taskStats());
  ASSERT_EQ(planStats.count("rpc-0"), 1);
  const auto& customStats = planStats.at("rpc-0").customStats;
  auto statSum = [&](const std::string& key) -> int64_t {
    auto it = customStats.find(key);
    return it == customStats.end() ? -1 : it->second.sum;
  };

  EXPECT_EQ(statSum(RPCOperator::kRpcErrorKindBackendError), kBurstRows);
  EXPECT_EQ(statSum(RPCOperator::kRpcCongestionOverloadShrinks), -1);
  EXPECT_EQ(statSum(RPCOperator::kRpcRateLimiterMinCap), kMaxLimit);
  EXPECT_EQ(statSum(RPCOperator::kRpcRateLimiterCap), kMaxLimit);
}

// The BATCH arm of the same closed loop: overload burst -> the backend's cap
// backs off through the real classifier -> a clean batch stream recovers it.
//
// This is the end-to-end complement to
// RPCOperatorTest.batchAimdRecoversPerBatchNotPerRow, which forces the shrink
// by calling onOutcome(kOverload) directly and pins the step size. Here the
// shrink arrives the way production produces one -- rate-limited responses
// classified by the function -- so classify -> onOverload -> onSuccess is
// exercised as a loop rather than as three separate calls.
TEST_F(RPCAimdUnderLoadTest, batchRateLimiterBacksOffThenRecovers) {
  std::vector<std::string> storage;
  storage.reserve(kTotalRows);
  for (int32_t i = 0; i < kTotalRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts;
  prompts.reserve(kTotalRows);
  for (const auto& prompt : storage) {
    prompts.emplace_back(prompt);
  }
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  // Chunks small enough that the burst window spans several flushes, so the
  // cap shrinks more than once, and small enough that many clean batches
  // follow it -- recovery is one step per batch, so it needs batches to count.
  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      /*dispatchBatchSize=*/16,
      "burst_batch_rpc");

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxLimit)
                    .config("rpc.ratelimiter.min_limit", kMinLimit)
                    .config("rpc.ratelimiter.decrease_factor", "0.5")
                    .config("rpc.congestion.max_window", kMaxWindow)
                    .config("rpc.congestion.min_window", 1)
                    .copyResults(pool(), task);

  ASSERT_EQ(result->size(), kTotalRows);

  auto planStats = toPlanStats(task->taskStats());
  ASSERT_EQ(planStats.count("rpc-0"), 1);
  const auto& customStats = planStats.at("rpc-0").customStats;
  auto statSum = [&](const std::string& key) -> int64_t {
    auto it = customStats.find(key);
    return it == customStats.end() ? -1 : it->second.sum;
  };

  // The burst actually landed and was classified by typed cause.
  EXPECT_GT(statSum(RPCOperator::kRpcErrorKindRateLimited), 0)
      << "the burst must reach the operator as rate-limit errors";
  EXPECT_GT(statSum(RPCOperator::kRpcCongestionOverloadShrinks), 0);

  EXPECT_GT(statSum(RPCOperator::kRpcCongestionShrinks), 0)
      << "the per-driver window must shrink on batch overload";

  // Overload drove the backend's cap below its ceiling.
  const int64_t minCap = statSum(RPCOperator::kRpcRateLimiterMinCap);
  ASSERT_GT(minCap, 0)
      << "the low-water stat must be emitted, and a zero cap would mean the "
         "backend stalled outright rather than backed off";
  EXPECT_LT(minCap, kMaxLimit) << "cap should have dipped below its ceiling";

  // And the clean batches after the burst recovered it. This is the assertion
  // the unit-level test cannot make: recovery here is credited by the operator
  // through the real congestion classifier, one unit per drained batch.
  const int64_t finalCap = statSum(RPCOperator::kRpcRateLimiterCap);
  EXPECT_GT(finalCap, minCap) << "cap should recover above its low-water mark";
}

TEST_F(RPCAimdUnderLoadTest, batchBackendErrorsDoNotBackOff) {
  std::vector<std::string> storage;
  storage.reserve(kTotalRows);
  for (int32_t i = 0; i < kTotalRows; ++i) {
    storage.push_back("row_" + std::to_string(i));
  }
  std::vector<StringView> prompts;
  prompts.reserve(kTotalRows);
  for (const auto& prompt : storage) {
    prompts.emplace_back(prompt);
  }
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});
  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      /*dispatchBatchSize=*/16,
      "backend_error_batch_rpc");

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .config("rpc.ratelimiter.adaptive_enabled", "true")
                    .config("rpc.ratelimiter.max_limit", kMaxLimit)
                    .config("rpc.ratelimiter.min_limit", kMinLimit)
                    .config("rpc.ratelimiter.decrease_factor", "0.5")
                    .config("rpc.congestion.max_window", kMaxWindow)
                    .config("rpc.congestion.min_window", 1)
                    .copyResults(pool(), task);

  ASSERT_EQ(result->size(), kTotalRows);
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  int64_t numNulls{0};
  for (vector_size_t i = 0; i < result->size(); ++i) {
    numNulls += results->isNullAt(i) ? 1 : 0;
  }
  EXPECT_EQ(numNulls, kBurstRows);

  auto planStats = toPlanStats(task->taskStats());
  ASSERT_EQ(planStats.count("rpc-0"), 1);
  const auto& customStats = planStats.at("rpc-0").customStats;
  auto statSum = [&](const std::string& key) -> int64_t {
    auto it = customStats.find(key);
    return it == customStats.end() ? -1 : it->second.sum;
  };

  EXPECT_EQ(statSum(RPCOperator::kRpcErrorKindBackendError), kBurstRows);
  EXPECT_EQ(statSum(RPCOperator::kRpcCongestionOverloadShrinks), -1);
  EXPECT_EQ(statSum(RPCOperator::kRpcRateLimiterMinCap), kMaxLimit);
  EXPECT_EQ(statSum(RPCOperator::kRpcRateLimiterCap), kMaxLimit);
}

} // namespace
} // namespace facebook::velox::exec::rpc
