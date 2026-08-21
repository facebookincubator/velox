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
/// ResponseSimulator::ErrorBurst), and PER_ROW dispatch preserves row order, so
/// the burst lands on a known, timing-independent slice of rows. The whole
/// trajectory is captured in a single close()-time stats snapshot: numShrinks>0
/// proves the window dipped; rpcRateLimiterMinCap (only emitted once the cap
/// shrank) proves the limiter backed off; the final cap sitting above that
/// low-water mark proves recovery.

#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/rpc/RPCOperator.h"
#include "velox/exec/rpc/RPCPlanNodeTranslator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
#include "velox/exec/rpc/tests/ResponseSimulator.h"
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

// A PER_ROW RPC function backed by a ResponseSimulator that rejects a
// deterministic window of requests with rate-limit errors. It classifies
// rate-limit/timeout failures as backend overload (kOverloaded -> both
// controllers back off) and clean drains as kSuccess (feed the RTT gradient /
// drive rate-limiter recovery), exactly as a production congestion policy
// would.
class BurstRPCFunction : public AsyncRPCFunction {
 public:
  struct Config {
    // Requests with call ordinal in [burstFirstCall, burstLastCall) are failed.
    int64_t burstFirstCall{0};
    int64_t burstLastCall{0};
    RPCErrorKind burstKind{RPCErrorKind::kRateLimited};
    std::chrono::milliseconds latency{std::chrono::milliseconds(1)};
    // Non-empty so the process-global rate limiter is keyed on a real tier.
    std::string tier{"layer2.test.tier"};
  };

  explicit BurstRPCFunction(Config config) : config_{std::move(config)} {}

  void initialize(
      const core::QueryConfig& /*queryConfig*/,
      const std::vector<TypePtr>& /*inputTypes*/,
      const std::vector<VectorPtr>& /*constantInputs*/,
      velox::rpc::RPCStreamingMode /*mode*/) override {
    // Call ordinals are what pin the burst to rows [kWarmupRows,
    // kWarmupRows + kBurstRows). A second initialize() would install a fresh
    // client whose ordinals restart at 0 and silently move the burst, so
    // require exactly one.
    VELOX_CHECK_NULL(simulator_, "initialize() must be called at most once");
    simulator_ =
        std::make_shared<ResponseSimulator>(config_.latency, /*errorRate=*/0.0);
    simulator_->setErrorBurst(
        {config_.burstFirstCall, config_.burstLastCall, config_.burstKind});
  }

  std::string name() const override {
    return "burst_rpc";
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  /// Per-row only; the AIMD-under-load test drives the per-row dispatch path.
  RpcCapability capabilities() const override {
    return {};
  }

  std::string tierKey() const override {
    return config_.tier;
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
            folly::makeSemiFuture<RPCResponse>(RPCResponse{
                .rowId = row,
                .error = "null_input",
                .errorKind = RPCErrorKind::kNullInput}));
        return;
      }
      std::string prompt = promptVector->valueAt(row).str();
      results.emplace_back(
          row,
          simulator_->nextCall().deferValue(
              [prompt = std::move(prompt)](RPCErrorKind kind) {
                RPCResponse response;
                if (kind != RPCErrorKind::kNone) {
                  response.error = "simulated backend overload";
                  response.errorKind = kind;
                  return response;
                }
                response.payload = makeTextPayload("burst: " + prompt);
                return response;
              }));
    });

    return results;
  }

  // Overload classifier: rate-limit / timeout failures are backend overload
  // (kOverloaded). A null-input error is a user error and must NOT move the
  // window (folded into kSuccess/kNone below since it is not
  // rate-limit/timeout). A clean drain feeds its RTT to the gradient and drives
  // rate-limiter recovery.
  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& responses) const override {
    for (const auto& response : responses) {
      if (response.hasError() &&
          (response.errorKind == RPCErrorKind::kRateLimited ||
           response.errorKind == RPCErrorKind::kTimeout)) {
        return CongestionSignal::kOverloaded;
      }
    }
    return responses.empty() ? CongestionSignal::kNone
                             : CongestionSignal::kSuccess;
  }

 private:
  Config config_;
  std::shared_ptr<ResponseSimulator> simulator_;

  VectorPtr buildOutput(
      const std::vector<RPCResponse>& responses,
      memory::MemoryPool* pool) const override {
    return buildTextOutput(responses, pool);
  }
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

  // Rate-limiter AIMD bounds for the test tier. Keeping the in-flight ceiling
  // (window 32, cap 64) below the burst size guarantees the burst drains in
  // several waves, so the cap shrinks multiple times (64 -> 32 -> 16 -> ...)
  // well below its ceiling before recovery.
  static constexpr int64_t kMaxLimit = 64;
  static constexpr int64_t kMinLimit = 4;
  static constexpr int64_t kMaxWindow = 32;

  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    registerRPCPlanNodeTranslator();
    AsyncRPCFunctionRegistry::registerFunction("burst_rpc", [] {
      return std::make_shared<BurstRPCFunction>(
          BurstRPCFunction::Config{/*burstFirstCall=*/kWarmupRows,
                                   /*burstLastCall=*/kWarmupRows + kBurstRows,
                                   /*burstKind=*/RPCErrorKind::kRateLimited});
    });
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
    RPCRateLimiter::testingResetAllState();
    OperatorTestBase::TearDown();
  }

  core::PlanNodePtr makeRPCNode(const core::PlanNodePtr& source) {
    const auto& sourceType = source->outputType();

    // The sole argument is the prompt column, read from the source at runtime.
    std::vector<core::TypedExprPtr> callInputs;
    callInputs.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(
            sourceType->findChild("prompt"), "prompt"));
    auto call = std::make_shared<core::CallTypedExpr>(
        VARCHAR(), std::move(callInputs), "burst_rpc");

    auto outputNames = sourceType->names();
    auto outputTypes = sourceType->children();
    outputNames.emplace_back("__rpc_result");
    outputTypes.push_back(VARCHAR());
    auto outputType = ROW(std::move(outputNames), std::move(outputTypes));

    // Default streaming mode is PER_ROW.
    return std::make_shared<core::RPCNode>(
        "rpc-0", source, std::move(call), "__rpc_result", outputType);
  }
};

TEST_F(RPCAimdUnderLoadTest, windowAndRateLimiterBackOffThenRecover) {
  // One input vector of distinct prompts. PER_ROW dispatch preserves row order,
  // so the burst lands exactly on rows [kWarmupRows, kWarmupRows + kBurstRows).
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

  auto plan = makeRPCNode(PlanBuilder().values({input}).planNode());

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

  // (2) Overload -> the per-driver congestion window shrank at least once
  //     (this stat is only emitted when numShrinks > 0).
  EXPECT_GT(statSum(RPCOperator::kRpcCongestionShrinks), 0);

  // (3) Overload -> the process-global rate-limiter cap backed off below its
  //     ceiling. rpcRateLimiterMinCap is only emitted once the cap shrank.
  const int64_t minCap = statSum(RPCOperator::kRpcRateLimiterMinCap);
  EXPECT_GT(minCap, 0);
  EXPECT_LT(minCap, kMaxLimit) << "cap should have dipped below its ceiling";

  // (4) After the burst, the clean success stream drove AIMD additive recovery:
  //     the final cap climbed back above the low-water mark.
  const int64_t finalCap = statSum(RPCOperator::kRpcRateLimiterCap);
  EXPECT_GT(finalCap, minCap) << "cap should recover above its low-water mark";
}

} // namespace
} // namespace facebook::velox::exec::rpc
