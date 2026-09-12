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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/rpc/RPCOperator.h"
#include "velox/exec/rpc/RPCPlanNodeTranslator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
#include "velox/exec/rpc/tests/DemoBatchRPCFunction.h"
#include "velox/exec/rpc/tests/DemoRPCFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>

#include <chrono>

#include "velox/exec/Task.h"

namespace facebook::velox::exec::rpc {

using namespace facebook::velox::exec::test;

class RPCOperatorTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    registerRPCPlanNodeTranslator();
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_rpc", []() { return std::make_shared<DemoAsyncRPCFunction>(); });
    AsyncRPCFunctionRegistry::registerFunction("demo_batch_rpc_held", []() {
      auto fn = std::make_shared<DemoBatchRPCFunction>();
      fn->testingHoldFlushes();
      return fn;
    });
    AsyncRPCFunctionRegistry::registerFunction("demo_batch_rpc", []() {
      return std::make_shared<DemoBatchRPCFunction>();
    });
    AsyncRPCFunctionRegistry::registerFunction("demo_batch_rpc_reversed", []() {
      return std::make_shared<DemoBatchRPCFunction>(
          DemoBatchRPCFunction::ResponseOrder::kReversed);
    });
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_partial_fail", []() {
          return std::make_shared<DemoBatchRPCFunction>(
              DemoBatchRPCFunction::ResponseOrder::kInOrder,
              std::unordered_set<int32_t>{1, 3});
        });
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_whole_fail", []() {
          return std::make_shared<DemoBatchRPCFunction>(
              DemoBatchRPCFunction::ResponseOrder::kInOrder,
              std::unordered_set<int32_t>{},
              /*failWholeBatch=*/true);
        });
    // Whole-batch failure AND a fail-on-error policy (mimics
    // meta_ai_on_error='fail'): the query must still hard-fail, not degrade.
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_whole_fail_strict", []() {
          return std::make_shared<DemoBatchRPCFunction>(
              DemoBatchRPCFunction::ResponseOrder::kInOrder,
              std::unordered_set<int32_t>{},
              /*failWholeBatch=*/true,
              /*failOnError=*/true);
        });
    // Returns fewer responses than rows (function-contract violation): the
    // operator's scatter must hard-fail on the count mismatch.
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_wrong_count", []() {
          return std::make_shared<DemoBatchRPCFunction>(
              DemoBatchRPCFunction::ResponseOrder::kInOrder,
              std::unordered_set<int32_t>{},
              /*failWholeBatch=*/false,
              /*failOnError=*/false,
              /*dropOneResponse=*/true);
        });
  }

  static void TearDownTestCase() {
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

  // Drives a query whose tier is fully held by the test body until a
  // background thread releases it. Defined below the fixture.
  void runContendedDrain(bool batchMode);

  /// Build a BATCH-mode RPCNode on top of a source plan node.
  core::PlanNodePtr makeBatchRPCNode(
      const core::PlanNodePtr& source,
      const std::vector<std::string>& argumentColumnNames,
      const std::string& functionName = "demo_batch_rpc",
      int32_t dispatchBatchSize = 0) {
    auto sourceType = source->outputType();

    std::vector<core::TypedExprPtr> callInputs;
    callInputs.reserve(argumentColumnNames.size());
    for (const auto& colName : argumentColumnNames) {
      callInputs.push_back(
          std::make_shared<core::FieldAccessTypedExpr>(
              sourceType->findChild(colName), colName));
    }
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

  /// Build a PER_ROW-mode RPCNode on top of a source plan node.
  /// argumentColumnNames specifies which source columns are RPC arguments.
  core::PlanNodePtr makeRPCNode(
      const core::PlanNodePtr& source,
      const std::vector<std::string>& argumentColumnNames) {
    auto sourceType = source->outputType();

    std::vector<core::TypedExprPtr> callInputs;
    callInputs.reserve(argumentColumnNames.size());
    for (const auto& colName : argumentColumnNames) {
      // Variable (column) argument, not a constant.
      callInputs.push_back(
          std::make_shared<core::FieldAccessTypedExpr>(
              sourceType->findChild(colName), colName));
    }
    auto call = std::make_shared<core::CallTypedExpr>(
        VARCHAR(), std::move(callInputs), "demo_rpc");

    // Output type = all source columns + RPC result column.
    auto outputNames = sourceType->names();
    auto outputTypes = sourceType->children();
    outputNames.emplace_back("__rpc_result");
    outputTypes.push_back(VARCHAR());
    auto outputType = ROW(std::move(outputNames), std::move(outputTypes));

    return std::make_shared<core::RPCNode>(
        "rpc-0", source, std::move(call), "__rpc_result", outputType);
  }
};

/// Runs Values(3 rows) → RPCNode → verifies passthrough + RPC result.
TEST_F(RPCOperatorTest, basicPerRow) {
  auto input = makeRowVector(
      {"prompt"},
      {makeFlatVector<StringView>(
          {"hello world", "test prompt", "third row"})});

  auto plan = makeRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 3);
  ASSERT_EQ(result->type()->size(), 2); // prompt + __rpc_result

  // Rows may arrive out of order (async dispatch). Collect and sort to verify.
  auto* prompts = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();

  std::map<std::string, std::string> rows;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    rows[prompts->valueAt(i).str()] = results->valueAt(i).str();
  }

  EXPECT_EQ(rows["hello world"], "Response for: hello world");
  EXPECT_EQ(rows["test prompt"], "Response for: test prompt");
  EXPECT_EQ(rows["third row"], "Response for: third row");
}

// kPerRow output is sized from QueryConfig::preferredOutputBatchRows: 50 rows
// with a cap of 10 emit at least 5 output vectors.
TEST_F(RPCOperatorTest, outputBatchSizeFromConfig) {
  auto input =
      makeRowVector({"prompt"}, {makeFlatVector<std::string>(50, [](auto row) {
                      return fmt::format("row {}", row);
                    })});
  auto plan = makeRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  std::shared_ptr<exec::Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "10")
                    .copyResults(pool(), task);
  EXPECT_EQ(result->size(), 50);

  // 50 rows capped at 10 per output vector => at least 5 vectors.
  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GE(planStats.at(plan->id()).outputVectors, 5);
}

/// Null input rows should produce null in the RPC result column.
TEST_F(RPCOperatorTest, nullInput) {
  auto promptVector =
      makeNullableFlatVector<StringView>({"valid prompt", std::nullopt});
  auto input = makeRowVector({"prompt"}, {promptVector});

  auto plan = makeRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 2);

  auto* prompts = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();

  // Find which row is the valid one vs the null one.
  for (vector_size_t i = 0; i < result->size(); ++i) {
    if (prompts->isNullAt(i)) {
      // Null input row should produce null result.
      EXPECT_TRUE(results->isNullAt(i));
    } else {
      EXPECT_EQ(prompts->valueAt(i).str(), "valid prompt");
      EXPECT_FALSE(results->isNullAt(i));
      EXPECT_EQ(results->valueAt(i).str(), "Response for: valid prompt");
    }
  }
}

/// Multiple source columns — verifies all passthrough columns are preserved.
TEST_F(RPCOperatorTest, multipleColumns) {
  auto input = makeRowVector(
      {"id", "prompt", "extra"},
      {makeFlatVector<int64_t>({100, 200}),
       makeFlatVector<StringView>({"question one", "question two"}),
       makeFlatVector<double>({1.5, 2.5})});

  // Only "prompt" is an RPC argument; "id" and "extra" are passthrough.
  auto plan = makeRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 2);
  ASSERT_EQ(result->type()->size(), 4); // id, prompt, extra, __rpc_result

  // Rows may arrive out of order. Index by prompt to verify.
  auto* prompts = result->childAt(1)->asFlatVector<StringView>();
  auto* ids = result->childAt(0)->asFlatVector<int64_t>();
  auto* extras = result->childAt(2)->asFlatVector<double>();
  auto* results = result->childAt(3)->asFlatVector<StringView>();

  std::map<std::string, vector_size_t> rowIndex;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    rowIndex[prompts->valueAt(i).str()] = i;
  }

  auto i1 = rowIndex["question one"];
  EXPECT_EQ(ids->valueAt(i1), 100);
  EXPECT_EQ(extras->valueAt(i1), 1.5);
  EXPECT_EQ(results->valueAt(i1).str(), "Response for: question one");

  auto i2 = rowIndex["question two"];
  EXPECT_EQ(ids->valueAt(i2), 200);
  EXPECT_EQ(extras->valueAt(i2), 2.5);
  EXPECT_EQ(results->valueAt(i2).str(), "Response for: question two");
}

// ============================================================
// BATCH mode tests — exercise accumulateBatch/flushBatch path
// ============================================================

// Basic batch mode: responses in order, verify passthrough + result.
TEST_F(RPCOperatorTest, batchBasic) {
  auto input = makeRowVector(
      {"prompt"}, {makeFlatVector<StringView>({"hello", "world", "batch"})});

  auto plan =
      makeBatchRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());

  ASSERT_EQ(result->size(), 3);

  auto* prompts = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();

  std::map<std::string, std::string> rows;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    rows[prompts->valueAt(i).str()] = results->valueAt(i).str();
  }

  EXPECT_EQ(rows["hello"], "Batch response for: hello");
  EXPECT_EQ(rows["world"], "Batch response for: world");
  EXPECT_EQ(rows["batch"], "Batch response for: batch");
}

// A backend is configured by the first query to reach it, and every query
// after shares that configuration. The limiter is a controller: its policy has
// to hold still while it adapts, so a later query cannot move the target
// mid-flight. Before this, initialize() rewrote the policy on every query --
// a pipeline that lowered the floor to match a small entitlement had it reset
// to 50 by the next default query, with nothing logged.
TEST_F(RPCOperatorTest, backendIsConfiguredByTheFirstQueryOnly) {
  auto input =
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b"})});
  auto& limiter = RPCRateLimiterRegistry::global().get("");

  // First query: a deliberate policy, as a small-entitlement pipeline sets.
  limiter.initializeOnce([](RPCRateLimiter::Config& config) {
    config.adaptive = true;
    config.floor = 4;
    config.decreaseFactor = 0.25;
  });

  // A later query on the same backend, carrying the defaults.
  auto plan =
      makeBatchRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});
  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
  ASSERT_EQ(result->size(), 2);

  const auto config = limiter.config();
  EXPECT_TRUE(config.adaptive);
  EXPECT_EQ(config.floor, 4) << "a later query reconfigured the backend";
  EXPECT_DOUBLE_EQ(config.decreaseFactor, 0.25);
}

// Dispatch must respect the tier's admission cap, not only the per-driver
// window. Other drivers can exhaust the tier while this driver's window is
// still open, and an ungated flush loop then pushes pending past the cap.
// The mock holds each flush open so several are genuinely in flight; the tier
// ceiling is set below the BATCH window's starting value so the tier is the
// binding constraint and the two gates are distinguishable.
// Intake is bounded by accumulator depth, and BATCH flushes from isBlocked()
// as well as addInput(). Both halves are needed: bounding intake without the
// flush from isBlocked() lets a full accumulator sit forever once needsInput()
// stops taking input, because BATCH has no other place to drain from -- the
// query then hangs rather than fails. Feeds many input vectors so needsInput()
// is actually consulted mid-stream, against a tier admitting one flush.
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
        makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)}));
  }

  auto& limiter = RPCRateLimiterRegistry::global().get("");
  limiter.initializeOnce(
      [](RPCRateLimiter::Config& config) { config.ceiling = kCeiling; });

  auto plan = makeBatchRPCNode(
      PlanBuilder().values(inputs).planNode(),
      {"prompt"},
      "demo_batch_rpc_held",
      /*dispatchBatchSize=*/4);
  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());

  // Every row comes out: throttling intake never stranded the accumulator.
  ASSERT_EQ(result->size(), kVectors * kRowsPerVector);
  EXPECT_LE(limiter.stats().peakPending, kCeiling);
}

// Contended admission: the tier's slots are held by someone else, so a refused
// dispatch meets numInFlight() == 0 -- the state the other operator tests never
// reach, since they run one driver against a tier nothing else holds. The test
// body holds the ceiling itself and releases from a background thread, which is
// the only way to produce a refusal this operator cannot resolve alone.
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
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  auto& limiter = RPCRateLimiterRegistry::global().get("");
  limiter.initializeOnce(
      [](RPCRateLimiter::Config& config) { config.ceiling = kCeiling; });

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
      ? makeBatchRPCNode(
            PlanBuilder().values({input}).planNode(),
            {"prompt"},
            "demo_batch_rpc",
            /*dispatchBatchSize=*/8)
      : makeRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});
  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
  releaser.join();

  // Nothing dropped while admission held the rows, and nothing hung.
  ASSERT_EQ(result->size(), kRows);
}

TEST_F(RPCOperatorTest, perRowDrainCompletesWhileTheTierIsHeld) {
  runContendedDrain(/*batchMode=*/false);
}

TEST_F(RPCOperatorTest, batchDrainCompletesWhileTheTierIsHeld) {
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
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b", "c"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "no_such_rpc_function_registered");

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool()),
      "Unknown RPC function");
}

TEST_F(RPCOperatorTest, batchDispatchRespectsTheTierCap) {
  constexpr int64_t kCeiling = 1;
  std::vector<std::string> storage;
  std::vector<StringView> prompts;
  storage.reserve(64);
  prompts.reserve(64);
  for (int i = 0; i < 64; ++i) {
    storage.push_back(fmt::format("row-{}", i));
    prompts.emplace_back(storage.back());
  }
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  auto& limiter = RPCRateLimiterRegistry::global().get("");
  limiter.initializeOnce(
      [](RPCRateLimiter::Config& config) { config.ceiling = kCeiling; });

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_held",
      /*dispatchBatchSize=*/4);
  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
  ASSERT_EQ(result->size(), 64);

  EXPECT_LE(limiter.stats().peakPending, kCeiling)
      << "dispatch admitted more than the tier cap";
}

// BATCH reserves one slot per flushBatch() regardless of row count, so its
// AIMD recovery has to be credited in batches too. onSuccess() steps capacity
// by units/capacity, so crediting the row count against a capacity counted in
// batches makes the step grow as capacity shrinks -- recovery accelerates
// exactly when it should be most cautious, undoing the multiplicative
// decrease.
//
// Four batches of 16 rows drain successfully from a capacity of 2. Credited in
// batches the capacity walks 2 -> 6, one step per batch. Credited in rows it
// jumps 2 -> 10 on the first batch alone (16/2 = 8) and lands at 13.
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
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  auto& limiter = RPCRateLimiterRegistry::global().get("");
  limiter.initializeOnce([](RPCRateLimiter::Config& config) {
    config.ceiling = kCeiling;
    config.adaptive = true;
    // Below the built-in floor of 50, so capacity can be driven low enough
    // for the two crediting schemes to separate.
    config.floor = 1;
    config.decreaseFactor = 0.5;
  });

  // 64 -> 32 -> 16 -> 8 -> 4 -> 2.
  for (int i = 0; i < 5; ++i) {
    limiter.onOutcome(RPCRateLimiter::Outcome::kOverload, /*units=*/0);
  }
  const int64_t shrunk = limiter.stats().capacity;
  ASSERT_EQ(shrunk, 2) << "precondition: capacity must start well below the "
                          "rows per batch for the two schemes to differ";

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc",
      kDispatchBatchSize);
  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
  ASSERT_EQ(result->size(), kRows);

  EXPECT_EQ(limiter.stats().capacity, shrunk + kNumBatches)
      << "capacity recovered by " << (limiter.stats().capacity - shrunk)
      << " over " << kNumBatches
      << " successful batches; additive increase on a batch-denominated "
         "capacity must be one step per batch";
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
  auto input = makeRowVector({"prompt"}, {makeFlatVector<StringView>(prompts)});

  auto& limiter = RPCRateLimiterRegistry::global().get("");
  limiter.initializeOnce(
      [](RPCRateLimiter::Config& config) { config.ceiling = kCeiling; });

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(), {"prompt"}, "demo_batch_rpc");
  std::shared_ptr<exec::Task> task;
  auto result =
      AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool(), task);
  ASSERT_EQ(result->size(), 8);

  // Nothing drove an overload, so the cap never shrank.
  ASSERT_EQ(limiter.stats().capacity, kCeiling);

  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at("rpc-0").customStats;
  ASSERT_EQ(customStats.count(RPCOperator::kRpcRateLimiterMinCap), 1)
      << "the low-water stat must land even when the backend never shrank";
  EXPECT_EQ(customStats.at(RPCOperator::kRpcRateLimiterMinCap).sum, kCeiling);
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

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_reversed");

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 5);

  auto* ids = result->childAt(0)->asFlatVector<int64_t>();
  auto* prompts = result->childAt(1)->asFlatVector<StringView>();
  auto* results = result->childAt(2)->asFlatVector<StringView>();

  std::map<int64_t, std::pair<std::string, std::string>> rowMap;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    rowMap[ids->valueAt(i)] = {
        prompts->valueAt(i).str(), results->valueAt(i).str()};
  }

  EXPECT_EQ(rowMap[1].second, "Batch response for: alpha");
  EXPECT_EQ(rowMap[2].second, "Batch response for: bravo");
  EXPECT_EQ(rowMap[3].second, "Batch response for: charlie");
  EXPECT_EQ(rowMap[4].second, "Batch response for: delta");
  EXPECT_EQ(rowMap[5].second, "Batch response for: echo");
}

// Partial batch failure: rows at indices 1 and 3 fail, others succeed.
// Verifies that failed rows produce NULL results while successful rows
// are correctly mapped to their prompts.
TEST_F(RPCOperatorTest, batchPartialFailure) {
  auto input = makeRowVector(
      {"prompt"},
      {makeFlatVector<StringView>(
          {"row0", "row1_fail", "row2", "row3_fail", "row4"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_partial_fail");

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 5);

  auto* prompts = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();

  std::map<std::string, vector_size_t> rowIndex;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    rowIndex[prompts->valueAt(i).str()] = i;
  }

  EXPECT_FALSE(results->isNullAt(rowIndex["row0"]));
  EXPECT_EQ(
      results->valueAt(rowIndex["row0"]).str(), "Batch response for: row0");
  EXPECT_TRUE(results->isNullAt(rowIndex["row1_fail"]));
  EXPECT_FALSE(results->isNullAt(rowIndex["row2"]));
  EXPECT_TRUE(results->isNullAt(rowIndex["row3_fail"]));
  EXPECT_FALSE(results->isNullAt(rowIndex["row4"]));
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
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b", "c"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_whole_fail");

  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());

  ASSERT_EQ(result->size(), 3);
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < result->size(); ++i) {
    EXPECT_TRUE(results->isNullAt(i))
        << "row " << i << " should degrade to NULL on whole-batch failure";
  }
}

// With a fail-on-error policy (meta_ai_on_error='fail'), a whole-batch failure
// must still HARD-FAIL the query — the degrade-to-per-row change must not
// silently turn a 'fail' request into all-NULL. The per-row errors produced by
// the operator flow to the function's buildOutput, which fails the query.
TEST_F(RPCOperatorTest, batchWholeBatchFailureWithFailPolicyThrows) {
  auto input =
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b", "c"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_whole_fail_strict");

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool()),
      "RPC call failed for row");
}

// A function that returns fewer responses than rows violates the batch
// contract. After moving deferError before deferValue, the scatter's
// count-mismatch check must STILL hard-fail the query (not be swallowed and
// degraded to NULL rows).
TEST_F(RPCOperatorTest, batchWrongResponseCountHardFails) {
  auto input =
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b", "c"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_wrong_count");

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool()),
      "does not match row count");
}

// Null inputs in batch mode produce null results.
TEST_F(RPCOperatorTest, batchNullInput) {
  auto input = makeRowVector(
      {"prompt"},
      {makeNullableFlatVector<StringView>(
          {"valid1"_sv, std::nullopt, "valid2"_sv, std::nullopt})});

  auto plan =
      makeBatchRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 4);

  auto* prompts = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();

  for (vector_size_t i = 0; i < result->size(); ++i) {
    if (prompts->isNullAt(i)) {
      EXPECT_TRUE(results->isNullAt(i));
    } else {
      EXPECT_FALSE(results->isNullAt(i));
      auto prompt = prompts->valueAt(i).str();
      EXPECT_EQ(results->valueAt(i).str(), "Batch response for: " + prompt);
    }
  }
}

// Multiple input batches: two separate addInput() calls, both processed
// correctly in batch mode.
TEST_F(RPCOperatorTest, batchMultipleInputBatches) {
  auto batch1 =
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b", "c"})});
  auto batch2 =
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"d", "e"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({batch1, batch2}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 5);

  auto* results = result->childAt(1)->asFlatVector<StringView>();

  std::set<std::string> resultSet;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    EXPECT_FALSE(results->isNullAt(i));
    resultSet.insert(results->valueAt(i).str());
  }

  EXPECT_EQ(resultSet.count("Batch response for: a"), 1);
  EXPECT_EQ(resultSet.count("Batch response for: b"), 1);
  EXPECT_EQ(resultSet.count("Batch response for: c"), 1);
  EXPECT_EQ(resultSet.count("Batch response for: d"), 1);
  EXPECT_EQ(resultSet.count("Batch response for: e"), 1);
}

// Pipelined batch dispatch: with dispatchBatchSize=2, the operator flushes
// mid-addInput() instead of waiting for noMoreInput(). Verifies all rows
// are still accounted for.
TEST_F(RPCOperatorTest, batchPipelinedDispatch) {
  auto input = makeRowVector(
      {"prompt"},
      {makeFlatVector<StringView>({"p1", "p2", "p3", "p4", "p5", "p6", "p7"})});

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc",
      /*dispatchBatchSize=*/2);

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 7);

  auto* results = result->childAt(1)->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < result->size(); ++i) {
    EXPECT_FALSE(results->isNullAt(i));
  }
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

  void initialize(
      const core::QueryConfig&,
      const std::vector<TypePtr>&,
      const std::vector<VectorPtr>&) override {}

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
      response.result = "ok";
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
  AsyncRPCFunctionRegistry::registerFunction(
      "slow_batch_rpc", [kLatency, rpcExecutor]() {
        return std::make_shared<SlowBatchRPCFunction>(kLatency, rpcExecutor);
      });

  constexpr int kRows = 8;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kRows);
  for (int i = 0; i < kRows; ++i) {
    inputs.push_back(makeRowVector(
        {"prompt"}, {makeFlatVector<StringView>({StringView("hi")})}));
  }

  auto plan = makeBatchRPCNode(
      PlanBuilder().values(inputs).planNode(),
      {"prompt"},
      "slow_batch_rpc",
      /*dispatchBatchSize=*/1);

  std::shared_ptr<exec::Task> task;
  auto result = AssertQueryBuilder(plan).copyResults(pool(), task);
  ASSERT_EQ(result->size(), kRows);

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

/// PER_ROW congestion path. On the function's overload verdict
/// (evaluateCongestion -> kError) both AIMD controllers back off: the
/// per-driver window (onUnitError) and the process-global rate limiter
/// (onRateLimited); on kSuccess the window's latency gradient is fed. Verifies
/// the query still completes correctly through that path. The controllers'
/// adjustments are unit-tested in RPCStateTest / RPCRateLimiterTest; here we
/// guard the operator-level materialization + signal plumbing against
/// crashes/regressions.
TEST_F(RPCOperatorTest, perRowCongestionPath) {
  // DemoAsyncRPCFunction::evaluateCongestion returns kError when a response
  // result contains "OVERLOAD" (the mock echoes the prompt into the result).
  auto input = makeRowVector(
      {"prompt"},
      {makeFlatVector<StringView>(
          {"OVERLOAD one", "OVERLOAD two", "normal three"})});

  auto plan = makeRPCNode(PlanBuilder().values({input}).planNode(), {"prompt"});

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 3);
  auto* prompts = result->childAt(0)->asFlatVector<StringView>();
  auto* results = result->childAt(1)->asFlatVector<StringView>();

  std::map<std::string, std::string> rows;
  for (vector_size_t i = 0; i < result->size(); ++i) {
    rows[prompts->valueAt(i).str()] = results->valueAt(i).str();
  }

  EXPECT_EQ(rows["OVERLOAD one"], "Response for: OVERLOAD one");
  EXPECT_EQ(rows["OVERLOAD two"], "Response for: OVERLOAD two");
  EXPECT_EQ(rows["normal three"], "Response for: normal three");
}

} // namespace facebook::velox::exec::rpc
