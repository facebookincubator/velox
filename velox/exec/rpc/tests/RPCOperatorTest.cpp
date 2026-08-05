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
#include "velox/exec/Task.h"
#include "velox/exec/rpc/RPCPlanNodeTranslator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
#include "velox/exec/rpc/tests/DemoBatchRPCFunction.h"
#include "velox/exec/rpc/tests/DemoRPCFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

namespace facebook::velox::exec::rpc {

using namespace facebook::velox::exec::test;

// A batch function that always reports congestion (kError) on each drained
// unit, so the operator deterministically exercises the AIMD backoff path.
// Default capabilities (native batch, kNativeBatch) — the per-driver window IS
// fed.
class CongestingBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  using DemoBatchRPCFunction::DemoBatchRPCFunction;

  CongestionSignal evaluateCongestion(
      const std::vector<RPCResponse>& /*responses*/) const override {
    return CongestionSignal::kError;
  }
};

// Same congestion behavior, but declares itself an async offline job
// (kAsyncJob). The async-job bypass must skip the per-driver latency window
// (its RTT is GPU-queue time) while the per-tier rate limiter still backs off.
class AsyncCongestingBatchRPCFunction : public CongestingBatchRPCFunction {
 public:
  using CongestingBatchRPCFunction::CongestingBatchRPCFunction;

  RpcCapability capabilities() const override {
    return {
        .supportedModes = {
            RpcCapabilityMode::kPerRow, RpcCapabilityMode::kAsyncJob}};
  }
};

// Declares a per-request byte budget (maxBatchBytes) and reports a fixed number
// of rows that fit it, so the operator must chunk a large backlog to keep each
// request under the cap. Records every flush's row count so the test can assert
// no flush exceeds the budget-derived limit and no rows are lost.
class ByteBudgetedBatchRPCFunction : public DemoBatchRPCFunction {
 public:
  using DemoBatchRPCFunction::DemoBatchRPCFunction;

  static constexpr int64_t kMaxBatchBytes = 3000;
  static constexpr int32_t kRowsPerBudget = 3;

  // Flush chunk sizes observed across a query. Cleared at the start of the
  // test.
  static std::vector<int32_t>& flushChunks() {
    static std::vector<int32_t> chunks;
    return chunks;
  }

  RpcCapability capabilities() const override {
    return {
        .supportedModes =
            {RpcCapabilityMode::kPerRow, RpcCapabilityMode::kNativeBatch},
        .maxBatchBytes = kMaxBatchBytes};
  }

  // Report a fixed prefix that "fits" the budget (>= 1), independent of the
  // actual byte math — this test exercises the operator's clamp, not the
  // function's estimator.
  int32_t rowsWithinByteBudget(int64_t /*budgetBytes*/) const override {
    return kRowsPerBudget;
  }

  folly::SemiFuture<std::vector<RPCResponse>> flushBatch(
      int32_t maxRows) override {
    flushChunks().push_back(maxRows);
    return DemoBatchRPCFunction::flushBatch(maxRows);
  }
};

class RPCOperatorTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    registerRPCPlanNodeTranslator();
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_rpc", []() { return std::make_shared<DemoAsyncRPCFunction>(); });
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
    // Congestion (kError on every drain) to exercise AIMD backoff: one
    // native-batch (per-driver window IS fed) and one async offline job (window
    // bypassed; only the rate limiter backs off).
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_congesting",
        []() { return std::make_shared<CongestingBatchRPCFunction>(); });
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_async_congesting",
        []() { return std::make_shared<AsyncCongestingBatchRPCFunction>(); });
    // Declares maxBatchBytes so the operator clamps each flush by the byte
    // budget (see batchClampedByByteBudget).
    AsyncRPCFunctionRegistry::registerFunction(
        "demo_batch_rpc_byte_budgeted",
        []() { return std::make_shared<ByteBudgetedBatchRPCFunction>(); });
  }

  static void TearDownTestCase() {
    OperatorTestBase::TearDownTestCase();
    // Reset MemoryManager to shut down SharedArbitrator executor threads.
    // Without this, TSAN reports a non-zero exit because the executor
    // threads are still running at process exit.
    memory::MemoryManager::testingSetInstance({});
  }

  void TearDown() override {
    RPCRateLimiter::testingResetAllState();
    OperatorTestBase::TearDown();
  }

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

// Byte-budget clamp: when the function declares maxBatchBytes, the operator
// caps each flush to rowsWithinByteBudget() — chunking a backlog so no single
// request exceeds the backend's per-request size cap (e.g. MetaGen's 5MB node
// limit) while still emitting every row. Uses the flush-all path
// (dispatchBatchSize=0), which noMoreInput() must also chunk.
TEST_F(RPCOperatorTest, batchClampedByByteBudget) {
  ByteBudgetedBatchRPCFunction::flushChunks().clear();

  auto input = makeRowVector(
      {"prompt"},
      {makeFlatVector<StringView>(
          {"r0", "r1", "r2", "r3", "r4", "r5", "r6"})}); // 7 rows

  auto plan = makeBatchRPCNode(
      PlanBuilder().values({input}).planNode(),
      {"prompt"},
      "demo_batch_rpc_byte_budgeted",
      /*dispatchBatchSize=*/0);

  auto result = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());

  // No rows lost.
  ASSERT_EQ(result->size(), 7);
  auto* results = result->childAt(1)->asFlatVector<StringView>();
  for (vector_size_t i = 0; i < result->size(); ++i) {
    EXPECT_FALSE(results->isNullAt(i));
  }

  // Every flush stayed within the budget-derived cap, and together they cover
  // all 7 rows (expected chunks [3, 3, 1]).
  const auto& chunks = ByteBudgetedBatchRPCFunction::flushChunks();
  ASSERT_FALSE(chunks.empty());
  int32_t total = 0;
  for (auto c : chunks) {
    EXPECT_GE(c, 1);
    EXPECT_LE(c, ByteBudgetedBatchRPCFunction::kRowsPerBudget);
    total += c;
  }
  EXPECT_EQ(total, 7);
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

/// Async-job window bypass. A BATCH backend that declares kAsyncJob (e.g.
/// MetaGen batch) has a job-completion RTT, not a latency signal, so the
/// operator must NOT feed the per-driver latency-gradient window on its
/// congestion verdict, while the per-tier rate limiter still backs off.
/// Observed via rpcCongestionShrinks: a native-batch congesting function
/// shrinks the window
/// (> 0); the async-job one does not (0).
TEST_F(RPCOperatorTest, asyncJobSkipsCongestionWindow) {
  auto input =
      makeRowVector({"prompt"}, {makeFlatVector<StringView>({"a", "b", "c"})});

  auto congestionShrinks = [&](const std::string& functionName) -> int64_t {
    auto plan = makeBatchRPCNode(
        PlanBuilder().values({input}).planNode(), {"prompt"}, functionName);
    std::shared_ptr<exec::Task> task;
    AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool(), task);
    auto planStats = exec::toPlanStats(task->taskStats());
    auto nodeIt = planStats.find(plan->id());
    if (nodeIt == planStats.end()) {
      return 0;
    }
    const auto& customStats = nodeIt->second.customStats;
    // Mirrors RPCOperator::kRpcCongestionShrinks (string literal avoids a BUCK
    // dep on the operator lib just for the constant).
    auto statIt = customStats.find("rpcCongestionShrinks");
    return statIt == customStats.end() ? 0 : statIt->second.sum;
  };

  // Native batch (kNativeBatch): the window is fed, so congestion shrinks it.
  EXPECT_GT(congestionShrinks("demo_batch_rpc_congesting"), 0);
  // Async job (kAsyncJob): the window is bypassed, so it never shrinks.
  EXPECT_EQ(congestionShrinks("demo_batch_rpc_async_congesting"), 0);
}

// resolveDispatchMode maps the coordinator's coarse streaming mode to the one
// concrete dispatch mode, clamped to what the backend actually supports.
TEST(ResolveDispatchModeTest, clampsToCapability) {
  using velox::rpc::resolveDispatchMode;
  using velox::rpc::RpcCapability;
  using velox::rpc::RpcCapabilityMode;
  using velox::rpc::RPCStreamingMode;

  const RpcCapability perRowOnly{
      .supportedModes = {RpcCapabilityMode::kPerRow}};
  const RpcCapability nativeBatch{
      .supportedModes = {
          RpcCapabilityMode::kPerRow, RpcCapabilityMode::kNativeBatch}};
  const RpcCapability asyncJob{
      .supportedModes = {
          RpcCapabilityMode::kPerRow, RpcCapabilityMode::kAsyncJob}};

  // PER_ROW is always kPerRow, regardless of capability.
  EXPECT_EQ(
      resolveDispatchMode(RPCStreamingMode::kPerRow, asyncJob),
      RpcCapabilityMode::kPerRow);
  // BATCH resolves to the backend's supported batch mode...
  EXPECT_EQ(
      resolveDispatchMode(RPCStreamingMode::kBatch, nativeBatch),
      RpcCapabilityMode::kNativeBatch);
  EXPECT_EQ(
      resolveDispatchMode(RPCStreamingMode::kBatch, asyncJob),
      RpcCapabilityMode::kAsyncJob);
  // ...and clamps to per-row when the backend supports no batch mode.
  EXPECT_EQ(
      resolveDispatchMode(RPCStreamingMode::kBatch, perRowOnly),
      RpcCapabilityMode::kPerRow);
}

} // namespace facebook::velox::exec::rpc
