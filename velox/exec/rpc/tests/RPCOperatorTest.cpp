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

    std::vector<std::string> argCols;
    std::vector<TypePtr> argTypes;
    std::vector<VectorPtr> constantInputs;
    for (const auto& colName : argumentColumnNames) {
      argCols.push_back(colName);
      argTypes.push_back(sourceType->findChild(colName));
      constantInputs.push_back(nullptr);
    }

    auto outputNames = sourceType->names();
    auto outputTypes = sourceType->children();
    outputNames.emplace_back("__rpc_result");
    outputTypes.push_back(VARCHAR());
    auto outputType = ROW(std::move(outputNames), std::move(outputTypes));

    return std::make_shared<core::RPCNode>(
        "rpc-0",
        source,
        functionName,
        VARCHAR(),
        "__rpc_result",
        outputType,
        argCols,
        argTypes,
        constantInputs,
        RPCStreamingMode::kBatch,
        dispatchBatchSize);
  }

  /// Build a PER_ROW-mode RPCNode on top of a source plan node.
  /// argumentColumnNames specifies which source columns are RPC arguments.
  core::PlanNodePtr makeRPCNode(
      const core::PlanNodePtr& source,
      const std::vector<std::string>& argumentColumnNames) {
    auto sourceType = source->outputType();

    std::vector<std::string> argCols;
    std::vector<TypePtr> argTypes;
    std::vector<VectorPtr> constantInputs;
    for (const auto& colName : argumentColumnNames) {
      argCols.push_back(colName);
      argTypes.push_back(sourceType->findChild(colName));
      constantInputs.push_back(nullptr); // Variable, not constant.
    }

    // Output type = all source columns + RPC result column.
    auto outputNames = sourceType->names();
    auto outputTypes = sourceType->children();
    outputNames.emplace_back("__rpc_result");
    outputTypes.push_back(VARCHAR());
    auto outputType = ROW(std::move(outputNames), std::move(outputTypes));

    return std::make_shared<core::RPCNode>(
        "rpc-0",
        source,
        "demo_rpc",
        VARCHAR(),
        "__rpc_result",
        outputType,
        argCols,
        argTypes,
        constantInputs);
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
