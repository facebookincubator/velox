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

#include "velox/exec/rpc/RPCPlanNodeTranslator.h"
#include "velox/exec/rpc/RPCRateLimiter.h"
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

  /// Build an RPCNode on top of a source plan node.
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

} // namespace facebook::velox::exec::rpc
