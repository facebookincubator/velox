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

/// RPCNodeTest - Tests plan node creation and AsyncRPCFunction behavior.
///
/// TESTS:
/// - rpcNodeCreation: RPCNode can be created with correct fields
/// - rpcNodeWithBatchMode: Batch mode configuration works
/// - functionDispatchPerRow: AsyncRPCFunction.dispatchPerRow() works
/// - functionBuildOutput: AsyncRPCFunction.buildOutput() works
/// - planNodeToString: toString() includes configuration info
/// - planNodeSingleSource: Plan node has exactly one source

#include "velox/core/PlanNode.h"

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/common/rpc/clients/MockRPCClient.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec::rpc {

using velox::rpc::MockRPCClient;

namespace {

/// Mock implementation of AsyncRPCFunction for testing.
class MockAsyncRPCFunction : public AsyncRPCFunction {
 public:
  explicit MockAsyncRPCFunction(std::shared_ptr<MockRPCClient> client)
      : client_(std::move(client)) {}

  std::string name() const override {
    return "mock_rpc_function";
  }

  TypePtr resultType() const override {
    return VARCHAR();
  }

  std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
  dispatchPerRow(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) override {
    std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
        results;

    if (args.empty()) {
      return results;
    }

    auto* promptVector = args[0]->as<SimpleVector<StringView>>();
    if (!promptVector) {
      return results;
    }

    rows.applyToSelected([&](vector_size_t row) {
      if (promptVector->isNullAt(row)) {
        results.emplace_back(
            row,
            folly::makeSemiFuture<RPCResponse>(RPCResponse{
                .rowId = static_cast<int64_t>(row),
                .result = "",
                .metadata = {},
                .error = "null_input"}));
        return;
      }
      RPCRequest request;
      request.payload = promptVector->valueAt(row).str();
      results.emplace_back(row, client_->call(request));
    });

    return results;
  }

 private:
  std::shared_ptr<MockRPCClient> client_;
};

class RPCNodeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    client_ =
        std::make_shared<MockRPCClient>(std::chrono::milliseconds(10), 0.0);
    function_ = std::make_shared<MockAsyncRPCFunction>(client_);
    pool_ = memory::memoryManager()->addLeafPool();
  }

  std::shared_ptr<MockRPCClient> client_;
  std::shared_ptr<MockAsyncRPCFunction> function_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(RPCNodeTest, rpcNodeCreation) {
  auto rpcNode = std::make_shared<core::RPCNode>(
      "rpc-1",
      nullptr,
      "mock_rpc_function",
      VARCHAR(),
      "response",
      ROW({"response"}, {VARCHAR()}),
      std::vector<std::string>{},
      std::vector<TypePtr>{},
      std::vector<VectorPtr>{});

  EXPECT_EQ(rpcNode->id(), "rpc-1");
  EXPECT_EQ(rpcNode->name(), "RPC");
  EXPECT_EQ(rpcNode->functionName(), "mock_rpc_function");
  EXPECT_EQ(rpcNode->outputColumn(), "response");
  EXPECT_EQ(rpcNode->streamingMode(), rpc::RPCStreamingMode::kPerRow);
  EXPECT_EQ(rpcNode->dispatchBatchSize(), 0);
}

TEST_F(RPCNodeTest, rpcNodeWithBatchMode) {
  auto rpcNode = std::make_shared<core::RPCNode>(
      "rpc-2",
      nullptr,
      "mock_rpc_function",
      VARCHAR(),
      "result",
      ROW({"result"}, {VARCHAR()}),
      std::vector<std::string>{},
      std::vector<TypePtr>{},
      std::vector<VectorPtr>{},
      rpc::RPCStreamingMode::kBatch,
      100);

  EXPECT_EQ(rpcNode->streamingMode(), rpc::RPCStreamingMode::kBatch);
  EXPECT_EQ(rpcNode->dispatchBatchSize(), 100);
}

TEST_F(RPCNodeTest, functionDispatchPerRow) {
  std::vector<std::string> prompts = {
      "What is 2+2?",
      "What is the capital of France?",
      "Explain quantum computing."};

  const auto numPrompts = static_cast<vector_size_t>(prompts.size());
  auto promptVector = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), numPrompts, pool_.get());
  for (vector_size_t i = 0; i < numPrompts; ++i) {
    promptVector->set(i, StringView(prompts[i]));
  }

  std::vector<VectorPtr> args = {promptVector};
  SelectivityVector rows(numPrompts);

  auto futures = function_->dispatchPerRow(rows, args);

  // Extract row indices and verify ordering.
  std::vector<vector_size_t> rowIndices;
  rowIndices.reserve(futures.size());
  for (const auto& [idx, _] : futures) {
    rowIndices.push_back(idx);
  }
  ASSERT_EQ(rowIndices.size(), 3);
  EXPECT_EQ(rowIndices[0], 0);
  EXPECT_EQ(rowIndices[1], 1);
  EXPECT_EQ(rowIndices[2], 2);

  // Resolve futures and verify responses.
  for (auto& [rowIdx, future] : futures) {
    auto response = std::move(future).get();
    EXPECT_FALSE(response.hasError());
    EXPECT_FALSE(response.result.empty());
  }
}

TEST_F(RPCNodeTest, functionBuildOutput) {
  std::vector<RPCResponse> responses;
  for (int i = 0; i < 3; ++i) {
    RPCResponse resp;
    resp.rowId = i;
    resp.result = "Response for prompt " + std::to_string(i);
    responses.push_back(std::move(resp));
  }

  auto result = function_->buildOutput(responses, pool_.get());

  EXPECT_EQ(result->size(), 3);
  EXPECT_EQ(result->type()->kind(), TypeKind::VARCHAR);

  auto* flatResult = result->asFlatVector<StringView>();
  EXPECT_EQ(flatResult->valueAt(0).str(), "Response for prompt 0");
  EXPECT_EQ(flatResult->valueAt(1).str(), "Response for prompt 1");
  EXPECT_EQ(flatResult->valueAt(2).str(), "Response for prompt 2");
}

TEST_F(RPCNodeTest, planNodeToString) {
  auto rpcNode = std::make_shared<core::RPCNode>(
      "rpc-1",
      nullptr,
      "mock_rpc_function",
      VARCHAR(),
      "response",
      ROW({"response"}, {VARCHAR()}),
      std::vector<std::string>{},
      std::vector<TypePtr>{},
      std::vector<VectorPtr>{});

  std::string str = rpcNode->toString(/*detailed=*/true);

  EXPECT_TRUE(str.find("RPC") != std::string::npos);
  EXPECT_TRUE(str.find("mock_rpc_function") != std::string::npos);
}

TEST_F(RPCNodeTest, planNodeSingleSource) {
  auto rpcNode = std::make_shared<core::RPCNode>(
      "rpc-1",
      nullptr,
      "mock_rpc_function",
      VARCHAR(),
      "response",
      ROW({"response"}, {VARCHAR()}),
      std::vector<std::string>{},
      std::vector<TypePtr>{},
      std::vector<VectorPtr>{});

  ASSERT_EQ(rpcNode->sources().size(), 1);
  EXPECT_EQ(rpcNode->source().get(), nullptr);
}

} // namespace
} // namespace facebook::velox::exec::rpc
