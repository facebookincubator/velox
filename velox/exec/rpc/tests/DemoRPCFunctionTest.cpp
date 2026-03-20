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

/// DemoRPCFunctionTest - End-to-end test for the reference AsyncRPCFunction
/// implementation.
///
/// Exercises the full lifecycle: initialize -> dispatchPerRow -> buildOutput,
/// verifying that DemoAsyncRPCFunction correctly follows the AsyncRPCFunction
/// contract.

#include "velox/exec/rpc/tests/DemoRPCFunction.h"

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec::rpc {
namespace {

class DemoRPCFunctionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    function_ = std::make_shared<DemoAsyncRPCFunction>();
    pool_ = memory::memoryManager()->addLeafPool();

    // Follow the lifecycle: initialize() before any dispatch.
    function_->initialize(core::QueryConfig{{}}, {}, {});
  }

  std::shared_ptr<DemoAsyncRPCFunction> function_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(DemoRPCFunctionTest, endToEnd) {
  // Build input vector.
  std::vector<std::string> prompts = {"hello world", "test prompt"};
  const auto numRows = static_cast<vector_size_t>(prompts.size());
  auto input = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), numRows, pool_.get());
  for (vector_size_t i = 0; i < numRows; ++i) {
    input->set(i, StringView(prompts[i]));
  }

  // Dispatch per-row and collect futures.
  SelectivityVector rows(numRows);
  auto futures = function_->dispatchPerRow(rows, {input});
  ASSERT_EQ(futures.size(), 2);
  EXPECT_EQ(futures[0].first, 0);
  EXPECT_EQ(futures[1].first, 1);

  // Resolve futures.
  std::vector<RPCResponse> responses;
  responses.reserve(futures.size());
  for (auto& [rowIdx, future] : futures) {
    responses.push_back(std::move(future).get());
  }
  ASSERT_EQ(responses.size(), 2);
  for (const auto& resp : responses) {
    EXPECT_FALSE(resp.hasError());
  }

  // Build output vector.
  auto result = function_->buildOutput(responses, pool_.get());
  ASSERT_EQ(result->size(), 2);
  auto* flat = result->asFlatVector<StringView>();
  EXPECT_FALSE(flat->isNullAt(0));
  EXPECT_FALSE(flat->isNullAt(1));
  // MockRPCClient returns "Response for: <payload>".
  EXPECT_EQ(flat->valueAt(0).str(), "Response for: hello world");
  EXPECT_EQ(flat->valueAt(1).str(), "Response for: test prompt");
}

TEST_F(DemoRPCFunctionTest, nullInput) {
  // Build input with a null row.
  auto input =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), 2, pool_.get());
  input->set(0, StringView("valid prompt"));
  input->setNull(1, true);

  SelectivityVector rows(2);
  auto futures = function_->dispatchPerRow(rows, {input});

  // Both rows produce futures (null row gets immediate error response).
  ASSERT_EQ(futures.size(), 2);
  EXPECT_EQ(futures[0].first, 0);
  EXPECT_EQ(futures[1].first, 1);

  // Non-null row should succeed.
  auto resp0 = std::move(futures[0].second).get();
  EXPECT_FALSE(resp0.hasError());

  // Null row should get error="null_input".
  auto resp1 = std::move(futures[1].second).get();
  EXPECT_TRUE(resp1.hasError());
  EXPECT_EQ(resp1.error.value(), "null_input");
}

TEST_F(DemoRPCFunctionTest, errorResponse) {
  // Build output from a response with an error.
  std::vector<RPCResponse> responses;
  RPCResponse ok;
  ok.rowId = 0;
  ok.result = "good result";
  responses.push_back(std::move(ok));

  RPCResponse err;
  err.rowId = 1;
  err.error = "RPC failed";
  responses.push_back(std::move(err));

  auto result = function_->buildOutput(responses, pool_.get());
  ASSERT_EQ(result->size(), 2);
  auto* flat = result->asFlatVector<StringView>();
  EXPECT_FALSE(flat->isNullAt(0));
  EXPECT_EQ(flat->valueAt(0).str(), "good result");
  EXPECT_TRUE(flat->isNullAt(1));
}

TEST_F(DemoRPCFunctionTest, signatures) {
  auto sigs = DemoAsyncRPCFunction::signatures();
  ASSERT_EQ(sigs.size(), 1);
  // demo_rpc(varchar) -> varchar
  EXPECT_EQ(sigs[0]->argumentTypes().size(), 1);
}

TEST_F(DemoRPCFunctionTest, metadata) {
  EXPECT_EQ(function_->name(), "demo_rpc");
  EXPECT_EQ(function_->resultType()->kind(), TypeKind::VARCHAR);
  EXPECT_EQ(function_->tierKey(), "");
}

} // namespace
} // namespace facebook::velox::exec::rpc
