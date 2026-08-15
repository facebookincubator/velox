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

/// MockRPCClientTest - Tests the mock RPC backend.
///
/// TESTS:
/// - basicCallAndError: Single call succeeds; error preserves rowId
/// - batchCallAndError: Batch call returns correct count; errors preserve
/// rowIds

#include "velox/common/rpc/clients/MockRPCClient.h"

#include <folly/futures/Future.h>
#include <gtest/gtest.h>

namespace facebook::velox::rpc {
namespace {

class MockRPCClientTest : public testing::Test {};

TEST_F(MockRPCClientTest, basicCallAndError) {
  // Success path
  MockRPCClient client(std::chrono::milliseconds(1), 0.0);

  RPCRequest request;
  request.rowId = 42;
  request.payload = "What is the capital of France?";
  request.options[std::string(rpc::keys::kModel)] = "test-model";

  auto response = client.call(request).get();

  EXPECT_FALSE(response.hasError());
  EXPECT_FALSE(response.result.empty());
  EXPECT_EQ(response.rowId, 42);
  EXPECT_EQ(client.callCount(), 1);

  // Error path: rowId must be preserved
  MockRPCClient errorClient(std::chrono::milliseconds(1), 1.0);

  RPCRequest errorRequest;
  errorRequest.rowId = 12345;
  errorRequest.payload = "Test";

  auto errorResponse = errorClient.call(errorRequest).get();

  EXPECT_TRUE(errorResponse.hasError());
  EXPECT_EQ(errorResponse.rowId, 12345);
}

TEST_F(MockRPCClientTest, batchCallAndError) {
  // Success path
  MockRPCClient client(std::chrono::milliseconds(1), 0.0);

  std::vector<RPCRequest> requests;
  for (int i = 0; i < 5; i++) {
    RPCRequest req;
    req.rowId = i;
    req.payload = "Prompt " + std::to_string(i);
    requests.push_back(std::move(req));
  }

  auto responses = client.callBatch(requests).get();

  EXPECT_EQ(responses.size(), 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_FALSE(responses[i].hasError());
    EXPECT_EQ(responses[i].rowId, i);
  }
  EXPECT_EQ(client.callCount(), 5);

  // Error path: rowIds must be preserved
  MockRPCClient errorClient(std::chrono::milliseconds(1), 1.0);

  std::vector<RPCRequest> errorRequests;
  for (int i = 0; i < 5; i++) {
    RPCRequest req;
    req.rowId = 100 + i;
    req.payload = "Test " + std::to_string(i);
    errorRequests.push_back(std::move(req));
  }

  auto errorResponses = errorClient.callBatch(errorRequests).get();

  EXPECT_EQ(errorResponses.size(), 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_TRUE(errorResponses[i].hasError());
    EXPECT_EQ(errorResponses[i].rowId, 100 + i);
  }
}

} // namespace
} // namespace facebook::velox::rpc
