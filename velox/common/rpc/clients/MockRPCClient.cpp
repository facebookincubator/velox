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

#include "velox/common/rpc/clients/MockRPCClient.h"

#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>

namespace facebook::velox::rpc {

namespace {
// Function-local static pattern for thread-local RNG to avoid
// NonPodStaticDeclaration lint warning.
std::mt19937& threadLocalRng() {
  thread_local std::mt19937 rng{std::random_device{}()};
  return rng;
}
} // namespace

MockRPCClient::MockRPCClient(
    std::chrono::milliseconds latency,
    double errorRate,
    std::shared_ptr<folly::CPUThreadPoolExecutor> executor)
    : latency_(latency), errorRate_(errorRate) {
  if (executor) {
    executor_ = std::move(executor);
  } else {
    ownedExecutor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
    executor_ = ownedExecutor_;
  }
}

MockRPCClient::~MockRPCClient() = default;

RPCResponse MockRPCClient::generateResponse(
    const RPCRequest& request,
    bool isError) {
  if (isError) {
    return RPCResponse{
        .rowId = request.rowId,
        .result = "",
        .metadata = {},
        .error = "Simulated error for row " + std::to_string(request.rowId)};
  }

  // Generate a mock response
  std::string responseText = "Response for: ";
  if (request.payload.size() > 30) {
    responseText += request.payload.substr(0, 30) + "...";
  } else {
    responseText += request.payload;
  }

  return RPCResponse{
      .rowId = request.rowId,
      .result = std::move(responseText),
      .metadata = {},
      .error = std::nullopt};
}

folly::SemiFuture<RPCResponse> MockRPCClient::call(const RPCRequest& request) {
  callCount_.fetch_add(1);

  // Determine if this request should fail
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  bool shouldError = dist(threadLocalRng()) < errorRate_;

  // Use folly::via with the thread pool executor for safe async execution
  return folly::via(
      executor_.get(),
      [this, request = request, shouldError, latency = latency_]()
          -> RPCResponse {
        // Simulate network latency
        /* sleep override */ std::this_thread::sleep_for(latency);
        // Generate and return the response
        return generateResponse(request, shouldError);
      });
}

folly::SemiFuture<std::vector<RPCResponse>> MockRPCClient::callBatch(
    const std::vector<RPCRequest>& requests) {
  // Capture error rate for thread safety
  double errorRate = errorRate_;

  // Use folly::via with the thread pool executor for safe async execution
  return folly::via(
      executor_.get(),
      [this, requests, errorRate, latency = latency_]()
          -> std::vector<RPCResponse> {
        // Simulate network latency (single batch = single latency)
        /* sleep override */ std::this_thread::sleep_for(latency);

        std::vector<RPCResponse> responses;
        responses.reserve(requests.size());

        // Create RNG inside lambda to avoid thread-local access issues.
        // Each executor thread will have its own properly initialized RNG.
        thread_local std::mt19937 localRng{std::random_device{}()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (const auto& request : requests) {
          callCount_.fetch_add(1);
          bool shouldError = dist(localRng) < errorRate;
          responses.push_back(generateResponse(request, shouldError));
        }

        return responses;
      });
}

} // namespace facebook::velox::rpc
