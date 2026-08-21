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

#include "velox/common/rpc/clients/MockTransport.h"

#include <fmt/format.h>
#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::rpc {

namespace {
// Function-local static pattern for thread-local RNG to avoid
// NonPodStaticDeclaration lint warning.
std::mt19937& threadLocalRng() {
  thread_local std::mt19937 rng{std::random_device{}()};
  return rng;
}

// Builds an error response tagged with a typed cause, for the error-burst
// path. Unlike MockTransport::generateResponse(request, /*isError=*/true),
// which leaves errorKind at kNone, this sets errorKind so the congestion path
// can classify the failure.
RPCResponse makeErrorResponse(const RPCRequest& request, RPCErrorKind kind) {
  // Every enumerator is listed explicitly (no default) so a newly added kind
  // trips -Wswitch-enum instead of silently reusing another kind's label.
  std::string_view label;
  switch (kind) {
    case RPCErrorKind::kRateLimited:
      label = "rate-limit";
      break;
    case RPCErrorKind::kTimeout:
      label = "timeout";
      break;
    case RPCErrorKind::kNullInput:
      label = "null-input";
      break;
    case RPCErrorKind::kBackendError:
      label = "backend";
      break;
    case RPCErrorKind::kEmptyResponse:
      label = "empty-response";
      break;
    case RPCErrorKind::kInvalidRequest:
      label = "invalid-request";
      break;
    case RPCErrorKind::kNone:
      VELOX_UNREACHABLE(
          "makeErrorResponse() requires a non-kNone error kind, "
          "it is only reached on the error-burst path");
  }
  return RPCResponse{
      .rowId = request.rowId,
      .result = "",
      .error =
          fmt::format("Simulated {} error for row {}", label, request.rowId),
      .errorKind = kind};
}
} // namespace

MockTransport::MockTransport(
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

MockTransport::~MockTransport() = default;

RPCResponse MockTransport::generateResponse(
    const RPCRequest& request,
    bool isError) {
  if (isError) {
    return RPCResponse{
        .rowId = request.rowId,
        .result = "",
        .error = "Simulated error for row " + std::to_string(request.rowId)};
  }

  // Generate a mock response — RPCRequest is correlation-only (rowId/isNull),
  // no payload.
  std::string responseText =
      "Response for row " + std::to_string(request.rowId);

  return RPCResponse{
      .rowId = request.rowId,
      .result = std::move(responseText),
      .error = std::nullopt};
}

void MockTransport::setErrorBurst(const ErrorBurst& burst) {
  // Install-once, before dispatch. burstErrorKind() reads errorBurst_ without
  // a lock, so the struct must never be written while a request could be
  // reading it. Rejecting a second install closes the resetCallCount() path:
  // the counter check alone would let a re-arm race with requests still in
  // flight from the previous burst.
  VELOX_CHECK(
      !burstInstalled_.load(std::memory_order_acquire),
      "setErrorBurst() may only be called once per client");
  VELOX_CHECK_EQ(
      callCount_.load(),
      0,
      "setErrorBurst() must be called before the first request is dispatched");
  errorBurst_ = burst;
  // Release: publishes the errorBurst_ write to any thread that subsequently
  // observes burstInstalled_ as true through the acquire load below.
  burstInstalled_.store(true, std::memory_order_release);
}

RPCErrorKind MockTransport::burstErrorKind(int64_t ordinal) const {
  // Acquire: pairs with the release store in setErrorBurst(), so the
  // errorBurst_ fields read below are guaranteed visible on this thread.
  // errorBurst_ is never mutated once a burst is installed, so no further
  // synchronization is needed.
  if (!burstInstalled_.load(std::memory_order_acquire)) {
    return RPCErrorKind::kNone;
  }
  if (errorBurst_.firstCall < errorBurst_.lastCall &&
      ordinal >= errorBurst_.firstCall && ordinal < errorBurst_.lastCall) {
    return errorBurst_.errorKind;
  }
  return RPCErrorKind::kNone;
}

folly::SemiFuture<RPCResponse> MockTransport::call(const RPCRequest& request) {
  const int64_t ordinal = callCount_.fetch_add(1);
  const RPCErrorKind burstKind = burstErrorKind(ordinal);

  // Draw the random error decision only on the non-burst path, so the RNG
  // stream is consumed identically to callBatch(), which skips the draw for
  // bursted requests. This keeps the two paths reproducible under a fixed seed.
  bool shouldError = false;
  if (burstKind == RPCErrorKind::kNone) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    shouldError = dist(threadLocalRng()) < errorRate_;
  }

  // Use folly::via with the thread pool executor for safe async execution
  return folly::via(
      executor_.get(),
      [this, request = request, shouldError, burstKind, latency = latency_]()
          -> RPCResponse {
        // Deterministic overload burst: fail fast with a typed cause, skipping
        // the latency sleep (overload rejections come back immediately).
        if (burstKind != RPCErrorKind::kNone) {
          return makeErrorResponse(request, burstKind);
        }
        // Simulate network latency
        /* sleep override */ std::this_thread::sleep_for(latency);
        // Generate and return the response
        return generateResponse(request, shouldError);
      });
}

folly::SemiFuture<std::vector<RPCResponse>> MockTransport::callBatch(
    const std::vector<RPCRequest>& requests) {
  // Capture error rate for thread safety
  double errorRate = errorRate_;

  // Reserve this batch's ordinals on the caller thread, as call() does, so the
  // burst window covers a fixed set of requests even when several batches are
  // in flight. Assigning them inside the lambda would tie the mapping to
  // executor scheduling order.
  const int64_t firstOrdinal =
      callCount_.fetch_add(static_cast<int64_t>(requests.size()));

  // Use folly::via with the thread pool executor for safe async execution
  return folly::via(
      executor_.get(),
      [this, requests, errorRate, firstOrdinal, latency = latency_]()
          -> std::vector<RPCResponse> {
        // Simulate network latency (single batch = single latency). Unlike
        // call(), which returns a bursted request immediately, a batch pays
        // this once up front even when some of its requests are in the burst
        // window: the batch is one round trip, and a backend that rejects part
        // of it still costs the caller that trip.
        /* sleep override */ std::this_thread::sleep_for(latency);

        std::vector<RPCResponse> responses;
        responses.reserve(requests.size());

        // Create RNG inside lambda to avoid thread-local access issues.
        // Each executor thread will have its own properly initialized RNG.
        thread_local std::mt19937 localRng{std::random_device{}()};
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < requests.size(); ++i) {
          const auto& request = requests[i];
          const RPCErrorKind burstKind =
              burstErrorKind(firstOrdinal + static_cast<int64_t>(i));
          if (burstKind != RPCErrorKind::kNone) {
            responses.push_back(makeErrorResponse(request, burstKind));
            continue;
          }
          bool shouldError = dist(localRng) < errorRate;
          responses.push_back(generateResponse(request, shouldError));
        }

        return responses;
      });
}

} // namespace facebook::velox::rpc
