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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <folly/futures/Future.h>

#include "velox/common/rpc/RPCTypes.h"

namespace facebook::velox::core {
class QueryConfig;
} // namespace facebook::velox::core

namespace facebook::velox::rpc {

/// Interface for RPC clients (transport layer).
///
/// IRPCClient is concerned with how to send requests and receive responses
/// over the network. It is decoupled from the business logic — domain-specific
/// request/response formatting is handled by AsyncRPCFunction (in
/// velox/expression/rpc/).
///
/// Implementations provide the actual transport (e.g., Thrift, gRPC, mock).
///
/// Thread safety: Implementations MUST be thread-safe for concurrent calls.
/// The operator may dispatch multiple RPCs concurrently from a single thread,
/// and completion callbacks run on the client's executor threads.
class IRPCClient {
 public:
  virtual ~IRPCClient() = default;

  /// Execute a single RPC call asynchronously.
  /// @param request The request to send.
  /// @return A SemiFuture that will contain the response when complete.
  virtual folly::SemiFuture<RPCResponse> call(const RPCRequest& request) = 0;

  /// Execute a batch of RPC calls as a single request.
  /// Default implementation fans out to individual call()s.
  /// Override for backends that support native batching (e.g., batch
  /// inference).
  /// @param requests The batch of requests to send.
  /// @return A SemiFuture that will contain all responses when complete.
  virtual folly::SemiFuture<std::vector<RPCResponse>> callBatch(
      const std::vector<RPCRequest>& requests) {
    std::vector<folly::SemiFuture<RPCResponse>> futures;
    futures.reserve(requests.size());
    for (const auto& request : requests) {
      futures.push_back(call(request));
    }
    // Capture rowIds to preserve them in error responses.
    std::vector<int64_t> rowIds;
    rowIds.reserve(requests.size());
    for (const auto& request : requests) {
      rowIds.push_back(request.rowId);
    }
    return folly::collectAll(std::move(futures))
        .deferValue([rowIds = std::move(rowIds)](
                        std::vector<folly::Try<RPCResponse>> tries) {
          std::vector<RPCResponse> responses;
          responses.reserve(tries.size());
          for (size_t i = 0; i < tries.size(); ++i) {
            if (tries[i].hasValue()) {
              responses.push_back(std::move(tries[i].value()));
            } else {
              RPCResponse errorResp;
              errorResp.rowId = rowIds[i];
              errorResp.error = tries[i].exception().what().toStdString();
              responses.push_back(std::move(errorResp));
            }
          }
          return responses;
        });
  }

  /// Returns the service tier key for rate limiting (e.g.,
  /// "service.backend.prod"). Requests from clients sharing the same tier key
  /// share a concurrency budget in RPCRateLimiter.
  /// Empty string means "no tier configured" — uses the global default limit.
  virtual std::string tierKey() const {
    return "";
  }

  /// Set the query config for session-level parameters.
  /// Called by the operator before dispatching RPCs.
  virtual void setQueryConfig(const core::QueryConfig* /*config*/) {}
};

} // namespace facebook::velox::rpc
