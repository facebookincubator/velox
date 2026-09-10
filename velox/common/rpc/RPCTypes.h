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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <folly/Expected.h>

#include "velox/common/base/Exceptions.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::rpc {

/// Streaming mode for RPC execution.
/// Controls how RPC results are emitted to downstream operators.
enum class RPCStreamingMode {
  /// Emit rows as they complete individually (default).
  /// Lower tail latency for high-variance workloads (e.g., LLM).
  kPerRow,

  /// Wait for all rows in batch before emitting.
  /// Lower overhead, useful for uniform-latency workloads.
  kBatch
};

/// Parse streaming mode from config string.
/// Returns kPerRow (default) unless explicitly set to "batch".
inline RPCStreamingMode parseStreamingMode(const std::string& value) {
  if (value == "batch") {
    return RPCStreamingMode::kBatch;
  }
  return RPCStreamingMode::kPerRow;
}

/// Typed cause of an RPC failure, carried alongside the human-readable error
/// string so consumers can classify failures without parsing message text.
///
/// The transports flatten backend-specific exceptions (rate-limit, timeout)
/// into the opaque 'error' string before a response reaches the framework,
/// which loses the signal a congestion controller needs. This enum preserves
/// it: the transport tags each failed response with why it failed, and a
/// congestion policy can then treat overload (kRateLimited / kTimeout)
/// differently from a user error (kNullInput) or a benign empty result.
enum class RPCErrorKind {
  /// Not an error, or cause not classified.
  kNone,
  /// Null primary input; a user error, not a backend problem.
  kNullInput,
  /// Backend rejected the call for rate limiting / quota (e.g. HTTP 429).
  kRateLimited,
  /// The call exceeded its deadline.
  kTimeout,
  /// Backend returned a non-overload error after retries.
  kBackendError,
  /// Backend returned successfully but with no usable result.
  kEmptyResponse,
  /// Backend rejected the request as invalid (e.g. malformed args, bad model).
  /// Non-retryable: the same request will fail again, so the transport fails
  /// fast rather than spending its retry budget.
  kInvalidRequest,
};

/// Function-owned payload of a response. The framework moves it from the
/// transport to the owning function's buildOutput() and never inspects it, so
/// each function defines its own concrete type and casts back on the way out.
///
/// Keeping the payload out of the framework's vocabulary is what lets a
/// function hand back the representation it already has: an embedding function
/// carries its vector of floats directly rather than rendering it to text for
/// a field nothing in the framework reads.
struct RPCResponsePayload {
  virtual ~RPCResponsePayload() = default;
};

/// Why a request failed: the typed cause a congestion policy reads, and the
/// message a failed row carries when the on-error policy asks for one.
struct RpcError {
  RPCErrorKind kind{RPCErrorKind::kNone};
  std::string message;
};

/// Framework-visible part of a response: correlation and outcome. Everything a
/// backend actually returns lives in the function-owned payload.
///
/// A response is a payload or an error, never both and never neither. The
/// outcome is only reachable through setPayload()/setError(), so the state
/// where a producer set neither cannot be built -- a response that nothing has
/// filled in yet already reads as an error.
struct RPCResponse {
  /// Row ID for correlating response with the original request.
  ///
  /// Two meanings depending on context:
  ///   - In flushBatch() return values: the 0-based index of this response
  ///     within the flushed batch (set by the function). The operator uses
  ///     this to scatter responses into the correct positions before
  ///     stamping the global row ID.
  ///   - After operator processing: a globally unique ID assigned by the
  ///     operator for downstream result tracking.
  int64_t rowId{0};

  static RPCResponse ok(
      int64_t rowId,
      std::shared_ptr<const RPCResponsePayload> payload) {
    RPCResponse response;
    response.rowId = rowId;
    response.setPayload(std::move(payload));
    return response;
  }

  static RPCResponse
  failed(int64_t rowId, RPCErrorKind kind, std::string message) {
    RPCResponse response;
    response.rowId = rowId;
    response.setError(kind, std::move(message));
    return response;
  }

  void setPayload(std::shared_ptr<const RPCResponsePayload> payload) {
    // A success with no payload is the third state this type exists to rule
    // out: it reads as succeeded, feeds the congestion signal, and only fails
    // later when a function tries to read it.
    VELOX_CHECK_NOT_NULL(payload, "RPC response payload must not be null");
    result_ = std::move(payload);
  }

  void setError(RPCErrorKind kind, std::string message) {
    result_ = folly::makeUnexpected(RpcError{kind, std::move(message)});
  }

  /// Returns true if this response represents an error.
  bool hasError() const {
    return result_.hasError();
  }

  /// The failure. Only valid when hasError().
  const RpcError& error() const {
    return result_.error();
  }

  /// Typed cause, or kNone on a successful response.
  RPCErrorKind errorKind() const {
    return result_.hasError() ? result_.error().kind : RPCErrorKind::kNone;
  }

  /// The function-owned payload. Only valid when !hasError().
  const std::shared_ptr<const RPCResponsePayload>& payload() const {
    return result_.value();
  }

 private:
  // Unfilled means failed: a producer that returns without saying anything
  // yields an error row rather than a response with no outcome at all. The
  // message is kept short enough for the small-string buffer -- every response
  // is default-constructed before it is filled, so a longer one would put a
  // heap allocation on the per-row path to say something no one should read.
  folly::Expected<std::shared_ptr<const RPCResponsePayload>, RpcError> result_{
      folly::makeUnexpected(
          RpcError{RPCErrorKind::kBackendError, "unset response"})};
};

} // namespace facebook::velox::rpc
