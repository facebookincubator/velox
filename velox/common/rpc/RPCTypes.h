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

/// Framework-visible part of a response: correlation, failure, and the typed
/// cause a congestion policy needs. Everything a backend actually returns
/// lives in the function-owned payload.
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

  /// Function-owned result. Opaque to the framework: it is moved from the
  /// transport to the owning function's buildOutput() and never inspected
  /// here.
  std::shared_ptr<const RPCResponsePayload> payload;

  /// Error message if the request failed.
  std::optional<std::string> error;

  /// Typed cause of the failure, set by the transport when 'error' is set.
  /// Defaults to kNone so existing aggregate initializers need not list it.
  RPCErrorKind errorKind{RPCErrorKind::kNone};

  /// Returns true if this response represents an error.
  bool hasError() const {
    return error.has_value();
  }
};

} // namespace facebook::velox::rpc
