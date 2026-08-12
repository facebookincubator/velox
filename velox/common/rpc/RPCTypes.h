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
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "velox/vector/TypeAliases.h"

namespace facebook::velox::rpc {

/// Well-known option key constants for RPCRequest.options.
/// Use these instead of raw string literals to prevent typo bugs.
// TODO Phase 2 (draft preview): delete entire namespace keys (12) — moved to
// FbLlmInference.h CompletionRequest fields
namespace keys {
inline constexpr std::string_view kModel = "model";
inline constexpr std::string_view kTemperature = "temperature";
inline constexpr std::string_view kMaxTokens = "max_tokens";
inline constexpr std::string_view kSystemPrompt = "systemPrompt";
inline constexpr std::string_view kJsonSchema = "json_schema";
inline constexpr std::string_view kMetagenKey = "metagen_key";
inline constexpr std::string_view kTierOverride = "tier_override";
inline constexpr std::string_view kCatToken = "cat_token";
inline constexpr std::string_view kPollIntervalMs = "poll_interval_ms";
inline constexpr std::string_view kOwnerUnixname = "owner_unixname";
inline constexpr std::string_view kIsQuery = "is_query";
inline constexpr std::string_view kPrefixDim = "prefix_dim";
} // namespace keys

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

/// Generic request structure for RPC calls.
/// This is a minimal, domain-agnostic structure that works for any backend.
/// Domain-specific formatting (e.g., LLM prompts, embedding inputs) is handled
/// by the plan node's buildRequests() method.
struct RPCRequest {
  /// Row ID for tracking which row this request belongs to.
  /// This is a globally unique ID assigned by the operator.
  int64_t rowId{0};

  /// Original row index in the input batch.
  /// This is used to slice the correct row from input columns when storing
  /// passthrough data. Unlike rowId (which is globally unique across batches),
  /// this is the index within the current input batch and is set by
  /// prepareRequests() based on the SelectivityVector iteration.
  /// CRITICAL: When prepareRequests() skips null rows, originalRowIndex
  /// tracks the actual input position to avoid slicing mismatch.
  vector_size_t originalRowIndex{0};

  /// Whether this row has a null primary input.
  /// When true, the transport should short-circuit and return an error
  /// response so that buildOutput() produces SQL NULL for this row.
  /// Replaces the former "__null_input" magic string in options.
  bool isNull{false};

  /// The request payload (opaque to the framework).
  std::string payload;

  /// Type-safe options for backend-specific parameters.
  // TODO Phase 2: delete options map — Functions use typed
  // CompletionRequest/EmbeddingRequest
  std::map<std::string, std::string> options;
};

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

/// A dispatch strategy a backend/model can execute.
enum class RpcCapabilityMode {
  /// One RPC per row (synchronous). Every backend supports this.
  kPerRow = 0,
  /// Native multi-input batch RPC (synchronous): one request carries many rows
  /// and returns one response.
  kNativeBatch = 1,
  /// Asynchronous offline job: submit -> poll -> fetch. Latency is queue/GPU
  /// time, not congestion, so the operator bypasses the RTT window for it.
  kAsyncJob = 2,
  /// Number of dispatch modes; keep last. Bounds the RpcCapabilityModeSet
  /// width.
  kNumModes,
};

/// A set of RpcCapabilityMode values; kPerRow is always present.
class RpcCapabilityModeSet {
  static_assert(
      static_cast<int>(RpcCapabilityMode::kNumModes) <= 32,
      "RpcCapabilityMode outgrew the 32-bit set; widen bits_");

 public:
  constexpr RpcCapabilityModeSet() {
    add(RpcCapabilityMode::kPerRow);
  }
  /* implicit */ constexpr RpcCapabilityModeSet(
      std::initializer_list<RpcCapabilityMode> modes) {
    add(RpcCapabilityMode::kPerRow);
    for (auto mode : modes) {
      add(mode);
    }
  }

  constexpr void add(RpcCapabilityMode mode) {
    bits_ |= bit(mode);
  }
  constexpr bool has(RpcCapabilityMode mode) const {
    return (bits_ & bit(mode)) != 0;
  }

 private:
  static constexpr uint32_t bit(RpcCapabilityMode mode) {
    return 1u << static_cast<int>(mode);
  }
  uint32_t bits_{0};
};

/// Per-mode flow-control bounds — only modes in supportedModes have an entry.
/// 0 means unlimited / backend default for that dimension.
struct RpcCapabilityBounds {
  int32_t maxBatchRows{0};
  int64_t maxBatchTokens{0};
  int64_t maxBatchBytes{0};
};

/// What dispatch strategies a (backend, model) supports, plus per-mode
/// flow-control bounds. kPerRow is always supported and never needs bounds.
/// Examples:
///   - an async-job backend:   {supportedModes = {kPerRow, kAsyncJob},
///                             boundsPerMode = {{kAsyncJob, {1000, 32000,
///                             5<<20}}}}
///   - a native-batch backend: {supportedModes = {kPerRow, kNativeBatch},
///                             boundsPerMode = {{kNativeBatch, {128, 8000,
///                             1<<20}}}}
struct RpcCapability {
  /// Dispatch modes this (backend, model) supports; kPerRow is always included.
  RpcCapabilityModeSet supportedModes;

  /// Per-mode bounds. Only modes in supportedModes have an entry; e.g.
  /// async-job 10k rows vs native-batch 128. kPerRow entries are ignored.
  std::map<RpcCapabilityMode, RpcCapabilityBounds> boundsPerMode;

  bool hasMode(RpcCapabilityMode mode) const {
    return supportedModes.has(mode);
  }

  RpcCapabilityBounds getBounds(RpcCapabilityMode mode) const {
    auto it = boundsPerMode.find(mode);
    if (it != boundsPerMode.end()) {
      return it->second;
    }
    return {};

 }
};

/// Generic response structure from RPC calls.
/// This is a minimal, domain-agnostic structure that works for any backend.
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

  /// The response result (opaque to the framework).
  std::string result;

  /// Type-safe metadata from the backend.
  std::map<std::string, std::string> metadata;

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
