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

#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "velox/vector/TypeAliases.h"

namespace facebook::velox::rpc {

/// Well-known option key constants for RPCRequest.options.
/// Use these instead of raw string literals to prevent typo bugs.
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
  std::map<std::string, std::string> options;
};

/// Generic response structure from RPC calls.
/// This is a minimal, domain-agnostic structure that works for any backend.
struct RPCResponse {
  /// Row ID for correlating response with the original request.
  int64_t rowId{0};

  /// The response result (opaque to the framework).
  std::string result;

  /// Type-safe metadata from the backend.
  std::map<std::string, std::string> metadata;

  /// Error message if the request failed.
  std::optional<std::string> error;

  /// Returns true if this response represents an error.
  bool hasError() const {
    return error.has_value();
  }
};

} // namespace facebook::velox::rpc
