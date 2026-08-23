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
#include <functional>
#include <memory>
#include <string>

namespace folly {
class Executor;
} // namespace folly

namespace facebook::velox::core {
class ExchangeNode;
class QueryConfig;
} // namespace facebook::velox::core

namespace facebook::velox::memory {
class MemoryPool;
} // namespace facebook::velox::memory

namespace facebook::velox::exec {

struct DriverCtx;
class Operator;
class ExchangeClient;

/// Builds a pipeline's exchange operator (e.g. Exchange or MergeExchange),
/// bound to the matching exchange client. The two are registered together in
/// ExchangeTransportRegistry so they cannot diverge.
using ExchangeOperatorFactory = std::function<std::unique_ptr<Operator>(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::ExchangeNode>& node,
    std::shared_ptr<ExchangeClient> client)>;

/// Caller-supplied context for building one exchange client. Grouping the
/// arguments lets the caller, not the transport, size the exchange buffer:
/// Task's plain-exchange path fills 'maxExchangeBufferSize' and
/// 'minExchangeOutputBatchBytes' from 'queryConfig', while a merge path can
/// pass its per-source budget and zero (deliver each page as it arrives) --
/// a distinction a single query-config-derived limit cannot express.
///
/// Always construct with designated initialisers. Five of the fields are
/// adjacent integers and pointers that a reorder would silently transpose.
struct ExchangeClientContext {
  /// Id of the consuming task, for logging.
  std::string taskId;

  /// Index of the producers' output buffer to fetch from.
  int destination;

  /// Number of exchange operators sharing this client.
  int32_t numberOfConsumers;

  /// Bytes the client may buffer before applying backpressure.
  int64_t maxExchangeBufferSize;

  /// Bytes to accumulate before unblocking a consumer; zero delivers each page
  /// as it arrives.
  uint64_t minExchangeOutputBatchBytes;

  /// Memory pool the received pages are allocated from.
  memory::MemoryPool* pool;

  /// Executor running the exchange sources' response callbacks.
  folly::Executor* executor;

  /// The running query's config, so a transport can honor session-level
  /// exchange tuning beyond the fields above. It outlives the client.
  const core::QueryConfig& queryConfig;
};

/// Creates the transport's exchange client for one pipeline of one task.
using ExchangeClientFactory = std::function<std::shared_ptr<ExchangeClient>(
    const ExchangeClientContext& context)>;

} // namespace facebook::velox::exec
