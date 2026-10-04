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

/// Builds an exchange operator using its transport's Task-level client. Merge
/// operators may additionally create a client for each source.
using ExchangeOperatorFactory = std::function<std::unique_ptr<Operator>(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::ExchangeNode>& node,
    std::shared_ptr<ExchangeClient> client)>;

/// Inputs supplied by Task when creating one pipeline's exchange client. The
/// buffer limits do not apply to per-source clients created by merge operators.
///
/// Always construct with designated initializers. Several fields share a type,
/// so positional initialization could silently swap them.
struct ExchangeClientContext {
  /// Id of the consuming task, for logging.
  std::string taskId;

  /// Index of the producers' output buffer to fetch from.
  int destination;

  /// Number of exchange operators sharing this client.
  int32_t numberOfConsumers;

  /// Bytes the client may buffer before applying backpressure.
  uint64_t maxExchangeBufferSize;

  /// Bytes to accumulate before unblocking a consumer; zero delivers each page
  /// as it arrives.
  uint64_t minExchangeOutputBatchBytes;

  /// Memory pool the received pages are allocated from.
  memory::MemoryPool* pool;

  /// Executor running the exchange sources' response callbacks.
  folly::Executor* executor;

  /// The query config, valid only during the factory call. Copy any settings
  /// needed afterward.
  const core::QueryConfig& queryConfig;
};

/// Creates one pipeline's exchange client while Task holds its mutex. The
/// factory must not block, call back into Task, or allocate from 'pool'. Defer
/// transport setup until remote task IDs are added.
using ExchangeClientFactory = std::function<std::shared_ptr<ExchangeClient>(
    const ExchangeClientContext& context)>;

} // namespace facebook::velox::exec
