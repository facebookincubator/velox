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

#include <deque>
#include <functional>
#include <memory>
#include <string>

#include <folly/container/F14Map.h>
#include <folly/io/IOBuf.h>
#include <string_view>

#include "velox/common/memory/MemoryPool.h"

namespace facebook::velox::exec {

/// Output produced after a durable sink finishes successfully.
struct CommittedExchangeOutput {
  folly::F14FastMap<int32_t, std::string> locations;
};

/// Writes partitioned data to a durable exchange backend.
class ExchangeSink {
 public:
  using Factory = std::function<std::shared_ptr<ExchangeSink>(
      const std::string& config,
      const std::string& taskId,
      velox::memory::MemoryPool* pool)>;

  virtual ~ExchangeSink() = default;

  /// Appends serialized data to a partition. Calls for different partitions
  /// may run concurrently; calls for one partition are serialized.
  virtual void append(int32_t partition, std::string_view data) = 0;

  /// Appends a serialized IOBuf chain to a partition. The default
  /// implementation coalesces the chain and delegates to the string-view
  /// overload.
  virtual void append(int32_t partition, std::unique_ptr<folly::IOBuf> data);

  /// Appends an ordered batch of independently serialized buffers. The
  /// default implementation chains the buffers and delegates to append().
  virtual void appendBatch(
      int32_t partition,
      std::deque<std::unique_ptr<folly::IOBuf>> data);

  /// Commits output and returns its backend-specific locations.
  virtual CommittedExchangeOutput finish() = 0;

  /// Aborts output and releases uncommitted backend resources.
  virtual void abort() = 0;

  /// Returns backend-specific runtime counters.
  virtual folly::F14FastMap<std::string, int64_t> stats() const = 0;
};

} // namespace facebook::velox::exec
