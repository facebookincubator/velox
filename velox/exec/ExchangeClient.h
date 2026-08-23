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

#include <string>

#include <folly/container/F14Map.h>
#include <folly/dynamic.h>

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox::exec {

/// Control plane of a task's receive side for one exchange transport. Owns the
/// set of producers a pipeline reads from and is what Task holds and drives.
///
/// The data plane is deliberately absent: page payloads are transport specific
/// (in-memory serialized pages, GPU buffers, ...), so there is nothing shared
/// to abstract. A transport registers its client factory and its exchange
/// operator factory together in ExchangeTransportRegistry, so the operator
/// always knows the concrete client type it was paired with and can reach the
/// transport's own data plane directly.
///
/// Implementations must be safe to call from multiple threads: Task adds remote
/// tasks from the split path while drivers consume data.
class ExchangeClient {
 public:
  virtual ~ExchangeClient() = default;

  /// Starts fetching data from the upstream task identified by 'remoteTaskId'.
  /// If close() has been called already, notifies the upstream task that its
  /// data is no longer needed. Repeated calls with the same 'remoteTaskId' are
  /// ignored.
  virtual void addRemoteTaskId(const std::string& remoteTaskId) = 0;

  /// Signals that no more calls to addRemoteTaskId() will follow.
  virtual void noMoreRemoteTasks() = 0;

  /// Releases the producers and unblocks consumers. Idempotent.
  virtual void close() = 0;

  /// Returns runtime statistics aggregated across all producers.
  /// Implementations are expected to report background CPU time as a metric
  /// named Operator::kBackgroundCpuTimeNanos.
  virtual folly::F14FastMap<std::string, RuntimeMetric> stats() = 0;

  /// Returns a human-readable description of the producers, for logging.
  virtual std::string toString() const = 0;

  /// Returns the client state as JSON, for the task's /v1/task endpoint.
  virtual folly::dynamic toJson() const = 0;
};

} // namespace facebook::velox::exec
