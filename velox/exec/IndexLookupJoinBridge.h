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

#include "velox/connectors/Connector.h"
#include "velox/exec/JoinBridge.h"

namespace facebook::velox::exec {

/// Coordinates sharing of index splits among multiple IndexLookupJoin operators
/// in the same pipeline. The leader operator (driverId 0) collects all splits
/// from the task and publishes them via setIndexSplits(). Follower operators
/// wait for splits via splitsOrFuture().
class IndexLookupJoinBridge : public JoinBridge {
 public:
  /// Called by the leader operator after collecting all index splits. Stores
  /// the splits and notifies all waiting followers. Must be called only once.
  /// May be called with an empty vector (e.g., when partition pruning
  /// eliminates all index partitions for a split group).
  void setIndexSplits(
      std::vector<std::shared_ptr<connector::ConnectorSplit>> splits);

  /// Returns the index splits if the leader has called setIndexSplits(),
  /// otherwise sets 'future' and returns an empty vector. The caller should
  /// check future->valid() to distinguish "not ready" (future set) from
  /// "ready with 0 splits" (future not set, empty vector).
  std::vector<std::shared_ptr<connector::ConnectorSplit>> splitsOrFuture(
      ContinueFuture* future);

 private:
  bool splitsSet_{false};
  std::vector<std::shared_ptr<connector::ConnectorSplit>> indexSplits_;
};

} // namespace facebook::velox::exec
