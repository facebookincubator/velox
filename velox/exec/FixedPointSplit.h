/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
#include <vector>

#include "velox/connectors/Connector.h"

namespace facebook::velox::exec {

/// Names the workers of a fixed point that reads its peers, by destination:
/// the worker whose task has destination d is 'workers[d]'.  There is at least
/// one name, and the names are non-empty and unique.  The application
/// adds one to each worker's task, on the FixedPointNode, then calls
/// noMoreSplits on the node.  The loop names itself 'workers[destination]'
/// and its peers by the other entries, and passes these names back to the
/// application through FixedPointOptions::LoopAddress.  Only a top-level loop
/// takes one: a nested loop runs on the workers of the loop that encloses it.
///
///   task->addSplit(
///       fixedPointNodeId,
///       exec::Split(std::make_shared<FixedPointSplit>(workerNames)));
///   task->noMoreSplits(fixedPointNodeId);
struct FixedPointSplit : public connector::ConnectorSplit {
  const std::vector<std::string> workers;

  explicit FixedPointSplit(std::vector<std::string> workers);

  static std::shared_ptr<FixedPointSplit> create(const folly::dynamic& obj);

  static void registerSerDe();

  folly::dynamic serialize() const override;

  std::string toString() const override;
};

} // namespace facebook::velox::exec
