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

#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec {

core::PlanNodePtr findPlanNodeById(
    const core::PlanNodePtr& planNode,
    const std::string& id) {
  if (planNode->id() == id) {
    return planNode;
  }

  for (const auto& child : planNode->sources()) {
    if (auto node = findPlanNodeById(child, id)) {
      return node;
    }
  }

  return nullptr;
}

std::function<core::PlanNodePtr(std::string, core::PlanNodePtr)> addTableWriter(
    const std::shared_ptr<const core::TableWriteNode>& node) {
  return [=](const core::PlanNodeId& nodeId,
             const core::PlanNodePtr& source) -> core::PlanNodePtr {
    return std::make_shared<core::TableWriteNode>(
        nodeId,
        node->columns(),
        node->columnNames(),
        node->aggregationNode(),
        node->insertTableHandle(),
        node->hasPartitioningScheme(),
        TableWriteTraits::outputType(node->aggregationNode()),
        node->commitStrategy(),
        source);
  };
}
} // namespace facebook::velox::exec
