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
#include "velox/core/PlanFragment.h"
#include "velox/core/QueryConfig.h"

namespace facebook::velox::core {

std::string_view PlanFragment::inputTransportType(
    const PlanNodeId& planNodeId) const {
  auto it = inputTransportTypes.find(planNodeId);
  return it != inputTransportTypes.end() ? std::string_view{it->second}
                                         : TransportKind::kHttp;
}

std::string_view PlanFragment::outputTransportType(
    const PlanNodeId& planNodeId) const {
  auto it = outputTransportTypes.find(planNodeId);
  return it != outputTransportTypes.end() ? std::string_view{it->second}
                                          : TransportKind::kHttp;
}

namespace {
// Checks that every node ID in 'transportTypes' refers to a node of type TNode
// in the plan tree rooted at 'root'. 'expectedNodeType' names TNode for error
// messages.
template <typename TNode>
void validateTransportNodeTypes(
    const PlanNode* root,
    const folly::F14FastMap<PlanNodeId, std::string>& transportTypes,
    std::string_view expectedNodeType) {
  for (const auto& entry : transportTypes) {
    const auto& planNodeId = entry.first;
    const auto* node = PlanNode::findNodeById(root, planNodeId);
    VELOX_USER_CHECK_NOT_NULL(
        node, "Transport type set for unknown plan node ID: {}", planNodeId);
    VELOX_USER_CHECK(
        node->is<TNode>(),
        "Transport type can only be set on {} nodes, but node '{}' is of "
        "type {}",
        expectedNodeType,
        planNodeId,
        node->name());
  }
}
} // namespace

void PlanFragment::validateTransportTypes() const {
  validateTransportNodeTypes<ExchangeNode>(
      planNode.get(), inputTransportTypes, "Exchange");
  validateTransportNodeTypes<PartitionedOutputNode>(
      planNode.get(), outputTransportTypes, "PartitionedOutput");
}

bool PlanFragment::canSpill(const QueryConfig& queryConfig) const {
  if (not queryConfig.spillEnabled()) {
    return false;
  }
  return PlanNode::findFirstNode(
             planNode.get(), [&](const core::PlanNode* node) {
               return node->canSpill(queryConfig);
             }) != nullptr;
}

const PlanNode* PlanFragment::firstNodeNotSupportingBarrier() const {
  return PlanNode::findFirstNode(
      planNode.get(),
      [&](const core::PlanNode* node) { return !node->supportsBarrier(); });
}

std::string executionStrategyToString(ExecutionStrategy strategy) {
  switch (strategy) {
    case ExecutionStrategy::kGrouped:
      return "GROUPED";
    case ExecutionStrategy::kUngrouped:
      return "UNGROUPED";
    default:
      return fmt::format("UNKNOWN: {}", static_cast<int>(strategy));
  }
}
} // namespace facebook::velox::core
