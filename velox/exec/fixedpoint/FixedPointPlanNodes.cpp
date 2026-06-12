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
#include "velox/exec/fixedpoint/FixedPointPlanNodes.h"

namespace facebook::velox::exec::fixedpoint {
namespace {

// The deepest node on a plan's primary (first-source) input chain -- the
// operator the plan "starts with".
const PlanNode* primaryLeaf(const PlanNode* node) {
  while (node != nullptr && !node->sources().empty()) {
    node = node->sources().front().get();
  }
  return node;
}

// True if 'node' or any of its sources requires splits (e.g. a TableScan or an
// Exchange) -- i.e. the coordinator must assign it source splits.
bool containsSplitSource(const PlanNodePtr& node) {
  if (node == nullptr) {
    return false;
  }
  if (node->requiresSplits()) {
    return true;
  }
  for (const auto& source : node->sources()) {
    if (containsSplitSource(source)) {
      return true;
    }
  }
  return false;
}

} // namespace

bool FixedPointNode::requiresSplits() const {
  if (!plans_.empty() &&
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
          plans_.front()) != nullptr) {
    return true;
  }
  for (const auto& declaration : stateDeclarations_) {
    if (containsSplitSource(declaration->initialPlan())) {
      return true;
    }
  }
  return false;
}

void FixedPointNode::validatePlans() const {
  VELOX_USER_CHECK(
      !plans_.empty(), "FixedPointNode requires at least one plan");
  const auto numPlans = plans_.size();
  for (size_t i = 0; i < numPlans; ++i) {
    const auto* root = plans_[i].get();
    const auto* leaf = primaryLeaf(root);
    const std::string leafName =
        leaf != nullptr ? std::string(leaf->name()) : "nothing";
    // The first plan reads from state; every later plan receives the previous
    // plan's shuffle through an Exchange.
    if (i == 0) {
      VELOX_USER_CHECK(
          dynamic_cast<const StateSourceNode*>(leaf) != nullptr,
          "FixedPointNode: the first plan must start with a StateSourceNode, "
          "but it starts with {}",
          leafName);
    } else {
      VELOX_USER_CHECK(
          dynamic_cast<const core::ExchangeNode*>(leaf) != nullptr,
          "FixedPointNode: every non-first plan must start with an Exchange, "
          "but plan {} starts with {}",
          i,
          leafName);
    }
    // The last plan writes state; every earlier plan shuffles to the next
    // through a PartitionedOutput.
    if (i + 1 == numPlans) {
      VELOX_USER_CHECK(
          dynamic_cast<const StateSinkNode*>(root) != nullptr,
          "FixedPointNode: the last plan must end with a StateSinkNode, but "
          "plan {} ends with {}",
          i,
          root->name());
    } else {
      VELOX_USER_CHECK(
          dynamic_cast<const core::PartitionedOutputNode*>(root) != nullptr,
          "FixedPointNode: every non-last plan must end with a "
          "PartitionedOutput, but plan {} ends with {}",
          i,
          root->name());
    }
  }
}

} // namespace facebook::velox::exec::fixedpoint
