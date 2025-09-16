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
#include "velox/experimental/streaming/StreamingPlanNode.h"

namespace facebook::velox::streaming {

const std::vector<core::PlanNodePtr>& StreamingPlanNode::sources() const {
  return node_->sources();
}

void StreamingPlanNode::addDetails(std::stringstream& stream) const {
  stream << "Node: " << node_->toString(true, true);
  stream << "Targets: [" << std::endl;
  for (auto target : targets_) {
    stream << target->toString(true, true) << "," << std::endl;
  }
  stream << "]" << std::endl;
}

folly::dynamic StreamingPlanNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["node"] = node_->serialize();
  obj["targets"] = folly::dynamic::array;
  for (const auto& target : targets_) {
    obj["targets"].push_back(target->serialize());
  }
  return obj;
}

// static
core::PlanNodePtr StreamingPlanNode::create(const folly::dynamic& obj, void* context) {
  auto node = ISerializable::deserialize<core::PlanNode>(
      obj["node"], context);
  auto targets = std::vector<core::PlanNodePtr>();
  if (obj.count("targets")) {
    targets = ISerializable::deserialize<std::vector<core::PlanNode>>(
        obj["targets"], context);
  }

  return std::make_shared<const StreamingPlanNode>(std::move(node), std::move(targets));
}

void StreamingPlanNode::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();

  registry.Register("StreamingPlanNode", StreamingPlanNode::create);
}

} // namespace facebook::velox::streaming
