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
#include "velox/core/FixedPointPlanNodes.h"

namespace facebook::velox::core {
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

// Returns the schema of the VectorStateDeclaration named 'name' -- the fixed
// point's mutable output entry, whose schema is the node's output type.  Throws
// if no vector entry of that name is declared.
RowTypePtr outputEntrySchema(
    const std::vector<StateDeclarationPtr>& declarations,
    const std::string& name) {
  for (const auto& declaration : declarations) {
    auto vector =
        std::dynamic_pointer_cast<const VectorStateDeclaration>(declaration);
    if (vector != nullptr && vector->name() == name) {
      return vector->schema();
    }
  }
  VELOX_USER_FAIL(
      "FixedPointNode: outputStateEntry must name a declared vector state "
      "entry: {}",
      name);
}

} // namespace

FixedPointNode::FixedPointNode(
    PlanNodeId id,
    std::vector<StateDeclarationPtr> stateDeclarations,
    std::vector<PlanNodePtr> plans,
    ConvergenceConfig convergenceConfig,
    std::string outputStateEntry)
    : PlanNode{std::move(id)},
      stateDeclarations_{std::move(stateDeclarations)},
      plans_{std::move(plans)},
      convergenceConfig_{std::move(convergenceConfig)},
      outputStateEntry_{std::move(outputStateEntry)},
      outputType_{outputEntrySchema(stateDeclarations_, outputStateEntry_)} {
  validatePlans();
  // errorWhenMaxIterationReached fails the loop when maxIterations is reached
  // without converging; that is only meaningful with a convergence plan (a
  // null plan never converges, so it would always fail).
  VELOX_USER_CHECK(
      !convergenceConfig_.errorWhenMaxIterationReached ||
          convergenceConfig_.plan != nullptr,
      "FixedPointNode: errorWhenMaxIterationReached requires a convergence "
      "plan; set it false for a fixed-count loop with no convergence plan");
}

void FixedPointNode::addDetails(std::stringstream& stream) const {
  stream << "maxIterations: " << convergenceConfig_.maxIterations
         << ", plans: " << plans_.size();
}

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
    // The last plan produces the rows the framework writes back to the output
    // state entry, so it must not end with a PartitionedOutput; every earlier
    // plan shuffles to the next through one.
    if (i + 1 == numPlans) {
      VELOX_USER_CHECK(
          dynamic_cast<const core::PartitionedOutputNode*>(root) == nullptr,
          "FixedPointNode: the last plan must produce the rows written back to "
          "the output state entry, not shuffle through a PartitionedOutput, but "
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

namespace {
// Deserializes the single source of a node that has exactly one source.
PlanNodePtr deserializeSingleSource(const folly::dynamic& obj, void* context) {
  auto sources = ISerializable::deserialize<std::vector<PlanNode>>(
      obj["sources"], context);
  VELOX_CHECK_EQ(1, sources.size());
  return sources[0];
}
} // namespace

folly::dynamic StateDeclaration::serializeBase(std::string_view type) const {
  folly::dynamic obj = folly::dynamic::object;
  obj["type"] = type;
  obj["name"] = name_;
  if (initialPlan_ != nullptr) {
    obj["initialPlan"] = initialPlan_->serialize();
  }
  return obj;
}

// static
std::shared_ptr<const StateDeclaration> StateDeclaration::deserialize(
    const folly::dynamic& obj,
    void* context) {
  const auto type = obj["type"].asString();
  auto name = obj["name"].asString();
  PlanNodePtr initialPlan;
  if (obj.count("initialPlan") != 0u) {
    initialPlan =
        ISerializable::deserialize<PlanNode>(obj["initialPlan"], context);
  }
  auto schema = ISerializable::deserialize<RowType>(obj["schema"]);
  if (type == "vector") {
    return std::make_shared<VectorStateDeclaration>(
        std::move(name),
        schema,
        std::move(initialPlan),
        obj["append"].asBool());
  }
  if (type == "hashTable") {
    auto keyColumns =
        ISerializable::deserialize<std::vector<std::string>>(obj["keyColumns"]);
    return std::make_shared<HashTableStateDeclaration>(
        std::move(name), schema, std::move(keyColumns), std::move(initialPlan));
  }
  VELOX_FAIL("Unknown StateDeclaration type: {}", type);
}

folly::dynamic VectorStateDeclaration::serialize() const {
  auto obj = serializeBase("vector");
  obj["schema"] = schema_->serialize();
  obj["append"] = append_;
  return obj;
}

folly::dynamic HashTableStateDeclaration::serialize() const {
  auto obj = serializeBase("hashTable");
  obj["schema"] = schema_->serialize();
  obj["keyColumns"] = ISerializable::serialize(keyColumns_);
  return obj;
}

folly::dynamic ConvergenceConfig::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  if (plan != nullptr) {
    obj["plan"] = plan->serialize();
  }
  obj["convergedColumn"] = convergedColumn;
  obj["maxIterations"] = maxIterations;
  obj["errorWhenMaxIterationReached"] = errorWhenMaxIterationReached;
  return obj;
}

// static
ConvergenceConfig ConvergenceConfig::deserialize(
    const folly::dynamic& obj,
    void* context) {
  ConvergenceConfig config;
  if (obj.count("plan") != 0u) {
    config.plan = ISerializable::deserialize<PlanNode>(obj["plan"], context);
  }
  config.convergedColumn = obj["convergedColumn"].asString();
  config.maxIterations = static_cast<int32_t>(obj["maxIterations"].asInt());
  config.errorWhenMaxIterationReached =
      obj["errorWhenMaxIterationReached"].asBool();
  return config;
}

folly::dynamic FixedPointNode::serialize() const {
  auto obj = PlanNode::serialize();
  folly::dynamic declarations = folly::dynamic::array;
  for (const auto& declaration : stateDeclarations_) {
    declarations.push_back(declaration->serialize());
  }
  obj["stateDeclarations"] = std::move(declarations);
  folly::dynamic plans = folly::dynamic::array;
  for (const auto& plan : plans_) {
    plans.push_back(plan->serialize());
  }
  obj["plans"] = std::move(plans);
  obj["convergenceConfig"] = convergenceConfig_.serialize();
  obj["outputStateEntry"] = outputStateEntry_;
  return obj;
}

// static
PlanNodePtr FixedPointNode::create(const folly::dynamic& obj, void* context) {
  std::vector<StateDeclarationPtr> stateDeclarations;
  for (const auto& declaration : obj["stateDeclarations"]) {
    stateDeclarations.push_back(
        StateDeclaration::deserialize(declaration, context));
  }
  std::vector<PlanNodePtr> plans;
  for (const auto& plan : obj["plans"]) {
    plans.push_back(ISerializable::deserialize<PlanNode>(plan, context));
  }
  return std::make_shared<FixedPointNode>(
      obj["id"].asString(),
      std::move(stateDeclarations),
      std::move(plans),
      ConvergenceConfig::deserialize(obj["convergenceConfig"], context),
      obj["outputStateEntry"].asString());
}

folly::dynamic StateSourceNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["stateName"] = stateName_;
  obj["outputType"] = outputType_->serialize();
  obj["delta"] = delta_;
  return obj;
}

// static
PlanNodePtr StateSourceNode::create(
    const folly::dynamic& obj,
    void* /*context*/) {
  return std::make_shared<StateSourceNode>(
      obj["id"].asString(),
      obj["stateName"].asString(),
      ISerializable::deserialize<RowType>(obj["outputType"]),
      obj["delta"].asBool());
}

folly::dynamic StateHashJoinNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["stateName"] = stateName_;
  obj["probeKeys"] = ISerializable::serialize(probeKeys_);
  obj["outputType"] = outputType_->serialize();
  return obj;
}

// static
PlanNodePtr StateHashJoinNode::create(
    const folly::dynamic& obj,
    void* context) {
  return std::make_shared<StateHashJoinNode>(
      obj["id"].asString(),
      obj["stateName"].asString(),
      ISerializable::deserialize<std::vector<std::string>>(obj["probeKeys"]),
      ISerializable::deserialize<RowType>(obj["outputType"]),
      deserializeSingleSource(obj, context));
}

} // namespace facebook::velox::core
