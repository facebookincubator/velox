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

#include <folly/String.h>
#include <unordered_set>

namespace facebook::velox::core {
namespace {

// Returns the deepest node on a plan's primary (first-source) input chain --
// the operator the plan "starts with".
const PlanNode* primaryLeaf(const PlanNode* node) {
  while (node != nullptr && !node->sources().empty()) {
    node = node->sources().front().get();
  }
  return node;
}

// Returns true if 'node' or any of its sources requires splits (e.g. a
// TableScan or an Exchange) -- i.e. the coordinator must assign it source
// splits.
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

// Collects every node of type NodeT reachable from 'root' (inclusive) by a
// depth-first walk over sources.
template <typename NodeT>
void collectNodes(const PlanNodePtr& root, std::vector<const NodeT*>& out) {
  if (root == nullptr) {
    return;
  }
  if (const auto* typed = dynamic_cast<const NodeT*>(root.get())) {
    out.push_back(typed);
  }
  for (const auto& source : root->sources()) {
    collectNodes<NodeT>(source, out);
  }
}

// Finds the declaration of subtype DeclT named 'name', or nullptr if none.
template <typename DeclT>
const DeclT* findDeclaration(
    const std::vector<StateDeclarationPtr>& declarations,
    const std::string& name) {
  for (const auto& declaration : declarations) {
    const auto* typed = dynamic_cast<const DeclT*>(declaration.get());
    if (typed != nullptr && typed->name() == name) {
      return typed;
    }
  }
  return nullptr;
}

// Returns the number of payload (dependent) columns of a hash-table entry: the
// columns after the leading key columns.
int32_t numPayloadColumns(const HashTableStateDeclaration& declaration) {
  return static_cast<int32_t>(declaration.schema()->size()) -
      static_cast<int32_t>(declaration.keyColumns().size());
}

// Joins the state declarations' toString() outputs with ", " for plan printing.
std::string declarationsToString(
    const std::vector<StateDeclarationPtr>& declarations) {
  std::vector<std::string> parts;
  parts.reserve(declarations.size());
  for (const auto& declaration : declarations) {
    parts.push_back(declaration->toString());
  }
  return folly::join(", ", parts);
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
  VELOX_USER_CHECK_GT(
      convergenceConfig_.maxIterations,
      0,
      "FixedPointNode: maxIterations must be positive");
  validatePlans();
  resolveAndValidateStateReferences();
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
  stream << "outputStateEntry: " << outputStateEntry_ << ", states: ["
         << declarationsToString(stateDeclarations_)
         << "], plans: " << plans_.size() << ", "
         << convergenceConfig_.toString();
}

bool FixedPointNode::requiresSplits() const {
  if (!plans_.empty() &&
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
          plans_.front()) != nullptr) {
    return true;
  }
  // A non-first body plan that shuffles in (an Exchange) consumes splits the
  // coordinator must assign; without this a multi-plan shuffling fixed point
  // would report requiresSplits()==false and deadlock waiting for peers.
  for (size_t i = 1; i < plans_.size(); ++i) {
    if (containsSplitSource(plans_[i])) {
      return true;
    }
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
    VELOX_USER_CHECK_NOT_NULL(
        plans_[i], "FixedPointNode: plan {} must not be null", i);
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
          root != nullptr ? std::string(root->name()) : "nothing");
    } else {
      VELOX_USER_CHECK(
          dynamic_cast<const core::PartitionedOutputNode*>(root) != nullptr,
          "FixedPointNode: every non-last plan must end with a "
          "PartitionedOutput, but plan {} ends with {}",
          i,
          root != nullptr ? std::string(root->name()) : "nothing");
    }
  }
}

void FixedPointNode::resolveAndValidateStateReferences() const {
  // State declaration names are the linking mechanism, so they must be unique
  // across all kinds (vector + hash table); a duplicate makes resolution
  // ambiguous.
  std::unordered_set<std::string> declaredNames;
  for (const auto& declaration : stateDeclarations_) {
    VELOX_USER_CHECK(
        declaredNames.insert(declaration->name()).second,
        "FixedPointNode: duplicate state declaration name: {}",
        declaration->name());
  }

  // Resolve and check every StateSource / StateHashJoin in every body plan and
  // in the convergence plan (which also reads state via a StateSource).
  std::vector<PlanNodePtr> referencingPlans = plans_;
  if (convergenceConfig_.plan != nullptr) {
    referencingPlans.push_back(convergenceConfig_.plan);
  }
  for (const auto& plan : referencingPlans) {
    std::vector<const StateSourceNode*> sources;
    collectNodes<StateSourceNode>(plan, sources);
    for (const auto* source : sources) {
      const auto* entry = findDeclaration<VectorStateDeclaration>(
          stateDeclarations_, source->stateName());
      VELOX_USER_CHECK_NOT_NULL(
          entry,
          "FixedPointNode: StateSource references no declared vector state "
          "entry: {}",
          source->stateName());
      VELOX_USER_CHECK(
          source->outputType()->equivalent(*entry->schema()),
          "FixedPointNode: StateSource output type must match the state entry "
          "schema for entry: {}",
          source->stateName());
    }

    std::vector<const StateHashJoinNode*> joins;
    collectNodes<StateHashJoinNode>(plan, joins);
    for (const auto* join : joins) {
      const auto* entry = findDeclaration<HashTableStateDeclaration>(
          stateDeclarations_, join->stateName());
      VELOX_USER_CHECK_NOT_NULL(
          entry,
          "FixedPointNode: StateHashJoin references no declared hash table "
          "state entry: {}",
          join->stateName());
      VELOX_USER_CHECK_EQ(
          join->probeKeys().size(),
          entry->keyColumns().size(),
          "FixedPointNode: StateHashJoin probe key count must match the hash "
          "table key count for entry: {}",
          join->stateName());

      // The output is the probe input columns followed by the table's payload
      // (dependent) columns -- the columns after the leading key columns.
      const auto& probeType = join->sources().front()->outputType();
      const auto numProbe = static_cast<int32_t>(probeType->size());
      const auto numPayload = numPayloadColumns(*entry);
      VELOX_USER_CHECK_EQ(
          static_cast<int32_t>(join->outputType()->size()),
          numProbe + numPayload,
          "FixedPointNode: StateHashJoin output arity must equal probe columns "
          "plus hash table payload columns for entry: {}",
          join->stateName());
      const auto& entrySchema = entry->schema();
      const auto numKeys = static_cast<int32_t>(entry->keyColumns().size());
      // Keys-first contract: the probe input's leading key columns must occupy
      // the same channels and share the type of the hash table's build keys.
      for (int32_t channel = 0; channel < numKeys; ++channel) {
        VELOX_USER_CHECK(
            probeType->childAt(channel)->equivalent(
                *entrySchema->childAt(channel)),
            "FixedPointNode: StateHashJoin probe key column type at channel {} "
            "must match the hash table build key type for entry: {}",
            channel,
            join->stateName());
      }
      for (int32_t channel = 0; channel < numProbe; ++channel) {
        VELOX_USER_CHECK(
            join->outputType()->childAt(channel)->equivalent(
                *probeType->childAt(channel)),
            "FixedPointNode: StateHashJoin output probe column type mismatch at "
            "channel {} for entry: {}",
            channel,
            join->stateName());
      }
      for (int32_t channel = 0; channel < numPayload; ++channel) {
        VELOX_USER_CHECK(
            join->outputType()
                ->childAt(numProbe + channel)
                ->equivalent(*entrySchema->childAt(numKeys + channel)),
            "FixedPointNode: StateHashJoin output payload column type mismatch "
            "at channel {} for entry: {}",
            numProbe + channel,
            join->stateName());
      }
    }
  }

  // Each initial plan's output must match its declared entry schema.
  for (const auto& declaration : stateDeclarations_) {
    const auto& initialPlan = declaration->initialPlan();
    if (initialPlan == nullptr) {
      continue;
    }
    // Initial plans run in Phase 1 before any state is populated, so they must
    // not read state.
    std::vector<const StateSourceNode*> initialSources;
    collectNodes<StateSourceNode>(initialPlan, initialSources);
    std::vector<const StateHashJoinNode*> initialJoins;
    collectNodes<StateHashJoinNode>(initialPlan, initialJoins);
    VELOX_USER_CHECK(
        initialSources.empty() && initialJoins.empty(),
        "FixedPointNode: initial plan must not read state (it runs in Phase 1 "
        "before state is populated) for entry: {}",
        declaration->name());
    if (const auto* vector =
            dynamic_cast<const VectorStateDeclaration*>(declaration.get())) {
      VELOX_USER_CHECK(
          initialPlan->outputType()->equivalent(*vector->schema()),
          "FixedPointNode: initial plan output must match the vector state "
          "entry schema for entry: {}",
          vector->name());
    } else if (
        const auto* hashTable =
            dynamic_cast<const HashTableStateDeclaration*>(declaration.get())) {
      VELOX_USER_CHECK(
          initialPlan->outputType()->equivalent(*hashTable->schema()),
          "FixedPointNode: initial plan output must match the hash table state "
          "entry schema for entry: {}",
          hashTable->name());
    }
  }

  // The last plan's output must match the node's output type (the output
  // entry's schema), since the framework writes that output back into the
  // entry.
  VELOX_USER_CHECK(
      plans_.back()->outputType()->equivalent(*outputType_),
      "FixedPointNode: last plan output must match the output state entry "
      "schema for entry: {}",
      outputStateEntry_);

  // The convergence plan (when present) starts with a StateSourceNode and emits
  // exactly one BOOLEAN column.
  if (convergenceConfig_.plan != nullptr) {
    const auto* leaf = primaryLeaf(convergenceConfig_.plan.get());
    VELOX_USER_CHECK(
        dynamic_cast<const StateSourceNode*>(leaf) != nullptr,
        "FixedPointNode: the convergence plan must start with a StateSourceNode,"
        " but it starts with {}",
        leaf != nullptr ? std::string(leaf->name()) : "nothing");
    const auto& convergenceType = convergenceConfig_.plan->outputType();
    VELOX_USER_CHECK_EQ(
        convergenceType->size(),
        1,
        "FixedPointNode: the convergence plan must produce exactly one output "
        "column");
    VELOX_USER_CHECK(
        convergenceType->childAt(0)->isBoolean(),
        "FixedPointNode: the convergence plan output column must be BOOLEAN, but"
        " got: {}",
        convergenceType->childAt(0)->toString());
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

std::string VectorStateDeclaration::toString() const {
  return fmt::format(
      "{} (vector, {}{})",
      name(),
      append_ ? "append" : "replace",
      initialPlan() != nullptr ? ", initialized" : "");
}

VectorState& VectorState::initial(PlanNodePtr initialPlan) {
  initialPlan_ = std::move(initialPlan);
  return *this;
}

HashTableStateDeclaration::HashTableStateDeclaration(
    std::string name,
    RowTypePtr schema,
    std::vector<std::string> keyColumns,
    PlanNodePtr initialPlan)
    : StateDeclaration{std::move(name), std::move(initialPlan)},
      schema_{std::move(schema)},
      keyColumns_{std::move(keyColumns)} {
  VELOX_USER_CHECK(
      !keyColumns_.empty(),
      "HashTableStateDeclaration requires at least one key column for entry: {}",
      this->name());
  std::unordered_set<std::string> seenKeys;
  for (const auto& key : keyColumns_) {
    VELOX_USER_CHECK(
        schema_->containsChild(key),
        "HashTableStateDeclaration key column is not in the schema: {}",
        key);
    VELOX_USER_CHECK(
        seenKeys.insert(key).second,
        "HashTableStateDeclaration key columns must be unique, but got a "
        "duplicate: {}",
        key);
  }
  // Keys must be the leading schema columns, in order (keys-first).  Downstream
  // StateHashJoin validation and payload-column computation rely on this
  // positional invariant.
  const auto numKeys = static_cast<int32_t>(keyColumns_.size());
  for (int32_t i = 0; i < numKeys; ++i) {
    VELOX_USER_CHECK_EQ(
        schema_->nameOf(i),
        keyColumns_[i],
        "HashTableStateDeclaration key columns must be the leading schema "
        "columns in order (keys-first) for entry: {}",
        this->name());
  }
}

folly::dynamic HashTableStateDeclaration::serialize() const {
  auto obj = serializeBase("hashTable");
  obj["schema"] = schema_->serialize();
  obj["keyColumns"] = ISerializable::serialize(keyColumns_);
  return obj;
}

std::string HashTableStateDeclaration::toString() const {
  return fmt::format(
      "{} (hashTable, keys: [{}]{})",
      name(),
      folly::join(", ", keyColumns_),
      initialPlan() != nullptr ? ", initialized" : "");
}

HashTableState& HashTableState::initial(PlanNodePtr initialPlan) {
  initialPlan_ = std::move(initialPlan);
  return *this;
}

// static
ConvergenceConfig ConvergenceConfig::withMaxIterations(int32_t maxIterations) {
  return ConvergenceConfig{
      .plan = nullptr,
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = false};
}

// static
ConvergenceConfig ConvergenceConfig::converging(
    PlanNodePtr plan,
    int32_t maxIterations) {
  return ConvergenceConfig{
      .plan = std::move(plan),
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = true};
}

folly::dynamic ConvergenceConfig::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  if (plan != nullptr) {
    obj["plan"] = plan->serialize();
  }
  obj["maxIterations"] = maxIterations;
  obj["errorWhenMaxIterationReached"] = errorWhenMaxIterationReached;
  return obj;
}

std::string ConvergenceConfig::toString() const {
  return fmt::format(
      "maxIterations: {}, errorWhenMaxIterationReached: {}, "
      "convergencePlan: {}",
      maxIterations,
      errorWhenMaxIterationReached ? "true" : "false",
      plan != nullptr ? "present" : "none");
}

// static
ConvergenceConfig ConvergenceConfig::deserialize(
    const folly::dynamic& obj,
    void* context) {
  ConvergenceConfig config;
  if (obj.count("plan") != 0u) {
    config.plan = ISerializable::deserialize<PlanNode>(obj["plan"], context);
  }
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

StateHashJoinNode::StateHashJoinNode(
    PlanNodeId id,
    std::string stateName,
    std::vector<std::string> probeKeys,
    RowTypePtr outputType,
    PlanNodePtr source)
    : PlanNode{std::move(id)},
      stateName_{std::move(stateName)},
      probeKeys_{std::move(probeKeys)},
      outputType_{std::move(outputType)},
      sources_{std::move(source)} {
  VELOX_USER_CHECK_NOT_NULL(
      sources_[0], "StateHashJoin requires a non-null probe source");
  VELOX_USER_CHECK(
      !probeKeys_.empty(), "StateHashJoin requires at least one probe key");
}

folly::dynamic StateHashJoinNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["stateName"] = stateName_;
  obj["probeKeys"] = ISerializable::serialize(probeKeys_);
  obj["outputType"] = outputType_->serialize();
  return obj;
}

void StateHashJoinNode::addDetails(std::stringstream& stream) const {
  stream << "state: " << stateName_ << ", probeKeys: ["
         << folly::join(", ", probeKeys_) << "]";
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
