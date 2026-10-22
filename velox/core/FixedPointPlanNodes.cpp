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
    const auto* vector = declaration->as<VectorStateDeclaration>();
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

// Every convergence plan in the order they run: 'plans', then 'verdict'.
std::vector<PlanNodePtr> allConvergencePlans(const ConvergenceConfig& config) {
  auto result = config.plans;
  if (config.verdict != nullptr) {
    result.push_back(config.verdict);
  }
  return result;
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
  // Contradictions in the config itself are reported before anything validates
  // the structure they describe, so the message names the contradiction rather
  // than a consequence of it.
  convergenceConfig_.checkConsistency();
  validatePlans();
  validateWorkerCounts();
  resolveAndValidateStateReferences();
  // The delta is the local shard's row count and the framework performs no
  // cross-worker reduction, so a peer's frontier may still be non-empty when
  // this one empties -- stopping then strands the peers reading this worker.
  if (convergenceConfig_.stopWhenDeltaEmpty) {
    VELOX_USER_CHECK_EQ(
        numWorkers(),
        1,
        "FixedPointNode: stopWhenDeltaEmpty requires a non-shuffling fixed "
        "point, because the delta is local to a worker");
  }
}

void FixedPointNode::addDetails(std::stringstream& stream) const {
  stream << "outputStateEntry: " << outputStateEntry_ << ", states: ["
         << declarationsToString(stateDeclarations_)
         << "], plans: " << plans_.size() << ", "
         << convergenceConfig_.toString();
}

namespace {

// The width a nested FixedPointNode shuffles across, or 1 when 'node' nests
// none.  Split requirements are a separate question, answered by
// containsSplitSource.
int32_t nestedWorkers(const PlanNodePtr& node) {
  if (node == nullptr) {
    return 1;
  }
  if (const auto* fixedPoint =
          dynamic_cast<const FixedPointNode*>(node.get())) {
    return fixedPoint->numWorkers();
  }
  int32_t workers{1};
  for (const auto& source : node->sources()) {
    workers = std::max(workers, nestedWorkers(source));
  }
  return workers;
}

// Whether a fixed point nested in 'node' reads its peers.
bool nestedReadsPeers(const PlanNodePtr& node) {
  if (node == nullptr) {
    return false;
  }
  if (const auto* fixedPoint = node->as<FixedPointNode>()) {
    return fixedPoint->readsPeers();
  }
  for (const auto& source : node->sources()) {
    if (nestedReadsPeers(source)) {
      return true;
    }
  }
  return false;
}

// Whether a plan chain makes the fixed point need coordinator-assigned peer
// splits: its first plan shuffles out, or any plan of it contains a node that
// consumes splits.  Without this a shuffling chain would report
// requiresSplits()==false and deadlock waiting for peers that were never
// named.
bool chainRequiresSplits(const std::vector<PlanNodePtr>& plans) {
  if (!plans.empty() &&
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
          plans.front()) != nullptr) {
    return true;
  }
  // Every plan, including the first: containsSplitSource reports any node
  // needing coordinator-assigned splits, which covers a later plan's Exchange,
  // a TableScan anywhere (including a join build side in the first plan), and
  // a nested FixedPointNode -- whose requiresSplits() it reaches, and which
  // gets its peers only from the enclosing loop.
  for (const auto& plan : plans) {
    if (containsSplitSource(plan)) {
      return true;
    }
  }
  return false;
}

// The number of workers a chain shuffles across -- the partition count its
// first plan's PartitionedOutput writes to -- or 1 when the chain does not
// shuffle.
int32_t chainWorkers(const std::vector<PlanNodePtr>& plans) {
  if (plans.empty()) {
    return 1;
  }
  if (auto partitioned =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
              plans.front())) {
    return partitioned->numPartitions();
  }
  return 1;
}

} // namespace

int32_t FixedPointNode::numWorkers() const {
  // validateWorkerCounts() rejected any stage that disagrees, so every value
  // here is either one or the single agreed width.
  int32_t workers = std::max(
      chainWorkers(plans_),
      chainWorkers(allConvergencePlans(convergenceConfig_)));
  for (const auto& plan : plans_) {
    workers = std::max(workers, nestedWorkers(plan));
  }
  for (const auto& plan : allConvergencePlans(convergenceConfig_)) {
    workers = std::max(workers, nestedWorkers(plan));
  }
  // An initial plan can host a nested loop too, and requiresSplits() already
  // counts it; leaving it out here would ask the coordinator for peers whose
  // width nothing validated.
  for (const auto& declaration : stateDeclarations_) {
    workers = std::max(workers, nestedWorkers(declaration->initialPlan()));
  }
  return workers;
}

void FixedPointNode::accept(
    const PlanNodeVisitor& visitor,
    PlanNodeVisitorContext& context) const {
  visitor.visit(*this, context);
}

void StateSourceNode::accept(
    const PlanNodeVisitor& visitor,
    PlanNodeVisitorContext& context) const {
  visitor.visit(*this, context);
}

void StateHashJoinNode::accept(
    const PlanNodeVisitor& visitor,
    PlanNodeVisitorContext& context) const {
  visitor.visit(*this, context);
}

std::vector<PlanNodePtr> FixedPointNode::nestedPlans() const {
  std::vector<PlanNodePtr> result;
  for (const auto& declaration : stateDeclarations_) {
    if (declaration->initialPlan() != nullptr) {
      result.push_back(declaration->initialPlan());
    }
  }
  result.insert(result.end(), plans_.begin(), plans_.end());
  const auto convergencePlans = allConvergencePlans(convergenceConfig_);
  result.insert(result.end(), convergencePlans.begin(), convergencePlans.end());
  return result;
}

bool FixedPointNode::readsPeers() const {
  // Either chain can be the shuffling one -- a convergence sequence that
  // reduces across workers needs peers just as a shuffling body does.  A loop
  // nested in an initial plan runs on this loop's workers too.
  if (chainRequiresSplits(plans_) ||
      chainRequiresSplits(allConvergencePlans(convergenceConfig_))) {
    return true;
  }
  for (const auto& declaration : stateDeclarations_) {
    if (nestedReadsPeers(declaration->initialPlan())) {
      return true;
    }
  }
  return false;
}

bool FixedPointNode::requiresSplits() const {
  if (readsPeers()) {
    return true;
  }
  for (const auto& declaration : stateDeclarations_) {
    if (containsSplitSource(declaration->initialPlan())) {
      return true;
    }
  }
  return false;
}

namespace {

// Validates the sub-plan chaining convention, which the body and convergence
// sequences share: the first plan reads state, every later one receives the
// previous one's shuffle through an Exchange, and only the last produces rows
// instead of shuffling through a PartitionedOutput.  'what' names a plan of the
// sequence and 'lastPlanProduces' what its last plan emits, both for the error
// messages.
// Counts the ExchangeNodes anywhere in 'node', not just on its primary input
// chain.  A fragment reading a second shuffle is a branching topology (a
// distributed join, say) rather than one link of a linear chain.
int32_t countExchanges(const PlanNode* node) {
  if (node == nullptr) {
    return 0;
  }
  int32_t numExchanges =
      dynamic_cast<const core::ExchangeNode*>(node) != nullptr ? 1 : 0;
  for (const auto& source : node->sources()) {
    numExchanges += countExchanges(source.get());
  }
  return numExchanges;
}

// Returns a node in 'node' that reads splits, or null.
// - Exchanges are left to the chain checks: the loop wires the one that links
//   a plan to its predecessor.
// - An index lookup reads splits for its lookup side only if the connector
//   asks for them.
// - A nested FixedPointNode gets its peers from this loop, but nothing feeds
//   its initial plans.  It validates its own per-iteration plans.
const PlanNode* findSplitSource(const PlanNode* node) {
  if (node->is<ExchangeNode>()) {
    return nullptr;
  }
  if (const auto* fixedPoint = node->as<FixedPointNode>()) {
    for (const auto& declaration : fixedPoint->stateDeclarations()) {
      if (declaration->initialPlan() == nullptr) {
        continue;
      }
      if (const auto* found =
              findSplitSource(declaration->initialPlan().get())) {
        return found;
      }
    }
    return nullptr;
  }
  const auto* lookupJoin = node->as<IndexLookupJoinNode>();
  if (node->requiresSplits() ||
      (lookupJoin != nullptr && lookupJoin->needsIndexSplit())) {
    return node;
  }
  for (const auto& source : node->sources()) {
    if (lookupJoin != nullptr && source == lookupJoin->lookupSource()) {
      continue;
    }
    if (const auto* found = findSplitSource(source.get())) {
      return found;
    }
  }
  return nullptr;
}

// 'lastLinkGathers' is true for a convergence sequence with a verdict plan,
// whose last shuffle gathers every worker's rows to one.
void validatePlanChain(
    const std::vector<PlanNodePtr>& plans,
    std::string_view what,
    std::string_view lastPlanProduces,
    bool lastLinkGathers) {
  const auto numPlans = plans.size();
  for (size_t i = 0; i < numPlans; ++i) {
    VELOX_USER_CHECK_NOT_NULL(
        plans[i], "FixedPointNode: {} {} must not be null", what, i);
    const auto* root = plans[i].get();
    const auto* leaf = primaryLeaf(root);
    const std::string leafName =
        leaf != nullptr ? std::string(leaf->name()) : "nothing";
    // The chain is one logical plan cut at its shuffle boundaries, so a
    // fragment reads at most the one shuffle that links it to its predecessor.
    // Counting every Exchange, not just the one on the primary input chain,
    // is what rejects a branching fragment.
    const auto numExchanges = countExchanges(root);
    if (i == 0) {
      VELOX_USER_CHECK(
          dynamic_cast<const StateSourceNode*>(leaf) != nullptr,
          "FixedPointNode: the first {} must start with a StateSourceNode, "
          "but it starts with {}",
          what,
          leafName);
      VELOX_USER_CHECK_EQ(
          numExchanges,
          0,
          "FixedPointNode: the first {} must not read a shuffle",
          what);
    } else {
      VELOX_USER_CHECK(
          dynamic_cast<const core::ExchangeNode*>(leaf) != nullptr,
          "FixedPointNode: every non-first {} must start with an Exchange, "
          "but {} {} starts with {}",
          what,
          what,
          i,
          leafName);
      VELOX_USER_CHECK_EQ(
          numExchanges,
          1,
          "FixedPointNode: every non-first {} must read exactly one shuffle, "
          "the link to its predecessor; a branching topology such as a "
          "distributed join is not supported. Plan: {}",
          what,
          i);
    }
    // The loop feeds splits only to its initial plans and to the Exchange that
    // links a plan to its predecessor; any other split source would wait for
    // splits forever.
    const auto* splitSource = findSplitSource(root);
    VELOX_USER_CHECK_NULL(
        splitSource,
        "FixedPointNode: only an initial plan can read splits; other plans "
        "must read their rows from state: {} {}, {} {}",
        what,
        i,
        splitSource->name(),
        splitSource->id());
    const std::string rootName =
        root != nullptr ? std::string(root->name()) : "nothing";
    if (i + 1 == numPlans) {
      VELOX_USER_CHECK(
          dynamic_cast<const core::PartitionedOutputNode*>(root) == nullptr,
          "FixedPointNode: the last {} must produce {}, not shuffle through a "
          "PartitionedOutput, but {} {} ends with {}",
          what,
          lastPlanProduces,
          what,
          i,
          rootName);
    } else {
      VELOX_USER_CHECK(
          dynamic_cast<const core::PartitionedOutputNode*>(root) != nullptr,
          "FixedPointNode: every non-last {} must end with a "
          "PartitionedOutput, but {} {} ends with {}",
          what,
          what,
          i,
          rootName);
    }
  }

  // Adjacent fragments are the two halves of one shuffle: what the
  // PartitionedOutput writes is what the next Exchange reads, so their schemas
  // must match, and every stage must shuffle across the same width or the
  // exchanges are built for peers that were never named.
  const auto workers = chainWorkers(plans);
  for (size_t i = 0; i + 1 < numPlans; ++i) {
    const auto partitioned =
        std::dynamic_pointer_cast<const core::PartitionedOutputNode>(plans[i]);
    if (!lastLinkGathers || i + 2 < numPlans) {
      VELOX_USER_CHECK_EQ(
          partitioned->numPartitions(),
          workers,
          "FixedPointNode: every shuffling {} must partition across the same "
          "number of workers. Plan: {}",
          what,
          i);
    }
    const auto* exchange = dynamic_cast<const core::ExchangeNode*>(
        primaryLeaf(plans[i + 1].get()));
    VELOX_USER_CHECK(
        exchange->outputType()->equivalent(*plans[i]->outputType()),
        "FixedPointNode: the schema a {} shuffles out must match what the next "
        "one reads back. Plan {} writes {}, plan {} reads {}",
        what,
        i,
        plans[i]->outputType()->toString(),
        i + 1,
        exchange->outputType()->toString());
  }
}

} // namespace

void FixedPointNode::validatePlans() const {
  VELOX_USER_CHECK(
      !plans_.empty(), "FixedPointNode requires at least one plan");
  validatePlanChain(
      plans_,
      "plan",
      "the rows written back to the output state entry",
      /*lastLinkGathers=*/false);
}

void FixedPointNode::validateWorkerCounts() const {
  // A stage that does not shuffle reports one and expresses no opinion.  Every
  // stage that does must name the same width: exchanges are wired for one peer
  // count, and a nested loop runs on the peers the enclosing one was assigned,
  // so a wider stage would wait on peers that were never named.
  int32_t agreed{1};
  std::string agreedBy;
  auto merge = [&](int32_t workers, const std::string& what) {
    if (workers == 1) {
      return;
    }
    VELOX_USER_CHECK(
        agreed == 1 || agreed == workers,
        "FixedPointNode: every stage must shuffle across the same number of "
        "workers. Stages: {} crosses {}, {} crosses {}",
        agreedBy,
        agreed,
        what,
        workers);
    agreed = workers;
    agreedBy = what;
  };
  merge(chainWorkers(plans_), "body");
  merge(chainWorkers(allConvergencePlans(convergenceConfig_)), "convergence");
  // Named with the index, because a nested loop's width is only traceable to
  // the plan hosting it.
  for (size_t i = 0; i < plans_.size(); ++i) {
    merge(nestedWorkers(plans_[i]), fmt::format("nested in body plan {}", i));
  }
  const auto convergencePlans = allConvergencePlans(convergenceConfig_);
  for (size_t i = 0; i < convergencePlans.size(); ++i) {
    merge(
        nestedWorkers(convergencePlans[i]),
        fmt::format("nested in convergence plan {}", i));
  }
  for (const auto& declaration : stateDeclarations_) {
    merge(
        nestedWorkers(declaration->initialPlan()),
        fmt::format(
            "nested in initial plan for state {}", declaration->name()));
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
  const auto convergencePlans = allConvergencePlans(convergenceConfig_);
  referencingPlans.insert(
      referencingPlans.end(), convergencePlans.begin(), convergencePlans.end());
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
      // Probe key i names the probe column matched against build key i, which
      // is column i of the hash table (keys first).
      for (int32_t key = 0; key < numKeys; ++key) {
        const auto& probeKey = join->probeKeys()[key];
        const auto probeChannel = probeType->getChildIdxIfExists(probeKey);
        VELOX_USER_CHECK(
            probeChannel.has_value(),
            "FixedPointNode: StateHashJoin probe key is not a probe input "
            "column: {}",
            probeKey);
        VELOX_USER_CHECK(
            probeType->childAt(*probeChannel)
                ->equivalent(*entrySchema->childAt(key)),
            "FixedPointNode: StateHashJoin probe key type must match the hash "
            "table build key type: {}",
            probeKey);
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
    if (const auto* vector = declaration->as<VectorStateDeclaration>()) {
      VELOX_USER_CHECK(
          initialPlan->outputType()->equivalent(*vector->schema()),
          "FixedPointNode: initial plan output must match the vector state "
          "entry schema for entry: {}",
          vector->name());
    } else if (
        const auto* hashTable = declaration->as<HashTableStateDeclaration>()) {
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
      .plans = {},
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = false};
}

// static
ConvergenceConfig ConvergenceConfig::whenDeltaEmpty(int32_t maxIterations) {
  return ConvergenceConfig{
      .plans = {},
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = true,
      .stopWhenDeltaEmpty = true};
}

// static
ConvergenceConfig ConvergenceConfig::converging(
    PlanNodePtr plan,
    int32_t maxIterations) {
  return ConvergenceConfig{
      .plans = {std::move(plan)},
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = true};
}

// static
ConvergenceConfig ConvergenceConfig::globalVerdict(
    std::vector<PlanNodePtr> partials,
    PlanNodePtr verdict,
    int32_t maxIterations) {
  return ConvergenceConfig{
      .plans = std::move(partials),
      .verdict = std::move(verdict),
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = true};
}

void ConvergenceConfig::checkConsistency() const {
  VELOX_USER_CHECK_GT(
      maxIterations, 0, "FixedPointNode: maxIterations must be positive");
  const auto convergencePlans = allConvergencePlans(*this);
  VELOX_USER_CHECK(
      !stopWhenDeltaEmpty || convergencePlans.empty(),
      "FixedPointNode: stopWhenDeltaEmpty and a convergence sequence are "
      "mutually exclusive; the delta row count is already the verdict");
  VELOX_USER_CHECK(
      verdict == nullptr || !plans.empty(),
      "FixedPointNode: a verdict plan requires convergence plans that "
      "gather to it");
  // Workers that each decide after a shuffle could disagree; one verdict plan
  // decides for all of them.
  VELOX_USER_CHECK(
      verdict != nullptr || plans.size() <= 1,
      "FixedPointNode: convergence plans that shuffle require a verdict plan; "
      "use ConvergenceConfig::globalVerdict");
  // Reaching maxIterations without converging fails the loop, which is only
  // meaningful with a convergence criterion: with none the loop never
  // converges, so it would always fail.
  VELOX_USER_CHECK(
      !errorWhenMaxIterationReached || !plans.empty() || stopWhenDeltaEmpty,
      "FixedPointNode: errorWhenMaxIterationReached requires a convergence "
      "criterion; set it false for a fixed-count loop with no convergence "
      "plan");
  if (convergencePlans.empty()) {
    return;
  }

  // The sequence chains like the body -- reading state, shuffling between
  // plans -- so the same check applies.
  validatePlanChain(
      convergencePlans,
      "convergence plan",
      "the BOOLEAN convergence verdict",
      /*lastLinkGathers=*/verdict != nullptr);

  // The verdict plan reads every worker's rows, so they must all reach the one
  // worker that runs it.
  if (verdict != nullptr) {
    static constexpr std::string_view kGatherRequired =
        "FixedPointNode: the convergence plans before a verdict plan must end "
        "in a PartitionedOutput with one partition";
    // validatePlanChain checked that the last of 'plans' ends in a
    // PartitionedOutput.
    const auto* gather = plans.back()->as<PartitionedOutputNode>();
    VELOX_USER_CHECK(gather->isPartitioned(), kGatherRequired);
    VELOX_USER_CHECK_EQ(gather->numPartitions(), 1, "{}", kGatherRequired);
    VELOX_USER_CHECK_EQ(
        nestedWorkers(verdict),
        1,
        "FixedPointNode: a fixed point nested in a verdict plan runs on worker 0 "
        "alone, so it cannot shuffle across workers");
  }

  const auto& verdictType = convergencePlans.back()->outputType();
  VELOX_USER_CHECK_EQ(
      verdictType->size(),
      1,
      "FixedPointNode: the convergence plan must produce exactly one output "
      "column");
  VELOX_USER_CHECK(
      verdictType->childAt(0)->isBoolean(),
      "FixedPointNode: the convergence plan output column must be BOOLEAN, but"
      " got: {}",
      verdictType->childAt(0)->toString());
}

folly::dynamic ConvergenceConfig::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  folly::dynamic serializedPlans = folly::dynamic::array;
  for (const auto& plan : plans) {
    serializedPlans.push_back(plan->serialize());
  }
  obj["plans"] = std::move(serializedPlans);
  if (verdict != nullptr) {
    obj["verdict"] = verdict->serialize();
  }
  obj["maxIterations"] = maxIterations;
  obj["errorWhenMaxIterationReached"] = errorWhenMaxIterationReached;
  obj["stopWhenDeltaEmpty"] = stopWhenDeltaEmpty;
  return obj;
}

std::string ConvergenceConfig::toString() const {
  std::string result = fmt::format("maxIterations: {}", maxIterations);
  if (stopWhenDeltaEmpty) {
    result += ", stopWhenDeltaEmpty";
  }
  if (!plans.empty()) {
    result += fmt::format(", convergencePlans: {}", plans.size());
  }
  if (verdict != nullptr) {
    result += ", verdict";
  }
  // Without a criterion the flag must be false, so it says something only
  // when a criterion exists.
  const bool hasCriterion = stopWhenDeltaEmpty || !plans.empty();
  if (hasCriterion && !errorWhenMaxIterationReached) {
    result += ", errorWhenMaxIterationReached: false";
  }
  return result;
}

// static
ConvergenceConfig ConvergenceConfig::deserialize(
    const folly::dynamic& obj,
    void* context) {
  ConvergenceConfig config;
  for (const auto& plan : obj["plans"]) {
    config.plans.push_back(ISerializable::deserialize<PlanNode>(plan, context));
  }
  if (obj.count("verdict")) {
    config.verdict =
        ISerializable::deserialize<PlanNode>(obj["verdict"], context);
  }
  config.maxIterations = static_cast<int32_t>(obj["maxIterations"].asInt());
  config.errorWhenMaxIterationReached =
      obj["errorWhenMaxIterationReached"].asBool();
  config.stopWhenDeltaEmpty = obj["stopWhenDeltaEmpty"].asBool();
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
