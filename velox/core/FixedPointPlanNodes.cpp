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

#include <folly/CppAttributes.h>
#include <folly/String.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include <algorithm>
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
  if (const auto* typed = root->template as<NodeT>()) {
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
    const auto* typed = declaration->template as<DeclT>();
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

// Returns the stages of a chain, in which each of 'plans' after the first
// reads the one before through its only Exchange.  A plan runs where the
// shuffle it reads delivers rows: on worker 0 for a gather, on one worker per
// partition for any other partitioned shuffle, and on every worker for a
// broadcast.  The first plan runs on every worker.  'what' names a plan of the
// chain for the error messages.
std::vector<Stage> chainStages(
    std::vector<PlanNodePtr> plans,
    std::string_view what) {
  VELOX_USER_CHECK(!plans.empty(), "FixedPointNode requires at least one plan");
  std::vector<Stage> stages;
  stages.reserve(plans.size());
  for (size_t i = 0; i < plans.size(); ++i) {
    VELOX_USER_CHECK_NOT_NULL(
        plans[i], "FixedPointNode: {} {} must not be null", what, i);
    Stage stage{.id = static_cast<int32_t>(i + 1)};
    if (i > 0) {
      std::vector<const ExchangeNode*> exchanges;
      collectNodes<ExchangeNode>(plans[i], exchanges);
      VELOX_USER_CHECK_EQ(
          exchanges.size(),
          1,
          "FixedPointNode: every chained plan after the first must read exactly "
          "one shuffle, the link to its predecessor. Plan: {}",
          i);
      stage.inputs.push_back({
          .exchangeId = exchanges.front()->id(),
          .producerStageId = static_cast<int32_t>(i),
      });
    }
    stage.plan = std::move(plans[i]);
    stages.push_back(std::move(stage));
  }
  return FixedPointNode::sizeStagesToShuffles(std::move(stages));
}

} // namespace

// static
std::vector<Stage> FixedPointNode::sizeStagesToShuffles(
    std::vector<Stage> stages) {
  folly::F14FastMap<int32_t, int32_t> numPartitions;
  for (const auto& stage : stages) {
    if (stage.plan == nullptr) {
      continue;
    }
    const auto* output = stage.plan->as<PartitionedOutputNode>();
    if (output != nullptr && output->isPartitioned()) {
      numPartitions.emplace(stage.id, output->numPartitions());
    }
  }
  for (auto& stage : stages) {
    for (const auto& input : stage.inputs) {
      const auto shuffle = numPartitions.find(input.producerStageId);
      if (shuffle == numPartitions.end()) {
        continue;
      }
      if (shuffle->second == 1) {
        stage.type = StageType::kSingle;
        stage.numRemotePartitions.reset();
      } else {
        stage.type = StageType::kFixed;
        stage.numRemotePartitions = shuffle->second;
      }
    }
  }
  return stages;
}

FixedPointNode::FixedPointNode(
    PlanNodeId id,
    std::vector<StateDeclarationPtr> stateDeclarations,
    MultiStagePlan body,
    ConvergenceConfig convergenceConfig,
    std::string outputStateEntry,
    bool outputOnWorkerZero)
    : PlanNode{std::move(id)},
      stateDeclarations_{std::move(stateDeclarations)},
      body_{std::move(body)},
      convergenceConfig_{std::move(convergenceConfig)},
      outputStateEntry_{std::move(outputStateEntry)},
      outputOnWorkerZero_{outputOnWorkerZero},
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
  // Worker 0 publishes iteration t+1's verdict only after every worker has
  // received t's (see FixedPointOptions::publishVerdict).  Gathering every
  // worker's partial rows ensures that; for a root that reads worker 0's own
  // state, the body's shuffle does: worker 0's iteration reads rows every peer
  // produces in the same iteration.
  if (convergenceConfig_.verdictOnWorkerZero() &&
      convergenceConfig_.plan->root().inputs.empty()) {
    VELOX_USER_CHECK_GT(
        body_.stages().size(),
        1,
        "FixedPointNode: a verdict plan without convergence plans requires a "
        "body that shuffles");
  }
}

FixedPointNode::FixedPointNode(
    PlanNodeId id,
    std::vector<StateDeclarationPtr> stateDeclarations,
    std::vector<PlanNodePtr> plans,
    ConvergenceConfig convergenceConfig,
    std::string outputStateEntry,
    bool outputOnWorkerZero)
    : FixedPointNode(
          std::move(id),
          std::move(stateDeclarations),
          MultiStagePlan(chainStages(std::move(plans), "plan")),
          std::move(convergenceConfig),
          std::move(outputStateEntry),
          outputOnWorkerZero) {}

void FixedPointNode::addDetails(std::stringstream& stream) const {
  stream << "outputStateEntry: " << outputStateEntry_ << ", states: ["
         << declarationsToString(stateDeclarations_)
         << "], stages: " << body_.stages().size() << ", "
         << convergenceConfig_.toString();
  if (outputOnWorkerZero_) {
    stream << ", outputOnWorkerZero";
  }
}

namespace {

// The width a nested FixedPointNode shuffles across, or 1 when 'node' nests
// none.  Split requirements are a separate question, answered by
// containsSplitSource.
int32_t nestedWorkers(const PlanNodePtr& node) {
  if (node == nullptr) {
    return 1;
  }
  if (const auto* fixedPoint = node->as<FixedPointNode>()) {
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

// Whether 'plan' makes the fixed point need coordinator-assigned peer splits:
// it shuffles, which takes more than one stage, or any of its plans contains a
// node that consumes splits.  Without this a shuffling fixed point would report
// requiresSplits()==false and deadlock waiting for peers that were never named.
bool stagesRequireSplits(const MultiStagePlan& plan) {
  if (plan.stages().size() > 1) {
    return true;
  }
  // containsSplitSource reports any node needing coordinator-assigned splits:
  // a TableScan anywhere, including a join build side, and a nested
  // FixedPointNode -- whose requiresSplits() it reaches, and which gets its
  // peers only from the enclosing loop.
  for (const auto& stage : plan.stages()) {
    if (containsSplitSource(stage.plan)) {
      return true;
    }
  }
  return false;
}

// Whether 'stage' runs on every worker of a fixed point that runs as
// 'numWorkers' workers (see FixedPointNode::numWorkers()).  Worker 0, which
// runs a kSingle stage, is every worker only when there is one, so
// FixedPointNode::requiresSingleWorker() then holds the loop to one.
bool runsOnEveryWorker(const Stage& stage, int32_t numWorkers) {
  return stage.type == StageType::kSource ||
      (stage.type == StageType::kFixed &&
       stage.numRemotePartitions == numWorkers) ||
      (stage.type == StageType::kSingle && numWorkers == 1);
}

// Whether 'plan' reads the fixed point's state.  A fixed point nested in it
// reads its own.
bool readsState(const PlanNodePtr& plan) {
  std::vector<const StateSourceNode*> sources;
  collectNodes<StateSourceNode>(plan, sources);
  std::vector<const StateHashJoinNode*> joins;
  collectNodes<StateHashJoinNode>(plan, joins);
  return !sources.empty() || !joins.empty();
}

// The root of 'convergence' when it is that of workerZeroVerdict: a kSingle
// root that reads no stage, over state every worker holds in full, so it may
// read state on worker 0 alone.  Null otherwise.
const Stage* FOLLY_NULLABLE
workerZeroVerdictRoot(const ConvergenceConfig& convergence) {
  if (!convergence.verdictOnWorkerZero()) {
    return nullptr;
  }
  const auto& root = convergence.plan->root();
  return root.inputs.empty() ? &root : nullptr;
}

// The most workers any stage of 'plan' runs on, counting a stage that runs on
// every worker as 1: a kFixed stage's tasks, or those of a fixed point nested
// in any stage.
int32_t widestStage(const MultiStagePlan& plan) {
  int32_t workers{1};
  for (const auto& stage : plan.stages()) {
    workers = std::max(
        {workers,
         stage.numTasks(/*numSourceTasks=*/1),
         nestedWorkers(stage.plan)});
  }
  return workers;
}

} // namespace

int32_t FixedPointNode::numWorkers() const {
  int32_t workers = widestStage(body_);
  if (convergenceConfig_.plan.has_value()) {
    workers = std::max(workers, widestStage(*convergenceConfig_.plan));
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
  for (const auto& stage : body_.stages()) {
    result.push_back(stage.plan);
  }
  if (convergenceConfig_.plan.has_value()) {
    for (const auto& stage : convergenceConfig_.plan->stages()) {
      result.push_back(stage.plan);
    }
  }
  return result;
}

bool FixedPointNode::requiresSingleWorker() const {
  if (body_.root().type == StageType::kSingle) {
    return true;
  }
  auto readsStateOnWorkerZero = [](const Stage& stage) {
    return stage.type == StageType::kSingle && readsState(stage.plan);
  };
  if (std::any_of(
          body_.stages().begin(),
          body_.stages().end(),
          readsStateOnWorkerZero)) {
    return true;
  }
  if (!convergenceConfig_.plan.has_value()) {
    return false;
  }
  const auto* exempt = workerZeroVerdictRoot(convergenceConfig_);
  return std::any_of(
      convergenceConfig_.plan->stages().begin(),
      convergenceConfig_.plan->stages().end(),
      [&](const Stage& stage) {
        return &stage != exempt && readsStateOnWorkerZero(stage);
      });
}

bool FixedPointNode::readsPeers() const {
  // Either the body or the convergence can be the shuffling one -- a
  // convergence that reduces across workers needs peers just as a shuffling
  // body does.  A loop nested in an initial plan runs on this loop's workers
  // too.
  if (stagesRequireSplits(body_) ||
      (convergenceConfig_.plan.has_value() &&
       stagesRequireSplits(*convergenceConfig_.plan))) {
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

// Returns a node in 'node' that reads splits, or null.
// - Exchanges are left to the tree checks: the loop wires each one to the
//   stage that names it.
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

// Validates the stages of a fixed point's body or convergence as a plan that a
// fixed point can run on its workers each iteration.  'what' names a plan of
// the stages and 'rootProduces' what their root emits, both for the error
// messages.
void validateStages(
    const MultiStagePlan& plan,
    std::string_view what,
    std::string_view rootProduces) {
  const auto& stages = plan.stages();
  for (size_t i = 0; i < stages.size(); ++i) {
    const auto& stage = stages[i];
    const auto& node = *stage.plan;
    VELOX_USER_CHECK_NE(
        stage.type,
        StageType::kCoordinator,
        "FixedPointNode: a fixed point runs on its workers, not on the "
        "coordinator. Plan: {} {}",
        what,
        i);
    VELOX_USER_CHECK(
        stage.type != StageType::kSource ||
            !stage.numRemotePartitions.has_value(),
        "FixedPointNode: a kSource {} runs on every worker, so it takes no "
        "numRemotePartitions. Plan: {}",
        what,
        i);
    VELOX_USER_CHECK(
        stage.type != StageType::kFixed ||
            stage.numRemotePartitions.value() > 1,
        "FixedPointNode: a kFixed {} must run more than one task; one task is "
        "kSingle. Plan: {}",
        what,
        i);
    if (&stage == &plan.root()) {
      VELOX_USER_CHECK(
          !node.is<PartitionedOutputNode>(),
          "FixedPointNode: the root {} must produce {}, not shuffle through a "
          "PartitionedOutput, but {} {} ends with {}",
          what,
          rootProduces,
          what,
          i,
          node.name());
    }
    if (stage.inputs.empty()) {
      const auto* leaf = primaryLeaf(&node);
      VELOX_USER_CHECK(
          leaf->is<StateSourceNode>(),
          "FixedPointNode: a {} that reads no shuffle must start with a "
          "StateSourceNode, but {} {} starts with {}",
          what,
          what,
          i,
          leaf->name());
    }

    // The loop does not feed the initial plans of a fixed point nested in a
    // stage; collectNodes() does not enter them.
    std::vector<const ExchangeNode*> exchanges;
    collectNodes<ExchangeNode>(stage.plan, exchanges);
    folly::F14FastSet<PlanNodeId> ownExchanges;
    for (const auto* exchange : exchanges) {
      ownExchanges.insert(exchange->id());
    }
    for (const auto& input : stage.inputs) {
      VELOX_USER_CHECK(
          ownExchanges.contains(input.exchangeId),
          "FixedPointNode: a {} must not feed an Exchange in a nested fixed "
          "point's initial plan. Plan: {}, Exchange: {}",
          what,
          i,
          input.exchangeId);
      // A partitioned shuffle hands each partition to the task with its
      // index, so its reader runs exactly one task per partition.
      const auto* producer =
          plan.stage(input.producerStageId).plan->as<PartitionedOutputNode>();
      VELOX_USER_CHECK(
          !producer->isPartitioned() || stage.type != StageType::kSource,
          "FixedPointNode: a {} that reads a partitioned shuffle must run one "
          "task per partition: kFixed, or kSingle for a gather. Plan: {}",
          what,
          i);
    }

    // The loop feeds splits only to its initial plans and to the Exchanges that
    // read other stages; any other split source would wait for splits forever.
    const auto* splitSource = findSplitSource(&node);
    VELOX_USER_CHECK_NULL(
        splitSource,
        "FixedPointNode: only an initial plan can read splits; other plans "
        "must read their rows from state: {} {}, {} {}",
        what,
        i,
        splitSource->name(),
        splitSource->id());

    // A fixed point nested in a stage runs on that stage's workers.
    if (stage.type != StageType::kSource) {
      const auto nested = nestedWorkers(stage.plan);
      const auto numTasks = stage.numTasks(/*numSourceTasks=*/1);
      VELOX_USER_CHECK(
          nested == 1 || nested == numTasks,
          "FixedPointNode: a fixed point nested in {} {} runs on the {} "
          "worker(s) that run it, so it cannot shuffle across {}",
          what,
          i,
          numTasks,
          nested);
    }
  }
}

} // namespace

void FixedPointNode::validatePlans() const {
  validateStages(
      body_, "plan", "the rows written back to the output state entry");
}

void FixedPointNode::validateWorkerCounts() const {
  const auto workers = numWorkers();
  // Every worker writes the rows the root produces into its own state.
  const auto& root = body_.root();
  VELOX_USER_CHECK(
      runsOnEveryWorker(root, workers),
      "FixedPointNode: the root plan writes each worker's state, so it must run "
      "on every worker: kSource, or kFixed across all {} workers, the most any "
      "stage or nested fixed point runs on, but it is {}",
      workers,
      root.type);

  // Each worker holds a partition of every state entry, so a plan that runs on
  // fewer workers would read part of it and produce wrong rows without an
  // error.  The root of workerZeroVerdict reads state every worker holds in
  // full.
  auto checkStateReads = [&](const MultiStagePlan& plan,
                             const Stage* FOLLY_NULLABLE exempt,
                             std::string_view what) {
    const auto& stages = plan.stages();
    for (size_t i = 0; i < stages.size(); ++i) {
      if (&stages[i] == exempt) {
        continue;
      }
      VELOX_USER_CHECK(
          runsOnEveryWorker(stages[i], workers) || !readsState(stages[i].plan),
          "FixedPointNode: a {} that reads state must run on every worker, "
          "because each worker holds part of it; a {} that runs on fewer "
          "workers reads its rows through a shuffle. Plan: {}, type: {}",
          what,
          what,
          i,
          stages[i].type);
    }
  };
  checkStateReads(body_, /*exempt=*/nullptr, "plan");
  if (convergenceConfig_.plan.has_value()) {
    checkStateReads(
        *convergenceConfig_.plan,
        workerZeroVerdictRoot(convergenceConfig_),
        "convergence plan");
  }

  // The rest of the stages' widths fit by construction: 'workers' is the
  // widest.  A fixed point nested in a plan that runs on every worker must
  // shuffle across all of them, or not at all.
  auto checkNested = [&](const PlanNodePtr& plan, const std::string& what) {
    const auto nested = nestedWorkers(plan);
    VELOX_USER_CHECK(
        nested == 1 || nested == workers,
        "FixedPointNode: every stage must shuffle across the same number of "
        "workers. A fixed point nested in {} shuffles across {}, but the fixed "
        "point runs on {}",
        what,
        nested,
        workers);
  };
  auto checkNestedInStages = [&](const MultiStagePlan& plan,
                                 std::string_view what) {
    const auto& stages = plan.stages();
    for (size_t i = 0; i < stages.size(); ++i) {
      if (stages[i].type == StageType::kSource) {
        checkNested(stages[i].plan, fmt::format("{} {}", what, i));
      }
    }
  };
  checkNestedInStages(body_, "body plan");
  if (convergenceConfig_.plan.has_value()) {
    checkNestedInStages(*convergenceConfig_.plan, "convergence plan");
  }
  for (const auto& declaration : stateDeclarations_) {
    checkNested(
        declaration->initialPlan(),
        fmt::format("the initial plan for state {}", declaration->name()));
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

  // Resolve and check every StateSource / StateHashJoin in every plan of the
  // body and of the convergence (which also reads state via a StateSource).
  std::vector<PlanNodePtr> referencingPlans;
  for (const auto& stage : body_.stages()) {
    referencingPlans.push_back(stage.plan);
  }
  if (convergenceConfig_.plan.has_value()) {
    for (const auto& stage : convergenceConfig_.plan->stages()) {
      referencingPlans.push_back(stage.plan);
    }
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

  // The root's output must match the node's output type (the output entry's
  // schema), since the framework writes that output back into the entry.
  VELOX_USER_CHECK(
      body_.root().plan->outputType()->equivalent(*outputType_),
      "FixedPointNode: root plan output must match the output state entry "
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

bool ConvergenceConfig::verdictOnWorkerZero() const {
  return plan.has_value() && plan->root().type == StageType::kSingle;
}

// static
ConvergenceConfig ConvergenceConfig::withMaxIterations(int32_t maxIterations) {
  return ConvergenceConfig{
      .maxIterations = maxIterations, .errorWhenMaxIterationReached = false};
}

// static
ConvergenceConfig ConvergenceConfig::whenDeltaEmpty(int32_t maxIterations) {
  return ConvergenceConfig{
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = true,
      .stopWhenDeltaEmpty = true};
}

// static
ConvergenceConfig ConvergenceConfig::converging(
    PlanNodePtr plan,
    int32_t maxIterations) {
  return converging(
      MultiStagePlan(chainStages({std::move(plan)}, "convergence plan")),
      maxIterations);
}

// static
ConvergenceConfig ConvergenceConfig::globalVerdict(
    std::vector<PlanNodePtr> partials,
    PlanNodePtr verdict,
    int32_t maxIterations) {
  partials.push_back(std::move(verdict));
  auto stages = chainStages(std::move(partials), "convergence plan");
  stages.back().type = StageType::kSingle;
  stages.back().numRemotePartitions.reset();
  return converging(MultiStagePlan(std::move(stages)), maxIterations);
}

// static
ConvergenceConfig ConvergenceConfig::workerZeroVerdict(
    PlanNodePtr verdict,
    int32_t maxIterations) {
  auto stages = chainStages({std::move(verdict)}, "convergence plan");
  stages.back().type = StageType::kSingle;
  return converging(MultiStagePlan(std::move(stages)), maxIterations);
}

// static
ConvergenceConfig ConvergenceConfig::converging(
    MultiStagePlan plan,
    int32_t maxIterations) {
  return ConvergenceConfig{
      .plan = std::move(plan),
      .maxIterations = maxIterations,
      .errorWhenMaxIterationReached = true};
}

void ConvergenceConfig::checkConsistency() const {
  VELOX_USER_CHECK_GT(
      maxIterations, 0, "FixedPointNode: maxIterations must be positive");
  VELOX_USER_CHECK(
      !stopWhenDeltaEmpty || !plan.has_value(),
      "FixedPointNode: stopWhenDeltaEmpty and a convergence sequence are "
      "mutually exclusive; the delta row count is already the verdict");
  // Reaching maxIterations without converging fails the loop, which is only
  // meaningful with a convergence criterion: with none the loop never
  // converges, so it would always fail.
  VELOX_USER_CHECK(
      !errorWhenMaxIterationReached || plan.has_value() || stopWhenDeltaEmpty,
      "FixedPointNode: errorWhenMaxIterationReached requires a convergence "
      "criterion; set it false for a fixed-count loop with no convergence "
      "plan");
  if (!plan.has_value()) {
    return;
  }

  validateStages(*plan, "convergence plan", "the BOOLEAN convergence verdict");

  // Every worker needs the verdict: each computes it, or worker 0 shares its.
  const auto& root = plan->root();
  VELOX_USER_CHECK(
      root.type == StageType::kSource || root.type == StageType::kSingle,
      "FixedPointNode: the root convergence plan must run on every worker "
      "(kSource) or on worker 0, which shares its verdict (kSingle), but it is "
      "{}",
      root.type);
  // Workers that each decide must decide alike, over state the body keeps
  // identical on every worker.  Rows read through a shuffle can arrive in a
  // different order on each worker, and a floating-point verdict over them
  // could differ.
  VELOX_USER_CHECK(
      root.type != StageType::kSource || root.inputs.empty(),
      "FixedPointNode: a convergence root that runs on every worker must read "
      "no stage; gather the rows to a kSingle root instead");

  const auto& verdictType = root.plan->outputType();
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
  if (plan.has_value()) {
    obj["plan"] = plan->serialize();
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
  if (plan.has_value()) {
    result += fmt::format(", convergenceStages: {}", plan->stages().size());
  }
  if (verdictOnWorkerZero()) {
    result += ", verdictOnWorkerZero";
  }
  // Without a criterion the flag must be false, so it says something only
  // when a criterion exists.
  const bool hasCriterion = stopWhenDeltaEmpty || plan.has_value();
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
  if (obj.count("plan") != 0u) {
    config.plan = MultiStagePlan::deserialize(obj["plan"], context);
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
  obj["body"] = body_.serialize();
  obj["convergenceConfig"] = convergenceConfig_.serialize();
  obj["outputStateEntry"] = outputStateEntry_;
  if (outputOnWorkerZero_) {
    obj["outputOnWorkerZero"] = true;
  }
  return obj;
}

// static
PlanNodePtr FixedPointNode::create(const folly::dynamic& obj, void* context) {
  std::vector<StateDeclarationPtr> stateDeclarations;
  for (const auto& declaration : obj["stateDeclarations"]) {
    stateDeclarations.push_back(
        StateDeclaration::deserialize(declaration, context));
  }
  return std::make_shared<FixedPointNode>(
      obj["id"].asString(),
      std::move(stateDeclarations),
      MultiStagePlan::deserialize(obj["body"], context),
      ConvergenceConfig::deserialize(obj["convergenceConfig"], context),
      obj["outputStateEntry"].asString(),
      obj.getDefault("outputOnWorkerZero", false).asBool());
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
