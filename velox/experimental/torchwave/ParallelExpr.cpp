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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <folly/container/F14Map.h>

#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {

namespace {

// Extra dependency edges injected by side-effect analysis (see
// computeSideEffectEdges): maps a node to in-place mutation nodes it must run
// after (or, for a mutation, the earlier touches of the same storage it must
// run after). Consulted by args() so the existing layering orders mutations
// relative to the reads/writes of the storage they alias. Null when no
// side-effect analysis is active (e.g. during reachability computation).
// thread_local so concurrent compiles of different graphs don't race on it
// (the rest of torchwave keeps compile/execution state thread_local).
thread_local const std::unordered_map<NodeCP, std::vector<NodeCP>>* gExtraArgs =
    nullptr;

// Returns the input nodes of 'expr'. Loops over inputs() of the node and adds
// the producer of the Value of each NamedArgument if not null, plus any
// side-effect dependency edges recorded for 'expr'.
std::vector<NodeCP> args(NodeCP expr) {
  std::vector<NodeCP> result;
  for (const auto& input : expr->inputs()) {
    auto* producer = input.value->producer();
    if (producer != nullptr && producer != expr) {
      result.push_back(producer);
    }
  }
  if (gExtraArgs != nullptr) {
    auto it = gExtraArgs->find(expr);
    if (it != gExtraArgs->end()) {
      result.insert(result.end(), it->second.begin(), it->second.end());
    }
  }
  return result;
}

// Returns true if 'expr' is a call-like expression (has arguments).
bool isCallExpr(NodeCP expr) {
  return !args(expr).empty();
}

// Adds 'expr' and all its transitive subexpressions to 'result'.
void subexpressionsInner(NodeCP expr, NodeSet& result) {
  for (auto arg : args(expr)) {
    if (result.count(arg)) {
      continue;
    }
    result.insert(arg);
    subexpressionsInner(arg, result);
  }
}

// Returns all transitive subexpressions of 'expr'.
NodeSet subexpressions(NodeCP expr) {
  NodeSet result;
  subexpressionsInner(expr, result);
  return result;
}

// Returns a unit cost for a node.
float selfCost(NodeCP /*expr*/) {
  return 1.0f;
}

struct LevelData {
  NodeSet exprs;
};

size_t levelOf(std::vector<LevelData>& levels, NodeCP expr) {
  for (size_t i = 0; i < levels.size(); ++i) {
    if (levels[i].exprs.count(expr)) {
      return i;
    }
  }
  __builtin_unreachable();
}

void pushdownExpr(
    NodeCP expr,
    int32_t level,
    std::vector<LevelData>& levelData) {
  const auto defined = levelOf(levelData, expr);
  if (defined >= static_cast<size_t>(level)) {
    return;
  }

  if (level >= static_cast<int32_t>(levelData.size())) {
    levelData.resize(level + 1);
  }
  TORCH_CHECK(defined < levelData.size());
  levelData[defined].exprs.erase(expr);
  TORCH_CHECK(static_cast<size_t>(level) < levelData.size());
  levelData[level].exprs.insert(expr);

  for (auto input : args(expr)) {
    pushdownExpr(input, level + 1, levelData);
  }
}

void makeLevelsInner(
    NodeCP expr,
    int32_t level,
    std::vector<LevelData>& levelData,
    std::unordered_map<NodeCP, int32_t>& refCount,
    NodeSet& counted) {
  if (counted.count(expr)) {
    ++refCount[expr];
    pushdownExpr(expr, level, levelData);
    return;
  }

  if (level >= static_cast<int32_t>(levelData.size())) {
    levelData.resize(level + 1);
  }

  counted.insert(expr);
  ++refCount[expr];
  TORCH_CHECK(static_cast<size_t>(level) < levelData.size());
  levelData[level].exprs.insert(expr);
  for (auto input : args(expr)) {
    makeLevelsInner(input, level + 1, levelData, refCount, counted);
  }
}

void makeExprLevels(
    const NodeSet& exprs,
    std::vector<LevelData>& levelData,
    std::unordered_map<NodeCP, int32_t>& refCount) {
  NodeSet counted;
  for (auto expr : exprs) {
    makeLevelsInner(expr, 0, levelData, refCount, counted);
  }
}

NodeSet makeCseBorder(
    const std::vector<LevelData>& levelData,
    const NodeSet& placed,
    std::unordered_map<NodeCP, int32_t>& refCount) {
  NodeSet border;
  for (const auto& data : levelData | std::views::reverse) {
    for (auto expr : data.exprs) {
      if (placed.count(expr)) {
        continue;
      }
      if (refCount[expr] > 1) {
        auto subs = subexpressions(expr);
        bool overlaps = false;
        for (auto sub : subs) {
          if (border.count(sub)) {
            overlaps = true;
            break;
          }
        }
        if (overlaps) {
          continue;
        }
        border.insert(expr);
      }
    }
  }
  return border;
}

float parallelBorder(NodeCP expr, const NodeSet& placed, NodeSet& result) {
  constexpr float kSplit = -1;
  constexpr float kTargetCost = 500;
  if (placed.count(expr)) {
    return 0;
  }

  if (!isCallExpr(expr)) {
    return selfCost(expr);
  }

  const float cost = selfCost(expr);
  auto exprArgs = args(expr);
  std::unordered_set<int32_t> splitArgs;
  float allArgsCost = 0;
  float highestArgCost = 0;
  for (int32_t i = 0; i < static_cast<int32_t>(exprArgs.size()); ++i) {
    auto argCost = parallelBorder(exprArgs[i], placed, result);
    highestArgCost = std::max(highestArgCost, argCost);
    if (argCost == kSplit) {
      splitArgs.insert(i);
    }
    allArgsCost += argCost;
  }

  if (!splitArgs.empty()) {
    for (int32_t i = 0; i < static_cast<int32_t>(exprArgs.size()); ++i) {
      if (!splitArgs.count(i) && isCallExpr(exprArgs[i])) {
        result.insert(exprArgs[i]);
      }
    }
    return kSplit;
  }

  if (allArgsCost > kTargetCost && highestArgCost < allArgsCost / 2) {
    for (auto arg : exprArgs) {
      if (isCallExpr(arg)) {
        result.insert(arg);
      }
    }
    return kSplit;
  }
  return cost + allArgsCost;
}

} // namespace

// ---------- ParallelNodes ----------

namespace {

// Collects all top-level exprs from a ProjectNode and all its transitive
// inputs.
NodeSet collectAllInputExprs(const ProjectNode* node) {
  NodeSet allExprs;
  for (auto* cur = node; cur != nullptr; cur = cur->input()) {
    allExprs.insert(cur->nodes().begin(), cur->nodes().end());
  }
  return allExprs;
}

// Traverses the subgraph reachable from 'expr', collecting nodes that are in
// 'inputExprs' into 'reachable'. Stops recursing once it hits a node in
// 'inputExprs' or a leaf with no inputs.
void collectReachable(
    NodeCP expr,
    const NodeSet& inputExprs,
    NodeSet& visited,
    NodeSet& reachable) {
  if (!visited.insert(expr).second) {
    return;
  }
  if (inputExprs.count(expr)) {
    reachable.insert(expr);
    return;
  }
  for (auto* child : args(expr)) {
    collectReachable(child, inputExprs, visited, reachable);
  }
}

// Computes the extra dependency edges needed to honor in-place side effects.
//
// A non-functionalized graph has imperative semantics: a "value" is a handle to
// storage, and an in-place op (FunctionSchema Tensor(a!)) mutates the storage
// of one of its arguments. Reads/writes of that storage through aliasing views
// must keep their program order relative to the mutation, even though there is
// no producer->input edge expressing it (and the mutation's own output may be
// dead, so the output-reachability walk never finds it).
//
// For every storage base that is mutated, this orders all touches (any node
// referencing the base or a view of it) relative to each mutation: touches
// after the mutation depend on it (read-after-write), and the mutation depends
// on touches before it (write-after-read / write-after-write). The graph output
// node is included as a final touch so a mutated value that is also returned
// pulls in and orders its mutation. Edges are emitted via 'extraArgs' (consumed
// by args()); 'mutationNodes' collects the mutations that gained a later
// dependent, so the layering can force them to be their own project nodes.
void computeSideEffectEdges(
    const nativert::Graph& graph,
    std::unordered_map<NodeCP, std::vector<NodeCP>>& extraArgs,
    NodeSet& mutationNodes) {
  std::unordered_map<NodeCP, int32_t> pos;
  int32_t idx = 0;
  for (const auto& node : graph.nodes()) {
    pos[&node] = idx++;
  }
  NodeCP outputNode = graph.outputNode();

  // The node list must be in program (topological) order: every input's
  // producer precedes its consumer. The memory-dependency analysis below uses
  // node-list position as program order, so a graph rewrite that inserts a
  // replacement node out of position (e.g. not at the replaced op's site, as
  // index_put_ -> tw.masked_put_ once did) would silently invert the
  // ordering edges. Fail loudly here instead. Rewrites must use
  // graph->insertBefore(newNode, replacedNode).
  for (const auto& node : graph.nodes()) {
    auto consumerPos = pos[&node];
    for (const auto& input : node.inputs()) {
      if (input.value == nullptr) {
        continue;
      }
      auto* producer = input.value->producer();
      if (producer == nullptr) {
        continue;
      }
      auto it = pos.find(producer);
      if (it == pos.end()) {
        continue;
      }
      TORCH_CHECK(
          it->second < consumerPos,
          "Graph node list is not in program order: node '",
          node.target(),
          "' consumes a value produced later by '",
          producer->target(),
          "'. A graph rewrite likely inserted a replacement node out of "
          "position; use graph->insertBefore(newNode, replacedNode).");
    }
  }

  std::unordered_map<ValueCP, ValueCP> baseMemo;
  auto baseOf = [&](ValueCP v) -> ValueCP {
    auto it = baseMemo.find(v);
    if (it != baseMemo.end()) {
      return it->second;
    }
    auto* b = viewStorageBase(v);
    baseMemo[v] = b;
    return b;
  };

  struct Touch {
    NodeCP node;
    int32_t pos;
  };
  std::unordered_map<ValueCP, std::vector<Touch>> touches;
  for (const auto& node : graph.nodes()) {
    if (&node == outputNode) {
      continue;
    }
    for (const auto& input : node.inputs()) {
      if (input.value != nullptr) {
        touches[baseOf(input.value)].push_back({&node, pos[&node]});
      }
    }
  }
  // Graph outputs are materialized after every node runs: model the output node
  // as a touch at the end so mutations feeding outputs are ordered/pulled in.
  constexpr int32_t kOutputPos = std::numeric_limits<int32_t>::max();
  for (const auto& input : outputNode->inputs()) {
    if (input.value != nullptr) {
      touches[baseOf(input.value)].push_back({outputNode, kOutputPos});
    }
  }

  auto addEdge = [&](NodeCP from, NodeCP to) {
    auto& deps = extraArgs[from];
    if (std::find(deps.begin(), deps.end(), to) == deps.end()) {
      deps.push_back(to);
    }
  };
  for (const auto& node : graph.nodes()) {
    if (&node == outputNode) {
      continue;
    }
    auto mutated = dataMutatedInputs(&node);
    if (mutated.empty()) {
      continue;
    }
    auto mPos = pos[&node];
    bool hasLaterUse = false;
    for (auto* mv : mutated) {
      auto it = touches.find(baseOf(mv));
      if (it == touches.end()) {
        continue;
      }
      for (const auto& t : it->second) {
        if (t.node == &node) {
          continue;
        }
        if (t.pos > mPos) {
          addEdge(t.node, &node); // read/write after the mutation depends on it
          hasLaterUse = true;
        } else {
          addEdge(&node, t.node); // mutation depends on the earlier touch
        }
      }
    }
    if (hasLaterUse) {
      mutationNodes.insert(&node);
    }
  }
}

} // namespace

ProjectNode* ParallelNodes::makeParallelProject(
    ProjectNode* input,
    const NodeSet& topExprs,
    std::vector<NodeCP> orderedExprs) {
  if (orderedExprs.empty()) {
    orderedExprs.assign(topExprs.begin(), topExprs.end());
    std::sort(orderedExprs.begin(), orderedExprs.end(), [](NodeCP a, NodeCP b) {
      auto idOf = [](NodeCP e) {
        return e->outputs().empty() ? 0 : e->outputs()[0]->id();
      };
      return idOf(a) < idOf(b);
    });
  }
  std::vector<NodeCP> nodes;
  NodeSet seen;
  for (auto* expr : orderedExprs) {
    if (seen.insert(expr).second) {
      nodes.push_back(expr);
    }
  }
  std::unordered_set<NodeCP> inputs;

  if (input != nullptr) {
    auto allInputExprs = collectAllInputExprs(input);

    NodeSet visited;
    NodeSet reachable;
    for (auto* expr : topExprs) {
      collectReachable(expr, allInputExprs, visited, reachable);
    }
    inputs = std::move(reachable);
  }

  auto projectNode = std::make_unique<ProjectNode>(
      std::move(nodes), std::move(inputs), input, nextId_++);
  auto* result = projectNode.get();
  projectNodes_.push_back(std::move(projectNode));
  return result;
}

namespace {

// True if 'v' is a list-typed value. getListElements() asserts on non-list
// values, so callers must gate on this.
bool isListValue(ValueCP v) {
  switch (v->type().kind()) {
    case nativert::Type::Kind::TensorList:
    case nativert::Type::Kind::NestedTensorList:
    case nativert::Type::Kind::OptionalTensorList:
      return true;
    default:
      return false;
  }
}

// Distinct boundary input values read by one top-level expr, stopping at the
// layer boundary (same boundary rule as countBoundaryAccesses).
void collectExprBoundaryInputs(
    NodeCP exprRoot,
    const std::unordered_set<NodeCP>& boundary,
    std::unordered_set<ValueCP>& inputs) {
  NodeSet visited;
  std::vector<NodeCP> stack{exprRoot};
  while (!stack.empty()) {
    NodeCP n = stack.back();
    stack.pop_back();
    if (!visited.insert(n).second) {
      continue;
    }
    for (const auto& in : n->inputs()) {
      ValueCP v = in.value;
      if (v == nullptr) {
        continue;
      }
      NodeCP producer = v->producer();
      if (producer == nullptr || boundary.count(producer) > 0) {
        inputs.insert(v);
      } else {
        stack.push_back(producer);
      }
    }
  }
}

// The layer's internal (non-boundary) nodes, in program order. Value ids are
// assigned in creation order, so the first-output id sorts nodes into program
// order; the alias analysis below depends on this so consumption is seen after
// the corresponding production.
std::vector<NodeCP> layerNodesInOrder(const ProjectNode* pn) {
  const auto& boundary = pn->inputs();
  NodeSet visited;
  std::vector<NodeCP> stack;
  for (auto* n : pn->nodes()) {
    if (boundary.count(n) == 0) {
      stack.push_back(n);
    }
  }
  std::vector<NodeCP> result;
  while (!stack.empty()) {
    NodeCP n = stack.back();
    stack.pop_back();
    if (!visited.insert(n).second) {
      continue;
    }
    result.push_back(n);
    for (const auto& in : n->inputs()) {
      ValueCP v = in.value;
      if (v == nullptr) {
        continue;
      }
      NodeCP producer = v->producer();
      if (producer != nullptr && boundary.count(producer) == 0) {
        stack.push_back(producer);
      }
    }
  }
  std::sort(result.begin(), result.end(), [](NodeCP a, NodeCP b) {
    auto idOf = [](NodeCP e) {
      return e->outputs().empty() ? 0 : e->outputs()[0]->id();
    };
    return idOf(a) < idOf(b);
  });
  return result;
}

} // namespace

void ParallelNodes::computeLastUse(const nativert::Graph& graph) {
  // Values that leave the graph must never be reused or released, so exclude
  // them from the last-use sets even if no later layer reads them.
  // Graph outputs (and elements of list-typed outputs) escape the graph and
  // must never be freed -- nor may any value that shares their storage.
  std::unordered_set<ValueCP> graphOutputs;
  if (NodeCP outputNode = graph.outputNode()) {
    std::function<void(ValueCP)> addOut = [&](ValueCP v) {
      if (v == nullptr || !graphOutputs.insert(v).second) {
        return;
      }
      if (isListValue(v)) {
        for (auto* e : v->getListElements()) {
          addOut(e);
        }
      }
    };
    for (const auto& output : outputNode->inputs()) {
      addOut(output.value);
    }
  }

  // User inputs, weights, and constants are externally managed frame values
  // that must persist across runs (the frame is reused; they are refilled, not
  // reproduced). They must never appear in a freeable list -- even if the
  // optimized graph gives them a producer node. Exclude them explicitly.
  std::unordered_set<ValueCP> frameInputs;
  for (auto* v : graph.userInputs()) {
    frameInputs.insert(v);
  }
  for (auto* v : graph.weightValues()) {
    frameInputs.insert(v);
  }

  // Last use of EVERY assigned value -- not just the values returned by a
  // layer's top-level exprs, but also the intra-layer intermediates of fused
  // elementwise chains. Those intermediates are never separately allocated, so
  // listing them as "freeable" costs nothing, but tracking them here (then
  // extending lifetimes through list membership and storage aliasing below)
  // frees a shared buffer only after its last reader, and makes the separate
  // kernel-op intermediate free mechanism unnecessary. A value's raw last use
  // is the highest project node that directly reads it; id() equals the index
  // in projectNodes_.
  std::unordered_map<NodeCP, int32_t> nodeToPn;
  for (auto& pnPtr : projectNodes_) {
    for (NodeCP n : layerNodesInOrder(pnPtr.get())) {
      nodeToPn[n] = pnPtr->id();
    }
  }
  folly::F14FastMap<ValueCP, int32_t> lastNode;
  for (auto* value : graph.values()) {
    if (value == nullptr) {
      continue;
    }
    int32_t last = -1;
    for (auto* user : value->users()) {
      auto it = nodeToPn.find(user);
      if (it != nodeToPn.end()) {
        last = std::max(last, it->second);
      }
    }
    if (last >= 0) {
      lastNode[value] = last;
    }
  }

  // Extend lifetimes through prim.ListPack membership: an element is read
  // whenever the list containing it is read, so its last use is at least the
  // list's -- recursively for lists of lists. Iterate to a fixpoint.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto* value : graph.values()) {
      if (value == nullptr || !isListValue(value)) {
        continue;
      }
      auto lit = lastNode.find(value);
      if (lit == lastNode.end()) {
        continue;
      }
      int32_t listLast = lit->second;
      for (auto* e : value->getListElements()) {
        if (e == nullptr) {
          continue;
        }
        auto& el = lastNode[e];
        if (el < listLast) {
          el = listLast;
          changed = true;
        }
      }
    }
  }

  // Extend lifetimes through storage aliasing: a view and its base share one
  // buffer, so the buffer lives until the last read of ANY value in the storage
  // group. Group by viewStorageBase and lift every member to the group's max
  // last use. If any member is a graph output, the group's storage escapes and
  // must never be freed.
  folly::F14FastMap<ValueCP, ValueCP> baseOf;
  folly::F14FastMap<ValueCP, int32_t> baseLast;
  std::unordered_set<ValueCP> nonFreeableBase;
  for (auto* v : graphOutputs) {
    nonFreeableBase.insert(viewStorageBase(v));
  }
  for (const auto& [value, ln] : lastNode) {
    ValueCP b = viewStorageBase(value);
    baseOf[value] = b;
    auto it = baseLast.find(b);
    if (it == baseLast.end() || it->second < ln) {
      baseLast[b] = ln;
    }
  }

  // Assign each value to its storage group's last-use layer. Leaves -- graph
  // inputs, weights, constants -- are externally managed and persist across
  // runs, so they are never freed. A leaf is a value whose producer is not a
  // graph computation node (it is null, or the graph input/constant node, which
  // is not in nodeToPn). Leaves still contribute their last read to baseLast
  // above so that views over them are freed at the right layer.
  for (const auto& [value, ln] : lastNode) {
    ValueCP b = baseOf[value];
    bool isLeaf =
        value->producer() == nullptr || nodeToPn.count(value->producer()) == 0;
    if (graphOutputs.count(value) > 0 || nonFreeableBase.count(b) > 0 ||
        isLeaf || frameInputs.count(value) > 0 || frameInputs.count(b) > 0) {
      continue;
    }
    projectNodes_[baseLast[b]]->lastUse.insert(value);
  }

  // Distinct boundary inputs of each top-level expr (for the reusable-input
  // third pass below).
  std::vector<std::vector<std::unordered_set<ValueCP>>> exprInputs(
      projectNodes_.size());
  for (auto& pnPtr : projectNodes_) {
    ProjectNode* pn = pnPtr.get();
    auto& perExpr = exprInputs[pn->id()];
    perExpr.resize(pn->nodes().size());
    for (size_t i = 0; i < pn->nodes().size(); ++i) {
      collectExprBoundaryInputs(pn->nodes()[i], pn->inputs(), perExpr[i]);
    }
  }

  // Third pass: a (now alias-corrected) lastUse value that is a boundary input
  // of exactly one top-level expr in its layer can be reused in place by that
  // expr's kernel op.
  for (auto& pnPtr : projectNodes_) {
    ProjectNode* pn = pnPtr.get();
    const auto& perExpr = exprInputs[pn->id()];
    pn->reusableValues_.assign(pn->nodes().size(), {});
    for (ValueCP v : pn->lastUse) {
      int32_t onlyExpr = -1;
      int32_t count = 0;
      for (size_t i = 0; i < perExpr.size(); ++i) {
        if (perExpr[i].count(v) > 0) {
          ++count;
          onlyExpr = static_cast<int32_t>(i);
          if (count > 1) {
            break;
          }
        }
      }
      if (count == 1) {
        pn->reusableValues_[onlyExpr].push_back(v);
      }
    }
  }

  // Diagnostic (kFrame): validate the (alias-corrected) lastUse against
  // viewStorageBase ground truth. A value freed in a layer whose storage is
  // still read by a later layer -- directly or through any storage alias -- is
  // a premature free. Reports which later use was missed so the alias tracking
  // can be fixed.
  if (WaveConfig::get().trace & WaveConfig::kFrame) {
    auto userMaxNode = [&](ValueCP v) -> int32_t {
      int32_t m = -1;
      for (auto* u : v->users()) {
        auto it = nodeToPn.find(u);
        if (it != nodeToPn.end()) {
          m = std::max(m, it->second);
        }
      }
      return m;
    };
    // storage base -> latest layer that reads any value sharing that storage.
    std::unordered_map<ValueCP, int32_t> baseTrueLast;
    for (auto* v : graph.values()) {
      if (v == nullptr) {
        continue;
      }
      ValueCP b = viewStorageBase(v);
      int32_t mu = userMaxNode(v);
      auto it = baseTrueLast.find(b);
      if (it == baseTrueLast.end() || it->second < mu) {
        baseTrueLast[b] = mu;
      }
    }
    for (auto& pnPtr : projectNodes_) {
      ProjectNode* pn = pnPtr.get();
      for (ValueCP v : pn->lastUse) {
        ValueCP b = viewStorageBase(v);
        auto it = baseTrueLast.find(b);
        int32_t trueLast = it != baseTrueLast.end() ? it->second : -1;
        if (trueLast > pn->id()) {
          LOG(INFO) << "LASTUSE-TOO-EARLY %" << v->id() << " freed at node "
                    << pn->id() << " but storage base %" << b->id()
                    << " (producer "
                    << (b->producer() ? b->producer()->target() : "<none>")
                    << ") is read through node " << trueLast;
        }
      }
    }
  }
}

ProjectNode* ParallelNodes::makeParallelNodes(const nativert::Graph& graph) {
  // Side-effect analysis: extra ordering edges for in-place mutations. Computed
  // from the raw graph (gExtraArgs is still null here), then installed so the
  // args()-based layering below orders mutations relative to the storage they
  // alias and discovers mutations whose SSA output is dead.
  std::unordered_map<NodeCP, std::vector<NodeCP>> extraArgs;
  NodeSet mutationNodes;
  computeSideEffectEdges(graph, extraArgs, mutationNodes);
  gExtraArgs = &extraArgs;
  // Clear on every exit path (incl. exceptions) so args() can never dereference
  // a dangling pointer to the now-destroyed local 'extraArgs'.
  SCOPE_EXIT {
    gExtraArgs = nullptr;
  };

  NodeCP root = graph.outputNode();
  auto topExprs = args(root);
  NodeSet top(topExprs.begin(), topExprs.end());

  std::vector<LevelData> levelData;
  std::unordered_map<NodeCP, int32_t> refCount;
  makeExprLevels(top, levelData, refCount);

  // Force in-place mutation nodes to be project-node borders so compileNode
  // emits them even when their SSA result is dead: their effect is observed
  // only through the mutated storage. (Reachable mutations are in refCount via
  // the extra edges; unreachable/dead ones are absent and stay dropped.)
  for (auto* m : mutationNodes) {
    auto it = refCount.find(m);
    if (it != refCount.end() && it->second < 2) {
      it->second = 2;
    }
  }

  NodeSet placed;
  ProjectNode* current = nullptr;

  for (;;) {
    auto cses = makeCseBorder(levelData, placed, refCount);
    if (cses.empty()) {
      break;
    }

    for (auto expr : cses) {
      auto subs = subexpressions(expr);
      placed.insert(subs.begin(), subs.end());
    }
    placed.insert(cses.begin(), cses.end());

    auto* project = makeParallelProject(current, cses);
    if (project == nullptr) {
      TORCH_CHECK(false, "makeParallelProject returned null");
    }
    current = project;
  }

  NodeSet parallel;
  for (auto expr : top) {
    parallelBorder(expr, placed, parallel);
  }

  if (!parallel.empty()) {
    for (auto expr : parallel) {
      auto subs = subexpressions(expr);
      placed.insert(subs.begin(), subs.end());
    }
    placed.insert(parallel.begin(), parallel.end());

    auto* project = makeParallelProject(current, parallel);
    if (project == nullptr) {
      TORCH_CHECK(false, "makeParallelProject returned null");
    }
    current = project;
  }

  auto* project = makeParallelProject(current, top, topExprs);
  if (project == nullptr) {
    TORCH_CHECK(false, "makeParallelProject returned null");
  }
  current = project;

  // All layers are now built in execution order; annotate each with its
  // last-use / reusable-last-use values.
  computeLastUse(graph);

  return current;
}

// Debugger helper - callable from GDB.
__attribute__((used)) void printSet(NodeSet& set) {
  fmt::print("{}\n", set.size());
  for (const auto* node : set) {
    auto ptr = reinterpret_cast<int64_t>(node);
    fmt::print("{:x} {}\n", (ptr >> 3) & 0xffff, std::string(node->target()));
  }
}

// Debugger helper - callable from GDB.
__attribute__((used)) void printRefcount(
    std::unordered_map<NodeCP, int32_t>& refCount,
    int32_t min) {
  fmt::print("{}\n", refCount.size());
  for (const auto& [expr, count] : refCount) {
    if (count >= min) {
      auto ptr = reinterpret_cast<int64_t>(expr);
      fmt::print("{} {:x} {}\n", count, (ptr >> 3) & 0xffff, expr->toString());
    }
  }
}

} // namespace torch::wave
