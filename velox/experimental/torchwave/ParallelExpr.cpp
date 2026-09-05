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
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <fmt/format.h>
#include <folly/CppAttributes.h>
#include <folly/ScopeGuard.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/experimental/torchwave/ParallelExpr.h"

#include <map>
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

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

// A metadata getter (sym_size / sym_numel) whose only role is to be a top-level
// graph output, over an operand something else also reads.
//
// Such a getter is in `top`, so makeLevelsInner counts a second reference to
// its operand's producer, and makeCseBorder turns that into a border: the
// producer moves to its own earlier layer and every other consumer of it
// follows a layer later than it needed to. The getter's own work is one scalar
// field read on the host; the whole cost is the layer split it forces.
//
// Users are counted through 'reachable', not through users(): torch.export
// leaves dead `_operator.ge` / `_operator.le` shape guards behind once the
// asserts are stripped -- 428 of them on the ROO graph -- and they appear in a
// value's users() while contributing to no level. Counting them refuses every
// candidate.
bool isDeferredSizeOutput(
    NodeCP node,
    NodeCP outputNode,
    const NodeSet& reachable) {
  const Metadata* meta = Registry::metadata(node->target());
  if (meta == nullptr || !meta->isMetadataGetter) {
    return false;
  }
  if (node->outputs().size() != 1 || node->outputs()[0] == nullptr) {
    return false;
  }
  // Consumed on the device as well as returned: it has to stay a real op or the
  // consumer has nothing to read.
  for (auto* user : node->outputs()[0]->users()) {
    if (user != outputNode && reachable.count(user) > 0) {
      return false;
    }
  }
  if (node->inputs().empty() || node->inputs()[0].value == nullptr) {
    return false;
  }
  ValueCP operand = node->inputs()[0].value;
  if (operand->producer() == nullptr) {
    return false;
  }
  // Sole user: removing the reference merges no layer, so there is nothing to
  // win and no reason to grow a second mechanism.
  int32_t others = 0;
  for (auto* user : operand->users()) {
    if (user != node && reachable.count(user) > 0) {
      ++others;
    }
  }
  return others > 0;
}

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

// Collects every node of the expr rooted at 'exprRoot', walking down to (but
// not through) the layer 'boundary'. The result is the set of nodes whose
// outputs are candidate expr-local temporaries.
void collectExprSubgraph(
    NodeCP exprRoot,
    const std::unordered_set<NodeCP>& boundary,
    NodeSet& subgraph) {
  std::vector<NodeCP> stack{exprRoot};
  while (!stack.empty()) {
    NodeCP n = stack.back();
    stack.pop_back();
    if (!subgraph.insert(n).second) {
      continue;
    }
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

// The value a writing op mutates in place (its mutatesArg "self"), or null when
// 'node' is not a registered writer or the ordinal is out of range.
ValueCP FOLLY_NULLABLE inPlaceSelf(NodeCP node) {
  const Metadata* meta = Registry::metadata(node->target());
  if (meta == nullptr || !meta->mutatesArg.has_value()) {
    return nullptr;
  }
  const auto ordinal = *meta->mutatesArg;
  const auto& inputs = node->inputs();
  if (ordinal < 0 || static_cast<size_t>(ordinal) >= inputs.size()) {
    return nullptr;
  }
  return inputs[ordinal].value;
}

// Rewrites every input slot of 'consumer' that reads 'from' to read 'to',
// maintaining Value::users() (a raw slot assignment alone leaves the old
// value's users() stale and never adds the node to the new value's users()).
void rewireInput(NodeCP consumer, ValueCP from, ValueCP to) {
  auto* node = const_cast<nativert::Node*>(consumer);
  auto* toMutable = const_cast<nativert::Value*>(to);
  bool replaced = false;
  for (auto& input : node->inputs()) {
    if (input.value == from) {
      input.value = toMutable;
      replaced = true;
    }
  }
  if (replaced) {
    // Safe: all of 'from's slots on this node were rewritten above, so the
    // node no longer references 'from'. addUser dedups if 'to' was already
    // read.
    const_cast<nativert::Value*>(from)->eraseUser(node);
    const_cast<nativert::Value*>(to)->addUser(node);
  }
}

// True if any input of 'consumer' other than slot 'selfOrdinal' shares a
// storage base with 'x' (C3: no other operand may alias the write target).
bool otherInputAliases(NodeCP consumer, int32_t selfOrdinal, ValueCP x) {
  ValueCP xBase = viewStorageBase(x);
  const auto& inputs = consumer->inputs();
  for (int32_t j = 0; j < static_cast<int32_t>(inputs.size()); ++j) {
    if (j == selfOrdinal || inputs[j].value == nullptr) {
      continue;
    }
    if (viewStorageBase(inputs[j].value) == xBase) {
      return true;
    }
  }
  return false;
}

// The alias steps from 'value' back to its storage base, outermost first: the
// producing node of every view / in-place edge viewStorageBase follows. Two
// values address the same elements only if these chains match op for op.
std::vector<NodeCP> viewChain(ValueCP value) {
  std::vector<NodeCP> chain;
  while (value != nullptr) {
    auto* producer = value->producer();
    if (producer == nullptr) {
      break;
    }
    ValueCP next = schemaAliasedInput(producer, value);
    if (next == nullptr) {
      const auto* meta = Registry::metadata(producer->target());
      if (meta == nullptr || !meta->viewOfArg.has_value() ||
          *meta->viewOfArg >= static_cast<int32_t>(producer->inputs().size())) {
        break;
      }
      next = producer->inputs()[*meta->viewOfArg].value;
    }
    chain.push_back(producer);
    value = next;
  }
  return chain;
}

// Constant is a variant; a subgraph alternative is move-only and never carries
// an index, so treat it as unequal rather than comparing it.
bool constantsEqual(const nativert::Constant& a, const nativert::Constant& b) {
  if (a.index() != b.index()) {
    return false;
  }
  return std::visit(
      [&b](const auto& av) -> bool {
        using T = std::decay_t<decltype(av)>;
        if constexpr (std::is_same_v<T, std::unique_ptr<nativert::Graph>>) {
          return false;
        } else {
          return av == std::get<T>(b);
        }
      },
      a);
}

bool sameAttributes(NodeCP a, NodeCP b) {
  const auto& aAttrs = a->attributes();
  const auto& bAttrs = b->attributes();
  if (aAttrs.size() != bAttrs.size()) {
    return false;
  }
  for (const auto& attr : aAttrs) {
    const auto* other = b->tryGetAttribute(attr.name);
    if (other == nullptr || !constantsEqual(attr.value, other->value)) {
      return false;
    }
  }
  return true;
}

// True if 'a' and 'b' address exactly the same elements of their shared base:
// the same alias steps, in the same order, with the same indices.
bool sameViewPath(ValueCP a, ValueCP b) {
  if (a == b) {
    return true;
  }
  auto aChain = viewChain(a);
  auto bChain = viewChain(b);
  if (aChain.size() != bChain.size()) {
    return false;
  }
  for (size_t i = 0; i < aChain.size(); ++i) {
    if (aChain[i]->target() != bChain[i]->target() ||
        !sameAttributes(aChain[i], bChain[i])) {
      return false;
    }
  }
  return true;
}

// True if some other operand of 'consumer' is computed from a read of the write
// target's storage that does not address the same elements as the write.
//
// otherInputAliases only inspects the operands themselves, so it clears an
// operand that is a materialized tensor (its own storage base) even when the
// expression producing it reads the target. Fusion puts that read in the same
// kernel as the write, so eliding the clone lets the write land before the read
// and the read returns post-write data. Walk each other operand back through
// its in-layer producers: a read of the target's base is only safe when it
// covers exactly the elements being overwritten (identical index path), which
// makes the pair element-to-element.
bool otherInputReadsTargetElsewhere(
    NodeCP consumer,
    int32_t selfOrdinal,
    ValueCP self,
    const NodeSet& layerNodes) {
  ValueCP base = viewStorageBase(self);
  std::vector<ValueCP> work;
  const auto& inputs = consumer->inputs();
  for (int32_t j = 0; j < static_cast<int32_t>(inputs.size()); ++j) {
    if (j != selfOrdinal && inputs[j].value != nullptr) {
      work.push_back(inputs[j].value);
    }
  }
  std::unordered_set<ValueCP> visited;
  while (!work.empty()) {
    ValueCP value = work.back();
    work.pop_back();
    if (value == nullptr || !visited.insert(value).second) {
      continue;
    }
    if (viewStorageBase(value) == base && !sameViewPath(value, self)) {
      return true;
    }
    auto* producer = value->producer();
    if (producer == nullptr || layerNodes.count(producer) == 0) {
      continue;
    }
    for (const auto& input : producer->inputs()) {
      work.push_back(input.value);
    }
  }
  return false;
}

// True if 'cloneNode' requests an explicit memory_format, i.e. it is a layout
// conversion whose output cannot be replaced by its (possibly differently laid
// out) input.
bool cloneForcesLayout(
    NodeCP cloneNode,
    ValueCP input,
    const ValueTypes& types) {
  const auto* attr = cloneNode->tryGetAttribute("memory_format");
  if (attr == nullptr) {
    return false;
  }
  bool concrete = false;
  bool contiguousFormat = false;
  if (std::holds_alternative<c10::MemoryFormat>(attr->value)) {
    concrete = true;
    contiguousFormat = std::get<c10::MemoryFormat>(attr->value) ==
        c10::MemoryFormat::Contiguous;
  } else if (std::holds_alternative<std::string>(attr->value)) {
    const auto& name = std::get<std::string>(attr->value);
    if (!name.empty() && name != "None") {
      concrete = true;
      contiguousFormat = name == "contiguous_format" || name == "Contiguous";
    }
  } else if (std::holds_alternative<int64_t>(attr->value)) {
    // c10::MemoryFormat::Contiguous is enum value 0.
    concrete = true;
    contiguousFormat = std::get<int64_t>(attr->value) == 0;
  }
  if (!concrete) {
    return false; // memory_format=None (or unrecognized): not a conversion.
  }
  // A non-contiguous target (channels_last, ...) always forces a real layout
  // conversion. A contiguous target only forces one when the input is not
  // already contiguous; otherwise the clone is a no-op copy.
  return contiguousFormat ? !types.contiguous(input) : true;
}

// True if 'op' reads its tensor inputs at arbitrary strides: an elementwise op
// (whose codegen handles strides and whose result values are layout
// independent -- e.g. a fused tw.bucketize) or a clone (which copies any
// layout). A clone that only densifies its input is unnecessary before such a
// consumer.
bool readsStridedInput(NodeCP op) {
  if (op->target() == "torch.ops.aten.clone.default") {
    return true;
  }
  const Metadata* meta = Registry::metadata(op->target());
  if (meta == nullptr) {
    return false;
  }
  return meta->elementwise != nullptr || meta->layoutAgnostic;
}

// True if 'node' is dead: it performs no in-place mutation and none of its
// outputs is read. Such nodes linger after an optimizer rewrite (e.g. a detach
// whose output uses were already replaced) but still reference their inputs, so
// they pollute users(); they must not constrain clone elision.
bool isDeadNode(NodeCP node) {
  if (!dataMutatedInputs(node).empty()) {
    return false; // an in-place mutation's effect is live even if its SSA
                  // output is dead
  }
  for (auto* out : node->outputs()) {
    if (out != nullptr && !out->users().empty()) {
      return false;
    }
  }
  return true;
}

// The single live op that reads 'value'. prim.ListPack is transparent: packing
// a value into a list moves no data, so the real reader is the list's consumer.
// Returns nullptr unless exactly one live reader resolves.
NodeCP FOLLY_NULLABLE soleLiveReader(ValueCP value) {
  NodeCP found = nullptr;
  for (auto* user : value->users()) {
    if (isDeadNode(user)) {
      continue;
    }
    if (found != nullptr) {
      return nullptr;
    }
    found = user;
  }
  if (found == nullptr || found->target() != "prim.ListPack") {
    return found;
  }
  if (found->outputs().empty()) {
    return nullptr;
  }
  // Packed twice into the same list means the reader visits it twice.
  int32_t occurrences = 0;
  for (const auto& input : found->inputs()) {
    if (input.value == value) {
      ++occurrences;
    }
  }
  if (occurrences != 1) {
    return nullptr;
  }
  return soleLiveReader(found->outputs()[0]);
}

// True when dropping a densifying clone before 'reader' is layout-neutral:
// 'reader' reads at arbitrary strides and visits each element once, so the
// strided addressing simply moves out of the clone's copy and into the reader.
// A reader that revisits elements (wholeTensor / randomAccess) would repeat
// that addressing, which one densifying copy amortizes, so it keeps the clone.
bool absorbsStrides(NodeCP reader, ValueCP value) {
  if (!readsStridedInput(reader)) {
    return false;
  }
  const Metadata* meta = Registry::metadata(reader->target());
  if (meta == nullptr) {
    return true; // a consuming clone copies each element once
  }
  bool seen = false;
  for (size_t i = 0; i < reader->inputs().size(); ++i) {
    if (reader->inputs()[i].value != value) {
      continue;
    }
    if (seen) {
      return false; // read from two argument slots
    }
    seen = true;
    const auto* am = argMetaForInput(meta, reader, i);
    if (am && (am->wholeTensor || am->randomAccess)) {
      return false;
    }
  }
  // Not found as a direct argument means 'value' arrived through a list, whose
  // element the reader copies once (see __copy's non-contiguous branch).
  return true;
}

// Attempts to replace clone(x) by x for one clone node in a layer.
// 'layerNodes' is the ProjectNode's node set; the clone's single consumer must
// be in the same layer so x's lifetime is not extended across layers (whether
// that consumer is fused or standalone does not matter). 'pn' provides the
// alias-aware reuse sets and 'types' the per-value constraints. Returns true
// when the clone is elided.
bool tryElideClone(
    NodeCP cloneNode,
    const ProjectNode* pn,
    const NodeSet& layerNodes,
    const ValueTypes& types) {
  if (cloneNode->target() != "torch.ops.aten.clone.default" ||
      cloneNode->inputs().empty() || cloneNode->outputs().empty()) {
    return false;
  }
  ValueCP source = cloneNode->inputs()[0].value;
  ValueCP cloneOut = cloneNode->outputs()[0];
  if (source == nullptr || cloneOut == nullptr) {
    return false;
  }
  // A single consumer in the same layer. Same-layer keeps source's lifetime
  // within one ProjectNode (no cross-layer extension); the consumer being fused
  // or standalone does not affect elidability.
  const auto& users = cloneOut->users();
  if (users.size() != 1 || layerNodes.count(users[0]) == 0) {
    return false;
  }
  NodeCP consumer = users[0];

  if (inPlaceSelf(consumer) == cloneOut) {
    // Mutating consumer (C3): eliding writes source in place. A memory_format
    // clone materialized the write buffer, so keep it; otherwise source must be
    // safe to overwrite -- last use here with a single in-layer storage
    // consumer (reusable/overwritable), not externally owned, not a graph
    // output, no zero strides, and no other operand of the writer aliasing it.
    if (cloneForcesLayout(cloneNode, source, types)) {
      return false;
    }
    if (!pn->isReusableInput(source) && !pn->isOverwritableTemp(source)) {
      return false;
    }
    if (types.zeroStrides(source) || types.externallyOwned(source) ||
        types.graphOutput(source)) {
      return false;
    }
    const int32_t selfOrdinal =
        *Registry::metadata(consumer->target())->mutatesArg;
    if (otherInputAliases(consumer, selfOrdinal, source)) {
      return false;
    }
    if (otherInputReadsTargetElsewhere(
            consumer, selfOrdinal, source, layerNodes)) {
      return false;
    }
    rewireInput(consumer, cloneOut, source);
    return true;
  }

  // Read-only consumer. The clone is unneeded when the consumer can read source
  // at its native layout: either source is already contiguous, or the consumer
  // is layout-agnostic (an elementwise op -- e.g. a fused tw.bucketize -- or
  // another clone) that reads arbitrary strides and yields layout-independent
  // values. A layout-agnostic consumer also makes any requested memory_format
  // moot; otherwise a memory_format densify must be kept.
  const bool consumerReadsStrided = readsStridedInput(consumer);
  if (!types.contiguous(source) && !consumerReadsStrided) {
    return false;
  }
  if (!consumerReadsStrided && cloneForcesLayout(cloneNode, source, types)) {
    return false;
  }
  if (types.graphOutput(cloneOut)) {
    return false;
  }
  const Metadata* meta = Registry::metadata(consumer->target());
  if (meta != nullptr && meta->isView() && types.zeroStrides(source)) {
    return false; // C5: a view over a stride-0 input needs a real buffer.
  }
  rewireInput(consumer, cloneOut, source);
  return true;
}

// Elides a multi-user clone by pointing every use at the input. Unlike
// tryElideClone this is not restricted to a single same-layer consumer:
// Graph::replaceAllUses rewires all users (maintaining users()) and
// computeLastUse is re-run afterward to refresh the input's now-extended
// lifetime. The clone is droppable when it is a no-op (input already
// contiguous) or every user is layout-agnostic (reads the input at native
// strides -- an elementwise op such as a fused tw.bucketize, or another clone).
// Safe only when the input's storage is never mutated (so the clone carries no
// pre-mutation snapshot), the output does not escape as a graph output, and no
// user writes the clone in place.
bool tryElideMultiUserClone(
    NodeCP cloneNode,
    nativert::Graph& graph,
    const std::unordered_set<ValueCP>& mutatedBases,
    const ValueTypes& types) {
  if (cloneNode->target() != "torch.ops.aten.clone.default" ||
      cloneNode->inputs().empty() || cloneNode->outputs().empty()) {
    return false;
  }
  ValueCP source = cloneNode->inputs()[0].value;
  ValueCP cloneOut = cloneNode->outputs()[0];
  if (source == nullptr || cloneOut == nullptr) {
    return false;
  }
  // Every live user must read the clone (never write it in place); track
  // whether all of them read at arbitrary strides. Dead users (e.g. an
  // already-elided detach still referencing the clone) do not constrain.
  bool allReadStrided = true;
  for (auto* user : cloneOut->users()) {
    if (isDeadNode(user)) {
      continue;
    }
    if (inPlaceSelf(user) == cloneOut) {
      return false;
    }
    if (!readsStridedInput(user)) {
      allReadStrided = false;
    }
  }
  // Droppable when the clone is a no-op (input already contiguous) or the
  // strided access it would have absorbed simply moves into its readers: either
  // every user reads at native strides, or there is exactly one reader and it
  // visits each element once. A memory_format densify still matters otherwise.
  bool stridesAbsorbed = allReadStrided;
  if (!stridesAbsorbed) {
    NodeCP reader = soleLiveReader(cloneOut);
    stridesAbsorbed = reader != nullptr && absorbsStrides(reader, cloneOut);
  }
  if (!types.contiguous(source) && !stridesAbsorbed) {
    return false;
  }
  if (!stridesAbsorbed && cloneForcesLayout(cloneNode, source, types)) {
    return false;
  }
  // The output must not escape as a graph output (would alias input to output),
  // and the input's storage must never be mutated (else the clone snapshots a
  // pre-mutation value).
  if (types.graphOutput(cloneOut) ||
      mutatedBases.count(viewStorageBase(source)) > 0) {
    return false;
  }
  graph.replaceAllUses(
      const_cast<nativert::Value*>(cloneOut),
      const_cast<nativert::Value*>(source));
  return true;
}

} // namespace

// Elides every clone whose users only read it, over the whole graph. Runs
// before partitioning, so it sees clones that rewriteInPlace cannot: that pass
// walks one ProjectNode layer at a time, and a source fanned out to many
// consumers lands its clones in different layers. In the ROO preproc graph one
// group_length_guard_sparse output is cloned 26 times, once per consumer, and
// no two of those clones ever appear in the same layer.
//
// A clone that survives fusion is a copy plus a barrier, never free, so
// dropping one is always a win when it is safe. Safety is exactly
// tryElideMultiUserClone's: no user writes it, it does not escape as a graph
// output, and the source's storage is never mutated anywhere.
int64_t elideReadOnlyClones(nativert::Graph& graph, const ValueTypes& types) {
  std::unordered_set<ValueCP> mutatedBases;
  for (const auto& node : graph.nodes()) {
    for (auto* mutated : dataMutatedInputs(&node)) {
      if (mutated != nullptr) {
        mutatedBases.insert(viewStorageBase(mutated));
      }
    }
  }

  // Snapshot first: eliding rewires users, and a live walk would revisit
  // clones that are already dead.
  std::vector<NodeCP> clones;
  for (const auto& node : graph.nodes()) {
    if (node.target() == "torch.ops.aten.clone.default") {
      clones.push_back(&node);
    }
  }

  int64_t elided = 0;
  for (NodeCP cloneNode : clones) {
    if (tryElideMultiUserClone(cloneNode, graph, mutatedBases, types)) {
      ++elided;
    }
  }

  // A clone whose source is not contiguous cannot be dropped: it densifies,
  // and that work is real. It only has to happen once per source, though.
  // Identical clones of one source are interchangeable as long as neither is
  // written in place, so keep the first and redirect the rest. In this graph a
  // single group_length_guard_sparse output is cloned once per consumer.
  int64_t cse = 0;
  std::map<std::pair<ValueCP, std::string>, ValueCP> firstCloneOf;
  for (NodeCP cloneNode : clones) {
    if (isDeadNode(cloneNode) || cloneNode->outputs()[0]->users().empty()) {
      continue;
    }
    ValueCP src = cloneNode->inputs()[0].value;
    ValueCP out = cloneNode->outputs()[0];
    bool written = false;
    for (auto* user : out->users()) {
      if (!isDeadNode(user) && inPlaceSelf(user) == out) {
        written = true;
      }
    }
    // A snapshot of storage that is mutated somewhere is only valid at the
    // point it was taken, so two such clones are not interchangeable.
    if (written || mutatedBases.count(viewStorageBase(src)) > 0) {
      continue;
    }
    // A graph output takes no part in the merge, in either direction.
    // Redirecting one away would leave it unproduced: replaceAllUses rewires
    // users, and the output node is not one of them. Merging others INTO one
    // would hand the caller a buffer that internal values also read -- safe
    // only as long as nothing writes it, which is an invariant of this pass
    // rather than of the returned tensor, and not one the caller can see. The
    // clones that remain are internal, so the survivor is always a value the
    // graph fully owns.
    if (types.graphOutput(out)) {
      continue;
    }
    // Two clones only match if they produce the same layout.
    const auto* fmt = cloneNode->tryGetAttribute("memory_format");
    std::pair<ValueCP, std::string> mapKey{
        src, fmt ? std::to_string(fmt->value.index()) : std::string("none")};
    auto it = firstCloneOf.find(mapKey);
    if (it == firstCloneOf.end()) {
      firstCloneOf.emplace(mapKey, out);
      continue;
    }
    graph.replaceAllUses(
        const_cast<nativert::Value*>(out),
        const_cast<nativert::Value*>(it->second));
    ++cse;
  }
  elided += cse;
  if ((WaveConfig::get().trace & WaveConfig::kTiming) && elided > 0) {
    LOG(INFO) << "pre-partition clone pass: elided " << elided << " clone(s)";
  }
  return elided;
}

namespace {

// Appends 'text' as a length-prefixed token. The punctuation cseKey separates
// its fields with also occurs inside the fields: constantToString renders a
// string attribute, a device or a list with commas, brackets and equals signs
// of its own. Concatenated raw, two different nodes could spell the same key
// and be merged, which rewires readers to the wrong buffer -- a wrong result
// rather than a missed optimization. A length prefix removes the ambiguity
// without having to escape anything.
void appendToken(std::string& key, std::string_view text) {
  key += std::to_string(text.size());
  key += ':';
  key += text;
}

// Identity of a node for common-subexpression purposes: op, operands in order,
// attributes, output arity. Output ids are deliberately absent -- they are what
// a merge rewrites. Empty for a node that must never be merged: one carrying a
// subgraph attribute, which has no value rendering to compare.
std::string cseKey(NodeCP node) {
  std::string key;
  appendToken(key, node->target());
  key += '(';
  for (const auto& input : node->inputs()) {
    appendToken(key, input.name);
    key += '=';
    // An id is digits only, so it needs no length prefix to stay unambiguous.
    key += input.value != nullptr ? std::to_string(input.value->id()) : "null";
    key += ',';
  }
  // Attribute order is not part of a node's identity, so compare them sorted.
  // Sorting the already-tokenized rendering keeps the order canonical; which
  // total order it is does not matter, only that it is deterministic.
  std::vector<std::string> attrs;
  attrs.reserve(node->attributes().size());
  for (const auto& attr : node->attributes()) {
    if (std::holds_alternative<std::unique_ptr<nativert::Graph>>(attr.value)) {
      return {};
    }
    std::string rendered;
    appendToken(rendered, attr.name);
    rendered += '=';
    appendToken(rendered, constantToString(attr.value));
    attrs.push_back(std::move(rendered));
  }
  std::sort(attrs.begin(), attrs.end());
  for (const auto& attr : attrs) {
    key += attr;
    key += ',';
  }
  key += ')';
  key += std::to_string(node->outputs().size());
  return key;
}

// A node is mergeable with an identical one when it is a pure function of its
// operands. It must write nothing, and no operand's storage may be written
// anywhere in the graph: a read is interchangeable with an earlier read only if
// nothing modified the buffer in between, and this pass compares identity, not
// position. An output that escapes as a graph output is left alone for the same
// reason clone elision leaves it alone -- the caller would be handed a buffer
// that other values also read.
bool cseEligible(
    NodeCP node,
    const ValueTypes& types,
    const folly::F14FastSet<ValueCP>& mutatedBases,
    bool cseViews,
    bool cseCompute) {
  if (node->outputs().empty() || !dataMutatedInputs(node).empty() ||
      inPlaceSelf(node) != nullptr) {
    return false;
  }
  for (const auto* out : node->outputs()) {
    if (out == nullptr || types.graphOutput(out)) {
      return false;
    }
    // An output written in place somewhere later is no more interchangeable
    // than a mutated operand: the survivor's buffer is not what the duplicate's
    // readers expect once the write has run. The operand loop below does not
    // cover it, because the write names the output of this node rather than
    // anything this node reads.
    if (mutatedBases.count(viewStorageBase(out)) > 0) {
      return false;
    }
    // A TensorList's elements are the outputs of its sole prim.ListUnpack user,
    // an invariant getListElements asserts on. Merging two list producers
    // leaves the survivor's list with two users -- its own unpack and the
    // dup's, which is dead but still counted -- and the next caller to ask for
    // its elements throws. Sound merging here needs the dead unpack removed,
    // which this pass does not do.
    if (out->type().kind() == nativert::Type::Kind::TensorList) {
      return false;
    }
  }
  for (const auto& input : node->inputs()) {
    if (input.value == nullptr ||
        mutatedBases.count(viewStorageBase(input.value)) > 0) {
      return false;
    }
  }
  const Metadata* meta = Registry::metadata(node->target());
  return (meta != nullptr && meta->isView()) ? cseViews : cseCompute;
}

// Points every reader of 'from' at 'to'.
//
// Graph::replaceAllUses walks the user list and TORCH_CHECKs that
// Graph::replace found something to swap in every entry, which two things
// break. A node reading the value in two argument positions is listed twice,
// and replace swaps both occurrences on the first visit, so the second visit
// finds nothing. And the list holds stale entries: nodes an earlier merge
// repointed without the entry being dropped. Both are common once views merge,
// where cat(t, t) and chains of repointed readers are the rule rather than the
// exception. Normalizing the list to the nodes that really do read the value,
// once each, is what makes the call safe.
void replaceReaders(nativert::Graph& graph, ValueCP from, nativert::Value* to) {
  auto* value = const_cast<nativert::Value*>(from);
  std::vector<nativert::Node*> readers;
  readers.reserve(value->users().size());
  for (auto* user : value->users()) {
    bool reads = false;
    for (const auto& input : user->inputs()) {
      if (input.value == value) {
        reads = true;
        break;
      }
    }
    if (!reads) {
      continue;
    }
    if (std::find(readers.begin(), readers.end(), user) == readers.end()) {
      readers.push_back(user);
    }
  }
  if (readers.size() != value->users().size()) {
    // eraseUser drops every occurrence of a node, so clearing and re-adding
    // leaves exactly the real readers, one entry apiece.
    std::vector<nativert::Node*> current(
        value->users().begin(), value->users().end());
    for (auto* user : current) {
      value->eraseUser(user);
    }
    for (auto* user : readers) {
      value->addUser(user);
    }
  }
  graph.replaceAllUses(value, to);
}

// Points every reader of 'dup's outputs at the matching output of 'keeper',
// then offers each survivor to 'push' so consumers that only became congruent
// through this merge get revisited. Returns false, having changed nothing, when
// the two disagree on output arity: equal cseKeys make that impossible for
// nodes of the same op, so it means they were never really congruent.
bool mergeCseNode(
    nativert::Graph& graph,
    NodeCP keeper,
    NodeCP dup,
    const std::function<void(ValueCP)>& push) {
  const auto& from = dup->outputs();
  const auto& to = keeper->outputs();
  if (from.size() != to.size()) {
    return false;
  }
  // No TensorList output reaches here: cseEligible rejects any node that has
  // one, so there are never element ids to remap alongside the list value.
  for (size_t i = 0; i < from.size(); ++i) {
    replaceReaders(graph, from[i], const_cast<nativert::Value*>(to[i]));
    push(to[i]);
  }
  return true;
}

} // namespace

int64_t commonSubexpressions(nativert::Graph& graph, const ValueTypes& types) {
  const bool cseViews = WaveConfig::get().cseViews;
  const bool cseCompute = WaveConfig::get().cseCompute;
  if (!cseViews && !cseCompute) {
    return 0;
  }

  folly::F14FastSet<ValueCP> mutatedBases;
  for (const auto& node : graph.nodes()) {
    for (auto* mutated : dataMutatedInputs(&node)) {
      if (mutated != nullptr) {
        mutatedBases.insert(viewStorageBase(mutated));
      }
    }
  }

  std::deque<ValueCP> work;
  folly::F14FastSet<ValueCP> queued;
  auto push = [&](ValueCP value) {
    if (value != nullptr && queued.insert(value).second) {
      work.push_back(value);
    }
  };

  // Program order of every node. The survivor of a merge has to be the earlier
  // of the two: keeping the later one would leave the earlier one's consumers
  // reading a value produced after them, which makeParallelNodes rejects. User
  // lists are not in program order, so the first candidate a bucket sees is not
  // necessarily the first in the graph.
  folly::F14FastMap<NodeCP, size_t> order;
  {
    size_t index = 0;
    for (const auto& node : graph.nodes()) {
      order.emplace(&node, index++);
    }
  }

  int64_t merged = 0;
  auto mergeBucket = [&](folly::F14FastMap<std::string, NodeCP>& firstOf,
                         NodeCP node) {
    auto key = cseKey(node);
    if (key.empty()) {
      return;
    }
    // A value in this graph can be read by a node in a variant subgraph, which
    // shows up in users() but is not in this graph's node list. Those are not
    // ours to merge or to order.
    if (!order.contains(node)) {
      return;
    }
    auto [it, isNew] = firstOf.emplace(key, node);
    if (isNew) {
      return;
    }
    NodeCP keeper = it->second;
    NodeCP dup = node;
    if (order.at(dup) < order.at(keeper)) {
      std::swap(keeper, dup);
    }
    if (mergeCseNode(graph, keeper, dup, push)) {
      it->second = keeper;
      ++merged;
    }
  };

  // A node with no operands appears in no user list, so nothing would ever
  // bring two of them together; bucket those once up front. Factory ops
  // (aten.zeros and friends) are the case that matters.
  {
    folly::F14FastMap<std::string, NodeCP> firstOf;
    for (const auto& node : graph.nodes()) {
      if (node.inputs().empty() && !isDeadNode(&node) &&
          cseEligible(&node, types, mutatedBases, cseViews, cseCompute)) {
        mergeBucket(firstOf, &node);
      }
    }
  }

  // Congruent nodes have identical operands, so both appear in the user list of
  // every value they read. Walking user lists is therefore complete for any
  // node with an operand, and never hashes a node that has no possible partner.
  for (const auto* value : graph.values()) {
    if (value != nullptr && !value->users().empty()) {
      push(value);
    }
  }

  while (!work.empty()) {
    ValueCP value = work.front();
    work.pop_front();
    queued.erase(value);
    // Snapshot: merging rewires the list being walked.
    std::vector<NodeCP> users(value->users().begin(), value->users().end());
    folly::F14FastMap<std::string, NodeCP> firstOf;
    for (NodeCP user : users) {
      if (!isDeadNode(user) &&
          cseEligible(user, types, mutatedBases, cseViews, cseCompute)) {
        mergeBucket(firstOf, user);
      }
    }
  }

  if ((WaveConfig::get().trace & WaveConfig::kTiming) && merged > 0) {
    LOG(INFO) << "pre-partition CSE: merged " << merged << " node(s)";
  }
  return merged;
}

int64_t decomposeListOps(nativert::Graph& graph, WaveGraph& waveGraph) {
  // Snapshot: a rule rewrites the node it is given and may add nodes, and a
  // live walk would visit the replacements.
  std::vector<NodeCP> nodes;
  nodes.reserve(graph.nodes().size());
  for (const auto& node : graph.nodes()) {
    nodes.push_back(&node);
  }

  int64_t rewritten = 0;
  for (NodeCP node : nodes) {
    if (isDeadNode(node)) {
      continue;
    }
    const Metadata* meta = Registry::metadata(node->target());
    if (meta == nullptr || !meta->decompose) {
      continue;
    }
    if (meta->decompose(node, waveGraph)) {
      ++rewritten;
    }
  }

  if ((WaveConfig::get().trace & WaveConfig::kTiming) && rewritten > 0) {
    LOG(INFO) << "pre-partition decomposition: rewrote " << rewritten
              << " node(s)";
  }
  return rewritten;
}

namespace {

// The ScalarType newScalarValue needs to reproduce a scalar Value's type kind,
// or nullopt when the kind is not a scalar. Inverse of the mapping in
// WaveGraph::newScalarValue.
std::optional<c10::ScalarType> scalarTypeOfKind(nativert::Type::Kind kind) {
  switch (kind) {
    case nativert::Type::Kind::SymInt:
      return c10::ScalarType::Long;
    case nativert::Type::Kind::SymFloat:
      return c10::ScalarType::Double;
    case nativert::Type::Kind::SymBool:
      return c10::ScalarType::Bool;
    default:
      return std::nullopt;
  }
}

// An attribute holds a Constant, a variant whose unique_ptr<Graph> alternative
// makes the whole variant move-only. A getter never carries a subgraph, but
// check rather than silently duplicating a node without its attributes.
bool attributesAreCopyable(NodeCP node) {
  for (const auto& attr : node->attributes()) {
    if (std::holds_alternative<std::unique_ptr<nativert::Graph>>(attr.value)) {
      return false;
    }
  }
  return true;
}

void copyAttributes(NodeCP from, nativert::Node* to) {
  for (const auto& attr : from->attributes()) {
    std::visit(
        [&](const auto& constant) {
          using T = std::decay_t<decltype(constant)>;
          if constexpr (!std::is_same_v<T, std::unique_ptr<nativert::Graph>>) {
            to->addAttribute({attr.name, constant});
          }
        },
        attr.value);
  }
}

// The nodes the partitioner can see. makeParallelNodes seeds from the graph
// output node and walks back through input producers, so a node reachable only
// from an _assert_scalar chain is never placed in a ProjectNode and never
// counts toward a CSE border. Duplicating for such a user would pay a node and
// remove no boundary. Side-effect edges are deliberately not followed: they
// only add users, and over-counting is the direction that costs.
std::unordered_set<NodeCP> reachableFromOutputs(const nativert::Graph& graph) {
  std::unordered_set<NodeCP> reachable{graph.outputNode()};
  std::vector<NodeCP> pending{graph.outputNode()};
  while (!pending.empty()) {
    NodeCP node = pending.back();
    pending.pop_back();
    for (const auto& input : node->inputs()) {
      if (input.value == nullptr) {
        continue;
      }
      const auto* producer = input.value->producer();
      if (producer != nullptr && reachable.insert(producer).second) {
        pending.push_back(producer);
      }
    }
  }
  return reachable;
}

// Attaches 'constraint' to a value the pass just created. A rewrite-created
// value carries nothing by default: types.types is a no-op for a scalar, and a
// rank left at -1 has produced a rank-0 result before, so copy the original's
// rank and layout explicitly. graphOutput / externallyOwned are deliberately
// not carried -- the duplicate is a fresh internal value.
void setDuplicateConstraint(
    ValueTypes& types,
    ValueCP value,
    const ValueConstraint& constraint) {
  auto id = value->id();
  if (id < 0) {
    return;
  }
  if (static_cast<size_t>(id) >= types.constraints.size()) {
    types.constraints.resize(id + 1);
  }
  // Grown to id + 1 just above.
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  auto& target = types.constraints[id];
  target.rank = constraint.rank;
  target.contiguity = constraint.contiguity;
  target.zeroStrides = constraint.zeroStrides;
}

} // namespace

int64_t duplicateMetadataOps(
    nativert::Graph& graph,
    ValueTypes& types,
    WaveGraph& waveGraph) {
  NodeCP outputNode = graph.outputNode();
  constexpr int32_t kOutputPos = std::numeric_limits<int32_t>::max();
  const auto reachable = reachableFromOutputs(graph);

  std::unordered_map<NodeCP, int32_t> pos;
  int32_t idx = 0;
  for (const auto& node : graph.nodes()) {
    pos[&node] = &node == outputNode ? kOutputPos : idx++;
  }

  // Program position of each value's last live read, from the graph as it is
  // now. Every decision below is taken against this snapshot: inserting a
  // duplicate adds a read of the getter's operand, and reading a stale lastUse
  // is what keeps that from being mistaken for a lifetime extension.
  folly::F14FastMap<ValueCP, int32_t> lastUse;
  for (const auto& node : graph.nodes()) {
    if (reachable.count(&node) == 0) {
      continue;
    }
    const int32_t nodePos = pos.at(&node);
    for (const auto& input : node.inputs()) {
      if (input.value == nullptr) {
        continue;
      }
      auto [it, inserted] = lastUse.emplace(input.value, nodePos);
      if (!inserted) {
        it->second = std::max(it->second, nodePos);
      }
    }
  }
  auto lastUseOf = [&](ValueCP value) {
    auto it = lastUse.find(value);
    return it == lastUse.end() ? -1 : it->second;
  };
  auto reachableUsers = [&](ValueCP value) {
    int32_t count = 0;
    for (auto* user : value->users()) {
      count += reachable.count(user) > 0 ? 1 : 0;
    }
    return count;
  };

  // Snapshot the candidates: duplicating rewires users, and a live walk would
  // revisit the duplicates it just inserted. Two families qualify as free to
  // recompute: a getter, whose output is a scalar read out of tensor metadata,
  // and a view, which is set up host-side from its base's sizes and strides and
  // allocates nothing.
  std::vector<NodeCP> candidates;
  for (const auto& node : graph.nodes()) {
    const Metadata* meta = Registry::metadata(node.target());
    if (meta == nullptr || (!meta->isMetadataGetter && !meta->metadataOnly)) {
      continue;
    }
    if (node.outputs().size() != 1 || node.inputs().empty() ||
        node.outputs()[0] == nullptr || !attributesAreCopyable(&node)) {
      continue;
    }
    const auto kind = node.outputs()[0]->type().kind();
    const bool isGetter =
        meta->isMetadataGetter && scalarTypeOfKind(kind).has_value();
    const bool isViewOp = meta->metadataOnly && meta->isView() &&
        kind == nativert::Type::Kind::Tensor;
    if (!isGetter && !isViewOp) {
      continue;
    }
    candidates.push_back(&node);
  }

  int64_t duplicated = 0;
  int64_t viewsDuplicated = 0;
  int64_t bordersRemoved = 0;
  for (NodeCP node : candidates) {
    ValueCP out = node->outputs()[0];
    const bool isViewOp = out->type().kind() == nativert::Type::Kind::Tensor;
    // A returned size keeps the original: replaceAllUses-style rewiring does
    // not reach the output node, and it costs nothing to leave it there.
    bool feedsGraphOutput = false;
    bool writtenInPlace = false;
    std::vector<NodeCP> users;
    for (auto* user : out->users()) {
      if (user == outputNode) {
        feedsGraphOutput = true;
      } else if (reachable.count(user) > 0) {
        // A view written through by one user and read by another is a single
        // shared buffer, not two interchangeable aliases: splitting it would
        // send the write to a copy the readers never see.
        writtenInPlace |= inPlaceSelf(user) == out;
        users.push_back(user);
      }
    }
    if (writtenInPlace) {
      continue;
    }
    // The original serves the graph output if there is one, else the first
    // user. Only a value read more than once is a border worth breaking.
    const size_t firstToRewire = feedsGraphOutput ? 0 : 1;
    if (users.size() <= firstToRewire) {
      continue;
    }

    // Every operand must already be available at the use sites, so that moving
    // the read there neither creates a new border nor extends a buffer's
    // lifetime. Three ways to qualify:
    //   - producer-less (a graph input, weight or constant): live throughout;
    //   - a scalar with more than one reachable reader: already a CSE border
    //     the partitioner materializes, and a SymInt owns no buffer, so reading
    //     it later costs nothing. This is what admits the dynamic slice bounds
    //     -- slice(input, item(..), item(..)) with both items already
    //     materialized -- which the lifetime rule alone would reject;
    //   - otherwise, it must already outlive this node's own last use, which
    //     also implies a reader besides this node.
    const int32_t outLastUse = lastUseOf(out);
    auto isAvailable = [&](ValueCP value) {
      if (value == nullptr) {
        return false;
      }
      if (types.externallyOwned(value)) {
        return true;
      }
      const auto kind = value->type().kind();
      if ((kind == nativert::Type::Kind::SymInt ||
           kind == nativert::Type::Kind::SymFloat ||
           kind == nativert::Type::Kind::SymBool) &&
          reachableUsers(value) > 1) {
        return true;
      }
      return lastUseOf(value) >= outLastUse;
    };
    bool operandsAvailable = true;
    for (const auto& input : node->inputs()) {
      if (!isAvailable(input.value)) {
        operandsAvailable = false;
        break;
      }
    }
    if (!operandsAvailable) {
      continue;
    }

    // A view's duplicate needs the same dtype as the original, which for a
    // tensor lives in the TensorMeta rather than the Value's type kind. Without
    // a TensorMeta there is nothing to reproduce, so leave the node alone.
    const nativert::TensorMeta* outMeta =
        isViewOp && static_cast<size_t>(out->id()) < types.types.size()
        ? types.types[out->id()]
        : nullptr;
    if (isViewOp && outMeta == nullptr) {
      continue;
    }
    const auto dtype =
        isViewOp ? outMeta->dtype() : *scalarTypeOfKind(out->type().kind());
    const ValueConstraint constraint =
        static_cast<size_t>(out->id()) < types.constraints.size()
        // Bound tested on the line above.
        // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
        ? types.constraints[out->id()]
        : ValueConstraint{};
    for (size_t i = firstToRewire; i < users.size(); ++i) {
      auto* user = const_cast<nativert::Node*>(users[i]);
      auto* duplicate =
          graph.createNode(std::string(node->target()), node->inputs());
      copyAttributes(node, duplicate);
      // Immediately before the user, so the node list stays in program order
      // (computeSideEffectEdges checks this) and the operands still precede it.
      graph.insertBefore(duplicate, user);
      auto* duplicateOut = isViewOp
          ? waveGraph.newTensorValue(duplicate, out->name(), dtype)
          : waveGraph.newScalarValue(duplicate, out->name(), dtype);
      setDuplicateConstraint(types, duplicateOut, constraint);
      rewireInput(user, out, duplicateOut);
      ++duplicated;
      viewsDuplicated += isViewOp ? 1 : 0;
    }
    ++bordersRemoved;
  }

  if (duplicated > 0) {
    LOG(INFO) << "pre-partition metadata duplication: " << duplicated
              << " node(s) duplicated (" << viewsDuplicated << " view, "
              << duplicated - viewsDuplicated << " getter), " << bordersRemoved
              << " shared value(s) reduced to a single use";
  }
  return duplicated;
}

void ParallelNodes::computeReuseEligibility(
    const std::unordered_set<ValueCP>& graphOutputs) {
  // Distinct boundary inputs of each top-level expr (for the reusable-input
  // pass below). Indexed by ProjectNode id, which makeParallelProject assigns
  // from nextId_ as each node is created, so it is always a valid slot; at()
  // makes that an exception rather than a silent overrun if it ever is not.
  std::vector<std::vector<std::unordered_set<ValueCP>>> exprInputs(
      projectNodes_.size());
  for (auto& pnPtr : projectNodes_) {
    ProjectNode* pn = pnPtr.get();
    auto& perExpr = exprInputs.at(pn->id());
    perExpr.resize(pn->nodes().size());
    for (size_t i = 0; i < pn->nodes().size(); ++i) {
      collectExprBoundaryInputs(pn->nodes()[i], pn->inputs(), perExpr[i]);
    }
  }

  // A (now alias-corrected) lastUse value may be reused in place by an expr's
  // kernel op only when its storage is consumed by exactly one expr in this
  // layer. "Consumed" is judged by storage group, not raw value: an operand
  // that is a view of -- or otherwise shares storage with -- a value counts as
  // touching the same buffer, so reusing it in place would corrupt that
  // aliasing read. No later layer reads the storage: lastUse membership already
  // guarantees that via the viewStorageBase/baseLast grouping in
  // computeLastUse. Comparing raw value identity would miss aliases read by a
  // sibling expr, so group each expr's boundary inputs by viewStorageBase.
  std::unordered_map<ValueCP, ValueCP> storageBaseMemo;
  auto storageBase = [&](ValueCP value) -> ValueCP {
    auto it = storageBaseMemo.find(value);
    if (it != storageBaseMemo.end()) {
      return it->second;
    }
    ValueCP base = viewStorageBase(value);
    storageBaseMemo[value] = base;
    return base;
  };
  for (auto& pnPtr : projectNodes_) {
    ProjectNode* pn = pnPtr.get();
    const auto& perExpr = exprInputs.at(pn->id());
    pn->reusableValues_.assign(pn->nodes().size(), {});
    pn->overwritableTemps_.assign(pn->nodes().size(), {});

    // For each storage base touched at this layer's boundary, the set of exprs
    // reading any value that aliases it. A base read by exactly one expr has a
    // single in-layer consumer, so that expr may mutate the buffer in place.
    std::unordered_map<ValueCP, std::unordered_set<int32_t>> baseToExprs;
    for (size_t i = 0; i < perExpr.size(); ++i) {
      for (ValueCP w : perExpr[i]) {
        baseToExprs[storageBase(w)].insert(static_cast<int32_t>(i));
      }
    }

    for (ValueCP v : pn->lastUse) {
      auto it = baseToExprs.find(storageBase(v));
      if (it == baseToExprs.end() || it->second.size() != 1) {
        continue;
      }
      int32_t onlyExpr = *it->second.begin();
      // Flag 'v' for the expr that actually reads 'v' as a boundary input, so
      // its kernel has 'v''s buffer to write into (an aliasing sibling value in
      // the same group is flagged in its own iteration).
      if (perExpr[onlyExpr].count(v) > 0) {
        pn->reusableValues_[onlyExpr].push_back(v);
      }
    }

    // Expr-local overwritable temps: for each top-level expr, the values it
    // produces and consumes entirely internally. A value qualifies when it owns
    // its storage (storageBase(v) == v) and neither it nor any value aliasing
    // it escapes the expr subgraph -- i.e. no use in a sibling expr, a later
    // layer, or a graph output. Escape is detected per storage base: any
    // produced value whose group leaves the subgraph disqualifies that whole
    // base.
    for (size_t i = 0; i < pn->nodes().size(); ++i) {
      NodeCP exprRoot = pn->nodes()[i];
      if (pn->inputs().count(exprRoot) > 0) {
        continue; // boundary node from an earlier layer, not an expr root here
      }
      NodeSet subgraph;
      collectExprSubgraph(exprRoot, pn->inputs(), subgraph);
      std::unordered_set<ValueCP> escapingBase;
      std::vector<ValueCP> produced;
      for (NodeCP n : subgraph) {
        for (auto* v : n->outputs()) {
          if (v == nullptr) {
            continue;
          }
          produced.push_back(v);
          bool escapes = graphOutputs.count(v) > 0;
          if (!escapes) {
            for (auto* u : v->users()) {
              if (subgraph.count(u) == 0) {
                escapes = true;
                break;
              }
            }
          }
          if (escapes) {
            escapingBase.insert(storageBase(v));
          }
        }
      }
      for (ValueCP v : produced) {
        if (storageBase(v) == v && escapingBase.count(v) == 0) {
          pn->overwritableTemps_[i].push_back(v);
        }
      }
    }
  }
}

void ParallelNodes::computeLastUse(const nativert::Graph& graph) {
  // Clear per-layer last-use sets so this is safe to re-run after the in-place
  // pass rewrites the graph (reusableValues_/overwritableTemps_ are reassigned
  // in the third pass below; lastUse is only inserted into, so clear it here).
  for (auto& pnPtr : projectNodes_) {
    pnPtr->lastUse.clear();
  }
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

  // Reuse eligibility (reusableValues_ / overwritableTemps_) is a distinct
  // concern from the lifetime computation above; compute it in its own pass so
  // the core last-use logic stays readable.
  computeReuseEligibility(graphOutputs);

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

  // Returned metadata getters, excluded from the level walk so they stop
  // creating a CSE border on their operand. Put back before the last layer is
  // built (below) so they still run and still fill their output slot.
  //
  // The reachable set is built here rather than at function scope so a run with
  // the flag off does not pay for a walk of the whole graph.
  std::vector<NodeCP> deferredSizes;
  if (WaveConfig::get().deferSizeOutputs) {
    NodeSet reachable;
    std::vector<NodeCP> stack(topExprs.begin(), topExprs.end());
    while (!stack.empty()) {
      NodeCP n = stack.back();
      stack.pop_back();
      if (!reachable.insert(n).second) {
        continue;
      }
      for (auto* in : args(n)) {
        stack.push_back(in);
      }
    }
    for (auto* expr : top) {
      if (isDeferredSizeOutput(expr, root, reachable)) {
        deferredSizes.push_back(expr);
      }
    }
    for (auto* expr : deferredSizes) {
      top.erase(expr);
    }
  }

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

  // The deferred getters rejoin here: topExprs still lists them, so they are
  // already in the layer's node set; putting them back in 'top' is what makes
  // collectReachable pull their operands in as layer inputs.
  for (auto* expr : deferredSizes) {
    top.insert(expr);
  }

  auto* project = makeParallelProject(current, top, topExprs);
  if (project == nullptr) {
    TORCH_CHECK(false, "makeParallelProject returned null");
  }
  current = project;

  if ((WaveConfig::get().trace & WaveConfig::kTiming) &&
      WaveConfig::get().deferSizeOutputs) {
    int32_t layers = 0;
    for (auto* p = current; p != nullptr; p = p->input()) {
      ++layers;
    }
    std::cout << fmt::format(
        "partition: {} layers, last layer has {} exprs, {} size outputs deferred\n",
        layers,
        current->nodes().size(),
        deferredSizes.size());
  }

  // All layers are now built in execution order; annotate each with its
  // last-use / reusable-last-use values.
  computeLastUse(graph);

  return current;
}

void ParallelNodes::rewriteInPlace(
    nativert::Graph& graph,
    const ValueTypes& types) {
  // Storage bases mutated anywhere in the graph. A clone of such a base may be
  // a required pre-mutation snapshot, so it is never treated as a no-op.
  std::unordered_set<ValueCP> mutatedBases;
  for (const auto& node : graph.nodes()) {
    for (auto* mv : dataMutatedInputs(&node)) {
      if (mv != nullptr) {
        mutatedBases.insert(viewStorageBase(mv));
      }
    }
  }

  int64_t elided = 0;
  for (auto& pnPtr : projectNodes_) {
    ProjectNode* pn = pnPtr.get();
    auto ordered = layerNodesInOrder(pn);
    NodeSet layerNodes(ordered.begin(), ordered.end());
    // Snapshot the clone nodes first: eliding one clone leaves the others
    // untouched, and a re-walk would see the now-dead clone.
    std::vector<NodeCP> clones;
    for (NodeCP n : ordered) {
      if (n->target() == "torch.ops.aten.clone.default") {
        clones.push_back(n);
      }
    }
    for (NodeCP cloneNode : clones) {
      if (cloneNode->outputs().empty() || cloneNode->inputs().empty()) {
        continue;
      }
      // Read the source before the rewrite: it is the buffer whose copy is
      // saved, and it identifies the elision once the clone output is dead.
      ValueCP source = cloneNode->inputs()[0].value;
      // A multi-user clone is a CSE border consumed across layers (every use is
      // rewired); a single-user clone is elided in place within its layer.
      // Fusion status of the consumer does not affect either path.
      const bool multiUser = cloneNode->outputs()[0]->users().size() > 1;
      const bool ok = multiUser
          ? tryElideMultiUserClone(cloneNode, graph, mutatedBases, types)
          : tryElideClone(cloneNode, pn, layerNodes, types);
      if (ok) {
        ++elided;
        if (source != nullptr) {
          ++pn->elidedCloneCounts[source->id()];
        }
      }
    }
  }
  // The graph changed (uses rewired, clones now dead); refresh the alias-aware
  // last-use / reuse sets so downstream freeing and reuse stay correct with the
  // possibly cross-layer extended input lifetimes.
  if (elided > 0) {
    computeLastUse(graph);
  }
  if ((WaveConfig::get().trace & WaveConfig::kTiming) && elided > 0) {
    LOG(INFO) << "in-place pass: elided " << elided << " clone(s)";
  }
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
