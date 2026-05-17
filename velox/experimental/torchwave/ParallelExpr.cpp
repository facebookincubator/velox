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
#include <memory>
#include <ranges>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Utils.h"

namespace torch::wave {

namespace {

// Returns the input nodes of 'expr'. Loops over inputs() of the node and adds
// the producer of the Value of each NamedArgument if not null.
std::vector<ExprCP> args(ExprCP expr) {
  std::vector<ExprCP> result;
  for (const auto& input : expr->inputs()) {
    auto* producer = input.value->producer();
    if (producer != nullptr && producer != expr) {
      result.push_back(producer);
    }
  }
  return result;
}

// Returns true if 'expr' is a call-like expression (has arguments).
bool isCallExpr(ExprCP expr) {
  return !args(expr).empty();
}

// Adds 'expr' and all its transitive subexpressions to 'result'.
void subexpressionsInner(ExprCP expr, PlanObjectSet& result) {
  for (auto arg : args(expr)) {
    if (result.count(arg)) {
      continue;
    }
    result.insert(arg);
    subexpressionsInner(arg, result);
  }
}

// Returns all transitive subexpressions of 'expr'.
PlanObjectSet subexpressions(ExprCP expr) {
  PlanObjectSet result;
  subexpressionsInner(expr, result);
  return result;
}

// Returns a unit cost for a node.
float selfCost(ExprCP /*expr*/) {
  return 1.0f;
}

struct LevelData {
  PlanObjectSet exprs;
};

size_t levelOf(std::vector<LevelData>& levels, ExprCP expr) {
  for (size_t i = 0; i < levels.size(); ++i) {
    if (levels[i].exprs.count(expr)) {
      return i;
    }
  }
  __builtin_unreachable();
}

void pushdownExpr(
    ExprCP expr,
    int32_t level,
    std::vector<LevelData>& levelData) {
  const auto defined = levelOf(levelData, expr);
  if (defined >= static_cast<size_t>(level)) {
    return;
  }

  if (level >= static_cast<int32_t>(levelData.size())) {
    levelData.resize(level + 1);
  }
  levelData[defined].exprs.erase(expr);
  levelData[level].exprs.insert(expr);

  for (auto input : args(expr)) {
    pushdownExpr(input, level + 1, levelData);
  }
}

void makeLevelsInner(
    ExprCP expr,
    int32_t level,
    std::vector<LevelData>& levelData,
    std::unordered_map<ExprCP, int32_t>& refCount,
    PlanObjectSet& counted) {
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
  levelData[level].exprs.insert(expr);
  for (auto input : args(expr)) {
    makeLevelsInner(input, level + 1, levelData, refCount, counted);
  }
}

void makeExprLevels(
    const PlanObjectSet& exprs,
    std::vector<LevelData>& levelData,
    std::unordered_map<ExprCP, int32_t>& refCount) {
  PlanObjectSet counted;
  for (auto expr : exprs) {
    makeLevelsInner(expr, 0, levelData, refCount, counted);
  }
}

PlanObjectSet makeCseBorder(
    const std::vector<LevelData>& levelData,
    const PlanObjectSet& placed,
    std::unordered_map<ExprCP, int32_t>& refCount) {
  PlanObjectSet border;
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

float parallelBorder(
    ExprCP expr,
    const PlanObjectSet& placed,
    PlanObjectSet& result) {
  constexpr float kSplit = -1;
  constexpr float kTargetCost = 50;
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
PlanObjectSet collectAllInputExprs(const ProjectNode* node) {
  PlanObjectSet allExprs;
  for (auto* cur = node; cur != nullptr; cur = cur->input()) {
    allExprs.insert(cur->nodes().begin(), cur->nodes().end());
  }
  return allExprs;
}

// Traverses the subgraph reachable from 'expr', collecting nodes that are in
// 'inputExprs' into 'reachable' and nodes with no inputs that are not in
// 'inputExprs' into 'leafInputs'. Stops recursing into a branch once it hits a
// node in 'inputExprs'.
void collectReachableAndLeaves(
    ExprCP expr,
    const PlanObjectSet& inputExprs,
    PlanObjectSet& visited,
    PlanObjectSet& reachable,
    PlanObjectSet& leafInputs) {
  if (!visited.insert(expr).second) {
    return;
  }
  if (inputExprs.count(expr)) {
    reachable.insert(expr);
    return;
  }
  auto inputs = args(expr);
  if (inputs.empty()) {
    leafInputs.insert(expr);
    return;
  }
  for (auto* child : inputs) {
    collectReachableAndLeaves(
        child, inputExprs, visited, reachable, leafInputs);
  }
}

} // namespace

ProjectNode* ParallelNodes::makeParallelProject(
    ProjectNode* input,
    const PlanObjectSet& topExprs,
    std::vector<ExprCP> orderedExprs) {
  if (orderedExprs.empty()) {
    orderedExprs.assign(topExprs.begin(), topExprs.end());
    std::sort(orderedExprs.begin(), orderedExprs.end(), [](ExprCP a, ExprCP b) {
      auto idOf = [](ExprCP e) {
        return e->outputs().empty() ? 0 : e->outputs()[0]->id();
      };
      return idOf(a) < idOf(b);
    });
  }
  std::vector<NodeCP> nodes;
  PlanObjectSet seen;
  for (auto* expr : orderedExprs) {
    if (seen.insert(expr).second) {
      nodes.push_back(expr);
    }
  }
  std::unordered_set<NodeCP> inputs;
  std::unordered_set<NodeCP> leafInputs;

  if (input != nullptr) {
    auto allInputExprs = collectAllInputExprs(input);

    PlanObjectSet visited;
    PlanObjectSet reachable;
    PlanObjectSet leaves;
    for (auto* expr : topExprs) {
      collectReachableAndLeaves(
          expr, allInputExprs, visited, reachable, leaves);
    }
    inputs = std::move(reachable);
    leafInputs = std::move(leaves);
  }

  auto projectNode = std::make_unique<ProjectNode>(
      std::move(nodes),
      std::move(inputs),
      std::move(leafInputs),
      input,
      nextId_++);
  auto* result = projectNode.get();
  projectNodes_.push_back(std::move(projectNode));
  return result;
}

ProjectNode* ParallelNodes::makeParallelNodes(const nativert::Graph& graph) {
  ExprCP root = graph.outputNode();
  auto topExprs = args(root);
  PlanObjectSet top(topExprs.begin(), topExprs.end());

  std::vector<LevelData> levelData;
  std::unordered_map<ExprCP, int32_t> refCount;
  makeExprLevels(top, levelData, refCount);

  PlanObjectSet placed;
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

    current = makeParallelProject(current, cses);
  }

  PlanObjectSet parallel;
  for (auto expr : top) {
    parallelBorder(expr, placed, parallel);
  }

  if (!parallel.empty()) {
    for (auto expr : parallel) {
      auto subs = subexpressions(expr);
      placed.insert(subs.begin(), subs.end());
    }
    placed.insert(parallel.begin(), parallel.end());

    current = makeParallelProject(current, parallel);
  }

  if (nextId_ > 1000) {
    printSet(placed);
    std::unordered_map<ExprCP, int32_t> refs;
    printRefcount(refs, 0);
  }
  current = makeParallelProject(current, top, topExprs);

  return current;
}

__attribute__((used)) void printSet(PlanObjectSet& set) {
  fmt::print("{}\n", set.size());
  for (const auto* node : set) {
    auto ptr = reinterpret_cast<int64_t>(node);
    fmt::print("{:x} {}\n", (ptr >> 3) & 0xffff, std::string(node->target()));
  }
}

__attribute__((used)) void printRefcount(
    std::unordered_map<ExprCP, int32_t>& refCount,
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
