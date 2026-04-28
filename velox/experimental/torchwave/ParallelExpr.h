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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/experimental/torchwave/Project.h"

namespace torch::wave {

using ExprCP = const nativert::Node*;

using PlanObjectSet = std::unordered_set<const nativert::Node*>;

class ParallelNodes {
 public:
  ProjectNode* makeParallelProject(
      ProjectNode* input,
      const PlanObjectSet& topExprs,
      std::vector<ExprCP> orderedExprs = {});

  ProjectNode* makeParallelNodes(const nativert::Graph& graph);

 private:
  std::vector<std::unique_ptr<ProjectNode>> projectNodes_;
  int32_t nextId_{0};
};

// Produce a human-readable expression string for a node.
// For a non-function node, returns the target. For a function node, prints
// target(arg1, arg2, ...) where each arg is either a literal shape, a border
// node target, or a recursive expression.
std::string nodeExprString(
    const nativert::Node* node,
    const nativert::Graph& graph,
    PlanObjectSet& border);

void printSet(PlanObjectSet& set);

void printRefcount(std::unordered_map<ExprCP, int32_t>& refCount, int32_t min);

} // namespace torch::wave
