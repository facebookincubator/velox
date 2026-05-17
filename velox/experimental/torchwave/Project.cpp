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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Project.h"

namespace torch::wave {

namespace {

void distinctFunctionsInner(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Node*>& inputs,
    std::unordered_set<const nativert::Node*>& visited,
    std::unordered_map<std::string, int32_t>& counts) {
  if (!visited.insert(node).second) {
    return;
  }
  if (inputs.count(node)) {
    return;
  }

  // Check if this node represents a function (has inputs from other nodes).
  bool isFunction = false;
  for (const auto& input : node->inputs()) {
    auto* producer = input.value->producer();
    if (producer != nullptr && producer != node) {
      isFunction = true;
      break;
    }
  }

  if (isFunction) {
    // Build key: target followed by sorted literal attributes.
    std::string key(node->target());
    const auto& attrs = node->attributes();
    if (!attrs.empty()) {
      std::vector<std::pair<std::string, std::string>> sorted;
      sorted.reserve(attrs.size());
      for (const auto& attr : attrs) {
        sorted.emplace_back(attr.name, constantToString(attr.value));
      }
      std::sort(sorted.begin(), sorted.end());
      for (const auto& [name, value] : sorted) {
        key += " ";
        key += name;
        key += "=";
        key += value;
      }
    }
    ++counts[key];
  }

  // Recurse into inputs.
  for (const auto& input : node->inputs()) {
    auto* producer = input.value->producer();
    if (producer != nullptr && producer != node) {
      distinctFunctionsInner(producer, inputs, visited, counts);
    }
  }
}

} // namespace

void ProjectNode::distinctFunctions(
    std::unordered_map<std::string, int32_t>& counts) const {
  std::unordered_set<const nativert::Node*> visited;
  for (const auto* node : nodes_) {
    distinctFunctionsInner(node, inputs_, visited, counts);
  }
}

} // namespace torch::wave
