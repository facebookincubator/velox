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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Project.h"
#include "velox/experimental/torchwave/Utils.h"

#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

std::string ProjectNode::toString(
    const nativert::Graph& graph,
    NodeSet& border,
    const ValueTypes* valueTypes) const {
  NodeSet localBorder(inputs_.begin(), inputs_.end());
  localBorder.insert(border.begin(), border.end());

  PrintOptions opts = NodePrinter::defaults();
  opts.inlineIntermediates = true;
  opts.compressOutputRanges = true;
  opts.boundaryNodes = &localBorder;
  opts.graph = &graph;
  opts.valueTypes = valueTypes;
  opts.showTypes = valueTypes != nullptr;
  NodePrinter printer(opts);

  std::stringstream ss;
  ss << fmt::format("ProjectNode {}:\n", id_);
  for (int32_t i = 0; i < static_cast<int32_t>(nodes_.size()); ++i) {
    ss << fmt::format("  {}.{}: ", id_, i);
    if (localBorder.count(nodes_[i])) {
      printer.printOutputIds(ss, nodes_[i]);
      ss << "\n";
    } else if (opts.showOutputIds) {
      ss << printer.print(nodes_[i]) << "\n";
    } else {
      printer.printOutputIds(ss, nodes_[i]);
      ss << " = " << printer.print(nodes_[i]) << "\n";
    }
  }
  if (input_ != nullptr) {
    ss << fmt::format("  input: ProjectNode {}\n", input_->id());
  }
  return ss.str();
}

namespace {

void distinctFunctionsInner(
    NodeCP node,
    const std::unordered_set<NodeCP>& inputs,
    std::unordered_set<NodeCP>& visited,
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
  std::unordered_set<NodeCP> visited;
  for (const auto* node : nodes_) {
    distinctFunctionsInner(node, inputs_, visited, counts);
  }
}

} // namespace torch::wave
