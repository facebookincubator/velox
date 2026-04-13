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

#include "velox/experimental/torchwave/Project.h"
#include "velox/experimental/torchwave/Utils.h"

namespace torch::wave {

namespace {

// Returns true if 'node' is a call-like expression (has producer inputs).
bool isCallExpr(NodeCP node) {
  for (const auto& input : node->inputs()) {
    auto* producer = input.value->producer();
    if (producer != nullptr && producer != node) {
      return true;
    }
  }
  return false;
}

// Formats a leaf value using tensor metadata from the graph.
std::string leafValueString(
    std::string_view valueName,
    const nativert::Graph& graph) {
  const nativert::TensorMeta* tm = nullptr;
  std::string name(valueName);
  auto it = graph.tensorValuesMeta().find(name);
  if (it != graph.tensorValuesMeta().end()) {
    tm = &it->second;
  } else {
    auto wit = graph.weightsMeta().find(name);
    if (wit != graph.weightsMeta().end()) {
      tm = &wit->second;
    }
  }
  if (tm != nullptr && !tm->hasSymbolicShape()) {
    auto sizes = tm->sizes();
    if (sizes.empty()) {
      return name; // scalar
    }
    std::string result = "<literal [";
    for (int64_t i = 0; i < static_cast<int64_t>(sizes.size()); ++i) {
      if (i > 0) {
        result += ", ";
      }
      result += std::to_string(sizes[i]);
    }
    result += "]>";
    return result;
  }
  return name;
}

void nodeExprStringImpl(
    std::stringstream& ss,
    NodeCP node,
    const nativert::Graph& graph,
    PlanObjectSet& border) {
  if (!isCallExpr(node)) {
    ss << node->target();
    if (!node->attributes().empty()) {
      ss << "(";
      bool first = true;
      for (const auto& attr : node->attributes()) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << attr.name << "=" << constantToString(attr.value);
      }
      ss << ")";
    }
    return;
  }

  ss << node->target() << "(";
  bool first = true;
  for (const auto& input : node->inputs()) {
    if (!first) {
      ss << ", ";
    }
    first = false;

    auto* value = input.value;
    auto* producer = value->producer();

    if (producer == nullptr || producer == node) {
      ss << leafValueString(value->name(), graph);
    } else if (border.count(producer)) {
      ss << "%" << value->id();
    } else {
      nodeExprStringImpl(ss, producer, graph, border);
    }
  }
  for (const auto& attr : node->attributes()) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << attr.name << "=" << constantToString(attr.value);
  }
  ss << ")";
}

void formatOutputIds(std::stringstream& ss, NodeCP node) {
  const auto& outputs = node->outputs();
  if (outputs.empty()) {
    return;
  }
  std::vector<int> ids;
  ids.reserve(outputs.size());
  for (const auto* v : outputs) {
    if (v != nullptr) {
      ids.push_back(v->id());
    }
  }
  if (ids.empty()) {
    return;
  }
  if (ids.size() == 1) {
    ss << "%" << ids[0];
    return;
  }
  std::sort(ids.begin(), ids.end());

  bool first = true;
  size_t i = 0;
  while (i < ids.size()) {
    size_t j = i;
    while (j + 1 < ids.size() && ids[j + 1] == ids[j] + 1) {
      ++j;
    }
    if (!first) {
      ss << ", ";
    }
    first = false;
    if (j > i) {
      ss << "[%" << ids[i] << " - %" << ids[j] << "]";
    } else {
      ss << "%" << ids[i];
    }
    i = j + 1;
  }
}

} // namespace

std::string nodeExprString(
    NodeCP node,
    const nativert::Graph& graph,
    PlanObjectSet& border) {
  std::stringstream ss;
  nodeExprStringImpl(ss, node, graph, border);
  return ss.str();
}

std::string ProjectNode::toString(
    const nativert::Graph& graph,
    PlanObjectSet& border) const {
  std::stringstream ss;
  PlanObjectSet localBorder(inputs_.begin(), inputs_.end());
  localBorder.insert(border.begin(), border.end());
  ss << fmt::format("ProjectNode {}:\n", id_);
  for (int32_t i = 0; i < static_cast<int32_t>(nodes_.size()); ++i) {
    ss << fmt::format("  {}.{}: ", id_, i);
    formatOutputIds(ss, nodes_[i]);
    ss << " = " << nodeExprString(nodes_[i], graph, localBorder) << "\n";
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
