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

#include "velox/experimental/torchwave/NodePrinter.h"

#include <algorithm>

#include "velox/experimental/torchwave/Project.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {

namespace {

char dtypeLetter(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return 'f';
    case c10::ScalarType::Double:
      return 'd';
    case c10::ScalarType::Half:
      return 'h';
    case c10::ScalarType::BFloat16:
      return 'B';
    case c10::ScalarType::Long:
      return 'L';
    case c10::ScalarType::Int:
      return 'I';
    case c10::ScalarType::Short:
      return 'S';
    case c10::ScalarType::Byte:
      return 'U';
    case c10::ScalarType::Char:
      return 'C';
    case c10::ScalarType::Bool:
      return 'b';
    default:
      return '?';
  }
}

void formatTypeAnnotation(
    std::stringstream& ss,
    ValueCP value,
    const ValueTypes& types) {
  auto kind = value->type().kind();
  if (kind == nativert::Type::Kind::Tensor) {
    auto r = types.rank(value);
    auto id = value->id();
    char dl = '?';
    if (static_cast<size_t>(id) < types.types.size() && types.types[id]) {
      dl = dtypeLetter(types.types[id]->dtype());
    }
    // '#' after the dtype letter marks a value not known to be contiguous.
    const char* nc = types.contiguous(value) ? "" : "#";
    if (r < 0) {
      ss << "(?" << dl << nc << ")";
    } else {
      ss << "(" << static_cast<int>(r) << "D" << dl << nc << ")";
    }
  } else if (kind == nativert::Type::Kind::SymInt) {
    ss << "(L)";
  } else if (kind == nativert::Type::Kind::SymFloat) {
    ss << "(f)";
  } else if (kind == nativert::Type::Kind::SymBool) {
    ss << "(b)";
  }
}

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
      return name;
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

void printOutputIdsCompressed(std::stringstream& ss, NodeCP node) {
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

std::string shortenTarget(std::string_view target) {
  std::string result(target);
  static const std::vector<std::string> prefixes = {
      "torch.ops.aten.", "torch.ops.fb.", "torch.ops.fbgemm."};
  for (const auto& prefix : prefixes) {
    if (result.size() > prefix.size() &&
        result.compare(0, prefix.size(), prefix) == 0) {
      result = result.substr(prefix.size());
      break;
    }
  }
  static const std::vector<std::string> suffixes = {
      ".Tensor_default", ".Tensor", ".default"};
  for (const auto& suffix : suffixes) {
    if (result.size() > suffix.size() &&
        result.compare(result.size() - suffix.size(), suffix.size(), suffix) ==
            0) {
      result = result.substr(0, result.size() - suffix.size());
      break;
    }
  }
  return result;
}

} // namespace

NodePrinter::NodePrinter(PrintOptions options) : options_(std::move(options)) {}

bool NodePrinter::isLeaf(ValueCP value) const {
  if (!value->producer()) {
    return true;
  }
  if (options_.boundaryValues && options_.boundaryValues->count(value)) {
    return true;
  }
  if (options_.boundaryNodes &&
      options_.boundaryNodes->count(value->producer())) {
    return true;
  }
  if (options_.allowedNodes &&
      !options_.allowedNodes->count(value->producer())) {
    return true;
  }
  return false;
}

std::string NodePrinter::formatTarget(std::string_view target) const {
  if (options_.shortNames) {
    return shortenTarget(target);
  }
  return std::string(target);
}

void NodePrinter::printValueId(std::stringstream& ss, ValueCP value) const {
  if (!value) {
    ss << "<null>";
    return;
  }
  if (options_.valueNames) {
    ss << "%" << value->name();
  } else {
    auto id = value->id();
    if (options_.formalToActual) {
      auto it = options_.formalToActual->find(id);
      if (it != options_.formalToActual->end()) {
        id = it->second;
      }
    }
    ss << "%" << id;
  }
}

void NodePrinter::printValueRef(std::stringstream& ss, ValueCP value) const {
  if (!value) {
    ss << "<null>";
    return;
  }
  if (options_.showTypes && options_.valueTypes) {
    formatTypeAnnotation(ss, value, *options_.valueTypes);
  }
  // Mark a reusable last use: the operand is a boundary input of only one expr
  // in this ProjectNode and never read again (directly or via alias), so its
  // buffer is free to mutate in place.
  if (options_.projectNode != nullptr &&
      options_.projectNode->isReusableInput(value)) {
    ss << "& ";
  }
  if (options_.useGraphNames && options_.graph) {
    ss << leafValueString(value->name(), *options_.graph);
  } else {
    printValueId(ss, value);
  }
}

void NodePrinter::printOutputIds(std::stringstream& ss, NodeCP node) const {
  const auto& outputs = node->outputs();
  if (outputs.empty()) {
    return;
  }
  if (!options_.valueNames && options_.compressOutputRanges) {
    printOutputIdsCompressed(ss, node);
    return;
  }
  ss << "(";
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    printValueId(ss, outputs[i]);
  }
  ss << ")";
}

void NodePrinter::collectSummaryForValue(
    ValueCP value,
    std::unordered_set<ValueCP>& seenLeaves,
    SubtreeSummary& summary) const {
  if (isLeaf(value)) {
    if (seenLeaves.insert(value).second) {
      ++summary.distinctLeaves;
    }
    return;
  }
  collectSummary(value->producer(), seenLeaves, summary);
}

void NodePrinter::collectSummary(
    NodeCP node,
    std::unordered_set<ValueCP>& seenLeaves,
    SubtreeSummary& summary) const {
  auto name = formatTarget(node->target());
  ++summary.functionCounts[name];
  for (const auto& input : node->inputs()) {
    collectSummaryForValue(input.value, seenLeaves, summary);
  }
}

void NodePrinter::printSummary(
    std::stringstream& ss,
    const SubtreeSummary& summary) const {
  std::vector<std::pair<int32_t, std::string>> entries;
  entries.reserve(summary.functionCounts.size());
  for (const auto& [name, count] : summary.functionCounts) {
    entries.emplace_back(count, name);
  }
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });
  ss << "<";
  bool first = true;
  for (const auto& [count, name] : entries) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << count << " " << name;
  }
  if (!first) {
    ss << ", ";
  }
  ss << summary.distinctLeaves << " values>";
}

void NodePrinter::printExprImpl(
    std::stringstream& ss,
    NodeCP node,
    int32_t depth) const {
  if (node->inputs().empty() && node->attributes().empty()) {
    ss << formatTarget(node->target());
    return;
  }
  ss << formatTarget(node->target()) << "(";
  bool first = true;
  int32_t argIndex = 0;
  SubtreeSummary truncatedSummary;
  std::unordered_set<ValueCP> truncatedLeaves;
  int32_t truncatedCount = 0;
  bool truncated = false;
  for (const auto& input : node->inputs()) {
    if (options_.maxLength > 0 && argIndex >= options_.maxLength) {
      collectSummaryForValue(input.value, truncatedLeaves, truncatedSummary);
      ++truncatedCount;
      truncated = true;
      continue;
    }
    if (!first) {
      ss << ", ";
    }
    first = false;
    auto* value = input.value;
    bool isBreakout =
        options_.breakoutValues && options_.breakoutValues->count(value);
    if (isLeaf(value) || isBreakout) {
      printValueRef(ss, value);
    } else if (options_.inlineIntermediates) {
      if (options_.maxDepth > 0 && depth >= options_.maxDepth) {
        SubtreeSummary depthSummary;
        std::unordered_set<ValueCP> depthLeaves;
        collectSummaryForValue(value, depthLeaves, depthSummary);
        printSummary(ss, depthSummary);
      } else {
        if (options_.showOutputIds) {
          printValueId(ss, value);
          ss << " = ";
        }
        printExprImpl(ss, value->producer(), depth + 1);
      }
    } else {
      printValueRef(ss, value);
    }
    ++argIndex;
  }
  if (truncated) {
    if (!first) {
      ss << ", ";
    }
    ss << "... " << truncatedCount << " more, ";
    printSummary(ss, truncatedSummary);
  }
  if (options_.showAttributes) {
    for (const auto& attr : node->attributes()) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      ss << attr.name << "=" << constantToString(attr.value);
    }
  }
  ss << ")";
}

void NodePrinter::printImpl(
    std::stringstream& ss,
    NodeCP node,
    std::unordered_set<NodeCP>& visited) const {
  if (!visited.insert(node).second) {
    return;
  }

  if (!options_.inlineIntermediates) {
    for (const auto& input : node->inputs()) {
      auto* producer = input.value->producer();
      if (producer && !isLeaf(input.value)) {
        printImpl(ss, producer, visited);
      }
    }
  }

  if (options_.showOutputIds) {
    printOutputIds(ss, node);
    ss << " = ";
  }
  printExprImpl(ss, node, 0);
  ss << "\n";
}

namespace {

void collectBreakoutNodes(
    NodeCP node,
    const std::unordered_set<ValueCP>& breakoutValues,
    const NodePrinter& printer,
    std::unordered_set<NodeCP>& visited,
    std::vector<NodeCP>& result) {
  if (!visited.insert(node).second) {
    return;
  }
  for (const auto& input : node->inputs()) {
    auto* producer = input.value->producer();
    if (!producer || printer.isLeaf(input.value)) {
      continue;
    }
    collectBreakoutNodes(producer, breakoutValues, printer, visited, result);
    if (breakoutValues.count(input.value)) {
      result.push_back(producer);
    }
  }
}

} // namespace

std::string NodePrinter::print(NodeCP node) const {
  std::stringstream ss;
  if (options_.inlineIntermediates) {
    if (options_.breakoutValues && !options_.breakoutValues->empty()) {
      std::unordered_set<NodeCP> visited;
      std::vector<NodeCP> breakouts;
      collectBreakoutNodes(
          node, *options_.breakoutValues, *this, visited, breakouts);
      for (auto* bNode : breakouts) {
        std::vector<ValueCP> bOutputs;
        for (auto* v : bNode->outputs()) {
          if (options_.breakoutValues->count(v)) {
            bOutputs.push_back(v);
          }
        }
        if (!bOutputs.empty()) {
          ss << "(";
          for (size_t i = 0; i < bOutputs.size(); ++i) {
            if (i > 0) {
              ss << ", ";
            }
            printValueId(ss, bOutputs[i]);
          }
          ss << ") = ";
          printExprImpl(ss, bNode, 0);
          ss << "\n";
        }
      }
    }
    if (options_.showOutputIds) {
      printOutputIds(ss, node);
      ss << " = ";
    }
    printExprImpl(ss, node, 0);
  } else {
    std::unordered_set<NodeCP> visited;
    printImpl(ss, node, visited);
  }
  return ss.str();
}

// --- Static presets ---

std::string NodePrinter::expr(NodeCP node) {
  PrintOptions opts;
  opts.inlineIntermediates = true;
  opts.showOutputIds = false;

  return NodePrinter(opts).print(node);
}

std::string NodePrinter::values(NodeCP node) {
  PrintOptions opts;
  opts.inlineIntermediates = false;
  opts.showOutputIds = true;

  return NodePrinter(opts).print(node);
}

std::string NodePrinter::detailed(NodeCP node) {
  PrintOptions opts;
  opts.inlineIntermediates = false;
  opts.showOutputIds = true;
  opts.showTypes = true;

  opts.compressOutputRanges = true;
  return NodePrinter(opts).print(node);
}

std::string NodePrinter::one(NodeCP node) {
  PrintOptions opts;
  opts.inlineIntermediates = false;
  opts.showOutputIds = true;
  opts.shortNames = true;

  std::unordered_set<NodeCP> empty;
  opts.allowedNodes = &empty;
  std::stringstream ss;
  NodePrinter printer(opts);
  printer.printOutputIds(ss, node);
  ss << " = ";
  printer.printExprImpl(ss, node, 0);
  return ss.str();
}

static PrintOptions*& threadLocalDefaults() {
  static thread_local PrintOptions* instance = nullptr;
  return instance;
}

PrintOptions& NodePrinter::defaults() {
  auto* ptr = threadLocalDefaults();
  if (ptr != nullptr) {
    return *ptr;
  }
  static PrintOptions instance;
  return instance;
}

void NodePrinter::setDefaults(const PrintOptions& opts) {
  defaults() = opts;
}

WithPrintOptions::WithPrintOptions(const std::string& opts)
    : previous_(threadLocalDefaults()),
      options_(NodePrinter::parsePrintOptions(opts)) {
  threadLocalDefaults() = &options_;
}

WithPrintOptions::WithPrintOptions(PrintOptions opts)
    : previous_(threadLocalDefaults()), options_(std::move(opts)) {
  threadLocalDefaults() = &options_;
}

WithPrintOptions::~WithPrintOptions() {
  threadLocalDefaults() = previous_;
}

PrintOptions NodePrinter::parsePrintOptions(const std::string& opts) {
  PrintOptions result;
  if (opts.empty()) {
    return result;
  }
  std::istringstream stream(opts);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    bool negate = false;
    std::string_view key = token;
    if (key[0] == '-') {
      negate = true;
      key = key.substr(1);
    }
    if (key[0] == 'D' && key.size() > 1) {
      result.maxDepth = std::stoi(std::string(key.substr(1)));
    } else if (key[0] == 'L' && key.size() > 1) {
      result.maxLength = std::stoi(std::string(key.substr(1)));
    } else if (key == "S") {
      result.shortNames = !negate;
    } else if (key == "V") {
      result.inlineIntermediates = negate;
    } else if (key == "NA") {
      result.showAttributes = negate;
    } else if (key == "VN") {
      result.valueNames = !negate;
    } else if (key == "II") {
      result.inlineIntermediates = !negate;
    } else if (key == "GN") {
      result.useGraphNames = !negate;
    } else if (key == "OI") {
      result.showOutputIds = !negate;
    } else if (key == "CR") {
      result.compressOutputRanges = !negate;
    } else if (key == "T") {
      result.showTypes = !negate;
    } else if (key == "OF") {
      result.showOutputFlags = !negate;
    }
  }
  return result;
}

} // namespace torch::wave
