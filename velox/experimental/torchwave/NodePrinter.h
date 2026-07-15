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

#include <sstream>
#include <string>
#include <unordered_set>

#include <torch/nativert/graph/Graph.h>

#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

class ProjectNode;

/// Controls formatting, recursion depth, and name style for NodePrinter output.
struct PrintOptions {
  // --- Value reference format ---

  /// If true, leaf values use graph tensor metadata to show shapes
  /// (e.g. "<literal [3, 4]>") instead of bare ids. Token: GN
  bool useGraphNames = false;

  /// If true, print values by name instead of prefix + id. Token: VN
  bool valueNames = false;

  // --- Intermediate node handling ---

  /// If true, intermediate nodes are inlined as nested expressions.
  /// If false, each node is printed on its own line with output ids.
  /// Token: II (set true), V (set false)
  bool inlineIntermediates = true;

  // --- Output ids ---

  /// If true, print output value ids before the expression:
  /// "(%v5, %v6) = target(...)". Token: OI
  bool showOutputIds = true;

  /// If true, compress consecutive output ids into ranges:
  /// "[%5 - %8]" instead of "%5, %6, %7, %8". Token: CR
  bool compressOutputRanges = false;

  // --- Type annotations ---

  /// If true, print type annotations like "(2Df)" before value references.
  /// Token: T
  bool showTypes = false;

  // --- Attributes ---

  /// If true, include node attributes in the output. Token: NA (set false)
  bool showAttributes = true;

  // --- Name formatting ---

  /// If true, strip "torch.ops.aten." prefix and trailing ".default",
  /// ".Tensor", ".Tensor_default" from node target names. Token: S
  bool shortNames = false;

  // --- Depth and length limits ---

  /// Max recursion depth for inlined expressions. 0 means no limit.
  /// When the limit is reached, a summary of the elided subtree is printed.
  /// Token: D<n>
  int32_t maxDepth = 0;

  /// Max number of arguments printed in an argument list before truncating.
  /// Remaining args are replaced with "... <n> more, <summary>".
  /// Token: L<n>
  int32_t maxLength = 0;

  // --- Output flags ---

  /// If true, append output descriptor flags (shapeondev, shape, reqh)
  /// after the expression. Token: OF
  bool showOutputFlags = false;

  // --- Boundaries ---

  /// Values to treat as leaves (print as references, don't recurse).
  const std::unordered_set<ValueCP>* boundaryValues = nullptr;

  /// Nodes to treat as leaves (their outputs are printed as references).
  const std::unordered_set<NodeCP>* boundaryNodes = nullptr;

  /// If set, only recurse into producers that are in this set.
  const std::unordered_set<NodeCP>* allowedNodes = nullptr;

  /// Values whose producing node is printed as a separate "(%vN) = expr(...)"
  /// line rather than inlined. Other intermediates are still inlined within
  /// those expressions. When set and inlineIntermediates is true, this creates
  /// the mixed mode used by KernelOperation::makeExprString.
  const std::unordered_set<ValueCP>* breakoutValues = nullptr;

  // --- Context for name/type lookups ---

  const nativert::Graph* graph = nullptr;
  const ValueTypes* valueTypes = nullptr;

  /// If set, map formal value IDs to actual value IDs when printing.
  const FormalToActual* formalToActual = nullptr;

  /// If set, operands that are a reusable input of this ProjectNode (a boundary
  /// input of only one expr, dead after here directly and via alias) are
  /// prefixed with '&' to flag that the value's buffer is mutable in place.
  const ProjectNode* projectNode = nullptr;
};

/// Renders nativert graph nodes as human-readable expression strings.
class NodePrinter {
 public:
  explicit NodePrinter(PrintOptions options = {});

  std::string print(NodeCP node) const;

  // --- Preset configurations ---

  // Inline expression: target(target2(%v1, %v2), %v3)
  static std::string expr(NodeCP node);

  // Per-line with all value ids:
  // (%v5) = mul(%v1, %v2)
  // (%v6) = add(%v5, %v3)
  static std::string values(NodeCP node);

  // Like values() but with type annotations.
  static std::string detailed(NodeCP node);

  // Debugger-friendly: prints a single node's call with output ids and
  // all attributes, no recursion.
  static std::string one(NodeCP node);

  static PrintOptions parsePrintOptions(const std::string& opts);

  static void setDefaults(const PrintOptions& opts);

  static PrintOptions& defaults();

  void printOutputIds(std::stringstream& ss, NodeCP node) const;

  bool isLeaf(ValueCP value) const;

 private:
  struct SubtreeSummary {
    std::unordered_map<std::string, int32_t> functionCounts;
    int32_t distinctLeaves = 0;
  };

  void printImpl(
      std::stringstream& ss,
      NodeCP node,
      std::unordered_set<NodeCP>& visited) const;

  void printExprImpl(std::stringstream& ss, NodeCP node, int32_t depth) const;

  void printValueId(std::stringstream& ss, ValueCP value) const;

  void printValueRef(std::stringstream& ss, ValueCP value) const;

  std::string formatTarget(std::string_view target) const;

  void collectSummary(
      NodeCP node,
      std::unordered_set<ValueCP>& seenLeaves,
      SubtreeSummary& summary) const;

  void collectSummaryForValue(
      ValueCP value,
      std::unordered_set<ValueCP>& seenLeaves,
      SubtreeSummary& summary) const;

  void printSummary(std::stringstream& ss, const SubtreeSummary& summary) const;

  PrintOptions options_;
};

/// RAII guard that parses a print-options string (e.g. "D3,L4,S") and installs
/// it as the thread-local override for NodePrinter::defaults(). Restores the
/// previous override on destruction.
class WithPrintOptions {
 public:
  explicit WithPrintOptions(const std::string& opts);
  /// Installs the given options directly (e.g. a copy of the current defaults
  /// with valueTypes/graph populated). Restores the previous override on exit.
  explicit WithPrintOptions(PrintOptions opts);
  ~WithPrintOptions();

  WithPrintOptions(const WithPrintOptions&) = delete;
  WithPrintOptions& operator=(const WithPrintOptions&) = delete;
  WithPrintOptions(WithPrintOptions&&) = delete;
  WithPrintOptions& operator=(WithPrintOptions&&) = delete;

 private:
  PrintOptions* previous_;
  PrintOptions options_;
};

} // namespace torch::wave
