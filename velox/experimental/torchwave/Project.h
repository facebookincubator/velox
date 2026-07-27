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

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

using NodeSet = std::unordered_set<NodeCP>;

/// Represents  set of independent computations. These are subgraphs of an FX
/// graph delimited by their root and a set of input nodes. The nodes reachable
/// from 'exprs_' and stopping at 'inputs_', excluding 'inputs_' constitute the
/// layer of the graph represented by 'this'.
class ProjectNode {
 public:
  ProjectNode(
      std::vector<NodeCP> nodes,
      std::unordered_set<NodeCP> inputs,
      ProjectNode* input,
      int32_t id)
      : nodes_{std::move(nodes)},
        inputs_{std::move(inputs)},
        input_{input},
        id_{id} {}

  int32_t id() const {
    return id_;
  }

  std::string toString(
      const nativert::Graph& graph,
      NodeSet& border,
      const ValueTypes* valueTypes = nullptr) const;

  const std::vector<NodeCP>& nodes() const {
    return nodes_;
  }

  const std::unordered_set<NodeCP>& inputs() const {
    return inputs_;
  }

  ProjectNode* input() const {
    return input_;
  }

  void distinctFunctions(
      std::unordered_map<std::string, int32_t>& counts) const;

  /// Values whose last access across all ProjectNodes happens in this node, so
  /// their buffers may be released after this node executes. Graph outputs are
  /// excluded (they escape the graph). Alias-corrected: a value kept alive by a
  /// view or by membership in a prim.ListPack is not listed until its last
  /// alias dies. Populated by ParallelNodes::computeLastUse.
  std::unordered_set<ValueCP> lastUse;

  /// For each top-level expr (parallel to nodes()), the lastUse values that are
  /// a boundary input of only that expr in this layer. A kernel op for the expr
  /// may reuse such a value's buffer in place, since nothing else reads it here
  /// or later (directly or via any alias). Populated by
  /// ParallelNodes::computeLastUse.
  std::vector<std::vector<ValueCP>> reusableValues_;

  /// True if 'value' is a reusable input of any expr in this node.
  bool isReusableInput(ValueCP value) const {
    for (const auto& perExpr : reusableValues_) {
      if (std::find(perExpr.begin(), perExpr.end(), value) != perExpr.end()) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<NodeCP> nodes_;
  std::unordered_set<NodeCP> inputs_;
  ProjectNode* input_;
  int32_t id_;
};

} // namespace torch::wave
