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

/// Functions for splitting a graph into consecutive layers where each layer
/// consists of independent computations.
/// Elides clones whose users only read them, across the whole graph. Call
/// before partitioning: ParallelNodes::rewriteInPlace walks one ProjectNode
/// layer at a time and cannot see a source whose clones land in different
/// layers. Returns the number elided.
int64_t elideReadOnlyClones(nativert::Graph& graph, const ValueTypes& types);

/// Merges nodes that compute the same value from the same operands, to a
/// fixpoint: merging two nodes can make their consumers congruent in turn. Only
/// pure nodes participate -- nothing that writes, reads a buffer written
/// elsewhere in the graph, or escapes as a graph output. Gated per category by
/// WaveConfig::cseCompute and cseViews; a no-op when both are off.
/// Call before partitioning, after the rewrites that create the duplicates, and
/// before duplicateMetadataOps, whose duplicates this would otherwise undo.
/// Returns the number of nodes merged away.
int64_t commonSubexpressions(nativert::Graph& graph, const ValueTypes& types);

/// Rewrites ops that produce a TensorList into per-tensor form, by calling each
/// op's own Metadata::decompose rule. One traversal serves every op, so a new
/// decomposition is a rule on the op rather than another pass over the graph.
/// Call after commonSubexpressions -- decomposition multiplies the node count,
/// and CSE is cheaper on the bundled form -- and before partitioning, which is
/// what the per-tensor nodes exist to inform. Returns the number rewritten.
int64_t decomposeListOps(nativert::Graph& graph, WaveGraph& waveGraph);

/// Rematerializes each multiply-used metadata getter (sym_size / sym_numel) at
/// its use sites, so it stops being a shared value the partitioner has to
/// materialize as a top-level output. Only duplicates a getter whose operand is
/// already live at every use site, so no new border appears in its place.
/// Call before partitioning and after any pass that merges equal values, which
/// would undo it. Returns the number of nodes duplicated.
int64_t duplicateMetadataOps(
    nativert::Graph& graph,
    ValueTypes& types,
    WaveGraph& waveGraph);

class ParallelNodes {
 public:
  /// Divides 'graph' into consecutive layers where each layer's expressions
  /// are known to be independent. A ProjectNode represents each layer.
  ProjectNode* makeParallelNodes(const nativert::Graph& graph);

  /// After the layers and last-use sets are built, elides redundant clones
  /// within each top-level expr so an in-place writer (e.g. index_put_) writes
  /// its original buffer instead of a defensive copy, per the in-place rewrite
  /// checklist. Localized to a single top-level expr: no op moves between
  /// ProjectNodes. Runs only when WaveConfig::enableReuse is set. 'types'
  /// supplies per-value layout and ownership constraints; graph nodes are
  /// mutated in place. Re-runs the last-use analysis afterward since the
  /// rewrites (clone elision) change value lifetimes.
  void rewriteInPlace(nativert::Graph& graph, const ValueTypes& types);

 private:
  ProjectNode* makeParallelProject(
      ProjectNode* input,
      const NodeSet& topExprs,
      std::vector<NodeCP> orderedExprs = {});

  /// After the graph is split into layers, fills each ProjectNode's lastUse
  /// (alias-corrected) and reusableValues_ sets by scanning every layer's
  /// boundary-value accesses and view/list aliases in execution order.
  void computeLastUse(const nativert::Graph& graph);

  /// Fills each ProjectNode's reusableValues_ and overwritableTemps_ from the
  /// finalized lastUse sets: boundary last-use values whose storage has a
  /// single in-layer consumer (reusable inputs), and expr-local values that
  /// never escape their subgraph (overwritable temps). Split out of
  /// computeLastUse so the core lifetime computation stays separate from
  /// reuse-eligibility.
  void computeReuseEligibility(const std::unordered_set<ValueCP>& graphOutputs);

  std::vector<std::unique_ptr<ProjectNode>> projectNodes_;
  int32_t nextId_{0};
};

void printSet(NodeSet& set);

void printRefcount(std::unordered_map<NodeCP, int32_t>& refCount, int32_t min);

} // namespace torch::wave
