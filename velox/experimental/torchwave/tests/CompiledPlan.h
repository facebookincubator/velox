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
#include <initializer_list>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace torch::wave {

class WaveGraph;

/// Provides a static, mode-parameterized view of a compiled WaveGraph for
/// asserting how ops are distributed into kernels and steps -- i.e. where
/// kernel boundaries are and are not placed. Built from WaveGraph::nodes()
/// after compilation, so no run is needed. Ops are identified by node target,
/// normalized by stripping a leading "torch.ops." so "aten.add.Tensor" matches
/// "torch.ops.aten.add.Tensor"; synthetic kernel pieces keep their "tw." target
/// (e.g. "tw.masked_select_head"). Matchers state relationships (co-fusion,
/// boundary, barrier, sequencing) rather than absolute indices, so they survive
/// small graph changes. describe() output is used only in failure messages,
/// never for matching.
class CompiledPlan {
 public:
  /// Which grid variant to inspect.
  enum class Mode { kMultiKernel, kSingleBlock, kCG };

  /// Represents one Launch: a fused kernel (the set of co-fused op targets) or
  /// a standalone op.
  struct KernelView {
    /// True if this launch runs a standalone (eager) node rather than a fused
    /// kernel.
    bool standalone{false};

    /// Normalized targets of every node fused into this kernel, or the single
    /// target of a standalone launch.
    std::set<std::string> names;

    /// Number of intra-kernel ordering barriers in this kernel.
    int32_t numBarriers{0};

    /// Returns true if the normalized 'op' is one of this kernel's targets.
    bool has(std::string_view op) const;
  };

  /// Holds the parallel launches at one step.
  struct StepView {
    std::vector<KernelView> kernels;
  };

  /// Holds the step sequence of one CompiledNode. A kernel boundary sits
  /// between consecutive steps.
  struct NodeView {
    std::vector<StepView> steps;
  };

  /// Flattens the already-compiled 'graph' into a queryable plan for 'mode'.
  static CompiledPlan from(WaveGraph& graph, Mode mode);

  const std::vector<NodeView>& nodes() const {
    return nodes_;
  }

  /// Passes if some single kernel fuses all of 'ops' together. Extra ops in
  /// that kernel are allowed, so unrelated fusions do not break the assertion.
  ::testing::AssertionResult fuses(
      std::initializer_list<std::string_view> ops) const;

  /// Passes if 'op' runs as a standalone launch and is not fused into any
  /// kernel.
  ::testing::AssertionResult standalone(std::string_view op) const;

  /// Passes if 'lhs' and 'rhs' both appear but never in the same kernel, i.e. a
  /// kernel boundary separates every occurrence.
  ::testing::AssertionResult kernelBoundaryBetween(
      std::string_view lhs,
      std::string_view rhs) const;

  /// Passes if 'lhs' and 'rhs' are co-fused in a kernel that carries an
  /// intra-kernel barrier (ordered within one kernel, not split by a boundary).
  ::testing::AssertionResult barrierBetween(
      std::string_view lhs,
      std::string_view rhs) const;

  /// Passes if, in some node, 'later' appears in a strictly later step than
  /// 'earlier'.
  ::testing::AssertionResult inLaterStep(
      std::string_view later,
      std::string_view earlier) const;

  /// Returns a human-readable dump of the plan for failure messages.
  std::string describe() const;

 private:
  std::vector<NodeView> nodes_;
};

} // namespace torch::wave
