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

#include "velox/experimental/torchwave/tests/CompiledPlan.h"

#include <algorithm>
#include <limits>

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {
namespace {

std::string normalizeTarget(std::string_view target) {
  constexpr std::string_view kPrefix = "torch.ops.";
  if (target.substr(0, kPrefix.size()) == kPrefix) {
    target.remove_prefix(kPrefix.size());
  }
  return std::string(target);
}

LaunchGrid& selectGrid(ProjectOperation* op, CompiledPlan::Mode mode) {
  switch (mode) {
    case CompiledPlan::Mode::kMultiKernel:
      return op->grid();
    case CompiledPlan::Mode::kSingleBlock:
      return op->singleBlockGrid();
    case CompiledPlan::Mode::kCG:
      return op->cgGrid();
  }
  return op->grid();
}

CompiledPlan::KernelView makeKernelView(const Launch& launch) {
  CompiledPlan::KernelView view;
  if (launch.op != nullptr) {
    for (const auto* node : launch.op->allNodes()) {
      view.names.insert(normalizeTarget(node->target()));
    }
    view.numBarriers =
        static_cast<int32_t>(launch.op->barrierCounters().size());
  } else if (launch.standalone != nullptr) {
    view.standalone = true;
    view.names.insert(normalizeTarget(launch.standalone->target()));
  }
  return view;
}

template <typename Range>
std::string join(const Range& range) {
  std::string out;
  for (const auto& item : range) {
    if (!out.empty()) {
      out += ", ";
    }
    out += std::string(item);
  }
  return out;
}

} // namespace

bool CompiledPlan::KernelView::has(std::string_view op) const {
  return names.count(normalizeTarget(op)) > 0;
}

CompiledPlan CompiledPlan::from(WaveGraph& graph, Mode mode) {
  CompiledPlan plan;
  for (const auto& node : graph.nodes()) {
    NodeView nodeView;
    const auto* composite = node->kernels();
    if (composite) {
      // A CompositeInvocation launches step i of every ProjectOperation
      // together before step i+1, so step i across all ops is one real
      // (synchronized) step: merge them by step index.
      for (const auto& op : composite->ops()) {
        auto& grid = selectGrid(op.projectOp(), mode);
        if (grid.size() > nodeView.steps.size()) {
          nodeView.steps.resize(grid.size());
        }
        for (size_t step = 0; step < grid.size(); ++step) {
          for (const auto& launch : grid[step]) {
            nodeView.steps[step].kernels.push_back(makeKernelView(launch));
          }
        }
      }
    }
    plan.nodes_.push_back(std::move(nodeView));
  }
  return plan;
}

::testing::AssertionResult CompiledPlan::fuses(
    std::initializer_list<std::string_view> ops) const {
  for (const auto& node : nodes_) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (std::all_of(ops.begin(), ops.end(), [&](std::string_view op) {
              return kernel.has(op);
            })) {
          return ::testing::AssertionSuccess();
        }
      }
    }
  }
  return ::testing::AssertionFailure()
      << "no single kernel fuses {" << join(ops) << "}\n"
      << describe();
}

::testing::AssertionResult CompiledPlan::standalone(std::string_view op) const {
  bool asStandalone = false;
  bool asFused = false;
  for (const auto& node : nodes_) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (kernel.has(op)) {
          (kernel.standalone ? asStandalone : asFused) = true;
        }
      }
    }
  }
  if (asStandalone && !asFused) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
      << op
      << (asStandalone ? " is also fused into a kernel"
                       : " is not run as a standalone")
      << "\n"
      << describe();
}

::testing::AssertionResult CompiledPlan::kernelBoundaryBetween(
    std::string_view lhs,
    std::string_view rhs) const {
  bool lhsFound = false, rhsFound = false, coFused = false;
  for (const auto& node : nodes_) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        bool hasLhs = kernel.has(lhs), hasRhs = kernel.has(rhs);
        lhsFound |= hasLhs;
        rhsFound |= hasRhs;
        coFused |= (hasLhs && hasRhs);
      }
    }
  }
  if (lhsFound && rhsFound && !coFused) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
      << (!lhsFound || !rhsFound ? "one of the ops is absent"
                                 : "the ops are co-fused (no boundary)")
      << ": " << lhs << ", " << rhs << "\n"
      << describe();
}

::testing::AssertionResult CompiledPlan::barrierBetween(
    std::string_view lhs,
    std::string_view rhs) const {
  bool coFused = false;
  for (const auto& node : nodes_) {
    for (const auto& step : node.steps) {
      for (const auto& kernel : step.kernels) {
        if (kernel.has(lhs) && kernel.has(rhs)) {
          coFused = true;
          if (kernel.numBarriers > 0) {
            return ::testing::AssertionSuccess();
          }
        }
      }
    }
  }
  return ::testing::AssertionFailure()
      << (coFused ? "co-fused but that kernel has no barrier"
                  : "the ops are not co-fused")
      << ": " << lhs << ", " << rhs << "\n"
      << describe();
}

::testing::AssertionResult CompiledPlan::inLaterStep(
    std::string_view later,
    std::string_view earlier) const {
  for (const auto& node : nodes_) {
    int32_t maxLater = -1;
    int32_t minEarlier = std::numeric_limits<int32_t>::max();
    for (size_t step = 0; step < node.steps.size(); ++step) {
      for (const auto& kernel : node.steps[step].kernels) {
        if (kernel.has(later)) {
          maxLater = std::max(maxLater, static_cast<int32_t>(step));
        }
        if (kernel.has(earlier)) {
          minEarlier = std::min(minEarlier, static_cast<int32_t>(step));
        }
      }
    }
    if (maxLater >= 0 && minEarlier != std::numeric_limits<int32_t>::max() &&
        maxLater > minEarlier) {
      return ::testing::AssertionSuccess();
    }
  }
  return ::testing::AssertionFailure()
      << later << " is not in a later step than " << earlier << "\n"
      << describe();
}

std::string CompiledPlan::describe() const {
  std::string out = "compiled plan:\n";
  for (size_t n = 0; n < nodes_.size(); ++n) {
    if (nodes_[n].steps.empty()) {
      continue;
    }
    out += "  node " + std::to_string(n) + ":\n";
    for (size_t step = 0; step < nodes_[n].steps.size(); ++step) {
      for (const auto& kernel : nodes_[n].steps[step].kernels) {
        out += "    step " + std::to_string(step) + ": ";
        if (kernel.standalone) {
          out += "standalone ";
        }
        out += "{" + join(kernel.names) + "}";
        if (kernel.numBarriers > 0) {
          out += " +barrier";
        }
        out += "\n";
      }
    }
  }
  return out;
}

} // namespace torch::wave
