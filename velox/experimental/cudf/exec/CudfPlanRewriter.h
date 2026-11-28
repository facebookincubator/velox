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

#include "velox/experimental/cudf/exec/CudfPlanNode.h"

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::cudf_velox {

/// Rewrites a plan of core plan nodes into one that places each node on the
/// CPU or the GPU. Nodes that run on the GPU become cuDF plan nodes, and
/// CudfFromVeloxNode and CudfToVeloxNode convert vectors where the two meet.
class CudfPlanRewriter {
 public:
  /// Where a plan node runs.
  enum class ExecutionMode { kCpu, kGpu };

  /// The settings of one rewrite.
  struct Config {
    /// The kind of vectors the root of the plan has to produce.
    VectorRepresentation taskOutput = VectorRepresentation::kVelox;
    /// Whether a plan node that cuDF cannot run stays on the CPU. When false,
    /// the rewrite fails with a user error that names the node. The caller
    /// sets this; the rewriter does not read 'cudf.allow_cpu_fallback'.
    bool allowCpuFallback = true;
    /// Required: capability decisions use the same context as execution.
    std::shared_ptr<core::QueryCtx> queryCtx;
  };

  /// Returns the rewritten plan of 'root', which passes validateCudfPlan().
  /// Throws a user error if two nodes share an ID, or if cuDF cannot run a
  /// node and 'config.allowCpuFallback' is false.
  static core::PlanNodePtr rewrite(
      const core::PlanNodePtr& root,
      const Config& config);
};

} // namespace facebook::velox::cudf_velox
