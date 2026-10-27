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
  enum class ExecutionMode { kCpu, kGpu };

  struct Config {
    VectorRepresentation taskOutput = VectorRepresentation::kVelox;
    /// Required: capability decisions use the same context as execution.
    std::shared_ptr<core::QueryCtx> queryCtx;
  };

  static core::PlanNodePtr rewrite(
      const core::PlanNodePtr& root,
      const Config& config);
};

} // namespace facebook::velox::cudf_velox
