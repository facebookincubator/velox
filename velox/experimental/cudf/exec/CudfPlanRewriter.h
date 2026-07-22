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

#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::cudf_velox {

class CudfPlanRewriter {
 public:
  enum class ExecutionMode { kCpu, kGpu };

  struct Config {
    int gpuDriverCount = 4;
    int cpuDriverCount = 32;
    bool addLocalPartitionBoundaries = true;
    std::shared_ptr<core::QueryCtx> queryCtx;
  };

  static core::PlanNodePtr rewrite(
      const core::PlanNodePtr& root,
      const Config& config);

  /// Transitional entry points for the legacy DriverAdapter path. These keep
  /// CPU-to-cuDF PlanNode conversion centralized in the plan rewriter while
  /// that path is still available.
  static core::PlanNodePtr translateForAdapter(
      const core::PlanNodePtr& node,
      int gpuDriverCount = 4);

  template <typename CudfNode>
  static std::shared_ptr<const CudfNode> translateForAdapterAs(
      const core::PlanNodePtr& node,
      int gpuDriverCount = 4) {
    auto translated = std::dynamic_pointer_cast<const CudfNode>(
        translateForAdapter(node, gpuDriverCount));
    VELOX_CHECK_NOT_NULL(translated);
    return translated;
  }

  static core::PlanNodePtr translateBatchConcatForAdapter(
      const core::PlanNodePtr& node,
      int gpuDriverCount = 4);
};

} // namespace facebook::velox::cudf_velox
