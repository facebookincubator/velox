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

namespace facebook::velox::cudf_velox {

class CudfPlanRewriter {
 public:
  enum class ExecutionMode { kCpu, kGpu };

  struct Config {
    int gpuDriverCount = 4;
    int cpuDriverCount = 32;
  };

  static core::PlanNodePtr rewrite(
      const core::PlanNodePtr& root,
      const Config& config);

 private:
  static bool canUseGpuAggregation(
      const std::shared_ptr<const core::AggregationNode>& aggNode);

  static core::PlanNodePtr convertToGpuAggregation(
      const std::shared_ptr<const core::AggregationNode>& aggNode,
      const core::PlanNodePtr& newSource,
      int preferredDriverCount);

  static ExecutionMode determineExecutionMode(const core::PlanNodePtr& node);

  static core::PlanNodePtr insertBoundary(
      const core::PlanNodePtr& source,
      ExecutionMode fromMode,
      ExecutionMode toMode,
      int targetPartitions);

  static core::PlanNodePtr rewriteNode(
      const core::PlanNodePtr& node,
      ExecutionMode parentMode,
      const Config& config);
};

} // namespace facebook::velox::cudf_velox
