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

#include <string>

namespace facebook::velox::cudf_velox {

/// Rules that decide whether a plan node can run on the GPU. The cuDF operator
/// adapters and the cuDF plan rewriter both use them, so the two paths place
/// nodes the same way. Each function returns false and sets 'reason' when the
/// node has to stay on the CPU. The rules look only at the plan node. Checks
/// that exist because the driver adapter sees one pipeline at a time stay in
/// the operator adapters.

/// 'filter' and 'project' are the parts of one FilterProject operator: a
/// Filter, a Project, or a Filter fused under a Project. LocalPlanner fuses a
/// Filter followed by a Project into one operator, so the pair is decided
/// together.
bool canRunFilterProjectOnGpu(
    const core::FilterNode* filter,
    const core::ProjectNode* project,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason);

bool canRunAggregationOnGpu(
    const core::AggregationNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason);

bool canRunHashJoinOnGpu(
    const core::HashJoinNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason);

bool canRunNestedLoopJoinOnGpu(
    const core::NestedLoopJoinNode& node,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool,
    std::string* reason);

bool canRunTopNRowNumberOnGpu(
    const core::TopNRowNumberNode& node,
    std::string* reason);

bool canRunLocalPartitionOnGpu(
    const std::shared_ptr<const core::LocalPartitionNode>& node,
    std::string* reason);

} // namespace facebook::velox::cudf_velox
