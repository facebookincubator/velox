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

/// Plan-node-level cuDF eligibility checks.
///
/// Each function decides whether a single plan node can run on GPU, without
/// building operators. The Velox operator adapters call these at runtime, and
/// the Presto coordinator calls them during pre-execution plan validation so a
/// query can fall back to CPU before it is dispatched to workers.
///
/// Functions whose eligibility depends on expressions accept an optional
/// 'queryCtx' and 'pool'. When both are provided the expression is
/// constant-folded and rewritten first, matching the form the operator
/// compiles at runtime; passing nullptr for both checks the plan-time
/// expression as-is, which is what the coordinator does when it has no
/// execution context. The nullptr path is therefore more conservative: it may
/// reject an expression that optimization would have made GPU-eligible.
namespace facebook::velox::cudf_velox {

/// Result of a plan-node cuDF eligibility check: whether the node can run on
/// GPU and, when it cannot, a human-readable reason for the fallback.
struct CudfNodeSupport {
  /// True if the node can be executed on GPU by cuDF.
  bool supported{false};

  /// Populated only when 'supported' is false: the specific reason the node
  /// cannot run on GPU, for a fallback log or a coordinator validation message.
  std::string reason;
};

/// Returns whether 'tableScanNode' reads through a cuDF-backed connector
/// (CudfHiveConnector or CudfIcebergConnector), with a reason when it does not.
CudfNodeSupport isTableScanNodeSupported(
    const core::TableScanNode* tableScanNode);

/// Returns whether 'filterNode's predicate can be evaluated on GPU, with a
/// reason when it cannot.
CudfNodeSupport isFilterNodeSupported(
    const core::FilterNode* filterNode,
    core::QueryCtx* queryCtx = nullptr,
    memory::MemoryPool* pool = nullptr);

/// Returns whether every projection in 'projectNode' can be evaluated on GPU,
/// with a reason when it cannot. A node whose single source produces no columns
/// but still carries projections is rejected, because cuDF cannot represent it.
CudfNodeSupport isProjectNodeSupported(
    const core::ProjectNode* projectNode,
    core::QueryCtx* queryCtx = nullptr,
    memory::MemoryPool* pool = nullptr);

/// Returns whether 'joinNode' has a cuDF-supported join type and, when present,
/// a GPU-evaluable filter, with a reason when it does not. Null-aware anti
/// joins that carry a filter are rejected until that combination is
/// implemented.
CudfNodeSupport isHashJoinNodeSupported(
    const core::HashJoinNode* joinNode,
    core::QueryCtx* queryCtx = nullptr,
    memory::MemoryPool* pool = nullptr);

/// Returns whether 'joinNode' has a cuDF-supported nested-loop join type and,
/// when present, a GPU-evaluable join condition, with a reason when it does
/// not.
CudfNodeSupport isNestedLoopJoinNodeSupported(
    const core::NestedLoopJoinNode* joinNode,
    core::QueryCtx* queryCtx = nullptr,
    memory::MemoryPool* pool = nullptr);

/// Returns whether 'aggregationNode' (groupby, global reduction, or distinct)
/// can be evaluated on GPU, with a reason when it cannot: supported functions
/// and steps, no unsupported mask or distinct aggregate, and GPU-evaluable
/// grouping keys and inputs.
CudfNodeSupport isAggregationNodeSupported(
    const core::AggregationNode* aggregationNode,
    core::QueryCtx* queryCtx = nullptr,
    memory::MemoryPool* pool = nullptr);

/// Returns whether every window function and frame in 'windowNode' is supported
/// by cuDF, with a reason when one is not.
CudfNodeSupport isWindowNodeSupported(const core::WindowNode* windowNode);

/// Returns whether 'topNRowNumberNode' uses the row_number ranking function,
/// the only ranking cuDF supports, with a reason when it does not.
CudfNodeSupport isTopNRowNumberNodeSupported(
    const core::TopNRowNumberNode* topNRowNumberNode);

/// Returns whether 'localPartitionNode' uses a partitioning scheme that
/// CudfLocalPartition can replace (hash, gather, or round-robin variants), with
/// a reason when it does not.
CudfNodeSupport isLocalPartitionNodeSupported(
    const core::LocalPartitionNode* localPartitionNode);

} // namespace facebook::velox::cudf_velox
