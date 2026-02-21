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
/*
 * Helper to check whether plan nodes can be evaluated by CUDF.
 */
#pragma once

#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::cudf_velox {

// TableScanNode validation: check if the node uses a supported connector
bool isTableScanNodeSupported(const core::TableScanNode* tableScanNode);

// FilterNode validation: check if filter expressions can be evaluated by CUDF
bool isFilterNodeSupported(const core::FilterNode* filterNode);

// ProjectNode validation: check if project expressions can be evaluated by CUDF
bool isProjectNodeSupported(const core::ProjectNode* projectNode);

// Step-aware aggregation validation function
bool canAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step);

// AggregationNode validation: check if aggregation can be evaluated by CUDF
bool isAggregationNodeSupported(const core::AggregationNode* aggregationNode);

// HashJoinNode validation: check if join can be evaluated by CUDF
bool isHashJoinNodeSupported(const core::HashJoinNode* joinNode);

} // namespace facebook::velox::cudf_velox
