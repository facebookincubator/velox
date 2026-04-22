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
#include "velox/vector/ComplexVector.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox {

/// Replaces TableScanNodes in a TPC-H plan with ValuesNode + ProjectNode.
/// Expects plans built with filtersAsNode=true (filters already as FilterNodes).
/// New node IDs are generated starting above the max existing ID in the plan.
///
/// @param plan The original plan containing TableScanNodes
/// @param preloadedTables Map from TPC-H table name (e.g. "lineitem", "orders")
///        to preloaded RowVector batches (CudfVectors for GPU path)
/// @return A new plan with TableScanNodes replaced by ValuesNode feeding
///         ProjectNode (for column selection)
core::PlanNodePtr replaceTableScansWithValues(
    const core::PlanNodePtr& plan,
    const std::unordered_map<std::string, std::vector<RowVectorPtr>>&
        preloadedTables);

} // namespace facebook::velox
