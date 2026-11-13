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
#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox::detail {
// Internal, immutable built-in lowering table. Application nodes continue to
// use CudfPlanNodeDescriptor and their own Operator::PlanNodeTranslator.
struct CudfNodeLowering {
  bool (*matches)(const core::PlanNodePtr&);
  bool (*canRun)(
      const core::PlanNodePtr&,
      core::QueryCtx*,
      memory::MemoryPool*,
      std::string*);
  core::PlanNodePtr (
      *lower)(const core::PlanNodePtr&, std::vector<core::PlanNodePtr>);
  bool batchFirstInput;
};
const CudfNodeLowering* findCudfNodeLowering(const core::PlanNodePtr& node);
// Physical-only entries need no core node or lowering rule.
std::unique_ptr<exec::Operator> createCudfOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node);
} // namespace facebook::velox::cudf_velox::detail
