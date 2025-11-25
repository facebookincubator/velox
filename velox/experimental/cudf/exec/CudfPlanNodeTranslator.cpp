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

#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeTranslator.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"

namespace facebook::velox::cudf_velox {

std::unique_ptr<exec::Operator> CudfPlanNodeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto fromVelox =
          std::dynamic_pointer_cast<const CudfFromVeloxNode>(node)) {
    return std::make_unique<CudfFromVelox>(
        id, fromVelox->outputType(), ctx, fromVelox->id());
  }

  if (auto toVelox = std::dynamic_pointer_cast<const CudfToVeloxNode>(node)) {
    return std::make_unique<CudfToVelox>(
        id, toVelox->outputType(), ctx, toVelox->id());
  }

  if (auto gpuAgg =
          std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
    return std::make_unique<CudfHashAggregation>(
        id, ctx, gpuAgg->aggregationNode());
  }

  return nullptr;
}

std::optional<uint32_t> CudfPlanNodeTranslator::maxDrivers(
    const core::PlanNodePtr& node) {
  if (auto gpuAgg =
          std::dynamic_pointer_cast<const CudfAggregationNode>(node)) {
    return gpuAgg->preferredDriverCount();
  }

  if (auto gpuJoin = std::dynamic_pointer_cast<const CudfHashJoinNode>(node)) {
    return gpuJoin->preferredProbeDriverCount();
  }

  return std::nullopt;
}

} // namespace facebook::velox::cudf_velox
