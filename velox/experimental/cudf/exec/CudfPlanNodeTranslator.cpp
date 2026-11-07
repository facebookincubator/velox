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
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfPlanNodeTranslator.h"
#include "velox/experimental/cudf/exec/CudfPlanNodes.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

namespace facebook::velox::cudf_velox {

std::unique_ptr<exec::Operator> CudfPlanNodeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (!cudfIsRegistered()) {
    return nullptr;
  }
  if (auto fromVelox =
          std::dynamic_pointer_cast<const CudfFromVeloxNode>(node)) {
    VELOX_CHECK(
        isTypeSupportedByCudf(fromVelox->outputType()),
        "Unsupported cuDF conversion type: {}",
        fromVelox->outputType()->toString());
    return std::make_unique<CudfFromVelox>(
        id, fromVelox->outputType(), ctx, fromVelox->id());
  }
  if (auto toVelox = std::dynamic_pointer_cast<const CudfToVeloxNode>(node)) {
    return std::make_unique<CudfToVelox>(
        id, toVelox->outputType(), ctx, toVelox->id());
  }
  if (auto orderBy = std::dynamic_pointer_cast<const CudfOrderByNode>(node)) {
    VELOX_CHECK(
        isTypeSupportedByCudf(orderBy->outputType()),
        "Unsupported CudfOrderBy type: {}",
        orderBy->outputType()->toString());
    return std::make_unique<CudfOrderBy>(id, ctx, orderBy);
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
