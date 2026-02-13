/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/core/PlanNode.h"

namespace facebook::velox::cudf_velox {

/// The user defined operator will inherit this operator, the operator accepts
/// CudfOperator and output CudfVector.
class CudfOperator : public NvtxHelper {
 public:
  CudfOperator(
      int32_t operatorId,
      const core::PlanNodeId& nodeId,
      nvtx3::color color = nvtx3::rgb{160, 82, 45} /* Sienna */)
      : facebook::velox::cudf_velox::NvtxHelper(
            color,
            operatorId,
            fmt::format("[{}]", nodeId)) {}
};
} // namespace facebook::velox::cudf_velox
