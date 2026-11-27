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

#include "velox/experimental/cudf/exec/CudfProjectionOperator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

/// GPU implementation of the Expand operator. Takes a single input batch and
/// produces one output batch per projection list. Each output column is either
/// an input column or a constant.
class CudfExpand : public CudfProjectionOperator {
 public:
  CudfExpand(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::ExpandNode>& expandNode);

  /// Returns true if every constant in the projections can be built as a cuDF
  /// scalar. Builds each one to find out. On failure, 'reason' is populated
  /// with an explanation when non-null.
  static bool canRunOnGPU(
      const core::ExpandNode& expandNode,
      std::string* reason);
};

} // namespace facebook::velox::cudf_velox
