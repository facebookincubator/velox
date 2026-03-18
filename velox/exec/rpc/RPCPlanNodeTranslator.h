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

#include "velox/exec/Operator.h"

namespace facebook::velox::exec::rpc {

/// PlanNodeTranslator for RPCNode.
///
/// Creates RPCOperator from RPCNode plan nodes.
///
/// maxDrivers() returns 1 for BATCH mode to prevent multiple drivers from
/// competing for batch results. PER_ROW mode allows parallel execution
/// since each driver claims individual rows atomically.
class RPCPlanNodeTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;

  /// Returns 1 for BATCH mode (single-driver), nullopt for PER_ROW mode.
  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override;
};

/// Register the RPCPlanNodeTranslator with Velox.
void registerRPCPlanNodeTranslator();

} // namespace facebook::velox::exec::rpc
