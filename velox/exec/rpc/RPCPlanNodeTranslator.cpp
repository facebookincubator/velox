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

#include "velox/exec/rpc/RPCPlanNodeTranslator.h"

#include <algorithm>

#include "velox/exec/rpc/RPCOperator.h"

namespace facebook::velox::exec::rpc {

std::unique_ptr<exec::Operator> RPCPlanNodeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto rpcNode = std::dynamic_pointer_cast<const core::RPCNode>(node)) {
    VLOG(1) << "[RPC_TRANSLATOR] Creating RPCOperator for node id="
            << node->id();
    return std::make_unique<RPCOperator>(id, ctx, rpcNode);
  }
  return nullptr;
}

std::optional<uint32_t> RPCPlanNodeTranslator::maxDrivers(
    const core::PlanNodePtr& node) {
  if (auto rpcNode = std::dynamic_pointer_cast<const core::RPCNode>(node)) {
    if (rpcNode->streamingMode() == rpc::RPCStreamingMode::kBatch) {
      // BATCH mode: Force single-driver execution. Multiple drivers would
      // race for batch results, with only one getting data and others
      // finishing empty.
      return 1;
    }

    // When all arguments are constants and no real data columns flow from the
    // source, the upstream is a synthetic single-row ValuesNode. The Java
    // AddLocalExchanges optimizer inserts a ROUND_ROBIN LocalExchange that
    // distributes this single row across N drivers, causing N-1 drivers to
    // finish empty and the result to be lost. Force single-driver execution.
    const auto& constantInputs = rpcNode->constantInputs();
    const auto& argumentColumns = rpcNode->argumentColumns();
    bool allConstant = !constantInputs.empty() &&
        std::all_of(
            constantInputs.begin(), constantInputs.end(), [](const auto& v) {
              return v != nullptr;
            });
    if (allConstant) {
      auto sourceType = rpcNode->source()->outputType();
      if (sourceType->size() <= argumentColumns.size()) {
        return 1;
      }
    }

    // PER_ROW mode: Allow parallel execution. Each driver claims individual
    // rows atomically via RPCState::tryClaimOrWait().
    return std::nullopt;
  }
  return std::nullopt;
}

void registerRPCPlanNodeTranslator() {
  exec::Operator::registerOperator(std::make_unique<RPCPlanNodeTranslator>());
}

} // namespace facebook::velox::exec::rpc
