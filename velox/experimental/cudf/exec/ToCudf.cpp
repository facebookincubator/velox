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

#include <cuda.h>
#include <cudf/detail/nvtx/ranges.hpp>
#include "velox/exec/Driver.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include <iostream>

namespace facebook::velox::cudf_velox {

bool CompileState::compile() {
  std::cout << "Calling cudfDriverAdapter" << std::endl;
  auto operators = driver_.operators();
  auto& nodes = driverFactory_.planNodes;
  std::cout << "Number of operators: " << operators.size() << std::endl;
  for (auto& op : operators) {
    std::cout << "  Operator: ID " << op->operatorId() << ": " << op->toString()
              << std::endl;
  }
  std::cout << "Number of plan nodes: " << nodes.size() << std::endl;
  for (auto& node : nodes) {
    std::cout << "  Plan node: ID " << node->id() << ": " << node->toString()
              << std::endl;
  }

  // Make sure operator states are initialized.  We will need to inspect some of
  // them during the transformation.
  driver_.initializeOperators();

  bool replacements_made = false;
  auto ctx = driver_.driverCtx();
  // Replace HashBuild and HashProbe operators with CudfHashJoinBuild and
  // CudfHashJoinProbe operators.
  for (int32_t operatorIndex = 0; operatorIndex < operators.size();
       ++operatorIndex) {
    std::vector<std::unique_ptr<exec::Operator>> replace_op;

    exec::Operator* oper = operators[operatorIndex];
    VELOX_CHECK(oper);
    if (auto joinBuildOp =
            dynamic_cast<exec::HashBuild*>(oper)) {
      auto plan_node_id = joinBuildOp->planNodeId();
      auto id = joinBuildOp->operatorId();
      replace_op.push_back(std::make_unique<CudfHashJoinBuild>(id, ctx, plan_node_id));
      replace_op[0]->initialize();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    } else if (
        auto joinProbeOp =
            dynamic_cast<exec::HashProbe*>(oper)) {
      auto plan_node_id = joinProbeOp->planNodeId();
      auto id = joinProbeOp->operatorId();
      replace_op.push_back(std::make_unique<CudfHashJoinProbe>(id, ctx, plan_node_id));
      replace_op[0]->initialize();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    }
  }
  return replacements_made;
}

bool cudfDriverAdapter(
    const exec::DriverFactory& factory,
    exec::Driver& driver) {
  auto state = CompileState(factory, driver);
  return state.compile();
}

void registerCudf() {
  CUDF_FUNC_RANGE();
  cudaFree(0); // to init context.
  std::cout << "Registering CudfHashJoinBridgeTranslator" << std::endl;
  exec::Operator::registerOperator(
      std::make_unique<CudfHashJoinBridgeTranslator>());
  std::cout << "Registering cudfDriverAdapter" << std::endl;
  exec::DriverAdapter cudfAdapter{"cuDF", {}, cudfDriverAdapter};
  exec::DriverFactory::registerAdapter(cudfAdapter);
}
} // namespace facebook::velox::cudf_velox
