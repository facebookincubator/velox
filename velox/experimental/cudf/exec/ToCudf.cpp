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

#include "velox/experimental/cudf/exec/ToCudf.h"
#include <cuda.h>
#include <cudf/detail/nvtx/ranges.hpp>
#include "velox/exec/Driver.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"

#include <iostream>

namespace facebook::velox::cudf_velox {

bool CompileState::compile() {
  std::cout << "Calling cudfDriverAdapter" << std::endl;
  auto operators = driver_.operators();
  auto& nodes = planNodes_;
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

  // Get plan node by id lookup.
  auto get_plan_node = [&](const core::PlanNodeId& id) {
    auto it =
        std::find_if(nodes.cbegin(), nodes.cend(), [&id](const auto& node) {
          return node->id() == id;
        });
    VELOX_CHECK(it != nodes.end());
    return *it;
  };
  // Replace HashBuild and HashProbe operators with CudfHashJoinBuild and
  // CudfHashJoinProbe operators.
  for (int32_t operatorIndex = 0; operatorIndex < operators.size();
       ++operatorIndex) {
    std::vector<std::unique_ptr<exec::Operator>> replace_op;

    exec::Operator* oper = operators[operatorIndex];
    VELOX_CHECK(oper);
    if (auto joinBuildOp = dynamic_cast<exec::HashBuild*>(oper)) {
      auto id = joinBuildOp->operatorId();
      auto plan_node = std::dynamic_pointer_cast<const core::HashJoinNode>(
          get_plan_node(joinBuildOp->planNodeId()));
      VELOX_CHECK(plan_node != nullptr);
      replace_op.push_back(
          std::make_unique<CudfHashJoinBuild>(id, ctx, plan_node));
      replace_op[0]->initialize();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    } else if (auto joinProbeOp = dynamic_cast<exec::HashProbe*>(oper)) {
      auto id = joinProbeOp->operatorId();
      auto plan_node = std::dynamic_pointer_cast<const core::HashJoinNode>(
          get_plan_node(joinProbeOp->planNodeId()));
      VELOX_CHECK(plan_node != nullptr);
      replace_op.push_back(
          std::make_unique<CudfHashJoinProbe>(id, ctx, plan_node));
      replace_op[0]->initialize();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    }
  }
  return replacements_made;
}

struct cudfDriverAdapter {
  std::shared_ptr<std::vector<std::shared_ptr<core::PlanNode const>>> planNodes;
  cudfDriverAdapter() {
    std::cout << "cudfDriverAdapter constructor" << std::endl;
    planNodes =
        std::make_shared<std::vector<std::shared_ptr<core::PlanNode const>>>();
  }
  ~cudfDriverAdapter() {
    std::cout << "cudfDriverAdapter destructor" << std::endl;
    printf(
        "cached planNodes %p, %ld\n", planNodes.get(), planNodes.use_count());
  }
  // driveradapter
  bool operator()(const exec::DriverFactory& factory, exec::Driver& driver) {
    auto state = CompileState(factory, driver, *planNodes);
    // Stored planNodes from inspect.
    printf("driver.planNodes=%p\n", planNodes.get());
    for (auto planNode : *planNodes) {
      std::cout << "PlanNode: " << (*planNode).toString() << std::endl;
    }
    auto res = state.compile();
    return res;
  }
  // Iterate recursively and store them in the planNodes_ptr.
  void storePlanNodes(const std::shared_ptr<const core::PlanNode>& planNode) {
    const auto& sources = planNode->sources();
    for (int32_t i = 0; i < sources.size(); ++i) {
      storePlanNodes(sources[i]);
    }
    planNodes->push_back(planNode);
  }

  // inspect
  void operator()(const core::PlanFragment& planFragment) {
    // signature: std::function<void(const core::PlanFragment&)> inspect;
    // call: adapter.inspect(planFragment);
    planNodes->clear();
    std::cout << "Inspecting PlanFragment: " << std::endl;
    if (planNodes) {
      printf("inspect.planNodes=%p\n", planNodes.get());
      storePlanNodes(planFragment.planNode);
    } else {
      std::cout << "planNodes_ptr is nullptr" << std::endl;
    }
  }
};

void registerCudf() {
  CUDF_FUNC_RANGE();
  cudaFree(0); // to init context.
  std::cout << "Registering CudfHashJoinBridgeTranslator" << std::endl;
  exec::Operator::registerOperator(
      std::make_unique<CudfHashJoinBridgeTranslator>());
  std::cout << "Registering cudfDriverAdapter" << std::endl;
  cudfDriverAdapter cda{};
  exec::DriverAdapter cudfAdapter{"cuDF", cda, cda};
  exec::DriverFactory::registerAdapter(cudfAdapter);
}

void unregisterCudf() {
  std::cout << "unRegistering cudfDriverAdapter" << std::endl;
  exec::DriverFactory::adapters.clear();
}
} // namespace facebook::velox::cudf_velox
