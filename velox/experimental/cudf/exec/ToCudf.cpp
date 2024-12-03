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
#include "velox/exec/OrderBy.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <iostream>

namespace facebook::velox::cudf_velox {

static bool _cudfIsRegistered = false;

bool CompileState::compile() {
  if (cudfDebugEnabled()) {
    std::cout << "Calling cudfDriverAdapter" << std::endl;
  }

  auto operators = driver_.operators();
  auto& nodes = planNodes_;

  if (cudfDebugEnabled()) {
    std::cout << "Number of operators: " << operators.size() << std::endl;
    for (auto& op : operators) {
      std::cout << "  Operator: ID " << op->operatorId() << ": "
                << op->toString() << std::endl;
    }
    std::cout << "Number of plan nodes: " << nodes.size() << std::endl;
    for (auto& node : nodes) {
      std::cout << "  Plan node: ID " << node->id() << ": " << node->toString()
                << std::endl;
    }
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
      replace_op.push_back(std::make_unique<CudfFromVelox>(
          id, plan_node->outputType(), ctx, plan_node->id()));
      replace_op[0]->initialize();
      replace_op.push_back(
          std::make_unique<CudfHashJoinBuild>(id, ctx, plan_node));
      replace_op[1]->initialize();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    } else if (auto joinProbeOp = dynamic_cast<exec::HashProbe*>(oper)) {
      auto id = joinProbeOp->operatorId();
      auto plan_node = std::dynamic_pointer_cast<const core::HashJoinNode>(
          get_plan_node(joinProbeOp->planNodeId()));
      VELOX_CHECK(plan_node != nullptr);
      // Each cudf operator is wrapped by CudfFromVelox, and CudfToVelox
      // operators
      // CudfFromVelox -> CudfHashJoinProbe -> CudfToVelox
      replace_op.push_back(std::make_unique<CudfFromVelox>(
          id, plan_node->outputType(), ctx, plan_node->id()));
      replace_op[0]->initialize();
      replace_op.push_back(
          std::make_unique<CudfHashJoinProbe>(id, ctx, plan_node));
      replace_op[1]->initialize();
      replace_op.push_back(std::make_unique<CudfToVelox>(
          id, plan_node->outputType(), ctx, plan_node->id()));
      replace_op[2]->initialize();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    } else if (auto orderByOp = dynamic_cast<exec::OrderBy*>(oper)) {
      auto id = orderByOp->operatorId();
      auto plan_node = std::dynamic_pointer_cast<const core::OrderByNode>(
          get_plan_node(orderByOp->planNodeId()));
      VELOX_CHECK(plan_node != nullptr);
      replace_op.push_back(std::make_unique<CudfFromVelox>(
          id, plan_node->outputType(), ctx, plan_node->id()));
      replace_op[0]->initialize();
      replace_op.push_back(std::make_unique<CudfOrderBy>(id, ctx, plan_node));
      replace_op[1]->initialize();
      replace_op.push_back(std::make_unique<CudfToVelox>(
          id, plan_node->outputType(), ctx, plan_node->id()));
      replace_op[2]->initialize();

      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_, operatorIndex, operatorIndex + 1, std::move(replace_op));
      replacements_made = true;
    }
  }
  return replacements_made;
}

struct cudfDriverAdapter {
  std::shared_ptr<rmm::mr::device_memory_resource> mr_;
  std::shared_ptr<std::vector<std::shared_ptr<core::PlanNode const>>>
      planNodes_;

  cudfDriverAdapter(std::shared_ptr<rmm::mr::device_memory_resource> mr)
      : mr_(mr) {
    if (cudfDebugEnabled()) {
      std::cout << "cudfDriverAdapter constructor" << std::endl;
    }
    planNodes_ =
        std::make_shared<std::vector<std::shared_ptr<core::PlanNode const>>>();
  }

  ~cudfDriverAdapter() {
    if (cudfDebugEnabled()) {
      std::cout << "cudfDriverAdapter destructor" << std::endl;
      printf(
          "cached planNodes_ %p, %ld\n",
          planNodes_.get(),
          planNodes_.use_count());
    }
  }

  // Call operator needed by DriverAdapter
  bool operator()(const exec::DriverFactory& factory, exec::Driver& driver) {
    auto state = CompileState(factory, driver, *planNodes_);
    // Stored planNodes_ from inspect.
    if (cudfDebugEnabled()) {
      printf("driver.planNodes_=%p\n", planNodes_.get());
      for (auto planNode : *planNodes_) {
        std::cout << "PlanNode: " << (*planNode).toString() << std::endl;
      }
    }
    auto res = state.compile();
    return res;
  }

  // Iterate recursively and store them in the planNodes_.
  void storePlanNodes(const std::shared_ptr<const core::PlanNode>& planNode) {
    const auto& sources = planNode->sources();
    for (int32_t i = 0; i < sources.size(); ++i) {
      storePlanNodes(sources[i]);
    }
    planNodes_->push_back(planNode);
  }

  // Call operator needed by plan inspection
  void operator()(const core::PlanFragment& planFragment) {
    // signature: std::function<void(const core::PlanFragment&)> inspect;
    // call: adapter.inspect(planFragment);
    planNodes_->clear();
    if (cudfDebugEnabled()) {
      std::cout << "Inspecting PlanFragment" << std::endl;
    }
    if (planNodes_) {
      storePlanNodes(planFragment.planNode);
    }
  }
};

void registerCudf() {
  const char* env_cudf_disabled = std::getenv("VELOX_CUDF_DISABLED");
  if (env_cudf_disabled != nullptr && std::stoi(env_cudf_disabled)) {
    return;
  }

  CUDF_FUNC_RANGE();
  cudaFree(0); // to init context.

  if (cudfDebugEnabled()) {
    std::cout << "Registering CudfHashJoinBridgeTranslator" << std::endl;
  }
  exec::Operator::registerOperator(
      std::make_unique<CudfHashJoinBridgeTranslator>());
  if (cudfDebugEnabled()) {
    std::cout << "Registering cudfDriverAdapter" << std::endl;
  }

  const char* env_cudf_mr = std::getenv("VELOX_CUDF_MEMORY_RESOURCE");
  auto mr_mode = env_cudf_mr != nullptr ? env_cudf_mr : "async";
  if (cudfDebugEnabled()) {
    std::cout << "Setting cuDF memory resource to " << mr_mode << std::endl;
  }
  auto mr = cudf_velox::create_memory_resource(mr_mode);
  cudf::set_current_device_resource(mr.get());
  cudfDriverAdapter cda{mr};
  exec::DriverAdapter cudfAdapter{"cuDF", cda, cda};
  exec::DriverFactory::registerAdapter(cudfAdapter);
  _cudfIsRegistered = true;
}

void unregisterCudf() {
  if (cudfDebugEnabled()) {
    std::cout << "Unregistering cudfDriverAdapter" << std::endl;
  }
  exec::DriverFactory::adapters.clear();
  _cudfIsRegistered = false;
}

bool cudfIsRegistered() {
  return _cudfIsRegistered;
}

} // namespace facebook::velox::cudf_velox
