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

#include "velox/experimental/cudf/tocudf.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h" // Compilation fails in Driver.h if Operator.h isn't included first!
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"

#include "CustomJoin.h"

#include <iostream>

namespace facebook::velox::cudf {

bool CompileState::compile() {
  std::cout << "Calling CustomDriverAdapter" << std::endl;
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
  // return false;

  int32_t first = 0;
  int32_t operatorIndex = 0;
  int32_t nodeIndex = 0;
  RowTypePtr outputType;
  // Make sure operator states are initialized.  We will need to inspect some of
  // them during the transformation.
  driver_.initializeOperators();
  // Replace HashBuild and HashProbe operators with CustomHashBuild and CustomHashProbe operators.
  // is that enough?
  std::cout<<"Operator replacement:\n";
  for (; operatorIndex < operators.size(); ++operatorIndex) {
      std::vector<std::unique_ptr<exec::Operator>> replace_op;
      std::vector<int32_t> replace_id;

      facebook::velox::exec::Operator* oper = operators[operatorIndex];
      std::cout<<"operator["<<operatorIndex<<"]:";
      std::cout << "  Operator: ID " << oper->operatorId() << ": " << oper->toString() << std::endl;
      if (auto joinBuildOp = dynamic_cast<facebook::velox::exec::HashBuild*>(oper)) {
        std::cout<<"replaced HashBuild\n";
        auto planid = joinBuildOp->planNodeId();
        // auto planNode = joinBuildOp->planNode();
        auto id =  joinBuildOp->operatorId(); //TODO should we reuse operator id?
        auto ctx = driver_.driverCtx();
        replace_id.push_back(operatorIndex);
        replace_op.push_back(std::make_unique<CustomJoinBuild>(id, ctx, planid));
        replace_op[0]->initialize();
            auto replaced = driverFactory_.replaceOperators(
      driver_, replace_id[0], replace_id[0]+1, std::move(replace_op));
      } else if (auto joinProbeOp = dynamic_cast<facebook::velox::exec::HashProbe*>(oper)) {
        std::cout<<"replaced HashProbe\n";
        auto planid = joinProbeOp->planNodeId();
        // auto planNode = joinProbeOp->planNode();
        auto id =  joinProbeOp->operatorId(); //TODO should we reuse operator id?
        auto ctx = driver_.driverCtx();
        replace_id.push_back(operatorIndex);
        replace_op.push_back(std::make_unique<CustomJoinProbe>(id, ctx, planid));
        replace_op[0]->initialize();
            auto replaced = driverFactory_.replaceOperators(
      driver_, replace_id[0], replace_id[0]+1, std::move(replace_op));
      }
      // if (not replace_id.empty()) {
      //     auto replaced = driverFactory_.replaceOperators(
      // driver_, replace_id[0], replace_id[0]+1, std::move(replace_op));
          // waveOp->setReplaced(std::move(replaced));
      // }
  }
  // for (auto& op : operators_) {
  //   op->finalize(*this);
  // }
  return true;
  /*
  for (; operatorIndex < operators.size(); ++operatorIndex) {
    if (!addOperator(operators[operatorIndex], nodeIndex, outputType)) {
      break;
    }
    ++nodeIndex;
    auto& identity = operators[operatorIndex]->identityProjections();
    for (auto i = 0; i < outputType->size(); ++i) {
      Value value = Value(toSubfield(outputType->nameOf(i)));
      if (isProjectedThrough(identity, i)) {
        continue;
      }
      auto operand = operators_.back()->defines(value);
      definedBy_[value] = operand;
    }
  }
  if (operators_.empty()) {
    return false;
  }
  for (auto& op : operators_) {
    op->finalize(*this);
  }
  std::vector<OperandId> resultOrder;
  for (auto i = 0; i < outputType->size(); ++i) {
    auto operand = findCurrentValue(Value(toSubfield(outputType->nameOf(i))));
    resultOrder.push_back(operand->id);
  }
  auto waveOpUnique = std::make_unique<WaveDriver>(
      driver_.driverCtx(),
      outputType,
      operators[first]->planNodeId(),
      operators[first]->operatorId(),
      std::move(arena_),
      std::move(operators_),
      std::move(resultOrder),
      std::move(subfields_),
      std::move(operands_));
  auto waveOp = waveOpUnique.get();
  waveOp->initialize();
  std::vector<std::unique_ptr<exec::Operator>> added;
  added.push_back(std::move(waveOpUnique));
  auto replaced = driverFactory_.replaceOperators(
      driver_, first, operatorIndex, std::move(added));
  waveOp->setReplaced(std::move(replaced));
  return true;
  */
}

bool CustomDriverAdapter(
    const exec::DriverFactory& factory,
    exec::Driver& driver) {
  auto state = CompileState(factory, driver);
  return state.compile();
}

void registerCustomDriver() {
  std::cout << "Registering CustomJoinBridgeTranslator" << std::endl;
  exec::Operator::registerOperator( std::make_unique<CustomJoinBridgeTranslator>());
  std::cout << "Registering CustomDriverAdapter" << std::endl;
  exec::DriverAdapter custAdapter{"cuDF", {}, CustomDriverAdapter};
  exec::DriverFactory::registerAdapter(custAdapter);
}
} // namespace facebook::velox::cudf

