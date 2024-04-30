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
#include "velox/exec/Operator.h" // Compilation fails in Driver.h if Operator.h isn't included first!
#include "velox/exec/Driver.h"

#include <iostream>

namespace facebook::velox::cudf_velox {

bool CompileState::compile() {
  std::cout << "Calling cudfDriverAdapter" << std::endl;
  auto operators = driver_.operators();
  auto& nodes = driverFactory_.planNodes;
  std::cout << "Number of operators: " << operators.size() << std::endl;
  for (auto& op : operators) {
    std::cout << "  Operator: ID " << op->operatorId() << ": " << op->toString() << std::endl;
  }
  std::cout << "Number of plan nodes: " << nodes.size() << std::endl;
  for (auto& node : nodes) {
    std::cout << "  Plan node: ID " << node->id() << ": " << node->toString() << std::endl;
  }
  return false;

  /*
  int32_t first = 0;
  int32_t operatorIndex = 0;
  int32_t nodeIndex = 0;
  RowTypePtr outputType;
  // Make sure operator states are initialized.  We will need to inspect some of
  // them during the transformation.
  driver_.initializeOperators();
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

bool cudfDriverAdapter(
    const exec::DriverFactory& factory,
    exec::Driver& driver) {
  auto state = CompileState(factory, driver);
  return state.compile();
}

void registerCudf() {
  std::cout << "Registering cudfDriverAdapter" << std::endl;
  exec::DriverAdapter cudfAdapter{"cuDF", {}, cudfDriverAdapter};
  exec::DriverFactory::registerAdapter(cudfAdapter);
}
} // namespace facebook::velox::cudf_velox
