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

#include "velox/experimental/wave/exec/WaveDriver.h"
#include "velox/experimental/wave/exec/Instruction.h"
#include "velox/experimental/wave/exec/WaveOperator.h"

namespace facebook::velox::wave {

WaveDriver::WaveDriver(
    exec::DriverCtx* driverCtx,
    RowTypePtr outputType,
    core::PlanNodeId planNodeId,
    int32_t operatorId,
    std::vector<std::unique_ptr<WaveOperator>> waveOperators,
    SubfieldMap subfields,
    std::vector<std::unique_ptr<AbstractOperand>> operands)
    : exec::SourceOperator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "Wave"),
      waveOperators_(std::move(waveOperators)),
      subfields_(std::move(subfields)),
      operands_(std::move(operands)) {}

RowVectorPtr WaveDriver::getOutput() {
  return nullptr;
}

std::string WaveDriver::toString() const {
  std::stringstream out;
  out << "{Wave" << std::endl;
  for (auto& op : waveOperators_) {
    out << op->toString() << std::endl;
  }
  return out.str();
}

} // namespace facebook::velox::wave
