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
#include "velox/experimental/wave/exec/WaveOperator.h"

namespace facebook::velox::wave {

using SubfieldMap =
    folly::F14FastMap<std::string, std::unique_ptr<common::Subfield>>;

class WaveDriver : public exec::SourceOperator {
 public:
  WaveDriver(
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      core::PlanNodeId planNodeId,
      int32_t operatorId,
      std::vector<std::unique_ptr<WaveOperator>> waveOperators,
      SubfieldMap subfields,
      std::vector<std::unique_ptr<AbstractOperand>> operands);

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    if (blockingFuture_.valid()) {
      *future = std::move(blockingFuture_);
      return blockingReason_;
    }
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() {
    return finished_;
  }

  void setReplaced(std::vector<std::unique_ptr<exec::Operator>> original) {
    cpuOperators_ = std::move(original);
  }

  std::string toString() const override;

 private:
  ContinueFuture blockingFuture_;
  exec::BlockingReason blockingReason_;

  bool finished_{false};

  // Wave operators replacing 'cpuOperators_' on GPU path.
  std::vector<std::unique_ptr<WaveOperator>> waveOperators_;
  // The replaced Operators from the Driver. Can be used for a CPU fallback.
  std::vector<std::unique_ptr<exec::Operator>> cpuOperators_;
  // Dedupped Subfields. Handed over by CompileState.
  SubfieldMap subfields_;

  std::vector<std::unique_ptr<AbstractOperand>> operands_;
  bool canAddDynamicFilter_{false};
};

} // namespace facebook::velox::wave
