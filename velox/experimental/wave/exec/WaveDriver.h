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
      std::unique_ptr<GpuArena> arena,
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

  GpuArena& arena() const {
    return *arena_;
  }

  std::string toString() const override;

 private:
  // True if all output from 'stream' is fetched.
  bool streamAtEnd(WaveStream& stream);

  // Makes a RowVector from the result buffers of the last stage of executables
  // in 'stream'.
  RowVectorPtr makeResult(WaveStream& stream, const OperandSet& outputIds);

  // Starts another WaveStream if the source operator indicates it has more data
  // and there is space in the arena.
  void startMore();

  // Enqueus a prefetch from device to host for the buffers of output vectors.
  void prefetchReturn(WaveStream& stream);

  std::unique_ptr<GpuArena> arena_;

  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};
  exec::BlockingReason blockingReason_;

  bool finished_{false};

  // Wave operators replacing 'cpuOperators_' on GPU path.
  std::vector<std::unique_ptr<WaveOperator>> waveOperators_;
  // The replaced Operators from the Driver. Can be used for a CPU fallback.
  std::vector<std::unique_ptr<exec::Operator>> cpuOperators_;
  // Dedupped Subfields. Handed over by CompileState.
  SubfieldMap subfields_;
  // Operands handed over by compilation.
  std::vector<std::unique_ptr<AbstractOperand>> operands_;

  // The set of currently pending kernel DAGs for this WaveDriver. If
  // the source operator can produce multiple consecutive batches
  // before the batch is executed to completion, multiple such batches
  // can be on device independently of each other. This is bounded by
  // device memory and the speed at which the source can produce new
  // batches.
  std::vector<std::unique_ptr<WaveStream>> streams_;
};

} // namespace facebook::velox::wave
