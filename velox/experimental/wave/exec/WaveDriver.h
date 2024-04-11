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

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/wave/exec/WaveOperator.h"

namespace facebook::velox::wave {

class WaveDriver : public exec::SourceOperator {
 public:
  WaveDriver(
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      core::PlanNodeId planNodeId,
      int32_t operatorId,
      std::unique_ptr<GpuArena> arena,
      std::vector<std::unique_ptr<WaveOperator>> waveOperators,
      std::vector<OperandId> resultOrder_,
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

  bool isFinished() override {
    return finished_;
  }

  void setReplaced(std::vector<std::unique_ptr<exec::Operator>> original) {
    cpuOperators_ = std::move(original);
  }

  GpuArena& arena() const {
    return *arena_;
  }

  const std::vector<std::unique_ptr<AbstractOperand>>& operands() {
    return operands_;
  }

  const SubfieldMap* subfields() {
    return &subfields_;
  }

  /// Returns the control block with thread block level sizes and statuses for
  /// input of  operator with id 'operator'. This is the control for the source
  /// or previous cardinality change.
  LaunchControl* inputControl(WaveStream& stream, int32_t operatorId);

  std::string toString() const override;

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override {
    pipelines_[0].operators[0]->addDynamicFilter(outputChannel, filter);
  }

  exec::OperatorCtx* operatorCtx() const {
    return operatorCtx_.get();
  }

 private:
  // True if all output from 'stream' is fetched.
  bool streamAtEnd(WaveStream& stream);

  // Makes a RowVector from the result buffers of the last stage of executables
  // in 'stream'.
  RowVectorPtr makeResult(WaveStream& stream, const OperandSet& outputIds);

  WaveVectorPtr makeWaveResult(
      const TypePtr& rowType,
      WaveStream& stream,
      const OperandSet& lastSet);

  // Starts another WaveStream if the source operator indicates it has more data
  // and there is space in the arena.
  void startMore();

  // Enqueus a prefetch from device to host for the buffers of output vectors.
  void prefetchReturn(WaveStream& stream);

  std::unique_ptr<GpuArena> arena_;
  std::unique_ptr<GpuArena> deviceArena_;
  std::unique_ptr<GpuArena> hostArena_;

  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};
  exec::BlockingReason blockingReason_;

  bool finished_{false};

  struct Pipeline {
    // Wave operators replacing 'cpuOperators_' on GPU path.
    std::vector<std::unique_ptr<WaveOperator>> operators;

    // The set of currently pending kernel DAGs for this Pipeline.  If the
    // source operator can produce multiple consecutive batches before the batch
    // is executed to completion, multiple such batches can be on device
    // independently of each other.  This is bounded by device memory and the
    // speed at which the source can produce new batches.
    std::list<std::unique_ptr<WaveStream>> streams;
  };

  std::vector<Pipeline> pipelines_;

  // The replaced Operators from the Driver. Can be used for a CPU fallback.
  std::vector<std::unique_ptr<exec::Operator>> cpuOperators_;

  // Top level column order in getOutput result.
  std::vector<OperandId> resultOrder_;

  // Dedupped Subfields. Handed over by CompileState.
  SubfieldMap subfields_;
  // Operands handed over by compilation.
  std::vector<std::unique_ptr<AbstractOperand>> operands_;
};

} // namespace facebook::velox::wave
