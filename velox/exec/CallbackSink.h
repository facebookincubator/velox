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
#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::exec {

/// A sink Operator at the end of a pipeline that hands each input batch to a
/// consumer callback instead of producing output for a downstream operator.
/// Used both for a pipeline's final output (delivering the fragment result to
/// its consumer) and to feed the source queues of LocalMerge / MixedUnion /
/// MergeJoin from their producer pipelines.
class CallbackSink : public Operator {
 public:
  /// @param consumeCb Invoked with each input batch; its BlockingReason
  /// controls back-pressure.
  /// @param startedCb Optional, invoked once before the first input.
  /// @param planNodeId Defaults to "N/A" because such a sink usually has no
  /// plan node of its own. When the sink belongs to a node (e.g. it feeds that
  /// node's merge source), pass the node's id so the sink's runtime stats
  /// aggregate into that node and query tracing can associate it.
  CallbackSink(
      int32_t operatorId,
      DriverCtx* driverCtx,
      Consumer consumeCb,
      std::function<BlockingReason(ContinueFuture*)> startedCb = nullptr,
      const std::string& planNodeId = "N/A")
      : Operator(
            driverCtx,
            nullptr,
            operatorId,
            planNodeId,
            OperatorType::kCallbackSink),
        startedCb_{std::move(startedCb)},
        consumeCb_{std::move(consumeCb)} {}

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr: a sink consumes its input via the callback and
  /// produces no output for a downstream operator.
  RowVectorPtr getOutput() override;

  bool startDrain() override;

  bool needsInput() const override {
    return consumeCb_ != nullptr;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return noMoreInput_;
  }

 private:
  void close() override;

  ContinueFuture future_;
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  std::function<BlockingReason(ContinueFuture*)> startedCb_;
  Consumer consumeCb_;
};

} // namespace facebook::velox::exec
