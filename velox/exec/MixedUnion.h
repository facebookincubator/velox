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

#include "velox/core/PlanNode.h"
#include "velox/exec/MergeSource.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

/// Union operator that processes splits from all inputs simultaneously as a
/// SourceOperator. Unlike traditional operators that receive input via
/// addInput(), MixedUnion manages multiple MergeSource objects internally,
/// pulling data from each source and combining results in a round-robin or
/// interleaved fashion.
class MixedUnion : public SourceOperator {
 public:
  MixedUnion(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::MixedUnionNode>& unionNode);

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override;

 private:
  /// Get merge sources from the task
  BlockingReason addMergeSources(ContinueFuture* future);

  /// Start reading from sources
  void startSources();

  /// Process inputs in mixed mode (all at once)
  RowVectorPtr getOutputMixed();

  /// Combine multiple row vectors into a single output vector
  RowVectorPtr combineResults(std::vector<RowVectorPtr>& results);

  /// Check if we have data from all active sources for mixed mode
  bool hasDataFromAllSources() const;

  const std::shared_ptr<const core::MixedUnionNode> unionNode_;

  /// MergeSource objects representing each input pipeline
  std::vector<std::shared_ptr<MergeSource>> sources_;

  /// Track which sources have been started
  bool sourcesStarted_{false};

  /// Store pending data from each source
  std::vector<RowVectorPtr> pendingData_;

  /// Track which sources have finished
  std::vector<bool> sourcesFinished_;

  /// List of blocking futures for sources
  std::vector<ContinueFuture> sourceBlockingFutures_;

  /// True when all sources are exhausted
  bool finished_{false};

  /// Maximum output batch size
  const vector_size_t maxOutputBatchRows_;
  const uint64_t maxOutputBatchBytes_;
};

} // namespace facebook::velox::exec
