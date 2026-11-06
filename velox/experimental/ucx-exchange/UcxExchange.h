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
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxQueues.h"

namespace facebook::velox::ucx_exchange {

using exec::BlockingReason;
using exec::DriverCtx;
using exec::SourceOperator;

/// @brief The UCX exchange operator receives data from upstream tasks via
/// UcxExchangeClient. exec::ExchangeTransportRegistry builds it for plan nodes
/// whose transportKind is core::TransportKind::kUcx. The operator receives
/// cudf::packed_columns from remote tasks and wraps them as CudfVectors.
class UcxExchange : public SourceOperator, public cudf_velox::NvtxHelper {
 public:
  /// @param ucxExchangeClient Client this operator reads from. Must not be
  /// null: it is created by the same exec::ExchangeTransportEntry that builds
  /// this operator and is shared by all consumers of the pipeline, so the Task
  /// that owns it -- not this operator -- closes it.
  UcxExchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::PlanNode>& planNode,
      std::shared_ptr<UcxExchangeClient> ucxExchangeClient,
      std::string_view operatorType = "UcxExchange");

  ~UcxExchange() override;

  [[nodiscard]] BlockingReason isBlocked(ContinueFuture* future) override;

  [[nodiscard]] bool isFinished() override;

  [[nodiscard]] RowVectorPtr getOutput() override;

  void close() override;

  /// Returns kInput for a MergeExchangeNode, whose output the CudfOrderBy
  /// spliced in behind this operator reports, and kBoth otherwise.
  core::PlanNode::Boundary planNodeBoundary() const override {
    return planNodeBoundary_;
  }

 private:
  // This is a no-op except when called from driver 0.
  // Fetches splits from the task until there are no more splits or task returns
  // a future that will be complete when more splits arrive. Records split
  // stats, and for a merge exchange also passes the splits and the end of
  // splits to exchangeClient_ (see forwardsSplits_). Sets "noMoreSplits_" if
  // the task returns not blocked without a split, this is the end-of-splits
  // signal.
  void getSplits(ContinueFuture* future);

  // Converts the cudf packed table into a CudfVector.
  RowVectorPtr getOutputFromPackedTable();

  // Drops a dequeued table that was not handed to a CudfVector and returns its
  // in-flight receive credit after stream-ordered destruction completes.
  void releaseCurrentData();

  // Fetches runtime stats from ExchangeClient and replaces these in this
  // operator's stats.
  void recordExchangeClientStats();

  void recordInputStats(uint64_t rawInputBytes, const RowVectorPtr& result);

  std::shared_ptr<UcxExchangeClient> exchangeClient_;

  const uint64_t preferredOutputBatchBytes_;

  // True if this operator takes splits from the Task, records split and
  // exchange client stats, and reports the splits finished. When running with
  // multiple drivers, this is done by the exchange running on driver 0.
  const bool processSplits_;
  const int pipelineId_;
  const int driverId_;

  // See planNodeBoundary().
  const core::PlanNode::Boundary planNodeBoundary_;

  // True for a MergeExchangeNode. The Task passes remote splits to the
  // exchange client of every other exchange node, but not of a merge exchange,
  // whose CPU operator fetches through clients of its own. This operator reads
  // the Task's client, so for a merge exchange it passes them on itself. A
  // merge exchange runs single-threaded, so driver 0 is the only reader.
  const bool forwardsSplits_;

  bool noMoreSplits_ = false;

  // A future received from Task::getSplitOrFuture(). It will be complete when
  // there are more splits available or no-more-splits signal has arrived.
  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  // Data returned from exchangeClient_->next().
  PackedTableWithStreamPtr currentData_;

  // Reusable result vector.
  RowVectorPtr result_;

  bool atEnd_{false};
  bool closed_{false};
};

} // namespace facebook::velox::ucx_exchange
