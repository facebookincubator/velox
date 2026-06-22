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

#include <random>
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
/// UcxExchangeClient. It is used as a replacement for the Velox Exchange
/// operator when the plan node's transport type is UCX. The operator receives
/// cudf::packed_columns from remote tasks and wraps them as CudfVectors.
class UcxExchange : public SourceOperator, public cudf_velox::NvtxHelper {
 public:
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

 private:
  // Invoked to create exchange client for remote tasks. The function shuffles
  // the source task ids first to randomize the source tasks we fetch data from.
  // This helps to avoid different tasks fetching from the same source task in a
  // distributed system.
  void addRemoteTaskIds(std::vector<std::string>& remoteTaskIds);

  // This is a no-op except when called from driver 0.
  // Fetches splits from the task until there are no more splits or task returns
  // a future that will be complete when more splits arrive. Adds splits to
  // exchangeClient_. Sets "noMoreSplits_" if the task returns not blocked
  // without a split, this is the end-of-splits signal.
  void getSplits(ContinueFuture* future);

  // Converts the cudf packed table into a CudfVector.
  RowVectorPtr getOutputFromPackedTable();

  // Fetches runtime stats from ExchangeClient and replaces these in this
  // operator's stats.
  void recordExchangeClientStats();

  void recordInputStats(uint64_t rawInputBytes, const RowVectorPtr& result);

  std::shared_ptr<UcxExchangeClient> exchangeClient_;

  const uint64_t preferredOutputBatchBytes_;

  /// True if this operator is responsible for fetching splits from the Task
  /// and passing these to ExchangeClient. When running with multile drivers,
  /// this is done by the exchange running on driver 0.
  const bool processSplits_;
  const int pipelineId_;
  const int driverId_;
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
  std::default_random_engine rng_{std::random_device{}()};
};

} // namespace facebook::velox::ucx_exchange
