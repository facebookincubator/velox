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
#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"
#include "velox/experimental/cudf-exchange/CudfExchangeSource.h"
#include "velox/experimental/cudf-exchange/CudfQueues.h"

using namespace facebook::velox::exec;

namespace facebook::velox::cudf_exchange {

class CudfExchange : public SourceOperator {
 public:
  CudfExchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::PlanNode>& planNode,
      std::shared_ptr<CudfExchangeClient> exchangeClient,
      const std::string& operatorType = "CudfExchange");

  ~CudfExchange() override {
    close();
  }

  RowVectorPtr getOutput() override;

  void close() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 private:
  // Invoked to create CudfExchangeSources for remote tasks. The function
  // shuffles the source task ids first to randomize the source tasks we fetch
  // data from. This helps to avoid different tasks fetching from the same
  // source task in a distributed system.
  void addRemoteTaskIds(std::vector<std::string>& remoteTaskIds);

  // Fetches splits from the task until there are no more splits or task returns
  // a future that will be complete when more splits arrive. Adds splits to
  // exchangeClient_. Returns true if received a future from the task and sets
  // the 'future' parameter. Returns false if fetched all splits or if this
  // operator is not the first operator in the pipeline and therefore is not
  // responsible for fetching splits and adding them to the exchangeClient_.
  bool getSplits(ContinueFuture* future);

  // Fetches runtime stats from ExchangeClient and replaces these in this
  // operator's stats.
  void recordExchangeClientStats();

  void recordInputStats(uint64_t rawInputBytes, RowVectorPtr result);

  /// True if this operator is responsible for fetching splits from the Task
  /// and passing these to ExchangeClient.
  const bool processSplits_;

  const int driverId_;

  bool noMoreSplits_ = false;

  std::shared_ptr<CudfExchangeClient> exchangeClient_;

  // A future received from Task::getSplitOrFuture(). It will be complete when
  // there are more splits available or no-more-splits signal has arrived.
  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  PackedTableWithStreamPtr currentData_;
  bool atEnd_{false};

  std::default_random_engine rng_{std::random_device{}()};
};

} // namespace facebook::velox::cudf_exchange
