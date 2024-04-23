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
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

/// Simple history based fallback row size estimator that is only used when
/// underlying 'DataSource' does not support row size estimation.
class RowSizeEstimator {
 public:
  static constexpr int64_t kDefaultConservativeRowSize = 2 << 20;
  static constexpr float kDefaultDecayFactor = 0.9;

  RowSizeEstimator(
      const RowTypePtr& rowType,
      int64_t conservativeRowSize = kDefaultConservativeRowSize,
      float decayFactor = kDefaultDecayFactor);

  void updateEstimator(const VectorPtr& resultVector);

  int64_t estimatedRowSize() const;

 private:
  // Currently we identify a type to be potential large memory user by checking
  // if it contains MAP or ARRAY types.
  bool maybeLargeMemoryUser(const TypePtr& type);

  // Returned as first batch estimation. For the first batch estimation, since
  // there is no historical data to base on, this number is used as a
  // conservative estimation.
  //
  // NOTE: This will only be applied if 'convervativeEstimate_' is true.
  const int64_t conseravtiveRowSize_;

  // If the vector provided to updateEstimator() has less average row size, we
  // reduce 'estimatedRowSizeBytes_' with at most (no less than) this decay
  // factor.
  const float decayFactor_;

  // When set to true, start initial estimation with 'conseravtiveRowSize_'. If
  // there are MAP or ARRAY types in the output type, we set this to true upon
  // estimator construction.
  bool conservativeEstimate_;
  int64_t estimatedRowSizeBytes_{connector::DataSource::kUnknownRowSize};
};

class TableScan : public SourceOperator {
 public:
  TableScan(
      int32_t operatorId,
      DriverCtx* driverCtx,
      std::shared_ptr<const core::TableScanNode> tableScanNode);

  folly::dynamic toJson() const override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override {
    if (blockingFuture_.valid()) {
      *future = std::move(blockingFuture_);
      return blockingReason_;
    }
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  bool canAddDynamicFilter() const override {
    return connector_->canAddDynamicFilter();
  }

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) override;

 private:
  // Checks if this table scan operator needs to yield before processing the
  // next split.
  bool shouldYield(StopReason taskStopReason, size_t startTimeMs) const;

  // Checks if this table scan operator needs to stop because the task has been
  // terminated.
  bool shouldStop(StopReason taskStopReason) const;

  // Sets 'maxPreloadSplits' and 'splitPreloader' if prefetching splits is
  // appropriate. The preloader will be applied to the 'first 'maxPreloadSplits'
  // of the Task's split queue for 'this' when getting splits.
  void checkPreload();

  // Sets 'split->dataSource' to be an AsyncSource that makes a DataSource to
  // read 'split'. This source will be prepared in the background on the
  // executor of the connector. If the DataSource is needed before prepare is
  // done, it will be made when needed.
  void preload(std::shared_ptr<connector::ConnectorSplit> split);

  const std::shared_ptr<connector::ConnectorTableHandle> tableHandle_;
  const std::
      unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
          columnHandles_;
  DriverCtx* const driverCtx_;
  memory::MemoryPool* const connectorPool_;
  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};
  BlockingReason blockingReason_;
  int64_t currentSplitWeight_{0};
  bool needNewSplit_ = true;
  std::shared_ptr<connector::Connector> connector_;
  std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<connector::DataSource> dataSource_;
  bool noMoreSplits_ = false;
  // Dynamic filters to add to the data source when it gets created.
  std::unordered_map<column_index_t, std::shared_ptr<common::Filter>>
      dynamicFilters_;

  int32_t maxPreloadedSplits_{0};

  const int32_t maxSplitPreloadPerDriver_{0};

  // Callback passed to getSplitOrFuture() for triggering async preload. The
  // callback's lifetime is the lifetime of 'this'. This callback can schedule
  // preloads on an executor. These preloads may outlive the Task and therefore
  // need to capture a shared_ptr to it.
  std::function<void(const std::shared_ptr<connector::ConnectorSplit>&)>
      splitPreloader_{nullptr};

  // Count of splits that started background preload.
  int32_t numPreloadedSplits_{0};

  // Count of splits that finished preloading before being read.
  int32_t numReadyPreloadedSplits_{0};

  int32_t readBatchSize_;
  int32_t maxReadBatchSize_;

  // Exits getOutput() method after this many milliseconds. Zero means 'no
  // limit'.
  size_t getOutputTimeLimitMs_{0};

  double maxFilteringRatio_{0};

  // String shown in ExceptionContext inside DataSource and LazyVector loading.
  std::string debugString_;

  // Holds the current status of the operator. Used when debugging to understand
  // what operator is doing.
  std::atomic<const char*> curStatus_{""};

  // Fallback row size estimator that will only be used if 'dataSource_' does
  // not support row size estimation.
  RowSizeEstimator rowSizeEstimator_;
};
} // namespace facebook::velox::exec
