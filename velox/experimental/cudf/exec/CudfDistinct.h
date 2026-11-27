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

#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/PartitionedBufferedState.h"

namespace facebook::velox::cudf_velox {

class CudfDistinct : public CudfOperatorBase {
 public:
  CudfDistinct(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::AggregationNode> const& aggregationNode);

  void initialize() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;

  RowVectorPtr doGetOutput() override;

  void doNoMoreInput() override;

  void doClose() override;

 private:
  // Keeps the distinct keys seen so far in cuDF tables. Defined in
  // CudfDistinct.cpp.
  class DistinctBufferedStateOps;

  // Returns the distinct rows of the columns 'groupByKeys' of 'tableView', or
  // nullptr if there are none.
  CudfVectorPtr getDistinctKeys(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      cuda::stream_ref stream);

  void recordPartialFlushStats(const CudfVector& output);

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;

  const bool isPartialOutput_;
  const bool isSingleStep_;
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t numInputRows_ = 0;

  bool finished_ = false;
  bool statePartitionsRecorded_ = false;

  // The keys of a partial or intermediate step. They are sent downstream when
  // they reach the row limit or the memory limit for partial aggregation.
  std::unique_ptr<FlushableBufferedState> flushableBufferedState_;
  // The keys of a final or single step, kept in hash partitions until the
  // input ends.
  std::unique_ptr<PartitionedBufferedState> partitionedBufferedState_;
};

} // namespace facebook::velox::cudf_velox
