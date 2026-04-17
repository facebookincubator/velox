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

 private:
  CudfVectorPtr getDistinctKeys(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      rmm::cuda_stream_view stream);

  CudfVectorPtr releaseAndResetBufferedResult();

  void computePartialDistinctStreaming(CudfVectorPtr tbl);

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;

  const bool isPartialOutput_;
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t numInputRows_ = 0;

  bool finished_ = false;

  std::vector<CudfVectorPtr> inputs_;
  TypePtr inputType_;
  CudfVectorPtr bufferedResult_;
};

} // namespace facebook::velox::cudf_velox
