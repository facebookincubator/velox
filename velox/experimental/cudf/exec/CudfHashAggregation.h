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

#include "velox/exec/GroupingSet.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/groupby.hpp>

namespace facebook::velox::cudf_velox {

class CudfHashAggregation : public exec::Operator, public NvtxHelper {
 public:
  struct Aggregator {
    core::AggregationNode::Step step;
    bool is_global;
    cudf::aggregation::Kind kind;
    uint32_t inputIndex;
    VectorPtr constant;

    virtual void addGroupbyRequest(
        cudf::table_view const& tbl,
        std::vector<cudf::groupby::aggregation_request>& requests) = 0;

    virtual std::unique_ptr<cudf::column> doReduce(
        cudf::table_view const& input,
        TypePtr const& output_type,
        rmm::cuda_stream_view stream) = 0;

    virtual std::unique_ptr<cudf::column> makeOutputColumn(
        std::vector<cudf::groupby::aggregation_result>& results,
        rmm::cuda_stream_view stream) = 0;

   protected:
    Aggregator(
        core::AggregationNode::Step step,
        cudf::aggregation::Kind kind,
        uint32_t inputIndex,
        VectorPtr constant,
        bool is_global)
        : step(step),
          is_global(is_global),
          kind(kind),
          inputIndex(inputIndex),
          constant(constant) {}
  };

  CudfHashAggregation(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::AggregationNode> const& aggregationNode);

  void initialize() override;

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  // Setups the projections for accessing grouping keys stored in grouping
  // set.
  // For 'groupingKeyInputChannels', the index is the key column index from
  // the grouping set, and the value is the key column channel from the input.
  // For 'outputChannelProjections', the index is the key column channel from
  // the output, and the value is the key column index from the grouping set.
  void setupGroupingKeyChannelProjections(
      std::vector<column_index_t>& groupingKeyInputChannels,
      std::vector<column_index_t>& groupingKeyOutputChannels) const;

  RowVectorPtr doGroupByAggregation(
      std::unique_ptr<cudf::table> tbl,
      rmm::cuda_stream_view stream);
  RowVectorPtr doGlobalAggregation(
      std::unique_ptr<cudf::table> tbl,
      rmm::cuda_stream_view stream);
  RowVectorPtr getDistinctKeys(
      std::unique_ptr<cudf::table> tbl,
      rmm::cuda_stream_view stream);

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  std::vector<std::unique_ptr<Aggregator>> aggregators_;

  // Partial aggregation is the first phase of aggregation. e.g. count(*) when
  // in partial phase will do a count_agg but in the final phase will do a sum
  // of the previous calculated counts
  const bool isPartialOutput_;
  // Global means it's an aggregation without groupby. Like cudf::reduce
  const bool isGlobal_;
  // Distinct means it's a count distinct on the groupby keys, without any
  // aggregations
  const bool isDistinct_;

  bool finished_ = false;

  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<cudf_velox::CudfVectorPtr> inputs_;
};

} // namespace facebook::velox::cudf_velox
