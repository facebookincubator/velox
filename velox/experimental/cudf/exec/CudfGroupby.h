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

#include <cudf/groupby.hpp>

namespace facebook::velox::cudf_velox {

struct GroupbyAggregator {
  core::AggregationNode::Step step;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;

  virtual void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests) = 0;

  virtual std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) = 0;

  virtual ~GroupbyAggregator() = default;

 protected:
  GroupbyAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : step(step),
        inputIndex(inputIndex),
        constant(constant),
        resultType(resultType) {}
};

// Factory functions for creating groupby aggregators from plan nodes.
std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants);

// Groupby-specific validation
bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

class CudfGroupby : public CudfOperatorBase {
 public:
  CudfGroupby(
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
  CudfVectorPtr doGroupByAggregation(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream);

  CudfVectorPtr releaseAndResetBufferedResult();

  void computePartialGroupbyStreaming(CudfVectorPtr tbl);
  void computeFinalGroupbyStreaming(CudfVectorPtr tbl);
  void computeSingleGroupbyStreaming(CudfVectorPtr tbl);

  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;
  std::vector<column_index_t> aggregationInputChannels_;

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators_;
  std::vector<std::unique_ptr<GroupbyAggregator>> intermediateAggregators_;
  // Used for kSingle streaming: partial-step aggregators (raw -> intermediate)
  // and final-step aggregators (intermediate -> final).
  std::vector<std::unique_ptr<GroupbyAggregator>> partialAggregators_;
  std::vector<std::unique_ptr<GroupbyAggregator>> finalAggregators_;

  const bool isPartialOutput_;
  const bool isSingleStep_;
  // Streaming aggregation is disabled if companion aggregates are present.
  bool streamingEnabled_{true};
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t numInputRows_ = 0;

  bool finished_ = false;
  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<CudfVectorPtr> inputs_;
  TypePtr inputType_;
  RowTypePtr bufferedResultType_;
  CudfVectorPtr bufferedResult_;
};

} // namespace facebook::velox::cudf_velox
