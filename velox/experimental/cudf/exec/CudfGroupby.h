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

#include <cudf/groupby.hpp>

#include <deque>
#include <limits>
#include <optional>
#include <string>

namespace facebook::velox::cudf_velox {

class GroupbyBufferedStateOps;
class BufferedGroupbyStateOps;
class StreamingGroupbyBufferedStateOps;
class StreamingGroupbyLeafState;

struct StreamingPreparedColumn {
  column_index_t inputIndex;
  //   TODO (dm): generalize this to support nested columns
  std::optional<column_index_t> childIndex;
  TypePtr type;
  std::string name;
};

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

struct StreamingGroupbyAggregator {
  core::AggregationNode::Step step;
  uint32_t aggregateIndex;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr inputType;
  TypePtr bufferedType;
  TypePtr finalType;

  virtual void addPreparedColumns(
      std::vector<StreamingPreparedColumn>& columns) = 0;

  virtual void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests) = 0;

  virtual std::unique_ptr<cudf::column> makeBufferedOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream) = 0;

  virtual std::unique_ptr<cudf::column> makeFinalOutputColumn(
      cudf::column_view const& bufferedColumn,
      rmm::cuda_stream_view stream) = 0;

  virtual ~StreamingGroupbyAggregator() = default;

 protected:
  StreamingGroupbyAggregator(
      core::AggregationNode::Step step,
      uint32_t aggregateIndex,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& inputType,
      const TypePtr& bufferedType,
      const TypePtr& finalType)
      : step(step),
        aggregateIndex(aggregateIndex),
        inputIndex(inputIndex),
        constant(constant),
        inputType(inputType),
        bufferedType(bufferedType),
        finalType(finalType) {}

  column_index_t addPreparedColumn(
      std::vector<StreamingPreparedColumn>& columns,
      std::optional<column_index_t> childIndex,
      const TypePtr& type,
      const std::string& name) const {
    columns.push_back(
        StreamingPreparedColumn{inputIndex, childIndex, type, name});
    return static_cast<column_index_t>(columns.size() - 1);
  }
};

std::vector<std::unique_ptr<StreamingGroupbyAggregator>>
toStreamingGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    RowTypePtr const& inputType,
    std::vector<column_index_t> const& aggregationInputChannels,
    TypePtr const& bufferedOutputType,
    TypePtr const& finalOutputType,
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
  friend class GroupbyBufferedStateOps;
  friend class BufferedGroupbyStateOps;
  friend class StreamingGroupbyBufferedStateOps;
  friend class StreamingGroupbyLeafState;

  CudfVectorPtr doGroupByAggregation(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream);

  bool canUseStreamingGroupbyApi(
      const RowTypePtr& inputRowSchema,
      const std::vector<VectorPtr>& constants) const;

  cudf::table_view makeStreamingPreparedInputView(
      cudf::table_view rawInputView) const;

  std::unique_ptr<cudf::groupby::streaming_groupby> createStreamingGroupby(
      size_t maxGroups) const;

  CudfVectorPtr materializeStreamingBufferedOutput(
      const cudf::groupby::streaming_groupby& groupby,
      rmm::cuda_stream_view stream) const;

  CudfVectorPtr finalizeStreamingBufferedOutput(
      CudfVectorPtr bufferedOutput) const;

  CudfVectorPtr releasePartialOutput(CudfVectorPtr output, int64_t inputRows);

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
  bool streamingGroupbyApiEnabled_{false};
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t numInputRows_ = 0;

  bool finished_ = false;
  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<CudfVectorPtr> inputs_;
  std::deque<std::pair<CudfVectorPtr, int64_t>> pendingPartialOutputs_;
  TypePtr inputType_;
  RowTypePtr bufferedResultType_;
  RowTypePtr streamingPreparedType_;
  std::vector<StreamingPreparedColumn> streamingPreparedColumns_;
  std::vector<std::unique_ptr<StreamingGroupbyAggregator>>
      streamingGroupbyAggregators_;
  std::unique_ptr<FlushableBufferedState> flushableBufferedState_;
  std::unique_ptr<PartitionedBufferedState> partitionedBufferedState_;
  size_t maxBufferedRows_{
      static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())};
};

} // namespace facebook::velox::cudf_velox
