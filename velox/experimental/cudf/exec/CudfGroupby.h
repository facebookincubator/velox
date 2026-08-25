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

#include <optional>
#include <utility>

namespace facebook::velox::cudf_velox {

class CudaEvent;

// Type-specific adapter between Velox final-aggregation state and libcudf's
// flattened streaming_groupby request/result interface.
struct StreamingGroupbyAggregator {
  column_index_t inputIndex;
  TypePtr resultType;

  virtual void prepareInput(
      cudf::table_view input,
      std::vector<cudf::column_view>& preparedColumns) = 0;

  virtual void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests) = 0;

  virtual std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) = 0;

  virtual ~StreamingGroupbyAggregator() = default;

 protected:
  StreamingGroupbyAggregator(column_index_t inputIndex, TypePtr resultType)
      : inputIndex(inputIndex), resultType(std::move(resultType)) {}

  column_index_t prepareColumn(
      cudf::table_view input,
      std::vector<cudf::column_view>& preparedColumns,
      std::optional<column_index_t> childIndex = std::nullopt) const {
    VELOX_CHECK_LT(inputIndex, input.num_columns());
    auto column = input.column(inputIndex);
    if (childIndex.has_value()) {
      VELOX_CHECK_LT(*childIndex, column.num_children());
      column = column.child(*childIndex);
    }
    VELOX_CHECK_EQ(column.size(), input.num_rows());
    preparedColumns.push_back(column);
    return static_cast<column_index_t>(preparedColumns.size() - 1);
  }
};

struct GroupbyAggregator {
  core::AggregationNode::Step step;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;

  virtual void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream) = 0;

  virtual std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) = 0;

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

std::optional<std::vector<std::unique_ptr<StreamingGroupbyAggregator>>>
toStreamingGroupbyAggregators(
    const core::AggregationNode& aggregationNode,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& aggregationInputChannels,
    const TypePtr& outputType,
    const std::vector<VectorPtr>& constants);

// Groupby-specific validation
bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool);

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

  void doClose() override;

 private:
  CudfVectorPtr doGroupByAggregation(
      cudf::table_view tableView,
      std::vector<column_index_t> const& groupByKeys,
      std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  CudfVectorPtr releaseAndResetBufferedResult();

  bool initializeStreamingGroupbyApi(
      const RowTypePtr& inputRowSchema,
      const std::vector<VectorPtr>& constants);

  cudf::table_view makeStreamingGroupbyInputView(cudf::table_view input);

  std::unique_ptr<cudf::groupby::streaming_groupby> createStreamingGroupby(
      size_t capacity);

  void computeFinalGroupbyWithStreamingApi(CudfVectorPtr input);

  CudfVectorPtr finalizeStreamingGroupby();

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
  bool nativeStreamingEnabled_{false};
  const int64_t maxPartialAggregationMemoryUsage_;
  int64_t numInputRows_ = 0;

  bool finished_ = false;
  size_t numAggregates_;
  bool ignoreNullKeys_;

  std::vector<CudfVectorPtr> inputs_;
  TypePtr inputType_;
  RowTypePtr bufferedResultType_;
  CudfVectorPtr bufferedResult_;

  std::vector<std::unique_ptr<StreamingGroupbyAggregator>>
      streamingGroupbyAggregators_;
  std::unique_ptr<cudf::groupby::streaming_groupby> streamingGroupby_;
  std::optional<rmm::cuda_stream_view> streamingGroupbyStream_;
  std::unique_ptr<CudaEvent> streamingGroupbyEvent_;
  size_t streamingGroupbyCapacity_{0};
};

} // namespace facebook::velox::cudf_velox
