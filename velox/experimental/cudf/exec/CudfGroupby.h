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
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox {

class CudaEvent;

inline constexpr std::string_view kStreamingGroupbyUsedStat{
    "streamingGroupbyUsed"};
inline constexpr std::string_view kStreamingGroupbyRebuildsStat{
    "streamingGroupbyRebuilds"};

// Type-specific adapter between Velox final-aggregation state and libcudf's
// flattened streaming_groupby request/result interface. prepareInput() must be
// called before addStreamingRequest(). The prepared input and result indices
// assigned by these methods must remain stable when requests are recreated for
// a capacity rebuild.
struct StreamingGroupbyAggregator {
  // Index in the unpermuted operator input and the final Velox result type.
  column_index_t inputIndex;
  TypePtr resultType;

  // Appends the input columns required by this aggregate and records their
  // positions in the prepared streaming_groupby input table.
  virtual void prepareInput(
      cudf::table_view input,
      std::vector<cudf::column_view>& preparedColumns) = 0;

  // Appends requests using the positions recorded by prepareInput() and records
  // their result positions.
  virtual void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests) = 0;

  // Consumes the result positions recorded by addStreamingRequest().
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
      std::optional<column_index_t> childIndex = std::nullopt) const;
};

struct GroupbyAggregator {
  core::AggregationNode::Step step;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;
  std::optional<uint32_t> maskIndex;

  virtual void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) = 0;

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
      const TypePtr& resultType,
      std::optional<uint32_t> maskIndex)
      : step(step),
        inputIndex(inputIndex),
        constant(constant),
        resultType(resultType),
        maskIndex(maskIndex) {}

  // Returns the value column for 'valueIdx'. When this aggregate has no mask,
  // returns tbl.column(valueIdx) directly. When it has a mask, materializes the
  // masked column into the owning member maskedValues_ and returns a view into
  // it -- so the returned view stays valid only until the next call on this
  // aggregator. doGroupByAggregation fully consumes 'requests' via aggregate()
  // before the next batch reuses the aggregator, so the view never dangles.
  cudf::column_view materializeMaskedInput(
      cudf::table_view const& tbl,
      uint32_t valueIdx,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

 private:
  std::unique_ptr<cudf::column> maskedValues_;
};

// Factory functions for creating groupby aggregators from plan nodes.
// 'maskChannels' carries the post-permutation mask column index per aggregate;
// pass the raw-input mask channels for raw base/partial steps and an empty
// vector for intermediate/final steps.
std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    std::vector<std::optional<uint32_t>> const& maskChannels);

std::optional<std::vector<std::unique_ptr<StreamingGroupbyAggregator>>>
toStreamingGroupbyAggregators(
    const core::AggregationNode& aggregationNode,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& aggregationInputChannels,
    const TypePtr& outputType,
    const std::vector<VectorPtr>& constants,
    const std::vector<std::optional<uint32_t>>& maskChannels);

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

  bool initializeStreamingGroupby(
      const RowTypePtr& inputRowSchema,
      const std::vector<VectorPtr>& constants,
      const std::vector<std::optional<uint32_t>>& maskChannels);

  cudf::table_view makeStreamingGroupbyInputView(cudf::table_view input);

  std::unique_ptr<cudf::groupby::streaming_groupby> createStreamingGroupby(
      size_t capacity);

  void computeFinalGroupbyStreaming(CudfVectorPtr input);

  CudfVectorPtr finalizeStreamingGroupby();

  void computePartialGroupbyIncrementally(CudfVectorPtr tbl);
  void computeFinalGroupbyIncrementally(CudfVectorPtr tbl);
  void computeSingleGroupbyIncrementally(CudfVectorPtr tbl);

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
  // Incremental aggregation is disabled if companion aggregates are present.
  bool incrementalAggregationEnabled_{true};
  bool streamingGroupbyEnabled_{false};
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
