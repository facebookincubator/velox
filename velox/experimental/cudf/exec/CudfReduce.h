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

struct ReduceAggregator {
  core::AggregationNode::Step step;
  uint32_t inputIndex;
  VectorPtr constant;
  TypePtr resultType;

  virtual std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      rmm::cuda_stream_view stream,
      vector_size_t inputRowCount) = 0;

  virtual ~ReduceAggregator() = default;

 protected:
  ReduceAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : step(step),
        inputIndex(inputIndex),
        constant(constant),
        resultType(resultType) {}
};

std::vector<std::unique_ptr<ReduceAggregator>> toReduceAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants);

bool canReduceBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx);

bool canReduceAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx);

class CudfReduce : public CudfOperatorBase {
 public:
  CudfReduce(
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
  CudfVectorPtr doGlobalAggregation(
      cudf::table_view tableView,
      rmm::cuda_stream_view stream);

  std::shared_ptr<const core::AggregationNode> aggregationNode_;
  std::vector<std::unique_ptr<ReduceAggregator>> aggregators_;

  std::vector<column_index_t> aggregationInputChannels_;

  const bool isPartialOutput_;
  // Number of input rows accumulated since the last output.
  int64_t numInputRows_ = 0;

  bool finished_ = false;
  size_t numAggregates_;

  std::vector<CudfVectorPtr> inputs_;
  TypePtr inputType_;
};

} // namespace facebook::velox::cudf_velox
