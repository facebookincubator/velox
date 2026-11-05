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

#include "velox/experimental/cudf/exec/CudfProjectionOperator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/column/column_factories.hpp>

namespace facebook::velox::cudf_velox {

CudfProjectionOperator::CudfProjectionOperator(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    RowTypePtr outputType,
    const core::PlanNodeId& planNodeId,
    const std::string& operatorName,
    std::optional<nvtx3::color> color,
    NvtxMethodFlag nvtxMethods,
    std::optional<std::shared_ptr<const core::PlanNode>> planNode,
    std::vector<std::vector<ProjectionColumn>> projections,
    size_t numInputColumns)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          std::move(outputType),
          planNodeId,
          operatorName,
          color,
          nvtxMethods,
          std::nullopt,
          std::move(planNode)),
      projections_(std::move(projections)),
      numInputColumns_(numInputColumns) {
  VELOX_CHECK(!projections_.empty());

  // Walk the projections backwards so the first sighting of a channel is its
  // last use.
  moveInputColumns_.resize(projections_.size());
  std::vector<bool> seenChannels(numInputColumns_, false);
  for (auto projection = projections_.size(); projection-- > 0;) {
    const auto& columns = projections_[projection];
    moveInputColumns_[projection].assign(columns.size(), false);
    for (auto column = columns.size(); column-- > 0;) {
      const auto channel = columns[column].channel;
      if (channel == kConstantChannel) {
        continue;
      }
      VELOX_CHECK_LT(channel, numInputColumns_);
      if (!seenChannels[channel]) {
        seenChannels[channel] = true;
        moveInputColumns_[projection][column] = true;
      }
    }
  }
}

bool CudfProjectionOperator::needsInput() const {
  return !noMoreInput_ && !hasInput_;
}

bool CudfProjectionOperator::isFinished() {
  return noMoreInput_ && !hasInput_;
}

void CudfProjectionOperator::doAddInput(RowVectorPtr input) {
  VELOX_CHECK(!hasInput_, "Previous input not fully consumed");
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(
      cudfInput, "{} expects CudfVector input", operatorType());

  // Read before release(), which invalidates the vector.
  numRows_ = static_cast<cudf::size_type>(cudfInput->size());
  stream_ = cudfInput->stream();
  inputColumns_ = cudfInput->release()->release();
  VELOX_CHECK_EQ(inputColumns_.size(), numInputColumns_);
  hasInput_ = true;
}

RowVectorPtr CudfProjectionOperator::doGetOutput() {
  if (!hasInput_) {
    return nullptr;
  }

  const auto outputMr = get_output_mr();
  if (scalars_.empty()) {
    scalarStream_ = stream_;
    scalars_.reserve(projections_.size());
    for (const auto& columns : projections_) {
      auto& scalars = scalars_.emplace_back();
      scalars.reserve(columns.size());
      for (const auto& column : columns) {
        scalars.push_back(
            column.makeScalar ? column.makeScalar(stream_, pool()) : nullptr);
      }
    }
  }

  // The scalars are used without synchronization, so every batch must run on
  // the stream that built them.
  VELOX_CHECK(
      stream_ == scalarStream_, "Input batches must share one CUDA stream");

  const auto& columns = projections_[projectionIndex_];
  const auto& moveInputColumns = moveInputColumns_[projectionIndex_];
  const auto& scalars = scalars_[projectionIndex_];

  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(columns.size());
  // TODO: Avoid deep copies by letting the output reference the input columns,
  // copying only when a downstream operator needs to own them.
  for (size_t i = 0; i < columns.size(); ++i) {
    const auto channel = columns[i].channel;
    if (channel == kConstantChannel) {
      outputColumns.push_back(
          cudf::make_column_from_scalar(
              *scalars[i], numRows_, stream_, outputMr));
    } else if (moveInputColumns[i]) {
      outputColumns.push_back(std::move(inputColumns_[channel]));
    } else {
      outputColumns.push_back(
          std::make_unique<cudf::column>(
              *inputColumns_[channel], stream_, outputMr));
    }
  }

  if (++projectionIndex_ == projections_.size()) {
    projectionIndex_ = 0;
    inputColumns_.clear();
    hasInput_ = false;
  }

  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
  return std::make_shared<CudfVector>(
      pool(), outputType_, numRows_, std::move(outputTable), stream_);
}

void CudfProjectionOperator::doClose() {
  Operator::close();
  inputColumns_.clear();
  scalars_.clear();
}

} // namespace facebook::velox::cudf_velox
