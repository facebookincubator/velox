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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

RowTypePtr getConcatOutputType(
    const std::shared_ptr<const core::PlanNode>& planNode) {
  const auto numSources = planNode->sources().size();
  if (planNode->is<core::AbstractJoinNode>()) {
    VELOX_CHECK_EQ(
        numSources,
        2,
        "CudfBatchConcat expects a join plan node to have exactly 2 sources");
  } else {
    VELOX_CHECK_EQ(
        numSources, 1, "CudfBatchConcat expects a single-source plan node");
  }
  return planNode->sources()[0]->outputType();
}

} // namespace

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          getConcatOutputType(planNode),
          planNode->id(),
          "CudfBatchConcat",
          nvtx3::rgb{211, 211, 211}, /* LightGrey */
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      driverCtx_(driverCtx),
      targetRows_(CudfConfig::getInstance().batchSizeMinThreshold) {}

void CudfBatchConcat::doAddInput(RowVectorPtr input) {
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "CudfBatchConcat expects CudfVector input");

  if (cudfVector->size() == 0) {
    return;
  }

  // Push input cudf table to buffer
  currentNumRows_ += cudfVector->size();
  buffer_.push_back(std::move(cudfVector));
}

RowVectorPtr CudfBatchConcat::doGetOutput() {
  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto output = std::move(outputQueue_.front());
    outputQueue_.pop();
    return output;
  }

  // Merge tables if there are enough rows
  if (!buffer_.empty() && (currentNumRows_ >= targetRows_ || noMoreInput_)) {
    // Use stream from existing buffer vectors
    const auto outputStream = buffer_[0]->stream();
    auto outputVectors = getConcatenatedCudfVectorsBatched(
        pool(),
        std::exchange(buffer_, {}),
        outputType_,
        outputStream,
        get_output_mr());

    currentNumRows_ = 0;
    VELOX_CHECK_GT(outputVectors.size(), 0);

    for (auto it = outputVectors.begin(); it + 1 != outputVectors.end(); ++it) {
      outputQueue_.push(std::move(*it));
    }

    // If last table is a smaller batch and we still expect more input and keep
    // it in buffer.
    auto& last = outputVectors.back();
    auto rowCount = last->size();

    if (!noMoreInput_ && rowCount < targetRows_) {
      currentNumRows_ = rowCount;
      buffer_.push_back(std::move(last));
    } else {
      outputQueue_.push(std::move(last));
    }

    // Return the first batch from the new queue
    if (!outputQueue_.empty()) {
      auto output = std::move(outputQueue_.front());
      outputQueue_.pop();
      return output;
    }
  }

  return nullptr;
}

bool CudfBatchConcat::isFinished() {
  return noMoreInput_ && buffer_.empty() && outputQueue_.empty();
}

} // namespace facebook::velox::cudf_velox
