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
#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

// Returns the byte target, or nullopt when unset or the output has no columns.
// Zero-column vectors own no GPU buffers to measure.
std::optional<uint64_t> getTargetBytes(const RowTypePtr& outputType) {
  if (outputType->size() == 0) {
    return std::nullopt;
  }
  const auto targetBytes = CudfConfig::getInstance().batchSizeMinBytes;
  if (targetBytes.has_value()) {
    VELOX_CHECK_GT(
        targetBytes.value(),
        0,
        "cuDF BatchConcat minimum byte target must be positive");
  }
  return targetBytes;
}

size_t getTargetRows() {
  const auto targetRows = CudfConfig::getInstance().batchSizeMinThreshold;
  VELOX_CHECK_GT(
      targetRows, 0, "cuDF BatchConcat minimum row target must be positive");
  return targetRows;
}

} // namespace

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          std::dynamic_pointer_cast<const CudfBatchConcatNode>(
              CudfPlanRewriter::translateBatchConcatForAdapter(planNode))) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const CudfBatchConcatNode> planNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          planNode->outputType(),
          planNode->id(),
          "CudfBatchConcat",
          nvtx3::rgb{211, 211, 211}, /* LightGrey */
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      targetBytes_(getTargetBytes(outputType_)),
      targetRows_(getTargetRows()) {}

bool CudfBatchConcat::meetsTarget(size_t numRows, uint64_t numBytes) const {
  if (targetBytes_.has_value()) {
    return numBytes >= targetBytes_.value();
  }
  return numRows >= targetRows_;
}

uint64_t CudfBatchConcat::estimateBytes(const CudfVector& vector) const {
  return targetBytes_.has_value() ? vector.estimateFlatSize() : 0;
}

void CudfBatchConcat::doAddInput(RowVectorPtr input) {
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "CudfBatchConcat expects CudfVector input");

  if (cudfVector->size() == 0) {
    return;
  }

  currentNumRows_ += cudfVector->size();
  currentBytes_ += estimateBytes(*cudfVector);
  buffer_.push_back(std::move(cudfVector));
}

RowVectorPtr CudfBatchConcat::doGetOutput() {
  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto output = std::move(outputQueue_.front());
    outputQueue_.pop();
    return output;
  }

  // Merge tables once the target is reached.
  if (!buffer_.empty() && (targetReached() || noMoreInput_)) {
    // Concatenating a single column-bearing input only materializes a copy of
    // the same table. Pass it through unchanged. Zero-column inputs still need
    // the batching helper below to preserve their row count and enforce the
    // maximum batch-size threshold.
    if (buffer_.size() == 1 && outputType_->size() > 0) {
      auto output = std::move(buffer_.front());
      buffer_.clear();
      currentNumRows_ = 0;
      currentBytes_ = 0;
      return output;
    }

    // Use stream from existing buffer vectors
    const auto outputStream = buffer_[0]->stream();
    auto outputVectors = getConcatenatedCudfVectorsBatched(
        pool(),
        std::exchange(buffer_, {}),
        outputType_,
        outputStream,
        get_output_mr());

    currentNumRows_ = 0;
    currentBytes_ = 0;
    VELOX_CHECK_GT(outputVectors.size(), 0);

    for (auto it = outputVectors.begin(); it + 1 != outputVectors.end(); ++it) {
      outputQueue_.push(std::move(*it));
    }

    // Keep the below-target tail of a split buffered while more input can
    // arrive. A lone output is emitted even if it now measures below the
    // target: concatenation merges null masks and string offsets, so byte
    // estimates shrink, and re-buffering would concatenate the rows twice.
    auto& last = outputVectors.back();
    const auto lastRows = static_cast<size_t>(last->size());
    const auto lastBytes = estimateBytes(*last);
    if (!noMoreInput_ && outputVectors.size() > 1 &&
        !meetsTarget(lastRows, lastBytes)) {
      currentNumRows_ = lastRows;
      currentBytes_ = lastBytes;
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

void CudfBatchConcat::doClose() {
  buffer_.clear();
  while (!outputQueue_.empty()) {
    outputQueue_.pop();
  }
  currentNumRows_ = 0;
  currentBytes_ = 0;
  Operator::close();
}

bool CudfBatchConcat::isFinished() {
  return noMoreInput_ && buffer_.empty() && outputQueue_.empty();
}

} // namespace facebook::velox::cudf_velox
