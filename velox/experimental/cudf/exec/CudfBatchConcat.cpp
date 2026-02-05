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
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/Utilities.h"

namespace facebook::velox::cudf_velox {

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode)
    : exec::Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfBatchConcat"),
      CudfOperator(operatorId, planNode->id()),
      driverCtx_(driverCtx),
      targetRows_(CudfConfig::getInstance().batchSizeMinThreshold) {}

void CudfBatchConcat::addInput(RowVectorPtr input) {
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "CudfBatchConcat expects CudfVector input");

  // Push input cudf table to buffer
  currentNumRows_ += cudfVector->getTableView().num_rows();
  buffer_.push_back(std::move(cudfVector));
}

RowVectorPtr CudfBatchConcat::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto table = std::move(outputQueue_.front());
    auto rowCount = table->num_rows();
    outputQueue_.pop();
    auto stream = cudfGlobalStreamPool().get_stream();
    return std::make_shared<CudfVector>(
        pool(), outputType_, rowCount, std::move(table), stream);
  }

  // Merge tables if there are enough rows
  if (currentNumRows_ >= targetRows_ || (noMoreInput_ && !buffer_.empty())) {
    // Use stream from existing buffer vectors
    auto stream = buffer_[0]->stream();
    auto tables = getConcatenatedTableBatched(buffer_, outputType_, stream);

    buffer_.clear();
    currentNumRows_ = 0;

    for (auto it = tables.begin(); it + 1 != tables.end(); ++it) {
      outputQueue_.push(std::move(*it));
    }

    // If last table is a smaller batch and we still expect more input and keep
    // it in buffer.
    auto& last = tables.back();
    auto rowCount = last->num_rows();

    if (!noMoreInput_ && rowCount < targetRows_) {
      currentNumRows_ = rowCount;
      buffer_.push_back(
          std::make_shared<CudfVector>(
              pool(), outputType_, rowCount, std::move(last), stream));
    } else {
      outputQueue_.push(std::move(last));
    }

    // Return the first batch from the new queue
    if (!outputQueue_.empty()) {
      auto table = std::move(outputQueue_.front());
      stream = cudfGlobalStreamPool().get_stream();
      auto rowCount = table->num_rows();
      outputQueue_.pop();
      return std::make_shared<CudfVector>(
          pool(), outputType_, rowCount, std::move(table), stream);
    }
  }

  return nullptr;
}

bool CudfBatchConcat::isFinished() {
  return noMoreInput_ && buffer_.empty() && outputQueue_.empty();
}

} // namespace facebook::velox::cudf_velox
