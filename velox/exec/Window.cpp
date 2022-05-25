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
#include "velox/exec/Window.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

Window::Window(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : Operator(
          driverCtx,
          windowNode->outputType(),
          operatorId,
          windowNode->id(),
          "Window"),
      inputColumnsSize_(windowNode->sources()[0]->outputType()->size()),
      data_(std::make_unique<RowContainer>(
          windowNode->sources()[0]->outputType()->children(),
          operatorCtx_->mappedMemory())),
      decodedInputVectors_(inputColumnsSize_),
      allKeysComparator_(
          windowNode->sources()[0]->outputType(),
          windowNode->partitionAndSortKeys(),
          windowNode->partitionAndSortOrders(),
          data_.get()),
      partitionKeysComparator_(
          windowNode->sources()[0]->outputType(),
          windowNode->partitionKeys(),
          {},
          data_.get()),
      windowPartitionsQueue_(allKeysComparator_),
      windowFunctions_(windowNode->windowFunctions()) {}

Window::Comparator::Comparator(
    const std::shared_ptr<const RowType>& type,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    RowContainer* rowContainer)
    : rowContainer_(rowContainer) {
  core::SortOrder defaultPartitionSortOrder(true, true);
  auto numKeys = sortingKeys.size();
  for (int i = 0; i < numKeys; ++i) {
    auto channel = exprToChannel(sortingKeys[i].get(), type);
    VELOX_CHECK(
        channel != kConstantChannel,
        "Window doesn't allow constant comparison keys");
    if (i < sortingOrders.size()) {
      keyInfo_.push_back(std::make_pair(channel, sortingOrders[i]));
    } else {
      keyInfo_.push_back(std::make_pair(channel, defaultPartitionSortOrder));
    }
  }
}

void Window::addInput(RowVectorPtr input) {
  SelectivityVector allRows(input->size());

  // TODO Decode keys first, then decode the rest only for passing positions
  for (int col = 0; col < input->childrenSize(); ++col) {
    decodedInputVectors_[col].decode(*input->childAt(col), allRows);
  }

  // Add all the rows into the RowContainer and update the partitionQueue
  for (int row = 0; row < input->size(); ++row) {
    char* newRow = data_->newRow();

    for (int col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedInputVectors_[col], row, newRow, col);
    }

    windowPartitionsQueue_.push(newRow);
  }
}

void Window::noMoreInput() {
  Operator::noMoreInput();
  if (windowPartitionsQueue_.empty()) {
    finished_ = true;
    return;
  }
  rows_.resize(windowPartitionsQueue_.size());
  for (int i = rows_.size(); i > 0; --i) {
    rows_[i - 1] = windowPartitionsQueue_.top();
    windowPartitionsQueue_.pop();
  }
}

RowVectorPtr Window::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  int numRowsToReturn = data_->numRows();
  auto result = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(outputType_, numRowsToReturn, operatorCtx_->pool()));

  // Set values of all output columns corresponding to the input columns.
  for (int i = 0; i < inputColumnsSize_; ++i) {
    data_->extractColumn(rows_.data(), numRowsToReturn, i, result->childAt(i));
  }

  std::vector<VectorPtr> windowFunctionOutputs;
  windowFunctionOutputs.reserve(outputType_->size() - inputColumnsSize_);
  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    auto windowOutputFlatVector = BaseVector::create(
        outputType_->childAt(j), numRowsToReturn, operatorCtx_->pool());
    windowFunctionOutputs.insert(
        windowFunctionOutputs.cend(), std::move(windowOutputFlatVector));
  }

  BufferPtr rowIndices = allocateIndices(numRowsToReturn, pool());
  auto* rawRowIndices = rowIndices->asMutable<vector_size_t>();
  for (int i = 0; i < numRowsToReturn; i++) {
    rawRowIndices[i] = i;
    if (i == 0 || partitionKeysComparator_(rows_[i - 1], rows_[i])) {
      // This is a new partition, so reset WindowFunction.
      for (int j = 0; j < outputType_->size() - inputColumnsSize_; j++) {
        windowFunctions_[j]->resetPartition();
      }
    }
    for (int j = 0; j < outputType_->size() - inputColumnsSize_; j++) {
      windowFunctions_[j]->evaluate(windowFunctionOutputs[j], i);
    }
  }

  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    result->childAt(j) = wrapChild(
        numRowsToReturn,
        rowIndices,
        windowFunctionOutputs[j - inputColumnsSize_]);
  }

  finished_ = true;
  return result;
}

bool Window::isFinished() {
  // Will operate one batch at a time and leaving this simple for now.
  return noMoreInput_ && input_ == nullptr;
}
} // namespace facebook::velox::exec
