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
      decodedInputVectors_(inputColumnsSize_) {
  std::vector<core::FieldAccessTypedExprPtr> partitionAndSortKeys;
  std::vector<core::SortOrder> partitionAndSortOrders;
  partitionAndSortKeys.reserve(
      windowNode->partitionKeys().size() + windowNode->sortingKeys().size());
  partitionAndSortOrders.reserve(
      windowNode->partitionKeys().size() + windowNode->sortingOrders().size());

  partitionAndSortKeys.insert(
      partitionAndSortKeys.cend(),
      windowNode->partitionKeys().begin(),
      windowNode->partitionKeys().end());
  partitionAndSortKeys.insert(
      partitionAndSortKeys.cend(),
      windowNode->sortingKeys().begin(),
      windowNode->sortingKeys().end());
  core::SortOrder defaultPartitionSortOrder(true, true);
  partitionAndSortOrders.insert(
      partitionAndSortOrders.cend(),
      windowNode->partitionKeys().size(),
      defaultPartitionSortOrder);
  partitionAndSortOrders.insert(
      partitionAndSortOrders.cend(),
      windowNode->sortingOrders().begin(),
      windowNode->sortingOrders().end());

  initKeyInfo(
      windowNode->sources()[0]->outputType(),
      partitionAndSortKeys,
      partitionAndSortOrders,
      allKeyInfo_);
  initKeyInfo(
      windowNode->sources()[0]->outputType(),
      windowNode->partitionKeys(),
      {},
      partitionKeyInfo_);
  for (auto i = 0; i < windowNode->windowFunctions().size(); i++) {
    const auto& windowNodeFunction = windowNode->windowFunctions()[i];
    std::vector<TypePtr> argTypes;
    for (auto& arg : windowNodeFunction.functionCall->inputs()) {
      argTypes.push_back(arg->type());
    }
    const auto& resultType = outputType_->childAt(inputColumnsSize_ + i);
    windowFunctions_.push_back(WindowFunction::create(
        windowNodeFunction.functionCall->name(), argTypes, resultType));

    std::optional<ChannelIndex> windowFrameStartChannel;
    std::optional<ChannelIndex> windowFrameEndChannel;
    if (windowNodeFunction.frame.startValue) {
      windowFrameStartChannel = exprToChannel(
          windowNodeFunction.frame.startValue.get(),
          windowNode->sources()[0]->outputType());
      VELOX_CHECK(
          windowFrameStartChannel.value() != kConstantChannel,
          "Window doesn't allow constant comparison keys");
    }
    if (windowNodeFunction.frame.endValue) {
      windowFrameEndChannel = exprToChannel(
          windowNodeFunction.frame.endValue.get(),
          windowNode->sources()[0]->outputType());
      VELOX_CHECK(
          windowFrameEndChannel.value() != kConstantChannel,
          "Window doesn't allow constant comparison keys");
    }
    windowFrames_.push_back(
        {windowNodeFunction.frame.type,
         windowNodeFunction.frame.startType,
         windowNodeFunction.frame.endType,
         windowFrameStartChannel,
         windowFrameEndChannel});
  }
}

void Window::initKeyInfo(
    const std::shared_ptr<const RowType>& type,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    std::vector<std::pair<ChannelIndex, core::SortOrder>>& keyInfo) {
  core::SortOrder defaultPartitionSortOrder(true, true);
  auto numKeys = sortingKeys.size();
  for (int i = 0; i < numKeys; ++i) {
    auto channel = exprToChannel(sortingKeys[i].get(), type);
    VELOX_CHECK(
        channel != kConstantChannel,
        "Window doesn't allow constant comparison keys");
    if (i < sortingOrders.size()) {
      keyInfo.push_back(std::make_pair(channel, sortingOrders[i]));
    } else {
      keyInfo.push_back(std::make_pair(channel, defaultPartitionSortOrder));
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
  }
  numRows_ += allRows.size();
}

void Window::noMoreInput() {
  Operator::noMoreInput();
  // No data.
  if (numRows_ == 0) {
    finished_ = true;
    return;
  }

  // Sort the pointers to the rows in RowContainer (data_) instead of sorting
  // the rows.
  returningRows_.resize(numRows_);
  RowContainerIterator iter;
  data_->listRows(&iter, numRows_, returningRows_.data());

  std::sort(
      returningRows_.begin(),
      returningRows_.end(),
      [this](const char* leftRow, const char* rightRow) {
        for (auto& [channelIndex, sortOrder] : allKeyInfo_) {
          if (auto result = data_->compare(
                  leftRow,
                  rightRow,
                  channelIndex,
                  {sortOrder.isNullsFirst(), sortOrder.isAscending(), false})) {
            return result < 0;
          }
        }
        return false; // lhs == rhs.
      });
}

std::pair<int32_t, int32_t> Window::findFrameEndPoints(
    int32_t windowFunctionIndex,
    int32_t rowNumber) {
  // TODO : We handle only the default window frame in this code. Add support
  // for all window frames subsequently.
  VELOX_CHECK_EQ(
      windowFrames_[windowFunctionIndex].windowType_,
      core::WindowNode::WindowType::kRange);
  VELOX_CHECK_EQ(
      windowFrames_[windowFunctionIndex].startBoundType_,
      core::WindowNode::BoundType::kUnboundedPreceding);
  VELOX_CHECK_EQ(
      windowFrames_[windowFunctionIndex].endBoundType_,
      core::WindowNode::BoundType::kCurrentRow);
  VELOX_CHECK(!windowFrames_[windowFunctionIndex].startChannel_.has_value());
  VELOX_CHECK(!windowFrames_[windowFunctionIndex].endChannel_.has_value());

  // Default window frame is Range UNBOUNDED PRECEDING CURRENT ROW.
  return std::make_pair(partitionStartRow_, rowNumber);
}

RowVectorPtr Window::getOutput() {
  if (finished_ || !noMoreInput_ || returningRows_.size() == numRowsReturned_) {
    return nullptr;
  }

  size_t numRows = data_->estimatedNumRowsPerBatch(kBatchSizeInBytes);
  int32_t numRowsToReturn =
      std::min(numRows, returningRows_.size() - numRowsReturned_);

  VELOX_CHECK_GT(numRowsToReturn, 0);

  auto result = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(outputType_, numRowsToReturn, operatorCtx_->pool()));

  // Set values of all output columns corresponding to the input columns.
  for (int i = 0; i < inputColumnsSize_; ++i) {
    data_->extractColumn(
        returningRows_.data() + numRowsReturned_,
        numRowsToReturn,
        i,
        result->childAt(i));
  }

  std::vector<VectorPtr> windowFunctionOutputs;
  windowFunctionOutputs.reserve(outputType_->size() - inputColumnsSize_);
  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    auto windowOutputFlatVector = BaseVector::create(
        outputType_->childAt(j), numRowsToReturn, operatorCtx_->pool());
    windowFunctionOutputs.insert(
        windowFunctionOutputs.cend(), std::move(windowOutputFlatVector));
  }

  auto partitionCompare = [&](const char* lhs, const char* rhs) -> bool {
    if (lhs == rhs) {
      return false;
    }
    for (auto& key : partitionKeyInfo_) {
      if (auto result = data_->compare(
              lhs,
              rhs,
              key.first,
              {key.second.isNullsFirst(), key.second.isAscending(), false})) {
        return result < 0;
      }
    }
    return false;
  };

  int partitionStartRow;
  int partitionEndRow;
  for (int i = 0; i < numRowsToReturn;) {
    partitionStartRow_ = numRowsReturned_ + i;
    partitionStartRow = i;
    if ((numRowsReturned_ + i == returningRows_.size() - 1)) {
      // This is the last row of the input.
      // Fake partitionEnd row to be one row beyond the output block. This works
      // for this logic but is a bit hacky.
      partitionEndRow = i + 1;
    } else {
      // Lookahead and find partition end. The java code uses a binary search
      // style lookup instead of iterating over consecutive rows.
      partitionEndRow = 0;
      for (int j = i + 1; j < numRowsToReturn; j++) {
        // Compare the partition start row with the current row to check if the
        // partition has changed.
        if (partitionCompare(returningRows_[i], returningRows_[j])) {
          // Partition end found.
          partitionEndRow = j;
          break;
        }
      }
    }

    if (partitionEndRow == 0) {
      // This means we cannot ascertain this partition can complete in this
      // getOutput call. So defer until next getOutput call.
      break;
    }

    std::vector<char*> partitionRows;
    partitionRows.resize(partitionEndRow - partitionStartRow);
    // partitionIter_ retains the positional information across getOutput calls.
    data_->listRows(
        &partitionIter_,
        partitionEndRow - partitionStartRow,
        partitionRows.data());
    for (int w = 0; w < outputType_->size() - inputColumnsSize_; w++) {
      windowFunctions_[w]->resetPartition(partitionRows);
    }

    for (int j = partitionStartRow; j < partitionEndRow; j++) {
      for (int w = 0; w < outputType_->size() - inputColumnsSize_; w++) {
        // TODO : The end point is a row within this block. Do the cross block
        // output case.
        auto frameEndPoints = findFrameEndPoints(w, j);
        // TODO : Figure how to find the peers, frameStarts and frameEnds
        // buffers for the function invocation.
        VELOX_CHECK_EQ(frameEndPoints.second, j);
        windowFunctions_[w]->apply(
            nullptr, nullptr, nullptr, nullptr, windowFunctionOutputs[w]);
      }
    }

    i = partitionEndRow;
  }

  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    result->childAt(j) = windowFunctionOutputs[j - inputColumnsSize_];
  }

  numRowsReturned_ += numRowsToReturn;

  finished_ = (numRowsReturned_ == returningRows_.size());
  return result;
}

} // namespace facebook::velox::exec
