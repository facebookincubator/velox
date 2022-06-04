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
  initKeyInfo(
      windowNode->sources()[0]->outputType(),
      windowNode->sortingKeys(),
      windowNode->sortingOrders(),
      sortKeyInfo_);
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

    std::vector<ChannelIndex> argChannels;
    for (auto& arg : windowNodeFunction.functionCall->inputs()) {
      if (auto fae =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  arg)) {
        std::optional<ChannelIndex> argChannel =
            exprToChannel(fae.get(), windowNode->sources()[0]->outputType());
        VELOX_CHECK(
            argChannel.value() != kConstantChannel,
            "Window doesn't allow constant comparison keys");
        argChannels.push_back(argChannel.value());
      }
    }
    funcArgChannels_.push_back(argChannels);
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
    int32_t partitionStartRow,
    int32_t /*partitionEndRow*/,
    int32_t currentRow) {
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
  return std::make_pair(partitionStartRow, currentRow);
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

  std::vector<std::vector<VectorPtr>> functionArgVectors;
  functionArgVectors.resize(outputType_->size() - inputColumnsSize_);
  for (auto i = 0; i < funcArgChannels_.size(); i++) {
    const auto& funcChannels = funcArgChannels_[i];
    functionArgVectors[i].resize(funcChannels.size());
    for (auto j = 0; j < funcChannels.size(); j++) {
      functionArgVectors[i][j] = result->childAt(funcChannels[j]);
    }
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

  auto peerCompare = [&](const char* lhs, const char* rhs) -> bool {
    if (lhs == rhs) {
      return false;
    }
    for (auto& key : sortKeyInfo_) {
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

  int bufferEndRow = partitionStartRow_ + numRowsToReturn;
  bool lastOutputBlock = (bufferEndRow == returningRows_.size()) ? true : false;
  // Both partitionStartRow_ and partitionEndRow are the offsets in the
  // returningRows_ buffer.
  int partitionEndRow;
  for (int i = 0; i < numRowsToReturn;) {
    if ((numRowsReturned_ + i == returningRows_.size() - 1)) {
      // This is the last row of the input.
      // Fake partitionEnd row to be one row beyond the output block. This works
      // for this logic but is a bit hacky.
      partitionEndRow = partitionStartRow_ + 1;
    } else {
      // Lookahead and find partition end. The java code uses a binary search
      // style lookup instead of iterating over consecutive rows.
      partitionEndRow = 0;
      for (int j = partitionStartRow_ + 1; j < bufferEndRow; j++) {
        // Compare the partition start row with the current row to check if the
        // partition has changed.
        if (partitionCompare(
                returningRows_[partitionStartRow_], returningRows_[j])) {
          // Partition end found.
          partitionEndRow = j;
          break;
        }
      }
    }

    if (partitionEndRow == 0) {
      if (!lastOutputBlock) {
        // This means we cannot ascertain this partition can complete in this
        // getOutput call. So don't continue outputing rows in this output
        // block. Defer until next getOutput call. But correctly set the number
        // of rows returned in this block.
        numRowsToReturn = i + 1;
        break;
      } else {
        partitionEndRow = returningRows_.size();
      }
    }

    std::vector<char*> partitionRows;
    size_t partitionSize = partitionEndRow - partitionStartRow_;
    partitionRows.resize(partitionSize);
    // partitionIter_ retains the positional information across getOutput calls.
    data_->listRows(&partitionIter_, partitionSize, partitionRows.data());
    for (int w = 0; w < outputType_->size() - inputColumnsSize_; w++) {
      windowFunctions_[w]->resetPartition(partitionRows);
    }

    int peerStartRow = partitionStartRow_;
    int peerEndRow = peerStartRow + 1;
    for (int j = 0; j < partitionSize; j++) {
      int currentRow = partitionStartRow_ + j;
      // Find peers for the row. Peer computation happens only after the
      // rows of the previous peer are traversed. The peer computation
      // is done only until the end of the partition, so the values can
      // be safely used for traversal of input and output buffers.
      if (currentRow == partitionStartRow_ ||
          (currentRow == peerEndRow && currentRow != (partitionEndRow - 1))) {
        if (peerStartRow != partitionStartRow_) {
          peerStartRow = peerEndRow;
        }
        peerEndRow = peerStartRow + 1;
        for (int p = peerEndRow; p < partitionEndRow; p++) {
          if (peerCompare(
                  returningRows_[peerStartRow], returningRows_[peerEndRow])) {
            peerEndRow = p;
            break;
          }
        }
      }

      for (int w = 0; w < outputType_->size() - inputColumnsSize_; w++) {
        // As we ensure that all the rows of a partition fit into the output
        // block, then the peer and frame rows also will belong to this block.
        auto frameEndPoints = findFrameEndPoints(
            w, partitionStartRow_, partitionEndRow, currentRow);
        VELOX_CHECK_EQ(frameEndPoints.second, currentRow);
        // Find offsets for peer rows and frame rows in the current
        // partitionRowBuffer for the function.
        int32_t peerStartOffset = peerStartRow - partitionStartRow_;
        int32_t peerEndOffset = peerEndRow - partitionStartRow_;
        int32_t frameStartOffset = frameEndPoints.first - partitionStartRow_;
        int32_t frameEndOffset = frameEndPoints.second - partitionStartRow_;
        // This is the offset of the current row in the output buffer.
        int32_t currentRowOffset = i + j;
        windowFunctions_[w]->apply(
            peerStartOffset,
            peerEndOffset,
            frameStartOffset,
            frameEndOffset,
            currentRowOffset,
            functionArgVectors[w],
            windowFunctionOutputs[w]);
      }
    }

    i += partitionSize;
    partitionStartRow_ = partitionEndRow;
  }

  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    result->childAt(j) = windowFunctionOutputs[j - inputColumnsSize_];
  }

  numRowsReturned_ += numRowsToReturn;

  finished_ = (numRowsReturned_ == returningRows_.size());
  return result;
}

} // namespace facebook::velox::exec
