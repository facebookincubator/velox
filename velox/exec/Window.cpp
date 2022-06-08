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

    std::vector<exec::RowColumn> argColumns;
    for (auto& arg : windowNodeFunction.functionCall->inputs()) {
      if (auto fae =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  arg)) {
        std::optional<ChannelIndex> argChannel =
            exprToChannel(fae.get(), windowNode->sources()[0]->outputType());
        VELOX_CHECK(
            argChannel.value() != kConstantChannel,
            "Window doesn't allow constant comparison keys");
        argColumns.push_back(data_->columnAt(argChannel.value()));
      }
    }
    windowFunctions_.push_back(WindowFunction::create(
        windowNodeFunction.functionCall->name(),
        argColumns,
        argTypes,
        resultType));

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

  numRowsPerOutput_ = data_->estimatedNumRowsPerBatch(kBatchSizeInBytes);
  // Randomly assuming that max 10000 partitions are in the data.
  // Find the number of rows per partition (in order as they are present in
  // returningRows_).
  partitionStartRows_.reserve(10000);
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
  // Using a sequential traversal to find changing partitions.
  // This algorithm can be changed to use a binary search kind of strategy. Or
  // if we use a HashTable for grouping then the count of rows in each group can
  // be used directly.
  int j = 0;
  partitionStartRows_[0] = 0;
  // We directly return above if there are no rows. So we can safely assume here
  // that we have atleast 1 input row.
  for (int64_t i = 1; i < returningRows_.size(); i++) {
    if (partitionCompare(returningRows_[i - 1], returningRows_[i])) {
      j++;
      partitionStartRows_[j] = i;
    }
  }
  currentPartitionIndex_ = 0;
  numberOfPartitions_ = j + 1;
  // Setting the startRow of the (last + 1) partition to be returningRows.size()
  // to help for last partition related calculations.
  partitionStartRows_[numberOfPartitions_] = returningRows_.size();
}

std::pair<int32_t, int32_t> Window::findFrameEndPoints(
    int32_t idx,
    int32_t partitionStartRow,
    int32_t /*partitionEndRow*/,
    int32_t currentRow) {
  // TODO : We handle only the default window frame in this code. Add support
  // for all window frames subsequently.
  VELOX_CHECK_EQ(
      windowFrames_[idx].windowType_, core::WindowNode::WindowType::kRange);
  VELOX_CHECK_EQ(
      windowFrames_[idx].startBoundType_,
      core::WindowNode::BoundType::kUnboundedPreceding);
  VELOX_CHECK_EQ(
      windowFrames_[idx].endBoundType_,
      core::WindowNode::BoundType::kCurrentRow);
  VELOX_CHECK(!windowFrames_[idx].startChannel_.has_value());
  VELOX_CHECK(!windowFrames_[idx].endChannel_.has_value());

  // Default window frame is Range UNBOUNDED PRECEDING CURRENT ROW.
  return std::make_pair(partitionStartRow, currentRow);
}

std::pair<RowVectorPtr, std::vector<VectorPtr>> Window::setupBufferForOutput(
    size_t noRows) {
  auto result = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(outputType_, noRows, operatorCtx_->pool()));

  // Set values of all output columns corresponding to the input columns.
  // ExtractColumn might be doing a copy. Try to avoid that.
  for (int i = 0; i < inputColumnsSize_; ++i) {
    data_->extractColumn(
        returningRows_.data() + numRowsReturned_,
        noRows,
        i,
        result->childAt(i));
  }

  std::vector<VectorPtr> windowFunctionOutputs;
  windowFunctionOutputs.reserve(outputType_->size() - inputColumnsSize_);
  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    auto windowOutputFlatVector = BaseVector::create(
        outputType_->childAt(j), noRows, operatorCtx_->pool());
    windowFunctionOutputs.insert(
        windowFunctionOutputs.cend(), std::move(windowOutputFlatVector));
  }
  return std::make_pair(result, windowFunctionOutputs);
}

void Window::callResetPartition(size_t idx) {
  size_t partitionSize =
      partitionStartRows_[idx + 1] - partitionStartRows_[idx];
  partitionRows_.resize(partitionSize);
  // partitionIter_ retains the positional information across getOutput calls.
  data_->listRows(&partitionIter_, partitionSize, partitionRows_.data());
  for (int w = 0; w < outputType_->size() - inputColumnsSize_; w++) {
    windowFunctions_[w]->resetPartition(partitionRows_);
  }
}

void Window::outputCurrentPartition(
    size_t startRow,
    size_t endRow,
    const std::vector<VectorPtr>& windowFunctionOutputs,
    const size_t bufferIndex) {
  bool isFirstRowOfPartition =
      (partitionStartRows_[currentPartitionIndex_] == startRow) ? true : false;
  if (isFirstRowOfPartition) {
    callResetPartition(currentPartitionIndex_);
  }

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

  // TODO: Is pool parameter correct ? Can we allocate in a pool that only lasts
  // for this function call ? Do we need to release the buffers ?
  size_t numRows = endRow - startRow;
  size_t numFuncs = outputType_->size() - inputColumnsSize_;
  BufferPtr allFuncsPeerStartBuffer[numFuncs];
  BufferPtr allFuncsPeerEndBuffer[numFuncs];
  BufferPtr allFuncsFrameStartBuffer[numFuncs];
  BufferPtr allFuncsFrameEndBuffer[numFuncs];
  size_t* allFuncsRawPeerStartBuffer[numFuncs];
  size_t* allFuncsRawPeerEndBuffer[numFuncs];
  size_t* allFuncsRawFrameStartBuffer[numFuncs];
  size_t* allFuncsRawFrameEndBuffer[numFuncs];

  for (int w = 0; w < numFuncs; w++) {
    BufferPtr peerStartBuffer =
        AlignedBuffer::allocate<size_t>(numRows, operatorCtx_->pool());
    BufferPtr peerEndBuffer =
        AlignedBuffer::allocate<size_t>(numRows, operatorCtx_->pool());
    BufferPtr frameStartBuffer =
        AlignedBuffer::allocate<size_t>(numRows, operatorCtx_->pool());
    BufferPtr frameEndBuffer =
        AlignedBuffer::allocate<size_t>(numRows, operatorCtx_->pool());
    peerStartBuffer->setSize(numRows);
    peerEndBuffer->setSize(numRows);
    frameStartBuffer->setSize(numRows);
    frameEndBuffer->setSize(numRows);
    allFuncsPeerStartBuffer[w] = peerStartBuffer;
    allFuncsPeerEndBuffer[w] = peerEndBuffer;
    allFuncsFrameStartBuffer[w] = frameStartBuffer;
    allFuncsFrameEndBuffer[w] = frameEndBuffer;

    auto rawPeerStartBuffer = peerStartBuffer->asMutable<size_t>();
    auto rawPeerEndBuffer = peerEndBuffer->asMutable<size_t>();
    auto rawFrameStartBuffer = frameStartBuffer->asMutable<size_t>();
    auto rawFrameEndBuffer = frameEndBuffer->asMutable<size_t>();
    allFuncsRawPeerStartBuffer[w] = rawPeerStartBuffer;
    allFuncsRawPeerEndBuffer[w] = rawPeerEndBuffer;
    allFuncsRawFrameStartBuffer[w] = rawFrameStartBuffer;
    allFuncsRawFrameEndBuffer[w] = rawFrameEndBuffer;
  }

  size_t lastPartitionRow = partitionStartRows_[currentPartitionIndex_ + 1] - 1;
  size_t firstPartitionRow = partitionStartRows_[currentPartitionIndex_];
  for (int i = startRow, j = 0; i < endRow; i++, j++) {
    // Compute the next peerStart and peerEnd rows (if this is the first row
    // of the partition or we are past previous peerGroup).
    if (i == partitionStartRows_[currentPartitionIndex_] || i >= peerEndRow_) {
      peerStartRow_ = i;
      peerEndRow_ = i;
      while (peerEndRow_ <= lastPartitionRow) {
        if (peerCompare(
                returningRows_[peerStartRow_], returningRows_[peerEndRow_])) {
          break;
        }
        peerEndRow_++;
      }
    }
    for (int w = 0; w < numFuncs; w++) {
      // The peer and frame values set in the input buffers to the function
      // are the offsets within the current partition.
      allFuncsRawPeerStartBuffer[w][j] = peerStartRow_ - firstPartitionRow;
      allFuncsRawPeerEndBuffer[w][j] = peerEndRow_ - 1 - firstPartitionRow;

      auto frameEndPoints = findFrameEndPoints(
          w, partitionStartRows_[currentPartitionIndex_], endRow, i);
      VELOX_CHECK_EQ(frameEndPoints.second, i);
      allFuncsRawFrameStartBuffer[w][j] =
          frameEndPoints.first - firstPartitionRow;
      allFuncsRawFrameEndBuffer[w][j] =
          frameEndPoints.second - firstPartitionRow;
    }
  }
  // Invoke the apply method for the WindowFunctions
  for (int w = 0; w < numFuncs; w++) {
    windowFunctions_[w]->apply(
        allFuncsPeerStartBuffer[w],
        allFuncsPeerEndBuffer[w],
        allFuncsFrameStartBuffer[w],
        allFuncsFrameEndBuffer[w],
        startRow - firstPartitionRow,
        bufferIndex,
        windowFunctionOutputs[w]);
  }

  numRowsReturned_ += numRows;
  if (endRow == partitionStartRows_[currentPartitionIndex_ + 1]) {
    currentPartitionIndex_++;
  }
}

RowVectorPtr Window::getOutput() {
  if (finished_ || !noMoreInput_ || returningRows_.size() == numRowsReturned_) {
    return nullptr;
  }

  size_t rowsForOutput =
      std::min(numRowsPerOutput_, returningRows_.size() - numRowsReturned_);
  auto result = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(outputType_, rowsForOutput, operatorCtx_->pool()));

  // Set values of all output columns corresponding to the input columns.
  // ExtractColumn might be doing a copy. Try to avoid that.
  for (int i = 0; i < inputColumnsSize_; ++i) {
    data_->extractColumn(
        returningRows_.data() + numRowsReturned_,
        rowsForOutput,
        i,
        result->childAt(i));
  }

  std::vector<VectorPtr> windowFunctionOutputs;
  windowFunctionOutputs.reserve(outputType_->size() - inputColumnsSize_);
  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    auto windowOutputFlatVector = BaseVector::create(
        outputType_->childAt(j), rowsForOutput, operatorCtx_->pool());
    windowFunctionOutputs.insert(
        windowFunctionOutputs.cend(), std::move(windowOutputFlatVector));
  }

  size_t rowsLeftForOutput = rowsForOutput;
  int bufferRowIndex = 0;
  while (rowsLeftForOutput > 0 &&
         currentPartitionIndex_ < numberOfPartitions_) {
    size_t rowsForCurrentPartition =
        partitionStartRows_[currentPartitionIndex_ + 1] - numRowsReturned_;
    if (rowsForCurrentPartition <= rowsLeftForOutput) {
      // Current partition can fit completely in the output buffer.
      // The currentPartitionIndex_ will be updated in outputCurrentPartition,
      // so no need to update it here.
      outputCurrentPartition(
          numRowsReturned_,
          numRowsReturned_ + rowsForCurrentPartition,
          windowFunctionOutputs,
          bufferRowIndex);
      bufferRowIndex += rowsForCurrentPartition;
      rowsLeftForOutput -= rowsForCurrentPartition;
    } else {
      // Current partition can fit partially in the output buffer.
      // Output "rowsLeftForOutput" number of rows and break from the outputing
      // rows.
      outputCurrentPartition(
          numRowsReturned_,
          numRowsReturned_ + rowsLeftForOutput,
          windowFunctionOutputs,
          bufferRowIndex);
      break;
    }
  }

  for (int j = inputColumnsSize_; j < outputType_->size(); j++) {
    result->childAt(j) = windowFunctionOutputs[j - inputColumnsSize_];
  }

  finished_ = (numRowsReturned_ == returningRows_.size());
  return result;
}

} // namespace facebook::velox::exec
