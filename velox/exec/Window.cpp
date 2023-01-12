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

namespace {
void initKeyInfo(
    const RowTypePtr& type,
    const std::vector<core::FieldAccessTypedExprPtr>& keys,
    const std::vector<core::SortOrder>& orders,
    std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo) {
  const core::SortOrder defaultPartitionSortOrder(true, true);

  keyInfo.reserve(keys.size());
  for (auto i = 0; i < keys.size(); ++i) {
    auto channel = exprToChannel(keys[i].get(), type);
    VELOX_CHECK(
        channel != kConstantChannel,
        "Window doesn't allow constant partition or sort keys");
    if (i < orders.size()) {
      keyInfo.push_back(std::make_pair(channel, orders[i]));
    } else {
      keyInfo.push_back(std::make_pair(channel, defaultPartitionSortOrder));
    }
  }
}

}; // namespace

bool isKPrecedingOrFollowing(core::WindowNode::BoundType bound) {
  return bound == core::WindowNode::BoundType::kPreceding ||
      bound == core::WindowNode::BoundType::kFollowing;
}

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
      outputBatchSizeInBytes_(
          driverCtx->queryConfig().preferredOutputBatchSize()),
      numInputColumns_(windowNode->sources()[0]->outputType()->size()),
      data_(std::make_unique<RowContainer>(
          windowNode->sources()[0]->outputType()->children(),
          pool())),
      decodedInputVectors_(numInputColumns_),
      stringAllocator_(pool()) {
  auto inputType = windowNode->sources()[0]->outputType();
  initKeyInfo(inputType, windowNode->partitionKeys(), {}, partitionKeyInfo_);
  initKeyInfo(
      inputType,
      windowNode->sortingKeys(),
      windowNode->sortingOrders(),
      sortKeyInfo_);
  allKeyInfo_.reserve(partitionKeyInfo_.size() + sortKeyInfo_.size());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), partitionKeyInfo_.begin(), partitionKeyInfo_.end());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), sortKeyInfo_.begin(), sortKeyInfo_.end());

  if (windowNode->windowFunctions()[0].frame.type ==
          core::WindowNode::WindowType::kRange &&
      (isKPrecedingOrFollowing(
           windowNode->windowFunctions()[0].frame.startType) ||
       isKPrecedingOrFollowing(
           windowNode->windowFunctions()[0].frame.endType))) {
    auto sortingKey = windowNode->sortingKeys()[0];
    // inputType needs to be replaced by sortingKey.type()
    sortingKeyIndex_ = exprToChannel(sortingKey.get(), inputType);
    sortingOrder_ = windowNode->sortingOrders()[0];
  } else {
    sortingKeyIndex_ = kConstantChannel;
  }

  std::vector<exec::RowColumn> inputColumns;
  for (int i = 0; i < inputType->children().size(); i++) {
    inputColumns.push_back(data_->columnAt(i));
  }
  // The WindowPartition is structured over all the input columns data.
  // Individual functions access its input argument column values from it.
  // The RowColumns are copied by the WindowPartition, so its fine to use
  // a local variable here.
  windowPartition_ =
      std::make_unique<WindowPartition>(inputColumns, inputType->children());

  createWindowFunctions(windowNode, inputType);
}

void Window::createWindowFunctions(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    const RowTypePtr& inputType) {
  auto constantArg = [&](const core::TypedExprPtr arg) -> const VectorPtr {
    if (auto typedExpr =
            dynamic_cast<const core::ConstantTypedExpr*>(arg.get())) {
      if (typedExpr->hasValueVector()) {
        return BaseVector::wrapInConstant(1, 0, typedExpr->valueVector());
      }
      if (typedExpr->value().isNull()) {
        return BaseVector::createNullConstant(typedExpr->type(), 1, pool());
      }
      return BaseVector::createConstant(typedExpr->value(), 1, pool());
    }
    return nullptr;
  };

  auto fieldArgToChannel =
      [&](const core::TypedExprPtr arg) -> std::optional<column_index_t> {
    if (arg) {
      std::optional<column_index_t> argChannel =
          exprToChannel(arg.get(), inputType);
      VELOX_CHECK(
          argChannel.value() != kConstantChannel,
          "Window doesn't allow constant arguments or frame end-points");
      return argChannel;
    }
    return std::nullopt;
  };

  for (const auto& windowNodeFunction : windowNode->windowFunctions()) {
    std::vector<WindowFunctionArg> functionArgs;
    functionArgs.reserve(windowNodeFunction.functionCall->inputs().size());
    for (auto& arg : windowNodeFunction.functionCall->inputs()) {
      if (auto constant = constantArg(arg)) {
        functionArgs.push_back({arg->type(), constant, std::nullopt});
      } else {
        functionArgs.push_back(
            {arg->type(), nullptr, fieldArgToChannel(arg).value()});
      }
    }

    windowFunctions_.push_back(WindowFunction::create(
        windowNodeFunction.functionCall->name(),
        functionArgs,
        windowNodeFunction.functionCall->type(),
        operatorCtx_->pool(),
        &stringAllocator_));

    auto createFrameChannelArg =
        [&](const core::TypedExprPtr& frame) -> FrameChannelArg {
      if (auto constant = constantArg(frame)) {
        auto constantOffset =
            constant->template as<ConstantVector<int64_t>>()->valueAt(0);
        VELOX_USER_CHECK_GE(
            constantOffset, 1, "k in frame bounds must be at least 1");
        return {kConstantChannel, constant};
      } else {
        return {
            fieldArgToChannel(frame).value(),
            BaseVector::create(BIGINT(), 0, operatorCtx_->pool())};
      }
    };

    bool hasKBasedStartBound =
        (windowNodeFunction.frame.startType ==
             core::WindowNode::BoundType::kPreceding ||
         windowNodeFunction.frame.startType ==
             core::WindowNode::BoundType::kFollowing);
    bool hasKBasedEndBound =
        (windowNodeFunction.frame.endType ==
             core::WindowNode::BoundType::kPreceding ||
         windowNodeFunction.frame.endType ==
             core::WindowNode::BoundType::kFollowing);

    WindowFrame windowFrame({
        windowNodeFunction.frame.type,
        windowNodeFunction.frame.startType,
        windowNodeFunction.frame.endType,
        (hasKBasedStartBound ? std::make_optional(createFrameChannelArg(
                                   windowNodeFunction.frame.startValue))
                             : std::nullopt),
        (hasKBasedEndBound ? std::make_optional(createFrameChannelArg(
                                 windowNodeFunction.frame.endValue))
                           : std::nullopt),
    });

    windowFrames_.push_back(windowFrame);
  }
}

void Window::addInput(RowVectorPtr input) {
  inputRows_.resize(input->size());

  for (auto col = 0; col < input->childrenSize(); ++col) {
    decodedInputVectors_[col].decode(*input->childAt(col), inputRows_);
  }

  // Add all the rows into the RowContainer.
  for (auto row = 0; row < input->size(); ++row) {
    char* newRow = data_->newRow();

    for (auto col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedInputVectors_[col], row, newRow, col);
    }
  }
  numRows_ += inputRows_.size();
}

inline bool Window::compareRowsWithKeys(
    const char* lhs,
    const char* rhs,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keys) {
  if (lhs == rhs) {
    return false;
  }
  for (auto& key : keys) {
    if (auto result = data_->compare(
            lhs,
            rhs,
            key.first,
            {key.second.isNullsFirst(), key.second.isAscending(), false})) {
      return result < 0;
    }
  }
  return false;
}

void Window::createPeerAndFrameBuffers() {
  // TODO: This computation needs to be revised. It only takes into account
  // the input columns size. We need to also account for the output columns.
  numRowsPerOutput_ = data_->estimatedNumRowsPerBatch(outputBatchSizeInBytes_);

  peerStartBuffer_ = AlignedBuffer::allocate<vector_size_t>(
      numRowsPerOutput_, operatorCtx_->pool());
  peerEndBuffer_ = AlignedBuffer::allocate<vector_size_t>(
      numRowsPerOutput_, operatorCtx_->pool());

  auto numFuncs = windowFunctions_.size();
  frameStartBuffers_.reserve(numFuncs);
  frameEndBuffers_.reserve(numFuncs);

  for (auto i = 0; i < numFuncs; i++) {
    BufferPtr frameStartBuffer = AlignedBuffer::allocate<vector_size_t>(
        numRowsPerOutput_, operatorCtx_->pool());
    BufferPtr frameEndBuffer = AlignedBuffer::allocate<vector_size_t>(
        numRowsPerOutput_, operatorCtx_->pool());
    frameStartBuffers_.push_back(frameStartBuffer);
    frameEndBuffers_.push_back(frameEndBuffer);
  }
}

void Window::computePartitionStartRows() {
  // Randomly assuming that max 10000 partitions are in the data.
  partitionStartRows_.reserve(numRows_);
  auto partitionCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, partitionKeyInfo_);
  };

  // Using a sequential traversal to find changing partitions.
  // This algorithm is inefficient and can be changed
  // i) Use a binary search kind of strategy.
  // ii) If we use a Hashtable instead of a full sort then the count
  // of rows in the partition can be directly used.
  partitionStartRows_.push_back(0);

  VELOX_CHECK_GT(sortedRows_.size(), 0);
  for (auto i = 1; i < sortedRows_.size(); i++) {
    if (partitionCompare(sortedRows_[i - 1], sortedRows_[i])) {
      partitionStartRows_.push_back(i);
    }
  }

  // Setting the startRow of the (last + 1) partition to be returningRows.size()
  // to help for last partition related calculations.
  partitionStartRows_.push_back(sortedRows_.size());
}

void Window::sortPartitions() {
  // This is a very inefficient but easy implementation to order the input rows
  // by partition keys + sort keys.
  // Sort the pointers to the rows in RowContainer (data_) instead of sorting
  // the rows.
  sortedRows_.resize(numRows_);
  RowContainerIterator iter;
  data_->listRows(&iter, numRows_, sortedRows_.data());

  std::sort(
      sortedRows_.begin(),
      sortedRows_.end(),
      [this](const char* leftRow, const char* rightRow) {
        return compareRowsWithKeys(leftRow, rightRow, allKeyInfo_);
      });

  computePartitionStartRows();

  currentPartition_ = 0;
}

void Window::noMoreInput() {
  Operator::noMoreInput();
  // No data.
  if (numRows_ == 0) {
    finished_ = true;
    return;
  }

  // At this point we have seen all the input rows. We can start
  // outputting rows now.
  // However, some preparation is needed. The rows should be
  // separated into partitions and sort by ORDER BY keys within
  // the partition. This will order the rows for getOutput().
  sortPartitions();
  createPeerAndFrameBuffers();
}

void Window::updateKRangeBoundsForPartition(
    const vector_size_t& partitionNumber) {
  auto totalPartitions = partitionStartRows_.size();
  auto firstPartitionRow = partitionStartRows_[partitionNumber];
  auto lastPartitionRow = (partitionNumber == totalPartitions - 1)
      ? numRows_ - 1
      : partitionStartRows_[partitionNumber + 1] - 1;
  auto numRows = lastPartitionRow - firstPartitionRow + 1;
  auto peerGroupCount = -1;
  auto peerStartRow = 0;
  auto peerEndRow = lastPartitionRow;

  auto peerCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, sortKeyInfo_);
  };

  rowToPeerGroup_.resize(numRows);
  rowToPeerGroupValue_.resize(numRows);
  // Assumption: sorting column is of type BIGINT().
  const int64_t* sortingColumn = NULL;

  if (sortingKeyIndex_ != kConstantChannel) {
    auto sortingColumnValues = BaseVector::create<FlatVector<int64_t>>(
        BIGINT(), numRows, operatorCtx_->pool());
    windowPartition_->extractColumn(
        sortingKeyIndex_, 0, numRows, 0, sortingColumnValues);
    sortingColumn = sortingColumnValues->values()->as<int64_t>();
  }

  for (auto i = firstPartitionRow, j = 0; i <= lastPartitionRow; i++, j++) {
    if (i == firstPartitionRow || i >= peerEndRow) {
      peerStartRow = i;
      peerEndRow = i;
      while (peerEndRow <= lastPartitionRow) {
        if (peerCompare(sortedRows_[peerStartRow], sortedRows_[peerEndRow])) {
          break;
        }
        peerEndRow++;
      }

      peerGroupCount++;
      partitionToPeerGroupBounds_[peerGroupCount] = {
          peerStartRow - firstPartitionRow, peerEndRow - 1 - firstPartitionRow};
    }

    rowToPeerGroup_[j] = peerGroupCount;
    if (sortingKeyIndex_ != kConstantChannel) {
      rowToPeerGroupValue_[j] = sortingColumn[j];
    }
  }
}

void Window::callResetPartition(vector_size_t partitionNumber) {
  auto partitionSize = partitionStartRows_[partitionNumber + 1] -
      partitionStartRows_[partitionNumber];
  auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[partitionNumber], partitionSize);
  windowPartition_->resetPartition(partition);
  for (int i = 0; i < windowFunctions_.size(); i++) {
    windowFunctions_[i]->resetPartition(windowPartition_.get());
  }

  updateKRangeBoundsForPartition(partitionNumber);
}

void Window::callApplyForPartitionRows(
    vector_size_t startRow,
    vector_size_t endRow,
    const std::vector<VectorPtr>& result,
    vector_size_t resultOffset) {
  if (partitionStartRows_[currentPartition_] == startRow) {
    callResetPartition(currentPartition_);
    partitionOffset_ = 0;
  }

  vector_size_t numRows = endRow - startRow;
  vector_size_t numFuncs = windowFunctions_.size();

  // Size buffers for the call to WindowFunction::apply.
  auto bufferSize = numRows * sizeof(vector_size_t);
  peerStartBuffer_->setSize(bufferSize);
  peerEndBuffer_->setSize(bufferSize);
  auto rawPeerStarts = peerStartBuffer_->asMutable<vector_size_t>();
  auto rawPeerEnds = peerEndBuffer_->asMutable<vector_size_t>();

  std::vector<vector_size_t*> rawFrameStartBuffers;
  std::vector<vector_size_t*> rawFrameEndBuffers;
  rawFrameStartBuffers.reserve(numFuncs);
  rawFrameEndBuffers.reserve(numFuncs);
  for (auto w = 0; w < numFuncs; w++) {
    frameStartBuffers_[w]->setSize(bufferSize);
    frameEndBuffers_[w]->setSize(bufferSize);

    auto rawFrameStartBuffer =
        frameStartBuffers_[w]->asMutable<vector_size_t>();
    auto rawFrameEndBuffer = frameEndBuffers_[w]->asMutable<vector_size_t>();
    rawFrameStartBuffers.push_back(rawFrameStartBuffer);
    rawFrameEndBuffers.push_back(rawFrameEndBuffer);
  }

  auto findPeerGroup = [&](const vector_size_t& row) -> vector_size_t {
    auto peerGroups = partitionToPeerGroupBounds_;
    auto findResult = std::find_if(
        peerGroups.begin(),
        peerGroups.end(),
        [&](std::pair<const int, std::pair<int, int>> peerGroup) {
          return peerGroup.second.first <= row &&
              peerGroup.second.second >= row;
        });
    if (findResult == std::end(partitionToPeerGroupBounds_)) {
      return -1;
    }
    return findResult->first;
  };

  auto peerCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, sortKeyInfo_);
  };

  auto firstPartitionRow = partitionStartRows_[currentPartition_];
  auto lastPartitionRow = partitionStartRows_[currentPartition_ + 1] - 1;
  auto rowToPeerGroup = rowToPeerGroup_;
  auto rowToPeerGroupValue = rowToPeerGroupValue_;
  auto partitionPeerGroups = partitionToPeerGroupBounds_;
  auto currentPartition = currentPartition_;
  auto partitionRowCount = windowPartition_->numRows();
  vector_size_t peerGroupCount = partitionPeerGroups.size();
  auto startingPeerGroup = findPeerGroup(partitionOffset_);
  VELOX_CHECK(startingPeerGroup != -1, "Invalid peer group");

  for (auto i = startingPeerGroup, j = 0; j < numRows; i++) {
    auto peerGroupStart = partitionPeerGroups[i].first;
    auto peerGroupEnd = partitionPeerGroups[i].second;

    for (auto k = std::max(peerGroupStart, partitionOffset_); k <= peerGroupEnd;
         k++, j++) {
      rawPeerStarts[j] = peerGroupStart;
      rawPeerEnds[j] = peerGroupEnd;
    }
  }

  auto readFrameValues = [&](const FrameChannelArg& frameArg) -> void {
    frameArg.value->resize(numRows);
    windowPartition_->extractColumn(
        frameArg.index, partitionOffset_, numRows, 0, frameArg.value);
  };

  auto computeKRangeFrameBound =
      [&](const vector_size_t& peerGroup,
          const vector_size_t& offset,
          const bool isStartBound,
          const bool isKPreceding,
          const vector_size_t& sortingColumnValue) -> vector_size_t {
    // The limiting value, after which all subsequent values belong to the
    // window frame.
    auto limitValue = 0;
    // When the sorting order is not ascending, the limitValue calculation
    // follows the sorted order and the offset value added for kPreceding and
    // kFollowing bounds are reversed.
    if (isKPreceding) {
      limitValue = sortingOrder_.isAscending() ? sortingColumnValue - offset
                                               : sortingColumnValue + offset;
    } else {
      limitValue = sortingOrder_.isAscending() ? sortingColumnValue + offset
                                               : sortingColumnValue - offset;
    }

    // Finds a value greater than or equal to the limiting value when sorting
    // order is ascending, finds a value not greater than the limiting value
    // otherwise.
    auto findValue = sortingOrder_.isAscending()
        ? std::lower_bound(
              rowToPeerGroupValue.begin(),
              rowToPeerGroupValue.end(),
              limitValue)
        : std::lower_bound(
              rowToPeerGroupValue.begin(),
              rowToPeerGroupValue.end(),
              limitValue,
              std::greater<int64_t>());

    // Row number of lower bound.
    auto idx = std::min(
        vector_size_t(findValue - rowToPeerGroupValue.begin()),
        partitionRowCount - 1);
    auto lowerBoundPeerGroup = rowToPeerGroup[idx];
    auto lowerBoundValue = rowToPeerGroupValue[idx];
    auto previousPeerGroup = peerGroup;
    auto resultPeerGroup = 0;

    if (sortingOrder_.isAscending()) {
      if (lowerBoundValue == limitValue) {
        resultPeerGroup = lowerBoundPeerGroup;
      } else {
        // resultValue > limitValue.
        resultPeerGroup = isKPreceding
            ? lowerBoundPeerGroup
            : std::max(lowerBoundPeerGroup - 1, peerGroup);
      }
    } else {
      if (lowerBoundValue == limitValue) {
        resultPeerGroup = lowerBoundPeerGroup;
      } else {
        // Because lower_bound for array sorted in descending order returns the
        // first value not greater than the limit, the bound for kFollowing
        // frames needs to be adjusted based on the positions of limitValue and
        // lowerBoundValue.
        if (isKPreceding) {
          resultPeerGroup = lowerBoundPeerGroup;
        } else if (lowerBoundValue > limitValue) {
          resultPeerGroup =
              std::min(lowerBoundPeerGroup + 1, peerGroupCount - 1);
        } else {
          resultPeerGroup = std::max(lowerBoundPeerGroup - 1, peerGroup);
        }
      }
    }

    auto resultBounds = partitionPeerGroups[resultPeerGroup];
    return isStartBound ? resultBounds.first : resultBounds.second;
  };

  auto updateKBoundFrameLimits =
      [&](WindowFrame windowFrame,
          vector_size_t* rawFrameBound,
          const bool isStartBound,
          const bool isKPreceding,
          const std::optional<FrameChannelArg>& frameArg =
              std::nullopt) -> void {
    if (frameArg->index == kConstantChannel) {
      auto constantOffset =
          frameArg->value->template as<ConstantVector<int64_t>>()->valueAt(0);
      for (auto i = 0; i < numRows; i++) {
        rawFrameBound[i] = computeKRangeFrameBound(
            rowToPeerGroup[i + partitionOffset_],
            constantOffset,
            isStartBound,
            isKPreceding,
            rowToPeerGroupValue[i + partitionOffset_]);
      }
    } else {
      readFrameValues(frameArg.value());
      auto offsetsVector = frameArg->value->asFlatVector<int64_t>();
      for (auto i = 0; i < numRows; i++) {
        rawFrameBound[i] = computeKRangeFrameBound(
            rowToPeerGroup[i + partitionOffset_],
            offsetsVector->valueAt(i),
            isStartBound,
            isKPreceding,
            rowToPeerGroupValue[i + partitionOffset_]);
      }
    }
  };

  auto updateFrameBounds = [&](const vector_size_t& functionIdx,
                               vector_size_t* rawFrameBound,
                               const bool isStartBound,
                               const std::optional<FrameChannelArg>& frameArg =
                                   std::nullopt) -> void {
    auto windowFrame = windowFrames_[functionIdx];
    auto frameBoundType =
        isStartBound ? windowFrame.startType : windowFrame.endType;
    switch (frameBoundType) {
      case core::WindowNode::BoundType::kUnboundedPreceding:
        std::memset(rawFrameBound, 0, numRows * sizeof(vector_size_t));
        break;
      case core::WindowNode::BoundType::kUnboundedFollowing:
        std::fill_n(
            rawFrameBound, numRows, lastPartitionRow - firstPartitionRow);
        break;
      case core::WindowNode::BoundType::kCurrentRow: {
        if (windowFrame.type == core::WindowNode::WindowType::kRange) {
          vector_size_t* rawPeerBuffer =
              isStartBound ? rawPeerStarts : rawPeerEnds;
          std::copy(rawPeerBuffer, rawPeerBuffer + numRows, rawFrameBound);
        } else {
          // Fills the frameBound buffer with increasing value of row indices
          // (corresponding to CURRENT ROW) from the startRow of the current
          // output buffer. The startRow has to be adjusted relative to the
          // partition start row.
          std::iota(
              rawFrameBound,
              rawFrameBound + numRows,
              startRow - firstPartitionRow);
        }
        break;
      }
      case core::WindowNode::BoundType::kPreceding: {
        if (windowFrame.type == core::WindowNode::WindowType::kRange) {
          updateKBoundFrameLimits(
              windowFrame, rawFrameBound, isStartBound, true, frameArg);
        } else {
          VELOX_NYI("k preceding/following frames not supported in ROWS mode");
        }
        break;
      }
      case core::WindowNode::BoundType::kFollowing: {
        if (windowFrame.type == core::WindowNode::WindowType::kRange) {
          updateKBoundFrameLimits(
              windowFrame, rawFrameBound, isStartBound, false, frameArg);
        } else {
          VELOX_NYI("k preceding/following frames not supported in ROWS mode");
        }
        break;
      }
      default:
        VELOX_USER_FAIL("Invalid frame bound type");
    }
  };

  for (auto i = 0; i < numFuncs; i++) {
    updateFrameBounds(
        i, rawFrameStartBuffers[i], true, windowFrames_[i].startFrameArg);
    updateFrameBounds(
        i, rawFrameEndBuffers[i], false, windowFrames_[i].endFrameArg);
  }

  // Invoke the apply method for the WindowFunctions.
  for (auto w = 0; w < numFuncs; w++) {
    windowFunctions_[w]->apply(
        peerStartBuffer_,
        peerEndBuffer_,
        frameStartBuffers_[w],
        frameEndBuffers_[w],
        resultOffset,
        result[w]);
  }

  numProcessedRows_ += numRows;
  partitionOffset_ += numRows;
  if (endRow == partitionStartRows_[currentPartition_ + 1]) {
    currentPartition_++;
  }
}

void Window::callApplyLoop(
    vector_size_t numOutputRows,
    const std::vector<VectorPtr>& windowOutputs) {
  // Compute outputs by traversing as many partitions as possible. This
  // logic takes care of partial partitions output also.

  vector_size_t resultIndex = 0;
  vector_size_t numOutputRowsLeft = numOutputRows;
  while (numOutputRowsLeft > 0) {
    auto rowsForCurrentPartition =
        partitionStartRows_[currentPartition_ + 1] - numProcessedRows_;
    if (rowsForCurrentPartition <= numOutputRowsLeft) {
      // Current partition can fit completely in the output buffer.
      // So output all its rows.
      callApplyForPartitionRows(
          numProcessedRows_,
          numProcessedRows_ + rowsForCurrentPartition,
          windowOutputs,
          resultIndex);
      resultIndex += rowsForCurrentPartition;
      numOutputRowsLeft -= rowsForCurrentPartition;
    } else {
      // Current partition can fit only partially in the output buffer.
      // Call apply for the rows that can fit in the buffer and break from
      // outputting.
      callApplyForPartitionRows(
          numProcessedRows_,
          numProcessedRows_ + numOutputRowsLeft,
          windowOutputs,
          resultIndex);
      numOutputRowsLeft = 0;
      break;
    }
  }
}

RowVectorPtr Window::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  auto numRowsLeft = numRows_ - numProcessedRows_;
  auto numOutputRows = std::min(numRowsPerOutput_, numRowsLeft);
  auto result = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(outputType_, numOutputRows, operatorCtx_->pool()));

  // Set all passthrough input columns.
  for (int i = 0; i < numInputColumns_; ++i) {
    data_->extractColumn(
        sortedRows_.data() + numProcessedRows_,
        numOutputRows,
        i,
        result->childAt(i));
  }

  // Construct vectors for the window function output columns.
  std::vector<VectorPtr> windowOutputs;
  windowOutputs.reserve(windowFunctions_.size());
  for (int i = numInputColumns_; i < outputType_->size(); i++) {
    auto output = BaseVector::create(
        outputType_->childAt(i), numOutputRows, operatorCtx_->pool());
    windowOutputs.emplace_back(std::move(output));
  }

  // Compute the output values of window functions.
  callApplyLoop(numOutputRows, windowOutputs);

  for (int j = numInputColumns_; j < outputType_->size(); j++) {
    result->childAt(j) = windowOutputs[j - numInputColumns_];
  }

  finished_ = (numProcessedRows_ == sortedRows_.size());
  return result;
}

} // namespace facebook::velox::exec
