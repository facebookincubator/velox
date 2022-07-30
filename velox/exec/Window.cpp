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
  core::SortOrder defaultPartitionSortOrder(true, true);

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

void checkDefaultWindowFrame(const core::WindowNode::Function& windowFunction) {
  VELOX_CHECK_EQ(
      windowFunction.frame.type, core::WindowNode::WindowType::kRange);
  VELOX_CHECK_EQ(
      windowFunction.frame.startType,
      core::WindowNode::BoundType::kUnboundedPreceding);
  VELOX_CHECK_EQ(
      windowFunction.frame.endType, core::WindowNode::BoundType::kCurrentRow);
  VELOX_CHECK_EQ(windowFunction.frame.startValue, nullptr);
  VELOX_CHECK_EQ(windowFunction.frame.endValue, nullptr);
}

}; // namespace

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
          operatorCtx_->mappedMemory())),
      decodedInputVectors_(numInputColumns_) {
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

  std::vector<exec::RowColumn> inputColumns;
  for (int i = 0; i < inputType->children().size(); i++) {
    inputColumns.push_back(data_->columnAt(i));
  }
  // The WindowPartition is structured over all the input columns.
  // Individual functions access all its input argument columns from it.
  // The RowColumns are copied by the WindowPartition, so its fine to use
  // a local variable here.
  windowPartition_ = std::make_unique<WindowPartition>(
      inputColumns, inputType->children(), pool());

  createWindowFunctions(windowNode, inputType);
}

void Window::createWindowFunctions(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    const RowTypePtr& inputType) {
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
    std::vector<TypePtr> argTypes;
    std::vector<column_index_t> argIndices;
    argTypes.reserve(windowNodeFunction.functionCall->inputs().size());
    argIndices.reserve(windowNodeFunction.functionCall->inputs().size());
    for (auto& arg : windowNodeFunction.functionCall->inputs()) {
      argTypes.push_back(arg->type());
      argIndices.push_back(fieldArgToChannel(arg).value());
    }

    windowFunctions_.push_back(WindowFunction::create(
        windowNodeFunction.functionCall->name(),
        argTypes,
        argIndices,
        windowNodeFunction.functionCall->type(),
        operatorCtx_->pool()));

    checkDefaultWindowFrame(windowNodeFunction);

    windowFrames_.push_back(
        {windowNodeFunction.frame.type,
         windowNodeFunction.frame.startType,
         windowNodeFunction.frame.endType,
         fieldArgToChannel(windowNodeFunction.frame.startValue),
         fieldArgToChannel(windowNodeFunction.frame.endValue)});
  }
}

void Window::addInput(RowVectorPtr input) {
  // TODO : Is resize() or resizeAll() needed here ?
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

// This is a common function used to compare rows in the RowContainer
// wrt a set of keys (could be partition keys, order by keys or their
// combination).
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

// This function is used to initialize the peer Buffers and frame buffers that
// are used in window function invocations.
void Window::createPeerAndFrameBuffers() {
  // TODO: This computation needs to be revised. It only takes into account
  // the input columns size. We need to also account for the output columns.
  numRowsPerOutput_ = data_->estimatedNumRowsPerBatch(outputBatchSizeInBytes_);

  peerStartBuffer_ = AlignedBuffer::allocate<vector_size_t>(
      numRowsPerOutput_, operatorCtx_->pool());
  peerEndBuffer_ = AlignedBuffer::allocate<vector_size_t>(
      numRowsPerOutput_, operatorCtx_->pool());
  rawPeerStartBuffer_ = peerStartBuffer_->asMutable<vector_size_t>();
  rawPeerEndBuffer_ = peerEndBuffer_->asMutable<vector_size_t>();

  auto numFuncs = windowFunctions_.size();
  allFuncsFrameStartBuffer_.reserve(numFuncs);
  allFuncsFrameEndBuffer_.reserve(numFuncs);
  allFuncsRawFrameStartBuffer_.reserve(numFuncs);
  allFuncsRawFrameEndBuffer_.reserve(numFuncs);

  for (auto i = 0; i < numFuncs; i++) {
    BufferPtr frameStartBuffer = AlignedBuffer::allocate<vector_size_t>(
        numRowsPerOutput_, operatorCtx_->pool());
    BufferPtr frameEndBuffer = AlignedBuffer::allocate<vector_size_t>(
        numRowsPerOutput_, operatorCtx_->pool());
    allFuncsFrameStartBuffer_.push_back(frameStartBuffer);
    allFuncsFrameEndBuffer_.push_back(frameEndBuffer);

    auto rawFrameStartBuffer = frameStartBuffer->asMutable<vector_size_t>();
    auto rawFrameEndBuffer = frameEndBuffer->asMutable<vector_size_t>();
    allFuncsRawFrameStartBuffer_.push_back(rawFrameStartBuffer);
    allFuncsRawFrameEndBuffer_.push_back(rawFrameEndBuffer);
  }
}

// Find the starting row number for each partition (in the order they
// are in sortedRows_). Having this array handy makes the getOutput
// logic simpler.
void Window::computePartitionStartRows() {
  // Randomly assuming that max 10000 partitions are in the data.
  partitionStartRows_.reserve(10000);
  auto partitionCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, partitionKeyInfo_);
  };

  // Using a sequential traversal to find changing partitions.
  // This algorithm is inefficient and can be changed
  // i) Use a binary search kind of strategy.
  // ii) If we use a Hashtable instead of a full sort then the count
  // of rows in the partition can be directly used.
  partitionStartRows_.push_back(0);
  // The invoker of this function has a fast-path if there are no
  // rows. So we should have atleast one input row at this point.
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

// This function orders the input rows by (partition keys + order by keys).
// Doing so orders all rows of a partition adjacent to each other
// and sorted by the ORDER BY clause. This is the order in which the rows
// will be output by this operator.
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

  // Compute an array of the start rows of each partition. This is used to
  // simplify further computations.
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

// This function is to find the frame end points for the current row
// being output.
// @param idx : Index of the window function whose frame we are
// computing.
// @param partitionStartRow : Index of the start row of the current
// partition being output.
// @param partitionEndRow : Index of the end row of the current
// partition being output.
// @param currentRow : Index of the current row.
// partitionStartRow, partitionEndRow and currentRow are indexes in
// the sortedRows_ ordering of input rows.
std::pair<vector_size_t, vector_size_t> Window::findFrameEndPoints(
    vector_size_t /*idx*/,
    vector_size_t partitionStartRow,
    vector_size_t /*partitionEndRow*/,
    vector_size_t currentRow) {
  // TODO : We handle only the default window frame in this code. Add support
  // for all window frames subsequently.

  // Default window frame is Range UNBOUNDED PRECEDING CURRENT ROW.
  return std::make_pair(partitionStartRow, currentRow);
}

void Window::callResetPartition(vector_size_t idx) {
  auto partitionSize = partitionStartRows_[idx + 1] - partitionStartRows_[idx];
  auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[idx], partitionSize);
  windowPartition_->resetPartition(partition);
  for (int i = 0; i < windowFunctions_.size(); i++) {
    windowFunctions_[i]->resetPartition(windowPartition_.get());
  }
}

// Call WindowFunction::apply for the rows between startRow and endRow in
// the sortedRows_ ordering of the data_ RowContainer.
void Window::callApplyForPartitionRows(
    vector_size_t startRow,
    vector_size_t endRow,
    const std::vector<VectorPtr>& windowFunctionOutputs,
    vector_size_t resultIndex) {
  if (partitionStartRows_[currentPartition_] == startRow) {
    callResetPartition(currentPartition_);
  }

  auto peerCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, sortKeyInfo_);
  };

  vector_size_t numRows = endRow - startRow;
  vector_size_t numFuncs = windowFunctions_.size();

  // Size buffers for the call to WindowFunction::apply.
  peerStartBuffer_->setSize(numRows);
  peerEndBuffer_->setSize(numRows);
  for (auto w = 0; w < numFuncs; w++) {
    allFuncsFrameStartBuffer_[w]->setSize(numRows);
    allFuncsFrameEndBuffer_[w]->setSize(numRows);
  }

  auto firstPartitionRow = partitionStartRows_[currentPartition_];
  auto lastPartitionRow = partitionStartRows_[currentPartition_ + 1] - 1;
  for (auto i = startRow, j = 0; i < endRow; i++, j++) {
    // When traversing input partition rows, the peers are the rows
    // with the same values for the ORDER BY clause. These rows
    // are equal in some ways and affect the results of ranking functions.
    // This logic exploits the fact that all rows between the peerStartRow_
    // and peerEndRow_ have the same values for peerStartRow_ and peerEndRow_.
    // So we can compute them just once and reuse across the rows in that peer
    // interval.

    // This logic computes the next values of peerStart and peerEnd rows.
    // This needs to happen for the  first row of the partition or
    // if we are past the previous peerGroup.
    if (i == firstPartitionRow || i >= peerEndRow_) {
      peerStartRow_ = i;
      peerEndRow_ = i;
      while (peerEndRow_ <= lastPartitionRow) {
        if (peerCompare(sortedRows_[peerStartRow_], sortedRows_[peerEndRow_])) {
          break;
        }
        peerEndRow_++;
      }
    }

    // The peer and frame values set in the WindowFunction::apply buffers
    // are the offsets within the current partition, so offset all row
    // indexes correctly. This is required since peerStartRow_ and
    // peerEndRow_ are computed wrt the start of all input rows.
    rawPeerStartBuffer_[j] = peerStartRow_ - firstPartitionRow;
    rawPeerEndBuffer_[j] = peerEndRow_ - 1 - firstPartitionRow;

    for (auto w = 0; w < numFuncs; w++) {
      auto frameEndPoints = findFrameEndPoints(w, firstPartitionRow, endRow, i);
      allFuncsRawFrameStartBuffer_[w][j] =
          frameEndPoints.first - firstPartitionRow;
      allFuncsRawFrameEndBuffer_[w][j] =
          frameEndPoints.second - firstPartitionRow;
    }
  }

  // Invoke the apply method for the WindowFunctions
  for (auto w = 0; w < numFuncs; w++) {
    windowFunctions_[w]->apply(
        peerStartBuffer_,
        peerEndBuffer_,
        allFuncsFrameStartBuffer_[w],
        allFuncsFrameEndBuffer_[w],
        resultIndex,
        windowFunctionOutputs[w]);
  }

  numProcessedRows_ += numRows;
  if (endRow == partitionStartRows_[currentPartition_ + 1]) {
    currentPartition_++;
  }
}

// Function to compute window function values for the current output
// buffer. The buffer has numOutputRows number of rows. windowOutputs
// has the vectors for window function columns.
void Window::computeWindowOutputs(
    vector_size_t numOutputRows,
    const std::vector<VectorPtr>& windowOutputs) {
  // Compute outputs by traversing as many partitions as possible. This
  // logic takes care of partial partitions output also.

  vector_size_t resultIndex = 0;
  while (numOutputRows > 0) {
    auto rowsForCurrentPartition =
        partitionStartRows_[currentPartition_ + 1] - numProcessedRows_;
    if (rowsForCurrentPartition <= numOutputRows) {
      // Current partition can fit completely in the output buffer.
      // So output all its rows.
      callApplyForPartitionRows(
          numProcessedRows_,
          numProcessedRows_ + rowsForCurrentPartition,
          windowOutputs,
          resultIndex);
      resultIndex += rowsForCurrentPartition;
      numOutputRows -= rowsForCurrentPartition;
    } else {
      // Current partition can fit only partially in the output buffer.
      // Call apply for the rows that can fit in the buffer and break from
      // outputting.
      callApplyForPartitionRows(
          numProcessedRows_,
          numProcessedRows_ + numOutputRows,
          windowOutputs,
          resultIndex);
      numOutputRows = 0;
      break;
    }
  }
}

RowVectorPtr Window::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  auto numRowsLeft = numRows_ - numProcessedRows_;
  auto numOutputRows =
      (numRowsPerOutput_ < numRowsLeft) ? numRowsPerOutput_ : numRowsLeft;
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
  computeWindowOutputs(numOutputRows, windowOutputs);

  for (int j = numInputColumns_; j < outputType_->size(); j++) {
    result->childAt(j) = windowOutputs[j - numInputColumns_];
  }

  finished_ = (numProcessedRows_ == sortedRows_.size());
  return result;
}

} // namespace facebook::velox::exec
