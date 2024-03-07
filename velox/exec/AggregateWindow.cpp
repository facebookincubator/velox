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

#include "velox/exec/AggregateWindow.h"
#include "velox/common/base/Exceptions.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/WindowFunction.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

namespace {

// A generic way to compute any aggregation used as a window function.
// Creates an Aggregate function object for the window function invocation.
// At each row, computes the aggregation across all rows from the frameStart
// to frameEnd boundaries at that row using singleGroup.
class AggregateWindowFunction : public exec::WindowFunction {
 public:
  AggregateWindowFunction(
      const std::string& name,
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      bool ignoreNulls,
      velox::memory::MemoryPool* pool,
      HashStringAllocator* stringAllocator,
      const core::QueryConfig& config)
      : WindowFunction(resultType, pool, stringAllocator) {
    VELOX_USER_CHECK(
        !ignoreNulls, "Aggregate window functions do not support IGNORE NULLS");
    argTypes_.reserve(args.size());
    argIndices_.reserve(args.size());
    argVectors_.reserve(args.size());
    for (const auto& arg : args) {
      argTypes_.push_back(arg.type);
      if (arg.constantValue) {
        argIndices_.push_back(kConstantChannel);
        argVectors_.push_back(arg.constantValue);
      } else {
        VELOX_CHECK(arg.index.has_value());
        argIndices_.push_back(arg.index.value());
        argVectors_.push_back(BaseVector::create(arg.type, 0, pool_));
      }
    }
    // Create an Aggregate function object to do result computation. Window
    // function usage only requires single group aggregation for calculating
    // the function value for each row.
    aggregate_ = exec::Aggregate::create(
        name,
        core::AggregationNode::Step::kSingle,
        argTypes_,
        resultType,
        config);
    aggregate_->setAllocator(stringAllocator_);
    orderSensitive_ =
        exec::getAggregateFunctionEntry(name)->metadata.orderSensitive;

    // Aggregate initialization.
    // Row layout is:
    //  - null flags - one bit per aggregate.
    //  - uint32_t row size,
    //  - fixed-width accumulators - one per aggregate
    //
    // Here we always make space for a row size since we only have one
    // row and no RowContainer. We also have a single aggregate here, so there
    // is only one null bit and one initialized bit.
    static const int32_t kAccumulatorFlagsOffset = 0;
    static const int32_t kRowSizeOffset = bits::nbytes(1);
    singleGroupRowSize_ = kRowSizeOffset + sizeof(int32_t);
    // Accumulator offset must be aligned by their alignment size.
    singleGroupRowSize_ = bits::roundUp(
        singleGroupRowSize_, aggregate_->accumulatorAlignmentSize());
    aggregate_->setOffsets(
        singleGroupRowSize_,
        exec::RowContainer::nullByte(kAccumulatorFlagsOffset),
        exec::RowContainer::nullMask(kAccumulatorFlagsOffset),
        exec::RowContainer::initializedByte(kAccumulatorFlagsOffset),
        exec::RowContainer::initializedMask(kAccumulatorFlagsOffset),
        /* needed for out of line allocations */ kRowSizeOffset);
    singleGroupRowSize_ += aggregate_->accumulatorFixedWidthSize();

    // Construct the single row in the MemoryPool.
    singleGroupRowBufferPtr_ =
        AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
    rawSingleGroupRow_ = singleGroupRowBufferPtr_->asMutable<char>();

    leftBufferPtr_ = AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
    leftLeavesState_ = leftBufferPtr_->asMutable<char>();
    upperBufferPtr_ = AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
    upperLevelsState_ = upperBufferPtr_->asMutable<char>();
    prevBufferPtr = AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
    prevState_ = prevBufferPtr->asMutable<char>();

    intermediateResultVector_ = BaseVector::create(
        exec::Aggregate::intermediateType(name, argTypes_), kTreeFanout, pool_);
    intermediateResultForCombine_ = BaseVector::create(
        exec::Aggregate::intermediateType(name, argTypes_), 1, pool_);
    // Constructing a vector of a single result value used for copying from
    // the aggregate to the final result.
    aggregateResultVector_ = BaseVector::create(resultType, 1, pool_);

    computeDefaultAggregateValue(resultType);
  }

  ~AggregateWindowFunction() {
    // Needed to delete any out-of-line storage for the accumulator in the
    // group row.
    if (aggregateInitialized_) {
      std::vector<char*> singleGroupRowVector = {rawSingleGroupRow_};
      aggregate_->destroy(folly::Range(singleGroupRowVector.data(), 1));
    }
    if (treeNodes_ != nullptr) {
      std::vector<char*> needDestroy = {
          leftLeavesState_, upperLevelsState_, prevState_};
      aggregate_->destroy(folly::Range(needDestroy.data(), needDestroy.size()));
      aggregate_->destroy(folly::Range(treeNodes_, treeNodeCount_));
    }
  }

  void initialize(const WindowFrame& windowFrame,
                  vector_size_t minFrameSizeUseSegmentTree,
                  bool enableSegmentTreeOpt) override {
    enableSegmentTreeOpt_ = enableSegmentTreeOpt;
    minFrameUseSegmentTree_ = minFrameSizeUseSegmentTree;
    // Enable use segment tree when frame size >= minFrameSizeUseSegmentTree.
    if (enableSegmentTreeOpt_ && windowFrame.start.has_value() &&
        windowFrame.start.value().constant.has_value() &&
        windowFrame.end.has_value() &&
        windowFrame.end.value().constant.has_value() &&
        windowFrame.start.value().constant.value() +
                windowFrame.end.value().constant.value() >=
            minFrameSizeUseSegmentTree) {
      useSegmentTreeByConstFrame_ = true;
    } else {
      useSegmentTreeByConstFrame_ = false;
    }
  }

  char** allocateNodes(
      memory::AllocationPool& allocationPool,
      const vector_size_t& n) const {
    auto* nodes = (char**)allocationPool.allocateFixed(sizeof(char*) * n);

    auto alignment = aggregate_->accumulatorAlignmentSize();
    for (vector_size_t i = 0; i < n; i++) {
      nodes[i] = allocationPool.allocateFixed(singleGroupRowSize_, alignment);
    }
    return nodes;
  }

  // Aggregate the leaf nodes in [begin, end) to targetNode.
  void
  aggregateLeafNodes(vector_size_t begin, vector_size_t end, char* targetNode) {
    rows_.setValidRange(begin, end, true);
    rows_.updateBounds(begin, end);
    aggregate_->addSingleGroupRawInput(targetNode, rows_, argVectors_, false);
    rows_.setValidRange(begin, end, false);
    rows_.updateBounds(0, 0);
  }

  // Aggregate the nodes in [begin, end) to targetNode.
  void aggregateNodes(
      vector_size_t levelIdx,
      vector_size_t begin,
      vector_size_t end,
      char* targetNode) {
    if (begin == end || partition_->numRows() == 0) {
      return;
    }

    const auto count = end - begin;
    if (levelIdx == 0) {
      // Aggregate the leaf nodes.
      aggregateLeafNodes(begin, end, targetNode);
    } else {
      // Find out where the node begins.
      auto beginNode = treeNodes_ + begin + levelsStart_[levelIdx - 1];
      intermediateResultVector_->resize(count);
      BaseVector::prepareForReuse(intermediateResultVector_, count);

      // Extract the upper nodes [begin, end) to intermediateResultVector_.
      aggregate_->extractAccumulators(
          beginNode, count, &intermediateResultVector_);

      if (count != selectivityForSegment_.size()) {
        selectivityForSegment_.resize(count);
      }
      selectivityForSegment_.setValidRange(0, count, true);
      // Since we know begin and end of selectivityForSegment_ is [0, count],
      // and we have already set it in selectivityForSegment_.resize(count)
      // so we don't need to call updateBounds() here.

      // Aggregate the upper nodes to targetNode.
      aggregate_->addSingleGroupIntermediateResults(
          targetNode,
          selectivityForSegment_,
          {intermediateResultVector_},
          false);
    }
  }

  inline vector_size_t getLevelSize(
      vector_size_t currentLevel,
      vector_size_t levelsOffset) {
    return currentLevel == 0 ? partition_->numRows()
                             : levelsOffset - levelsStart_[currentLevel - 1];
  }

  // Construct segment tree.
  void constructSegmentTree() {
    vector_size_t numRows = partition_->numRows();
    fillArgVectors(0, numRows - 1);
    rows_.resize(numRows, false);
    if (treeNodes_ != nullptr) {
      aggregate_->destroy(folly::Range(treeNodes_, treeNodeCount_));
      levelsStart_.clear();
    }

    // Compute node count of the segment tree.
    treeNodeCount_ = 0;
    vector_size_t levelNodes = partition_->numRows();
    do {
      levelNodes = (levelNodes + (kTreeFanout - 1)) / kTreeFanout;
      treeNodeCount_ += levelNodes;
    } while (levelNodes > 1);

    // Allocate treeNodeCount_ nodes for the segment tree.
    memory::AllocationPool allocationPool{pool_};
    treeNodes_ = allocateNodes(allocationPool, treeNodeCount_);
    std::vector<vector_size_t> allSelectedRange;
    for (vector_size_t i = 0; i < treeNodeCount_; i++) {
      allSelectedRange.push_back(i);
    }
    aggregate_->clear();
    // Initialize all the tree nodes.
    aggregate_->initializeNewGroups(treeNodes_, allSelectedRange);

    // Level 0 is data itself.
    levelsStart_.push_back(0);

    vector_size_t levelsOffset = 0;
    vector_size_t currentLevel = 0;
    vector_size_t levelSize;
    // Iterate over the levels of the segment tree.
    while ((levelSize = getLevelSize(currentLevel, levelsOffset)) > 1) {
      for (vector_size_t pos = 0; pos < levelSize; pos += kTreeFanout) {
        // Compute the aggregate for node in the segment tree.
        char* targetNode = treeNodes_[levelsOffset];
        aggregateNodes(
            currentLevel,
            pos,
            std::min(levelSize, pos + kTreeFanout),
            targetNode);
        levelsOffset++;
      }

      levelsStart_.push_back(levelsOffset);
      currentLevel++;
    }
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;

    previousFrameMetadata_.reset();
    currentPartitionUseSegmentTree_ = false;
    newPartition_ = true;
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds,
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) override {
    if (handleAllEmptyFrames(validRows, resultOffset, result)) {
      return;
    }

    auto rawFrameStarts = frameStarts->as<vector_size_t>();
    auto rawFrameEnds = frameEnds->as<vector_size_t>();

    FrameMetadata frameMetadata =
        analyzeFrameValues(validRows, rawFrameStarts, rawFrameEnds);

    if (frameMetadata.incrementalAggregation) {
      vector_size_t startRow;
      if (frameMetadata.usePreviousAggregate) {
        // If incremental aggregation can be resumed from the previous block,
        // then the argument vectors also can be populated from the previous
        // frameEnd to the current frameEnd. Only the new values are
        // required for computing aggregates.
        startRow = previousFrameMetadata_->lastRow + 1;
      } else {
        startRow = frameMetadata.firstRow;

        // This is the start of a new incremental aggregation. So the
        // aggregate_ function object should be initialized.
        auto singleGroup = std::vector<vector_size_t>{0};
        aggregate_->clear();
        aggregate_->initializeNewGroups(&rawSingleGroupRow_, singleGroup);
        aggregateInitialized_ = true;
      }

      fillArgVectors(startRow, frameMetadata.lastRow);
      incrementalAggregation(
          validRows,
          startRow,
          frameMetadata.lastRow,
          rawFrameEnds,
          resultOffset,
          result);
    } else {
      if (enableSegmentTreeOpt_
          && newPartition_
          && validRows.countSelected() > minFrameUseSegmentTree_) {
        newPartition_ = false;
        if (useSegmentTreeByConstFrame_) {
          currentPartitionUseSegmentTree_ = true;
          constructSegmentTree();
        } else {
          auto count = 0;
          auto sum = 0;
          validRows.applyToSelected([&](auto i) {
            sum = sum + rawFrameEnds[i] + 1 - rawFrameStarts[i];
            count += 1;
          });
          if (count > 0 && sum / count >= minFrameUseSegmentTree_) {
            currentPartitionUseSegmentTree_ = true;
            constructSegmentTree();
          }
        }
      }

      if (currentPartitionUseSegmentTree_) {
        segmentTreeAggregation(
            validRows,
            frameMetadata.firstRow,
            frameMetadata.lastRow,
            rawFrameStarts,
            rawFrameEnds,
            resultOffset,
            result);
      } else {
        fillArgVectors(frameMetadata.firstRow, frameMetadata.lastRow);
        simpleAggregation(
            validRows,
            frameMetadata.firstRow,
            frameMetadata.lastRow,
            rawFrameStarts,
            rawFrameEnds,
            resultOffset,
            result);
      }
    }
    previousFrameMetadata_ = frameMetadata;
  }

 private:
  struct FrameMetadata {
    // Min frame start row required for aggregation.
    vector_size_t firstRow;

    // Max frame end required for the aggregation.
    vector_size_t lastRow;

    // If all the rows in the block have the same start row, and the
    // end frame rows are non-decreasing, then the aggregation can be done
    // incrementally. With incremental aggregation new frame rows are
    // accumulated over the previous result to obtain the new result.
    bool incrementalAggregation;

    // Resume incremental aggregation from the prior block.
    bool usePreviousAggregate;
  };

  bool handleAllEmptyFrames(
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    if (!validRows.hasSelections()) {
      setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
      return true;
    }
    return false;
  }

  // Computes the least frameStart row and the max frameEnds row
  // indices for the valid frames of this output block. These indices are used
  // as bounds when reading input parameter vectors for aggregation.
  // This method expects to have at least 1 valid frame in the block.
  // Blocks with all empty frames are handled before this point.
  FrameMetadata analyzeFrameValues(
      const SelectivityVector& validRows,
      const vector_size_t* rawFrameStarts,
      const vector_size_t* rawFrameEnds) {
    VELOX_DCHECK(validRows.hasSelections());

    // Use first valid frame row for the initialization.
    auto firstValidRow = validRows.begin();
    vector_size_t firstRow = rawFrameStarts[firstValidRow];
    vector_size_t fixedFrameStartRow = firstRow;
    vector_size_t lastRow = rawFrameEnds[firstValidRow];
    vector_size_t prevFrameEnds = lastRow;

    bool incrementalAggregation = true;
    validRows.applyToSelected([&](auto i) {
      firstRow = std::min(firstRow, rawFrameStarts[i]);
      lastRow = std::max(lastRow, rawFrameEnds[i]);

      // Incremental aggregation can be done if :
      // i) All rows have the same frameStart value.
      // ii) The frame end values are non-decreasing.
      incrementalAggregation &= (rawFrameStarts[i] == fixedFrameStartRow);
      incrementalAggregation &= rawFrameEnds[i] >= prevFrameEnds;
      prevFrameEnds = rawFrameEnds[i];
    });

    bool usePreviousAggregate = false;
    if (previousFrameMetadata_.has_value()) {
      auto previousFrame = previousFrameMetadata_.value();
      // Incremental aggregation continues between blocks if :
      // i) Their starting firstRow values are the same.
      // ii) The nonDecreasing frameEnd property is also applicable between the
      // lastRow of the first block and the first row of the current block.
      if (incrementalAggregation && previousFrame.incrementalAggregation &&
          previousFrame.firstRow == firstRow &&
          previousFrame.lastRow <= rawFrameEnds[firstValidRow]) {
        usePreviousAggregate = true;
      }
    }

    return {firstRow, lastRow, incrementalAggregation, usePreviousAggregate};
  }

  void fillArgVectors(vector_size_t firstRow, vector_size_t lastRow) {
    vector_size_t numFrameRows = lastRow + 1 - firstRow;
    for (int i = 0; i < argIndices_.size(); i++) {
      argVectors_[i]->resize(numFrameRows);
      // Only non-constant field argument vectors need to be populated. The
      // constant vectors are correctly set during aggregate initialization
      // itself.
      if (argIndices_[i] != kConstantChannel) {
        partition_->extractColumn(
            argIndices_[i], firstRow, numFrameRows, 0, argVectors_[i]);
      }
    }
  }

  void computeAggregate(
      SelectivityVector rows,
      vector_size_t startFrame,
      vector_size_t endFrame) {
    rows.clearAll();
    rows.setValidRange(startFrame, endFrame, true);
    rows.updateBounds();

    BaseVector::prepareForReuse(aggregateResultVector_, 1);

    aggregate_->addSingleGroupRawInput(
        rawSingleGroupRow_, rows, argVectors_, false);
    aggregate_->extractValues(&rawSingleGroupRow_, 1, &aggregateResultVector_);
  }

  void incrementalAggregation(
      const SelectivityVector& validRows,
      vector_size_t startFrame,
      vector_size_t endFrame,
      const vector_size_t* rawFrameEnds,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    SelectivityVector rows;
    rows.resize(endFrame + 1 - startFrame);

    auto prevFrameEnd = 0;
    // This is a simple optimization for frames that have a fixed startFrame
    // and increasing frameEnd values. In that case, we can
    // incrementally aggregate over the new rows seen in the frame between
    // the previous and current row.
    validRows.applyToSelected([&](auto i) {
      auto currentFrameEnd = rawFrameEnds[i] - startFrame + 1;
      if (currentFrameEnd > prevFrameEnd) {
        computeAggregate(rows, prevFrameEnd, currentFrameEnd);
      }

      result->copy(aggregateResultVector_.get(), resultOffset + i, 0, 1);
      prevFrameEnd = currentFrameEnd;
    });

    // Set null values for empty (non valid) frames in the output block.
    setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
  }

  void simpleAggregation(
      const SelectivityVector& validRows,
      vector_size_t minFrame,
      vector_size_t maxFrame,
      const vector_size_t* frameStartsVector,
      const vector_size_t* frameEndsVector,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    SelectivityVector rows;
    rows.resize(maxFrame + 1 - minFrame);
    static auto kSingleGroup = std::vector<vector_size_t>{0};

    validRows.applyToSelected([&](auto i) {
      // This is a very naive algorithm.
      // It evaluates the entire aggregation for each row by iterating over
      // input rows from frameStart to frameEnd in the SelectivityVector.
      // TODO : Try to re-use previous computations by advancing and retracting
      // the aggregation based on the frame changes with each row. This would
      // require adding new APIs to the Aggregate framework.
      aggregate_->clear();
      aggregate_->initializeNewGroups(&rawSingleGroupRow_, kSingleGroup);
      aggregateInitialized_ = true;

      auto frameStartIndex = frameStartsVector[i] - minFrame;
      auto frameEndIndex = frameEndsVector[i] - minFrame + 1;
      computeAggregate(rows, frameStartIndex, frameEndIndex);
      result->copy(aggregateResultVector_.get(), resultOffset + i, 0, 1);
    });

    // Set null values for empty (non valid) frames in the output block.
    setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
  }

  // Compute the upper level nodes in [frameStart, frameEnd).
  char* evaluateUpperLevels(
      const vector_size_t& frameStart,
      const vector_size_t& frameEnd,
      const vector_size_t maxLevel,
      vector_size_t rowIdx,
      char* statePtr) {
    auto begin = frameStart;
    auto end = frameEnd;

    vector_size_t levelIdx = 0;
    vector_size_t rightMax = 0;

    for (; levelIdx < maxLevel; levelIdx++) {
      auto parentBegin = begin / kTreeFanout;
      auto parentEnd = end / kTreeFanout;
      /// If two record frame has a same upper level, we can reuse upper level
      /// result. For example:
      ///   level_3          0
      ///                  /   \
      ///                 /     \
      ///                /       \
      ///   level_2     0         1
      ///              / \       / \
      ///             /   \     /   \
      ///   level_1   0    1    2    3
      ///            / \  / \  / \  / \
      ///   level_0  0 1  2 3  4 5  6 7
      /// rowIdx=3, frame is [1~6), parent is [1, 3)
      /// rowIdx=4, frame is [2~7), parent is [1, 3)
      /// For rowIdx=4, the parent node is the same as rowIdx=3, we can cache the
      /// result of [1, 3) in rowIdx=3 and reuse it in rowIdx=4.
      if (levelIdx == 1 && parentBegin == prevBegin_ && parentEnd == prevEnd_) {
        // Just return the previous result.
        tempState_ = statePtr;
        statePtr = prevState_;
        return statePtr;
      }

      if (!orderSensitive_ && levelIdx == 1) {
        tempState_ = prevState_;
        prevState_ = statePtr;
        prevBegin_ = begin;
        prevEnd_ = end;
      }

      if (parentBegin == parentEnd) {
        // Skip level 0, level 0 nodes compute in evaluateLeaves().
        if (levelIdx) {
          aggregateNodes(levelIdx, begin, end, statePtr);
        }
        break;
      }

      vector_size_t groupBegin = parentBegin * kTreeFanout;
      if (begin != groupBegin) {
        // Skip level 0, level 0 nodes compute in evaluateLeaves().
        if (levelIdx) {
          aggregateNodes(levelIdx, begin, groupBegin + kTreeFanout, statePtr);
        }
        parentBegin++;
      }

      vector_size_t groupEnd = parentEnd * kTreeFanout;
      if (end != groupEnd) {
        // Skip level 0, level 0 nodes compute in evaluateLeaves().
        if (levelIdx) {
          if (!orderSensitive_) {
            aggregateNodes(levelIdx, groupEnd, end, statePtr);
          } else {
            // If order sensitive, we should compute left side before right
            // side, so here we only record the ranges in rightStack_.
            rightStack_[levelIdx] = {groupEnd, end};
            rightMax = levelIdx;
          }
        }
      }
      begin = parentBegin;
      end = parentEnd;
    }

    // For order sensitive aggregates. As we go up the tree, we can just
    // reverse scan the array and append the cached ranges.
    for (levelIdx = rightMax; levelIdx > 0; --levelIdx) {
      auto& rightEntry = rightStack_[levelIdx];
      const auto groupEnd = rightEntry.first;
      const auto end = rightEntry.second;
      if (end) {
        aggregateNodes(levelIdx, groupEnd, end, statePtr);
        rightEntry = {0, 0};
      }
    }
    return statePtr;
  }

  // Compute the ragged leaf nodes in [begin, end), for order sensitive
  // aggregates, we should firstly compute the ragged left side, the upper
  // level nodes, finally compute the ragged right side.
  void evaluateLeaves(
      const vector_size_t& begin,
      const vector_size_t& end,
      vector_size_t rowIdx,
      char* targetNode,
      FramePart leafPart) {
    const bool computeLeft = leafPart != FramePart::RIGHT;
    const bool computeRight = leafPart != FramePart::LEFT;

    auto parentBegin = begin / kTreeFanout;
    auto parentEnd = end / kTreeFanout;
    if (parentBegin == parentEnd) {
      // Only compute when parentBegin == parentEnd and computeLeft.
      if (computeLeft) {
        aggregateLeafNodes(begin, end, targetNode);
      }
      return;
    }

    vector_size_t groupBegin = parentBegin * kTreeFanout;
    vector_size_t groupEnd = parentEnd * kTreeFanout;

    // Compute ragged left leaf nodes.
    if (begin != groupBegin && computeLeft) {
      aggregateLeafNodes(begin, groupBegin + kTreeFanout, targetNode);
    }

    // Compute ragged right leaf nodes.
    if (end != groupEnd && computeRight) {
      aggregateLeafNodes(groupEnd, end, targetNode);
    }
  }

  // Combine the sourceState to targetNode.
  void combine(char* sourceState, char* targetNode, SelectivityVector& rows) {
    aggregate_->extractAccumulators(
        &sourceState, 1, &intermediateResultForCombine_);

    // Aggregate the intermediateResultForCombine_ to targetNode.
    aggregate_->addSingleGroupIntermediateResults(
        targetNode, rows, {intermediateResultForCombine_}, false);
  }

  void segmentTreeAggregation(
      const SelectivityVector& validRows,
      vector_size_t minFrame,
      vector_size_t maxFrame,
      const vector_size_t* frameStartsVector,
      const vector_size_t* frameEndsVector,
      vector_size_t resultOffset,
      const VectorPtr& result) {
    const auto maxLevel = levelsStart_.size() + 1;
    rightStack_.resize(maxLevel, {0, 0});
    prevBegin_ = 1;
    prevEnd_ = 0;
    SelectivityVector rowsForCombine;
    rowsForCombine.resizeFill(1, true);
    auto singleGroup = std::vector<vector_size_t>{0};
    aggregate_->initializeNewGroups(&prevState_, singleGroup);

    validRows.applyToSelected([&](auto i) {
      aggregate_->clear();
      aggregate_->initializeNewGroups(&leftLeavesState_, singleGroup);
      aggregate_->initializeNewGroups(&upperLevelsState_, singleGroup);

      if (!orderSensitive_) {
        // Aggregate the upper level nodes.
        upperLevelsState_ = evaluateUpperLevels(
            frameStartsVector[i],
            frameEndsVector[i] + 1,
            maxLevel,
            i,
            upperLevelsState_);

        memcpy(leftLeavesState_, upperLevelsState_, singleGroupRowSize_);
        // Aggregate the ragged leaf nodes.
        evaluateLeaves(
            frameStartsVector[i],
            frameEndsVector[i] + 1,
            i,
            leftLeavesState_,
            FramePart::FULL);
      } else {
        // Aggregate the ragged left leaf nodes.
        evaluateLeaves(
            frameStartsVector[i],
            frameEndsVector[i] + 1,
            i,
            leftLeavesState_,
            FramePart::LEFT);

        // Aggregate the upper level nodes.
        upperLevelsState_ = evaluateUpperLevels(
            frameStartsVector[i],
            frameEndsVector[i] + 1,
            maxLevel,
            i,
            upperLevelsState_);

        combine(upperLevelsState_, leftLeavesState_, rowsForCombine);

        // Aggregate the ragged right leaf nodes.
        evaluateLeaves(
            frameStartsVector[i],
            frameEndsVector[i] + 1,
            i,
            leftLeavesState_,
            FramePart::RIGHT);
      }
      if (upperLevelsState_ == prevState_) {
        upperLevelsState_ = tempState_;
        tempState_ = nullptr;
      }

      aggregate_->extractValues(&leftLeavesState_, 1, &aggregateResultVector_);
      result->copy(aggregateResultVector_.get(), resultOffset + i, 0, 1);
    });

    // Set null values for empty (non valid) frames in the output block.
    setEmptyFramesResult(validRows, resultOffset, emptyResult_, result);
  }

  // Precompute and save the aggregate output for empty input in emptyResult_.
  // This value is returned for rows with empty frames.
  void computeDefaultAggregateValue(const TypePtr& resultType) {
    aggregate_->clear();
    aggregate_->initializeNewGroups(
        &rawSingleGroupRow_, std::vector<vector_size_t>{0});
    aggregateInitialized_ = true;

    emptyResult_ = BaseVector::create(resultType, 1, pool_);
    aggregate_->extractValues(&rawSingleGroupRow_, 1, &emptyResult_);
    aggregate_->clear();
  }

  // Aggregate function object required for this window function evaluation.
  std::unique_ptr<exec::Aggregate> aggregate_;

  bool aggregateInitialized_{false};

  // If the frame is constant and larger than minFrameUseSegmentTree, direct
  // enable use segment tree, no need to compute the average frame size.
  bool useSegmentTreeByConstFrame_{false};

  // Whether turn on the optimization of segment tree.
  bool enableSegmentTreeOpt_{false};

  // The min average frame size can use segment tree.
  int32_t minFrameUseSegmentTree_;

  // Current WindowPartition used for accessing rows in the apply method.
  const exec::WindowPartition* partition_;

  // Args information : their types, column indexes in inputs and vectors
  // used to populate values to pass to the aggregate function.
  // For a constant argument a column index of kConstantChannel is used in
  // argIndices_, and its ConstantVector value from the Window operator
  // is saved in argVectors_.
  std::vector<TypePtr> argTypes_;
  std::vector<column_index_t> argIndices_;
  std::vector<VectorPtr> argVectors_;

  // This is a single aggregate row needed by the aggregate function for its
  // computation. These values are for the row and its various components.
  BufferPtr singleGroupRowBufferPtr_;
  char* rawSingleGroupRow_;
  vector_size_t singleGroupRowSize_;

  // Used for per-row aggregate computations.
  // This vector is used to copy from the aggregate to the result.
  VectorPtr aggregateResultVector_;

  // Stores metadata about the previous output block of the partition
  // to optimize aggregate computation and reading argument vectors.
  std::optional<FrameMetadata> previousFrameMetadata_;

  // Stores default result value for empty frame aggregation. Window functions
  // return the default value of an aggregate (aggregation with no rows) for
  // empty frames. e.g. count for empty frames should return 0 and not null.
  VectorPtr emptyResult_;

  // Right side nodes need to be cached and processed in reverse order.
  // The first value in pair is groupEnd, the second is end：
  // level n + 1    0   1   2
  //               / \ / \ / \
  // level n       1 2 3 4 5 6
  // [frameStart, frameEnd) = [1,6), The ragged right node is
  // [groupEnd, end) = [5, 6)
  using RightEntry = std::pair<vector_size_t, vector_size_t>;

  // Cache of right side tree ranges for ordered aggregates.
  std::vector<RightEntry> rightStack_;

  // The actual window segment tree, an array of aggregate states that
  // represent all the intermediate nodes.
  char** treeNodes_;

  // The total number of internal nodes of the segment tree.
  vector_size_t treeNodeCount_{0};

  // For each level, the starting location in the treeNodes_ array.
  std::vector<vector_size_t> levelsStart_;

  // Fanout of the segment tree.
  static constexpr vector_size_t kTreeFanout = 16;

  // Intermediate result for construct the segment tree.
  VectorPtr intermediateResultVector_;

  // Intermediate result for combine aggregate result.
  VectorPtr intermediateResultForCombine_;

  // Left BufferPtr use for ragged left leaves.
  BufferPtr leftBufferPtr_;

  // Upper BufferPtr use for upper segment tree nodes.
  BufferPtr upperBufferPtr_;

  // BufferPtr use for prev upper segment tree nodes.
  BufferPtr prevBufferPtr;

  // State for ragged left leaves.
  char* leftLeavesState_;

  // State for upper segment tree nodes.
  char* upperLevelsState_;

  // State for prev upper segment tree nodes.
  char* prevState_;

  // Temp state.
  char* tempState_;

  // Prev begin upper node.
  vector_size_t prevBegin_{1};

  // Prev end upper node.
  vector_size_t prevEnd_{0};

  // Current partition should use segment tree for aggregate.
  bool currentPartitionUseSegmentTree_{false};

  // Whether the first time aggregate current partition.
  bool newPartition_{false};

  // Whether the aggregate function is order sensitive.
  bool orderSensitive_;

  // SelectivityVector for the partition.
  SelectivityVector rows_;

  // SelectivityVector for the upper nodes in segment three.
  SelectivityVector selectivityForSegment_;
};

} // namespace

void registerAggregateWindowFunction(const std::string& name) {
  auto aggregateFunctionSignatures = exec::getAggregateFunctionSignatures(name);
  if (aggregateFunctionSignatures.has_value()) {
    // This copy is needed to obtain a vector of the base FunctionSignaturePtr
    // from the AggregateFunctionSignaturePtr type of
    // aggregateFunctionSignatures variable.
    std::vector<exec::FunctionSignaturePtr> signatures(
        aggregateFunctionSignatures.value().begin(),
        aggregateFunctionSignatures.value().end());

    exec::registerWindowFunction(
        name,
        std::move(signatures),
        [name](
            const std::vector<exec::WindowFunctionArg>& args,
            const TypePtr& resultType,
            bool ignoreNulls,
            velox::memory::MemoryPool* pool,
            HashStringAllocator* stringAllocator,
            const core::QueryConfig& config)
            -> std::unique_ptr<exec::WindowFunction> {
          return std::make_unique<AggregateWindowFunction>(
              name,
              args,
              resultType,
              ignoreNulls,
              pool,
              stringAllocator,
              config);
        });
  }
}
} // namespace facebook::velox::exec
