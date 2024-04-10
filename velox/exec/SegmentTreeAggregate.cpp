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

#include "velox/exec/SegmentTreeAggregate.h"

namespace facebook::velox::exec {

SegmentTreeAggregate::SegmentTreeAggregate(
    const std::string& name,
    const std::vector<TypePtr>& argTypes,
    const vector_size_t singleGroupRowSize,
    const std::vector<VectorPtr>& argVectors,
    VectorPtr aggregateResultVector,
    velox::memory::MemoryPool* pool)
    : singleGroupRowSize_(singleGroupRowSize),
      pool_(pool),
      argVectors_(argVectors),
      aggregateResultVector_(aggregateResultVector) {
  orderSensitive_ =
      exec::getAggregateFunctionEntry(name)->metadata.orderSensitive;

  leftBufferPtr_ = AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
  leftLeavesState_ = leftBufferPtr_->asMutable<char>();
  upperBufferPtr_ = AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
  upperLevelsState_ = upperBufferPtr_->asMutable<char>();
  prevBufferPtr = AlignedBuffer::allocate<char>(singleGroupRowSize_, pool_);
  prevState_ = prevBufferPtr->asMutable<char>();

  intermediateResultVector_ = BaseVector::create(
      exec::Aggregate::intermediateType(name, argTypes), kTreeFanout, pool_);
  intermediateResultForCombine_ = BaseVector::create(
      exec::Aggregate::intermediateType(name, argTypes), 1, pool_);
}

void SegmentTreeAggregate::destroy(
    const std::unique_ptr<exec::Aggregate>& aggregate) {
  if (treeNodes_ != nullptr) {
    std::vector<char*> needDestroy = {
        leftLeavesState_, upperLevelsState_, prevState_};
    aggregate->destroy(folly::Range(needDestroy.data(), needDestroy.size()));
    aggregate->destroy(folly::Range(treeNodes_, treeNodeCount_));
  }
}

char** SegmentTreeAggregate::allocateNodes(
    const std::unique_ptr<exec::Aggregate>& aggregate,
    memory::AllocationPool& allocationPool,
    const vector_size_t& n) {
  auto* nodes = (char**)allocationPool.allocateFixed(sizeof(char*) * n);

  auto alignment = aggregate->accumulatorAlignmentSize();
  for (vector_size_t i = 0; i < n; i++) {
    nodes[i] = allocationPool.allocateFixed(singleGroupRowSize_, alignment);
  }
  return nodes;
}

// Aggregate the leaf nodes in [begin, end) to targetNode.
void SegmentTreeAggregate::aggregateLeafNodes(
    const std::unique_ptr<exec::Aggregate>& aggregate,
    vector_size_t begin,
    vector_size_t end,
    char* targetNode) {
  rows_.setValidRange(begin, end, true);
  rows_.updateBounds(begin, end);
  aggregate->addSingleGroupRawInput(targetNode, rows_, argVectors_, false);
  rows_.setValidRange(begin, end, false);
  rows_.updateBounds(0, 0);
}

// Aggregate the nodes in [begin, end) to targetNode.
void SegmentTreeAggregate::aggregateNodes(
    const std::unique_ptr<exec::Aggregate>& aggregate,
    vector_size_t levelIdx,
    vector_size_t begin,
    vector_size_t end,
    char* targetNode) {
  if (begin == end) {
    return;
  }

  const auto count = end - begin;
  if (levelIdx == 0) {
    // Aggregate the leaf nodes.
    aggregateLeafNodes(aggregate, begin, end, targetNode);
  } else {
    // Find out where the node begins.
    auto beginNode = treeNodes_ + begin + levelsStart_[levelIdx - 1];
    intermediateResultVector_->resize(count);
    BaseVector::prepareForReuse(intermediateResultVector_, count);

    // Extract the upper nodes [begin, end) to intermediateResultVector_.
    aggregate->extractAccumulators(
        beginNode, count, &intermediateResultVector_);

    if (count != selectivityForSegment_.size()) {
      selectivityForSegment_.resize(count);
    }
    selectivityForSegment_.setValidRange(0, count, true);
    // Since we know begin and end of selectivityForSegment_ is [0, count],
    // and we have already set it in selectivityForSegment_.resize(count)
    // so we don't need to call updateBounds() here.

    // Aggregate the upper level nodes to targetNode.
    aggregate->addSingleGroupIntermediateResults(
        targetNode, selectivityForSegment_, {intermediateResultVector_}, false);
  }
}

// Construct segment tree.
void SegmentTreeAggregate::constructSegmentTree(
    const std::unique_ptr<exec::Aggregate>& aggregate,
    const exec::WindowPartition* partition) {
  vector_size_t numRows = partition->numRows();
  if (numRows == 0) {
    return;
  }
  rows_.resize(numRows, false);
  if (treeNodes_ != nullptr) {
    aggregate->destroy(folly::Range(treeNodes_, treeNodeCount_));
    levelsStart_.clear();
  }

  // Compute node count of the segment tree.
  treeNodeCount_ = 0;
  vector_size_t levelNodes = partition->numRows();
  do {
    levelNodes = (levelNodes + (kTreeFanout - 1)) / kTreeFanout;
    treeNodeCount_ += levelNodes;
  } while (levelNodes > 1);

  // Allocate treeNodeCount_ nodes for the segment tree.
  memory::AllocationPool allocationPool{pool_};
  treeNodes_ = allocateNodes(aggregate, allocationPool, treeNodeCount_);
  std::vector<vector_size_t> allSelectedRange;
  for (vector_size_t i = 0; i < treeNodeCount_; i++) {
    allSelectedRange.push_back(i);
  }
  aggregate->clear();
  // Initialize all the tree nodes.
  aggregate->initializeNewGroups(treeNodes_, allSelectedRange);

  // Level 0 is data itself.
  levelsStart_.push_back(0);

  vector_size_t levelsOffset = 0;
  vector_size_t currentLevel = 0;
  vector_size_t levelSize;
  // Iterate over the levels of the segment tree.
  while ((levelSize = getLevelSize(
              partition->numRows(), currentLevel, levelsOffset)) > 1) {
    for (vector_size_t pos = 0; pos < levelSize; pos += kTreeFanout) {
      // Compute the aggregate for node in the segment tree.
      char* targetNode = treeNodes_[levelsOffset];
      aggregateNodes(
          aggregate,
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

void SegmentTreeAggregate::segmentTreeAggregation(
    const std::unique_ptr<exec::Aggregate>& aggregate,
    const SelectivityVector& validRows,
    const vector_size_t minFrame,
    const vector_size_t maxFrame,
    const vector_size_t* frameStartsVector,
    const vector_size_t* frameEndsVector,
    const vector_size_t resultOffset,
    const VectorPtr& result) {
  const auto maxLevel = levelsStart_.size() + 1;
  rightStack_.resize(maxLevel, {0, 0});
  prevBegin_ = 1;
  prevEnd_ = 0;
  SelectivityVector rowsForCombine;
  rowsForCombine.resizeFill(1, true);
  auto singleGroup = std::vector<vector_size_t>{0};
  aggregate->initializeNewGroups(&prevState_, singleGroup);

  validRows.applyToSelected([&](auto i) {
    aggregate->clear();
    aggregate->initializeNewGroups(&leftLeavesState_, singleGroup);
    aggregate->initializeNewGroups(&upperLevelsState_, singleGroup);

    if (!orderSensitive_) {
      // Aggregate the upper level nodes.
      upperLevelsState_ = evaluateUpperLevels(
          aggregate,
          frameStartsVector[i],
          frameEndsVector[i] + 1,
          maxLevel,
          i,
          upperLevelsState_);

      memcpy(leftLeavesState_, upperLevelsState_, singleGroupRowSize_);
      // Aggregate the ragged leaf nodes.
      evaluateLeaves(
          aggregate,
          frameStartsVector[i],
          frameEndsVector[i] + 1,
          i,
          leftLeavesState_,
          FramePart::FULL);
    } else {
      // Aggregate the ragged left leaf nodes.
      evaluateLeaves(
          aggregate,
          frameStartsVector[i],
          frameEndsVector[i] + 1,
          i,
          leftLeavesState_,
          FramePart::LEFT);

      // Aggregate the upper level nodes.
      upperLevelsState_ = evaluateUpperLevels(
          aggregate,
          frameStartsVector[i],
          frameEndsVector[i] + 1,
          maxLevel,
          i,
          upperLevelsState_);

      combine(aggregate, upperLevelsState_, leftLeavesState_, rowsForCombine);

      // Aggregate the ragged right leaf nodes.
      evaluateLeaves(
          aggregate,
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

    aggregate->extractValues(&leftLeavesState_, 1, &aggregateResultVector_);
    result->copy(aggregateResultVector_.get(), resultOffset + i, 0, 1);
  });
}

// Compute the upper level nodes in [frameStart, frameEnd).
char* SegmentTreeAggregate::evaluateUpperLevels(
    const std::unique_ptr<exec::Aggregate>& aggregate,
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
    /// For rowIdx=4, the parent node is the same as rowIdx=3, we can cache the result of [1, 3) in rowIdx=3 and reuse it in rowIdx=4.
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
        aggregateNodes(aggregate, levelIdx, begin, end, statePtr);
      }
      break;
    }

    vector_size_t groupBegin = parentBegin * kTreeFanout;
    if (begin != groupBegin) {
      // Skip level 0, level 0 nodes compute in evaluateLeaves().
      if (levelIdx) {
        aggregateNodes(
            aggregate, levelIdx, begin, groupBegin + kTreeFanout, statePtr);
      }
      parentBegin++;
    }

    vector_size_t groupEnd = parentEnd * kTreeFanout;
    if (end != groupEnd) {
      // Skip level 0, level 0 nodes compute in evaluateLeaves().
      if (levelIdx) {
        if (!orderSensitive_) {
          aggregateNodes(aggregate, levelIdx, groupEnd, end, statePtr);
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
      aggregateNodes(aggregate, levelIdx, groupEnd, end, statePtr);
      rightEntry = {0, 0};
    }
  }
  return statePtr;
}

// Compute the ragged leaf nodes in [begin, end), for order sensitive
// aggregates, we should firstly compute the ragged left side, the upper
// level nodes, finally compute the ragged right side.
void SegmentTreeAggregate::evaluateLeaves(
    const std::unique_ptr<exec::Aggregate>& aggregate,
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
      aggregateLeafNodes(aggregate, begin, end, targetNode);
    }
    return;
  }

  vector_size_t groupBegin = parentBegin * kTreeFanout;
  vector_size_t groupEnd = parentEnd * kTreeFanout;

  // Compute ragged left leaf nodes.
  if (begin != groupBegin && computeLeft) {
    aggregateLeafNodes(aggregate, begin, groupBegin + kTreeFanout, targetNode);
  }

  // Compute ragged right leaf nodes.
  if (end != groupEnd && computeRight) {
    aggregateLeafNodes(aggregate, groupEnd, end, targetNode);
  }
}

// Combine the sourceState to targetNode.
void SegmentTreeAggregate::combine(
    const std::unique_ptr<exec::Aggregate>& aggregate,
    char* sourceState,
    char* targetNode,
    SelectivityVector& rows) {
  aggregate->extractAccumulators(
      &sourceState, 1, &intermediateResultForCombine_);

  // Aggregate the intermediateResultForCombine_ to targetNode.
  aggregate->addSingleGroupIntermediateResults(
      targetNode, rows, {intermediateResultForCombine_}, false);
}

} // namespace facebook::velox::exec
