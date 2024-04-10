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

#include "velox/exec/Aggregate.h"
#include "velox/exec/WindowFunction.h"

namespace facebook::velox::exec {

/// When the window frame size is relatively large, simple aggregation will have
/// a large number of repeated calculations. The optimization idea based on
/// segment trees to reduce the repeated calculation of aggregate functions and
/// reduce the time complexity of calculating a record from O(n) to O(log n)
/// when the window frame size changes arbitrarily.
class SegmentTreeAggregate {
 public:
  enum class FramePart : uint8_t { FULL = 0, LEFT = 1, RIGHT = 2 };

  SegmentTreeAggregate(
      const std::string& name,
      const std::vector<TypePtr>& argTypes,
      const vector_size_t singleGroupRowSize,
      const std::vector<VectorPtr>& argVectors,
      VectorPtr aggregateResultVector,
      velox::memory::MemoryPool* pool);

  /// Destroy any storage for the accumulator in the group row.
  void destroy(const std::unique_ptr<exec::Aggregate>& aggregate);

  /// Allocate n nodes, each size is singleGroupRowSize_.
  char** allocateNodes(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      memory::AllocationPool& allocationPool,
      const vector_size_t& n);

  /// Get level size in the segmentTree.
  inline vector_size_t getLevelSize(
      vector_size_t partitionCount,
      vector_size_t currentLevel,
      vector_size_t levelsOffset) {
    return currentLevel == 0 ? partitionCount
                             : levelsOffset - levelsStart_[currentLevel - 1];
  }

  /// Aggregate the leaf nodes in segmentTree [begin, end) to targetNode.
  void aggregateLeafNodes(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      vector_size_t begin,
      vector_size_t end,
      char* targetNode);

  /// Aggregate the nodes in segmentTree [begin, end) to targetNode. It can be
  /// leaf nodes or upper level nodes in the segmentTree.
  void aggregateNodes(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      vector_size_t levelIdx,
      vector_size_t begin,
      vector_size_t end,
      char* targetNode);

  /// Construct a segmentTree. Compute node count, allocate nodes and
  /// initialize, Then iterate over the levels of the segment tree, for each
  /// upper level nodes, fill it with aggregate intermediate result.
  void constructSegmentTree(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      const exec::WindowPartition* partition);

  /// For each row, aggregate the frameStart to frameEnd by the intermediate
  /// result in the segmentTree.
  void segmentTreeAggregation(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      const SelectivityVector& validRows,
      vector_size_t minFrame,
      vector_size_t maxFrame,
      const vector_size_t* frameStartsVector,
      const vector_size_t* frameEndsVector,
      vector_size_t resultOffset,
      const VectorPtr& result);

  /// Aggregate the leaf nodes in segmentTree [begin, end) to targetNode.
  void evaluateLeaves(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      const vector_size_t& begin,
      const vector_size_t& end,
      vector_size_t rowIdx,
      char* targetNode,
      FramePart leafPart);

  /// Aggregate the upper level nodes in segmentTree [begin, end) to targetNode.
  char* evaluateUpperLevels(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      const vector_size_t& frameStart,
      const vector_size_t& frameEnd,
      const vector_size_t maxLevel,
      vector_size_t rowIdx,
      char* statePtr);

  /// Combine the intermediate source state into targetNode.
  void combine(
      const std::unique_ptr<exec::Aggregate>& aggregate,
      char* sourceState,
      char* targetNode,
      SelectivityVector& rows);

 private:
  memory::MemoryPool* pool_;

  // ConstantVector value from the Window operator is saved in argVectors_
  std::vector<VectorPtr> argVectors_;

  // This is a single aggregate row needed by the aggregate function for its
  // computation. These values are for the row and its various components.
  vector_size_t singleGroupRowSize_;

  // This vector is used to copy from the aggregate to the result.
  VectorPtr aggregateResultVector_;

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

  // Whether the aggregate function is order sensitive.
  bool orderSensitive_;

  // SelectivityVector for the partition.
  SelectivityVector rows_;

  // SelectivityVector for the upper nodes in segment three.
  SelectivityVector selectivityForSegment_;
};

} // namespace facebook::velox::exec


