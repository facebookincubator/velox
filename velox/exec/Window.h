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
#pragma once

#include "velox/exec/Operator.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/WindowFunction.h"

namespace facebook::velox::exec {

class Window : public Operator {
 public:
  Window(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* /* unused */) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void close() override {
    Operator::close();
  }

 private:
  struct WindowFrame {
    const core::WindowNode::WindowType windowType_;
    const core::WindowNode::BoundType startBoundType_;
    const core::WindowNode::BoundType endBoundType_;
    std::optional<ChannelIndex> startChannel_;
    std::optional<ChannelIndex> endChannel_;
  };

  void initKeyInfo(
      const std::shared_ptr<const RowType>& type,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      std::vector<std::pair<ChannelIndex, core::SortOrder>>& keyInfo);

  void callResetPartition(size_t idx);

  void callApplyForPartitionRows(
      size_t startRow,
      size_t endRow,
      const std::vector<VectorPtr>& windowFunctionOutputs,
      const size_t bufferIndex);

  std::pair<int32_t, int32_t> findFrameEndPoints(
      int32_t idx,
      int32_t partitionStartRow,
      int32_t partitionEndRow,
      int32_t currentRow);

  bool finished_ = false;

  // As the inputs are added to Window operator, we store the rows in the
  // RowContainer (data_).
  // Once all inputs are available, we can separate the input rows into Window
  // partitions by sorting all rows with a combination of Window partitionKeys
  // followed by sortKeys.
  // This WindowFunction is invoked with these partition rows to compute its
  // output values.
  const int inputColumnsSize_;
  std::unique_ptr<RowContainer> data_;
  std::vector<DecodedVector> decodedInputVectors_;

  std::vector<std::pair<ChannelIndex, core::SortOrder>> allKeyInfo_;
  std::vector<std::pair<ChannelIndex, core::SortOrder>> partitionKeyInfo_;
  std::vector<std::pair<ChannelIndex, core::SortOrder>> sortKeyInfo_;

  static const int32_t kBatchSizeInBytes{2 * 1024 * 1024};
  size_t numRows_ = 0;

  std::vector<char*> rows_;
  std::vector<char*> returningRows_;

  // The RowContainer data_ is traversed a partition at a time.
  // At the beginning of the partition we call the resetPartition function.
  // The partitionIter_ points to starting row of the next partition.
  // It is updated each time resetPartition is called.
  RowContainerIterator partitionIter_;
  std::vector<char*> partitionRows_;

  std::vector<std::unique_ptr<exec::WindowFunction>> windowFunctions_;
  std::vector<WindowFrame> windowFrames_;

  // Number of rows that be fit into an output block.
  size_t numRowsPerOutput_;

  // This is a vector that gives the start row of each partition
  // in the RowContainer data_. This auxiliary structure helps
  // demarcate partitions in getOutput calls.
  std::vector<size_t> partitionStartRows_;
  size_t numberOfPartitions_;

  // Some variables that maintain state across getOutput() calls.
  // Number of rows traversed in data_ RowContainer so far. This
  // value is updated as the apply() function is called on the
  // partition blocks.
  size_t numRowsApplied_ = 0;
  // This indicates the partition currently being output, and the peers
  // demarcated. Since a partition can be spread over multiple
  // getOutput calls, these variables allow us to continue outputting
  // data in subsequent getOutput calls.
  size_t currentPartitionIndex_;
  size_t peerStartRow_ = 0;
  size_t peerEndRow_ = 0;
};

} // namespace facebook::velox::exec
