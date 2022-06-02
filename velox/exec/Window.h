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

  static const int32_t kBatchSizeInBytes{2 * 1024 * 1024};

  void initKeyInfo(
      const std::shared_ptr<const RowType>& type,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      std::vector<std::pair<ChannelIndex, core::SortOrder>>& keyInfo);

  std::pair<int32_t, int32_t> findFrameEndPoints(
      int32_t windowFunctionIndex,
      int32_t rowNumber);

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

  size_t numRows_ = 0;
  size_t numRowsReturned_ = 0;
  std::vector<char*> rows_;
  std::vector<char*> returningRows_;
  // This index captures the row index in RowContainer data_ that marks the
  // start of the current partition being processed. (The partitionIter_
  // captures an iterator to the same row index). This state is maintained in
  // the class as this offset carries over across output row batches.
  // In generality for window frame computation we need to know the size of the
  // full partition. Those partition sizes can be computed during the big sort
  // in noMoreInput (use HashLookup structure for this). We could also use a
  // hash + sort based algorithm to partition the rows. If doing so, then
  // the HashTable can maintain a count of the rows in the partition.

  // The current code also assumes that all partition rows fit within a single
  // output block of rows. We do not consider partitions that overlap over output
  // blocks.
  int partitionStartRow_ = 0;
  RowContainerIterator partitionIter_;

  std::vector<std::unique_ptr<exec::WindowFunction>> windowFunctions_;
  std::vector<WindowFrame> windowFrames_;
};

} // namespace facebook::velox::exec
