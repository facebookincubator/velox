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

 private:
  struct WindowFrame {
    const core::WindowNode::WindowType type;
    const core::WindowNode::BoundType startType;
    const core::WindowNode::BoundType endType;
    const std::optional<ChannelIndex> startChannel;
    const std::optional<ChannelIndex> endChannel;
  };

  void setupPeerAndFrameBuffers();

  void computePartitionStartRows();

  void sortPartitions();

  void callResetPartition(vector_size_t idx);

  void callApplyForPartitionRows(
      vector_size_t startRow,
      vector_size_t endRow,
      const std::vector<VectorPtr>& windowFunctionOutputs,
      vector_size_t resultIndex);

  std::pair<vector_size_t, vector_size_t> findFrameEndPoints(
      vector_size_t idx,
      vector_size_t partitionStartRow,
      vector_size_t partitionEndRow,
      vector_size_t currentRow);

  inline bool compareRowsWithKeys(
      const char* lhs,
      const char* rhs,
      const std::vector<std::pair<ChannelIndex, core::SortOrder>>& keys);

  bool finished_ = false;
  const vector_size_t outputBatchSizeInBytes_;

  const vector_size_t numInputColumns_;

  // The Window operator needs to see all the input rows before starting
  // any function computation. As the inputs are added to Window operator,
  // we store the rows in the RowContainer (data_).
  std::unique_ptr<RowContainer> data_;
  // The decodedInputVectors_ are reused across addInput() calls to decode
  // the partition and sort keys for the above RowContainer.
  std::vector<DecodedVector> decodedInputVectors_;

  // The below 3 vectors represent the ChannelIndex of the partition keys,
  // the order by keys and the concatenation of the 2. These keyInfo are
  // used for sorting by those key combinations during the processing.
  std::vector<std::pair<ChannelIndex, core::SortOrder>> partitionKeyInfo_;
  std::vector<std::pair<ChannelIndex, core::SortOrder>> sortKeyInfo_;
  std::vector<std::pair<ChannelIndex, core::SortOrder>> allKeyInfo_;

  // Vector of WindowFunction objects required by this operator.
  // WindowFunction is the base API implemented by all the window functions.
  // The functions are ordered by their positions in the output columns.
  std::vector<std::unique_ptr<exec::WindowFunction>> windowFunctions_;
  // Vector of WindowFrames corresponding to each windowFunction above.
  // It represents the frame spec for the function computation.
  std::vector<WindowFrame> windowFrames_;

  // This SelectivityVector is used across addInput calls for decoding.
  SelectivityVector allRows_;
  // Number of input rows.
  vector_size_t numRows_ = 0;
  // Vector of pointers to each input row in the data_ RowContainer.
  // The rows are sorted by partitionKeys + sortKeys. This total
  // ordering can be used to split partitions (with the correct
  // order by) for the processing.
  std::vector<char*> sortedRows_;

  // Number of rows that be fit into an output block.
  vector_size_t numRowsPerOutput_;

  // This is a vector that gives the start row of each partition
  // in the RowContainer data_. This auxiliary structure helps
  // demarcate partitions in getOutput calls.
  std::vector<vector_size_t> partitionStartRows_;

  // These Buffers are used to pass peer and frame start and
  // end values to the WindowFunction::apply method. These
  // buffers can be allocated once and reused across all the getOutput
  // calls.
  // Only a single peer start and peer end buffer is needed across
  // functions (as the peer values are based on the ORDER BY clause).
  BufferPtr peerStartBuffer_;
  BufferPtr peerEndBuffer_;
  // A separate BufferPtr is needed for each function window frame.
  std::vector<BufferPtr> allFuncsFrameStartBuffer_;
  std::vector<BufferPtr> allFuncsFrameEndBuffer_;

  // The 4 below are for the raw pointers to the above BufferPtr.
  vector_size_t* rawPeerStartBuffer_;
  vector_size_t* rawPeerEndBuffer_;
  std::vector<vector_size_t*> allFuncsRawFrameStartBuffer_;
  std::vector<vector_size_t*> allFuncsRawFrameEndBuffer_;

  // Number of rows traversed in data_ RowContainer so far. This
  // value is updated as the apply() function is called on the
  // partition blocks.
  vector_size_t numProcessedRows_ = 0;
  // Current partition being output. The partition might be
  // output across multiple getOutput() calls so this needs to
  // be tracked in the operator.
  vector_size_t currentPartition_;
  // When traversing input partition rows, the peers are the rows
  // with the same values for the ORDER BY clause. These rows
  // are equal in some ways and affect the results of ranking functions.
  // All rows between the peerStartRow_ and peerEndRow_ have the same
  // values for peerStartRow_ and peerEndRow_. So we needn't compute
  // them for each row independently. Since these rows might
  // cross getOutput boundaries they are saved in the operator.
  vector_size_t peerStartRow_ = 0;
  vector_size_t peerEndRow_ = 0;
};

} // namespace facebook::velox::exec
