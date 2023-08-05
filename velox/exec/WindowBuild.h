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

#include "velox/exec/RowContainer.h"
#include "velox/exec/WindowPartition.h"

namespace facebook::velox::exec {

// The Window operator needs to see all input rows, and separate them into
// partitions based on a partitioning key. There are many approaches to do
// this. e.g with a full-sort, HashTable, streaming etc. This abstraction
// is used by the Window operator to hold the input rows and provide
// partitions to it for processing. Varied implementations of the
// WindowBuild can use different algorithms.
class WindowBuild {
 public:
  WindowBuild(
      const std::shared_ptr<const core::WindowNode>& windowNode,
      velox::memory::MemoryPool* pool);

  virtual ~WindowBuild() = default;

  // Adds new input rows to the WindowBuild.
  void addInput(RowVectorPtr input);

  // This signals the end of all the input rows to the WindowBuild.
  // For streaming windows addInput and getColumns are interleaved.
  // For blocking windows, getColumns and resetPartition are only
  // called after end of all input rows.
  virtual void noMoreInput() = 0;

  // This signals that the operator has consumed all the rows of the
  // current partition and wants to get the next one. Returning
  // nullptr means the next partition is not yet ready for output.
  virtual WindowPartition* nextPartition() = 0;

  std::optional<int64_t> estimateRowSize() {
    return data_->estimateRowSize();
  }

 protected:
  bool compareRowsWithKeys(
      const char* lhs,
      const char* rhs,
      const std::vector<std::pair<column_index_t, core::SortOrder>>& keys);

  // The below 2 vectors represent the ChannelIndex of the partition keys
  // and the order by keys. These keyInfo are used for sorting by those
  // key combinations during the processing.
  // partitionKeyInfo_ is used to separate partitions in the rows.
  // sortKeyInfo_ is used to identify peer rows in a partition.
  std::vector<std::pair<column_index_t, core::SortOrder>> partitionKeyInfo_;
  std::vector<std::pair<column_index_t, core::SortOrder>> sortKeyInfo_;

  const vector_size_t numInputColumns_;

  // The Window operator needs to see all the input rows before starting
  // any function computation. As the Window operators gets input rows
  // we store the rows in the RowContainer (data_).
  std::unique_ptr<RowContainer> data_;

  // The decodedInputVectors_ are reused across addInput() calls to decode
  // the partition and sort keys for the above RowContainer.
  std::vector<DecodedVector> decodedInputVectors_;

  // Window partition object used to provide per-partition
  // data to the window function.
  std::unique_ptr<WindowPartition> windowPartition_;

  // Number of input rows.
  vector_size_t numRows_ = 0;
};

// This class a type of WindowBuild which does a full sort of the
// input data by {partition keys, sort keys}. This sort fully orders
// rows as needed for window function computation.
class SortWindowBuild : public WindowBuild {
 public:
  SortWindowBuild(
      const std::shared_ptr<const core::WindowNode>& windowNode,
      velox::memory::MemoryPool* pool);

  void noMoreInput() override;

  WindowPartition* nextPartition() override;

 private:
  // Main sorting function loop done after all input rows are received
  // by WindowBuild.
  void sortPartitions();

  // Function to compute the partitionStartRows_ structure.
  // partitionStartRows_ is vector of the starting rows index
  // of each partition in the data. This is an auxiliary
  // structure that helps simplify the window function computations.
  void computePartitionStartRows();

  // allKeyInfo_ is a combination of (partitionKeyInfo_ and sortKeyInfo_).
  // It is used to perform a full sorting of the input rows to be able to
  // separate partitions and sort the rows in it. The rows are output in
  // this order by the operator.
  std::vector<std::pair<column_index_t, core::SortOrder>> allKeyInfo_;

  // Vector of pointers to each input row in the data_ RowContainer.
  // The rows are sorted by partitionKeys + sortKeys. This total
  // ordering can be used to split partitions (with the correct
  // order by) for the processing.
  std::vector<char*> sortedRows_;

  // This is a vector that gives the index of the start row
  // (in sortedRows_) of each partition in the RowContainer data_.
  // This auxiliary structure helps demarcate partitions.
  std::vector<vector_size_t> partitionStartRows_;

  // Current partition being output. Used to construct WindowPartitions
  // during resetPartition.
  vector_size_t currentPartition_ = -1;
};

} // namespace facebook::velox::exec
