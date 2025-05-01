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
#include "velox/exec/spiller.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {
class SortInputSpiller;
class SortOutputSpiller;

/// A utility class to accumulate data inside and output the merged result.
/// Spilling would be triggered if spilling is enabled and memory usage exceeds
/// limit.
class MergeBuffer {
 public:
  MergeBuffer(
      RowTypePtr type,
      velox::memory::MemoryPool* pool,
      common::SpillConfig spillConfig,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  ~MergeBuffer();

  void addInput(RowVectorPtr input);

  /// Indicates no more input and triggers finish spilling and setup the sort
  ///  merge reader for processing spilled data.
  void noMoreInput();

  /// Gathers and returns the sort merge output rows in batch.
  RowVectorPtr getOutput(vector_size_t maxOutputRows);

 private:
  // Ensures there is sufficient memory reserved to process 'input'.
  void ensureInputFits(const VectorPtr& input);

  void updateEstimatedOutputRowSize();

  // Invoked to initialize or reset the reusable output buffer to get output.
  void prepareOutput(vector_size_t outputBatchSize);

  // Invoked to initialize reader to read the spilled data from storage for
  // output processing.
  void prepareOutputWithSpill();

  void getOutputWithSpill();

  // Spill during input stage.
  void spillInput();

  // Finish spill, and we shouldn't get any rows from non-spilled partition as
  // there is only one hash partition for MergeBuffer.
  void finishSpill();

  const RowTypePtr type_;
  velox::memory::MemoryPool* const pool_;
  const common::SpillConfig spillConfig_;
  folly::Synchronized<common::SpillStats>* const spillStats_;
  const std::unique_ptr<NoRowContainerSpiller> inputSpiller_;

  // Indicates no more input. Once it is set, addInput() can't be called on this
  // sort buffer object.
  bool noMoreInput_ = false;

  // The number of received input rows.
  uint64_t numInputRows_ = 0;

  // Used to merge the sorted runs from in-memory rows and spilled rows on disk.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> spillMerger_;

  // Records the source rows to copy to 'output_' in order.
  std::vector<const RowVector*> spillSources_;

  std::vector<vector_size_t> spillSourceRows_;

  SpillPartitionSet spillPartitionSet_;

  // Reusable output vector.
  RowVectorPtr output_;

  // The number of rows that has been returned.
  uint64_t numOutputRows_{0};
};
} // namespace facebook::velox::exec
