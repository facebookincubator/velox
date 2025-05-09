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
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/RowContainer.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {

/// A utility class to spill merged vectors from 'LocalMerge' operator.
/// It finishes spill per partial merge run, and creates a sort-merge reader to
/// restore and output the final merge results.
class MergeBuffer {
 public:
  MergeBuffer(
      const RowTypePtr& type,
      velox::memory::MemoryPool* const pool,
      const std::vector<std::pair<column_index_t, CompareFlags>>& sortingKeys,
      const common::SpillConfig* spillConfig = nullptr,
      folly::Synchronized<velox::common::SpillStats>* spillStats = nullptr);

  /// Spill the merged vector.
  void addInput(const RowVectorPtr vector);

  /// Finish current merge run.
  void finishSpill(bool lastRun);

  /// Returns the sorted output rows in batch.
  RowVectorPtr getOutput(vector_size_t maxOutputRows);

 private:
  // Indicates no more data to spill, finish spill for the current round, and
  // sets up a sort-merge reader to process and produce the output.
  void noMoreInput();

  void createSortMergeReader();

  // Invoked to initialize or reset the reusable output buffer to get output.
  void prepareOutput(vector_size_t outputBatchSize);

  void getOutputInternal();

  const RowTypePtr type_;
  velox::memory::MemoryPool* const pool_;
  const std::vector<std::pair<column_index_t, CompareFlags>> sortingKeys_;
  const common::SpillConfig* const spillConfig_;
  folly::Synchronized<common::SpillStats>* const spillStats_;

  std::unique_ptr<NoRowContainerSpiller> inputSpiller_;
  // SpillFiles per partial spill merge.
  std::vector<SpillFiles> spillFilesLists_;
  // Indicates no more input. Once it is set, addInput() can't be called on this
  // sort buffer object.
  bool noMoreInput_{false};
  // Used to merge the sorted runs from in-memory rows and spilled rows on disk.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> spillMerger_;
  // Records the source rows to copy to 'output_' in order.
  std::vector<const RowVector*> spillSources_;
  std::vector<vector_size_t> spillSourceRows_;
  // Reusable output vector.
  RowVectorPtr output_;
  // The number of received input rows.
  uint64_t numInputRows_{0};
  // The number of rows that has been returned.
  uint64_t numOutputRows_{0};
};
} // namespace facebook::velox::exec
