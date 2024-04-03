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

/// RowsStreamingWindowPartition is to facilitate RowsStreamingWindowBuild by
/// processing rows within WindowPartition in a streaming manner.
class RowsStreamingWindowPartition : public WindowPartition {
 public:
  RowsStreamingWindowPartition(
      RowContainer* data,
      const folly::Range<char**>& rows,
      const std::vector<column_index_t>& inputMapping,
      const std::vector<std::pair<column_index_t, core::SortOrder>>&
          sortKeyInfo);

  // Returns the number of rows in the current partial window partition,
  // including the offset within the full partition.
  vector_size_t numRows() const override {
    if (currentPartition_ == -1) {
      return 0;
    } else {
      return partition_.size() + partitionStartRows_[currentPartition_];
    }
  }

  // Returns the starting offset of the current partial window partition within
  // the full partition.
  vector_size_t offsetInPartition() const override {
    return partitionStartRows_[currentPartition_];
  }

  // Indicates support for row-level streaming processing.
  bool supportRowLevelStreaming() const override {
    return true;
  }

  // Sets the flag indicating that all input rows have been processed on the
  // producer side.
  void setInputRowsFinished() override {
    inputRowsFinished_ = true;
  }

  // Adds new rows to the partition using a streaming approach on the producer
  // side.
  void addNewRows(std::vector<char*> rows) override;

  // Builds the next set of available rows on the consumer side.
  bool buildNextRows() override;

  // Determines if the current partition is complete and then proceed to the
  // next partition.
  bool processFinished() const override {
    return (
        inputRowsFinished_ &&
        currentPartition_ == partitionStartRows_.size() - 2);
  }

 private:
  // Indicates whether all input rows have been added to sortedRows_
  bool inputRowsFinished_ = false;

  // Stores new rows added to the WindowPartition.
  std::vector<char*> sortedRows_;

  // Indices of the start row (in sortedRows_) of each partitial partition.
  std::vector<vector_size_t> partitionStartRows_;

  // Current partial partition being output.
  vector_size_t currentPartition_ = -1;
};
} // namespace facebook::velox::exec
