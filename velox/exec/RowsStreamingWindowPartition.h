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
      const std::vector<exec::RowColumn>& columns,
      const std::vector<std::pair<column_index_t, core::SortOrder>>&
          sortKeyInfo);

  vector_size_t numRows() const override {
    if (currentPartition_ == -1) {
      return 0;
    } else {
      return partition_.size() + partitionStartRows_[currentPartition_];
    }
  }

  vector_size_t offsetInPartition() const override {
    return partitionStartRows_[currentPartition_];
  }

  bool supportRowLevelStreaming() const override {
    return true;
  }

  void setInputRowsFinished() override {
    inputRowsFinished_ = true;
  }

  void addNewRows(std::vector<char*> rows) override;

  bool buildNextRows() override;

  bool processFinished() const override {
    return (
        inputRowsFinished_ &&
        currentPartition_ == partitionStartRows_.size() - 2);
  }

 private:
  // Whether the input rows is all added into the sortedRows_.
  bool inputRowsFinished_ = false;

  // Add new rows in WindowPartition.
  std::vector<char*> sortedRows_;

  // Indices of the start row (in sortedRows_) of each partitial partition.
  std::vector<vector_size_t> partitionStartRows_;

  // Current partial partition being output.
  vector_size_t currentPartition_ = -1;
};
} // namespace facebook::velox::exec
