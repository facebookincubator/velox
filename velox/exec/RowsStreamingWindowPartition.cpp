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
#include "velox/exec/RowsStreamingWindowPartition.h"

namespace facebook::velox::exec {

RowsStreamingWindowPartition::RowsStreamingWindowPartition(
    RowContainer* data,
    const folly::Range<char**>& rows,
    const std::vector<column_index_t>& inputMapping,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& sortKeyInfo)
    : WindowPartition(data, rows, inputMapping, sortKeyInfo) {
  partitionStartRows_.push_back(0);
}

void RowsStreamingWindowPartition::addNewRows(std::vector<char*> rows) {
  partitionStartRows_.push_back(partitionStartRows_.back() + rows.size());

  sortedRows_.insert(sortedRows_.end(), rows.begin(), rows.end());
}

bool RowsStreamingWindowPartition::buildNextRows() {
  if (currentPartition_ >= int(partitionStartRows_.size() - 2))
    return false;

  currentPartition_++;

  // Erase previous rows in current partition.
  if (currentPartition_ > 0) {
    auto numPreviousPartitionRows = partitionStartRows_[currentPartition_] -
        partitionStartRows_[currentPartition_ - 1];
    data_->eraseRows(
        folly::Range<char**>(sortedRows_.data(), numPreviousPartitionRows));
    sortedRows_.erase(
        sortedRows_.begin(), sortedRows_.begin() + numPreviousPartitionRows);
  }

  auto partitionSize = partitionStartRows_[currentPartition_ + 1] -
      partitionStartRows_[currentPartition_];

  partition_ = folly::Range(sortedRows_.data(), partitionSize);
  return true;
}

} // namespace facebook::velox::exec
