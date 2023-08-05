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

#include "velox/exec/WindowBuild.h"

#include "velox/exec/Operator.h"

namespace facebook::velox::exec {

namespace {
void initKeyInfo(
    const RowTypePtr& type,
    const std::vector<core::FieldAccessTypedExprPtr>& keys,
    const std::vector<core::SortOrder>& orders,
    std::vector<std::pair<column_index_t, core::SortOrder>>& keyInfo) {
  const core::SortOrder defaultPartitionSortOrder(true, true);

  keyInfo.reserve(keys.size());
  for (auto i = 0; i < keys.size(); ++i) {
    auto channel = exprToChannel(keys[i].get(), type);
    VELOX_CHECK(
        channel != kConstantChannel,
        "Window doesn't allow constant partition or sort keys");
    if (i < orders.size()) {
      keyInfo.push_back(std::make_pair(channel, orders[i]));
    } else {
      keyInfo.push_back(std::make_pair(channel, defaultPartitionSortOrder));
    }
  }
}
}; // namespace

WindowBuild::WindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool)
    : numInputColumns_(windowNode->sources()[0]->outputType()->size()),
      data_(std::make_unique<RowContainer>(
          windowNode->sources()[0]->outputType()->children(),
          pool)),
      decodedInputVectors_(windowNode->sources()[0]->outputType()->size()) {
  auto inputType = windowNode->sources()[0]->outputType();
  std::vector<exec::RowColumn> inputColumns;
  for (int i = 0; i < inputType->children().size(); i++) {
    inputColumns.push_back(data_->columnAt(i));
  }

  initKeyInfo(inputType, windowNode->partitionKeys(), {}, partitionKeyInfo_);
  initKeyInfo(
      inputType,
      windowNode->sortingKeys(),
      windowNode->sortingOrders(),
      sortKeyInfo_);

  // The WindowPartition is structured over all the input columns data.
  // Individual functions access its input argument column values from it.
  // The RowColumns are copied by the WindowPartition, so its fine to use
  // a local variable here.
  windowPartition_ = std::make_unique<WindowPartition>(
      data_.get(), inputColumns, inputType->children(), sortKeyInfo_);
}

void WindowBuild::addInput(RowVectorPtr input) {
  for (auto col = 0; col < input->childrenSize(); ++col) {
    decodedInputVectors_[col].decode(*input->childAt(col));
  }

  // Add all the rows into the RowContainer.
  for (auto row = 0; row < input->size(); ++row) {
    char* newRow = data_->newRow();

    for (auto col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedInputVectors_[col], row, newRow, col);
    }
  }
  numRows_ += input->size();
}

inline bool WindowBuild::compareRowsWithKeys(
    const char* lhs,
    const char* rhs,
    const std::vector<std::pair<column_index_t, core::SortOrder>>& keys) {
  if (lhs == rhs) {
    return false;
  }
  for (auto& key : keys) {
    if (auto result = data_->compare(
            lhs,
            rhs,
            key.first,
            {key.second.isNullsFirst(), key.second.isAscending(), false})) {
      return result < 0;
    }
  }
  return false;
}

SortWindowBuild::SortWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool)
    : WindowBuild(windowNode, pool) {
  allKeyInfo_.reserve(partitionKeyInfo_.size() + sortKeyInfo_.size());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), partitionKeyInfo_.begin(), partitionKeyInfo_.end());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), sortKeyInfo_.begin(), sortKeyInfo_.end());
  partitionStartRows_.resize(0);
}

void SortWindowBuild::computePartitionStartRows() {
  partitionStartRows_.reserve(numRows_);
  auto partitionCompare = [&](const char* lhs, const char* rhs) -> bool {
    return compareRowsWithKeys(lhs, rhs, partitionKeyInfo_);
  };

  // Using a sequential traversal to find changing partitions.
  // This algorithm is inefficient and can be changed
  // i) Use a binary search kind of strategy.
  // ii) If we use a Hashtable instead of a full sort then the count
  // of rows in the partition can be directly used.
  partitionStartRows_.push_back(0);

  VELOX_CHECK_GT(sortedRows_.size(), 0);
  for (auto i = 1; i < sortedRows_.size(); i++) {
    if (partitionCompare(sortedRows_[i - 1], sortedRows_[i])) {
      partitionStartRows_.push_back(i);
    }
  }

  // Setting the startRow of the (last + 1) partition to be returningRows.size()
  // to help for last partition related calculations.
  partitionStartRows_.push_back(sortedRows_.size());
}

void SortWindowBuild::sortPartitions() {
  // This is a very inefficient but easy implementation to order the input rows
  // by partition keys + sort keys.
  // Sort the pointers to the rows in RowContainer (data_) instead of sorting
  // the rows.
  sortedRows_.resize(numRows_);
  RowContainerIterator iter;
  data_->listRows(&iter, numRows_, sortedRows_.data());

  std::sort(
      sortedRows_.begin(),
      sortedRows_.end(),
      [this](const char* leftRow, const char* rightRow) {
        return compareRowsWithKeys(leftRow, rightRow, allKeyInfo_);
      });

  computePartitionStartRows();
}

void SortWindowBuild::noMoreInput() {
  if (numRows_ == 0) {
    return;
  }
  // At this point we have seen all the input rows. The operator is
  // being prepared to output rows now.
  // To prepare the rows for output in SortWindowBuild they need to
  // be separated into partitions and sort by ORDER BY keys within
  // the partition. This will order the rows for getOutput().
  sortPartitions();
}

WindowPartition* SortWindowBuild::nextPartition() {
  if (partitionStartRows_.size() == 0) {
    // There are no partitions available at this point.
    return nullptr;
  }

  currentPartition_++;
  if (currentPartition_ > partitionStartRows_.size() - 2) {
    // All partitions are output. No more partitions available.
    return nullptr;
  }

  // There is partition data available now.
  auto partitionSize = partitionStartRows_[currentPartition_ + 1] -
      partitionStartRows_[currentPartition_];
  auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[currentPartition_],
      partitionSize);
  windowPartition_->resetPartition(partition);

  return windowPartition_.get();
}

} // namespace facebook::velox::exec
