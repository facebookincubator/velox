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

#include "velox/exec/HashWindowBuild.h"

namespace facebook::velox::exec {

HashWindowBuild::HashWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool)
    : WindowBuild(windowNode, pool),
      inputType_(windowNode->sources()[0]->outputType()),
      comparator_(
          inputType_,
          windowNode->sortingKeys(),
          windowNode->sortingOrders(),
          data_.get()) {
  const auto& keys = windowNode->partitionKeys();
  const auto numKeys = keys.size();

  if (numKeys > 0) {
    Accumulator accumulator{true, sizeof(SortedRows), false, 1, [](auto) {}};

    table_ = std::make_unique<HashTable<false>>(
        createVectorHashers(inputType_, keys),
        std::vector<Accumulator>{accumulator},
        std::vector<TypePtr>{},
        false, // allowDuplicates
        false, // isJoinBuild
        false, // hasProbedFlag
        0, // minTableSizeForParallelJoinBuild
        pool);
    sortedRowsOffset_ = table_->rows()->columnAt(numKeys).offset();
    lookup_ = std::make_unique<HashLookup>(table_->hashers());
  } else {
    allocator_ = std::make_unique<HashStringAllocator>(pool);
    singlePartition_ =
        std::make_unique<SortedRows>(allocator_.get(), comparator_);
  }
}

void HashWindowBuild::addInput(RowVectorPtr input) {
  const auto numInput = input->size();

  for (auto i = 0; i < inputType_->size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(i));
  }

  if (table_) {
    SelectivityVector rows(numInput);
    lookup_->reset(numInput);
    table_->prepareForProbe(*lookup_, input, rows, false);
    table_->groupProbe(*lookup_);

    // Initialize new partitions.
    initializeNewPartitions();

    // Process input rows. For each row, lookup the partition. Add the
    // new row to the partition. The SortedRows priority queue will
    // insert the row maintaining the order of the ORDER BY keys.
    for (auto i = 0; i < numInput; ++i) {
      auto& partition = sortedRowsAt(lookup_->hits[i]);
      addRowToPartition(input, i, partition);
    }
  } else {
    for (auto i = 0; i < numInput; ++i) {
      addRowToPartition(input, i, *singlePartition_);
    }
  }
}

void HashWindowBuild::initializeNewPartitions() {
  for (auto index : lookup_->newGroups) {
    new (lookup_->hits[index] + sortedRowsOffset_)
        SortedRows(table_->stringAllocator(), comparator_);
  }
}

void HashWindowBuild::addRowToPartition(
    const RowVectorPtr& input,
    vector_size_t index,
    SortedRows& partition) {
  char* newRow = data_->newRow();
  for (auto col = 0; col < input->childrenSize(); ++col) {
    data_->store(decodedInputVectors_[col], index, newRow, col);
  }

  auto& sortedRows = partition.rows;
  sortedRows.push(newRow);
}

bool HashWindowBuild::hasNextPartition() {
  if (!noMoreInput_) {
    // HashBuild not completed so no partitions available.
    return false;
  }

  currentSortedRows_ = nextSortedRows();
  if (!currentSortedRows_) {
    return false;
  }

  return true;
}

std::unique_ptr<WindowPartition> HashWindowBuild::nextPartition() {
  VELOX_CHECK(noMoreInput_, "No window partitions available");
  VELOX_CHECK(currentSortedRows_, "All window partitions consumed");

  partitionSortedRows_.clear();
  auto numSortedRows = currentSortedRows_->rows.size();
  partitionSortedRows_.reserve(numSortedRows);
  for (auto i = 0; i < numSortedRows; i++) {
    partitionSortedRows_.push_back(currentSortedRows_->rows.top());
    currentSortedRows_->rows.pop();
  }
  std::reverse(partitionSortedRows_.begin(), partitionSortedRows_.end());

  auto windowPartition = std::make_unique<WindowPartition>(
      data_.get(), inputColumns_, sortKeyInfo_);

  auto partition =
      folly::Range(partitionSortedRows_.data(), partitionSortedRows_.size());
  windowPartition->resetPartition(partition);

  return windowPartition;
}

HashWindowBuild::SortedRows* HashWindowBuild::nextSortedRows() {
  if (!table_) {
    if (!currentPartition_) {
      currentPartition_ = 0;
      return singlePartition_.get();
    }
    return nullptr;
  }

  if (!currentPartition_) {
    numPartitions_ = table_->listAllRows(
        &partitionIt_,
        partitions_.size(),
        RowContainer::kUnlimited,
        partitions_.data());
    if (numPartitions_ == 0) {
      // No more partitions.
      return nullptr;
    }

    currentPartition_ = 0;
  } else {
    ++currentPartition_.value();
    if (currentPartition_ >= numPartitions_) {
      currentPartition_.reset();
      return nextSortedRows();
    }
  }

  return &sortedRowsAt(partitions_[currentPartition_.value()]);
}

} // namespace facebook::velox::exec
