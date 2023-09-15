
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

#include "velox/exec/HashTable.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/WindowBuild.h"

namespace facebook::velox::exec {

// This WindowBuild class uses a HashTable to obtain Window partitions
// The input rows are hashed by Window partition keys.
// Each Window partition row has an Accumulator which is a priority
// queue of pointers to the rows in it. The priority queue
// uses the sorting keys of the Window partition to
// order the rows in it.
// All input rows are seen before any partitions are processed for
// output.
class HashWindowBuild : public WindowBuild {
 public:
  HashWindowBuild(
      const std::shared_ptr<const core::WindowNode>& windowNode,
      velox::memory::MemoryPool* pool);

  bool needsInput() override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override {
    noMoreInput_ = true;
  }

  bool hasNextPartition() override;

  std::unique_ptr<WindowPartition> nextPartition() override;

 private:
  // A priority queue to keep track of rows for a given partition.
  struct SortedRows {
    struct Compare {
      RowComparator& comparator;

      bool operator()(const char* lhs, const char* rhs) {
        return comparator(lhs, rhs);
      }
    };

    std::priority_queue<char*, std::vector<char*, StlAllocator<char*>>, Compare>
        rows;

    SortedRows(HashStringAllocator* allocator, RowComparator& comparator)
        : rows{{comparator}, StlAllocator<char*>(allocator)} {}
  };

  void initializeNewPartitions();

  // Adds input row to a partition or discards the row.
  void addRowToPartition(
      const RowVectorPtr& input,
      vector_size_t index,
      SortedRows& partition);

  SortedRows& sortedRowsAt(char* group) {
    return *reinterpret_cast<SortedRows*>(group + sortedRowsOffset_);
  }

  SortedRows* nextSortedRows();

  const RowTypePtr inputType_;
  RowComparator comparator_;

  // Hash table to keep track of partitions. Not used if there are no
  // partitioning keys. For each partition, stores an instance of SortedRows
  // struct.
  std::unique_ptr<BaseHashTable> table_;
  std::unique_ptr<HashLookup> lookup_;
  int32_t sortedRowsOffset_;

  // SortedRows struct to keep track of rows for a single partition, when
  // there are no partitioning keys.
  std::unique_ptr<HashStringAllocator> allocator_;
  std::unique_ptr<SortedRows> singlePartition_;

  // Number of partitions to fetch from a HashTable in a single listAllRows
  // call.
  static const size_t kPartitionBatchSize = 100;

  // The below variables are populated for the listAllRows call.
  BaseHashTable::RowsIterator partitionIt_;
  std::vector<char*> partitions_{kPartitionBatchSize};
  size_t numPartitions_{0};

  std::optional<int32_t> currentPartition_;
  SortedRows* currentSortedRows_;
  std::vector<char*> partitionSortedRows_;

  bool noMoreInput_ = false;
};

} // namespace facebook::velox::exec
