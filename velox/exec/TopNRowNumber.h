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
#include "velox/exec/Operator.h"
#include "velox/exec/Spiller.h"

namespace facebook::velox::exec {
class TopNRowNumberSpiller;

/// TopNRowNumber is an optimized version of a Window operator with a
/// single row_number or rank or dense_rank window function followed by a
/// rank <= N filter. N must be >= 0. If the TopNRowNumber has no partition
/// keys, then all the rows belong to a single partition. However, the
/// TopNRowNumber should have at least one sorting key specified.
///
/// TopNRowNumber is more efficient than a general Window operator as it does
/// not store all rows of a partition. Instead, it only keeps the top N
/// rows of the partition at any point.
///
/// The operator partitions the input using specified partitioning keys,
/// and maintains a TopRows structure per partition in a HashTable. The TopRows
/// maintains a priority queue of row pointers. The priority queue is
/// kept ordered by sorting keys of the TopNRowNumber. The TopRows only retains
/// rows whose ranks satisfy the filter condition (so rank <= N). N is also
/// called the limit of the operator. To aid this filtering, the TopRows tracks
/// the greatest rank seen for each partition.
///
/// The operator processes all input rows before beginning to output rows.
///
/// For each input row, it retrieves the TopRows corresponding to the partition
/// keys. The TopRows is first filled until it has N rows. Thereafter, new rows
/// are compared with the top row in the TopRows priority queue.
/// If the new rows order by values are less than (for ASC) or greater than
/// (for DESC) so row rank <= topRank, then the row is added to TopRows.
/// For each outcome, the greatest rank of the TopRows is updated as per the
/// ranking function logic.
/// For each function type, the rank maintenance logic is in:
/// - processRowWithinLimit() function when the TopRows is filling the first
///   N rows.
/// - processRowExceedingLimit() function when the TopRows already has N rows.
///
/// After processing all the input rows, the operator proceeds to output the
/// rows. The rows might all be in memory or spilled to disk if memory
/// reclamation was triggered during processing.
///
/// If the rows are in memory, then the operator iterates over each partition
/// in the HashTable, and starts outputting rows from the partition. The
/// TopRows structure maintains the rows in descending order of their ranks
/// (greatest rank at the top of the priority queue). So when outputting,
/// the operator first fixes the top rank of the partition using fixTopRank()
/// and then computes the ranks of each row using computeNextRankInMemory().
/// The logic of the next rank differs based on the ranking function.
///
/// If the rows are in the spill, then the spiller iterates over each spilled
/// partition in order of the ranks. For each row from the spill, the next
/// rank is computed using computeNextRankInSpill() function. The logic of
/// the next rank differs based on the ranking function.
/// Note : The spill could have > limit rows for a partition as each spill
/// resets the TopRows for the partition. So stop outputting rows after
/// reaching the limit for each partition.

class TopNRowNumber : public Operator {
 public:
  TopNRowNumber(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  bool needsInput() const override {
    if (abandonedPartial_ && (data_->numRows() > 0 || input_ != nullptr)) {
      // This operator switched to a pass-through and needs to produce output
      // before receiving more input.
      return false;
    }

    return true;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  void noMoreInput() override;

  bool isFinished() override;

  void close() override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

 private:
  // This structure holds the top rows for a partition. It uses a priority
  // queue to maintain the top rows in order of their ranks. Note the rank
  // logic depends on the respective function (row_number, rank or dense_rank).
  // However, a common requirement across all three is to maintain the rows in
  // order of their sort keys so that the greatest rank row is always at the top
  // of the queue. This ordering is done using the RowComparator passed to the
  // TopRows.
  //
  // The number of rows in TopRows are limited to 'limit' specified for the
  // operator. The greatest rank of the rows in TopRows is maintained in the
  // 'topRank' variable.
  //
  // The TopRows structure is first filled in order to collect 'limit'
  // rows. Thereafter, new rows are compared with the top row and either kept
  // or discarded and the new top rank is updated. The rank computation differs
  // based on the ranking function. This structure has methods for abstractions
  // used for the top rank maintenance algorithms.
  struct TopRows {
    struct Compare {
      RowComparator& comparator;

      bool operator()(const char* lhs, const char* rhs) {
        return comparator(lhs, rhs);
      }
    };

    std::priority_queue<char*, std::vector<char*, StlAllocator<char*>>, Compare>
        rows;

    RowComparator& rowComparator;

    // This is the greatest rank seen so far in the input rows. Note: rank is
    // the result of the respective function computation (row_number, rank or
    // dense_rank). It is compared with the expected limit for the operator.
    int64_t topRank = 0;

    // Number of rows with the highest rank in the partition.
    vector_size_t numTopRankRows();

    // Remove all rows with the highest rank in the partition.
    // Returns a pointer to the last removed row.
    char* removeTopRankRows();

    // Returns true if the row at position index in decodedVectors
    // has the same order by keys as another row in the TopRows
    // priority_vector.
    bool isDuplicate(
        const std::vector<DecodedVector>& decodedVectors,
        vector_size_t index);

    TopRows(HashStringAllocator* allocator, RowComparator& comparator)
        : rows{{comparator}, StlAllocator<char*>(allocator)},
          rowComparator(comparator) {}
  };

  void initializeNewPartitions();

  // Cleans up any newly inserted but uninitialized partitions from the hash
  // table. This is called when groupProbe throws (e.g., due to OOM) to ensure
  // close() doesn't crash trying to destroy uninitialized TopRows structures.
  void cleanupNewPartitions();

  TopRows& partitionAt(char* group) {
    return *reinterpret_cast<TopRows*>(group + partitionOffset_);
  }

  // Decodes and potentially loads input if lazy vector.
  void prepareInput(RowVectorPtr& input);

  // Handles input row when the partition has not yet accumulated 'limit' rows.
  // Returns a pointer to the row to add to the partition accumulator.
  template <core::TopNRowNumberNode::RankFunction TRank>
  char* processRowWithinLimit(vector_size_t index, TopRows& partition);

  // Handles input row when the partition has already accumulated 'limit' rows.
  // Returns a pointer to the row to add to the partition accumulator.
  template <core::TopNRowNumberNode::RankFunction TRank>
  char* processRowExceedingLimit(vector_size_t index, TopRows& partition);

  // Loop to process the numInput input rows received by the operator.
  template <core::TopNRowNumberNode::RankFunction TRank>
  void processInputRowLoop(vector_size_t numInput);

  // Adds input row to a partition or discards the row.
  template <core::TopNRowNumberNode::RankFunction TRank>
  void processInputRow(vector_size_t index, TopRows& partition);

  // Returns next partition to add to output or nullptr if there are no
  // partitions left.
  TopRows* nextPartition();

  // If there are many rows with the highest rank, then the topRank
  // of the partition can oscillate between a very small value and a
  // value > limit. Fix the partition for this condition before starting to
  // output the partition.
  vector_size_t fixTopRank(TopRows& partition);

  // Computes the rank for the next row to be output
  // (all output rows in memory).
  template <core::TopNRowNumberNode::RankFunction TRank>
  void computeNextRankInMemory(
      const TopRows& partition,
      vector_size_t rowIndex);

  // Appends numRows of the current partition to the output. Note: The rows are
  // popped in reverse order of the rank.
  // NOTE: This function erases the yielded output rows from the partition
  // and the next call starts with the remaining rows.
  template <core::TopNRowNumberNode::RankFunction TRank>
  void appendPartitionRows(
      TopRows& partition,
      vector_size_t numRows,
      vector_size_t outputOffset,
      FlatVector<int64_t>* rowNumbers);

  bool spillEnabled() const {
    return spillConfig_.has_value();
  }

  void ensureInputFits(const RowVectorPtr& input);

  // Sorts, spills and clears all of 'data_'. Clears 'table_'.
  void spill();

  void setupSpiller();

  template <core::TopNRowNumberNode::RankFunction TRank>
  RowVectorPtr getOutputFromSpill();

  RowVectorPtr getOutputFromMemory();

  // Returns true if 'next' row belongs to a different partition then index-1
  // row of output.
  bool isNewPartition(
      const RowVectorPtr& output,
      vector_size_t index,
      const SpillMergeStream* next);

  // Returns true if 'next' row is a new rank (rows differ on order by keys)
  // of the previous row in the partition (at output[index] of the
  // output block).
  bool isNewRank(
      const RowVectorPtr& output,
      vector_size_t index,
      const SpillMergeStream* next);

  // Utility method to compare values from startColumn to endColumn for
  // 'next' row from SpillMergeStream with current row of output (at index).
  bool compareSpillRowColumns(
      const RowVectorPtr& output,
      vector_size_t index,
      const SpillMergeStream* next,
      vector_size_t startColumn,
      vector_size_t endColumn);

  // Computes next rank value for spill output.
  template <core::TopNRowNumberNode::RankFunction TRank>
  inline void computeNextRankInSpill(
      const RowVectorPtr& output,
      vector_size_t index,
      const SpillMergeStream* next);

  // Checks if next row in 'merge_' belongs to a different partition than last
  // row in 'output' and if so updates nextRank_ and numPeers_ to 1.
  // Also, checks current partition reached the limit on rank and
  // if so advances 'merge_' to the first row on the next
  // partition and sets nextRank_ and numPeers_ to 0.
  //
  // @post 'merge_->next()' is either at end or points to a row that should be
  // included in the next output batch using 'nextRank_'.
  template <core::TopNRowNumberNode::RankFunction TRank>
  void setupNextOutput(const RowVectorPtr& output);

  // Called in noMoreInput() and spill().
  void updateEstimatedOutputRowSize();

  // Return true if this operator runs a 'partial' stage and doesn't not reduce
  // cardinality sufficiently. Returns false if spilling was triggered earlier.
  bool abandonPartialEarly() const;

  // Rank function semantics of operator.
  const core::TopNRowNumberNode::RankFunction rankFunction_;

  const int32_t limit_;

  const bool generateRowNumber_;

  const size_t numPartitionKeys_;
  const size_t numSortingKeys_;

  // Input columns in the order of: partition keys, sorting keys, the rest.
  const std::vector<column_index_t> inputChannels_;

  // Input column types in 'inputChannels_' order.
  const RowTypePtr inputType_;

  // Compare flags for partition and sorting keys. Compare flags for partition
  // keys are set to default values. Compare flags for sorting keys match
  // sorting order specified in the plan node.
  //
  // Used to sort 'data_' while spilling.
  const std::vector<CompareFlags> spillCompareFlags_;

  const vector_size_t abandonPartialMinRows_;

  const int32_t abandonPartialMinPct_;

  // True if this operator runs a 'partial' stage without sufficient reduction
  // in cardinality. In this case, it becomes a pass-through.
  bool abandonedPartial_{false};

  // Hash table to keep track of partitions. Not used if there are no
  // partitioning keys. For each partition, stores an instance of TopRows
  // struct.
  std::unique_ptr<BaseHashTable> table_;

  std::unique_ptr<HashLookup> lookup_;

  int32_t partitionOffset_;

  // TopRows struct to keep track of top rows for a single partition, when
  // there are no partitioning keys.
  std::unique_ptr<HashStringAllocator> allocator_;

  std::unique_ptr<TopRows> singlePartition_;

  // Stores input data. For each partition, only up to 'limit_' rows are stored.
  // Order of columns matches 'inputChannels_': partition keys, sorting keys,
  // the rest.
  //
  // Partition and sorting columns are specified as 'keys'. The rest of the
  // columns are specified as 'dependents'. This enables sorting 'data_' using
  // 'spillCompareFlags_' when spilling.
  std::unique_ptr<RowContainer> data_;

  RowComparator comparator_;

  std::vector<DecodedVector> decodedVectors_;

  bool finished_{false};

  // Size of a single output row estimated using 'data_->estimateRowSize()'.
  // If spilling, this value is set to max 'data_->estimateRowSize()' across all
  // accumulated 'data_'.
  std::optional<int64_t> estimatedOutputRowSize_;

  // Maximum number of rows in the output batch.
  vector_size_t outputBatchSize_;

  // The below variables are used when outputting from memory.
  // Vector of pointers to individual rows in the RowContainer for the current
  // output block.
  std::vector<char*> outputRows_;
  // Number of partitions to fetch from a HashTable in a single listAllRows
  // call.
  static const size_t kPartitionBatchSize = 100;

  BaseHashTable::RowsIterator partitionIt_;
  std::vector<char*> partitions_{kPartitionBatchSize};
  size_t numPartitions_{0};

  // This is the index of the current partition within partitions_ which is
  // obtained from the HashTable iterator.
  std::optional<int32_t> outputPartitionNumber_;
  // This is the currentPartition being output. It is possible that the
  // partition is output across multiple output blocks.
  TopNRowNumber::TopRows* outputPartition_{nullptr};

  // The below variables are used when outputting from the spiller.
  // Spiller for contents of the 'data_'.
  std::unique_ptr<SortInputSpiller> spiller_;

  // Used to sort-merge spilled data.
  std::unique_ptr<TreeOfLosers<SpillMergeStream>> merge_;

  // Row number/rank or dense_rank for the first row in the next output batch
  // from the spiller.
  vector_size_t nextRank_{1};
  // Number of peers of first row in the previous output batch. This is used
  // in rank calculation.
  vector_size_t numPeers_{1};
};
} // namespace facebook::velox::exec
