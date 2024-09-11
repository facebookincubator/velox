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

#include "velox/common/base/SpillConfig.h"
#include "velox/common/compression/Compression.h"
#include "velox/exec/HashBitRange.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {

/// Manages spilling data for an operator.
class Spiller {
 public:
  using SpillRows = std::vector<char*, memory::StlAllocator<char*>>;

  /// Spills all the rows from 'this' to disk. The spilled rows stays in the
  /// row container. The caller needs to erase the spilled rows from the row
  /// container.
  virtual void spill();

  /// Finishes spilling and accumulate the spilled partition metadata in
  /// 'partitionSet' indexed by spill partition id.
  void finishSpill(SpillPartitionSet& partitionSet);

  common::SpillStats stats() const;

  std::string toString() const;

 protected:
  /// Extracts up to 'maxRows' or 'maxBytes' from 'rows' into 'spillVector'. The
  /// extract starts at nextBatchIndex and updates nextBatchIndex to be the
  /// index of the first non-extracted element of 'rows'. Returns the byte size
  /// of the extracted rows.
  int64_t extractSpillVector(
      SpillRows& rows,
      int32_t maxRows,
      int64_t maxBytes,
      RowVectorPtr& spillVector,
      size_t& nextBatchIndex);

  const SpillState& state() const {
    return state_;
  }

  const HashBitRange& hashBits() const {
    return bits_;
  }

  bool isSpilled(int32_t partition) const {
    return state_.isPartitionSpilled(partition);
  }

  /// Indicates if all the partitions have spilled.
  bool isAllSpilled() const {
    return state_.isAllPartitionSpilled();
  }

  /// Indicates if any one of the partitions has spilled.
  bool isAnySpilled() const {
    return state_.isAnyPartitionSpilled();
  }

  /// Returns the spilled partition number set.
  SpillPartitionNumSet spilledPartitionSet() const {
    return state_.spilledPartitionSet();
  }

#if 0
  /// Invokes to set a set of 'partitions' as spilling.
  void setPartitionsSpilled(const SpillPartitionNumSet& partitions) {
    VELOX_CHECK_EQ(
        type_,
        Spiller::Type::kHashJoinProbe,
        "Unexpected spiller type: ",
        typeName(type_));
    for (const auto& partition : partitions) {
      state_.setPartitionSpilled(partition);
    }
  }
#endif

  /// Indicates if this spiller has finalized or not.
  bool finalized() const {
    return finalized_;
  }

  Spiller(
      RowContainer* container,
      RowTypePtr rowType,
      HashBitRange bits,
      int32_t numSortingKeys,
      const std::vector<CompareFlags>& sortCompareFlags,
      bool recordProbedFlag,
      const common::GetSpillDirectoryPathCB& getSpillDirPathCb,
      const common::UpdateAndCheckSpillLimitCB& updateAndCheckSpillLimitCb,
      const std::string& fileNamePrefix,
      uint64_t targetFileSize,
      uint64_t writeBufferSize,
      common::CompressionKind compressionKind,
      folly::Executor* executor,
      uint64_t maxSpillRunRows,
      const std::string& fileCreateConfig,
      folly::Synchronized<common::SpillStats>* spillStats);

  // Invoked to spill. If 'startRowIter' is not null, then we only spill rows
  // from row container starting at the offset pointed by 'startRowIter'.
  void spill(const RowContainerIterator* startRowIter);

  // Extracts the keys, dependents or accumulators for 'rows' into '*result'.
  // Creates '*results' in spillPool() if nullptr. Used from Spiller and
  // RowContainerSpillMergeStream.
  void extractSpill(folly::Range<char**> rows, RowVectorPtr& result);

  // Returns a mergeable stream that goes over unspilled in-memory
  // rows for the spill partition  'partition'. finishSpill()
  // first and 'partition' must specify a partition that has started spilling.
  std::unique_ptr<SpillMergeStream> spillMergeStreamOverRows(int32_t partition);

  // Invoked to finalize the spiller and flush any buffered spill to disk.
  void finalizeSpill();

  // Represents a run of rows from a spillable partition of
  // a RowContainer. Rows that hash to the same partition are accumulated here
  // and sorted in the case of sorted spilling. The run is then
  // spilled into storage as multiple batches. The rows are deleted
  // from this and the RowContainer as they are written. When 'rows'
  // goes empty this is refilled from the RowContainer for the next
  // spill run from the same partition.
  struct SpillRun {
    explicit SpillRun(memory::MemoryPool& pool)
        : rows(0, memory::StlAllocator<char*>(pool)) {}
    // Spillable rows from the RowContainer.
    SpillRows rows;
    // The total byte size of rows referenced from 'rows'.
    uint64_t numBytes{0};
    // True if 'rows' are sorted on their key.
    bool sorted{false};

    void clear() {
      rows.clear();
      // Clears the memory allocated in rows after a spill run finishes.
      rows.shrink_to_fit();
      numBytes = 0;
      sorted = false;
    }

    std::string toString() const {
      return fmt::format(
          "[{} ROWS {} BYTES {}]",
          rows.size(),
          numBytes,
          sorted ? "SORTED" : "UNSORTED");
    }
  };

  struct SpillStatus {
    const int32_t partition;
    const int32_t rowsWritten;
    const std::exception_ptr error;

    SpillStatus(
        int32_t _partition,
        int32_t _numWritten,
        std::exception_ptr _error)
        : partition(_partition), rowsWritten(_numWritten), error(_error) {}
  };

  void checkEmptySpillRuns() const;

  // Marks all the partitions have been spilled as we don't support
  // fine-grained spilling as for now.
  void markAllPartitionsSpilled();

  // Prepares spill runs for the spillable data from all the hash partitions.
  // If 'startRowIter' is not null, we prepare runs starting from the offset
  // pointed by 'startRowIter'.
  // The function returns true if it is the last spill run.
  bool fillSpillRuns(RowContainerIterator* startRowIter = nullptr);

  // Prepares spill run of a single partition for the spillable data from the
  // rows.
  void fillSpillRun(std::vector<char*>& rows);

  // Writes out all the rows collected in spillRuns_.
  void runSpill(bool lastRun);

  // Sorts 'run' if not already sorted.
  void ensureSorted(SpillRun& run);

  // Function for writing a spill partition on an executor. Writes to
  // 'partition' until all rows in spillRuns_[partition] are written
  // or spill file size limit is exceeded. Returns the number of rows
  // written.
  std::unique_ptr<SpillStatus> writeSpill(int32_t partition);

  void updateSpillFillTime(uint64_t timeUs);

  void updateSpillSortTime(uint64_t timeUs);

  // NOTE: for hash join probe type, there is no associated row container for
  // the spiller.
  RowContainer* const container_{nullptr};
  folly::Executor* const executor_;
  const HashBitRange bits_;
  const RowTypePtr rowType_;
  const bool spillProbedFlag_;
  const uint64_t maxSpillRunRows_;
  folly::Synchronized<common::SpillStats>* const spillStats_;

  // True if all rows of spilling partitions are in 'spillRuns_', so
  // that one can start reading these back. This means that the rows
  // that are not written out and deleted will be captured by
  // spillMergeStreamOverRows().
  bool finalized_{false};

  SpillState state_;

  // Collects the rows to spill for each partition.
  std::vector<SpillRun> spillRuns_;
};

spiller_ = std::make_unique<Spiller>(
    Spiller::Type::kHashJoinBuild,
    joinType_,
    table_->rows(),
    spillType_,
    HashBitRange(
        startPartitionBit, startPartitionBit + config->numPartitionBits),
    config,
    &spillStats_);

class SpillerWithoutSort : public Spiller {
  SpillerWithoutSort(
      RowContainer* rows,
      RowTypePtr rowType,
      const HashBitRange& hashBitRange,
      bool recordProbedFlag,
      const common::SpillConfig* spillConfig,
      folly::Synchronized<common::SpillStats>* spillStats);

  /// Append 'spillVector' into the spill file of given 'partition'. It is now
  /// only used by the spilling operator which doesn't need data sort, such as
  /// hash join build and hash join probe.
  ///
  /// NOTE: the spilling operator should first mark 'partition' as spilling and
  /// spill any data buffered in row container before call this.
  virtual void spill(uint32_t partition, const RowVectorPtr& spillVector);
};

class HashProbeSpiller : public SpillerWithoutSort {
 public:
  HashProbeSpiller(
      RowTypePtr rowType,
      const HashBitRange& hashBitRange,
      const common::SpillConfig* spillConfig,
      folly::Synchronized<common::SpillStats>* spillStats);

  void spill() override {
    VELOX_UNSUPPORTED("{} for SortOutputSpiller");
  }
};

class HashTableSpiller : public Spiller {
 public:
  RowNumberSpiller(
      RowTypePtr rowType,
      const HashBitRange& hashBitRange,
      const common::SpillConfig* spillConfig,
      folly::Synchronized<common::SpillStats>* spillStats);
};

/// Manages spilling data for an operator which needs to sort the spill data.
class SortedSpiller : public Spiller {};

/// Manages spilling data during the aggregate output processing .
class AggregateOutputSpiller : public Spiller {
  // AggregateOutputSpiller();

  /// Spills all rows starting from 'startRowIter'. The spilled rows still stays
  /// in the row container. The caller needs to erase them from the row
  /// container.
  void spill(const RowContainerIterator& startRowIter);
};

/// Manages spilling data during sort buffer output processing.
class SortOutputSpiller : public Spiller {
 public:
  SortOutputSpiller(
      RowContainer* rows,
      const RowTypePtr& rowType,
      const common::SpillConfig* spillConfig,
      folly::Synchronized<common::SpillStats>* spillStats);

  void spill() override {
    VELOX_UNSUPPORTED("{} for SortOutputSpiller");
  }

  /// Spills all the rows pointed by rows. The spilled rows still stays in the
  /// row container. The caller needs to erase them from the row container.
  void spill(std::vector<char*>& rows);
};
} // namespace facebook::velox::exec
