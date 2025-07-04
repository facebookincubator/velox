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

#include "velox/exec/HashBuild.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/Operator.h"
#include "velox/exec/ProbeOperatorState.h"
#include "velox/exec/VectorHasher.h"

namespace facebook::velox::exec {

// Probes a hash table made by HashBuild.
class HashProbe : public Operator {
 public:
  HashProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::HashJoinNode>& hashJoinNode);

  void initialize() override;

  bool needsInput() const override {
    if (state_ == ProbeOperatorState::kFinish || noMoreInput_ ||
        noMoreSpillInput_ || input_ != nullptr) {
      return false;
    }
    if (table_) {
      return true;
    }
    // NOTE: if we can't apply dynamic filtering, then we can start early to
    // read input even before the hash table has been built.
    return operatorCtx_->driverCtx()
        ->driver->canPushdownFilters(this, keyChannels_)
        .empty();
  }

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

  void close() override;

  bool canReclaim() const override;

  const std::vector<IdentityProjection>& tableOutputProjections() const {
    return tableOutputProjections_;
  }

  ExprSet* filterExprSet() const {
    return filter_.get();
  }

  /// Returns the type for the hash table row. Build side keys first,
  /// then dependent build side columns.

  static RowTypePtr makeTableType(
      const RowType* type,
      const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
          keys);

  const std::shared_ptr<HashJoinBridge>& joinBridge() const {
    return joinBridge_;
  }

  bool testingHasInputSpiller() const {
    return inputSpiller_ != nullptr;
  }

  bool testingExceededMaxSpillLevelLimit() const {
    return exceededMaxSpillLevelLimit_;
  }

  bool testingHasPendingInput() const {
    return input_ != nullptr;
  }

  std::shared_ptr<BaseHashTable> testingTable() const {
    return table_;
  }

  ProbeOperatorState testingState() const {
    return state_;
  }

 private:
  // Indicates if the join type includes misses from the left side in the
  // output.
  static bool joinIncludesMissesFromLeft(core::JoinType joinType) {
    return isLeftJoin(joinType) || isFullJoin(joinType) ||
        isAntiJoin(joinType) || isLeftSemiProjectJoin(joinType);
  }

  void setState(ProbeOperatorState state);
  void checkStateTransition(ProbeOperatorState state);

  void setRunning();
  void checkRunning() const;
  bool isRunning() const;
  bool isWaitingForPeers() const;

  // Returns true if all probe groups finished execution. If false, the join
  // bridge will prepare reprocessing for the next execution group from the
  // probe side. This is not a reliable signal, but rather a best effort signal.
  // Applies only for mixed grouped execution mode.
  bool allProbeGroupFinished() const;

  void pushdownDynamicFilters();

  // Invoked to wait for the hash table to be built by the hash build operators
  // asynchronously. The function also sets up the internal state for
  // potentially spilling input or reading spilled input or recursively spill
  // the hash table.
  void asyncWaitForHashTable();

  // Sets up 'filter_' and related members.
  void initializeFilter(
      const core::TypedExprPtr& filter,
      const RowTypePtr& probeType,
      const RowTypePtr& tableType);

  // Setup 'resultIter_'.
  void initializeResultIter();

  // If 'toSpillOutput', the produced output is spilled to disk for memory
  // arbitration.
  RowVectorPtr getOutputInternal(bool toSpillOutput);

  // Check if output_ can be re-used and if not make a new one.
  void prepareOutput(vector_size_t size);

  // Populate output columns.
  void fillOutput(vector_size_t size);

  // Populate 'match' output column for the left semi join project,
  void fillLeftSemiProjectMatchColumn(vector_size_t size);

  // Clears the columns of 'output_' that are projected from
  // 'input_'. This should be done when preparing to produce a next
  // batch of output to drop any lingering references to row
  // number mappings or input vectors. In this way input vectors do
  // not have to be copied and will be singly referenced by their
  // producer.
  void clearProjectedOutput();

  // Populate output columns with matching build-side rows
  // for the right semi join and non-matching build-side rows
  // for right join and full join.
  RowVectorPtr getBuildSideOutput();

  // Applies 'filter_' to 'outputTableRows_' and updates 'outputRowMapping_'.
  // Returns the number of passing rows.
  vector_size_t evalFilter(vector_size_t numRows);

  inline bool filterPassed(vector_size_t row) {
    return filterInputRows_.isValid(row) &&
        !decodedFilterResult_.isNullAt(row) &&
        decodedFilterResult_.valueAt<bool>(row);
  }

  // Create a temporary input vector to be passed to the filter. This ensures it
  // gets destroyed in case its wrapping an unloaded vector which eventually
  // needs to be wrapped in fillOutput().
  RowVectorPtr createFilterInput(vector_size_t size);

  // Prepare filter row selectivity for null-aware join. 'numRows'
  // specifies the number of rows in 'filterInputRows_' to process. If
  // 'filterPropagateNulls' is true, the probe input row which has null in any
  // probe filter column can't pass the filter.
  void prepareFilterRowsForNullAwareJoin(
      RowVectorPtr& filterInput,
      vector_size_t numRows,
      bool filterPropagateNulls);

  // Evaluate the filter for null-aware anti or left semi project join.
  SelectivityVector evalFilterForNullAwareJoin(
      vector_size_t numRows,
      bool filterPropagateNulls);

  // Prepares the hashers for probing with null keys.
  // Initializes `nullKeyProbeHashers_` if empty, ensuring it has exactly one
  // hasher. If the table's hash mode is `kHash`, creates and decodes a null
  // input vector.
  void prepareNullKeyProbeHashers();

  // Combine the selected probe-side rows with all or null-join-key (depending
  // on the iterator) build side rows and evaluate the filter.  Mark probe rows
  // that pass the filter in 'filterPassedRows'. Used in null-aware join
  // processing.
  void applyFilterOnTableRowsForNullAwareJoin(
      const SelectivityVector& rows,
      SelectivityVector& filterPassedRows,
      std::function<int32_t(char**, int32_t)> iterator);

  void ensureLoadedIfNotAtEnd(column_index_t channel);

  void ensureLoaded(column_index_t channel);

  // Indicates if the operator has more probe inputs from either the upstream
  // operator or the spill input reader.
  bool hasMoreInput() const;

  // Indicates if the join type such as right, right semi and full joins,
  // require to produce the output results from the join results after all the
  // probe operators have finished processing the probe inputs.
  bool needLastProbe() const;

  // Indicates if the join type can skip processing probe inputs with empty
  // build table such as inner, right and semi joins.
  //
  // NOTE: if spilling is triggered at the build side, then we still need to
  // process probe inputs to spill the probe rows if the corresponding
  // partitions have been spilled at the build side.
  bool skipProbeOnEmptyBuild() const;

  // If 'spillPartitionIds' is not empty, then spilling has been triggered at
  // the build side and the function will set up 'inputSpiller_' to spill probe
  // inputs.
  void maybeSetupInputSpiller(const SpillPartitionIdSet& spillPartitionIds);

  // If 'restoredSpillPartitionId' is set, then setup 'spillInputReader_' to
  // read probe inputs from spilled data on disk.
  void maybeSetupSpillInputReader(
      const std::optional<SpillPartitionId>& restoredSpillPartitionId);

  // Checks the hash table's spill level limit from the restored table. Sets the
  // 'exceededMaxSpillLevelLimit_' accordingly.
  void checkMaxSpillLevel(
      const std::optional<SpillPartitionId>& restoredPartitionId);

  bool canSpill() const override;

  // Indicates if the probe input is read from spilled data or not.
  bool isSpillInput() const;

  // Indicates if there is more spill data to restore after finishes processing
  // the current probe inputs.
  bool hasMoreSpillData() const;

  // Indicates if the operator needs to spill probe inputs. It is true if parts
  // of the build-side rows have been spilled. Hence, the probe operator needs
  // to spill the corresponding probe-side rows as well.
  bool needToSpillInput() const;

  // This ensures there is sufficient buffer reserved to produce the next output
  // batch. This might trigger memory arbitration underneath and the probe
  // operator is set to reclaimable at this stage.
  void ensureOutputFits();

  // Setups spilled output reader if 'spillOutputPartitionSet_' is not empty.
  void maybeSetupSpillOutputReader();

  // Reads from the spilled output if the spilling has been triggered during the
  // middle of an input processing. The latter produces all the outputs and
  // spill them on to disk in case the output is too large to fit in memory in
  // some edge case, like one input row has many matches with the build side.
  // The function returns true if it has read the spilled output and saves in
  // 'output_'.
  bool maybeReadSpillOutput();

  // Invoked after finishes processing the probe inputs and there is spill data
  // remaining to restore. The function will reset the internal states which
  // are relevant to the last finished probe run. The last finished probe
  // operator will also notify the hash build operators to build the next hash
  // table from spilled data.
  void prepareForSpillRestore();

  // Invoked to read next batch of spilled probe inputs from disk to process.
  void addSpillInput();

  // Produces and spills outputs from operator which has pending input to
  // process in probe 'operators'.
  void spillOutput(const std::vector<HashProbe*>& operators);
  // Produces and spills output from this probe operator.
  void spillOutput();

  // Invoked to spill rows in 'input' to disk directly if the corresponding
  // partitions have been spilled at the build side.
  //
  // NOTE: this method keeps 'input' as is if no row needs spilling; resets it
  // to null if all rows have been spilled; wraps in a dictionary using rows
  // number that do not need spilling otherwise.
  void spillInput(RowVectorPtr& input);

  // Invoked to prepare indices buffers for input spill processing.
  void prepareInputIndicesBuffers(
      vector_size_t numInput,
      const SpillPartitionIdSet& spillPartitionIds);

  /// Decode join key inputs and populate 'nonNullInputRows_'.
  void decodeAndDetectNonNullKeys();

  // Invoked when there is no more input from either upstream task or spill
  // input. If there is remaining spilled data, then the last finished probe
  // operator is responsible for notifying the hash build operators to build the
  // next hash table from the spilled data.
  void noMoreInputInternal();

  // Indicates if this hash probe operator is under non-reclaimable state or
  // not.
  bool nonReclaimableState() const;

  // Returns the index of the 'match' column in the output for semi project
  // joins.
  VectorPtr& matchColumn() const {
    VELOX_DCHECK(
        isRightSemiProjectJoin(joinType_) || isLeftSemiProjectJoin(joinType_));
    return output_->children().back();
  }

  // Returns true if build side has no data.
  // NOTE: if build side has triggered spilling, then the first hash table
  // might be empty, but we still have spilled partition data remaining to
  // restore. Also note that the spilled partition at build side must not be
  // empty.
  bool emptyBuildSide() const {
    return table_->numDistinct() == 0 && inputSpillPartitionSet_.empty() &&
        spillInputPartitionIds_.empty();
  }

  // Find the peer hash probe operators in the same pipeline.
  std::vector<HashProbe*> findPeerOperators();

  // Wake up the peer hash probe operators when last probe operator finishes.
  void wakeupPeerOperators();

  // Invoked to release internal buffers to free up memory resources after
  // memory reclamation or operator close.
  void clearBuffers();

  // Returns the estimated row size of the projected output columns. nullopt
  // will be returned if insufficient column stats is presented in 'table_', or
  // the row size variation is too large. The row size is too large if ratio of
  // max row size and avg row size is larger than 'kToleranceRatio' which is set
  // to 10.
  std::optional<uint64_t> estimatedRowSize(
      const std::vector<vector_size_t>& varColumnsStats,
      uint64_t totalFixedColumnsBytes);

  // Returns the aggregated column stats at 'columnIndex' of 'table_'. Returns
  // nullopt if the column stats is not available.
  //
  // NOTE: The column stats is collected by default for hash join table but it
  // could be invalidated in case of spilling. But we should never expect usage
  // of an invalidated table as we always spill the entire table.
  std::optional<RowColumn::Stats> columnStats(int32_t columnIndex) const;

  // TODO: Define batch size as bytes based on RowContainer row sizes.
  const vector_size_t outputBatchSize_;

  const std::shared_ptr<const core::HashJoinNode> joinNode_;

  const core::JoinType joinType_;

  const bool nullAware_;

  const RowTypePtr probeType_;

  std::shared_ptr<HashJoinBridge> joinBridge_;

  ProbeOperatorState state_{ProbeOperatorState::kWaitForBuild};

  // Used for synchronization with the hash probe operators of the same pipeline
  // to handle the last probe processing for certain types of join and notify
  // the hash build operators to build the next hash table from spilled data if
  // disk spilling has been triggered.
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  // Used with 'future_' for the same purpose. 'promises_' will be fulfilled
  // after finishes the last probe processing and hash build operator
  // notification.
  //
  // NOTE: for a given probe operator, it has either 'future_' or 'promises_'
  // set, but not both. All but last probe operators get futures that are
  // fulfilled when the last probe operator finishes. The last probe operator
  // doesn't get a future, but collects the promises from the Task and uses
  // those to wake up peers.
  std::vector<ContinuePromise> promises_;

  // True if this is the last hash probe operator in the pipeline that finishes
  // probe input processing. It is responsible for producing matching build-side
  // rows for the right semi join and non-matching build-side rows for right
  // join and full join. If the disk spilling is enabled, the last operator is
  // also responsible to notify the hash build operators to build the next hash
  // table from the previously spilled data.
  bool lastProber_{false};

  std::unique_ptr<HashLookup> lookup_;

  // Channel of probe keys in 'input_'.
  std::vector<column_index_t> keyChannels_;

  folly::F14FastSet<column_index_t> dynamicFiltersProducedOnChannels_;

  // True if the join can become a no-op starting with the next batch of input.
  bool canReplaceWithDynamicFilter_{false};

  // True if the join became a no-op after pushing down the filter.
  bool replacedWithDynamicFilter_{false};

  std::vector<std::unique_ptr<VectorHasher>> hashers_;

  // Current working hash table that is shared between other HashProbes in other
  // Drivers of the same pipeline.
  std::shared_ptr<BaseHashTable> table_;

  // Indicates whether there was no input. Used for right semi join project.
  bool noInput_{true};

  // Indicates whether to skip probe input data processing or not. It only
  // applies for a specific set of join types (see skipProbeOnEmptyBuild()), and
  // the build table is empty and the probe input is read from non-spilled
  // source. This ensures the hash probe operator keeps running until all the
  // probe input from the sources have been processed. It prevents the exchange
  // hanging problem at the producer side caused by the early query finish.
  bool skipInput_{false};

  // Indicates whether there are rows with null join keys on the build
  // side. Used by anti and left semi project join.
  bool buildSideHasNullKeys_{false};

  // Indicates whether there are rows with null join keys on the probe
  // side. Used by right semi project join.
  bool probeSideHasNullKeys_{false};

  // Rows in the filter columns to apply 'filter_' to.
  SelectivityVector filterInputRows_;

  // Join filter.
  std::unique_ptr<ExprSet> filter_;

  std::vector<VectorPtr> filterResult_;
  DecodedVector decodedFilterResult_;

  // Type of the RowVector for filter inputs.
  RowTypePtr filterInputType_;

  // The input channels that are projected to the output.
  folly::F14FastMap<column_index_t, column_index_t> projectedInputColumns_;

  // Maps input channels to channels in 'filterInputType_'.
  std::vector<IdentityProjection> filterInputProjections_;

  // Maps from column index in hash table to channel in 'filterInputType_'.
  std::vector<IdentityProjection> filterTableProjections_;

  // The following six fields are used in null-aware anti join filter
  // processing.

  // Used to decode a probe side filter input column to check nulls.
  DecodedVector filterInputColumnDecodedVector_;

  // Rows that have null value in any probe side filter columns to skip the
  // null-propagating filter evaluation. The corresponding probe input rows can
  // be added to the output result directly as they won't pass the filter.
  SelectivityVector nullFilterInputRows_;

  // Used to store the null-key joined rows for filter processing.
  RowVectorPtr filterTableInput_;
  SelectivityVector filterTableInputRows_;

  // Used to store the filter result for null-key joined rows.
  std::vector<VectorPtr> filterTableResult_;
  DecodedVector decodedFilterTableResult_;

  // Row number in 'input_' for each output row.
  BufferPtr outputRowMapping_;

  // For left join with filter, we could overwrite the row which we have not
  // checked if there is a carryover.  Use a temporary buffer in this case.
  BufferPtr tempOutputRowMapping_;

  // maps from column index in 'table_' to channel in 'output_'.
  std::vector<IdentityProjection> tableOutputProjections_;

  // Rows of table found by join probe, later filtered by 'filter_'.
  BufferPtr outputTableRows_;
  vector_size_t outputTableRowsCapacity_;

  // For left join with filter, we could overwrite the row which we have not
  // checked if there is a carryover.  Use a temporary buffer in this case.
  BufferPtr tempOutputTableRows_;

  // Indicates probe-side rows which should produce a NULL in left semi project
  // with filter.
  SelectivityVector leftSemiProjectIsNull_;

  // Tracks probe side rows which had one or more matches on the build side, but
  // didn't pass the filter.
  class NoMatchDetector {
   public:
    // Called for each row that the filter was evaluated on. Expects that probe
    // side rows with multiple matches on the build side are next to each other.
    template <typename TOnMiss>
    void advance(vector_size_t row, bool passed, TOnMiss&& onMiss) {
      if (currentRow_ != row) {
        if (hasLastMissedRow()) {
          onMiss(currentRow_);
        }
        currentRow_ = row;
        currentRowPassed_ = false;
      }
      if (passed) {
        currentRowPassed_ = true;
      }
    }

    // Invoked at the end of all output batches.
    template <typename TOnMiss>
    void finish(TOnMiss&& onMiss) {
      if (hasLastMissedRow()) {
        onMiss(currentRow_);
      }
      currentRow_ = -1;
    }

    // Returns if we're carrying forward a missed input row. Notably, if this is
    // true, we're not yet done processing the input batch.
    bool hasLastMissedRow() const {
      return currentRow_ != -1 && !currentRowPassed_;
    }

   private:
    // Row number being processed.
    vector_size_t currentRow_{-1};

    // True if currentRow_ has a match.
    bool currentRowPassed_{false};
  };

  // For left semi join filter with extra filter, de-duplicates probe side rows
  // with multiple matches.
  class LeftSemiFilterJoinTracker {
   public:
    // Called for each row that the filter passes. Expects that probe
    // side rows with multiple matches are next to each other. Calls onLastMatch
    // just once for each probe side row with at least one match.
    template <typename TOnLastMatch>
    void advance(vector_size_t row, TOnLastMatch onLastMatch) {
      if (currentRow != row) {
        if (currentRow != -1) {
          onLastMatch(currentRow);
        }
        currentRow = row;
      }
    }

    // Called when all rows from the current input batch were processed. Calls
    // onLastMatch for the last probe row with at least one match.
    template <typename TOnLastMatch>
    void finish(TOnLastMatch onLastMatch) {
      if (currentRow != -1) {
        onLastMatch(currentRow);
      }

      currentRow = -1;
    }

   private:
    // The last row number passed to advance for the current input batch.
    vector_size_t currentRow{-1};
  };

  // For left semi join project with filter, de-duplicates probe side rows
  // with multiple matches.
  class LeftSemiProjectJoinTracker {
   public:
    // Called for each row and indicates whether the filter passed or not.
    // Expects that probe side rows with multiple matches are next to each
    // other. Calls onLast just once for each probe side row.
    template <typename TOnLast>
    void
    advance(vector_size_t row, std::optional<bool> passed, TOnLast onLast) {
      if (currentRow != row) {
        if (currentRow != -1) {
          onLast(currentRow, currentRowPassed);
        }
        currentRow = row;
        currentRowPassed = std::nullopt;
      }

      if (passed.has_value()) {
        if (currentRowPassed.has_value()) {
          currentRowPassed = currentRowPassed.value() || passed.value();
        } else {
          currentRowPassed = passed;
        }
      }
    }

    // Called when all rows from the current input batch were processed. Calls
    // onLast for the last probe row.
    template <typename TOnLast>
    void finish(TOnLast onLast) {
      if (currentRow != -1) {
        onLast(currentRow, currentRowPassed);
      }

      currentRow = -1;
      currentRowPassed = std::nullopt;
    }

   private:
    // The last row number passed to advance for the current input batch.
    vector_size_t currentRow{-1};

    // True if currentRow has a match.
    std::optional<bool> currentRowPassed;
  };

  BaseHashTable::RowsIterator lastProbeIterator_;

  // For left and anti join with filter, tracks the probe side rows which had
  // matches on the build side but didn't pass the filter.
  NoMatchDetector noMatchDetector_;

  // For left semi join filter with extra filter, de-duplicates probe side rows
  // with multiple matches.
  LeftSemiFilterJoinTracker leftSemiFilterJoinTracker_;

  // For left semi join project with filter, de-duplicates probe side rows with
  // multiple matches.
  LeftSemiProjectJoinTracker leftSemiProjectJoinTracker_;

  // Keeps track of returned results between successive batches of
  // output for a batch of input.
  std::unique_ptr<BaseHashTable::JoinResultIterator> resultIter_;

  RowVectorPtr output_;

  // Input rows with no nulls in the join keys.
  SelectivityVector nonNullInputRows_;

  // Input rows with a hash match. This is a subset of rows with no nulls in the
  // join keys and a superset of rows that have a match on the build side.
  SelectivityVector activeRows_;

  // True if passingInputRows is up to date.
  bool passingInputRowsInitialized_ = false;

  // Set of input rows for which there is at least one join hit. All
  // set if right side optional. Used when loading lazy vectors for
  // cases where there is more than one batch of output or join filter
  // input.
  SelectivityVector passingInputRows_;

  // Indicates if this hash probe has exceeded max spill limit which is not
  // allowed to spill. This is reset when hash probe operator starts to probe
  // the next previously spilled hash table partition.
  tsan_atomic<bool> exceededMaxSpillLevelLimit_{false};

  // The partition bits used to spill the hash table.
  HashBitRange tableSpillHashBits_;

  // The spilled output partition set which is cleared after setup
  // 'spillOutputReader_'.
  SpillPartitionSet spillOutputPartitionSet_;

  // The reader used to read the spilled output produced by pending input during
  // the spill processing.
  std::unique_ptr<UnorderedStreamReader<BatchStream>> spillOutputReader_;

  // 'inputSpiller_' is created if some part of build-side rows have been
  // spilled. It is used to spill probe-side rows if the corresponding
  // build-side rows have been spilled.
  std::unique_ptr<NoRowContainerSpiller> inputSpiller_;

  // If not empty, the probe inputs with partition id set in
  // 'spillInputPartitionIds_' needs to spill. It is set along with 'spiller_'
  // to the partition ids that have been spilled at build side when built
  // 'table_'.
  SpillPartitionIdSet spillInputPartitionIds_;

  // Used to calculate the spill partition numbers of the probe inputs.
  std::unique_ptr<SpillPartitionFunction> spillPartitionFunction_;

  // Reusable memory for spill hash partition calculation.
  std::vector<SpillPartitionId> spillPartitions_;

  // Reusable memory for probe input spilling processing.
  folly::F14FastMap<SpillPartitionId, vector_size_t> numSpillInputs_;
  folly::F14FastMap<SpillPartitionId, BufferPtr> spillInputIndicesBuffers_;
  folly::F14FastMap<SpillPartitionId, vector_size_t*>
      rawSpillInputIndicesBuffers_;
  BufferPtr nonSpillInputIndicesBuffer_;
  vector_size_t* rawNonSpillInputIndicesBuffer_;

  // 'spillInputReader_' is only created if 'table_' is built from the
  // previously spilled data. It is used to read the probe inputs from the
  // corresponding spilled data on disk.
  std::unique_ptr<UnorderedStreamReader<BatchStream>> spillInputReader_;

  // The spill partition id for the currently restoring input partition,
  // corresponding to 'spillInputReader_'. Not set if hash probe hasn't spilled
  // yet.
  std::optional<SpillPartitionId> restoringPartitionId_;

  // Sets to true after read all the probe inputs from 'spillInputReader_'.
  bool noMoreSpillInput_{false};

  // The spilled probe partitions remaining to restore.
  SpillPartitionSet inputSpillPartitionSet_;

  // VectorHashers used for listing rows with null keys.
  std::vector<std::unique_ptr<VectorHasher>> nullKeyProbeHashers_;

  // Input vector used for listing rows with null keys.
  VectorPtr nullKeyProbeInput_;
};

inline std::ostream& operator<<(std::ostream& os, ProbeOperatorState state) {
  os << probeOperatorStateName(state);
  return os;
}

} // namespace facebook::velox::exec
