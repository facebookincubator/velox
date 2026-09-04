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

#include <functional>

#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"

namespace facebook::velox::exec {

/// Accumulates build side input into a join hash table.
///
/// This holds the state and the logic which is shared between the 'HashBuild'
/// operator and the users which build a join table outside of a Velox driver,
/// e.g. Gluten builds the broadcast build side directly from a set of vectors.
/// Hence it knows nothing about plan nodes, drivers, spilling, stats or the
/// hash table cache: the owner drives it and plugs its own processing in
/// between the 'addInput()' phases.
///
/// Not thread safe. One instance accumulates the input of one build thread.
/// Tables built by multiple instances are merged with
/// 'BaseHashTable::prepareJoinTable()'.
class JoinTableBuilder {
 public:
  struct Options {
    core::JoinType joinType;

    bool nullAware{false};

    bool nullAsValue{false};

    /// True if the join has an extra filter, i.e. 'HashJoinNode::filter()' is
    /// set.
    bool withFilter{false};

    /// The type of the vectors passed to 'addInput()'. Note that this is the
    /// build source type, which is not necessarily 'tableType()'.
    RowTypePtr inputType;

    /// The build side join keys, resolved against 'inputType'.
    std::vector<core::FieldAccessTypedExprPtr> joinKeys;

    /// See the query config options of the same name.
    uint32_t minTableRowsForParallelJoinBuild{1'000};
    uint32_t vectorHasherMaxNumDistinct{1'000'000};
    int32_t abandonHashBuildDedupMinRows{100'000};
    int32_t abandonHashBuildDedupMinPct{0};
    int64_t bloomFilterPushdownMaxSize{0};

    /// Invoked at most once when the build of a deduplicated hash table is
    /// abandoned while processing input. Not invoked if dedup was disabled by
    /// 'abandonHashBuildDedupMinPct' being zero. Used by 'HashBuild' to record
    /// a runtime stat.
    std::function<void()> onDedupAbandoned{nullptr};
  };

  /// Describes the join filter of an anti join. Only used to skip the build
  /// rows which can never pass a null propagating filter.
  struct AntiJoinFilterInfo {
    /// True if the join filter is null propagating, e.g.
    /// 'exec::Expr::propagatesNulls()'.
    bool propagatesNulls{false};

    /// Channels in 'Options::inputType' of the build side columns referenced by
    /// the filter. Must be sorted. Channels which are not build side columns
    /// are ignored.
    std::vector<column_index_t> inputChannels;
  };

  explicit JoinTableBuilder(Options options);

  /// Creates the hash table. Must be called once before 'addInput()'. This is
  /// separate from the constructor as neither the pools nor the compiled filter
  /// are necessarily known at construction time, see 'HashBuild::initialize()'.
  /// 'tablePool' is used by the hash table and its row container,
  /// 'auxiliaryPool' by the transient state of the build, e.g. the hash lookup.
  /// They can be the same pool.
  void initialize(
      memory::MemoryPool* tablePool,
      memory::MemoryPool* auxiliaryPool,
      const AntiJoinFilterInfo& filterInfo);

  void initialize(
      memory::MemoryPool* tablePool,
      memory::MemoryPool* auxiliaryPool) {
    initialize(tablePool, auxiliaryPool, AntiJoinFilterInfo{});
  }

  /// Accumulates 'input' into the table. Returns false if the build can stop
  /// early, i.e. this is a null-aware anti join without a filter and 'input'
  /// has a null join key, in which case the join returns no rows.
  ///
  /// This is the composition of the four phases below and is meant for the
  /// owners which have nothing to do in between.
  bool addInput(const RowVectorPtr& input);

  /// The phases of 'addInput()'. Decodes the join keys of 'input' into the
  /// hashers and resets 'activeRows()' to all the rows of 'input'.
  void decodeKeys(const RowVectorPtr& input);

  /// Deselects the rows of 'activeRows()' with a null join key unless the join
  /// needs to retain them, and tracks whether the build side has null keys.
  /// Returns false if the build can stop early, see 'addInput()'.
  bool processNullKeys();

  /// Decodes the non-key columns of 'input' and, for an anti join with a null
  /// propagating filter, deselects the rows with a null in a filter column.
  void decodeDependents(const RowVectorPtr& input);

  /// Inserts the rows still selected in 'activeRows()' into the table, either
  /// by deduplicating them into the hash table or by appending them to the row
  /// container. 'spillProbedFlags', if set, carries the probed flag of each row
  /// restored from a spilled table.
  void insertRows(
      const RowVectorPtr& input,
      const FlatVector<bool>* spillProbedFlags = nullptr);

  /// Clears the table and re-creates an empty one. Used when restoring
  /// previously spilled data: the columns of the spilled input are already
  /// ordered as 'tableType()', hence the key and dependent channels are reset
  /// to their identity mapping.
  void resetForSpillInput();

  /// Frees the table. The builder can not be used afterwards unless
  /// 'resetForSpillInput()' is called.
  void clearTable() {
    table_.reset();
    lookup_.reset();
  }

  BaseHashTable* table() const {
    return table_.get();
  }

  /// Transfers the ownership of the table out of the builder.
  std::unique_ptr<BaseHashTable> takeTable() {
    lookup_.reset();
    return std::move(table_);
  }

  /// Replaces the table, e.g. with one restored from a serialized image.
  void setTable(std::unique_ptr<BaseHashTable> table) {
    lookup_.reset();
    table_ = std::move(table);
  }

  const std::vector<std::unique_ptr<VectorHasher>>& hashers() const {
    return table_->hashers();
  }

  /// The rows of the last 'input' which are still being processed. Exposed as
  /// the owner may deselect rows in between the 'addInput()' phases, e.g.
  /// 'HashBuild' deselects the rows it spills.
  SelectivityVector& activeRows() {
    return activeRows_;
  }

  /// The row type of the hash table, which is also the type used to spill it.
  const RowTypePtr& tableType() const {
    return tableType_;
  }

  const std::vector<column_index_t>& keyChannels() const {
    return keyChannels_;
  }

  const std::vector<column_index_t>& dependentChannels() const {
    return dependentChannels_;
  }

  bool dropDuplicates() const {
    return dropDuplicates_;
  }

  /// True if the build of the deduplicated hash table has been abandoned.
  bool dedupAbandoned() const {
    return abandonHashBuildDedup_;
  }

  /// True if this is the build side of an anti or left semi project join and
  /// has at least one entry with null join keys.
  bool joinHasNullKeys() const {
    return joinHasNullKeys_;
  }

  void setJoinHasNullKeys(bool joinHasNullKeys) {
    joinHasNullKeys_ = joinHasNullKeys;
  }

  core::JoinType joinType() const {
    return options_.joinType;
  }

  uint32_t vectorHasherMaxNumDistinct() const {
    return options_.vectorHasherMaxNumDistinct;
  }

 private:
  // Invoked to set up the hash table to build.
  void setupTable();

  // Maps the filter input channels to the key and the dependent channels of
  // the table. Set up for null-aware and regular anti join with a
  // null-propagating filter.
  void setupFilterChannels(const AntiJoinFilterInfo& filterInfo);

  // Invoked when preparing for null-aware and regular anti join with a
  // null-propagating filter. The function deselects the input rows which have
  // any null in the filter input columns. This is an optimization for
  // null-aware and regular anti join processing at the probe side as any probe
  // matches with the deselected rows can't pass the null-propagating filter
  // and will be added to the joined output.
  void removeInputRowsForAntiJoinFilter();

  // True if we have enough rows and not enough duplicate join keys, i.e. more
  // than 'abandonHashBuildDedupMinRows' rows and more than
  // 'abandonHashBuildDedupMinPct' % of rows are unique.
  bool abandonHashBuildDedupEarly(int64_t numDistinct) const;

  // Invoked to abandon the build of the deduped hash table.
  void abandonHashBuildDedup();

  const Options options_;

  // Indicates whether to drop duplicate rows. Rows containing duplicate keys
  // can be removed for left semi and anti join.
  const bool dropDuplicates_;

  memory::MemoryPool* tablePool_{nullptr};
  memory::MemoryPool* auxiliaryPool_{nullptr};

  // Indicates whether the join filter is null-propagating.
  bool filterPropagatesNulls_{false};

  // The row type used for the hash table build and the disk spilling.
  RowTypePtr tableType_;

  // Container for the rows being accumulated.
  std::unique_ptr<BaseHashTable> table_;

  // Used for building the hash table while adding input rows.
  std::unique_ptr<HashLookup> lookup_;

  // Key channels in the input.
  std::vector<column_index_t> keyChannels_;

  // Non-key channels in the input.
  std::vector<column_index_t> dependentChannels_;

  // Corresponds 1:1 to 'dependentChannels_'.
  std::vector<std::unique_ptr<DecodedVector>> decoders_;

  // Maps a key channel in the input to a channel in the key.
  folly::F14FastMap<column_index_t, column_index_t> keyChannelMap_;

  // Indices of the key columns used by the filter in the build side table.
  std::vector<column_index_t> keyFilterChannels_;

  // Indices of the dependent columns used by the filter in 'decoders_'.
  std::vector<column_index_t> dependentFilterChannels_;

  // True if we are considering use of normalized keys or array hash tables.
  // Set to false when the dataset is no longer suitable.
  bool analyzeKeys_{false};

  // Temporary space for hash numbers.
  raw_vector<uint64_t> hashes_;

  // Set of active rows during 'addInput()'.
  SelectivityVector activeRows_;

  bool joinHasNullKeys_{false};

  // Whether to abandon building a hash table without duplicates while adding
  // input for left semi/anti join.
  bool abandonHashBuildDedup_{false};

  // Counts the number of hash table input rows for building the deduped hash
  // table. It is not updated after 'abandonHashBuildDedup_' is true.
  int64_t numHashInputRows_{0};
};

} // namespace facebook::velox::exec
