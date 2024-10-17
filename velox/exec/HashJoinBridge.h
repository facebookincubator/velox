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
#include "velox/exec/JoinBridge.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Spill.h"

namespace facebook::velox::exec {

namespace test {
class HashJoinBridgeTestHelper;
}

/// Hands over a hash table from a multi-threaded build pipeline to a
/// multi-threaded probe pipeline. This is owned by shared_ptr by all the build
/// and probe Operator instances concerned. Corresponds to the Presto concept of
/// the same name.
class HashJoinBridge : public JoinBridge {
 public:
  struct SpillResult {
    Spiller* spiller{nullptr};
    const std::exception_ptr error{nullptr};

    explicit SpillResult(std::exception_ptr _error) : error(_error) {}
    explicit SpillResult(Spiller* _spiller) : spiller(_spiller) {}
  };

  void start() override;

  /// Invoked by HashBuild operator ctor to add to this bridge by incrementing
  /// 'numBuilders_'. The latter is used to split the spill partition data among
  /// HashBuild operators to parallelize the restoring operation.
  void addBuilder();

  /// Invoked to spill 'table' and returns spilled partitions. This method
  /// should only be invoked when the 'table' is a ready-to-use one, meaning it
  /// should not be one in the middle of building. Hence it is normally invoked
  /// by probe side.
  SpillPartitionSet spillTable(
      std::shared_ptr<BaseHashTable> table,
      folly::Synchronized<common::SpillStats>* stats);

  /// Triggers the parallel spilling directly from the provided 'spillers'. It
  /// does not do other operations other than spill. Hence it can be invoked to
  /// spill partially built table, and hence invoked by build side.
  std::vector<std::unique_ptr<SpillResult>> spillTableFromSpillers(
      const std::vector<Spiller*>& spillers);

  /// Invoked by the build operator to set the built hash table.
  /// 'spillPartitionSet' contains the spilled partitions while building
  /// 'table' which only applies if the disk spilling is enabled.
  void setHashTable(
      std::unique_ptr<BaseHashTable> table,
      SpillPartitionSet spillPartitionSet,
      bool hasNullKeys);

  /// Invoked by the probe operator to set the spilled hash table while the
  /// probing. The function puts the spilled table partitions into
  /// 'spillPartitionSets_' stack. This only applies if the disk spilling is
  /// enabled.
  void setSpilledHashTable(SpillPartitionSet spillPartitionSet);

  void setAntiJoinHasNullKeys();

  /// Represents the result of HashBuild operators. In case of an anti join, a
  /// build side entry with a null in a join key makes the join return nothing.
  /// In this case, HashBuild operators finishes early without processing all
  /// the input and without finishing building the hash table.
  struct HashBuildResult {
    HashBuildResult(
        std::shared_ptr<BaseHashTable> _table,
        std::optional<SpillPartitionId> _restoredPartitionId,
        SpillPartitionIdSet _spillPartitionIds,
        bool _hasNullKeys)
        : hasNullKeys(_hasNullKeys),
          table(std::move(_table)),
          restoredPartitionId(std::move(_restoredPartitionId)),
          spillPartitionIds(std::move(_spillPartitionIds)) {}

    HashBuildResult() : hasNullKeys(true) {}

    bool hasNullKeys;
    std::shared_ptr<BaseHashTable> table;

    /// Restored spill partition id associated with 'table', null if 'table' is
    /// not built from restoration.
    std::optional<SpillPartitionId> restoredPartitionId;

    /// Spilled partitions while building hash table. Either 'table' is empty or
    /// 'spillPartitionIds' is empty.
    SpillPartitionIdSet spillPartitionIds;
  };

  /// Invoked by HashProbe operator to get the table to probe which is built by
  /// HashBuild operators. If HashProbe operator calls this early, 'future' will
  /// be set to wait asynchronously, otherwise the built table along with
  /// optional spilling related information will be returned in HashBuildResult.
  std::optional<HashBuildResult> tableOrFuture(ContinueFuture* future);

  /// Invoked by HashProbe operator after finishes probing the built table to
  /// set one of the previously spilled partition to restore. The HashBuild
  /// operators will then build the next hash table from the selected spilled
  /// one. The function returns true if there is spill data to be restored by
  /// HashBuild operators next.
  bool probeFinished();

  /// Contains the spill input for one HashBuild operator: a shard of previously
  /// spilled partition data. 'spillPartition' is null if there is no more spill
  /// data to restore.
  struct SpillInput {
    explicit SpillInput(
        std::unique_ptr<SpillPartition> spillPartition = nullptr)
        : spillPartition(std::move(spillPartition)) {}

    std::unique_ptr<SpillPartition> spillPartition;
  };

  /// Invoked by HashBuild operator to get one of previously spilled partition
  /// shard to restore. The spilled partition to restore is set by HashProbe
  /// operator after finishes probing on the previously built hash table.
  /// If HashBuild operator calls this early, 'future' will be set to wait
  /// asynchronously. If there is no more spill data to restore, then
  /// 'spillPartition' will be set to null in the returned SpillInput.
  std::optional<SpillInput> spillInputOrFuture(ContinueFuture* future);

  /// Sets the build table type.
  void maybeSetTableType(const RowTypePtr& tableType);

  /// Sets the spill configs.
  void maybeSetSpillConfig(const common::SpillConfig* spillConfig);

  /// Sets the join plan node 'this' is responsible for.
  void maybeSetJoinNode(
      const std::shared_ptr<const core::HashJoinNode>& joinNode);

 private:
  // Spills the row container from one of the sub-table from
  // 'table' to parallelize the table spilling. The function
  // spills all the rows from the row container and returns the spiller for the
  // caller to collect the spilled partitions and stats.
  std::unique_ptr<Spiller> createSpiller(
      RowContainer* subTableRows,
      folly::Synchronized<common::SpillStats>* stats);

  // Returns the spill hash bit range for spilling the current
  // 'buildResult_->table'.
  HashBitRange tableSpillHashBitRange() const;

  const common::SpillConfig* spillConfig() const;

  uint32_t numBuilders_{0};

  // The result of the build side of the current processing unit. There could be
  // multiple processing units in the case of spill. Each partition/recursive
  // partition would be its own processing unit.
  std::optional<HashBuildResult> buildResult_;

  // restoringSpillPartitionXxx member variables are populated by the
  // bridge itself. When probe side finished processing, the bridge picks the
  // first partition from 'spillPartitionSets_', splits it into "even" shards
  // among the HashBuild operators and notifies these operators that they can
  // start building HashTables from these shards.

  // If not null, set to the currently restoring table spill partition id.
  std::optional<SpillPartitionId> restoringSpillPartitionId_;

  // If 'restoringSpillPartitionId_' is not null, this set to the restoring
  // spill partition data shards. Each shard is expected to have the same number
  // of spill files and will be processed by one of the HashBuild operator.
  std::vector<std::unique_ptr<SpillPartition>> restoringSpillShards_;

  // The spill partitions remaining to restore. This set is populated using
  // information provided by the HashBuild operators if spilling is enabled.
  // This set can grow if HashBuild operator cannot load full partition in
  // memory and engages in recursive spilling.
  SpillPartitionSet spillPartitionSets_;

  // The row type used for hash table spilling.
  RowTypePtr tableType_;
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  std::optional<common::SpillConfig> spillConfig_;
  friend test::HashJoinBridgeTestHelper;
};

// Indicates if 'joinNode' is null-aware anti or left semi project join type and
// has filter set.
bool isLeftNullAwareJoinWithFilter(
    const std::shared_ptr<const core::HashJoinNode>& joinNode);

class HashJoinMemoryReclaimer final : public MemoryReclaimer {
 public:
  static std::unique_ptr<memory::MemoryReclaimer> create() {
    return std::unique_ptr<memory::MemoryReclaimer>(
        new HashJoinMemoryReclaimer());
  }

  uint64_t reclaim(
      memory::MemoryPool* pool,
      uint64_t targetBytes,
      uint64_t maxWaitMs,
      memory::MemoryReclaimer::Stats& stats) final;

 private:
  HashJoinMemoryReclaimer() : MemoryReclaimer() {}
};

/// Returns true if 'pool' is a hash build operator's memory pool. The check is
/// currently based on the pool name.
bool isHashBuildMemoryPool(const memory::MemoryPool& pool);

/// Returns true if 'pool' is a hash probe operator's memory pool. The check is
/// currently based on the pool name.
bool isHashProbeMemoryPool(const memory::MemoryPool& pool);

bool needRightSideJoin(core::JoinType joinType);

/// Returns the type used to spill a given hash table type. The function
/// might attach a boolean column at the end of 'tableType' if 'joinType' needs
/// right side join processing. It is used by the hash join table spilling
/// triggered at the probe side to record if each row has been probed or not.
RowTypePtr hashJoinTableSpillType(
    const RowTypePtr& tableType,
    core::JoinType joinType);

/// Checks if a given type is a hash table spill type or not based on
/// 'joinType'.
bool isHashJoinTableSpillType(
    const RowTypePtr& spillType,
    core::JoinType joinType);
} // namespace facebook::velox::exec
