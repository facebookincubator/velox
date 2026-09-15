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

#include "velox/exec/HashBuild.h"
#include <fmt/format.h>
#include "velox/common/base/Counters.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/HashTableCache.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorHasher.h"
#include "velox/expression/FieldReference.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::exec {
namespace {
// Map HashBuild 'state' to the corresponding driver blocking reason.
BlockingReason fromStateToBlockingReason(HashBuild::State state) {
  switch (state) {
    case HashBuild::State::kRunning:
      [[fallthrough]];
    case HashBuild::State::kFinish:
      return BlockingReason::kNotBlocked;
    case HashBuild::State::kYield:
      return BlockingReason::kYield;
    case HashBuild::State::kWaitForBuild:
      return BlockingReason::kWaitForJoinBuild;
    case HashBuild::State::kWaitForProbe:
      return BlockingReason::kWaitForJoinProbe;
    default:
      VELOX_UNREACHABLE(HashBuild::stateName(state));
  }
}
} // namespace

HashBuild::HashBuild(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : Operator(
          driverCtx,
          nullptr,
          operatorId,
          joinNode->id(),
          OperatorType::kHashBuild,
          joinNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId, OperatorType::kHashBuild)
              : std::nullopt),
      joinNode_(std::move(joinNode)),
      joinType_{joinNode_->joinType()},
      nullAware_{joinNode_->isNullAware()},
      nullAsValue_{joinNode_->isNullAsValue()},
      needProbedFlagSpill_{needRightSideJoin(joinType_)},
      joinBridge_(operatorCtx_->task()->getHashJoinBridgeLocked(
          operatorCtx_->driverCtx()->splitGroupId,
          planNodeId())) {
  VELOX_CHECK(pool()->trackUsage());
  VELOX_CHECK_NOT_NULL(joinBridge_);

  joinBridge_->addBuilder();

  setupTableBuilder();

  stateCleared_ = false;
}

void HashBuild::setupTableBuilder() {
  const auto& queryConfig = operatorCtx_->driverCtx()->queryConfig();

  JoinTableBuilder::Options options;
  options.joinType = joinType_;
  options.nullAware = nullAware_;
  options.nullAsValue = nullAsValue_;
  options.withFilter = joinNode_->filter() != nullptr;
  options.inputType = joinNode_->sources()[1]->outputType();
  options.joinKeys = joinNode_->rightKeys();
  options.minTableRowsForParallelJoinBuild =
      queryConfig.minTableRowsForParallelJoinBuild();
  options.vectorHasherMaxNumDistinct =
      queryConfig.joinBuildVectorHasherMaxNumDistinct();
  options.abandonHashBuildDedupMinRows =
      queryConfig.abandonHashBuildDedupMinRows();
  options.abandonHashBuildDedupMinPct =
      queryConfig.abandonHashBuildDedupMinPct();
  options.bloomFilterPushdownMaxSize =
      queryConfig.hashProbeBloomFilterPushdownMaxSize();
  options.onDedupAbandoned = [this]() {
    // The hash table is no longer directly constructed in addInput. The data
    // that was previously inserted into the hash table is already in the
    // RowContainer.
    addRuntimeStat(
        std::string(HashBuild::kAbandonBuildNoDupHash), RuntimeCounter(1));
  };

  tableBuilder_ = std::make_unique<JoinTableBuilder>(std::move(options));
}

void HashBuild::initialize() {
  Operator::initialize();

  if (setupCachedHashTable()) {
    return;
  }

  // Set up table and spiller now that cache state is initialized.
  // This ensures tableMemoryPool() returns the cache's tablePool when enabled.
  tableBuilder_->initialize(tableMemoryPool(), pool(), analyzeAntiJoinFilter());
  setupSpiller();
}

bool HashBuild::setupCachedHashTable() {
  if (!joinNode_->useHashTableCache()) {
    return false;
  }

  if (joinNode_->cacheKey().has_value()) {
    cacheKey_ = joinNode_->cacheKey().value();
  } else {
    const auto& queryId = operatorCtx_->task()->queryCtx()->queryId();
    cacheKey_ = fmt::format("{}:{}", queryId, planNodeId());
  }

  VELOX_CHECK(
      !cacheKey_.empty(),
      "Hash table cache requires a non-empty cache key when "
      "useHashTableCache is enabled");

  // Get or create the cache entry (which includes the pool).
  // If another task is already building, future_ will be set.
  auto* cache = HashTableCache::instance();
  auto* queryCtx = operatorCtx_->task()->queryCtx().get();
  cacheEntry_ = cache->get(cacheKey_, taskId(), queryCtx, &future_);
  VELOX_CHECK_NOT_NULL(cacheEntry_);
  VELOX_CHECK_NOT_NULL(cacheEntry_->tablePool);

  // Check if table is already built.
  if (cacheEntry_->buildComplete) {
    noMoreInput();
    return true;
  }

  // Check if we're a waiter task (future was set by get).
  if (future_.valid()) {
    setState(State::kWaitForBuild);
    return true;
  }

  // This is the builder task - proceed with building.
  return false;
}

bool HashBuild::getHashTableFromCache() {
  if (!useHashTableCache()) {
    return false;
  }

  if (!cacheEntry_->buildComplete) {
    // Cache miss - we need to build the table.
    stats_.wlock()->addRuntimeStat(
        std::string(BaseHashTable::kHashTableCacheMiss), RuntimeCounter(1));
    return false;
  }

  // Table already built by a previous task! Use it directly.
  // Notify the bridge with the cached table.
  // We pass a shared_ptr copy (not std::move) since the cache retains
  // ownership.
  joinBridge_->setHashTable(
      cacheEntry_->table, {}, cacheEntry_->hasNullKeys, nullptr);
  // Record cache hit metric.
  stats_.wlock()->addRuntimeStat(
      std::string(BaseHashTable::kHashTableCacheHit), RuntimeCounter(1));
  return true;
}

void HashBuild::maybeSetHashTableInCache(
    const std::shared_ptr<BaseHashTable>& table) {
  if (!useHashTableCache()) {
    return;
  }
  auto* cache = HashTableCache::instance();
  cache->put(cacheKey(), table, tableBuilder_->joinHasNullKeys());
}

bool HashBuild::receivedCachedHashTable() {
  if (!useHashTableCache() || future_.valid()) {
    return false;
  }
  // Builder task drivers coordinate via allPeersFinished and should fall
  // through to the kWaitForProbe path in isBlocked(). Only waiter task
  // drivers (different taskId than the builder) should enter here.
  VELOX_CHECK_NOT_NULL(cacheEntry_);
  if (hashTableCacheBuilderTask()) {
    return false;
  }
  // We were waiting on cached table from another task.
  // Ensure that table is ready.
  VELOX_CHECK(
      cacheEntry_->buildComplete,
      "Hash table cache build failed for key '{}'. "
      "The builder task may have encountered an error (e.g., OOM).",
      cacheKey_);
  // Proceed through normal noMoreInput flow which will use the cache.
  setRunning();
  noMoreInput();
  return true;
}

void HashBuild::setupSpiller(SpillPartition* spillPartition) {
  VELOX_CHECK_NULL(spiller_);
  VELOX_CHECK_NULL(spillInputReader_);

  if (!canSpill()) {
    return;
  }
  if (spillType_ == nullptr) {
    spillType_ = hashJoinTableSpillType(tableBuilder_->tableType(), joinType_);
    if (needProbedFlagSpill_) {
      spillProbedFlagChannel_ = spillType_->size() - 1;
      VELOX_CHECK_NULL(spillProbedFlagVector_);
      // Creates a constant probed flag vector with all values false for build
      // side table spilling.
      spillProbedFlagVector_ = std::make_shared<ConstantVector<bool>>(
          pool(), 0, /*isNull=*/false, BOOLEAN(), false);
    }
  }

  const auto* config = spillConfig();
  uint8_t startPartitionBit = config->startPartitionBit;
  if (spillPartition != nullptr) {
    spillInputReader_ = spillPartition->createUnorderedReader(
        config->readBufferSize, pool(), spillStats_.get());
    VELOX_CHECK(!restoringPartitionId_.has_value());
    restoringPartitionId_ = spillPartition->id();
    const auto numPartitionBits = config->numPartitionBits;
    startPartitionBit =
        partitionBitOffset(
            spillPartition->id(), startPartitionBit, numPartitionBits) +
        numPartitionBits;
    // Disable spilling if exceeding the max spill level and the query might run
    // out of memory if the restored partition still can't fit in memory.
    if (FOLLY_UNLIKELY(config->exceedSpillLevelLimit(startPartitionBit))) {
      RECORD_METRIC_VALUE(kMetricMaxSpillLevelExceededCount);
      LOG(WARNING) << "Exceeded spill level limit: " << config->maxSpillLevel
                   << ", and disable spilling for memory pool: "
                   << pool()->name()
                   << ", root pool: " << pool()->root()->name()
                   << ", used: " << succinctBytes(pool()->usedBytes())
                   << ", reservation: "
                   << succinctBytes(pool()->reservedBytes())
                   << ", root pool reservation: "
                   << succinctBytes(pool()->root()->reservedBytes());
      spillStats_->spillMaxLevelExceededCount.fetch_add(
          1, std::memory_order_relaxed);
      exceededMaxSpillLevelLimit_ = true;
      return;
    }
    exceededMaxSpillLevelLimit_ = false;
  }

  spiller_ = std::make_unique<HashBuildSpiller>(
      joinType_,
      restoringPartitionId_,
      tableBuilder_->table()->rows(),
      spillType_,
      HashBitRange(
          startPartitionBit, startPartitionBit + config->numPartitionBits),
      config,
      spillStats_.get());

  const int32_t numPartitions = spiller_->hashBits().numPartitions();
  spillInputIndicesBuffers_.resize(numPartitions);
  rawSpillInputIndicesBuffers_.resize(numPartitions);
  numSpillInputs_.resize(numPartitions, 0);
  spillChildVectors_.resize(spillType_->size());
}

bool HashBuild::isInputFromSpill() const {
  return spillInputReader_ != nullptr;
}

RowTypePtr HashBuild::inputType() const {
  return isInputFromSpill() ? tableBuilder_->tableType()
                            : joinNode_->sources()[1]->outputType();
}

JoinTableBuilder::AntiJoinFilterInfo HashBuild::analyzeAntiJoinFilter() {
  JoinTableBuilder::AntiJoinFilterInfo filterInfo;
  if (!isAntiJoin(joinType_) || joinNode_->filter() == nullptr) {
    return filterInfo;
  }

  ExprSet exprs({joinNode_->filter()}, operatorCtx_->execCtx());
  VELOX_DCHECK_EQ(exprs.exprs().size(), 1);
  const auto& expr = exprs.expr(0);
  filterInfo.propagatesNulls = expr->propagatesNulls();
  if (!filterInfo.propagatesNulls) {
    return filterInfo;
  }

  const auto& inputType = joinNode_->sources()[1]->outputType();
  for (const auto& field : expr->distinctFields()) {
    const auto index = inputType->getChildIdxIfExists(field->field());
    if (!index.has_value()) {
      continue;
    }
    filterInfo.inputChannels.push_back(*index);
  }
  std::sort(filterInfo.inputChannels.begin(), filterInfo.inputChannels.end());
  return filterInfo;
}

void HashBuild::updateNullKeysStats() {
  // Update statistics for null keys in join operator.
  // We use the active rows to store which rows have some null keys,
  // and reset it after using it.
  auto& activeRows = tableBuilder_->activeRows();
  auto lockedStats = stats_.wlock();
  deselectRowsWithNulls(tableBuilder_->hashers(), activeRows);
  lockedStats->numNullKeys += activeRows.size() - activeRows.countSelected();
  activeRows.setAll();
}

const FlatVector<bool>* HashBuild::spillProbedFlags(
    const RowVectorPtr& input) const {
  if (!isInputFromSpill() || !needProbedFlagSpill_) {
    return nullptr;
  }
  return input->childAt(spillProbedFlagChannel_)->asFlatVector<bool>();
}

void HashBuild::addInput(RowVectorPtr input) {
  checkRunning();

  VELOX_CHECK(
      !useHashTableCache() ||
      (cacheEntry_->builderTaskId == taskId() && !cacheEntry_->buildComplete));

  ensureInputFits(input);

  TestValue::adjust("facebook::velox::exec::HashBuild::addInput", this);

  tableBuilder_->decodeKeys(input);

  // Only update the null keys stats when input is not spilled, to avoid
  // overcounting.
  if (!isInputFromSpill()) {
    updateNullKeysStats();
  }

  if (!tableBuilder_->processNullKeys()) {
    // Null-aware anti join with no extra filter returns no rows if build side
    // has nulls in join keys. Hence, we can stop processing on first null.
    noMoreInput();
    return;
  }

  tableBuilder_->decodeDependents(input);

  spillInput(input);

  tableBuilder_->insertRows(input, spillProbedFlags(input));
}

void HashBuild::ensureInputFits(RowVectorPtr& input) {
  // NOTE: we don't need memory reservation if all the partitions are spilling
  // as we spill all the input rows to disk directly.
  if (!canSpill() || spiller_ == nullptr || spiller_->spillTriggered()) {
    return;
  }

  // NOTE: we simply reserve memory all inputs even though some of them are
  // spilling directly. It is okay as we will accumulate the extra reservation
  // in the operator's memory pool, and won't make any new reservation if there
  // is already sufficient reservations.
  VELOX_CHECK(canSpill());

  auto* rows = tableBuilder_->table()->rows();
  const auto numRows = rows->numRows();

  auto [freeRows, outOfLineFreeBytes] = rows->freeSpace();
  const auto outOfLineBytes =
      rows->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const auto currentUsage = pool()->usedBytes();

  if (numRows != 0) {
    // Test-only spill path.
    if (testingTriggerSpill(pool()->name())) {
      Operator::ReclaimableSectionGuard guard(this);
      memory::testingRunArbitration(pool());
      return;
    }
  }

  const auto minReservationBytes =
      currentUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool()->availableReservation();
  const auto tableIncrementBytes =
      tableBuilder_->table()->hashTableSizeIncrease(input->size());
  const int64_t flatBytes = input->estimateFlatSize();
  const auto rowContainerIncrementBytes = numRows == 0
      ? flatBytes * 2
      : rows->sizeIncrement(
            input->size(), outOfLineBytes > 0 ? flatBytes * 2 : 0);
  const auto incrementBytes = rowContainerIncrementBytes + tableIncrementBytes;

  // First to check if we have sufficient minimal memory reservation.
  if (availableReservationBytes >= minReservationBytes) {
    if (freeRows > input->size() &&
        (outOfLineBytes == 0 || outOfLineFreeBytes >= flatBytes)) {
      // Enough free rows for input rows and enough variable length free
      // space for the flat size of the whole vector. If outOfLineBytes
      // is 0 there is no need for variable length space.
      return;
    }

    // If there is variable length data we take the flat size of the
    // input as a cap on the new variable length data needed. There must be at
    // least 2x the increments in reservation.
    if (pool()->availableReservation() > 2 * incrementBytes) {
      return;
    }
  }

  // Check if we can increase reservation. The increment is the larger of
  // twice the maximum increment from this input and
  // 'spillableReservationGrowthPct_' of the current reservation.
  const auto targetIncrementBytes = std::max<int64_t>(
      incrementBytes * 2,
      currentUsage * spillConfig_->spillableReservationGrowthPct / 100);

  {
    Operator::ReclaimableSectionGuard guard(this);
    if (pool()->maybeReserve(targetIncrementBytes)) {
      // If above reservation triggers the spilling of 'HashBuild' operator
      // itself, we will no longer need the reserved memory for building hash
      // table as the table is spilled, and the input will be directly spilled,
      // too.
      if (spiller_->spillTriggered()) {
        pool()->release();
      }
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", root pool: " << pool()->root()->name()
               << ", used: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(pool()->root()->reservedBytes());
}

void HashBuild::spillInput(const RowVectorPtr& input) {
  auto& activeRows = tableBuilder_->activeRows();
  VELOX_CHECK_EQ(input->size(), activeRows.size());

  if (!canSpill() || spiller_ == nullptr || !spiller_->spillTriggered() ||
      !activeRows.hasSelections()) {
    return;
  }

  const auto numInput = input->size();
  prepareInputIndicesBuffers(numInput);
  computeSpillPartitions(input);

  vector_size_t numSpillInputs = 0;
  for (auto row = 0; row < numInput; ++row) {
    const auto partition = spillPartitions_[row];
    if (FOLLY_UNLIKELY(!activeRows.isValid(row))) {
      continue;
    }
    activeRows.setValid(row, false);
    ++numSpillInputs;
    rawSpillInputIndicesBuffers_[partition][numSpillInputs_[partition]++] = row;
  }
  if (numSpillInputs == 0) {
    return;
  }

  maybeSetupSpillChildVectors(input);

  for (uint32_t partition = 0; partition < numSpillInputs_.size();
       ++partition) {
    const int numInputs = numSpillInputs_[partition];
    if (numInputs == 0) {
      continue;
    }
    spillPartition(
        partition, numInputs, spillInputIndicesBuffers_[partition], input);
    VELOX_CHECK(
        spiller_->state().isPartitionSpilled(SpillPartitionId(partition)));
  }
  activeRows.updateBounds();
}

void HashBuild::maybeSetupSpillChildVectors(const RowVectorPtr& input) {
  if (isInputFromSpill()) {
    return;
  }
  int32_t spillChannel = 0;
  for (const auto& channel : tableBuilder_->keyChannels()) {
    spillChildVectors_[spillChannel++] = input->childAt(channel);
  }
  for (const auto& channel : tableBuilder_->dependentChannels()) {
    spillChildVectors_[spillChannel++] = input->childAt(channel);
  }
  if (needProbedFlagSpill_) {
    VELOX_CHECK_NOT_NULL(spillProbedFlagVector_);
    spillProbedFlagVector_->resize(input->size());
    spillChildVectors_[spillChannel] = spillProbedFlagVector_;
  }
}

void HashBuild::prepareInputIndicesBuffers(vector_size_t numInput) {
  const auto maxIndicesBufferBytes = numInput * sizeof(vector_size_t);
  for (auto partition = 0; partition < (1UL << spillConfig_->numPartitionBits);
       ++partition) {
    if (spillInputIndicesBuffers_[partition] == nullptr ||
        (spillInputIndicesBuffers_[partition]->size() <
         maxIndicesBufferBytes)) {
      spillInputIndicesBuffers_[partition] = allocateIndices(numInput, pool());
      rawSpillInputIndicesBuffers_[partition] =
          spillInputIndicesBuffers_[partition]->asMutable<vector_size_t>();
    }
  }
  std::fill(numSpillInputs_.begin(), numSpillInputs_.end(), 0);
}

void HashBuild::computeSpillPartitions(const RowVectorPtr& input) {
  auto& activeRows = tableBuilder_->activeRows();
  if (spillHashes_.size() < activeRows.end()) {
    spillHashes_.resize(activeRows.end());
  }
  const auto& hashers = tableBuilder_->hashers();
  for (auto i = 0; i < hashers.size(); ++i) {
    auto& hasher = hashers[i];
    if (hasher->channel() != kConstantChannel) {
      hashers[i]->hash(activeRows, i > 0, spillHashes_);
    } else {
      hashers[i]->hashPrecomputed(activeRows, i > 0, spillHashes_);
    }
  }

  spillPartitions_.resize(input->size());
  activeRows.applyToSelected([&](int32_t row) {
    spillPartitions_[row] = spiller_->hashBits().partition(spillHashes_[row]);
  });
}

void HashBuild::spillPartition(
    uint32_t partition,
    vector_size_t size,
    const BufferPtr& indices,
    const RowVectorPtr& input) {
  VELOX_DCHECK(canSpill());

  if (isInputFromSpill()) {
    spiller_->spill(SpillPartitionId(partition), wrap(size, indices, input));
  } else {
    spiller_->spill(
        SpillPartitionId(partition),
        wrap(size, indices, spillType_, spillChildVectors_, input->pool()));
  }
}

void HashBuild::noMoreInput() {
  checkRunning();

  if (noMoreInput_) {
    return;
  }

  Operator::noMoreInput();

  noMoreInputInternal();
}

void HashBuild::noMoreInputInternal() {
  if (!finishHashBuild()) {
    return;
  }

  postHashBuildProcess();
}

bool HashBuild::finishHashBuild() {
  checkRunning();

  // Release the unused memory reservation before building the merged join
  // table.
  pool()->release();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Driver>> peers;
  // The last Driver to hit HashBuild::finish gathers the data from
  // all build Drivers and hands it over to the probe side. At this
  // point all build Drivers are continued and will free their
  // state. allPeersFinished is true only for the last Driver of the
  // build pipeline.
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    if (useHashTableCache() && !hashTableCacheBuilderTask()) {
      // Waiter task non-last driver: no partial table was built (we used the
      // cached table). Nothing to contribute — finish immediately. Clear the
      // future since allPeersFinished() set it but we don't need to wait.
      VELOX_CHECK_NULL(
          tableBuilder_->table(),
          "Waiter task should not have built a partial hash table");
      future_ = folly::SemiFuture<folly::Unit>::makeEmpty();
      setState(State::kFinish);
    } else {
      // Builder task non-last driver: the last driver needs our partial
      // table. Wait in kWaitForBuild until it has moved our table out.
      setState(State::kWaitForBuild);
    }
    return false;
  }

  TestValue::adjust("facebook::velox::exec::HashBuild::finishHashBuild", this);

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (getHashTableFromCache()) {
    return true;
  }

  if (tableBuilder_->joinHasNullKeys() && isAntiJoin(joinType_) && nullAware_ &&
      !joinNode_->filter()) {
    joinBridge_->setAntiJoinHasNullKeys();
    return true;
  }

  std::vector<HashBuild*> otherBuilds;
  otherBuilds.reserve(peers.size());
  uint64_t numRows{0};
  {
    std::lock_guard<std::mutex> l(mutex_);
    numRows += tableBuilder_->table()->rows()->numRows();
  }
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    HashBuild* build = dynamic_cast<HashBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    if (build->tableBuilder_->joinHasNullKeys()) {
      tableBuilder_->setJoinHasNullKeys(true);
      if (isAntiJoin(joinType_) && nullAware_ && !joinNode_->filter()) {
        joinBridge_->setAntiJoinHasNullKeys();
        return true;
      }
    }
    {
      std::lock_guard<std::mutex> l(build->mutex_);
      VELOX_CHECK(
          !build->stateCleared_,
          "Internal state for a peer is empty. It might have already"
          " been closed.");
      numRows += build->tableBuilder_->table()->rows()->numRows();
    }
    otherBuilds.push_back(build);
  }

  ensureTableFits(numRows);

  std::vector<std::unique_ptr<BaseHashTable>> otherTables;
  otherTables.reserve(peers.size());
  SpillPartitionSet spillPartitions;
  for (auto* build : otherBuilds) {
    std::unique_ptr<HashBuildSpiller> spiller;
    {
      std::lock_guard<std::mutex> l(build->mutex_);
      VELOX_CHECK(
          !build->stateCleared_,
          "Internal state for a peer is empty. It might have already"
          " been closed.");
      build->stateCleared_ = true;
      VELOX_CHECK_NOT_NULL(build->tableBuilder_->table());
      otherTables.push_back(build->tableBuilder_->takeTable());
      spiller = std::move(build->spiller_);
    }
    if (spiller != nullptr) {
      spiller->finishSpill(spillPartitions);
    }
  }

  if (spiller_ != nullptr) {
    spiller_->finishSpill(spillPartitions);
    removeEmptyPartitions(spillPartitions);
  }

  // TODO: Get accurate signal if parallel join build is going to be applied
  //  from hash table. Currently there is still a chance inside hash table that
  //  it might decide it is not going to trigger parallel join build.
  const bool allowParallelJoinBuild =
      !otherTables.empty() && spillPartitions.empty();

  SCOPE_EXIT {
    // Make a guard to release the unused memory reservation since we have
    // finished the merged table build.
    pool()->release();
  };

  CpuWallTiming timing;
  {
    CpuWallTimer cpuWallTimer{timing};
    tableBuilder_->table()->prepareJoinTable(
        std::move(otherTables),
        isInputFromSpill() ? spillConfig()->startPartitionBit
                           : BaseHashTable::kNoSpillInputStartPartitionBit,
        tableBuilder_->vectorHasherMaxNumDistinct(),
        tableBuilder_->dropDuplicates(),
        allowParallelJoinBuild ? operatorCtx_->task()->queryCtx()->executor()
                               : nullptr);
  }
  stats_.wlock()->addRuntimeStat(
      std::string(BaseHashTable::kBuildWallNanos),
      RuntimeCounter(timing.wallNanos, RuntimeCounter::Unit::kNanos));

  addRuntimeStats();

  // Setup spill function for spilling hash table directly from hash join
  // bridge after transferring of table ownership.
  HashJoinTableSpillFunc tableSpillFunc;
  if (canReclaim()) {
    VELOX_CHECK_NOT_NULL(spiller_);
    tableSpillFunc =
        [hashBitRange = spiller_->hashBits(),
         restoringPartitionId = restoringPartitionId_,
         joinNode = joinNode_,
         spillConfig = spillConfig(),
         spillStats = spillStats_.get()](std::shared_ptr<BaseHashTable> table) {
          return spillHashJoinTable(
              table,
              restoringPartitionId,
              hashBitRange,
              joinNode,
              spillConfig,
              spillStats);
        };
  }

  // For hash table caching: the last driver caches the merged table.
  std::shared_ptr<BaseHashTable> table = tableBuilder_->takeTable();
  maybeSetHashTableInCache(table);
  joinBridge_->setHashTable(
      table,
      std::move(spillPartitions),
      tableBuilder_->joinHasNullKeys(),
      std::move(tableSpillFunc));

  if (canSpill()) {
    stateCleared_ = true;
  }
  return true;
}

void HashBuild::ensureTableFits(uint64_t numRows) {
  // NOTE: we don't need memory reservation if all the partitions have been
  // spilled as nothing need to be built.
  if (!canSpill() || spiller_ == nullptr || spiller_->spillTriggered() ||
      numRows == 0) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool()->name())) {
    Operator::ReclaimableSectionGuard guard(this);
    memory::testingRunArbitration(pool());
    return;
  }

  TestValue::adjust("facebook::velox::exec::HashBuild::ensureTableFits", this);

  // NOTE: reserve a bit more memory to consider the extra memory used for
  // parallel table build operation.
  //
  // TODO: make this query configurable.
  const uint64_t memoryBytesToReserve =
      tableBuilder_->table()->estimateHashTableSize(numRows) * 1.1;
  {
    Operator::ReclaimableSectionGuard guard(this);
    if (pool()->maybeReserve(memoryBytesToReserve)) {
      // If reservation triggers the spilling of 'HashBuild' operator itself, we
      // will no longer need the reserved memory for building hash table as the
      // table is spilled.
      if (spiller_->spillTriggered()) {
        pool()->release();
      }
      return;
    }
  }

  LOG(WARNING) << "Failed to reserve " << succinctBytes(memoryBytesToReserve)
               << " for join table build from last hash build operator "
               << pool()->name() << ", root pool: " << pool()->root()->name()
               << ", used: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(pool()->root()->reservedBytes());
}

void HashBuild::postHashBuildProcess() {
  checkRunning();
  if (!canSpill()) {
    setState(State::kFinish);
    return;
  }

  auto spillInput = joinBridge_->spillInputOrFuture(&future_);
  if (!spillInput.has_value()) {
    VELOX_CHECK(future_.valid());
    setState(State::kWaitForProbe);
    return;
  }
  setupSpillInput(std::move(spillInput.value()));
}

void HashBuild::setupSpillInput(HashJoinBridge::SpillInput spillInput) {
  checkRunning();

  if (spillInput.spillPartition == nullptr) {
    setState(State::kFinish);
    return;
  }

  spiller_.reset();
  spillInputReader_.reset();
  restoringPartitionId_.reset();

  tableBuilder_->resetForSpillInput();
  setupSpiller(spillInput.spillPartition.get());
  stateCleared_ = false;

  // Start to process spill input.
  processSpillInput();
}

void HashBuild::processSpillInput() {
  checkRunning();

  while (spillInputReader_->nextBatch(spillInput_)) {
    addInput(std::move(spillInput_));
    if (!isRunning()) {
      return;
    }
    if (shouldYield()) {
      state_ = State::kYield;
      future_ = ContinueFuture{folly::Unit{}};
      return;
    }
  }
  noMoreInputInternal();
}

void HashBuild::addRuntimeStats() {
  // Report range sizes and number of distinct values for the join keys.
  const auto& hashers = tableBuilder_->table()->hashers();
  uint64_t asRange{0};
  uint64_t asDistinct{0};
  auto lockedStats = stats_.wlock();

  for (const auto& timing :
       tableBuilder_->table()->parallelJoinBuildStats().partitionTimings) {
    lockedStats->getOutputTiming.add(timing);
    lockedStats->addRuntimeStat(
        std::string(BaseHashTable::kParallelJoinPartitionWallNanos),
        RuntimeCounter(timing.wallNanos, RuntimeCounter::Unit::kNanos));
    lockedStats->addRuntimeStat(
        std::string(BaseHashTable::kParallelJoinPartitionCpuNanos),
        RuntimeCounter(timing.cpuNanos, RuntimeCounter::Unit::kNanos));
  }

  for (const auto& timing :
       tableBuilder_->table()->parallelJoinBuildStats().buildTimings) {
    lockedStats->getOutputTiming.add(timing);
    lockedStats->addRuntimeStat(
        std::string(BaseHashTable::kParallelJoinBuildWallNanos),
        RuntimeCounter(timing.wallNanos, RuntimeCounter::Unit::kNanos));
    lockedStats->addRuntimeStat(
        std::string(BaseHashTable::kParallelJoinBuildCpuNanos),
        RuntimeCounter(timing.cpuNanos, RuntimeCounter::Unit::kNanos));
  }

  for (const auto& timing : tableBuilder_->table()
                                ->parallelJoinBuildStats()
                                .bloomFilterPartitionTimings) {
    lockedStats->getOutputTiming.add(timing);
    if (timing.wallNanos > 0) {
      lockedStats->addRuntimeStat(
          std::string(
              BaseHashTable::kParallelJoinBloomFilterPartitionWallNanos),
          RuntimeCounter(timing.wallNanos, RuntimeCounter::Unit::kNanos));
    }
    if (timing.cpuNanos > 0) {
      lockedStats->addRuntimeStat(
          std::string(BaseHashTable::kParallelJoinBloomFilterPartitionCpuNanos),
          RuntimeCounter(timing.cpuNanos, RuntimeCounter::Unit::kNanos));
    }
  }

  for (const auto& timing : tableBuilder_->table()
                                ->parallelJoinBuildStats()
                                .bloomFilterBuildTimings) {
    lockedStats->getOutputTiming.add(timing);
    if (timing.wallNanos > 0) {
      lockedStats->addRuntimeStat(
          std::string(BaseHashTable::kParallelJoinBloomFilterBuildWallNanos),
          RuntimeCounter(timing.wallNanos, RuntimeCounter::Unit::kNanos));
    }
    if (timing.cpuNanos > 0) {
      lockedStats->addRuntimeStat(
          std::string(BaseHashTable::kParallelJoinBloomFilterBuildCpuNanos),
          RuntimeCounter(timing.cpuNanos, RuntimeCounter::Unit::kNanos));
    }
  }

  for (auto i = 0; i < hashers.size(); i++) {
    hashers[i]->cardinality(0, asRange, asDistinct);
    if (asRange != VectorHasher::kRangeTooLarge) {
      lockedStats->addRuntimeStat(
          fmt::format("rangeKey{}", i), RuntimeCounter(asRange));
    }
    if (asDistinct != VectorHasher::kRangeTooLarge) {
      lockedStats->addRuntimeStat(
          fmt::format("distinctKey{}", i), RuntimeCounter(asDistinct));
    }
  }

  tableBuilder_->table()->addRuntimeStats(lockedStats->runtimeStats);

  // Add max spilling level stats if spilling has been triggered.
  if (spiller_ != nullptr && spiller_->spillTriggered()) {
    lockedStats->addRuntimeStat(
        std::string(HashBuild::kMaxSpillLevel),
        RuntimeCounter(
            spillConfig()->spillLevel(spiller_->hashBits().begin())));
  }

  lockedStats->addRuntimeStat(
      std::string(BaseHashTable::kVectorHasherMergeCpuNanos),
      RuntimeCounter(
          tableBuilder_->table()->vectorHasherMergeTiming().cpuNanos,
          RuntimeCounter::Unit::kNanos));
}

BlockingReason HashBuild::isBlocked(ContinueFuture* future) {
  switch (state_) {
    case State::kRunning:
      if (isInputFromSpill()) {
        processSpillInput();
      }
      break;
    case State::kYield:
      setRunning();
      VELOX_CHECK(isInputFromSpill());
      processSpillInput();
      break;
    case State::kFinish:
      break;
    case State::kWaitForBuild:
      if (receivedCachedHashTable()) {
        break;
      }
      // We were waiting for peer drivers to finish - fall through to
      // kWaitForProbe which has the same logic.
      [[fallthrough]];
    case State::kWaitForProbe:
      if (!future_.valid()) {
        setRunning();
        postHashBuildProcess();
      }
      break;
    default:
      VELOX_UNREACHABLE("Unexpected state: {}", stateName(state_));
      break;
  }
  if (future_.valid()) {
    VELOX_CHECK(!isRunning() && !isFinished());
    *future = std::move(future_);
  }
  return fromStateToBlockingReason(state_);
}

bool HashBuild::isFinished() {
  return state_ == State::kFinish;
}

bool HashBuild::isRunning() const {
  return state_ == State::kRunning;
}

void HashBuild::checkRunning() const {
  VELOX_CHECK(isRunning(), stateName(state_));
}

void HashBuild::setRunning() {
  setState(State::kRunning);
}

void HashBuild::setState(State state) {
  checkStateTransition(state);
  state_ = state;
}

void HashBuild::checkStateTransition(State state) {
  VELOX_CHECK_NE(state_, state);
  switch (state) {
    case State::kRunning:
      if (!canSpill()) {
        VELOX_CHECK_EQ(state_, State::kWaitForBuild);
      } else {
        VELOX_CHECK_NE(state_, State::kFinish);
      }
      break;
    case State::kWaitForBuild:
      [[fallthrough]];
    case State::kWaitForProbe:
      [[fallthrough]];
    case State::kFinish:
      VELOX_CHECK_EQ(state_, State::kRunning);
      break;
    default:
      VELOX_UNREACHABLE(stateName(state_));
      break;
  }
}

std::string HashBuild::stateName(State state) {
  switch (state) {
    case State::kRunning:
      return "RUNNING";
    case State::kYield:
      return "YIELD";
    case State::kWaitForBuild:
      return "WAIT_FOR_BUILD";
    case State::kWaitForProbe:
      return "WAIT_FOR_PROBE";
    case State::kFinish:
      return "FINISH";
    default:
      return fmt::format("UNKNOWN: {}", static_cast<int>(state));
  }
}

bool HashBuild::canSpill() const {
  if (!Operator::canSpill()) {
    return false;
  }
  // For Cached hash table, we don't support spill either by the
  // task thats building or by the task that is re-using it
  if (useHashTableCache()) {
    return false;
  }
  if (joinNode_->isCountingJoin()) {
    return false;
  }
  if (operatorCtx_->task()->hasMixedExecutionGroupJoin(joinNode_.get())) {
    return operatorCtx_->driverCtx()
               ->queryConfig()
               .mixedGroupedModeHashJoinSpillEnabled() &&
        operatorCtx_->task()->concurrentSplitGroups() == 1;
  }
  return true;
}

bool HashBuild::canReclaim() const {
  return canSpill() && !exceededMaxSpillLevelLimit_;
}

void HashBuild::reclaim(
    uint64_t /*unused*/,
    memory::MemoryReclaimer::Stats& stats) {
  TestValue::adjust("facebook::velox::exec::HashBuild::reclaim", this);
  VELOX_CHECK(canSpill());
  auto* driver = operatorCtx_->driver();
  VELOX_CHECK_NOT_NULL(driver);
  VELOX_CHECK(!nonReclaimableSection_);

  const auto* config = spillConfig();
  VELOX_CHECK_NOT_NULL(config);
  if (UNLIKELY(exceededMaxSpillLevelLimit_)) {
    // 'canReclaim()' already checks the spill limit is not exceeding max, there
    // is only a small chance from the time 'canReclaim()' is checked to the
    // actual reclaim happens that the operator has spilled such that the spill
    // level exceeds max.
    LOG(WARNING)
        << "Can't reclaim from hash build operator, exceeded maximum spill "
           "level of "
        << config->maxSpillLevel << ", " << pool()->name()
        << ", root pool: " << pool()->root()->name()
        << ", used: " << succinctBytes(pool()->usedBytes())
        << ", reservation: " << succinctBytes(pool()->reservedBytes())
        << ", root pool reservation: "
        << succinctBytes(pool()->root()->reservedBytes());
    return;
  }

  // NOTE: a hash build operator is reclaimable if it is in the middle of table
  // build processing and is not under non-reclaimable execution section.
  if (nonReclaimableState()) {
    // TODO: reduce the log frequency if it is too verbose.
    RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
    ++stats.numNonReclaimableAttempts;
    LOG(WARNING) << "Can't reclaim from hash build operator, state_["
                 << stateName(state_) << "], nonReclaimableSection_["
                 << nonReclaimableSection_ << "], spiller_["
                 << (stateCleared_               ? "cleared"
                         : spiller_ == nullptr   ? "null"
                         : spiller_->finalized() ? "finalized"
                                                 : "non-finalized")
                 << "] " << pool()->name()
                 << ", root pool: " << pool()->root()->name()
                 << ", used: " << succinctBytes(pool()->usedBytes())
                 << ", reservation: " << succinctBytes(pool()->reservedBytes())
                 << ", root pool reservation: "
                 << succinctBytes(pool()->root()->reservedBytes());
    return;
  }

  const auto& task = driver->task();
  VELOX_CHECK(task->pauseRequested());
  const std::vector<Operator*> operators =
      task->findPeerOperators(operatorCtx_->driverCtx()->pipelineId, this);

  for (auto* op : operators) {
    HashBuild* buildOp = dynamic_cast<HashBuild*>(op);
    VELOX_CHECK_NOT_NULL(buildOp);
    VELOX_CHECK(buildOp->canSpill());
    if (buildOp->nonReclaimableState()) {
      // TODO: reduce the log frequency if it is too verbose.
      RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
      ++stats.numNonReclaimableAttempts;
      LOG(WARNING) << "Can't reclaim from hash build operator, state_["
                   << stateName(buildOp->state_) << "], nonReclaimableSection_["
                   << buildOp->nonReclaimableSection_ << "], spiller_["
                   << (buildOp->stateCleared_               ? "cleared"
                           : buildOp->spiller_ == nullptr   ? "null"
                           : buildOp->spiller_->finalized() ? "finalized"
                                                            : "non-finalized")
                   << "], " << buildOp->pool()->name()
                   << ", root pool: " << buildOp->pool()->root()->name()
                   << ", used: " << succinctBytes(buildOp->pool()->usedBytes())
                   << ", reservation: "
                   << succinctBytes(buildOp->pool()->reservedBytes())
                   << ", root pool reservation: "
                   << succinctBytes(buildOp->pool()->root()->reservedBytes());
      return;
    }
  }

  std::vector<HashBuildSpiller*> spillers;
  for (auto* op : operators) {
    HashBuild* buildOp = static_cast<HashBuild*>(op);
    spillers.push_back(buildOp->spiller_.get());
  }

  spillHashJoinTable(spillers, config);

  for (auto* op : operators) {
    HashBuild* buildOp = static_cast<HashBuild*>(op);
    buildOp->tableBuilder_->table()->clear(true);
    buildOp->pool()->release();
  }
}

memory::MemoryPool* HashBuild::tableMemoryPool() const {
  if (useHashTableCache()) {
    // Cached hash tables use a leaf pool under the query pool (from cache
    // entry). This allows the table to outlive the task while still supporting
    // allocations.
    VELOX_CHECK_NOT_NULL(cacheEntry_);
    VELOX_CHECK_NOT_NULL(cacheEntry_->tablePool);
    return cacheEntry_->tablePool.get();
  }
  // Regular joins use operator pool
  return pool();
}

bool HashBuild::nonReclaimableState() const {
  // Apart from being in the nonReclaimable section, it's also not reclaimable
  // if:
  // 1) the hash table has been built by the last build thread (indicated by
  //    state_)
  // 2) the last build operator has transferred ownership of 'this operator's
  //    internal state (the build table and spiller_) to itself.
  // 3) it has completed spilling before reaching either of the previous
  //    two states.
  return ((state_ != State::kRunning) && (state_ != State::kWaitForBuild) &&
          (state_ != State::kYield)) ||
      nonReclaimableSection_ || !spiller_ || spiller_->finalized();
}

void HashBuild::close() {
  Operator::close();

  if (useHashTableCache() && cacheEntry_ != nullptr &&
      !cacheEntry_->buildComplete && hashTableCacheBuilderTask()) {
    HashTableCache::instance()->drop(cacheKey_);
  }

  {
    // Free up major memory usage. Gate access to them as they can be accessed
    // by the last build thread that finishes building the hash table.
    std::lock_guard<std::mutex> l(mutex_);
    stateCleared_ = true;
    joinBridge_.reset();
    spiller_.reset();
    tableBuilder_->clearTable();
  }

  // Release the entry here rather than at operator destruction:
  // Driver::closeOperators() closes every operator but never destroys
  // 'operators_', so a failed build's leaf pool would otherwise stay attached
  // to the shared query pool until the Driver goes away. Keep this after
  // 'table_' is reset, whose allocations live in that pool, and outside
  // 'mutex_' -- dropping the last reference destroys the pool, which takes the
  // parent's lock via MemoryPool::dropChild().
  cacheEntry_.reset();
}

HashBuildSpiller::HashBuildSpiller(
    core::JoinType joinType,
    std::optional<SpillPartitionId> parentId,
    RowContainer* container,
    RowTypePtr rowType,
    HashBitRange bits,
    const common::SpillConfig* spillConfig,
    exec::SpillStats* spillStats)
    : SpillerBase(
          container,
          std::move(rowType),
          bits,
          {},
          spillConfig->maxFileSize,
          spillConfig->maxSpillRunRows,
          parentId,
          spillConfig,
          spillStats),
      spillProbeFlag_(needRightSideJoin(joinType)) {
  VELOX_CHECK(container_->accumulators().empty());
}

void HashBuildSpiller::spill() {
  spillTriggered_ = true;
  SpillerBase::spill(nullptr);
}

void HashBuildSpiller::spill(
    const SpillPartitionId& partitionId,
    const RowVectorPtr& spillVector) {
  VELOX_CHECK(spillTriggered_);
  VELOX_CHECK(!finalized_);
  if (FOLLY_UNLIKELY(spillVector == nullptr)) {
    return;
  }
  if (!state_.isPartitionSpilled(partitionId)) {
    state_.setPartitionSpilled(partitionId);
  }
  state_.appendToPartition(partitionId, spillVector);
}

void HashBuildSpiller::extractSpill(
    folly::Range<char**> rows,
    facebook::velox::RowVectorPtr& resultPtr) {
  if (resultPtr == nullptr) {
    resultPtr = BaseVector::create<RowVector>(
        rowType_, rows.size(), memory::spillMemoryPool());
  } else {
    resultPtr->prepareForReuse();
    resultPtr->resize(rows.size());
  }

  auto* result = resultPtr.get();
  const auto& types = container_->columnTypes();
  for (auto i = 0; i < types.size(); ++i) {
    container_->extractColumn(rows.data(), rows.size(), i, result->childAt(i));
  }
  if (spillProbeFlag_) {
    container_->extractProbedFlags(
        rows.data(), rows.size(), false, false, result->childAt(types.size()));
  }
}

} // namespace facebook::velox::exec
