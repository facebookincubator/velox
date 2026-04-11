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

#include "velox/exec/MarkDistinct.h"

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/vector/FlatVector.h"

#include <algorithm>
#include <utility>

namespace facebook::velox::exec {

MarkDistinct::MarkDistinct(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::MarkDistinctNode>& planNode)
    : Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          OperatorType::kMarkDistinct,
          planNode->canSpill(driverCtx->queryConfig())
              ? driverCtx->makeSpillConfig(operatorId, planNode->name())
              : std::nullopt) {
  inputType_ = planNode->sources()[0]->outputType();

  // Set all input columns as identity projection.
  for (auto i = 0; i < inputType_->size(); ++i) {
    identityProjections_.emplace_back(i, i);
  }

  // Use result[0] for distinct mask output.
  resultProjections_.emplace_back(0, inputType_->size());

  for (const auto& key : planNode->distinctKeys()) {
    distinctKeyChannels_.push_back(inputType_->getChildIdx(key->name()));
  }

  groupingSet_ = GroupingSet::createForDistinct(
      inputType_,
      createVectorHashers(inputType_, planNode->distinctKeys()),
      /*preGroupedKeys=*/{},
      operatorCtx_.get(),
      &nonReclaimableSection_);

  results_.resize(1);

  if (spillEnabled()) {
    setSpillPartitionBits();
  }
}

void MarkDistinct::addInput(RowVectorPtr input) {
  ensureInputFits(input);

  if (inputSpiller_ != nullptr) {
    spillInput(input, pool());
    return;
  }

  // Don't add to the hash table here. We defer it to getOutput() so that if
  // spill() is called between addInput() and getOutput(), the hash table spill
  // won't include this input's keys. This prevents those keys from being
  // re-suppressed during restore.
  input_ = std::move(input);
}

void MarkDistinct::noMoreInput() {
  Operator::noMoreInput();

  if (inputSpiller_ != nullptr) {
    finishSpillInputAndRestoreNext();
  }
}

void MarkDistinct::finishSpillInputAndRestoreNext() {
  VELOX_CHECK_NOT_NULL(inputSpiller_);
  inputSpiller_->finishSpill(spillInputPartitionSet_);
  inputSpiller_.reset();
  removeEmptyPartitions(spillInputPartitionSet_);
  restoreNextSpillPartition();
}

void MarkDistinct::restoreNextSpillPartition() {
  if (spillInputPartitionSet_.empty()) {
    return;
  }

  auto it = spillInputPartitionSet_.begin();
  restoringPartitionId_ = it->first;

  spillInputReader_ = it->second->createUnorderedReader(
      spillConfig_->readBufferSize, pool(), spillStats_.get());

  auto hashTableIt = spillHashTablePartitionSet_.find(it->first);
  if (hashTableIt != spillHashTablePartitionSet_.end()) {
    auto reader = hashTableIt->second->createUnorderedReader(
        spillConfig_->readBufferSize, pool(), spillStats_.get());

    setSpillPartitionBits(&(it->first));

    auto* table = groupingSet_->table();
    const auto& hashers = table->hashers();
    auto lookup = std::make_unique<HashLookup>(hashers, pool());

    std::vector<VectorPtr> columns(inputType_->size());
    RowVectorPtr data;
    while (reader->nextBatch(data)) {
      for (auto i = 0; i < hashers.size(); ++i) {
        columns[hashers[i]->channel()] = data->childAt(i);
      }

      auto input = std::make_shared<RowVector>(
          pool(), inputType_, nullptr, data->size(), std::move(columns));

      SelectivityVector rows(data->size());
      table->prepareForGroupProbe(
          *lookup, input, rows, spillConfig_->startPartitionBit);
      table->groupProbe(*lookup, spillConfig_->startPartitionBit);

      columns.assign(inputType_->size(), nullptr);
    }
  }

  spillInputPartitionSet_.erase(it);

  RowVectorPtr spilledInput;
  spillInputReader_->nextBatch(spilledInput);
  VELOX_CHECK_NOT_NULL(spilledInput);
  addInput(std::move(spilledInput));
}

void MarkDistinct::recursiveSpillInput() {
  RowVectorPtr spilledInput;
  while (spillInputReader_->nextBatch(spilledInput)) {
    spillInput(spilledInput, pool());

    if (shouldYield()) {
      yield_ = true;
      return;
    }
  }

  finishSpillInputAndRestoreNext();
}

RowVectorPtr MarkDistinct::getOutput() {
  if (isFinished()) {
    return nullptr;
  }

  if (input_ == nullptr) {
    if (spillInputReader_ == nullptr) {
      return nullptr;
    }

    recursiveSpillInput();
    if (yield_) {
      yield_ = false;
      return nullptr;
    }

    if (input_ == nullptr) {
      return nullptr;
    }
  }

  // Add the current input to the hash table now, just before producing output.
  // This is deferred from addInput() so that if spill() is called between
  // addInput() and getOutput(), the hash table doesn't contain this input's
  // keys — ensuring they are correctly marked as new during restore.
  groupingSet_->addInput(input_, /*mayPushdown=*/false);

  const auto outputSize = input_->size();

  VectorPtr& result = results_[0];
  if (result && result.use_count() == 1) {
    BaseVector::prepareForReuse(result, outputSize);
  } else {
    result = BaseVector::create(BOOLEAN(), outputSize, operatorCtx_->pool());
  }

  auto* resultBits =
      results_[0]->as<FlatVector<bool>>()->mutableRawValues<uint64_t>();
  bits::fillBits(resultBits, 0, outputSize, false);
  for (const auto i : groupingSet_->hashLookup().newGroups) {
    bits::setBit(resultBits, i, true);
  }

  auto output = fillOutput(outputSize, nullptr);
  input_ = nullptr;

  if (spillInputReader_ != nullptr) {
    RowVectorPtr spilledInput;
    if (spillInputReader_->nextBatch(spilledInput)) {
      addInput(std::move(spilledInput));
    } else {
      spillInputReader_.reset();
      restoringPartitionId_.reset();
      groupingSet_->resetTable(true);
      restoreNextSpillPartition();
    }
  }

  return output;
}

bool MarkDistinct::isFinished() {
  return noMoreInput_ && input_ == nullptr && spillInputReader_ == nullptr;
}

void MarkDistinct::ensureInputFits(const RowVectorPtr& input) {
  if (!spillEnabled() || inputSpiller_ != nullptr) {
    return;
  }

  const auto numDistinct = groupingSet_->numDistinct();
  if (numDistinct == 0) {
    return;
  }

  auto* table = groupingSet_->table();
  auto* rows = table->rows();
  const auto [freeRows, outOfLineFreeBytes] = rows->freeSpace();
  const auto outOfLineBytes =
      rows->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const auto outOfLineBytesPerRow = outOfLineBytes / numDistinct;

  if (testingTriggerSpill(pool()->name())) {
    Operator::ReclaimableSectionGuard guard(this);
    memory::testingRunArbitration(pool());
    return;
  }

  const auto currentUsage = pool()->usedBytes();
  const auto minReservationBytes =
      currentUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool()->availableReservation();
  const auto tableIncrementBytes =
      static_cast<int64_t>(table->hashTableSizeIncrease(input->size()));
  const auto incrementBytes =
      static_cast<int64_t>(rows->sizeIncrement(
          input->size(), outOfLineBytesPerRow * input->size())) +
      tableIncrementBytes;

  if (availableReservationBytes >= minReservationBytes) {
    if ((tableIncrementBytes == 0) && (freeRows > input->size()) &&
        (outOfLineBytes == 0 ||
         outOfLineFreeBytes >= outOfLineBytesPerRow * input->size())) {
      return;
    }

    if (availableReservationBytes > 2 * incrementBytes) {
      return;
    }
  }

  const auto targetIncrementBytes = std::max<int64_t>(
      incrementBytes * 2,
      currentUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    Operator::ReclaimableSectionGuard guard(this);
    if (pool()->maybeReserve(targetIncrementBytes)) {
      if (inputSpiller_ != nullptr) {
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

void MarkDistinct::reclaim(
    uint64_t /* unused */,
    memory::MemoryReclaimer::Stats& /* unused */) {
  VELOX_CHECK(canReclaim());
  VELOX_CHECK(!nonReclaimableSection_);

  if (groupingSet_->numDistinct() == 0) {
    return;
  }

  if (FOLLY_UNLIKELY(exceededMaxSpillLevelLimit_)) {
    LOG(WARNING) << "Exceeded mark distinct spill level limit: "
                 << spillConfig_->maxSpillLevel
                 << ", and abandon spilling for memory pool: " << pool()->name()
                 << ", root pool: " << pool()->root()->name()
                 << ", used: " << succinctBytes(pool()->usedBytes())
                 << ", reservation: " << succinctBytes(pool()->reservedBytes())
                 << ", root pool reservation: "
                 << succinctBytes(pool()->root()->reservedBytes());
    spillStats_->spillMaxLevelExceededCount.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }

  spill();
}

SpillPartitionIdSet MarkDistinct::spillHashTable() {
  VELOX_CHECK_GT(groupingSet_->numDistinct(), 0);

  auto* table = groupingSet_->table();
  auto columnTypes = table->rows()->columnTypes();
  auto tableType = ROW(std::move(columnTypes));

  auto hashTableSpiller = std::make_unique<MarkDistinctHashTableSpiller>(
      table->rows(),
      restoringPartitionId_,
      tableType,
      spillPartitionBits_,
      &spillConfig_.value(),
      spillStats_.get());

  hashTableSpiller->spill();
  hashTableSpiller->finishSpill(spillHashTablePartitionSet_);

  groupingSet_->resetTable(true);
  pool()->release();
  return hashTableSpiller->state().spilledPartitionIdSet();
}

void MarkDistinct::setupInputSpiller(
    const SpillPartitionIdSet& spillPartitionIdSet) {
  VELOX_CHECK(!spillPartitionIdSet.empty());

  inputSpiller_ = std::make_unique<NoRowContainerSpiller>(
      inputType_,
      restoringPartitionId_,
      spillPartitionBits_,
      &spillConfig_.value(),
      spillStats_.get());

  spillHashFunction_ = std::make_unique<HashPartitionFunction>(
      inputSpiller_->hashBits(), inputType_, distinctKeyChannels_);
}

void MarkDistinct::spill() {
  VELOX_CHECK(spillEnabled());

  spilled_ = true;

  const auto spillPartitionIdSet = spillHashTable();
  VELOX_CHECK_EQ(groupingSet_->numDistinct(), 0);

  setupInputSpiller(spillPartitionIdSet);
  if (input_ != nullptr) {
    spillInput(input_, memory::spillMemoryPool());
    input_ = nullptr;
  }
  results_.clear();
  results_.resize(1);
}

void MarkDistinct::spillInput(
    const RowVectorPtr& input,
    memory::MemoryPool* pool) {
  const auto numRows = input->size();

  std::vector<uint32_t> partitionAssignments(numRows);
  const auto singlePartition =
      spillHashFunction_->partition(*input, partitionAssignments);

  const auto numPartitions = spillHashFunction_->numPartitions();

  std::vector<BufferPtr> partitionIndices(numPartitions);
  std::vector<vector_size_t*> rawPartitionIndices(numPartitions);

  for (auto i = 0; i < numPartitions; ++i) {
    partitionIndices[i] = allocateIndices(numRows, pool);
    rawPartitionIndices[i] = partitionIndices[i]->asMutable<vector_size_t>();
  }

  std::vector<vector_size_t> numSpillInputs(numPartitions, 0);

  for (auto row = 0; row < numRows; ++row) {
    const auto partition = singlePartition.has_value()
        ? singlePartition.value()
        : partitionAssignments[row];
    rawPartitionIndices[partition][numSpillInputs[partition]++] = row;
  }

  // Ensure vector are lazy loaded before spilling.
  for (auto i = 0; i < input->childrenSize(); ++i) {
    input->childAt(i)->loadedVector();
  }

  for (auto partition = 0; partition < numSpillInputs.size(); ++partition) {
    const auto numInputs = numSpillInputs[partition];
    if (numInputs == 0) {
      continue;
    }

    inputSpiller_->spill(
        SpillPartitionId(partition),
        wrap(numInputs, partitionIndices[partition], input));
  }
}

void MarkDistinct::setSpillPartitionBits(
    const SpillPartitionId* restoredPartitionId) {
  const auto startPartitionBitOffset = restoredPartitionId == nullptr
      ? spillConfig_->startPartitionBit
      : partitionBitOffset(
            *restoredPartitionId,
            spillConfig_->startPartitionBit,
            spillConfig_->numPartitionBits) +
          spillConfig_->numPartitionBits;
  if (spillConfig_->exceedSpillLevelLimit(startPartitionBitOffset)) {
    exceededMaxSpillLevelLimit_ = true;
    return;
  }

  exceededMaxSpillLevelLimit_ = false;
  spillPartitionBits_ = HashBitRange(
      startPartitionBitOffset,
      startPartitionBitOffset + spillConfig_->numPartitionBits);
}

MarkDistinctHashTableSpiller::MarkDistinctHashTableSpiller(
    RowContainer* container,
    std::optional<SpillPartitionId> parentId,
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
          spillStats) {}

void MarkDistinctHashTableSpiller::spill() {
  SpillerBase::spill(nullptr);
}
} // namespace facebook::velox::exec
