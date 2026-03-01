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

  // We will use result[0] for distinct mask output.
  resultProjections_.emplace_back(0, inputType_->size());

  for (const auto& key : planNode->distinctKeys()) {
    distinctKeyChannels_.push_back(inputType_->getChildIdx(key->name()));
  }

  groupingSet_ = GroupingSet::createForDistinct(
      inputType_,
      createVectorHashers(inputType_, planNode->distinctKeys()),
      /*preGroupedKeys=*/{},
      /*spillConfig=*/nullptr,
      operatorCtx_.get(),
      &nonReclaimableSection_);

  results_.resize(1);

  if (spillEnabled()) {
    setSpillPartitionBits();
  }
}

void MarkDistinct::addInput(RowVectorPtr input) {
  if (spillEnabled()) {
    ensureInputFits(input);

    if (inputSpiller_ != nullptr) {
      spillInput(input, pool());
      return;
    }
  }

  groupingSet_->addInput(input, /*mayPushdown=*/false);

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

  // Set up spill partition bits for potential recursive spilling.
  setSpillPartitionBits(&(it->first));
  spillInputPartitionSet_.erase(it);
  processSpilledInput();
}

void MarkDistinct::processSpilledInput() {
  RowVectorPtr spilledInput;
  if (spillInputReader_->nextBatch(spilledInput)) {
    addInput(std::move(spilledInput));

    // Check if recursive spill was triggered during addInput().
    // If so, we need to spill all remaining data from the current partition
    // before moving to the restored partition.
    if (inputSpiller_ != nullptr) {
      recursiveSpillInput();
    }
  }
}

void MarkDistinct::recursiveSpillInput() {
  VELOX_CHECK_NOT_NULL(inputSpiller_);
  VELOX_CHECK_NOT_NULL(spillInputReader_);

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

  if (spillInputReader_ != nullptr) {
    if (input_ == nullptr) {
      processSpilledInput();
      if (yield_) {
        yield_ = false;
        return nullptr;
      }
      if (input_ == nullptr) {
        spillInputReader_.reset();
        restoringPartitionId_.reset();
        groupingSet_->resetTable(true);
        restoreNextSpillPartition();
        return nullptr;
      }
    }
  }

  if (!input_) {
    return nullptr;
  }

  auto outputSize = input_->size();
  // Re-use memory for the ID vector if possible.
  VectorPtr& result = results_[0];
  if (result && result.use_count() == 1) {
    BaseVector::prepareForReuse(result, outputSize);
  } else {
    result = BaseVector::create(BOOLEAN(), outputSize, operatorCtx_->pool());
  }

  // newGroups contains the indices of distinct rows.
  // For each index in newGroups, we mark the index'th bit true in the result
  // vector.
  auto resultBits =
      results_[0]->as<FlatVector<bool>>()->mutableRawValues<uint64_t>();

  bits::fillBits(resultBits, 0, outputSize, false);
  for (const auto i : groupingSet_->hashLookup().newGroups) {
    bits::setBit(resultBits, i, true);
  }
  auto output = fillOutput(outputSize, nullptr);

  // Drop reference to input_ to make it singly-referenced at the producer and
  // allow for memory reuse.
  input_ = nullptr;

  if (spillInputReader_ != nullptr) {
    processSpilledInput();
  }

  return output;
}

bool MarkDistinct::isFinished() {
  return noMoreInput_ && !input_ && spillInputReader_ == nullptr;
}

void MarkDistinct::ensureInputFits(const RowVectorPtr& input) {
  if (!spillEnabled() || inputSpiller_ != nullptr) {
    return;
  }

  const auto numDistinct = groupingSet_->numDistinct();
  if (numDistinct == 0) {
    return;
  }

  // Check hash table free space to see if we already have room.
  auto* rows = groupingSet_->rows();
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
      groupingSet_->hashTableSizeIncrease(input->size());
  const auto incrementBytes =
      rows->sizeIncrement(input->size(), outOfLineBytesPerRow * input->size()) +
      tableIncrementBytes;

  // First check if we have sufficient minimal memory reservation.
  if (availableReservationBytes >= minReservationBytes) {
    // Check if we have enough free row slots and variable-length space.
    if ((tableIncrementBytes == 0) && (freeRows > input->size()) &&
        (outOfLineBytes == 0 ||
         outOfLineFreeBytes >= outOfLineBytesPerRow * input->size())) {
      return;
    }

    // Check if available reservation is plenty for the increment.
    if (availableReservationBytes > 2 * incrementBytes) {
      return;
    }
  }

  // Check if we can increase reservation.
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
               << ", usage: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes());
}

void MarkDistinct::reclaim(
    /*targetBytes=*/uint64_t,
    /*stats=*/memory::MemoryReclaimer::Stats&) {
  VELOX_CHECK(canReclaim());
  VELOX_CHECK(!nonReclaimableSection_);

  if (groupingSet_->numDistinct() == 0) {
    return;
  }

  if (exceededMaxSpillLevelLimit_) {
    LOG(WARNING) << "Exceeded mark distinct spill level limit: "
                 << spillConfig_->maxSpillLevel
                 << ", and abandon spilling for memory pool: "
                 << pool()->name();
    spillStats_->spillMaxLevelExceededCount.fetch_add(
        1, std::memory_order_relaxed);
    return;
  }

  spill();
}

SpillPartitionIdSet MarkDistinct::spillHashTable() {
  VELOX_CHECK_GT(groupingSet_->numDistinct(), 0);

  // MarkDistinct only tracks key presence, so we simply discard the hash
  // table and rebuild it from spilled input during restore.
  // No hash table contents are written to disk.
  // The returned partitionIdSet is only used for assertion safety and code
  // alignment with other spillable operators.
  SpillPartitionIdSet partitionIdSet;
  const auto numPartitions = 1 << spillPartitionBits_.numBits();
  for (auto i = 0; i < numPartitions; ++i) {
    partitionIdSet.emplace(SpillPartitionId(i));
  }

  groupingSet_->resetTable(true);
  pool()->release();
  return partitionIdSet;
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
  const auto numInput = input->size();

  std::vector<uint32_t> spillPartitions(numInput);
  const auto singlePartition =
      spillHashFunction_->partition(*input, spillPartitions);

  const auto numPartitions = spillHashFunction_->numPartitions();

  std::vector<BufferPtr> partitionIndices(numPartitions);
  std::vector<vector_size_t*> rawPartitionIndices(numPartitions);

  for (auto i = 0; i < numPartitions; ++i) {
    partitionIndices[i] = allocateIndices(numInput, pool);
    rawPartitionIndices[i] = partitionIndices[i]->asMutable<vector_size_t>();
  }

  std::vector<vector_size_t> numSpillInputs(numPartitions, 0);

  for (auto row = 0; row < numInput; ++row) {
    const auto partition = singlePartition.has_value() ? singlePartition.value()
                                                       : spillPartitions[row];
    rawPartitionIndices[partition][numSpillInputs[partition]++] = row;
  }

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
} // namespace facebook::velox::exec
