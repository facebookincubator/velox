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

#include "velox/experimental/cxl/CxlHashAggregation.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <numeric>

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateInfo.h"
#include "velox/exec/Driver.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/VectorHasher.h"
#include "velox/experimental/cxl/CxlHashTable.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"

namespace facebook::velox::cxl {
namespace {

// Process-wide prototype diagnostics; see accessors in the header.
std::atomic<int64_t> numInitialized{0};
std::atomic<int64_t> numWithCxlPool{0};
std::atomic<int64_t> numMigrated{0};

// v1 reads back the whole table in batches of this many groups per getOutput().
constexpr int32_t kOutputBatchRows = 10'000;

} // namespace

// A hash partition: a CxlHashTable bound to one memory pool, plus the aggregate
// function instances bound to that table's RowContainer. The table's rows() is
// the partition's row container (DRAM- or CXL-resident depending on 'pool').
struct CxlHashAggregation::HashPartition {
  // A CxlHashTable holding a DRAM row container (rows_) and, when a CXL pool is
  // provided, a CXL row container (cxlRows_). New groups land in DRAM;
  // relocateAllToCxl() moves them to CXL.
  std::unique_ptr<CxlHashTable<false>> table;
  std::vector<exec::AggregateInfo> aggregates;
  std::unique_ptr<exec::HashLookup> lookup;
  // True once this partition's rows have been relocated to the CXL container.
  bool onCxl{false};
};

CxlHashAggregation::CxlHashAggregation(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::AggregationNode>& aggregationNode)
    : exec::Operator(
          driverCtx,
          aggregationNode->outputType(),
          operatorId,
          aggregationNode->id(),
          std::string{kOperatorType}),
      aggregationNode_(aggregationNode) {}

std::unique_ptr<CxlHashAggregation::HashPartition>
CxlHashAggregation::makePartition(memory::MemoryPool* pool) {
  auto partition = std::make_unique<HashPartition>();

  auto hashers =
      exec::createVectorHashers(inputType_, groupingKeyInputChannels_);
  const auto numKeys = hashers.size();

  std::shared_ptr<core::ExpressionEvaluator> expressionEvaluator;
  partition->aggregates = exec::toAggregateInfo(
      *aggregationNode_, *operatorCtx_, numKeys, expressionEvaluator);

  std::vector<exec::Accumulator> accumulators;
  accumulators.reserve(partition->aggregates.size());
  for (auto& aggregate : partition->aggregates) {
    accumulators.push_back(
        exec::Accumulator{
            aggregate.function.get(), aggregate.intermediateType});
  }

  // Give the table a CXL-backed second container when a CXL tier is configured
  // and relocation is safe for this aggregation's row layout.
  auto* cxlPool =
      (cxlPool_ != nullptr && relocationIsSafe(partition->aggregates))
      ? cxlPool_
      : nullptr;
  partition->table = CxlHashTable<false>::createForAggregation(
      std::move(hashers), accumulators, pool, cxlPool);

  // Bind each aggregate to the table's row container (allocator + offsets).
  auto& rows = *partition->table->rows();
  for (auto i = 0; i < partition->aggregates.size(); ++i) {
    auto& function = partition->aggregates[i].function;
    function->setAllocator(&rows.stringAllocator());
    const auto rowColumn = rows.columnAt(numKeys + i);
    function->setOffsets(
        rowColumn.offset(),
        rowColumn.nullByte(),
        rowColumn.nullMask(),
        rowColumn.initializedByte(),
        rowColumn.initializedMask(),
        rows.rowSizeOffset());
  }

  partition->lookup =
      std::make_unique<exec::HashLookup>(partition->table->hashers(), pool);
  return partition;
}

void CxlHashAggregation::initialize() {
  exec::Operator::initialize();
  VELOX_CHECK(pool()->trackUsage());

  inputType_ = aggregationNode_->sources()[0]->outputType();

  const auto& groupingKeys = aggregationNode_->groupingKeys();
  VELOX_USER_CHECK(
      !groupingKeys.empty(),
      "CxlHashAggregation v1 requires at least one grouping key");
  groupingKeyInputChannels_.reserve(groupingKeys.size());
  for (const auto& key : groupingKeys) {
    groupingKeyInputChannels_.push_back(
        exec::exprToChannel(key.get(), inputType_));
  }
  // v1: grouping key output order matches the row container column order.
  groupingKeyOutputChannels_.resize(groupingKeys.size());
  std::iota(
      groupingKeyOutputChannels_.begin(), groupingKeyOutputChannels_.end(), 0);

  cxlPool_ = customPool(kCxlResourceTag);

  partitions_.reserve(numPartitions_);
  for (size_t i = 0; i < numPartitions_; ++i) {
    partitions_.push_back(makePartition(pool()));
  }

  numInitialized.fetch_add(1, std::memory_order_relaxed);
  if (cxlPool_ != nullptr) {
    numWithCxlPool.fetch_add(1, std::memory_order_relaxed);
  }
}

void CxlHashAggregation::ensureInputFits(
    HashPartition& partition,
    const RowVectorPtr& input) {
  auto* table = partition.table.get();
  if (table == nullptr || table->numDistinct() == 0) {
    // Nothing DRAM-resident to relocate yet; the first allocations either fit
    // or fail on their own.
    return;
  }

  auto* rows = table->rows();
  const auto [freeRows, outOfLineFreeBytes] = rows->freeSpace();
  const auto outOfLineBytes =
      rows->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const int64_t flatBytes = input->estimateFlatSize();
  const auto tableIncrementBytes = table->hashTableSizeIncrease(input->size());

  // Existing free capacity absorbs the worst case: every input row is a new
  // group, and variable-width data grows by twice the input's flat size.
  if (tableIncrementBytes == 0 && freeRows > input->size() &&
      (outOfLineBytes == 0 || outOfLineFreeBytes >= flatBytes * 2)) {
    return;
  }

  const auto incrementBytes =
      rows->sizeIncrement(input->size(), outOfLineBytes ? flatBytes * 2 : 0) +
      tableIncrementBytes;
  // Reserve twice the increment so the reservation (and any arbitration it
  // triggers) is amortized over multiple input batches.
  const auto targetIncrementBytes = incrementBytes * 2;
  {
    exec::Operator::ReclaimableSectionGuard guard(this);
    if (pool()->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }
  // A failed reservation is not fatal: the arbitration inside maybeReserve()
  // may already have relocated this operator's groups to CXL, and the actual
  // increment is below the reservation target.
  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", used: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes());
}

void CxlHashAggregation::addInputToPartition(
    HashPartition& partition,
    const RowVectorPtr& input) {
  ensureInputFits(partition, input);

  const auto numRows = input->size();
  SelectivityVector rows(numRows);

  partition.table->prepareForGroupProbe(
      *partition.lookup,
      input,
      rows,
      exec::BaseHashTable::kNoSpillInputStartPartitionBit);
  if (partition.lookup->rows.empty()) {
    return;
  }
  partition.table->groupProbe(
      *partition.lookup, exec::BaseHashTable::kNoSpillInputStartPartitionBit);

  auto* groups = partition.lookup->hits.data();
  const auto& newGroups = partition.lookup->newGroups;

  for (auto i = 0; i < partition.aggregates.size(); ++i) {
    auto& function = partition.aggregates[i].function;
    if (!newGroups.empty()) {
      function->initializeNewGroups(groups, newGroups);
    }
    const auto& channels = partition.aggregates[i].inputs;
    const auto& constants = partition.aggregates[i].constantInputs;
    tempVectors_.resize(channels.size());
    for (auto j = 0; j < channels.size(); ++j) {
      if (channels[j] == kConstantChannel) {
        tempVectors_[j] = BaseVector::wrapInConstant(numRows, 0, constants[j]);
      } else {
        tempVectors_[j] = input->childAt(channels[j]);
      }
    }
    function->addRawInput(groups, rows, tempVectors_, /*mayPushdown=*/false);
  }
  tempVectors_.clear();
}

void CxlHashAggregation::addInput(RowVectorPtr input) {
  // v1: a single partition takes all input. TODO: when numPartitions_ > 1,
  // split 'input' by grouping-key hash and route each row subset to its
  // partition (partition = hash % numPartitions_).
  addInputToPartition(*partitions_[0], input);
  common::testutil::TestValue::adjust(
      "facebook::velox::cxl::CxlHashAggregation::addInput", this);
}

void CxlHashAggregation::noMoreInput() {
  exec::Operator::noMoreInput();
  // Release the memory reserved for input processing; output drains in place.
  pool()->release();
}

bool CxlHashAggregation::relocationIsSafe(
    const std::vector<exec::AggregateInfo>& aggregates) const {
  // Variable-width keys store out-of-line StringViews into the row container's
  // HashStringAllocator; a byte copy to CXL would leave dangling pointers.
  for (const auto channel : groupingKeyInputChannels_) {
    if (!inputType_->childAt(channel)->isFixedWidth()) {
      return false;
    }
  }
  // Accumulators with external (HashStringAllocator-backed) state can't be
  // byte-copied either.
  for (const auto& aggregate : aggregates) {
    if (aggregate.function->accumulatorUsesExternalMemory()) {
      return false;
    }
  }
  return true;
}

bool CxlHashAggregation::canReclaim() const {
  for (const auto& partition : partitions_) {
    // Reclaimable only if the table has a CXL container and DRAM-resident rows.
    if (partition != nullptr && partition->table != nullptr &&
        partition->table->cxlRows() != nullptr &&
        partition->table->rows()->numRows() > 0) {
      return true;
    }
  }
  return false;
}

void CxlHashAggregation::reclaim(
    uint64_t /*targetBytes*/,
    memory::MemoryReclaimer::Stats& /*stats*/) {
  // Pick the partition with the most DRAM-resident memory and relocate its rows
  // into its own CXL container (in place; the same table indexes them after).
  int64_t maxBytes = -1;
  size_t victim = partitions_.size();
  for (size_t i = 0; i < partitions_.size(); ++i) {
    auto& partition = partitions_[i];
    if (partition == nullptr || partition->table == nullptr ||
        partition->table->cxlRows() == nullptr ||
        partition->table->rows()->numRows() == 0) {
      continue;
    }
    const int64_t bytes = partition->table->rows()->allocatedBytes();
    if (bytes > maxBytes) {
      maxBytes = bytes;
      victim = i;
    }
  }
  if (victim == partitions_.size()) {
    return;
  }
  partitions_[victim]->table->relocateAllToCxl();
  partitions_[victim]->onCxl = true;
  numMigrated.fetch_add(1, std::memory_order_relaxed);
  // Return the unused reservation so the arbitrator sees the freed capacity.
  pool()->release();
}

RowVectorPtr CxlHashAggregation::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  const auto numKeys = groupingKeyOutputChannels_.size();
  std::vector<char*> groups(kOutputBatchRows);
  while (outputPartition_ < partitions_.size()) {
    auto& partition = *partitions_[outputPartition_];
    // Drain the partition's row containers in turn (DRAM rows_, then CXL
    // cxlRows_). Groups live in exactly one of them per row, so each is emitted
    // once.
    const auto containers = partition.table->allRows();
    while (outputContainerIdx_ < containers.size()) {
      auto* rowContainer = containers[outputContainerIdx_];
      const int32_t numGroups = rowContainer->listRows(
          &outputIterator_,
          kOutputBatchRows,
          std::numeric_limits<uint64_t>::max(),
          groups.data());
      if (numGroups == 0) {
        ++outputContainerIdx_;
        outputIterator_ = exec::RowContainerIterator{};
        continue;
      }

      auto result = std::static_pointer_cast<RowVector>(
          BaseVector::create(outputType_, numGroups, operatorCtx_->pool()));
      for (auto i = 0; i < numKeys; ++i) {
        rowContainer->extractColumn(
            groups.data(),
            numGroups,
            groupingKeyOutputChannels_[i],
            result->childAt(i));
      }
      for (auto i = 0; i < partition.aggregates.size(); ++i) {
        auto& function = partition.aggregates[i].function;
        auto& aggregateVector = result->childAt(numKeys + i);
        function->extractValues(groups.data(), numGroups, &aggregateVector);
      }
      return result;
    }
    // This partition is drained; advance to the next one.
    ++outputPartition_;
    outputContainerIdx_ = 0;
    outputIterator_ = exec::RowContainerIterator{};
  }

  finished_ = true;
  return nullptr;
}

bool CxlHashAggregation::isFinished() {
  return noMoreInput_ && finished_;
}

void CxlHashAggregation::close() {
  partitions_.clear();
  exec::Operator::close();
}

namespace {

std::shared_ptr<const core::AggregationNode> findAggregationNode(
    const exec::DriverFactory& factory,
    const core::PlanNodeId& planNodeId) {
  for (const auto& node : factory.planNodes) {
    if (node->id() == planNodeId) {
      return std::dynamic_pointer_cast<const core::AggregationNode>(node);
    }
  }
  return nullptr;
}

// Replaces every stock HashAggregation in the driver with a standalone
// CxlHashAggregation built from the same AggregationNode.
bool adaptDriver(const exec::DriverFactory& factory, exec::Driver& driver) {
  const auto operators = driver.operators();
  auto* driverCtx = driver.driverCtx();
  bool replaced = false;
  for (size_t i = 0; i < operators.size(); ++i) {
    auto* aggregation = dynamic_cast<exec::HashAggregation*>(operators[i]);
    if (aggregation == nullptr) {
      continue;
    }
    auto aggregationNode =
        findAggregationNode(factory, aggregation->planNodeId());
    if (aggregationNode == nullptr) {
      continue;
    }
    std::vector<std::unique_ptr<exec::Operator>> replacement;
    replacement.push_back(
        std::make_unique<CxlHashAggregation>(
            aggregation->operatorId(), driverCtx, aggregationNode));
    factory.replaceOperators(
        driver,
        static_cast<int32_t>(i),
        static_cast<int32_t>(i + 1),
        std::move(replacement));
    replaced = true;
  }
  return replaced;
}

} // namespace

void registerCxlHashAggregationAdapter() {
  exec::DriverAdapter adapter{
      std::string{kCxlResourceTag}, /*inspect=*/{}, adaptDriver};
  exec::DriverFactory::registerAdapter(std::move(adapter));
}

int64_t numCxlHashAggregationsInitialized() {
  return numInitialized.load(std::memory_order_relaxed);
}

int64_t numCxlHashAggregationsWithCxlPool() {
  return numWithCxlPool.load(std::memory_order_relaxed);
}

int64_t numCxlPartitionsMigrated() {
  return numMigrated.load(std::memory_order_relaxed);
}

void resetCxlHashAggregationCounters() {
  numInitialized.store(0, std::memory_order_relaxed);
  numWithCxlPool.store(0, std::memory_order_relaxed);
  numMigrated.store(0, std::memory_order_relaxed);
}

} // namespace facebook::velox::cxl
