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

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/exec/AggregateInfo.h"
#include "velox/exec/Operator.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::cxl {

/// Experimental standalone CXL-aware hash aggregation. Drives a CxlHashTable
/// that tiers group payload from DRAM to a CXL pool under memory pressure.
class CxlHashAggregation : public exec::Operator {
 public:
  /// Operator type reported to the memory subsystem and task stats.
  static constexpr std::string_view kOperatorType{"CxlAggregation"};

  CxlHashAggregation(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::AggregationNode>& aggregationNode);

  void initialize() override;

  void addInput(RowVectorPtr input) override;

  void noMoreInput() override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

  /// Reclaimable while a partition with a CXL tier still holds DRAM-resident
  /// groups; relocating them to CXL frees DRAM so the build can continue.
  bool canReclaim() const override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

  void close() override;

 private:
  // Holds one hash partition: a CxlHashTable bound to a single memory pool plus
  // the aggregate function instances bound to that table's RowContainer. The
  // table's own rows() is the partition's row container; whether it is DRAM- or
  // CXL-resident depends on the pool it was built with.
  struct HashPartition;

  // Builds an empty partition whose CxlHashTable (and therefore row container,
  // string allocator and bucket array) is backed by 'pool'.
  std::unique_ptr<HashPartition> makePartition(memory::MemoryPool* pool);

  // Aggregates the active partition's groups for 'input'.
  void addInputToPartition(HashPartition& partition, const RowVectorPtr& input);

  // Reserves memory for 'input' before the group probe, briefly leaving the
  // driver's non-reclaimable section so arbitration can relocate this
  // operator's groups to CXL. The only point where arbitration can reach the
  // operator; a later allocation hitting the cap fails the query instead.
  void ensureInputFits(HashPartition& partition, const RowVectorPtr& input);

  // Returns true if every grouping key is fixed-width and every accumulator is
  // fixed-size without external (HashStringAllocator-backed) memory, so a row
  // can be relocated DRAM -> CXL by a plain byte copy (v1 constraint).
  bool relocationIsSafe(
      const std::vector<exec::AggregateInfo>& aggregates) const;

  const std::shared_ptr<const core::AggregationNode> aggregationNode_;

  // Input row type of the source plan node (the schema addInput() receives).
  RowTypePtr inputType_;

  // Grouping key channels in the input, and their column order in the row
  // container (identity in v1).
  std::vector<column_index_t> groupingKeyInputChannels_;
  std::vector<column_index_t> groupingKeyOutputChannels_;

  // Hash partitions. v1: a single partition (no repartition). Kept as a vector
  // so growing to N partitions (routing input by grouping-key hash) is a
  // localized change. reclaim() may replace a partition with a CXL-resident
  // copy.
  std::vector<std::unique_ptr<HashPartition>> partitions_;

  // Number of partitions. v1: 1.
  size_t numPartitions_{1};

  // Partition currently being drained by getOutput(), and the index into that
  // partition's row containers (DRAM then CXL).
  size_t outputPartition_{0};
  size_t outputContainerIdx_{0};

  // Per-query CXL custom pool, resolved by tag in initialize(); null when the
  // query has no CXL tier.
  memory::MemoryPool* cxlPool_{nullptr};

  // Output iteration state.
  exec::RowContainerIterator outputIterator_;
  bool finished_{false};

  // Reusable args buffer for Aggregate::addRawInput.
  std::vector<VectorPtr> tempVectors_;
};

/// Registers a DriverAdapter that replaces every exec::HashAggregation in a
/// pipeline with a standalone CxlHashAggregation. Call once at startup.
void registerCxlHashAggregationAdapter();

/// Prototype diagnostics for tests (process-wide). Not a stable API.
int64_t numCxlHashAggregationsInitialized();
int64_t numCxlHashAggregationsWithCxlPool();
int64_t numCxlPartitionsMigrated();

/// Resets the diagnostics above.
void resetCxlHashAggregationCounters();

} // namespace facebook::velox::cxl
