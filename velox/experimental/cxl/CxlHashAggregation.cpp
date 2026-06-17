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
#include <limits>

#include <glog/logging.h>

#include "velox/experimental/cxl/CxlHashTable.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"

namespace facebook::velox::cxl {
namespace {

// Process-wide prototype diagnostics; see accessors in the header.
std::atomic<int64_t> numInitialized{0};
std::atomic<int64_t> numWithCxlPool{0};
std::atomic<int64_t> numMigrated{0};

// Groups emitted per getOutput() call when draining the CXL container.
constexpr int32_t kCxlOutputBatchRows = 10'000;
// Stats/pool label, distinct from the HashAggregation it replaces.
constexpr std::string_view kCxlPartialAggregation{"CxlPartialAggregation"};
constexpr std::string_view kCxlAggregation{"CxlAggregation"};
} // namespace

CxlHashAggregation::CxlHashAggregation(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::AggregationNode>& aggregationNode)
    : exec::HashAggregation(
          operatorId,
          driverCtx,
          aggregationNode,
          aggregationNode->step() == core::AggregationNode::Step::kPartial
            ? kCxlPartialAggregation
            : kCxlAggregation) {}

void CxlHashAggregation::initialize() {
  exec::HashAggregation::initialize();
  auto* cxlPool = customPool(kCxlResourceTag);
  VELOX_CHECK_NOT_NULL(
      cxlPool,
      "CxlHashAggregation requires a CXL pool; the adapter must not install it "
      "without one");
  // The base created the GroupingSet but not the table yet (it is built lazily
  // on the first input), so installing the factory now takes effect.
  groupingSet()->setHashTableFactory(
      [cxlPool](
          bool ignoreNullKeys,
          std::vector<std::unique_ptr<exec::VectorHasher>>&& hashers,
          const std::vector<exec::Accumulator>& accumulators,
          memory::MemoryPool* pool) -> std::unique_ptr<exec::BaseHashTable> {
        VELOX_CHECK(
            !ignoreNullKeys,
            "CxlHashAggregation does not support ignoreNullKeys aggregations");
        return CxlHashTable<false>::createForAggregation(
            std::move(hashers), accumulators, pool, cxlPool);
      });
  numInitialized.fetch_add(1, std::memory_order_relaxed);
  numWithCxlPool.fetch_add(1, std::memory_order_relaxed);
}

RowVectorPtr CxlHashAggregation::getOutput() {
  // Drain the CXL-relocated groups first. GroupingSet::getOutput reads only the
  // DRAM row container and clears the whole table (including the CXL container)
  // once that drains, so the relocated groups must be emitted before delegating.
  // Only after all input: relocation keeps filling 'cxlRows_' during the build,
  // and draining early would mark it done before it holds the final groups.
  if (noMoreInput_ && !cxlDrained_) {
    auto* groupingSet = this->groupingSet();
    auto* table = groupingSet != nullptr
        ? dynamic_cast<CxlHashTable<false>*>(groupingSet->table())
        : nullptr;
    auto* cxlRows = table != nullptr ? table->cxlRows() : nullptr;
    if (cxlRows != nullptr) {
      std::vector<char*> groups(kCxlOutputBatchRows);
      const auto numGroups = cxlRows->listRows(
          &cxlIterator_,
          kCxlOutputBatchRows,
          std::numeric_limits<uint64_t>::max(),
          groups.data());
      if (numGroups > 0) {
        auto result = std::static_pointer_cast<RowVector>(
            BaseVector::create(outputType_, numGroups, pool()));
        groupingSet->extractGroups(
            cxlRows, folly::Range<char**>(groups.data(), numGroups), result);
        return result;
      }
    }
    cxlDrained_ = true;
  }
  return exec::HashAggregation::getOutput();
}

bool CxlHashAggregation::canReclaim() const {
  auto* groupingSet = this->groupingSet();
  if (groupingSet != nullptr && groupingSet->table() != nullptr &&
      groupingSet->table()->rows()->numRows() > 0) {
    return true;
  }
  return exec::HashAggregation::canReclaim();
}

void CxlHashAggregation::reclaim(
    uint64_t targetBytes,
    memory::MemoryReclaimer::Stats& stats) {
  auto* groupingSet = this->groupingSet();
  auto* table = groupingSet != nullptr
      ? dynamic_cast<CxlHashTable<false>*>(groupingSet->table())
      : nullptr;
  if (table != nullptr && table->rows()->numRows() > 0) {
    try {
      table->relocateRowsToCxl();
      numMigrated.fetch_add(1, std::memory_order_relaxed);
      pool()->release();
      return;
    } catch (const VeloxException& e) {
      // The CXL pool is exhausted. relocateRowsToCxl empties the source only
      // after a full copy, so the table is still valid; fall back to spill.
      LOG(WARNING) << "CXL relocation failed, falling back to spill: "
                   << e.what();
    }
  }
  exec::HashAggregation::reclaim(targetBytes, stats);
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
