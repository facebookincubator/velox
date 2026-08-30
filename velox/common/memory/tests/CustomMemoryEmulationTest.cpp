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

#include <gtest/gtest.h>
#include <array>

#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::memory::test {
namespace {

struct EmulatedRow {
  int64_t key;
  int64_t sum;
};

class EmulatedCxlHashAggregation;

// Reclaimer installed on the operator's DRAM leaf row pool. Forwards
// reclaim() to the operator's DRAM -> CXL spill method.
class DramReclaimer : public MemoryReclaimer {
 public:
  explicit DramReclaimer(EmulatedCxlHashAggregation* op)
      : MemoryReclaimer(0), op_(op) {}

  uint64_t reclaim(
      MemoryPool* pool,
      uint64_t targetBytes,
      uint64_t maxWaitMs,
      Stats& stats) override;

 private:
  EmulatedCxlHashAggregation* const op_;
};

// Mini hash-aggregation operator emulating the CXL-aware HashAggregation
// described in spilling.rst (`Reclaim Across Memory Resources`). Allocates
// row bodies through MemoryPool to keep used-byte counters honest,
// partitions rows by key, and exposes the DRAM -> CXL spill method that
// the DRAM-leaf reclaimer invokes.
class EmulatedCxlHashAggregation {
 public:
  static constexpr int kNumPartitions = 8;

  explicit EmulatedCxlHashAggregation(core::QueryCtx* ctx)
      : ctx_(ctx),
        dramRowPool_(ctx->pool()->addLeafChild("emulated-dram-rows")),
        cxlRootPool_(ctx->customPool("cxl")),
        cxlRowPool_(cxlRootPool_->addLeafChild("emulated-cxl-rows")) {
    VELOX_CHECK_NOT_NULL(cxlRootPool_);
    // MemoryPool::setReclaimer requires the parent to have a reclaimer
    // first. The default root pool typically has none, so install a noop
    // one before attaching the DRAM-leaf reclaimer.
    if (ctx_->pool()->reclaimer() == nullptr) {
      ctx_->pool()->setReclaimer(MemoryReclaimer::create(0));
    }
    dramRowPool_->setReclaimer(std::make_unique<DramReclaimer>(this));
  }

  ~EmulatedCxlHashAggregation() {
    // Free every row still resident in DRAM or CXL before the pool
    // shared_ptrs go out of scope; otherwise the pool destructors abort
    // on outstanding usage.
    for (auto& [_, entry] : hashTable_) {
      auto* pool = entry.location == Location::kDram ? dramRowPool_.get()
                                                     : cxlRowPool_.get();
      pool->free(entry.row, sizeof(EmulatedRow));
    }
    hashTable_.clear();
  }

  EmulatedCxlHashAggregation(const EmulatedCxlHashAggregation&) = delete;
  EmulatedCxlHashAggregation& operator=(const EmulatedCxlHashAggregation&) =
      delete;

  void addInput(int64_t key, int64_t value) {
    auto it = hashTable_.find(key);
    if (it != hashTable_.end()) {
      it->second.row->sum += value;
      return;
    }
    auto* row =
        static_cast<EmulatedRow*>(dramRowPool_->allocate(sizeof(EmulatedRow)));
    row->key = key;
    row->sum = value;
    hashTable_.emplace(key, Entry{row, Location::kDram});
  }

  // Reads every row still resident in the hash table (DRAM and CXL both
  // CPU-addressable in this emulation) to produce the final per-key
  // aggregate.
  std::unordered_map<int64_t, int64_t> finalize() const {
    std::unordered_map<int64_t, int64_t> result;
    for (const auto& [_, entry] : hashTable_) {
      result[entry.row->key] += entry.row->sum;
    }
    return result;
  }

  // Returns the DRAM leaf pool — has the operator's DramReclaimer
  // installed, so the test can trigger DRAM -> CXL spill via
  // dramPool()->reclaim().
  MemoryPool* dramPool() const {
    return dramRowPool_.get();
  }

  size_t hashTableSize() const {
    return hashTable_.size();
  }

  size_t dramRowCount() const {
    return countByLocation(Location::kDram);
  }

  size_t cxlRowCount() const {
    return countByLocation(Location::kCxl);
  }

  // Moves the partition with the most DRAM-resident rows into CXL. The
  // hash-table bucket for each row is swizzled to its new CXL address;
  // the entry is not removed.
  uint64_t spillTopPartitionToCxl(uint64_t /*targetBytes*/) {
    const int partition = pickPartition(Location::kDram);
    if (partition < 0) {
      return 0;
    }
    uint64_t freed = 0;
    for (auto& [key, entry] : hashTable_) {
      if (entry.location != Location::kDram || partitionOf(key) != partition) {
        continue;
      }
      auto* cxlRow =
          static_cast<EmulatedRow*>(cxlRowPool_->allocate(sizeof(EmulatedRow)));
      *cxlRow = *entry.row;
      dramRowPool_->free(entry.row, sizeof(EmulatedRow));
      entry.row = cxlRow;
      entry.location = Location::kCxl;
      freed += sizeof(EmulatedRow);
    }
    return freed;
  }

 private:
  enum class Location { kDram, kCxl };
  struct Entry {
    EmulatedRow* row;
    Location location;
  };

  static int partitionOf(int64_t key) {
    return static_cast<int>(static_cast<uint64_t>(key) % kNumPartitions);
  }

  size_t countByLocation(Location target) const {
    size_t count = 0;
    for (const auto& [_, entry] : hashTable_) {
      if (entry.location == target) {
        ++count;
      }
    }
    return count;
  }

  // Picks the partition with the most rows of the given location, or
  // -1 if no rows match.
  int pickPartition(Location target) const {
    std::array<int, kNumPartitions> counts{};
    for (const auto& [key, entry] : hashTable_) {
      if (entry.location == target) {
        ++counts[partitionOf(key)];
      }
    }
    int best = -1;
    int bestCount = 0;
    for (int i = 0; i < kNumPartitions; ++i) {
      if (counts[i] > bestCount) {
        best = i;
        bestCount = counts[i];
      }
    }
    return best;
  }

  core::QueryCtx* const ctx_;
  std::shared_ptr<MemoryPool> dramRowPool_;
  // Root of the CXL custom resource. Reserved for child-pool creation;
  // allocations go through 'cxlRowPool_' below.
  std::shared_ptr<MemoryPool> cxlRootPool_;
  // Leaf child of 'cxlRootPool_'. Row bodies are allocated and freed
  // here.
  std::shared_ptr<MemoryPool> cxlRowPool_;
  std::unordered_map<int64_t, Entry> hashTable_;
};

uint64_t DramReclaimer::reclaim(
    MemoryPool* /*pool*/,
    uint64_t targetBytes,
    uint64_t /*maxWaitMs*/,
    Stats& /*stats*/) {
  return op_->spillTopPartitionToCxl(targetBytes);
}

std::shared_ptr<CustomMemoryResource> makeCxlResource() {
  MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = 1L << 30;
  return std::make_shared<CustomMemoryResource>(
      "cxl",
      std::make_shared<MallocAllocator>(allocatorOptions),
      MemoryArbitrator::create({}),
      []() { return MemoryReclaimer::create(0); });
}

// Materializes the CXL custom pool through the registered resource and
// attaches it to a fresh QueryCtx keyed by 'tag'.
std::shared_ptr<core::QueryCtx> buildQueryCtxWithCxl(
    std::shared_ptr<CustomMemoryResourceRegistry::Registry> registry,
    std::shared_ptr<CustomMemoryResource> resource,
    const std::string& queryId) {
  auto* manager = memoryManager();
  registry->insert("cxl", resource);
  auto registered = registry->find("cxl");
  VELOX_CHECK_NOT_NULL(registered);
  auto pool =
      manager->addCustomRootPool(fmt::format("{}.cxl", queryId), registered);
  return core::QueryCtx::Builder()
      .customPool("cxl", std::move(pool))
      .queryId(queryId)
      .build();
}

} // namespace

// End-to-end coverage for the CXL-backed HashAggregation flow documented in
// spilling.rst (`Reclaim Across Memory Resources` -> `Example: CXL-Backed
// Hash Aggregation`), restricted to the DRAM -> CXL hop. The CXL custom
// memory resource is backed by MallocAllocator so the test runs on
// hardware without real CXL devices.
class CustomMemoryEmulationTest : public testing::Test {
 protected:
  void SetUp() override {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
    registry_ = CustomMemoryResourceRegistry::createRegistry(nullptr);
  }

  std::shared_ptr<CustomMemoryResourceRegistry::Registry> registry_;
};

TEST_F(CustomMemoryEmulationTest, baselineAggregationWithoutSpill) {
  auto queryCtx =
      buildQueryCtxWithCxl(registry_, makeCxlResource(), "cxl-baseline");
  EmulatedCxlHashAggregation op(queryCtx.get());

  std::unordered_map<int64_t, int64_t> expected;
  for (int i = 0; i < 64; ++i) {
    const int64_t key = i % 16;
    const int64_t value = i;
    op.addInput(key, value);
    expected[key] += value;
  }

  EXPECT_EQ(op.hashTableSize(), expected.size());
  EXPECT_EQ(op.dramRowCount(), expected.size());
  EXPECT_EQ(op.cxlRowCount(), 0);
  EXPECT_EQ(op.finalize(), expected);
}

TEST_F(CustomMemoryEmulationTest, dramToCxlPreservesIntegrity) {
  auto queryCtx =
      buildQueryCtxWithCxl(registry_, makeCxlResource(), "cxl-chain");
  EmulatedCxlHashAggregation op(queryCtx.get());

  std::unordered_map<int64_t, int64_t> expected;
  for (int i = 0; i < 64; ++i) {
    const int64_t key = i % 16;
    const int64_t value = i;
    op.addInput(key, value);
    expected[key] += value;
  }
  const size_t totalKeys = expected.size();
  ASSERT_EQ(op.dramRowCount(), totalKeys);
  ASSERT_EQ(op.cxlRowCount(), 0);

  MemoryReclaimer::Stats stats;

  // DRAM -> CXL. The operator's DramReclaimer moves the top partition's
  // rows into the CXL pool and swizzles the hash-table bucket pointers to
  // the new addresses. The entries themselves remain; probe and finalize
  // logic continue to read them directly.
  const uint64_t freed1 = op.dramPool()->reclaim(
      /*targetBytes=*/64, /*maxWaitMs=*/0, stats);
  EXPECT_GT(freed1, 0);
  const size_t cxlAfterFirst = op.cxlRowCount();
  EXPECT_GT(cxlAfterFirst, 0);
  EXPECT_LT(op.dramRowCount(), totalKeys);
  EXPECT_EQ(op.hashTableSize(), totalKeys)
      << "DRAM -> CXL swizzles bucket pointers; size must not change.";

  // Second reclaim moves another partition to CXL.
  const uint64_t freed2 = op.dramPool()->reclaim(
      /*targetBytes=*/64, /*maxWaitMs=*/0, stats);
  EXPECT_GT(freed2, 0);
  EXPECT_GT(op.cxlRowCount(), cxlAfterFirst);
  EXPECT_EQ(op.hashTableSize(), totalKeys);

  // After two DRAM -> CXL hops, the CXL-resident rows are still directly
  // readable: finalize() returns the same per-key sums it would in the
  // baseline run.
  EXPECT_EQ(op.finalize(), expected);
}

} // namespace facebook::velox::memory::test
