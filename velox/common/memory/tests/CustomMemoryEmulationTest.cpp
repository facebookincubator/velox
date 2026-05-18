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

#include <array>
#include <functional>
#include <gtest/gtest.h>

#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::memory::test {
namespace {

// Mirrors the row layout the docs assume for the CXL-backed HashAggregation
// example: a fixed-width group key plus a single int64_t accumulator.
struct EmulatedRow {
  int64_t key;
  int64_t sum;
};

class EmulatedCxlHashAggregation;

// Reclaimer installed on the operator's DRAM leaf row pool. Forwards
// reclaim() to the operator's DRAM -> CXL spill method, modeling Phase 1
// in the spilling docs.
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

// Reclaimer attached to the CXL custom root pool. Constructed by the
// resource's reclaimerFactory before the operator exists; the operator
// fills in 'callback' after construction and clears it on destruction.
class CxlReclaimer : public MemoryReclaimer {
 public:
  using Callback = std::function<uint64_t(uint64_t)>;

  explicit CxlReclaimer(std::shared_ptr<Callback> callback)
      : MemoryReclaimer(0), callback_(std::move(callback)) {}

  uint64_t reclaim(
      MemoryPool* /*pool*/,
      uint64_t targetBytes,
      uint64_t /*maxWaitMs*/,
      Stats& /*stats*/) override {
    if (!callback_ || !*callback_) {
      return 0;
    }
    return (*callback_)(targetBytes);
  }

 private:
  const std::shared_ptr<Callback> callback_;
};

// Mini hash-aggregation operator emulating the CXL-aware HashAggregation
// described in spilling.rst (`Reclaim Across Memory Resources`). Allocates
// row bodies through MemoryPool to keep used-byte counters honest,
// partitions rows by key, and exposes spill methods that the two
// reclaimers invoke.
class EmulatedCxlHashAggregation {
 public:
  static constexpr int kNumPartitions = 8;

  EmulatedCxlHashAggregation(
      core::QueryCtx* ctx,
      std::shared_ptr<CxlReclaimer::Callback> cxlCallbackSlot)
      : ctx_(ctx),
        dramRowPool_(ctx->pool()->addLeafChild("emulated-dram-rows")),
        cxlRootPool_(ctx->customPool("cxl")),
        cxlRowPool_(cxlRootPool_->addLeafChild("emulated-cxl-rows")),
        cxlCallbackSlot_(std::move(cxlCallbackSlot)) {
    VELOX_CHECK_NOT_NULL(cxlRootPool_);
    // MemoryPool::setReclaimer requires the parent to have a reclaimer
    // first. The default root pool typically has none, so install a noop
    // one before attaching the DRAM-leaf reclaimer.
    if (ctx_->pool()->reclaimer() == nullptr) {
      ctx_->pool()->setReclaimer(MemoryReclaimer::create(0));
    }
    dramRowPool_->setReclaimer(std::make_unique<DramReclaimer>(this));
    *cxlCallbackSlot_ = [this](uint64_t targetBytes) {
      return spillCxlPartitionToDisk(targetBytes);
    };
  }

  ~EmulatedCxlHashAggregation() {
    // Clear the CXL reclaimer's callback so a stale reclaim invocation
    // after operator destruction cannot dereference 'this'.
    if (cxlCallbackSlot_) {
      *cxlCallbackSlot_ = {};
    }
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
    auto* row = static_cast<EmulatedRow*>(
        dramRowPool_->allocate(sizeof(EmulatedRow)));
    row->key = key;
    row->sum = value;
    hashTable_.emplace(key, Entry{row, Location::kDram});
  }

  // Reads every row still resident in the hash table (DRAM and CXL both
  // CPU-addressable in this emulation) and merges with disk-resident
  // rows to produce the final per-key aggregate.
  std::unordered_map<int64_t, int64_t> finalize() const {
    std::unordered_map<int64_t, int64_t> result;
    for (const auto& [_, entry] : hashTable_) {
      result[entry.row->key] += entry.row->sum;
    }
    for (const auto& row : diskRows_) {
      result[row.key] += row.sum;
    }
    return result;
  }

  // Returns the DRAM leaf pool — has the operator's DramReclaimer
  // installed, so the test can trigger Phase 1 via dramPool()->reclaim().
  MemoryPool* dramPool() const {
    return dramRowPool_.get();
  }

  // Returns the CXL root pool — has the resource's CxlReclaimer installed
  // via reclaimerFactory, so the test can trigger Phase 2 via
  // cxlPool()->reclaim().
  MemoryPool* cxlPool() const {
    return cxlRootPool_.get();
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

  size_t diskRowCount() const {
    return diskRows_.size();
  }

  // Phase 1 (DRAM -> CXL): move the partition with the most
  // DRAM-resident rows into CXL. The hash-table bucket for each row is
  // swizzled to its new CXL address; the entry is not removed.
  uint64_t spillTopPartitionToCxl(uint64_t /*targetBytes*/) {
    const int partition = pickPartition(Location::kDram);
    if (partition < 0) {
      return 0;
    }
    uint64_t freed = 0;
    for (auto& [key, entry] : hashTable_) {
      if (entry.location != Location::kDram ||
          partitionOf(key) != partition) {
        continue;
      }
      auto* cxlRow = static_cast<EmulatedRow*>(
          cxlRowPool_->allocate(sizeof(EmulatedRow)));
      *cxlRow = *entry.row;
      dramRowPool_->free(entry.row, sizeof(EmulatedRow));
      entry.row = cxlRow;
      entry.location = Location::kCxl;
      freed += sizeof(EmulatedRow);
    }
    return freed;
  }

  // Phase 2 (CXL -> disk): move the partition with the most CXL-resident
  // rows into the disk-resident vector. The hash-table entries are
  // erased, because the rows are no longer directly addressable from the
  // CPU after the on-disk copy is the only canonical copy.
  uint64_t spillCxlPartitionToDisk(uint64_t /*targetBytes*/) {
    const int partition = pickPartition(Location::kCxl);
    if (partition < 0) {
      return 0;
    }
    uint64_t freed = 0;
    for (auto it = hashTable_.begin(); it != hashTable_.end();) {
      if (it->second.location != Location::kCxl ||
          partitionOf(it->first) != partition) {
        ++it;
        continue;
      }
      diskRows_.push_back(*it->second.row);
      cxlRowPool_->free(it->second.row, sizeof(EmulatedRow));
      it = hashTable_.erase(it);
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
  // Root of the CXL custom resource; holds the CxlReclaimer installed by
  // the resource's reclaimerFactory. Reserved for reclaim() dispatch only;
  // allocations go through 'cxlRowPool_' below.
  std::shared_ptr<MemoryPool> cxlRootPool_;
  // Leaf child of 'cxlRootPool_'. Row bodies are allocated and freed
  // here.
  std::shared_ptr<MemoryPool> cxlRowPool_;
  std::shared_ptr<CxlReclaimer::Callback> cxlCallbackSlot_;
  std::unordered_map<int64_t, Entry> hashTable_;
  std::vector<EmulatedRow> diskRows_;
};

uint64_t DramReclaimer::reclaim(
    MemoryPool* /*pool*/,
    uint64_t targetBytes,
    uint64_t /*maxWaitMs*/,
    Stats& /*stats*/) {
  return op_->spillTopPartitionToCxl(targetBytes);
}

// Bundles a CustomMemoryResource backed by MallocAllocator with the
// callback slot the operator fills in after QueryCtx construction.
struct CxlResourceBundle {
  CustomMemoryResource resource;
  std::shared_ptr<CxlReclaimer::Callback> callbackSlot;
};

CxlResourceBundle makeCxlResource() {
  MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = 1L << 30;
  CxlResourceBundle bundle;
  bundle.callbackSlot = std::make_shared<CxlReclaimer::Callback>();
  bundle.resource.tag = "cxl";
  bundle.resource.maxCapacity = 1L << 30;
  bundle.resource.allocator =
      std::make_shared<MallocAllocator>(allocatorOptions);
  bundle.resource.arbitrator = MemoryArbitrator::create({});
  auto slot = bundle.callbackSlot;
  bundle.resource.reclaimerFactory = [slot]() {
    return std::make_unique<CxlReclaimer>(slot);
  };
  return bundle;
}

// Materializes the CXL custom pool through the registered resource and
// attaches it to a fresh QueryCtx keyed by 'tag'. Mirrors the recommended
// caller-builds-pool flow for custom memory resources.
std::shared_ptr<core::QueryCtx> buildQueryCtxWithCxl(
    CxlResourceBundle& bundle,
    const std::string& queryId) {
  auto* manager = memoryManager();
  manager->registerCustomResource(std::move(bundle.resource));
  const auto& registered = *manager->customResources().at("cxl");
  auto reclaimer = registered.reclaimerFactory();
  auto pool = manager->addRootPool(
      fmt::format("{}.cxl", queryId),
      registered.maxCapacity,
      std::move(reclaimer),
      std::nullopt,
      "cxl");
  return core::QueryCtx::Builder()
      .customPool("cxl", std::move(pool))
      .queryId(queryId)
      .build();
}

} // namespace

// End-to-end coverage for the CXL-backed HashAggregation flow documented in
// spilling.rst (`Reclaim Across Memory Resources` -> `Example: CXL-Backed
// Hash Aggregation`). The CXL custom memory resource is backed by
// MallocAllocator so the test runs on hardware without real CXL devices.
class CustomMemoryEmulationTest : public testing::Test {
 protected:
  void SetUp() override {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }
};

TEST_F(CustomMemoryEmulationTest, baselineAggregationWithoutSpill) {
  auto cxl = makeCxlResource();
  auto queryCtx = buildQueryCtxWithCxl(cxl, "cxl-baseline");
  EmulatedCxlHashAggregation op(queryCtx.get(), cxl.callbackSlot);

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
  EXPECT_EQ(op.diskRowCount(), 0);
  EXPECT_EQ(op.finalize(), expected);
}

TEST_F(CustomMemoryEmulationTest, dramToCxlToDiskChainPreservesIntegrity) {
  auto cxl = makeCxlResource();
  auto queryCtx = buildQueryCtxWithCxl(cxl, "cxl-chain");
  EmulatedCxlHashAggregation op(queryCtx.get(), cxl.callbackSlot);

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
  ASSERT_EQ(op.diskRowCount(), 0);

  MemoryReclaimer::Stats stats;

  // Phase 1: DRAM -> CXL. The operator's DramReclaimer moves the top
  // partition's rows into the CXL pool and swizzles the hash-table
  // bucket pointers to the new addresses. The entries themselves remain;
  // probe and finalize logic continue to read them directly.
  const uint64_t freed1 = op.dramPool()->reclaim(
      /*targetBytes=*/64, /*maxWaitMs=*/0, stats);
  EXPECT_GT(freed1, 0);
  const size_t cxlAfterFirst = op.cxlRowCount();
  EXPECT_GT(cxlAfterFirst, 0);
  EXPECT_LT(op.dramRowCount(), totalKeys);
  EXPECT_EQ(op.diskRowCount(), 0);
  EXPECT_EQ(op.hashTableSize(), totalKeys)
      << "Phase 1 swizzles bucket pointers; size must not change.";

  // Second DRAM reclaim moves another partition to CXL.
  const uint64_t freed2 = op.dramPool()->reclaim(
      /*targetBytes=*/64, /*maxWaitMs=*/0, stats);
  EXPECT_GT(freed2, 0);
  EXPECT_GT(op.cxlRowCount(), cxlAfterFirst);
  EXPECT_EQ(op.diskRowCount(), 0);
  EXPECT_EQ(op.hashTableSize(), totalKeys);

  // After two DRAM -> CXL hops, the CXL-resident rows are still directly
  // readable: finalize() returns the same per-key sums it would in the
  // baseline run.
  EXPECT_EQ(op.finalize(), expected);

  // Phase 2: CXL -> disk. Arbitration on the CXL custom pool triggers the
  // CxlReclaimer; the operator copies the top CXL partition's rows into
  // the disk-resident vector and erases their hash-table entries.
  const size_t cxlBeforePhase2 = op.cxlRowCount();
  const size_t hashSizeBeforePhase2 = op.hashTableSize();
  const uint64_t freed3 = op.cxlPool()->reclaim(
      /*targetBytes=*/64, /*maxWaitMs=*/0, stats);
  EXPECT_GT(freed3, 0);
  EXPECT_GT(op.diskRowCount(), 0);
  EXPECT_LT(op.cxlRowCount(), cxlBeforePhase2);
  EXPECT_LT(op.hashTableSize(), hashSizeBeforePhase2)
      << "Phase 2 erases hash-table entries for spilled keys.";

  // Phase 3: finalize merges DRAM-resident + CXL-resident + disk rows
  // and yields the same per-key sums the baseline (no-spill) run would.
  EXPECT_EQ(op.finalize(), expected);
}

} // namespace facebook::velox::memory::test
