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

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/core/QueryCtx.h"

// End-to-end check that the badness-score abort victim selection
// ('memory-pool-abort-scoring') is wired through a real QueryCtx ->
// MemoryReclaimer -> SharedArbitrator path, not just the mock unit tests.
// Unlike the mock tests which build pools with a hand-rolled reclaimer, this
// drives the priority through QueryConfig (query_memory_reclaimer_priority) the
// same way a production query does.
namespace facebook::velox::memory {
namespace {

class MemoryArbitrationAbortScoringTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    SharedArbitrator::registerFactory();
  }

  static void TearDownTestCase() {
    SharedArbitrator::unregisterFactory();
  }

  // Builds a MemoryManager whose SharedArbitrator selects abort victims by
  // badness score with the given priority weight.
  std::unique_ptr<MemoryManager>
  makeManager(int64_t capacity, bool scoring, uint64_t priorityWeightBytes) {
    MemoryManager::Options options;
    options.allocatorCapacity = capacity;
    options.arbitratorCapacity = capacity;
    options.arbitratorKind = "SHARED";
    options.checkUsageLeak = true;
    using ExtraConfig = SharedArbitrator::ExtraConfig;
    options.extraArbitratorConfigs = {
        {std::string(ExtraConfig::kMemoryPoolInitialCapacity), "0B"},
        {std::string(ExtraConfig::kMemoryPoolReservedCapacity), "0B"},
        {std::string(ExtraConfig::kReservedCapacity), "0B"},
        {std::string(ExtraConfig::kGlobalArbitrationEnabled), "true"},
        {std::string(ExtraConfig::kGlobalArbitrationWithoutSpill), "true"},
        {std::string(ExtraConfig::kMemoryPoolAbortCapacityLimit),
         folly::to<std::string>(capacity) + "B"},
        {std::string(ExtraConfig::kMemoryPoolAbortScoring),
         scoring ? "true" : "false"},
        {std::string(ExtraConfig::kMemoryPoolAbortScoringPriorityWeight),
         folly::to<std::string>(priorityWeightBytes) + "B"}};
    return std::make_unique<MemoryManager>(options);
  }

  // Builds a real QueryCtx whose root pool reclaimer carries 'priority',
  // mirroring how a production query sets query_memory_reclaimer_priority.
  std::shared_ptr<core::QueryCtx> makeQueryCtx(
      MemoryManager* manager,
      int64_t capacity,
      int32_t priority,
      const std::string& id) {
    std::unordered_map<std::string, std::string> config{
        {core::QueryConfig::kQueryMemoryReclaimerPriority,
         folly::to<std::string>(priority)}};
    return core::QueryCtx::Builder()
        .executor(executor_.get())
        // Pass a reclaimer-less root pool so QueryCtx installs its own
        // priority-carrying MemoryReclaimer from the config above.
        .pool(manager->addRootPool(id, capacity))
        .queryConfig(core::QueryConfig{std::move(config)})
        .queryId(id)
        .build();
  }

  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(4)};
};

TEST_F(MemoryArbitrationAbortScoringTest, hugeHigherPriorityAbortedFirst) {
  const int64_t capacity = 512 << 20;
  // One priority step is worth only 32MB, so the ~384MB capacity gap dominates.
  auto manager =
      makeManager(capacity, /*scoring=*/true, /*priorityWeightBytes=*/32 << 20);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  // hugeQuery: slightly higher priority (smaller number) but holds most memory.
  auto hugeQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "huge");
  auto hugeLeaf = hugeQuery->pool()->addLeafChild("huge.leaf");
  void* hugeBuf = hugeLeaf->allocate(448 << 20);

  // tinyQuery: lowest priority but only a sliver of memory.
  auto tinyQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/2, "tiny");
  auto tinyLeaf = tinyQuery->pool()->addLeafChild("tiny.leaf");
  void* tinyBuf = tinyLeaf->allocate(64 << 20);

  // Force a synchronous abort to reclaim memory. Strict priority ordering would
  // abort 'tiny' (lowest priority); scoring aborts 'huge' because freeing it
  // reclaims far more memory and the priority gap is only one weighted step.
  arbitrator->shrinkCapacity(64 << 20, /*allowSpill=*/false, /*force=*/true);

  EXPECT_TRUE(hugeQuery->pool()->aborted());
  EXPECT_FALSE(tinyQuery->pool()->aborted());

  // Release the buffers; the aborted pool tolerates frees.
  hugeLeaf->free(hugeBuf, 448 << 20);
  tinyLeaf->free(tinyBuf, 64 << 20);
}

// Contrast: with scoring disabled the same setup falls back to strict
// priority-first ordering, which aborts the lowest-priority 'tiny' query even
// though that reclaims far less memory. This is the behavior the scoring mode
// is designed to improve on, and it confirms the test setup actually exercises
// the victim-selection path.
TEST_F(MemoryArbitrationAbortScoringTest, legacyOrderAbortsLowestPriority) {
  const int64_t capacity = 512 << 20;
  auto manager = makeManager(
      capacity, /*scoring=*/false, /*priorityWeightBytes=*/32 << 20);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  auto hugeQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "huge");
  auto hugeLeaf = hugeQuery->pool()->addLeafChild("huge.leaf");
  void* hugeBuf = hugeLeaf->allocate(448 << 20);

  auto tinyQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/2, "tiny");
  auto tinyLeaf = tinyQuery->pool()->addLeafChild("tiny.leaf");
  void* tinyBuf = tinyLeaf->allocate(64 << 20);

  arbitrator->shrinkCapacity(64 << 20, /*allowSpill=*/false, /*force=*/true);

  EXPECT_TRUE(tinyQuery->pool()->aborted());
  EXPECT_FALSE(hugeQuery->pool()->aborted());

  hugeLeaf->free(hugeBuf, 448 << 20);
  tinyLeaf->free(tinyBuf, 64 << 20);
}

// A large priority weight makes the priority term dominate the badness score,
// so scoring degenerates to the legacy priority-first ordering even for a huge
// capacity gap. With weight = capacity, one priority step (512MB) outweighs the
// ~384MB footprint difference, so the lowest-priority 'tiny' is aborted - the
// same victim the legacy order picks. This pins the 'weight' knob's behavior.
TEST_F(MemoryArbitrationAbortScoringTest, largeWeightDegeneratesToPriority) {
  const int64_t capacity = 512 << 20;
  auto manager =
      makeManager(capacity, /*scoring=*/true, /*priorityWeightBytes=*/capacity);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  auto hugeQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "huge");
  auto hugeLeaf = hugeQuery->pool()->addLeafChild("huge.leaf");
  void* hugeBuf = hugeLeaf->allocate(448 << 20);

  auto tinyQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/2, "tiny");
  auto tinyLeaf = tinyQuery->pool()->addLeafChild("tiny.leaf");
  void* tinyBuf = tinyLeaf->allocate(64 << 20);

  arbitrator->shrinkCapacity(64 << 20, /*allowSpill=*/false, /*force=*/true);

  EXPECT_TRUE(tinyQuery->pool()->aborted());
  EXPECT_FALSE(hugeQuery->pool()->aborted());

  hugeLeaf->free(hugeBuf, 448 << 20);
  tinyLeaf->free(tinyBuf, 64 << 20);
}

// With equal priority the badness score reduces to reclaimable capacity, so the
// larger participant is aborted first. The legacy order also breaks equal
// priority by capacity, so this case is expected to match it - it is a
// compatibility check that scoring does not regress the equal-priority path,
// not a demonstration of scoring-only behavior.
TEST_F(MemoryArbitrationAbortScoringTest, equalPriorityAbortsLargest) {
  const int64_t capacity = 512 << 20;
  auto manager =
      makeManager(capacity, /*scoring=*/true, /*priorityWeightBytes=*/32 << 20);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  auto bigQuery = makeQueryCtx(manager.get(), capacity, /*priority=*/1, "big");
  auto bigLeaf = bigQuery->pool()->addLeafChild("big.leaf");
  void* bigBuf = bigLeaf->allocate(320 << 20);

  auto smallQuery =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "small");
  auto smallLeaf = smallQuery->pool()->addLeafChild("small.leaf");
  void* smallBuf = smallLeaf->allocate(192 << 20);

  arbitrator->shrinkCapacity(64 << 20, /*allowSpill=*/false, /*force=*/true);

  EXPECT_TRUE(bigQuery->pool()->aborted());
  EXPECT_FALSE(smallQuery->pool()->aborted());

  bigLeaf->free(bigBuf, 320 << 20);
  smallLeaf->free(smallBuf, 192 << 20);
}

// Scoring ranks the whole candidate set, not just a pair. With three queries
// spanning two priorities and different footprints, repeated forced aborts
// remove them in descending badness-score order. With weight = 32MB:
//   midPrioHuge:  priority 1, 300MB -> score 300 + 1*32 = 332 (highest)
//   lowPrioMid:   priority 2, 160MB -> score 160 + 2*32 = 224
//   highPrioTiny: priority 0,  40MB -> score  40 + 0*32 =  40 (lowest)
// so the order is midPrioHuge, then lowPrioMid, then highPrioTiny - the large
// high-ish-priority query goes first because its footprint dominates, which the
// strict priority order would never do.
TEST_F(MemoryArbitrationAbortScoringTest, multiQueryScoreOrder) {
  const int64_t capacity = 512 << 20;
  auto manager =
      makeManager(capacity, /*scoring=*/true, /*priorityWeightBytes=*/32 << 20);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  auto midPrioHuge =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "midPrioHuge");
  auto hugeLeaf = midPrioHuge->pool()->addLeafChild("huge.leaf");
  void* hugeBuf = hugeLeaf->allocate(300 << 20);

  auto lowPrioMid =
      makeQueryCtx(manager.get(), capacity, /*priority=*/2, "lowPrioMid");
  auto midLeaf = lowPrioMid->pool()->addLeafChild("mid.leaf");
  void* midBuf = midLeaf->allocate(160 << 20);

  auto highPrioTiny =
      makeQueryCtx(manager.get(), capacity, /*priority=*/0, "highPrioTiny");
  auto tinyLeaf = highPrioTiny->pool()->addLeafChild("tiny.leaf");
  void* tinyBuf = tinyLeaf->allocate(40 << 20);

  // First forced abort takes the highest-scoring victim: midPrioHuge.
  arbitrator->shrinkCapacity(32 << 20, /*allowSpill=*/false, /*force=*/true);
  EXPECT_TRUE(midPrioHuge->pool()->aborted());
  EXPECT_FALSE(lowPrioMid->pool()->aborted());
  EXPECT_FALSE(highPrioTiny->pool()->aborted());

  // Second forced abort takes the next-highest: lowPrioMid.
  arbitrator->shrinkCapacity(32 << 20, /*allowSpill=*/false, /*force=*/true);
  EXPECT_TRUE(lowPrioMid->pool()->aborted());
  EXPECT_FALSE(highPrioTiny->pool()->aborted());

  hugeLeaf->free(hugeBuf, 300 << 20);
  midLeaf->free(midBuf, 160 << 20);
  tinyLeaf->free(tinyBuf, 40 << 20);
}

// When two candidates have the same badness score (equal priority and equal
// capacity), the younger one (created later, larger participant id) is aborted
// first, letting the older long-running query proceed. This preserves the
// sunk-cost tie-break the legacy abort order applies within a capacity bucket.
TEST_F(MemoryArbitrationAbortScoringTest, equalScorePrefersAbortingYounger) {
  const int64_t capacity = 512 << 20;
  auto manager =
      makeManager(capacity, /*scoring=*/true, /*priorityWeightBytes=*/32 << 20);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  // Same priority and same capacity -> identical score; only age differs.
  auto older = makeQueryCtx(manager.get(), capacity, /*priority=*/1, "older");
  auto olderLeaf = older->pool()->addLeafChild("older.leaf");
  void* olderBuf = olderLeaf->allocate(200 << 20);

  auto younger =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "younger");
  auto youngerLeaf = younger->pool()->addLeafChild("younger.leaf");
  void* youngerBuf = youngerLeaf->allocate(200 << 20);

  arbitrator->shrinkCapacity(32 << 20, /*allowSpill=*/false, /*force=*/true);

  EXPECT_TRUE(younger->pool()->aborted());
  EXPECT_FALSE(older->pool()->aborted());

  olderLeaf->free(olderBuf, 200 << 20);
  youngerLeaf->free(youngerBuf, 200 << 20);
}

} // namespace
} // namespace facebook::velox::memory
