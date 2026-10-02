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

  std::unique_ptr<MemoryManager> makeManager(
      int64_t capacity,
      double priorityWeightPct) {
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
        {std::string(ExtraConfig::kAbortPriorityWeightPct),
         folly::to<std::string>(priorityWeightPct)}};
    return std::make_unique<MemoryManager>(options);
  }

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
  auto manager = makeManager(capacity, 0.0625);
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

  EXPECT_TRUE(hugeQuery->pool()->aborted());
  EXPECT_FALSE(tinyQuery->pool()->aborted());

  hugeLeaf->free(hugeBuf, 448 << 20);
  tinyLeaf->free(tinyBuf, 64 << 20);
}

TEST_F(MemoryArbitrationAbortScoringTest, highWeightAbortsLowestPriority) {
  const int64_t capacity = 512 << 20;
  auto manager = makeManager(capacity, 1.0);
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

TEST_F(MemoryArbitrationAbortScoringTest, largeWeightDegeneratesToPriority) {
  const int64_t capacity = 512 << 20;
  auto manager = makeManager(capacity, 1.0);
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

TEST_F(MemoryArbitrationAbortScoringTest, equalPriorityAbortsLargest) {
  const int64_t capacity = 512 << 20;
  auto manager = makeManager(capacity, 0.0625);
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

TEST_F(MemoryArbitrationAbortScoringTest, multiQueryScoreOrder) {
  const int64_t capacity = 512 << 20;
  auto manager = makeManager(capacity, 0.0625);
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

  arbitrator->shrinkCapacity(32 << 20, /*allowSpill=*/false, /*force=*/true);
  EXPECT_TRUE(midPrioHuge->pool()->aborted());
  EXPECT_FALSE(lowPrioMid->pool()->aborted());
  EXPECT_FALSE(highPrioTiny->pool()->aborted());

  arbitrator->shrinkCapacity(32 << 20, /*allowSpill=*/false, /*force=*/true);
  EXPECT_TRUE(lowPrioMid->pool()->aborted());
  EXPECT_FALSE(highPrioTiny->pool()->aborted());

  hugeLeaf->free(hugeBuf, 300 << 20);
  midLeaf->free(midBuf, 160 << 20);
  tinyLeaf->free(tinyBuf, 40 << 20);
}

TEST_F(MemoryArbitrationAbortScoringTest, equalScorePrefersAbortingYounger) {
  const int64_t capacity = 512 << 20;
  auto manager = makeManager(capacity, 0.0625);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

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

TEST_F(MemoryArbitrationAbortScoringTest, smallPriorityGapNotAbsoluteShield) {
  const int64_t capacity = 1LL << 30;

  auto run = [&](double weightPct) {
    auto manager = makeManager(capacity, weightPct);
    auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

    auto bigEtl =
        makeQueryCtx(manager.get(), capacity, /*priority=*/1, "big_etl");
    auto bigLeaf = bigEtl->pool()->addLeafChild("big_etl.leaf");
    void* bigBuf = bigLeaf->allocate(600 << 20);

    auto q1 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q1");
    auto q1Leaf = q1->pool()->addLeafChild("q1.leaf");
    void* q1Buf = q1Leaf->allocate(100 << 20);

    auto q2 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q2");
    auto q2Leaf = q2->pool()->addLeafChild("q2.leaf");
    void* q2Buf = q2Leaf->allocate(80 << 20);

    auto q3 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q3");
    auto q3Leaf = q3->pool()->addLeafChild("q3.leaf");
    void* q3Buf = q3Leaf->allocate(70 << 20);

    auto q4 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q4");
    auto q4Leaf = q4->pool()->addLeafChild("q4.leaf");
    void* q4Buf = q4Leaf->allocate(50 << 20);

    auto q5 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q5");
    auto q5Leaf = q5->pool()->addLeafChild("q5.leaf");
    void* q5Buf = q5Leaf->allocate(40 << 20);

    arbitrator->shrinkCapacity(300 << 20, /*allowSpill=*/false, /*force=*/true);

    bool bigAborted = bigEtl->pool()->aborted();
    bool allSmallSurvived = !q1->pool()->aborted() && !q2->pool()->aborted() &&
        !q3->pool()->aborted() && !q4->pool()->aborted() &&
        !q5->pool()->aborted();

    bigLeaf->free(bigBuf, 600 << 20);
    q1Leaf->free(q1Buf, 100 << 20);
    q2Leaf->free(q2Buf, 80 << 20);
    q3Leaf->free(q3Buf, 70 << 20);
    q4Leaf->free(q4Buf, 50 << 20);
    q5Leaf->free(q5Buf, 40 << 20);

    return std::make_pair(bigAborted, allSmallSurvived);
  };

  auto [highWeightBigAborted, highWeightSmallSurvived] = run(1.0);
  EXPECT_FALSE(highWeightBigAborted);
  EXPECT_FALSE(highWeightSmallSurvived);

  auto [lowWeightBigAborted, lowWeightSmallSurvived] = run(0.0625);
  EXPECT_TRUE(lowWeightBigAborted);
  EXPECT_TRUE(lowWeightSmallSurvived);
}

TEST_F(MemoryArbitrationAbortScoringTest, wastedAbortsCascade) {
  const int64_t capacity = 1LL << 30;

  auto runWithWeight = [&](double weightPct) {
    auto manager = makeManager(capacity, weightPct);
    auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

    struct QState {
      std::shared_ptr<core::QueryCtx> ctx;
      std::shared_ptr<MemoryPool> leaf;
      void* buf;
      int64_t size;
    };
    std::vector<QState> queries;

    auto addQuery = [&](const std::string& id, int32_t priority, int64_t mb) {
      auto ctx = makeQueryCtx(manager.get(), capacity, priority, id);
      auto leaf = ctx->pool()->addLeafChild(id + ".leaf");
      void* buf = leaf->allocate(mb << 20);
      queries.push_back({ctx, leaf, buf, mb << 20});
    };

    addQuery("big_etl", 1, 600);
    addQuery("q1", 2, 100);
    addQuery("q2", 2, 80);
    addQuery("q3", 2, 70);
    addQuery("q4", 2, 50);
    addQuery("q5", 2, 40);

    auto statsBefore = arbitrator->stats();

    uint64_t totalReclaimed = 0;
    const uint64_t target = 300UL << 20;
    while (totalReclaimed < target) {
      uint64_t reclaimed =
          arbitrator->shrinkCapacity(target - totalReclaimed, false, true);
      if (reclaimed == 0) {
        break;
      }
      totalReclaimed += reclaimed;
    }

    auto statsAfter = arbitrator->stats();
    uint64_t numAborted = statsAfter.numAborted - statsBefore.numAborted;

    uint64_t surviving = 0;
    for (auto& q : queries) {
      if (!q.ctx->pool()->aborted()) {
        surviving++;
      }
      q.leaf->free(q.buf, q.size);
    }

    return std::make_pair(numAborted, surviving);
  };

  auto [highWeightAborts, highWeightSurviving] = runWithWeight(1.0);
  auto [lowWeightAborts, lowWeightSurviving] = runWithWeight(0.0625);

  EXPECT_GT(highWeightAborts, lowWeightAborts);
  EXPECT_LT(highWeightSurviving, lowWeightSurviving);

  EXPECT_EQ(lowWeightAborts, 1);
  EXPECT_EQ(lowWeightSurviving, 5);
}

TEST_F(
    MemoryArbitrationAbortScoringTest,
    highWeightPreservesPriorityDominance) {
  const int64_t capacity = 1LL << 30;
  auto manager = makeManager(capacity, 1.0);
  auto* arbitrator = static_cast<SharedArbitrator*>(manager->arbitrator());

  auto bigEtl =
      makeQueryCtx(manager.get(), capacity, /*priority=*/1, "big_etl");
  auto bigLeaf = bigEtl->pool()->addLeafChild("big_etl.leaf");
  void* bigBuf = bigLeaf->allocate(600 << 20);

  auto q1 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q1");
  auto q1Leaf = q1->pool()->addLeafChild("q1.leaf");
  void* q1Buf = q1Leaf->allocate(100 << 20);

  auto q2 = makeQueryCtx(manager.get(), capacity, /*priority=*/2, "q2");
  auto q2Leaf = q2->pool()->addLeafChild("q2.leaf");
  void* q2Buf = q2Leaf->allocate(80 << 20);

  arbitrator->shrinkCapacity(64 << 20, /*allowSpill=*/false, /*force=*/true);

  EXPECT_FALSE(bigEtl->pool()->aborted());

  bigLeaf->free(bigBuf, 600 << 20);
  q1Leaf->free(q1Buf, 100 << 20);
  q2Leaf->free(q2Buf, 80 << 20);
}

} // namespace
} // namespace facebook::velox::memory
