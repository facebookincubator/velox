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
#include <thread>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/HashJoinBridge.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/HashTableCache.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/PlanNodeStats.h"

#include "velox/exec/tests/utils/ArbitratorTestUtil.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include "velox/exec/tests/utils/VectorTestUtil.h"

namespace facebook::velox::exec::test {
namespace {

// Test fixture for hash join with hash table caching tests.
class HashJoinWithCacheTest : public HiveConnectorTestBase {};

// Tests hash table caching for broadcast joins.
// First task builds the table (cache miss), second task reuses it (cache hit).
TEST_F(HashJoinWithCacheTest, sequential) {
  // Use a unique query ID for this test to ensure clean cache state.
  const std::string queryId =
      "hashTableCachingTest_" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());

  // Create probe and build vectors with distinct column names.
  std::vector<RowVectorPtr> probeVectors = makeBatches(10, [&](int32_t) {
    return makeRowVector(
        {"t_k", "t_v"},
        {
            makeFlatVector<int32_t>(100, [](auto row) { return row % 23; }),
            makeFlatVector<int64_t>(100, [](auto row) { return row; }),
        });
  });

  std::vector<RowVectorPtr> buildVectors = makeBatches(5, [&](int32_t) {
    return makeRowVector(
        {"u_k", "u_v"},
        {
            makeFlatVector<int32_t>(50, [](auto row) { return row % 31; }),
            makeFlatVector<int64_t>(50, [](auto row) { return row * 10; }),
        });
  });

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Build the plan using HashJoinNode::Builder to set useHashTableCache.
  auto buildPlanNode =
      PlanBuilder(planNodeIdGenerator).values(buildVectors).planNode();

  auto probePlanNode =
      PlanBuilder(planNodeIdGenerator).values(probeVectors).planNode();

  // Create HashJoinNode with useHashTableCache = true.
  auto outputType = ROW(
      {"t_k", "t_v", "u_k", "u_v"}, {INTEGER(), BIGINT(), INTEGER(), BIGINT()});

  auto joinNode =
      core::HashJoinNode::Builder()
          .id(planNodeIdGenerator->next())
          .joinType(core::JoinType::kInner)
          .nullAware(false)
          .leftKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "t_k")})
          .rightKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "u_k")})
          .left(probePlanNode)
          .right(buildPlanNode)
          .outputType(outputType)
          .useHashTableCache(true)
          .build();
  const auto joinNodeId = joinNode->id();

  const int numDrivers = 3;

  // Create a shared QueryCtx for all tasks.
  auto queryCtx = core::QueryCtx::create(
      driverExecutor_.get(),
      core::QueryConfig({}),
      std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
      cache::AsyncDataCache::getInstance(),
      nullptr,
      nullptr,
      queryId);

  // Helper to run a task and return the completed task.
  // Both tasks use the same queryCtx so they share the cache entry.
  auto runTask = [&]() {
    return AssertQueryBuilder(joinNode, duckDbQueryRunner_)
        .queryCtx(queryCtx)
        .maxDrivers(numDrivers)
        .assertResults(
            "SELECT t.t_k, t.t_v, u.u_k, u.u_v FROM t, u WHERE t.t_k = u.u_k");
  };

  // First task - should build the table (cache miss).
  auto task1 = runTask();

  // Get stats from first task - expect cache miss.
  auto opStats1 = toOperatorStats(task1->taskStats());
  ASSERT_EQ(opStats1.count("HashBuild"), 1);
  auto& hashBuildStats1 = opStats1.at("HashBuild");

  // The last driver that finishes building reports the cache miss stat.
  ASSERT_EQ(
      hashBuildStats1.runtimeStats.count(
          std::string(BaseHashTable::kHashTableCacheMiss)),
      1)
      << "First task should report cache miss";
  EXPECT_EQ(
      hashBuildStats1.runtimeStats
          .at(std::string(BaseHashTable::kHashTableCacheMiss))
          .count,
      1)
      << "Exactly one driver should report cache miss (the one that builds)";
  EXPECT_EQ(
      hashBuildStats1.runtimeStats.count(
          std::string(BaseHashTable::kHashTableCacheHit)),
      0)
      << "First task should not have any cache hits";

  // Second task - should reuse the cached table (cache hit).
  auto task2 = runTask();

  // Get stats from second task - expect cache hit.
  auto opStats2 = toOperatorStats(task2->taskStats());
  ASSERT_EQ(opStats2.count("HashBuild"), 1);
  auto& hashBuildStats2 = opStats2.at("HashBuild");

  // The last driver that finishes reports the cache hit stat.
  ASSERT_EQ(
      hashBuildStats2.runtimeStats.count(
          std::string(BaseHashTable::kHashTableCacheHit)),
      1)
      << "Second task should report cache hit";
  EXPECT_EQ(
      hashBuildStats2.runtimeStats
          .at(std::string(BaseHashTable::kHashTableCacheHit))
          .count,
      1)
      << "Exactly one driver should report cache hit (the one after barrier)";
  EXPECT_EQ(
      hashBuildStats2.runtimeStats.count(
          std::string(BaseHashTable::kHashTableCacheMiss)),
      0)
      << "Second task should not have any cache misses";

  // Clean up cache entry before tasks are destroyed.
  // The release callback on QueryCtx fires too late (after Task destruction
  // starts), so we need explicit cleanup in tests.
  const auto cacheKey = fmt::format("{}:{}", queryId, joinNodeId);
  HashTableCache::instance()->drop(cacheKey);
}

// Tests that multiple tasks running concurrently share the cached hash table.
// One task builds the table (cache miss), all others wait and reuse it
// (cache hits).
TEST_F(HashJoinWithCacheTest, concurrent) {
  // Use a unique query ID for this test to ensure clean cache state.
  const std::string queryId =
      "hashTableCachingConcurrentTest_" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());

  // Create probe and build vectors with distinct column names.
  std::vector<RowVectorPtr> probeVectors = makeBatches(10, [&](int32_t) {
    return makeRowVector(
        {"t_k", "t_v"},
        {
            makeFlatVector<int32_t>(100, [](auto row) { return row % 23; }),
            makeFlatVector<int64_t>(100, [](auto row) { return row; }),
        });
  });

  std::vector<RowVectorPtr> buildVectors = makeBatches(5, [&](int32_t) {
    return makeRowVector(
        {"u_k", "u_v"},
        {
            makeFlatVector<int32_t>(50, [](auto row) { return row % 31; }),
            makeFlatVector<int64_t>(50, [](auto row) { return row * 10; }),
        });
  });

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Build the plan using HashJoinNode::Builder to set useHashTableCache.
  auto buildPlanNode =
      PlanBuilder(planNodeIdGenerator).values(buildVectors).planNode();

  auto probePlanNode =
      PlanBuilder(planNodeIdGenerator).values(probeVectors).planNode();

  // Create HashJoinNode with useHashTableCache = true.
  auto outputType = ROW(
      {"t_k", "t_v", "u_k", "u_v"}, {INTEGER(), BIGINT(), INTEGER(), BIGINT()});

  auto joinNode =
      core::HashJoinNode::Builder()
          .id(planNodeIdGenerator->next())
          .joinType(core::JoinType::kInner)
          .nullAware(false)
          .leftKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "t_k")})
          .rightKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "u_k")})
          .left(probePlanNode)
          .right(buildPlanNode)
          .outputType(outputType)
          .useHashTableCache(true)
          .build();
  const auto joinNodeId = joinNode->id();

  const int numDrivers = 3;

  // Create a shared QueryCtx for all tasks.
  auto queryCtx = core::QueryCtx::create(
      driverExecutor_.get(),
      core::QueryConfig({}),
      std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
      cache::AsyncDataCache::getInstance(),
      nullptr,
      nullptr,
      queryId);

  // Helper to run a task and return the completed task.
  // All tasks use the same queryCtx so they share the cache entry.
  auto runTask = [&]() {
    return AssertQueryBuilder(joinNode, duckDbQueryRunner_)
        .queryCtx(queryCtx)
        .maxDrivers(numDrivers)
        .assertResults(
            "SELECT t.t_k, t.t_v, u.u_k, u.u_v FROM t, u WHERE t.t_k = u.u_k");
  };

  // Run 10 threads concurrently, each executing 5 tasks sequentially.
  constexpr int kNumThreads = 10;
  constexpr int kTasksPerThread = 5;
  constexpr int kTotalTasks = kNumThreads * kTasksPerThread;

  // Each thread maintains its own local vector of tasks.
  std::vector<std::vector<std::shared_ptr<Task>>> threadTasks(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  // Launch all threads at once.
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      // Each thread runs multiple tasks sequentially into its local vector.
      threadTasks[i].reserve(kTasksPerThread);
      for (int j = 0; j < kTasksPerThread; ++j) {
        threadTasks[i].push_back(runTask());
      }
    });
  }

  // Wait for all threads to complete.
  for (auto& thread : threads) {
    thread.join();
  }

  // Merge all thread-local task vectors into a single vector.
  std::vector<std::shared_ptr<Task>> allTasks;
  allTasks.reserve(kTotalTasks);
  for (auto& tasks : threadTasks) {
    for (auto& task : tasks) {
      allTasks.push_back(std::move(task));
    }
  }

  ASSERT_EQ(allTasks.size(), kTotalTasks);

  // Collect stats from all tasks.
  int totalCacheMisses = 0;
  int totalCacheHits = 0;

  for (const auto& task : allTasks) {
    auto opStats = toOperatorStats(task->taskStats());
    ASSERT_EQ(opStats.count("HashBuild"), 1);
    auto& hashBuildStats = opStats.at("HashBuild");

    if (hashBuildStats.runtimeStats.count(
            std::string(BaseHashTable::kHashTableCacheMiss))) {
      totalCacheMisses +=
          hashBuildStats.runtimeStats
              .at(std::string(BaseHashTable::kHashTableCacheMiss))
              .count;
    }
    if (hashBuildStats.runtimeStats.count(
            std::string(BaseHashTable::kHashTableCacheHit))) {
      totalCacheHits += hashBuildStats.runtimeStats
                            .at(std::string(BaseHashTable::kHashTableCacheHit))
                            .count;
    }
  }

  // Exactly one task should build (cache miss) and all others should reuse
  // (cache hits).
  EXPECT_EQ(totalCacheMisses, 1)
      << "Exactly one task should report a cache miss (the builder)";
  EXPECT_EQ(totalCacheHits, kTotalTasks - 1)
      << "All other tasks should report cache hits";

  // Clean up cache entry before tasks are destroyed.
  const auto cacheKey = fmt::format("{}:{}", queryId, joinNodeId);
  HashTableCache::instance()->drop(cacheKey);
}

// Tests that HashBuild and HashProbe cannot reclaim when using a cached hash
// table. When useHashTableCache() is true, canReclaim() returns false for both
// operators because spilling would clear the cached table and corrupt it for
// other tasks. This test uses TestValue to verify canReclaim() returns false.
DEBUG_ONLY_TEST_F(HashJoinWithCacheTest, probeCannotSpillWithCachedTable) {
  const std::string queryId =
      "probeCannotSpillTest_" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());

  // Create build and probe vectors.
  std::vector<RowVectorPtr> buildVectors = makeBatches(1, [&](int32_t) {
    return makeRowVector(
        {"u_k", "u_v"},
        {
            makeFlatVector<int32_t>(100, [](auto row) { return row % 23; }),
            makeFlatVector<int64_t>(100, [](auto row) { return row * 10; }),
        });
  });

  std::vector<RowVectorPtr> probeVectors = makeBatches(5, [&](int32_t) {
    return makeRowVector(
        {"t_k", "t_v"},
        {
            makeFlatVector<int32_t>(100, [](auto row) { return row % 23; }),
            makeFlatVector<int64_t>(100, [](auto row) { return row; }),
        });
  });

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlanNode =
      PlanBuilder(planNodeIdGenerator).values(buildVectors).planNode();
  auto probePlanNode =
      PlanBuilder(planNodeIdGenerator).values(probeVectors).planNode();

  auto outputType = ROW(
      {"t_k", "t_v", "u_k", "u_v"}, {INTEGER(), BIGINT(), INTEGER(), BIGINT()});

  auto joinNode =
      core::HashJoinNode::Builder()
          .id(planNodeIdGenerator->next())
          .joinType(core::JoinType::kInner)
          .nullAware(false)
          .leftKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "t_k")})
          .rightKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "u_k")})
          .left(probePlanNode)
          .right(buildPlanNode)
          .outputType(outputType)
          .useHashTableCache(true)
          .build();
  const auto joinNodeId = joinNode->id();

  auto queryCtx = core::QueryCtx::create(
      driverExecutor_.get(),
      core::QueryConfig({}),
      std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
      cache::AsyncDataCache::getInstance(),
      nullptr,
      nullptr,
      queryId);

  // Use TestValue to verify canReclaim() returns false for HashBuild.
  std::atomic_bool hashBuildChecked{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::addInput",
      std::function<void(HashBuild*)>([&](HashBuild* build) {
        if (hashBuildChecked.exchange(true)) {
          return;
        }
        ASSERT_FALSE(build->canReclaim())
            << "HashBuild should not be reclaimable with cached hash table";
      }));

  // Use TestValue to verify canReclaim() returns false for HashProbe.
  std::atomic_bool hashProbeChecked{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::addInput",
      std::function<void(Operator*)>([&](Operator* op) {
        if (!isHashProbeMemoryPool(*op->pool())) {
          return;
        }
        if (hashProbeChecked.exchange(true)) {
          return;
        }
        auto* probe = dynamic_cast<HashProbe*>(op);
        ASSERT_NE(probe, nullptr);
        ASSERT_FALSE(probe->canReclaim())
            << "HashProbe should not be reclaimable with cached hash table";
      }));

  // Run the query and verify results.
  auto task =
      AssertQueryBuilder(joinNode, duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .assertResults(
              "SELECT t.t_k, t.t_v, u.u_k, u.u_v FROM t, u WHERE t.t_k = u.u_k");

  // Verify that both operators were checked.
  ASSERT_TRUE(hashBuildChecked) << "HashBuild canReclaim check was not reached";
  ASSERT_TRUE(hashProbeChecked) << "HashProbe canReclaim check was not reached";

  // Verify that HashProbe operator stats show no spilling.
  auto opStats = toOperatorStats(task->taskStats());
  ASSERT_EQ(opStats.count("HashProbe"), 1);
  auto& probeStats = opStats.at("HashProbe");

  EXPECT_EQ(probeStats.spilledInputBytes, 0)
      << "HashProbe should not spill when using cached hash table";
  EXPECT_EQ(probeStats.spilledBytes, 0)
      << "HashProbe should not spill when using cached hash table";

  // Clean up cache entry before task is destroyed.
  const auto cacheKey = fmt::format("{}:{}", queryId, joinNodeId);
  HashTableCache::instance()->drop(cacheKey);
}

// Tests OOM behavior with cached hash tables via memory arbitration.
// This test triggers memory arbitration during HashProbe to verify:
// 1. HashProbe::canReclaim() returns false when useHashTableCache=true
// 2. When an allocation exceeds capacity, arbitration runs but can't reclaim
// 3. OOM is thrown by the arbitration framework
// 4. Cleanup works correctly via QueryCtx release callbacks
DEBUG_ONLY_TEST_F(HashJoinWithCacheTest, probeOOMWithCachedTable) {
  const std::string queryId =
      "probeOOMTest_" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());

  // Create build side with ~1MB hash table.
  // 10 batches × 10,000 rows = 100,000 rows × 12 bytes = ~1.2MB raw data.
  std::vector<RowVectorPtr> buildVectors = makeBatches(10, [&](int32_t) {
    return makeRowVector(
        {"u_k", "u_v"},
        {
            makeFlatVector<int32_t>(10000, [](auto row) { return row % 5000; }),
            makeFlatVector<int64_t>(10000, [](auto row) { return row * 10; }),
        });
  });

  // Create probe vectors with matching key range.
  std::vector<RowVectorPtr> probeVectors = makeBatches(10, [&](int32_t) {
    return makeRowVector(
        {"t_k", "t_v"},
        {
            makeFlatVector<int32_t>(1000, [](auto row) { return row % 5000; }),
            makeFlatVector<int64_t>(1000, [](auto row) { return row; }),
        });
  });

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlanNode =
      PlanBuilder(planNodeIdGenerator).values(buildVectors).planNode();
  auto probePlanNode =
      PlanBuilder(planNodeIdGenerator).values(probeVectors).planNode();

  auto outputType = ROW(
      {"t_k", "t_v", "u_k", "u_v"}, {INTEGER(), BIGINT(), INTEGER(), BIGINT()});

  auto joinNode =
      core::HashJoinNode::Builder()
          .id(planNodeIdGenerator->next())
          .joinType(core::JoinType::kInner)
          .nullAware(false)
          .leftKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "t_k")})
          .rightKeys(
              {std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "u_k")})
          .left(probePlanNode)
          .right(buildPlanNode)
          .outputType(outputType)
          .useHashTableCache(true)
          .build();

  // Create QueryCtx with sufficient memory for build but we'll exhaust it
  // during probe via TestValue injection.
  auto queryCtx = core::QueryCtx::create(
      driverExecutor_.get(),
      core::QueryConfig({}),
      std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
      cache::AsyncDataCache::getInstance(),
      nullptr,
      nullptr,
      queryId);

  // Use a pool with limited capacity. Build uses ~10MB, so 20MB should be
  // enough for build but tight for additional allocations during probe.
  constexpr int64_t kPoolCapacity = 20 * 1024 * 1024; // 20MB
  queryCtx->testingOverrideMemoryPool(
      memory::memoryManager()->addRootPool(
          queryCtx->queryId(), kPoolCapacity, exec::MemoryReclaimer::create()));

  // Use TestValue at the addInput injection point to trigger OOM during
  // HashProbe. We allocate more memory than the pool has available, which
  // triggers arbitration. Since HashProbe::canReclaim() returns false when
  // useHashTableCache=true, arbitration cannot reclaim and OOM is thrown.
  std::atomic_bool injected{false};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::Driver::runInternal::addInput",
      std::function<void(Operator*)>([&](Operator* op) {
        // Only inject once, and only for HashProbe operator.
        if (!isHashProbeMemoryPool(*op->pool())) {
          return;
        }
        if (injected.exchange(true)) {
          return;
        }

        auto* probe = dynamic_cast<HashProbe*>(op);
        ASSERT_NE(probe, nullptr);

        // Verify that HashProbe cannot reclaim when using cached hash table.
        // canReclaim() returns false because canSpill() is false.
        ASSERT_FALSE(probe->canReclaim())
            << "HashProbe should not be reclaimable with cached hash table";

        // Allocate memory equal to pool capacity.
        // If HashProbe could spill, arbitration would reclaim memory and this
        // allocation would succeed. But since canReclaim() returns false with
        // cached hash table, arbitration can't free memory and OOM is thrown.
        auto* pool = op->pool();
        // This allocation will trigger arbitration and throw OOM.
        pool->allocate(kPoolCapacity);
      }));

  // This should throw OOM during probe. The cleanup should work correctly.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(joinNode, duckDbQueryRunner_)
          .queryCtx(queryCtx)
          .maxDrivers(1)
          .copyResults(pool()),
      "Exceeded memory pool capacity");

  waitForAllTasksToBeDeleted();
  queryCtx.reset();
  // Cache should be cleaned up by QueryCtx destructor via release callback.
}

} // namespace
} // namespace facebook::velox::exec::test
