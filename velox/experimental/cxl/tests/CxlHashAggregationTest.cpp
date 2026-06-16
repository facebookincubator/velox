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

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cxl/CxlHashAggregation.h"
#include "velox/experimental/cxl/CxlMemoryResource.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cxl {
namespace {

// CXL custom resource backed by MallocAllocator so the seam test runs on
// hardware without real CXL devices. The real CxlMemoryAllocator path is
// covered by CxlMemoryResourceTest, which is skipped when NUMA is unavailable.
// Here we only need a custom pool tagged "cxl" that the operator can resolve.
std::shared_ptr<memory::CustomMemoryResource> makeEmulatedCxlResource() {
  memory::MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCxlResourceTag},
      std::make_shared<memory::MallocAllocator>(options),
      memory::MemoryArbitrator::create({}),
      [] { return memory::MemoryReclaimer::create(0); });
}

class CxlHashAggregationTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    common::testutil::TestValue::enable();
    // DriverAdapters are process-global and cannot be unregistered, so
    // register exactly once for the whole test binary.
    static const bool registered = [] {
      registerCxlHashAggregationAdapter();
      return true;
    }();
    (void)registered;
    resetCxlHashAggregationCounters();
  }

  // Builds a QueryCtx carrying a per-query "cxl" custom root pool plus an
  // isolated registry, so the Task can resolve the resource when mirroring the
  // custom pool tree and the operator can reach its CXL leaf via
  // customPool("cxl").
  std::shared_ptr<core::QueryCtx> makeQueryCtxWithCxl(
      const std::string& queryId) {
    auto* manager = memory::memoryManager();
    auto resource = makeEmulatedCxlResource();
    auto pool =
        manager->addCustomRootPool(fmt::format("{}.cxl", queryId), resource);
    auto queryCtx =
        core::QueryCtx::Builder()
            .executor(driverExecutor_.get())
            .customPool(std::string{kCxlResourceTag}, std::move(pool))
            .queryId(queryId)
            .build();
    auto registry =
        memory::CustomMemoryResourceRegistry::createRegistry(nullptr);
    queryCtx->setRegistry<memory::CustomMemoryResourceRegistry::Registry>(
        memory::kCustomMemoryResourceRegistryKey, registry);
    registry->insert(std::string{kCxlResourceTag}, resource);
    return queryCtx;
  }

  // Flattens the operator types across all pipelines of a finished task.
  static std::vector<std::string> operatorTypes(const exec::Task& task) {
    std::vector<std::string> types;
    for (const auto& pipeline : task.taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        types.push_back(op.operatorType);
      }
    }
    return types;
  }

  RowVectorPtr makeAggInput() {
    return makeRowVector({
        makeFlatVector<int64_t>(1'000, [](auto row) { return row % 10; }),
        makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
    });
  }

  core::PlanNodePtr makeAggPlan(const RowVectorPtr& data) {
    return PlanBuilder()
        .values({data})
        .singleAggregation({"c0"}, {"sum(c1)"})
        .planNode();
  }
};

// With a CXL tier registered on the query, the adapter swaps the stock
// HashAggregation for a CxlHashAggregation, and the replacement resolves its
// per-query CXL pool. Results stay correct.
TEST_F(CxlHashAggregationTest, adapterSwapsInCxlAggregationAndReachesCxlPool) {
  auto data = makeAggInput();
  createDuckDbTable({data});

  exec::CursorParameters params;
  params.planNode = makeAggPlan(data);
  params.queryCtx = makeQueryCtxWithCxl("cxl-agg-seam");

  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  // The stock HashAggregation was replaced: its distinct operator type shows
  // up in the task stats, and no plain "Aggregation" operator remains.
  EXPECT_THAT(
      operatorTypes(*task),
      testing::Contains(std::string(CxlHashAggregation::kOperatorType)));
  EXPECT_THAT(
      operatorTypes(*task),
      testing::Not(
          testing::Contains(std::string(exec::OperatorType::kAggregation))));

  // The replacement initialized and resolved a non-null per-query CXL pool.
  EXPECT_GT(numCxlHashAggregationsInitialized(), 0);
  EXPECT_EQ(
      numCxlHashAggregationsWithCxlPool(), numCxlHashAggregationsInitialized());
}

// Without a CXL tier on the query, the adapter leaves the stock
// HashAggregation in place: there is nothing to relocate to, so no
// CxlHashAggregation is installed. Results match the reference query.
TEST_F(CxlHashAggregationTest, withoutCxlTierKeepsStockAggregation) {
  auto data = makeAggInput();
  createDuckDbTable({data});

  auto task =
      assertQuery(makeAggPlan(data), "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_THAT(
      operatorTypes(*task),
      testing::Contains(std::string(exec::OperatorType::kAggregation)));
  EXPECT_THAT(
      operatorTypes(*task),
      testing::Not(testing::Contains(
          std::string(CxlHashAggregation::kOperatorType))));
  EXPECT_EQ(numCxlHashAggregationsInitialized(), 0);
}

// Under memory pressure the operator relocates its partition's groups DRAM ->
// CXL mid-build (here triggered deterministically by reclaiming after each
// batch while input is still pending). Later batches then re-probe the
// swizzled, CXL-resident table and update those accumulators. Results must
// still match the reference, proving the relocate + bucket swizzle preserved
// every group and kept the index valid for continued aggregation.
DEBUG_ONLY_TEST_F(
    CxlHashAggregationTest,
    relocatesToCxlUnderMemoryPressureAndPreservesResults) {
  // Multiple batches over the same 10 grouping keys, so batches after the first
  // relocate hit groups that already live in the CXL container.
  constexpr int32_t kNumBatches = 8;
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < kNumBatches; ++batch) {
    batches.push_back(makeRowVector({
        makeFlatVector<int64_t>(1'000, [](auto row) { return row % 10; }),
        makeFlatVector<int64_t>(
            1'000, [batch](auto row) { return row + batch; }),
    }));
  }
  createDuckDbTable(batches);

  // Relocate to CXL after each batch while input remains, exercising the bucket
  // swizzle on a live table.
  std::atomic<int64_t> reclaims{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cxl::CxlHashAggregation::addInput",
      std::function<void(CxlHashAggregation*)>([&](CxlHashAggregation* op) {
        if (!op->canReclaim()) {
          return;
        }
        memory::ScopedMemoryArbitrationContext ctx(op->pool());
        memory::MemoryReclaimer::Stats stats;
        op->reclaim(/*targetBytes=*/0, stats);
        ++reclaims;
      }));

  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithCxl("cxl-agg-pressure");

  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(reclaims.load(), 0);
  EXPECT_GT(numCxlPartitionsMigrated(), 0);
  EXPECT_GT(numCxlHashAggregationsWithCxlPool(), 0);
}

// The arbitrator itself must be able to drive relocation: with the query's
// DRAM pool capped below the group table, hitting the cap mid-build must
// reclaim via relocateRowsToCxl() instead of failing with "Exceeded memory pool
// capacity". This exercises the real arbitration path (no TestValue): the
// operator reserves memory at a safe point (ensureInputFits) inside a
// reclaimable section, since the driver marks operators non-reclaimable for
// the whole of addInput() otherwise.
TEST_F(CxlHashAggregationTest, relocatesViaMemoryArbitratorUnderCappedPool) {
  // Growing distinct BIGINT keys so the group table keeps growing across
  // batches and eventually exceeds the capped DRAM pool.
  constexpr int32_t kNumBatches = 64;
  constexpr int32_t kRowsPerBatch = 4'096;
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < kNumBatches; ++batch) {
    batches.push_back(makeRowVector({
        makeFlatVector<int64_t>(
            kRowsPerBatch,
            [batch](auto row) {
              return static_cast<int64_t>(batch) * kRowsPerBatch + row;
            }),
        makeFlatVector<int64_t>(
            kRowsPerBatch, [batch](auto row) { return row + batch; }),
    }));
  }
  createDuckDbTable(batches);

  // Cap the query's DRAM pool well below the final group table (~260k groups)
  // but above the bucket array, which stays DRAM-resident by design.
  constexpr int64_t kDramCap = 8L << 20;
  auto queryCtx = makeQueryCtxWithCxl("cxl-agg-arbitrator");
  queryCtx->testingOverrideMemoryPool(
      memory::memoryManager()->addRootPool(
          queryCtx->queryId(), kDramCap, memory::MemoryReclaimer::create()));

  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = queryCtx;

  assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  // Relocation fired through the arbitrator, not through a test hook.
  EXPECT_GT(numCxlPartitionsMigrated(), 0);
}

// A null grouping key forms a valid group that must survive relocation to CXL.
// The byte copy preserves each row's null flag, but the destination container's
// column stats (which gate the null-aware extract path) must be transferred too
// — otherwise the null group is read back as a non-null garbage key.
// Relocation is injected via TestValue, which only exists in debug builds.
DEBUG_ONLY_TEST_F(CxlHashAggregationTest, nullGroupingKeySurvivesRelocation) {
  constexpr int32_t kNumBatches = 4;
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < kNumBatches; ++batch) {
    batches.push_back(makeRowVector({
        // Every tenth row carries a null grouping key (its own NULL group).
        makeFlatVector<int64_t>(
            1'000,
            [](auto row) { return row % 10; },
            [](auto row) { return row % 10 == 0; }),
        makeFlatVector<int64_t>(
            1'000, [batch](auto row) { return row + batch; }),
    }));
  }
  createDuckDbTable(batches);

  std::atomic<int64_t> reclaims{0};
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cxl::CxlHashAggregation::addInput",
      std::function<void(CxlHashAggregation*)>([&](CxlHashAggregation* op) {
        if (!op->canReclaim()) {
          return;
        }
        memory::ScopedMemoryArbitrationContext ctx(op->pool());
        memory::MemoryReclaimer::Stats stats;
        op->reclaim(/*targetBytes=*/0, stats);
        ++reclaims;
      }));

  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithCxl("cxl-agg-null-key");

  auto task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(reclaims.load(), 0);
  EXPECT_GT(numCxlPartitionsMigrated(), 0);
}

} // namespace
} // namespace facebook::velox::cxl

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
