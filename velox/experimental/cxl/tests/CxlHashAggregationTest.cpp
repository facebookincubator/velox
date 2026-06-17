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
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/core/QueryConfig.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/Spill.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/experimental/cxl/CxlDriverAdapter.h"
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
    filesystems::registerLocalFileSystem();
    common::testutil::TestValue::enable();
    // DriverAdapters are process-global and cannot be unregistered, so
    // register exactly once for the whole test binary.
    static const bool registered = [] {
      registerCxlDriverAdapter();
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
            // CXL relocation reuses the spill reservation machinery, so spill
            // must be enabled for the operator to be reclaimable. A minimal
            // spillable reservation keeps DRAM lean so relocation fires
            // promptly under pressure.
            .queryConfig(
                core::QueryConfig{std::unordered_map<std::string, std::string>{
                    {core::QueryConfig::kSpillEnabled, "true"},
                    {core::QueryConfig::kAggregationSpillEnabled, "true"},
                    {core::QueryConfig::kMinSpillableReservationPct, "0"},
                    {core::QueryConfig::kSpillableReservationGrowthPct, "10"},
                }})
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
// HashAggregation for a CxlHashAggregation that resolves its per-query CXL
// pool. The swap reports through the diagnostics counter (the operator inherits
// the "Aggregation" type), and results stay correct.
TEST_F(CxlHashAggregationTest, adapterSwapsInCxlAggregationAndReachesCxlPool) {
  auto data = makeAggInput();
  createDuckDbTable({data});

  exec::CursorParameters params;
  params.planNode = makeAggPlan(data);
  params.queryCtx = makeQueryCtxWithCxl("cxl-agg-seam");

  assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(numCxlHashAggregationsInitialized(), 0);
  EXPECT_EQ(
      numCxlHashAggregationsWithCxlPool(), numCxlHashAggregationsInitialized());
}

// Without a CXL tier on the query, the adapter leaves the stock HashAggregation
// in place: no CxlHashAggregation is installed. Results match the reference.
TEST_F(CxlHashAggregationTest, withoutCxlTierKeepsStockAggregation) {
  auto data = makeAggInput();
  createDuckDbTable({data});

  assertQuery(makeAggPlan(data), "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_EQ(numCxlHashAggregationsInitialized(), 0);
}

// Under memory arbitration the operator reclaims by relocating its groups to
// CXL rather than spilling to disk. Spill injection drives the reclaim path
// deterministically. The data carries a null grouping key whose group must
// survive the relocation byte copy: that requires the destination container's
// column stats to be transferred, otherwise the null group reads back as a
// non-null garbage key. Results must match the reference query.
TEST_F(CxlHashAggregationTest, relocatesToCxlUnderArbitration) {
  constexpr int32_t kNumBatches = 8;
  constexpr int32_t kRowsPerBatch = 4'096;
  std::vector<RowVectorPtr> batches;
  for (auto batch = 0; batch < kNumBatches; ++batch) {
    batches.push_back(makeRowVector({
        // Distinct keys per batch, plus a null key every 1000th row.
        makeFlatVector<int64_t>(
            kRowsPerBatch,
            [batch](auto row) {
              return static_cast<int64_t>(batch) * kRowsPerBatch + row;
            },
            [](auto row) { return row % 1'000 == 0; }),
        makeFlatVector<int64_t>(
            kRowsPerBatch, [batch](auto row) { return row + batch; }),
    }));
  }
  createDuckDbTable(batches);

  // Force the reclaim path once; the operator's reclaim() relocates to CXL.
  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();

  exec::CursorParameters params;
  params.planNode = PlanBuilder()
                        .values(batches)
                        .singleAggregation({"c0"}, {"sum(c1)"})
                        .planNode();
  params.queryCtx = makeQueryCtxWithCxl("cxl-agg-arbitrator");
  params.spillDirectory = spillDirectory->getPath();

  assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_GT(numCxlPartitionsMigrated(), 0);
}

} // namespace
} // namespace facebook::velox::cxl

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
