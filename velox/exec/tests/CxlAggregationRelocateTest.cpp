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
#include "velox/core/QueryCtx.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/GroupingSet.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Spill.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/experimental/cxl/CxlResourceTag.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cxl {
namespace {

// A memory tier resource tagged "cxl" but backed by MallocAllocator, so the
// relocation seam runs on hardware without real CXL devices. The libnuma path
// is covered by CxlMemoryResourceTest (skipped when NUMA is unavailable); here
// we only need a custom pool the stock aggregation can resolve by tag.
std::shared_ptr<memory::CustomMemoryResource> makeEmulatedTierResource() {
  memory::MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  return std::make_shared<memory::CustomMemoryResource>(
      std::string{kCxlResourceTag},
      std::make_shared<memory::MallocAllocator>(options),
      memory::MemoryArbitrator::create({}),
      [] { return memory::MemoryReclaimer::create(0); });
}

class CxlAggregationRelocateTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    TestValue::enable();
  }

  // Builds a QueryCtx carrying a per-query "cxl" custom root pool plus an
  // isolated registry, so the Task can resolve the resource when mirroring the
  // custom pool tree and the operator can reach its tier leaf via
  // customPool("cxl"). No disk spill is configured: the tier is the only
  // reclaim destination.
  std::shared_ptr<core::QueryCtx> makeQueryCtxWithTier(
      const std::string& queryId) {
    auto* manager = memory::memoryManager();
    auto resource = makeEmulatedTierResource();
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

  // Counts GroupingSet::relocate() calls for the duration of a query.
  template <typename F>
  int countRelocations(F&& runQuery) {
    std::atomic<int> relocations{0};
    SCOPED_TESTVALUE_SET(
        "facebook::velox::exec::GroupingSet::relocate",
        std::function<void(exec::GroupingSet*)>(
            [&](exec::GroupingSet*) { ++relocations; }));
    runQuery();
    return relocations.load();
  }
};

// Under memory arbitration a grouped, fixed-width aggregation reclaims by
// relocating its groups to the configured tier rather than spilling to disk.
// Relocation rides on the spill reclaim path, so aggregation spill is enabled
// and a spill directory is provided; the tier intercepts reclaim before any
// disk write, so the spill machinery is armed but never used. Spill injection
// drives the reclaim path deterministically. The data carries a null grouping
// key whose group must survive the relocation byte copy -- that requires the
// destination container's column stats to be carried over, else the null group
// reads back as a non-null garbage key. Results must match the reference query,
// relocation must fire, and no disk spill must occur (either/or with spill).
TEST_F(CxlAggregationRelocateTest, relocatesToTierUnderArbitration) {
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

  // Force the reclaim path once; reclaim() relocates to the tier instead of
  // spilling to disk.
  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  std::shared_ptr<exec::Task> task;
  const int relocations = countRelocations([&] {
    exec::CursorParameters params;
    params.planNode = PlanBuilder()
                          .values(batches)
                          .singleAggregation({"c0"}, {"sum(c1)"})
                          .planNode();
    params.queryCtx = makeQueryCtxWithTier("cxl-agg-arbitrator");
    params.spillDirectory = spillDirectory->getPath();
    params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
    params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
    task = assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");
  });

  EXPECT_GT(relocations, 0);
  // Either/or: relocation replaced disk spill, so nothing reached disk.
  for (const auto& [_, nodeStats] : exec::toPlanStats(task->taskStats())) {
    EXPECT_EQ(nodeStats.spilledBytes, 0);
  }
}

// A varchar grouping key uses external (out-of-line) memory, so its rows cannot
// be byte-relocated. The gate must exclude it: even with spill enabled and a
// tier configured, reclaim falls back to disk spill (which would otherwise
// throw in relocateRunsTo) and the query returns correct results.
TEST_F(CxlAggregationRelocateTest, varcharKeyDoesNotRelocate) {
  auto data = makeRowVector({
      makeFlatVector<std::string>(
          1'000, [](auto row) { return fmt::format("k{}", row % 17); }),
      makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  exec::TestScopedSpillInjection injectSpill(
      /*spillPct=*/100, /*poolRegExp=*/".*", /*maxInjections=*/1);

  auto spillDirectory = exec::test::TempDirectoryPath::create();
  const int relocations = countRelocations([&] {
    exec::CursorParameters params;
    params.planNode = PlanBuilder()
                          .values({data})
                          .singleAggregation({"c0"}, {"sum(c1)"})
                          .planNode();
    params.queryCtx = makeQueryCtxWithTier("cxl-agg-varchar");
    params.spillDirectory = spillDirectory->getPath();
    params.queryConfigs[core::QueryConfig::kSpillEnabled] = "true";
    params.queryConfigs[core::QueryConfig::kAggregationSpillEnabled] = "true";
    assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");
  });

  EXPECT_EQ(relocations, 0);
}

// A partial aggregation is not a complete reclaim point, so the gate excludes
// it: relocation never fires and results are correct.
TEST_F(CxlAggregationRelocateTest, partialAggregationDoesNotRelocate) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(1'000, [](auto row) { return row % 10; }),
      makeFlatVector<int64_t>(1'000, [](auto row) { return row; }),
  });
  createDuckDbTable({data});

  const int relocations = countRelocations([&] {
    exec::CursorParameters params;
    params.planNode = PlanBuilder()
                          .values({data})
                          .partialAggregation({"c0"}, {"sum(c1)"})
                          .finalAggregation()
                          .planNode();
    params.queryCtx = makeQueryCtxWithTier("cxl-agg-partial");
    assertQuery(params, "SELECT c0, sum(c1) FROM tmp GROUP BY c0");
  });

  // The partial stage never relocates; the final stage relocates only under
  // pressure, which this unpressured run does not create.
  EXPECT_EQ(relocations, 0);
}

} // namespace
} // namespace facebook::velox::cxl
