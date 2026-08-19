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

#include <cstring>
#include <type_traits>
#include <unordered_map>

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/RowIntervalSet.h"
#include "velox/dwio/parquet/common/PageIndex.h"
#include "velox/dwio/parquet/reader/PagePruningPlan.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

namespace {

template <typename T>
std::string encode(T value) {
  std::string bytes(sizeof(T), '\0');
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

thrift::OffsetIndex makeOffsetIndex(
    std::vector<int64_t> firstRows,
    std::vector<int64_t> offsets,
    std::vector<int32_t> sizes) {
  thrift::OffsetIndex index;
  for (size_t i = 0; i < firstRows.size(); ++i) {
    thrift::PageLocation location;
    location.first_row_index() = firstRows[i];
    location.offset() = offsets[i];
    location.compressed_page_size() = sizes[i];
    index.page_locations()->push_back(std::move(location));
  }
  return index;
}

thrift::ColumnIndex makeColumnIndex(
    std::vector<bool> nullPages,
    std::vector<std::string> minimums,
    std::vector<std::string> maximums,
    std::vector<int64_t> nullCounts) {
  thrift::ColumnIndex index;
  index.null_pages() = std::move(nullPages);
  index.min_values() = std::move(minimums);
  index.max_values() = std::move(maximums);
  index.boundary_order() = thrift::BoundaryOrder::ASCENDING;
  index.null_counts() = std::move(nullCounts);
  return index;
}

ValidatedOffsetIndex makeValidatedOffsetIndex(
    const thrift::OffsetIndex& offsetIndex,
    std::optional<uint64_t> dataPageOffset = std::nullopt,
    std::optional<common::Region> columnChunkRegion = std::nullopt) {
  auto result = decodeOffsetIndex(
      offsetIndex, 20, 1'000, dataPageOffset, columnChunkRegion);
  EXPECT_TRUE(result);
  return result.value.value_or(ValidatedOffsetIndex{});
}

} // namespace

TEST(PageIndexTest, decodesTypedPhysicalBounds) {
  auto columnIndex = makeColumnIndex(
      {false}, {encode<int32_t>(-10)}, {encode<int32_t>(20)}, {0});
  auto offsetIndex = makeOffsetIndex({0}, {100}, {20});

  auto decoded = decodeColumnIndex(
      columnIndex,
      makeValidatedOffsetIndex(offsetIndex),
      thrift::Type::INT32,
      std::nullopt,
      PageBoundsCapability::kOrderedBounds,
      20,
      1'000);

  ASSERT_TRUE(decoded);
  ASSERT_TRUE(decoded.value->column.has_value());
  ASSERT_EQ(decoded.value->offset.pages.size(), 1);
  EXPECT_EQ(decoded.value->offset.pages[0].firstRow, 0);
  EXPECT_EQ(decoded.value->offset.pages[0].numRows, 20);
  ASSERT_TRUE(decoded.value->column->bounds(0).minimum.has_value());
  EXPECT_EQ(std::get<int32_t>(*decoded.value->column->bounds(0).minimum), -10);
}

TEST(PageIndexTest, rejectsCardinalityAndRowOrderErrors) {
  auto columnIndex =
      makeColumnIndex({false}, {encode<int32_t>(1)}, {encode<int32_t>(2)}, {0});
  auto offsetIndex = makeOffsetIndex({1}, {100}, {20});

  auto decoded = decodeOffsetIndex(offsetIndex, 20, 1'000);

  EXPECT_FALSE(decoded);
  EXPECT_EQ(decoded.reason, PageIndexFallbackReason::kInvalidRowOrder);
}

TEST(PageIndexTest, rejectsInvalidLocationsBeforeArithmetic) {
  auto negativeOffset = makeOffsetIndex({0}, {-1}, {20});

  auto decoded = decodeOffsetIndex(negativeOffset, 20, 1'000);

  EXPECT_FALSE(decoded);
  EXPECT_EQ(decoded.reason, PageIndexFallbackReason::kInvalidLocation);
}

TEST(PageIndexTest, rejectsPageOutsideColumnChunk) {
  auto offsetIndex = makeOffsetIndex({0}, {200}, {20});

  auto decoded =
      decodeOffsetIndex(offsetIndex, 20, 1'000, 100, common::Region{100, 20});

  EXPECT_FALSE(decoded);
  EXPECT_EQ(decoded.reason, PageIndexFallbackReason::kInvalidLocation);
}

TEST(PageIndexTest, rejectsContradictoryOrderedBounds) {
  auto columnIndex = makeColumnIndex(
      {false}, {encode<int32_t>(20)}, {encode<int32_t>(0)}, {0});
  auto offsetIndex = makeOffsetIndex({0}, {100}, {20});

  auto decoded = decodeColumnIndex(
      columnIndex,
      makeValidatedOffsetIndex(offsetIndex),
      thrift::Type::INT32,
      std::nullopt,
      PageBoundsCapability::kOrderedBounds,
      20,
      1'000);

  ASSERT_TRUE(decoded);
  ASSERT_TRUE(decoded.value->column.has_value());
  EXPECT_FALSE(decoded.value->column->bounds(0).minimum.has_value());
  EXPECT_FALSE(decoded.value->column->bounds(0).maximum.has_value());
}

TEST(PageIndexTest, rejectsContradictoryNullMetadata) {
  auto columnIndex =
      makeColumnIndex({true}, {encode<int32_t>(0)}, {encode<int32_t>(0)}, {0});
  auto offsetIndex = makeOffsetIndex({0}, {100}, {20});

  auto decoded = decodeColumnIndex(
      columnIndex,
      makeValidatedOffsetIndex(offsetIndex),
      thrift::Type::INT32,
      std::nullopt,
      PageBoundsCapability::kNullabilityOnly,
      20,
      1'000);

  EXPECT_FALSE(decoded);
  EXPECT_EQ(decoded.reason, PageIndexFallbackReason::kInvalidBounds);
}

TEST(PageIndexTest, evaluatesUnorderedPagesIndependently) {
  auto columnIndex = makeColumnIndex(
      {false}, {encode<int32_t>(10)}, {encode<int32_t>(20)}, {0});
  columnIndex.boundary_order() = thrift::BoundaryOrder::UNORDERED;
  auto offsetIndex = makeOffsetIndex({0}, {100}, {20});

  auto decoded = decodeColumnIndex(
      columnIndex,
      makeValidatedOffsetIndex(offsetIndex),
      thrift::Type::INT32,
      std::nullopt,
      PageBoundsCapability::kOrderedBounds,
      20,
      1'000);

  ASSERT_TRUE(decoded);
  ASSERT_TRUE(decoded.value->column.has_value());
  EXPECT_EQ(
      decoded.value->column->capability(),
      PageBoundsCapability::kOrderedBounds);
  EXPECT_TRUE(decoded.value->column->bounds(0).minimum.has_value());
}

TEST(PageIndexTest, rejectsMalformedBooleanBounds) {
  auto columnIndex = makeColumnIndex({false}, {"\x80"}, {"\x80"}, {0});
  auto offsetIndex = makeOffsetIndex({0}, {100}, {20});

  auto decoded = decodeColumnIndex(
      columnIndex,
      makeValidatedOffsetIndex(offsetIndex),
      thrift::Type::BOOLEAN,
      std::nullopt,
      PageBoundsCapability::kOrderedBounds,
      20,
      1'000);

  EXPECT_FALSE(decoded);
  EXPECT_EQ(decoded.reason, PageIndexFallbackReason::kInvalidBounds);
}

TEST(PagePruningPlanTest, keepsGappedPagesInSeparateLogicalRuns) {
  ValidatedOffsetIndex offset;
  offset.pages = {
      {100, 50, 0, 10},
      {150, 50, 10, 10},
      {250, 25, 20, 10},
  };
  RowIntervalSet retained;
  retained.add({0, 20});
  retained.add({20, 30});

  auto column = buildColumnPageReadPlan(
      3, offset, retained, common::Region{50, 50}, 200, 40);

  EXPECT_EQ(column.numSkippedPages, 0);
  ASSERT_EQ(column.retainedRuns.size(), 2);
  EXPECT_EQ(column.retainedRuns[0].region.offset, 100);
  EXPECT_EQ(column.retainedRuns[0].region.length, 100);
  EXPECT_EQ(column.retainedRuns[1].region.offset, 250);
  EXPECT_EQ(column.retainedRuns[1].region.length, 25);
  EXPECT_EQ(column.dataPageToRun, (std::vector<int32_t>{0, 0, 1}));
}

TEST(PagePruningPlanTest, skipsPagesFromImmutableRows) {
  ValidatedOffsetIndex offset;
  offset.pages = {
      {100, 50, 0, 10},
      {150, 50, 10, 10},
      {200, 50, 20, 10},
  };
  RowIntervalSet retained;
  retained.add({10, 20});

  auto column =
      buildColumnPageReadPlan(1, offset, retained, std::nullopt, 150, 20);

  EXPECT_EQ(column.numSkippedPages, 2);
  ASSERT_EQ(column.retainedRuns.size(), 1);
  EXPECT_EQ(column.retainedRuns[0].firstDataPageOrdinal, 1);
  EXPECT_EQ(column.dataPageToRun[0], -1);
  EXPECT_EQ(column.dataPageToRun[1], 0);
  EXPECT_EQ(column.dataPageToRun[2], -1);
}

TEST(PagePruningPlanTest, fallsBackWhenPhysicalSavingsAreNotMaterial) {
  ValidatedOffsetIndex offset;
  offset.pages = {
      {100, 50, 0, 10},
      {150, 50, 10, 10},
  };
  RowIntervalSet retained;
  retained.add({0, 10});

  auto preloaded =
      buildColumnPageReadPlan(1, offset, retained, std::nullopt, 50, 20, true);
  EXPECT_FALSE(preloaded.useWholeChunkStream);

  auto noSavings =
      buildColumnPageReadPlan(1, offset, retained, std::nullopt, 50, 20, false);
  EXPECT_TRUE(noSavings.useWholeChunkStream);
}

TEST(PagePruningPlanTest, treatsIndexBytesAsSunkCost) {
  ValidatedOffsetIndex offset;
  offset.pages = {
      {100, 50, 0, 10},
      {150, 50, 10, 10},
  };
  RowIntervalSet retained;
  retained.add({0, 10});

  auto column =
      buildColumnPageReadPlan(1, offset, retained, std::nullopt, 100, 60);

  EXPECT_FALSE(column.useWholeChunkStream);
  EXPECT_FALSE(column.costModelFallback);
  EXPECT_EQ(column.plannedPhysicalBytes, 50);
}

TEST(PagePruningPlanTest, modelsPhysicalGapCoalescingAndCostFallback) {
  ValidatedOffsetIndex offset;
  offset.pages = {
      {100, 10, 0, 10},
      {120, 10, 10, 10},
      {1'000, 10, 20, 10},
  };
  RowIntervalSet retained;
  retained.add({0, 10});
  retained.add({20, 30});

  const PagePruningCostModelOptions coalescing{
      .maxCoalesceDistance = 900,
      .maxCoalesceBytes = 1'000,
      .loadQuantum = 100,
  };
  auto coalesced = buildColumnPageReadPlan(
      1, offset, retained, std::nullopt, 1'000, 0, false, coalescing);
  EXPECT_EQ(coalesced.retainedRuns.size(), 2);
  EXPECT_EQ(coalesced.plannedPhysicalBytes, 910);
  EXPECT_EQ(coalesced.plannedPhysicalLoads, 10);
  EXPECT_FALSE(coalesced.useWholeChunkStream);

  const PagePruningCostModelOptions split{
      .maxCoalesceDistance = 10,
      .maxCoalesceBytes = 1'000,
      .loadQuantum = 100,
  };
  auto splitRuns = buildColumnPageReadPlan(
      1, offset, retained, std::nullopt, 1'000, 0, false, split);
  EXPECT_EQ(splitRuns.plannedPhysicalBytes, 20);
  EXPECT_EQ(splitRuns.plannedPhysicalLoads, 2);

  auto fallback = buildColumnPageReadPlan(
      1, offset, retained, std::nullopt, 20, 0, false, coalescing);
  EXPECT_TRUE(fallback.useWholeChunkStream);
  EXPECT_TRUE(fallback.costModelFallback);
}

TEST(PagePruningPlanTest, representsAllPagesRejectedExplicitly) {
  ValidatedOffsetIndex offset;
  offset.pages = {
      {100, 50, 0, 10},
      {150, 50, 10, 10},
  };
  RowIntervalSet retained;

  auto column = buildColumnPageReadPlan(
      1, offset, retained, std::nullopt, 100, 20, false);

  EXPECT_TRUE(column.allPagesSkipped);
  EXPECT_FALSE(column.useWholeChunkStream);
  EXPECT_EQ(column.retainedRuns.size(), 0);
  EXPECT_EQ(column.dataPageToRun, (std::vector<int32_t>{-1, -1}));
}

TEST(PagePlanMemoryBudgetTest, enforcesLimitAndSharedLifetime) {
  MemoryManager manager(MemoryManager::Options{});
  auto root = manager.addRootPool("pagePlanBudget");
  auto pool = root->addLeafChild("pagePlanBudget");
  const auto initialUsage = pool->usedBytes();
  PagePlanMemoryBudget budget(*pool, 100);

  auto first = budget.tryReserve(60);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(budget.usedBytes(), 60);
  auto atLimit = budget.tryReserve(40);
  ASSERT_NE(atLimit, nullptr);
  EXPECT_EQ(budget.usedBytes(), 100);
  EXPECT_FALSE(budget.tryReserve(1));

  auto second = first;
  first.reset();
  EXPECT_EQ(budget.usedBytes(), 100);
  atLimit.reset();
  EXPECT_EQ(budget.usedBytes(), 60);
  second.reset();
  EXPECT_EQ(budget.usedBytes(), 0);
  EXPECT_EQ(pool->usedBytes(), initialUsage);
}

TEST(PagePlanMemoryBudgetTest, releasesAfterPlanReferenceAndMapEviction) {
  MemoryManager manager(MemoryManager::Options{});
  auto root = manager.addRootPool("pagePlanEviction");
  auto pool = root->addLeafChild("pagePlanEviction");
  const auto initialUsage = pool->usedBytes();
  auto budget = std::make_shared<PagePlanMemoryBudget>(*pool, 100);
  auto reservation = budget->tryReserve(60);
  ASSERT_NE(reservation, nullptr);

  auto plan = std::make_shared<RowGroupPagePruningPlan>();
  plan->decodedPageIndexBytes = 60;
  plan->decodedPageIndexMemory = reservation;
  reservation.reset();
  std::unordered_map<uint32_t, RowGroupPagePruningPlanPtr> plans;
  plans.emplace(0, plan);
  plans.erase(0);
  EXPECT_EQ(budget->usedBytes(), 60);
  plan.reset();
  EXPECT_EQ(budget->usedBytes(), 0);
  EXPECT_EQ(pool->usedBytes(), initialUsage);
}

TEST(PagePlanMemoryBudgetTest, poolFailureRollsBackCounters) {
  MemoryManager::Options options;
  options.allocatorCapacity = 64;
  options.arbitratorCapacity = 64;
  MemoryManager manager(options);
  auto root = manager.addRootPool("pagePlanFailure", 64);
  auto pool = root->addLeafChild("pagePlanFailure");
  PagePlanMemoryBudget budget(*pool, 128);
  const auto initialStats = pool->stats();

  VELOX_ASSERT_THROW(budget.tryReserve(65), "Exceeded memory pool capacity");
  EXPECT_EQ(budget.usedBytes(), 0);
  EXPECT_EQ(pool->usedBytes(), 0);
  const auto finalStats = pool->stats();
  EXPECT_EQ(finalStats.numExternalAllocs, initialStats.numExternalAllocs);
  EXPECT_EQ(finalStats.numExternalFrees, initialStats.numExternalFrees);
}

TEST(PagePlanMemoryBudgetTest, reservationIsNotCopyable) {
  EXPECT_FALSE(std::is_copy_constructible_v<PagePlanMemoryReservation>);
  EXPECT_FALSE(std::is_copy_assignable_v<PagePlanMemoryReservation>);
  EXPECT_FALSE(std::is_move_constructible_v<PagePlanMemoryReservation>);
  EXPECT_FALSE(std::is_move_assignable_v<PagePlanMemoryReservation>);
}
