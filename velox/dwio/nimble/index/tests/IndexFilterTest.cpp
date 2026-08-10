/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/IndexFilter.h"
#include "velox/type/Filter.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index::test {

using namespace velox;
using namespace velox::common;

class IndexFilterTestBase : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::initialize({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("IndexFilterTest");
  }

  std::unique_ptr<ScanSpec> createScanSpec(
      const RowTypePtr& rowType,
      const std::unordered_map<std::string, std::unique_ptr<Filter>>& filters) {
    auto scanSpec = std::make_unique<ScanSpec>("root");
    for (size_t i = 0; i < rowType->size(); ++i) {
      auto* childSpec = scanSpec->addField(
          rowType->nameOf(i), static_cast<column_index_t>(i));
      auto it = filters.find(rowType->nameOf(i));
      if (it != filters.end()) {
        childSpec->setFilter(it->second->clone());
      }
    }
    return scanSpec;
  }

  template <typename T>
  void verifyBound(
      const RowVectorPtr& bound,
      const std::string& columnName,
      T expectedValue) {
    const auto& rowType = bound->type()->asRow();
    auto idx = rowType.getChildIdx(columnName);
    auto* flatVector = bound->childAt(idx)->as<FlatVector<T>>();
    ASSERT_NE(flatVector, nullptr);
    EXPECT_EQ(flatVector->valueAt(0), expectedValue);
  }

  void verifyNullBound(
      const RowVectorPtr& bound,
      const std::string& columnName) {
    const auto& rowType = bound->type()->asRow();
    auto idx = rowType.getChildIdx(columnName);
    EXPECT_TRUE(bound->childAt(idx)->isNullAt(0));
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

class IndexFilterTest : public IndexFilterTestBase {};

TEST_F(IndexFilterTest, basic) {
  auto rowType = ROW({"a", "b"}, {BIGINT(), BIGINT()});

  {
    // Empty index columns.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    NIMBLE_ASSERT_THROW(
        convertFilterToIndexBounds({}, {}, rowType, *scanSpec, pool_.get()),
        "Index columns cannot be empty");
  }

  {
    // Index columns size mismatch with sort orders size.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    NIMBLE_ASSERT_THROW(
        convertFilterToIndexBounds(
            {"a", "b"},
            {SortOrder{.ascending = true}},
            rowType,
            *scanSpec,
            pool_.get()),
        "Index columns size must match sort orders size");
  }

  {
    // Index column not in rowType - returns nullopt since there's no filter.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"nonexistent"},
        {SortOrder{.ascending = true}},
        rowType,
        *scanSpec,
        pool_.get());
    EXPECT_FALSE(result.has_value());
  }
}

TEST_F(IndexFilterTest, noMatchingFilter) {
  auto rowType = ROW({"a", "b"}, {BIGINT(), BIGINT()});

  {
    // No filter on index column "a".
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["b"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
    // Filter on "b" should still be set.
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Unsupported filter type: IsNull.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<IsNull>();
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Unsupported filter type: IsNotNull.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<IsNotNull>();
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Unsupported filter type: BigintValuesUsingHashTable.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    // Use values with large spread to force hash table instead of bitmask.
    filters["a"] = common::createBigintValues({1, 1000, 10000, 100000}, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Unsupported filter type: NegatedBigintRange.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<NegatedBigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Filter allows nulls - should not be convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 100, true);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Gap in filter chain: filter on "a" and "c", but not "b".
    // Extraction stops at missing filter on "b", so only "a" is converted.
    auto threeColRowType = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), BIGINT()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["c"] = std::make_unique<BigintRange>(100, 100, false);
    auto scanSpec = createScanSpec(threeColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"},
        {SortOrder{.ascending = true},
         SortOrder{.ascending = true},
         SortOrder{.ascending = true}},
        threeColRowType,
        *scanSpec,
        pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only "a" should be included - stops at missing filter on "b".
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);

    // Filter on "a" removed, but "c" should still have filter since it wasn't
    // converted (due to gap at "b").
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("c")->filter(), nullptr);
  }
}

TEST_F(IndexFilterTest, removedFilters) {
  // Test that removed filters are returned correctly.
  const auto rowType = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), BIGINT()});

  {
    // Single column filter - one filter removed.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"},
        {SortOrder{.ascending = true},
         SortOrder{.ascending = true},
         SortOrder{.ascending = true}},
        rowType,
        *scanSpec,
        pool_.get());
    ASSERT_TRUE(result.has_value());

    // Verify removed filters.
    ASSERT_EQ(result->removedFilters.size(), 1);
    EXPECT_EQ(result->removedFilters[0].first, "a");
    ASSERT_NE(result->removedFilters[0].second, nullptr);
    EXPECT_EQ(
        result->removedFilters[0].second->kind(), FilterKind::kBigintRange);
  }

  {
    // Two column filters - two filters removed.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] = std::make_unique<BigintRange>(100, 100, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"},
        {SortOrder{.ascending = true},
         SortOrder{.ascending = true},
         SortOrder{.ascending = true}},
        rowType,
        *scanSpec,
        pool_.get());
    ASSERT_TRUE(result.has_value());

    // Verify removed filters.
    ASSERT_EQ(result->removedFilters.size(), 2);
    EXPECT_EQ(result->removedFilters[0].first, "a");
    EXPECT_EQ(
        result->removedFilters[0].second->kind(), FilterKind::kBigintRange);
    EXPECT_EQ(result->removedFilters[1].first, "b");
    EXPECT_EQ(
        result->removedFilters[1].second->kind(), FilterKind::kBigintRange);
  }

  {
    // Gap in filter chain - only first filter removed.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["c"] = std::make_unique<BigintRange>(100, 100, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"},
        {SortOrder{.ascending = true},
         SortOrder{.ascending = true},
         SortOrder{.ascending = true}},
        rowType,
        *scanSpec,
        pool_.get());
    ASSERT_TRUE(result.has_value());

    // Only "a" should be removed due to gap at "b".
    ASSERT_EQ(result->removedFilters.size(), 1);
    EXPECT_EQ(result->removedFilters[0].first, "a");
    EXPECT_EQ(
        result->removedFilters[0].second->kind(), FilterKind::kBigintRange);
  }

  {
    // Range + range - only first filter removed (range stops extraction).
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 100, false);
    filters["b"] = std::make_unique<BigintRange>(200, 300, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"},
        {SortOrder{.ascending = true}, SortOrder{.ascending = true}},
        rowType,
        *scanSpec,
        pool_.get());
    ASSERT_TRUE(result.has_value());

    // Only "a" should be removed because it's a range scan.
    ASSERT_EQ(result->removedFilters.size(), 1);
    EXPECT_EQ(result->removedFilters[0].first, "a");
    EXPECT_EQ(
        result->removedFilters[0].second->kind(), FilterKind::kBigintRange);
  }

  {
    // No matching filter - no result, no removed filters.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["b"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
  }

  {
    // Unsupported filter - no result, no removed filters.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<IsNull>();
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, {SortOrder{.ascending = true}}, rowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
  }
}

TEST_F(IndexFilterTest, sortOrder) {
  // Test with different sort orders for different index columns.
  // Point lookup on first two columns, range on third - all three columns
  // should be included in the index bounds.
  // Loop over the last column's sort order (ASC or DESC).
  auto rowType = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), BIGINT()});

  std::vector<SortOrder> sortOrdersToTest = {
      SortOrder{.ascending = true}, SortOrder{.ascending = false}};
  for (const auto& lastColSortOrder : sortOrdersToTest) {
    SCOPED_TRACE(
        fmt::format(
            "lastColSortOrder: {}",
            lastColSortOrder.ascending ? "ASC" : "DESC"));

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] = std::make_unique<BigintRange>(100, 100, false);
    filters["c"] = std::make_unique<BigintRange>(10, 20, false);
    auto scanSpec = createScanSpec(rowType, filters);

    // Mixed sort orders: asc, desc, and lastColSortOrder for "c".
    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"},
        {SortOrder{.ascending = true},
         SortOrder{.ascending = false},
         lastColSortOrder},
        rowType,
        *scanSpec,
        pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // All three columns should be included: point lookup on "a" and "b",
    // range on "c".
    EXPECT_EQ(bounds.indexColumns.size(), 3);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");
    EXPECT_EQ(bounds.indexColumns[2], "c");

    // Point lookup on "a": same value for both bounds.
    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);
    // Point lookup on "b": same value for both bounds.
    verifyBound<int64_t>(bounds.lowerBound->bound, "b", 100);
    verifyBound<int64_t>(bounds.upperBound->bound, "b", 100);
    // Range on "c": bounds swapped for descending sort order.
    if (lastColSortOrder.ascending) {
      verifyBound<int64_t>(bounds.lowerBound->bound, "c", 10);
      verifyBound<int64_t>(bounds.upperBound->bound, "c", 20);
    } else {
      verifyBound<int64_t>(bounds.lowerBound->bound, "c", 20);
      verifyBound<int64_t>(bounds.upperBound->bound, "c", 10);
    }

    // All filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("c")->filter(), nullptr);
  }
}

class IndexFilterTestWithSortOrder
    : public IndexFilterTestBase,
      public ::testing::WithParamInterface<SortOrder> {
 protected:
  SortOrder sortOrder() const {
    return GetParam();
  }

  std::vector<SortOrder> sortOrders(size_t count) const {
    return std::vector<SortOrder>(count, sortOrder());
  }

  bool isAscending() const {
    return sortOrder().ascending;
  }

  // For range filters, lower/upper bounds are swapped for descending sort.
  template <typename T>
  void verifyRangeBounds(
      const velox::serializer::IndexBounds& bounds,
      const std::string& columnName,
      T lower,
      T upper) {
    if (isAscending()) {
      verifyBound<T>(bounds.lowerBound->bound, columnName, lower);
      verifyBound<T>(bounds.upperBound->bound, columnName, upper);
    } else {
      verifyBound<T>(bounds.lowerBound->bound, columnName, upper);
      verifyBound<T>(bounds.upperBound->bound, columnName, lower);
    }
  }
};

TEST_P(IndexFilterTestWithSortOrder, bigintFilter) {
  const auto rowType = ROW({"a", "b"}, {BIGINT(), BIGINT()});

  {
    // Prefix point lookup: single column point lookup on the first column.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    // Point lookup: both bounds have same value regardless of sort order.
    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Prefix range: range filter on the first index column only.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 100, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    verifyRangeBounds<int64_t>(bounds, "a", 10, 100);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Point lookup + range: point lookup on first column, range on second,
    // spanning two index columns.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] = std::make_unique<BigintRange>(10, 100, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookup on "a": same value regardless of sort order.
    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);
    // Range on "b": bounds swapped for descending sort.
    verifyRangeBounds<int64_t>(bounds, "b", 10, 100);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Range + range: range filter on the first column stops extraction,
    // so only the first column is converted, not the second.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 100, false);
    filters["b"] = std::make_unique<BigintRange>(200, 300, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only column "a" should be included because it's a range scan.
    // The range filter stops extraction from including subsequent columns.
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyRangeBounds<int64_t>(bounds, "a", 10, 100);

    // Filter on "a" removed, but "b" should still have filter since it wasn't
    // converted.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Point lookup + point lookup: point lookups on both index columns.
    // Both columns should be included since point lookups allow extraction to
    // continue.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] = std::make_unique<BigintRange>(100, 100, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookups: same value for both bounds regardless of sort order.
    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.lowerBound->bound, "b", 100);
    verifyBound<int64_t>(bounds.upperBound->bound, "b", 100);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }
}

// Tests for different integer types (TINYINT, SMALLINT, INTEGER) to verify
// the INTEGER_TYPE_DISPATCH macro works correctly with all integer types.
TEST_P(IndexFilterTestWithSortOrder, integerTypesFilter) {
  // Local template function to test a single integer type with range filter.
  auto testIntegerType =
      [this]<typename T>(const TypePtr& type, int64_t lower, int64_t upper) {
        const auto rowType = ROW({"a"}, {type});
        std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
        filters["a"] = std::make_unique<BigintRange>(lower, upper, false);
        auto scanSpec = createScanSpec(rowType, filters);

        auto result = convertFilterToIndexBounds(
            {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
        ASSERT_TRUE(result.has_value());

        const auto& bounds = result->indexBounds;
        EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
        verifyRangeBounds<T>(bounds, "a", lower, upper);
        EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
      };

  // Test all integer types.
  testIntegerType.template operator()<int8_t>(TINYINT(), 10, 20);
  testIntegerType.template operator()<int16_t>(SMALLINT(), 1000, 2000);
  testIntegerType.template operator()<int32_t>(INTEGER(), 100000, 200000);
  testIntegerType.template operator()<int64_t>(BIGINT(), 1000000, 2000000);

  // Local template function to test mixed integer types: point lookup on first
  // column + range on second column.
  auto testMixedIntegerTypes = [this]<typename T1, typename T2>(
                                   const TypePtr& type1,
                                   int64_t pointValue,
                                   const TypePtr& type2,
                                   int64_t lower,
                                   int64_t upper) {
    const auto rowType = ROW({"a", "b"}, {type1, type2});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(pointValue, pointValue, false);
    filters["b"] = std::make_unique<BigintRange>(lower, upper, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookup on "a".
    verifyBound<T1>(bounds.lowerBound->bound, "a", pointValue);
    verifyBound<T1>(bounds.upperBound->bound, "a", pointValue);
    // Range on "b".
    verifyRangeBounds<T2>(bounds, "b", lower, upper);

    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  };

  // Test mixed integer types: TINYINT point lookup + INTEGER range.
  testMixedIntegerTypes.template operator()<int8_t, int32_t>(
      TINYINT(), 42, INTEGER(), 10000, 20000);
  // Test SMALLINT + BIGINT combination.
  testMixedIntegerTypes.template operator()<int16_t, int64_t>(
      SMALLINT(), 100, BIGINT(), 1000000000, 2000000000);
}

TEST_P(IndexFilterTestWithSortOrder, doubleFilter) {
  const auto rowType = ROW({"a", "b"}, {DOUBLE(), DOUBLE()});

  {
    // Prefix point lookup: single column point lookup on the first column.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, false, 1.5, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    // Point lookup: both bounds have same value regardless of sort order.
    verifyBound<double>(bounds.lowerBound->bound, "a", 1.5);
    verifyBound<double>(bounds.upperBound->bound, "a", 1.5);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Prefix range: range filter on the first index column only.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, false, 10.5, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    verifyRangeBounds<double>(bounds, "a", 1.5, 10.5);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Point lookup + range: point lookup on first column, range on second,
    // spanning two index columns.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, false, 1.5, false, false, false);
    filters["b"] = std::make_unique<DoubleRange>(
        10.5, false, false, 100.5, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookup on "a": same value regardless of sort order.
    verifyBound<double>(bounds.lowerBound->bound, "a", 1.5);
    verifyBound<double>(bounds.upperBound->bound, "a", 1.5);
    // Range on "b": bounds swapped for descending sort.
    verifyRangeBounds<double>(bounds, "b", 10.5, 100.5);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Range + range: range filter on the first column stops extraction,
    // so only the first column is converted, not the second.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, false, 10.5, false, false, false);
    filters["b"] = std::make_unique<DoubleRange>(
        20.5, false, false, 30.5, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only column "a" should be included because it's a range scan.
    // The range filter stops extraction from including subsequent columns.
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyRangeBounds<double>(bounds, "a", 1.5, 10.5);

    // Filter on "a" removed, but "b" should still have filter since it wasn't
    // converted.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Point lookup + point lookup: point lookups on both index columns.
    // Both columns should be included since point lookups allow extraction to
    // continue.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, false, 1.5, false, false, false);
    filters["b"] = std::make_unique<DoubleRange>(
        10.5, false, false, 10.5, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookups: same value for both bounds regardless of sort order.
    verifyBound<double>(bounds.lowerBound->bound, "a", 1.5);
    verifyBound<double>(bounds.upperBound->bound, "a", 1.5);
    verifyBound<double>(bounds.lowerBound->bound, "b", 10.5);
    verifyBound<double>(bounds.upperBound->bound, "b", 10.5);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Exclusive bounds.
    auto singleColRowType = ROW({"a"}, {DOUBLE()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, true, 10.5, false, true, false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_FALSE(bounds.lowerBound->inclusive);
    EXPECT_FALSE(bounds.upperBound->inclusive);
    verifyRangeBounds<double>(bounds, "a", 1.5, 10.5);
  }

  {
    // Lower unbounded: -inf to 10.5 (inclusive).
    auto singleColRowType = ROW({"a"}, {DOUBLE()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        std::numeric_limits<double>::lowest(),
        true,
        false,
        10.5,
        false,
        false,
        false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    if (isAscending()) {
      verifyNullBound(bounds.lowerBound->bound, "a");
      verifyBound<double>(bounds.upperBound->bound, "a", 10.5);
    } else {
      verifyBound<double>(bounds.lowerBound->bound, "a", 10.5);
      verifyNullBound(bounds.upperBound->bound, "a");
    }
  }

  {
    // Upper unbounded: 1.5 (inclusive) to +inf.
    auto singleColRowType = ROW({"a"}, {DOUBLE()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5,
        false,
        false,
        std::numeric_limits<double>::max(),
        true,
        false,
        false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    if (isAscending()) {
      verifyBound<double>(bounds.lowerBound->bound, "a", 1.5);
      verifyNullBound(bounds.upperBound->bound, "a");
    } else {
      verifyNullBound(bounds.lowerBound->bound, "a");
      verifyBound<double>(bounds.upperBound->bound, "a", 1.5);
    }
  }
}

TEST_P(IndexFilterTestWithSortOrder, floatFilter) {
  const auto rowType = ROW({"a", "b"}, {REAL(), REAL()});

  {
    // Prefix point lookup: single column point lookup on the first column.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, false, 1.5f, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    // Point lookup: both bounds have same value regardless of sort order.
    verifyBound<float>(bounds.lowerBound->bound, "a", 1.5f);
    verifyBound<float>(bounds.upperBound->bound, "a", 1.5f);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Prefix range: range filter on the first index column only.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, false, 10.5f, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    verifyRangeBounds<float>(bounds, "a", 1.5f, 10.5f);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Point lookup + range: point lookup on first column, range on second,
    // spanning two index columns.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, false, 1.5f, false, false, false);
    filters["b"] = std::make_unique<FloatRange>(
        10.5f, false, false, 100.5f, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookup on "a": same value regardless of sort order.
    verifyBound<float>(bounds.lowerBound->bound, "a", 1.5f);
    verifyBound<float>(bounds.upperBound->bound, "a", 1.5f);
    // Range on "b": bounds swapped for descending sort.
    verifyRangeBounds<float>(bounds, "b", 10.5f, 100.5f);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Range + range: range filter on the first column stops extraction,
    // so only the first column is converted, not the second.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, false, 10.5f, false, false, false);
    filters["b"] = std::make_unique<FloatRange>(
        20.5f, false, false, 30.5f, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only column "a" should be included because it's a range scan.
    // The range filter stops extraction from including subsequent columns.
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyRangeBounds<float>(bounds, "a", 1.5f, 10.5f);

    // Filter on "a" removed, but "b" should still have filter since it wasn't
    // converted.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Point lookup + point lookup: point lookups on both index columns.
    // Both columns should be included since point lookups allow extraction to
    // continue.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, false, 1.5f, false, false, false);
    filters["b"] = std::make_unique<FloatRange>(
        10.5f, false, false, 10.5f, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookups: same value for both bounds regardless of sort order.
    verifyBound<float>(bounds.lowerBound->bound, "a", 1.5f);
    verifyBound<float>(bounds.upperBound->bound, "a", 1.5f);
    verifyBound<float>(bounds.lowerBound->bound, "b", 10.5f);
    verifyBound<float>(bounds.upperBound->bound, "b", 10.5f);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Exclusive bounds.
    auto singleColRowType = ROW({"a"}, {REAL()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, true, 10.5f, false, true, false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_FALSE(bounds.lowerBound->inclusive);
    EXPECT_FALSE(bounds.upperBound->inclusive);
    verifyRangeBounds<float>(bounds, "a", 1.5f, 10.5f);
  }

  {
    // Lower unbounded: -inf to 10.5 (inclusive).
    auto singleColRowType = ROW({"a"}, {REAL()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        std::numeric_limits<float>::lowest(),
        true,
        false,
        10.5f,
        false,
        false,
        false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    if (isAscending()) {
      verifyNullBound(bounds.lowerBound->bound, "a");
      verifyBound<float>(bounds.upperBound->bound, "a", 10.5f);
    } else {
      verifyBound<float>(bounds.lowerBound->bound, "a", 10.5f);
      verifyNullBound(bounds.upperBound->bound, "a");
    }
  }

  {
    // Upper unbounded: 1.5 (inclusive) to +inf.
    auto singleColRowType = ROW({"a"}, {REAL()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f,
        false,
        false,
        std::numeric_limits<float>::max(),
        true,
        false,
        false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    if (isAscending()) {
      verifyBound<float>(bounds.lowerBound->bound, "a", 1.5f);
      verifyNullBound(bounds.upperBound->bound, "a");
    } else {
      verifyNullBound(bounds.lowerBound->bound, "a");
      verifyBound<float>(bounds.upperBound->bound, "a", 1.5f);
    }
  }
}

// Test VARCHAR (bytes) filters. For ascending order, BytesValues (single value)
// and BytesRange filters are convertible. For descending order, VARCHAR filters
// are NOT convertible.
TEST_P(IndexFilterTestWithSortOrder, bytesFilter) {
  const auto rowType = ROW({"a", "b"}, {VARCHAR(), VARCHAR()});
  const bool ascending = isAscending();

  {
    // Prefix point lookup using BytesValues with single value.
    // Ascending: convertible. Descending: not convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] =
        std::make_unique<BytesValues>(std::vector<std::string>{"hello"}, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
      verifyBound<StringView>(
          bounds.lowerBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "a", StringView("hello"));
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
    }
  }

  {
    // Prefix point lookup using BytesRange with same lower and upper.
    // Ascending: convertible. Descending: not convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesRange>(
        "hello", false, false, "hello", false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
      verifyBound<StringView>(
          bounds.lowerBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "a", StringView("hello"));
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
    }
  }

  {
    // Prefix range: range filter on the first index column only.
    // Ascending: convertible. Descending: not convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesRange>(
        "abc", false, false, "xyz", false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
      verifyBound<StringView>(bounds.lowerBound->bound, "a", StringView("abc"));
      verifyBound<StringView>(bounds.upperBound->bound, "a", StringView("xyz"));
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
    }
  }

  {
    // Point lookup + range: point lookup on first column (BytesValues),
    // range on second (BytesRange), spanning two index columns.
    // Ascending: both convertible. Descending: not convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] =
        std::make_unique<BytesValues>(std::vector<std::string>{"hello"}, false);
    filters["b"] = std::make_unique<BytesRange>(
        "abc", false, false, "xyz", false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns.size(), 2);
      verifyBound<StringView>(
          bounds.lowerBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(bounds.lowerBound->bound, "b", StringView("abc"));
      verifyBound<StringView>(bounds.upperBound->bound, "b", StringView("xyz"));
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    }
  }

  {
    // Range + range: range filter on the first column stops extraction.
    // Ascending: only first column convertible. Descending: not convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesRange>(
        "aaa", false, false, "bbb", false, false, false);
    filters["b"] = std::make_unique<BytesRange>(
        "ccc", false, false, "ddd", false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns.size(), 1);
      verifyBound<StringView>(bounds.lowerBound->bound, "a", StringView("aaa"));
      verifyBound<StringView>(bounds.upperBound->bound, "a", StringView("bbb"));
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    }
  }

  {
    // Point lookup + point lookup: point lookups on both index columns.
    // Ascending: both convertible. Descending: not convertible.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] =
        std::make_unique<BytesValues>(std::vector<std::string>{"hello"}, false);
    filters["b"] =
        std::make_unique<BytesValues>(std::vector<std::string>{"world"}, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns.size(), 2);
      verifyBound<StringView>(
          bounds.lowerBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(
          bounds.lowerBound->bound, "b", StringView("world"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "b", StringView("world"));
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    }
  }

  {
    // Exclusive bounds.
    // Ascending: convertible. Descending: not convertible.
    auto singleColRowType = ROW({"a"}, {VARCHAR()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesRange>(
        "abc", false, true, "xyz", false, true, false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_FALSE(bounds.lowerBound->inclusive);
      EXPECT_FALSE(bounds.upperBound->inclusive);
      verifyBound<StringView>(bounds.lowerBound->bound, "a", StringView("abc"));
      verifyBound<StringView>(bounds.upperBound->bound, "a", StringView("xyz"));
    }
  }

  {
    // Lower unbounded: empty string to "xyz" (inclusive).
    // Ascending: convertible. Descending: not convertible.
    auto singleColRowType = ROW({"a"}, {VARCHAR()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesRange>(
        "", true, false, "xyz", false, false, false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      verifyNullBound(bounds.lowerBound->bound, "a");
      verifyBound<StringView>(bounds.upperBound->bound, "a", StringView("xyz"));
    }
  }

  {
    // Upper unbounded: "abc" (inclusive) to unbounded.
    // Ascending: convertible. Descending: not convertible.
    auto singleColRowType = ROW({"a"}, {VARCHAR()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesRange>(
        "abc", false, false, "", true, false, false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      verifyBound<StringView>(bounds.lowerBound->bound, "a", StringView("abc"));
      verifyNullBound(bounds.upperBound->bound, "a");
    }
  }

  {
    // BytesValues with multiple values - should not be convertible for any
    // sort order.
    auto singleColRowType = ROW({"a"}, {VARCHAR()});
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BytesValues>(
        std::vector<std::string>{"hello", "world"}, false);
    auto scanSpec = createScanSpec(singleColRowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), singleColRowType, *scanSpec, pool_.get());
    EXPECT_FALSE(result.has_value());
  }
}

TEST_P(IndexFilterTestWithSortOrder, boolValueFilter) {
  {
    // Single column point lookup.
    auto rowType = ROW({"a"}, {BOOLEAN()});

    for (bool value : {true, false}) {
      SCOPED_TRACE(fmt::format("BoolValue: {}", value));

      std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
      filters["a"] = std::make_unique<BoolValue>(value, false);
      auto scanSpec = createScanSpec(rowType, filters);

      auto result = convertFilterToIndexBounds(
          {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
      ASSERT_TRUE(result.has_value());

      const auto& bounds = result->indexBounds;
      verifyBound<bool>(bounds.lowerBound->bound, "a", value);
      verifyBound<bool>(bounds.upperBound->bound, "a", value);
    }
  }

  {
    // Point lookup + point lookup: point lookups on both index columns.
    // Both columns should be included since point lookups allow extraction to
    // continue.
    auto rowType = ROW({"a", "b"}, {BOOLEAN(), BOOLEAN()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BoolValue>(true, false);
    filters["b"] = std::make_unique<BoolValue>(false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookups: same value for both bounds regardless of sort order.
    verifyBound<bool>(bounds.lowerBound->bound, "a", true);
    verifyBound<bool>(bounds.upperBound->bound, "a", true);
    verifyBound<bool>(bounds.lowerBound->bound, "b", false);
    verifyBound<bool>(bounds.upperBound->bound, "b", false);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Point lookup (bigint) + point lookup (bool): mixed types, both point
    // lookups so both columns should be included.
    auto rowType = ROW({"a", "b"}, {BIGINT(), BOOLEAN()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] = std::make_unique<BoolValue>(true, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookups: same value for both bounds regardless of sort order.
    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);
    verifyBound<bool>(bounds.lowerBound->bound, "b", true);
    verifyBound<bool>(bounds.upperBound->bound, "b", true);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Range (bigint) + point lookup (bool): range filter on first column stops
    // extraction, so only the first column is converted.
    auto rowType = ROW({"a", "b"}, {BIGINT(), BOOLEAN()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(10, 100, false);
    filters["b"] = std::make_unique<BoolValue>(true, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only column "a" should be included because it's a range scan.
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyRangeBounds<int64_t>(bounds, "a", 10, 100);

    // Filter on "a" removed, but "b" should still have filter since it wasn't
    // converted.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }
}

TEST_P(IndexFilterTestWithSortOrder, timestampFilter) {
  const auto rowType = ROW({"a", "b"}, {TIMESTAMP(), TIMESTAMP()});
  Timestamp lower(100, 0);
  Timestamp upper(200, 0);

  {
    // Prefix point lookup: single column point lookup on the first column.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<TimestampRange>(lower, lower, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    // Point lookup: both bounds have same value regardless of sort order.
    verifyBound<Timestamp>(bounds.lowerBound->bound, "a", lower);
    verifyBound<Timestamp>(bounds.upperBound->bound, "a", lower);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Prefix range: range filter on the first index column only.
    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<TimestampRange>(lower, upper, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a"}, sortOrders(1), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns, std::vector<std::string>{"a"});
    ASSERT_TRUE(bounds.lowerBound.has_value());
    ASSERT_TRUE(bounds.upperBound.has_value());
    EXPECT_TRUE(bounds.lowerBound->inclusive);
    EXPECT_TRUE(bounds.upperBound->inclusive);

    verifyRangeBounds<Timestamp>(bounds, "a", lower, upper);

    // Filter should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
  }

  {
    // Point lookup + range: point lookup on first column, range on second,
    // spanning two index columns.
    Timestamp lower2(300, 0);
    Timestamp upper2(400, 0);

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<TimestampRange>(lower, lower, false);
    filters["b"] = std::make_unique<TimestampRange>(lower2, upper2, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookup on "a": same value regardless of sort order.
    verifyBound<Timestamp>(bounds.lowerBound->bound, "a", lower);
    verifyBound<Timestamp>(bounds.upperBound->bound, "a", lower);
    // Range on "b": bounds swapped for descending sort.
    verifyRangeBounds<Timestamp>(bounds, "b", lower2, upper2);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Range + range: range filter on the first column stops extraction,
    // so only the first column is converted, not the second.
    Timestamp lower2(300, 0);
    Timestamp upper2(400, 0);

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<TimestampRange>(lower, upper, false);
    filters["b"] = std::make_unique<TimestampRange>(lower2, upper2, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only column "a" should be included because it's a range scan.
    // The range filter stops extraction from including subsequent columns.
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyRangeBounds<Timestamp>(bounds, "a", lower, upper);

    // Filter on "a" removed, but "b" should still have filter since it wasn't
    // converted.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Point lookup + point lookup: point lookups on both index columns.
    // Both columns should be included since point lookups allow extraction to
    // continue.
    Timestamp value2(300, 0);

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<TimestampRange>(lower, lower, false);
    filters["b"] = std::make_unique<TimestampRange>(value2, value2, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    // Point lookups: same value for both bounds regardless of sort order.
    verifyBound<Timestamp>(bounds.lowerBound->bound, "a", lower);
    verifyBound<Timestamp>(bounds.upperBound->bound, "a", lower);
    verifyBound<Timestamp>(bounds.lowerBound->bound, "b", value2);
    verifyBound<Timestamp>(bounds.upperBound->bound, "b", value2);

    // Both filters should be removed from scanSpec.
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }
}

// Test mixed filter data types. For ascending order, all types are convertible.
// For descending order, VARCHAR filters are NOT convertible.
TEST_P(IndexFilterTestWithSortOrder, mixedFilterDataTypes) {
  const bool ascending = isAscending();

  {
    // Bigint point lookup + varchar point lookup.
    // Ascending: both convertible. Descending: only bigint convertible.
    auto rowType = ROW({"a", "b"}, {BIGINT(), VARCHAR()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] =
        std::make_unique<BytesValues>(std::vector<std::string>{"hello"}, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    const size_t expectedColumns = ascending ? 2 : 1;
    EXPECT_EQ(bounds.indexColumns.size(), expectedColumns);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);

    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    if (ascending) {
      EXPECT_EQ(bounds.indexColumns[1], "b");
      verifyBound<StringView>(
          bounds.lowerBound->bound, "b", StringView("hello"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "b", StringView("hello"));
      EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    }
  }

  {
    // Double point lookup + bool point lookup - works for both sort orders.
    auto rowType = ROW({"a", "b"}, {DOUBLE(), BOOLEAN()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<DoubleRange>(
        1.5, false, false, 1.5, false, false, false);
    filters["b"] = std::make_unique<BoolValue>(true, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 2);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");

    verifyBound<double>(bounds.lowerBound->bound, "a", 1.5);
    verifyBound<double>(bounds.upperBound->bound, "a", 1.5);
    verifyBound<bool>(bounds.lowerBound->bound, "b", true);
    verifyBound<bool>(bounds.upperBound->bound, "b", true);

    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Varchar point lookup + timestamp range.
    // Ascending: both convertible. Descending: not convertible (VARCHAR first).
    auto rowType = ROW({"a", "b"}, {VARCHAR(), TIMESTAMP()});
    Timestamp lower(100, 0);
    Timestamp upper(200, 0);

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] =
        std::make_unique<BytesValues>(std::vector<std::string>{"hello"}, false);
    filters["b"] = std::make_unique<TimestampRange>(lower, upper, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    EXPECT_EQ(result.has_value(), ascending);
    if (ascending) {
      const auto& bounds = result->indexBounds;
      EXPECT_EQ(bounds.indexColumns.size(), 2);
      verifyBound<StringView>(
          bounds.lowerBound->bound, "a", StringView("hello"));
      verifyBound<StringView>(
          bounds.upperBound->bound, "a", StringView("hello"));
      verifyRangeBounds<Timestamp>(bounds, "b", lower, upper);
      EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("a")->filter(), nullptr);
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    }
  }

  {
    // Float range + bigint point lookup: range on first column stops
    // extraction - works for both sort orders.
    auto rowType = ROW({"a", "b"}, {REAL(), BIGINT()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<FloatRange>(
        1.5f, false, false, 10.5f, false, false, false);
    filters["b"] = std::make_unique<BigintRange>(42, 42, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b"}, sortOrders(2), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    // Only "a" because it's a range filter.
    EXPECT_EQ(bounds.indexColumns.size(), 1);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyRangeBounds<float>(bounds, "a", 1.5f, 10.5f);

    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
  }

  {
    // Timestamp point lookup + double point lookup + bool point lookup:
    // three different types, all point lookups - works for both sort orders.
    auto rowType = ROW({"a", "b", "c"}, {TIMESTAMP(), DOUBLE(), BOOLEAN()});
    Timestamp ts(100, 0);

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<TimestampRange>(ts, ts, false);
    filters["b"] = std::make_unique<DoubleRange>(
        2.5, false, false, 2.5, false, false, false);
    filters["c"] = std::make_unique<BoolValue>(false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"}, sortOrders(3), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    EXPECT_EQ(bounds.indexColumns.size(), 3);
    EXPECT_EQ(bounds.indexColumns[0], "a");
    EXPECT_EQ(bounds.indexColumns[1], "b");
    EXPECT_EQ(bounds.indexColumns[2], "c");

    verifyBound<Timestamp>(bounds.lowerBound->bound, "a", ts);
    verifyBound<Timestamp>(bounds.upperBound->bound, "a", ts);
    verifyBound<double>(bounds.lowerBound->bound, "b", 2.5);
    verifyBound<double>(bounds.upperBound->bound, "b", 2.5);
    verifyBound<bool>(bounds.lowerBound->bound, "c", false);
    verifyBound<bool>(bounds.upperBound->bound, "c", false);

    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    EXPECT_EQ(scanSpec->childByName("c")->filter(), nullptr);
  }

  {
    // Bigint point lookup + varchar range + float point lookup.
    // Ascending: bigint + varchar convertible (range stops extraction).
    // Descending: only bigint convertible (VARCHAR not supported).
    auto rowType = ROW({"a", "b", "c"}, {BIGINT(), VARCHAR(), REAL()});

    std::unordered_map<std::string, std::unique_ptr<Filter>> filters;
    filters["a"] = std::make_unique<BigintRange>(42, 42, false);
    filters["b"] = std::make_unique<BytesRange>(
        "abc", false, false, "xyz", false, false, false);
    filters["c"] = std::make_unique<FloatRange>(
        1.5f, false, false, 1.5f, false, false, false);
    auto scanSpec = createScanSpec(rowType, filters);

    auto result = convertFilterToIndexBounds(
        {"a", "b", "c"}, sortOrders(3), rowType, *scanSpec, pool_.get());
    ASSERT_TRUE(result.has_value());

    const auto& bounds = result->indexBounds;
    const size_t expectedColumns = ascending ? 2 : 1;
    EXPECT_EQ(bounds.indexColumns.size(), expectedColumns);
    EXPECT_EQ(bounds.indexColumns[0], "a");

    verifyBound<int64_t>(bounds.lowerBound->bound, "a", 42);
    verifyBound<int64_t>(bounds.upperBound->bound, "a", 42);
    EXPECT_EQ(scanSpec->childByName("a")->filter(), nullptr);

    if (ascending) {
      EXPECT_EQ(bounds.indexColumns[1], "b");
      verifyRangeBounds<StringView>(
          bounds, "b", StringView("abc"), StringView("xyz"));
      EXPECT_EQ(scanSpec->childByName("b")->filter(), nullptr);
    } else {
      EXPECT_NE(scanSpec->childByName("b")->filter(), nullptr);
    }
    // "c" is never converted (range on "b" stops extraction for ascending,
    // VARCHAR not supported stops for descending).
    EXPECT_NE(scanSpec->childByName("c")->filter(), nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SortOrders,
    IndexFilterTestWithSortOrder,
    ::testing::Values(
        SortOrder{.ascending = true},
        SortOrder{.ascending = false}),
    [](const ::testing::TestParamInfo<SortOrder>& info) {
      return info.param.ascending ? "Ascending" : "Descending";
    });

} // namespace facebook::nimble::index::test
