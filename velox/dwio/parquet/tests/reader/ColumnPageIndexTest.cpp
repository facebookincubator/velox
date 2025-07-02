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

#include "velox/dwio/parquet/reader/ColumnPageIndex.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;
using namespace facebook::velox::dwio::common;

class ColumnPageIndexTest : public ParquetTestBase {
 public:
  thrift::ColumnIndex makeColumnIndex(
      const std::vector<bool>& nullPages,
      const std::vector<std::string>& mins,
      const std::vector<std::string>& maxs,
      const std::vector<int64_t>& nullCounts) {
    thrift::ColumnIndex ci;
    ci.__set_null_pages(nullPages);
    ci.__set_min_values(mins);
    ci.__set_max_values(maxs);
    ci.__set_null_counts(nullCounts);
    return ci;
  }

  thrift::OffsetIndex makeOffsetIndex(
      const std::vector<int64_t>& firstRowIndex,
      const std::vector<int64_t>& offsets,
      const std::vector<int32_t>& sizes) {
    thrift::OffsetIndex oi;
    std::vector<thrift::PageLocation> locs;
    size_t n = offsets.size();
    locs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      thrift::PageLocation pl;
      pl.__set_first_row_index(firstRowIndex[i]);
      pl.__set_offset(offsets[i]);
      pl.__set_compressed_page_size(sizes[i]);
      locs.push_back(std::move(pl));
    }
    oi.__set_page_locations(locs);
    return oi;
  }

  template <typename T>
  std::string encode(T v) {
    std::string s(sizeof(T), '\0');
    std::memcpy(s.data(), &v, sizeof(T));
    return s;
  }
};

TEST_F(ColumnPageIndexTest, basic) {
  // Pages: [0..29], [30..69], [70..99]
  auto ci = makeColumnIndex(
      std::vector<bool>{false, false, false},
      std::vector<std::string>{"a", "m", "x"},
      std::vector<std::string>{"l", "w", "z"},
      std::vector<int64_t>{0, 2, 5});
  auto oi = makeOffsetIndex(
      std::vector<int64_t>{0, 30, 70},
      std::vector<int64_t>{100, 200, 300},
      std::vector<int32_t>{50, 60, 70});
  ColumnPageIndex idx(std::move(ci), std::move(oi), 100);

  EXPECT_EQ(idx.numPages(), 3);
  EXPECT_EQ(idx.pageRowCount(0), 30);
  EXPECT_EQ(idx.pageRowCount(1), 40);
  EXPECT_EQ(idx.pageRowCount(2), 30);
  EXPECT_EQ(idx.pageFirstRowIndex(2), 70);
  EXPECT_EQ(idx.pageOffset(1), 200);
  EXPECT_EQ(idx.compressedPageSize(2), 70);
}

TEST_F(ColumnPageIndexTest, updateSkippedPages) {
  auto ci = makeColumnIndex(
      {false, false, false}, {"a", "m", "x"}, {"l", "w", "z"}, {0, 2, 5});
  auto oi = makeOffsetIndex({0, 30, 70}, {100, 200, 300}, {50, 60, 70});
  ColumnPageIndex idx(std::move(ci), std::move(oi), 100);

  RowRanges ranges;
  ranges.add(RowRange(10, 20)); // overlaps page0
  ranges.add(RowRange(35, 65)); // overlaps page1
  idx.updateSkippedPages(ranges);

  EXPECT_FALSE(idx.isPageSkipped(0));
  EXPECT_FALSE(idx.isPageSkipped(1));
  EXPECT_TRUE(idx.isPageSkipped(2)); // no overlap with [70..99]
}

TEST_F(ColumnPageIndexTest, booleanStatistics) {
  // one page
  auto ci = makeColumnIndex({false}, {""}, {""}, {3}); // three nulls
  auto oi = makeOffsetIndex({0}, {0}, {10}); // pageRowCount = 10
  ColumnPageIndex idx(std::move(ci), std::move(oi), 10);

  auto stats = idx.buildColumnStatisticsForPage(0, *BOOLEAN());
  auto bs = dynamic_cast<BooleanColumnStatistics*>(stats.get());
  EXPECT_NE(bs, nullptr);
  EXPECT_EQ(bs->hasNull(), std::optional<bool>(true));
  EXPECT_EQ(bs->getNumberOfValues(), std::optional<uint64_t>(7));
}

TEST_F(ColumnPageIndexTest, integerStatisticsNoNulls) {
  // one page
  auto ci = makeColumnIndex(
      {false}, {encode<int32_t>(-42)}, {encode<int32_t>(99)}, {0});
  auto oi = makeOffsetIndex({0}, {0}, {5}); // pageRowCount = 5
  ColumnPageIndex idx(std::move(ci), std::move(oi), 5);

  auto stats = idx.buildColumnStatisticsForPage(0, *INTEGER());
  auto is = dynamic_cast<IntegerColumnStatistics*>(stats.get());
  EXPECT_NE(is, nullptr);
  EXPECT_EQ(is->hasNull(), std::optional<bool>(false));
  EXPECT_EQ(is->getNumberOfValues(), std::optional<uint64_t>(5));
  EXPECT_EQ(is->getMinimum(), std::optional<int32_t>(-42));
  EXPECT_EQ(is->getMaximum(), std::optional<int32_t>(99));
}

TEST_F(ColumnPageIndexTest, integerStatisticsNullPage) {
  // one page, all nulls
  auto ci = makeColumnIndex({true}, {""}, {""}, {5});
  auto oi = makeOffsetIndex({0}, {0}, {5});
  ColumnPageIndex idx(std::move(ci), std::move(oi), 5);

  auto stats = idx.buildColumnStatisticsForPage(0, *INTEGER());
  auto is = dynamic_cast<IntegerColumnStatistics*>(stats.get());
  EXPECT_NE(is, nullptr);
  // entire page null → min/max nullopt, valueCount=0, hasNull=true
  EXPECT_EQ(is->getMinimum(), std::nullopt);
  EXPECT_EQ(is->getMaximum(), std::nullopt);
  EXPECT_EQ(is->getNumberOfValues(), std::optional<uint64_t>(0));
  EXPECT_EQ(is->hasNull(), std::optional<bool>(true));
}

TEST_F(ColumnPageIndexTest, doubleAndStringStatistics) {
  // double page
  auto ci = makeColumnIndex(
      {false, false},
      {encode<double>(3.14), "apple"},
      {encode<double>(6.28), "zebra"},
      {1, 0});
  auto oi = makeOffsetIndex({0, 0}, {0, 0}, {4, 3});
  ColumnPageIndex idx(std::move(ci), std::move(oi), /*totalRows*/ 7);

  // double
  {
    auto stats = idx.buildColumnStatisticsForPage(0, *DOUBLE());
    auto ds = dynamic_cast<DoubleColumnStatistics*>(stats.get());
    EXPECT_NE(ds, nullptr);
    EXPECT_EQ(ds->getMinimum(), std::optional<double>(3.14));
    EXPECT_EQ(ds->getMaximum(), std::optional<double>(6.28));
  }
  // string
  {
    auto stats = idx.buildColumnStatisticsForPage(1, *VARCHAR());
    auto ss = dynamic_cast<StringColumnStatistics*>(stats.get());
    EXPECT_NE(ss, nullptr);
    EXPECT_EQ(ss->getMinimum(), std::optional<std::string>("apple"));
    EXPECT_EQ(ss->getMaximum(), std::optional<std::string>("zebra"));
  }
}
