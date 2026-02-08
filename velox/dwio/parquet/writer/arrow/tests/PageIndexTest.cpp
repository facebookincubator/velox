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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/PageIndex.h"

#include <gtest/gtest.h>

#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

namespace facebook::velox::parquet::arrow {

struct PageIndexRanges {
  int64_t columnIndexOffset;
  int64_t columnIndexLength;
  int64_t offsetIndexOffset;
  int64_t offsetIndexLength;
};

using RowGroupRanges = std::vector<PageIndexRanges>;

/// Creates an FileMetaData object w/ single row group based on data in.
/// 'Row_group_ranges'. It sets the offsets and sizes of the column index and.
/// Offset index members of the row group. It doesn't set the member if the.
/// Input value is -1.
std::shared_ptr<FileMetaData> constructFakeMetaData(
    const RowGroupRanges& rowGroupRanges) {
  facebook::velox::parquet::thrift::RowGroup rowGroup;
  for (auto& pageIndexRanges : rowGroupRanges) {
    facebook::velox::parquet::thrift::ColumnChunk colChunk;
    if (pageIndexRanges.columnIndexOffset != -1) {
      colChunk.__set_column_index_offset(pageIndexRanges.columnIndexOffset);
    }
    if (pageIndexRanges.columnIndexLength != -1) {
      colChunk.__set_column_index_length(
          static_cast<int32_t>(pageIndexRanges.columnIndexLength));
    }
    if (pageIndexRanges.offsetIndexOffset != -1) {
      colChunk.__set_offset_index_offset(pageIndexRanges.offsetIndexOffset);
    }
    if (pageIndexRanges.offsetIndexLength != -1) {
      colChunk.__set_offset_index_length(
          static_cast<int32_t>(pageIndexRanges.offsetIndexLength));
    }
    rowGroup.columns.push_back(colChunk);
  }

  facebook::velox::parquet::thrift::FileMetaData metadata;
  metadata.row_groups.push_back(rowGroup);

  metadata.schema.emplace_back();
  schema::NodeVector fields;
  for (size_t i = 0; i < rowGroupRanges.size(); ++i) {
    fields.push_back(schema::int64(std::to_string(i)));
    metadata.schema.emplace_back();
    fields.back()->toParquet(&metadata.schema.back());
  }
  schema::GroupNode::make("schema", Repetition::kRepeated, fields)
      ->toParquet(&metadata.schema.front());

  auto sink = createOutputStream();
  ThriftSerializer{}.serialize(&metadata, sink.get());
  auto buffer = sink->Finish().MoveValueUnsafe();
  uint32_t len = static_cast<uint32_t>(buffer->size());
  return FileMetaData::make(buffer->data(), &len);
}

/// Validates that 'DeterminePageIndexRangesInRowGroup()' selects the expected.
/// File offsets and sizes or returns false when the row group doesn't have a.
/// Page index.
void validatePageIndexRange(
    const RowGroupRanges& rowGroupRanges,
    const std::vector<int32_t>& columnIndices,
    bool expectedHasColumnIndex,
    bool expectedHasOffsetIndex,
    int expectedCiStart,
    int expectedCiSize,
    int expectedOiStart,
    int expectedOiSize) {
  auto fileMetadata = constructFakeMetaData(rowGroupRanges);
  auto readRange = PageIndexReader::determinePageIndexRangesInRowGroup(
      *fileMetadata->rowGroup(0), columnIndices);
  ASSERT_EQ(expectedHasColumnIndex, readRange.columnIndex.has_value());
  ASSERT_EQ(expectedHasOffsetIndex, readRange.offsetIndex.has_value());
  if (expectedHasColumnIndex) {
    EXPECT_EQ(expectedCiStart, readRange.columnIndex->offset);
    EXPECT_EQ(expectedCiSize, readRange.columnIndex->length);
  }
  if (expectedHasOffsetIndex) {
    EXPECT_EQ(expectedOiStart, readRange.offsetIndex->offset);
    EXPECT_EQ(expectedOiSize, readRange.offsetIndex->length);
  }
}

/// This test constructs a couple of artificial row groups with page index.
/// Offsets in them. Then it validates if.
/// PageIndexReader::DeterminePageIndexRangesInRowGroup() properly computes the.
/// File range that contains the whole page index.
TEST(PageIndex, determinePageIndexRangesInRowGroup) {
  // No Column chunks.
  validatePageIndexRange({}, {}, false, false, -1, -1, -1, -1);
  // No page index at all.
  validatePageIndexRange({{-1, -1, -1, -1}}, {}, false, false, -1, -1, -1, -1);
  // Page index for single column chunk.
  validatePageIndexRange({{10, 5, 15, 5}}, {}, true, true, 10, 5, 15, 5);
  // Page index for two column chunks.
  validatePageIndexRange(
      {{10, 5, 30, 25}, {15, 15, 50, 20}}, {}, true, true, 10, 20, 30, 40);
  // Page index for second column chunk.
  validatePageIndexRange(
      {{-1, -1, -1, -1}, {20, 10, 30, 25}}, {}, true, true, 20, 10, 30, 25);
  // Page index for first column chunk.
  validatePageIndexRange(
      {{10, 5, 15, 5}, {-1, -1, -1, -1}}, {}, true, true, 10, 5, 15, 5);
  // Missing offset index for first column chunk. Gap in column index.
  validatePageIndexRange(
      {{10, 5, -1, -1}, {20, 10, 30, 25}}, {}, true, true, 10, 20, 30, 25);
  // Missing offset index for second column chunk.
  validatePageIndexRange(
      {{10, 5, 25, 5}, {20, 10, -1, -1}}, {}, true, true, 10, 20, 25, 5);
  // Four column chunks.
  validatePageIndexRange(
      {{100, 10, 220, 30},
       {110, 25, 250, 10},
       {140, 30, 260, 40},
       {200, 10, 300, 100}},
      {},
      true,
      true,
      100,
      110,
      220,
      180);
}

/// This test constructs a couple of artificial row groups with page index.
/// Offsets in them. Then it validates if.
/// PageIndexReader::DeterminePageIndexRangesInRowGroup() properly computes the.
/// File range that contains the page index of selected columns.
TEST(PageIndex, DeterminePageIndexRangesInRowGroupWithPartialColumnsSelected) {
  // No page index at all.
  validatePageIndexRange({{-1, -1, -1, -1}}, {0}, false, false, -1, -1, -1, -1);
  // Page index for single column chunk.
  validatePageIndexRange({{10, 5, 15, 5}}, {0}, true, true, 10, 5, 15, 5);
  // Page index for the 1st column chunk.
  validatePageIndexRange(
      {{10, 5, 30, 25}, {15, 15, 50, 20}}, {0}, true, true, 10, 5, 30, 25);
  // Page index for the 2nd column chunk.
  validatePageIndexRange(
      {{10, 5, 30, 25}, {15, 15, 50, 20}}, {1}, true, true, 15, 15, 50, 20);
  // Only 2nd column is selected among four column chunks.
  validatePageIndexRange(
      {{100, 10, 220, 30},
       {110, 25, 250, 10},
       {140, 30, 260, 40},
       {200, 10, 300, 100}},
      {1},
      true,
      true,
      110,
      25,
      250,
      10);
  // Only 2nd and 3rd columns are selected among four column chunks.
  validatePageIndexRange(
      {{100, 10, 220, 30},
       {110, 25, 250, 10},
       {140, 30, 260, 40},
       {200, 10, 300, 100}},
      {1, 2},
      true,
      true,
      110,
      60,
      250,
      50);
  // Only 2nd and 4th columns are selected among four column chunks.
  validatePageIndexRange(
      {{100, 10, 220, 30},
       {110, 25, 250, 10},
       {140, 30, 260, 40},
       {200, 10, 300, 100}},
      {1, 3},
      true,
      true,
      110,
      100,
      250,
      150);
  // Only 1st, 2nd and 4th columns are selected among four column chunks.
  validatePageIndexRange(
      {{100, 10, 220, 30},
       {110, 25, 250, 10},
       {140, 30, 260, 40},
       {200, 10, 300, 100}},
      {0, 1, 3},
      true,
      true,
      100,
      110,
      220,
      180);
  // 3Rd column is selected but not present in the row group.
  EXPECT_THROW(
      validatePageIndexRange(
          {{10, 5, 30, 25}, {15, 15, 50, 20}},
          {2},
          false,
          false,
          -1,
          -1,
          -1,
          -1),
      ParquetException);
}

/// This test constructs a couple of artificial row groups with page index.
/// Offsets in them. Then it validates if.
/// PageIndexReader::DeterminePageIndexRangesInRowGroup() properly detects if.
/// Column index or offset index is missing.
TEST(PageIndex, DeterminePageIndexRangesInRowGroupWithMissingPageIndex) {
  // No column index at all.
  validatePageIndexRange({{-1, -1, 15, 5}}, {}, false, true, -1, -1, 15, 5);
  // No offset index at all.
  validatePageIndexRange({{10, 5, -1, -1}}, {}, true, false, 10, 5, -1, -1);
  // No column index at all among two column chunks.
  validatePageIndexRange(
      {{-1, -1, 30, 25}, {-1, -1, 50, 20}}, {}, false, true, -1, -1, 30, 40);
  // No offset index at all among two column chunks.
  validatePageIndexRange(
      {{10, 5, -1, -1}, {15, 15, -1, -1}}, {}, true, false, 10, 20, -1, -1);
}

TEST(PageIndex, WriteOffsetIndex) {
  /// Create offset index via the OffsetIndexBuilder interface.
  auto Builder = OffsetIndexBuilder::make();
  const size_t numPages = 5;
  const std::vector<int64_t> offsets = {100, 200, 300, 400, 500};
  const std::vector<int32_t> pageSizes = {1024, 2048, 3072, 4096, 8192};
  const std::vector<int64_t> firstRowIndices = {0, 10000, 20000, 30000, 40000};
  for (size_t i = 0; i < numPages; ++i) {
    Builder->addPage(offsets[i], pageSizes[i], firstRowIndices[i]);
  }
  const int64_t finalPosition = 4096;
  Builder->finish(finalPosition);

  std::vector<std::unique_ptr<OffsetIndex>> offsetIndexes;
  /// 1St element is the offset index just built.
  offsetIndexes.emplace_back(Builder->build());
  /// 2Nd element is the offset index restored by serialize-then-deserialize.
  /// Round trip.
  auto sink = createOutputStream();
  Builder->writeTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  offsetIndexes.emplace_back(
      OffsetIndex::make(
          buffer->data(),
          static_cast<uint32_t>(buffer->size()),
          defaultReaderProperties()));

  /// Verify the data of the offset index.
  for (const auto& offsetIndex : offsetIndexes) {
    ASSERT_EQ(numPages, offsetIndex->pageLocations().size());
    for (size_t i = 0; i < numPages; ++i) {
      const auto& pageLocation = offsetIndex->pageLocations().at(i);
      ASSERT_EQ(offsets[i] + finalPosition, pageLocation.offset);
      ASSERT_EQ(pageSizes[i], pageLocation.compressedPageSize);
      ASSERT_EQ(firstRowIndices[i], pageLocation.firstRowIndex);
    }
  }
}

void testWriteTypedColumnIndex(
    schema::NodePtr Node,
    const std::vector<EncodedStatistics>& pageStats,
    BoundaryOrder::type boundaryOrder,
    bool hasNullCounts) {
  auto descr =
      std::make_unique<ColumnDescriptor>(Node, /*max_definition_level=*/1, 0);

  auto Builder = ColumnIndexBuilder::make(descr.get());
  for (const auto& stats : pageStats) {
    Builder->addPage(stats);
  }
  ASSERT_NO_THROW(Builder->finish());

  std::vector<std::unique_ptr<ColumnIndex>> columnIndexes;
  /// 1St element is the column index just built.
  columnIndexes.emplace_back(Builder->build());
  /// 2Nd element is the column index restored by serialize-then-deserialize.
  /// Round trip.
  auto sink = createOutputStream();
  Builder->writeTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  columnIndexes.emplace_back(
      ColumnIndex::make(
          *descr,
          buffer->data(),
          static_cast<uint32_t>(buffer->size()),
          defaultReaderProperties()));

  /// Verify the data of the column index.
  for (const auto& columnIndex : columnIndexes) {
    ASSERT_EQ(boundaryOrder, columnIndex->boundaryOrder());
    ASSERT_EQ(hasNullCounts, columnIndex->hasNullCounts());
    const size_t numPages = columnIndex->nullPages().size();
    for (size_t i = 0; i < numPages; ++i) {
      ASSERT_EQ(pageStats[i].allNullValue, columnIndex->nullPages()[i]);
      ASSERT_EQ(pageStats[i].min(), columnIndex->encodedMinValues()[i]);
      ASSERT_EQ(pageStats[i].max(), columnIndex->encodedMaxValues()[i]);
      if (hasNullCounts) {
        ASSERT_EQ(pageStats[i].nullCount, columnIndex->nullCounts()[i]);
      }
    }
  }
}

TEST(PageIndex, WriteInt32ColumnIndex) {
  auto encode = [=](int32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int32_t));
  };

  // Integer values in the ascending order.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setNullCount(1).setMin(encode(1)).setMax(encode(2));
  pageStats.at(1).setNullCount(2).setMin(encode(2)).setMax(encode(3));
  pageStats.at(2).setNullCount(3).setMin(encode(3)).setMax(encode(4));

  testWriteTypedColumnIndex(
      schema::int32("c1"),
      pageStats,
      BoundaryOrder::kAscending,
      /*has_null_counts=*/true);
}

TEST(PageIndex, WriteInt64ColumnIndex) {
  auto encode = [=](int64_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int64_t));
  };

  // Integer values in the descending order.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setNullCount(4).setMin(encode(-1)).setMax(encode(-2));
  pageStats.at(1).setNullCount(0).setMin(encode(-2)).setMax(encode(-3));
  pageStats.at(2).setNullCount(4).setMin(encode(-3)).setMax(encode(-4));

  testWriteTypedColumnIndex(
      schema::int64("c1"),
      pageStats,
      BoundaryOrder::kDescending,
      /*has_null_counts=*/true);
}

TEST(PageIndex, WriteFloatColumnIndex) {
  auto encode = [=](float value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(float));
  };

  // Float values with no specific order.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setNullCount(0).setMin(encode(2.2F)).setMax(encode(4.4F));
  pageStats.at(1).setNullCount(0).setMin(encode(1.1F)).setMax(encode(5.5F));
  pageStats.at(2).setNullCount(0).setMin(encode(3.3F)).setMax(encode(6.6F));

  testWriteTypedColumnIndex(
      schema::floatType("c1"),
      pageStats,
      BoundaryOrder::kUnordered,
      /*has_null_counts=*/true);
}

TEST(PageIndex, WriteDoubleColumnIndex) {
  auto encode = [=](double value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(double));
  };

  // Double values with no specific order and without null count.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setMin(encode(1.2)).setMax(encode(4.4));
  pageStats.at(1).setMin(encode(2.2)).setMax(encode(5.5));
  pageStats.at(2).setMin(encode(3.3)).setMax(encode(-6.6));

  testWriteTypedColumnIndex(
      schema::doubleType("c1"),
      pageStats,
      BoundaryOrder::kUnordered,
      /*has_null_counts=*/false);
}

TEST(PageIndex, WriteByteArrayColumnIndex) {
  // Byte array values with identical min/max.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setMin("bar").setMax("foo");
  pageStats.at(1).setMin("bar").setMax("foo");
  pageStats.at(2).setMin("bar").setMax("foo");

  testWriteTypedColumnIndex(
      schema::byteArray("c1"),
      pageStats,
      BoundaryOrder::kAscending,
      /*has_null_counts=*/false);
}

TEST(PageIndex, WriteFLBAColumnIndex) {
  // FLBA values in the ascending order with some null pages.
  std::vector<EncodedStatistics> pageStats(5);
  pageStats.at(0).setMin("abc").setMax("ABC");
  pageStats.at(1).allNullValue = true;
  pageStats.at(2).setMin("foo").setMax("FOO");
  pageStats.at(3).allNullValue = true;
  pageStats.at(4).setMin("xyz").setMax("XYZ");

  auto Node = schema::PrimitiveNode::make(
      "c1",
      Repetition::kOptional,
      Type::kFixedLenByteArray,
      ConvertedType::kNone,
      /*length=*/3);
  testWriteTypedColumnIndex(
      std::move(Node),
      pageStats,
      BoundaryOrder::kAscending,
      /*has_null_counts=*/false);
}

TEST(PageIndex, WriteColumnIndexWithAllNullPages) {
  // All values are null.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setNullCount(100).allNullValue = true;
  pageStats.at(1).setNullCount(100).allNullValue = true;
  pageStats.at(2).setNullCount(100).allNullValue = true;

  testWriteTypedColumnIndex(
      schema::int32("c1"),
      pageStats,
      BoundaryOrder::kUnordered,
      /*has_null_counts=*/true);
}

TEST(PageIndex, WriteColumnIndexWithInvalidNullCounts) {
  auto encode = [=](int32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int32_t));
  };

  // Some pages do not provide null_count.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setMin(encode(1)).setMax(encode(2)).setNullCount(0);
  pageStats.at(1).setMin(encode(1)).setMax(encode(3));
  pageStats.at(2).setMin(encode(2)).setMax(encode(3)).setNullCount(0);

  testWriteTypedColumnIndex(
      schema::int32("c1"),
      pageStats,
      BoundaryOrder::kAscending,
      /*has_null_counts=*/false);
}

TEST(PageIndex, WriteColumnIndexWithCorruptedStats) {
  auto encode = [=](int32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int32_t));
  };

  // 2Nd page does not set anything.
  std::vector<EncodedStatistics> pageStats(3);
  pageStats.at(0).setMin(encode(1)).setMax(encode(2));
  pageStats.at(2).setMin(encode(3)).setMax(encode(4));

  ColumnDescriptor descr(schema::int32("c1"), /*max_definition_level=*/1, 0);
  auto Builder = ColumnIndexBuilder::make(&descr);
  for (const auto& stats : pageStats) {
    Builder->addPage(stats);
  }
  ASSERT_NO_THROW(Builder->finish());
  ASSERT_EQ(nullptr, Builder->build());

  auto sink = createOutputStream();
  Builder->writeTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  EXPECT_EQ(0, buffer->size());
}

TEST(PageIndex, TestPageIndexBuilderWithZeroRowGroup) {
  schema::NodeVector fields = {schema::int32("c1"), schema::byteArray("c2")};
  schema::NodePtr root =
      schema::GroupNode::make("schema", Repetition::kRepeated, fields);
  SchemaDescriptor schema;
  schema.init(root);

  auto Builder = PageIndexBuilder::make(&schema);

  // AppendRowGroup() is not called and expect throw.
  ASSERT_THROW(Builder->getColumnIndexBuilder(0), ParquetException);
  ASSERT_THROW(Builder->getOffsetIndexBuilder(0), ParquetException);

  // Finish the builder without calling AppendRowGroup().
  ASSERT_NO_THROW(Builder->finish());

  // Verify WriteTo does not write anything.
  auto sink = createOutputStream();
  PageIndexLocation location;
  Builder->writeTo(sink.get(), &location);
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  ASSERT_EQ(0, buffer->size());
  ASSERT_TRUE(location.columnIndexLocation.empty());
  ASSERT_TRUE(location.offsetIndexLocation.empty());
}

class PageIndexBuilderTest : public ::testing::Test {
 public:
  void writePageIndexes(
      int numRowGroups,
      int numColumns,
      const std::vector<std::vector<EncodedStatistics>>& pageStats,
      const std::vector<std::vector<PageLocation>>& pageLocations,
      int finalPosition) {
    auto Builder = PageIndexBuilder::make(&schema_);
    for (int rowGroup = 0; rowGroup < numRowGroups; ++rowGroup) {
      ASSERT_NO_THROW(Builder->appendRowGroup());

      for (int column = 0; column < numColumns; ++column) {
        if (static_cast<size_t>(column) < pageStats[rowGroup].size()) {
          auto ColumnIndexBuilder = Builder->getColumnIndexBuilder(column);
          ASSERT_NO_THROW(
              ColumnIndexBuilder->addPage(pageStats[rowGroup][column]));
          ASSERT_NO_THROW(ColumnIndexBuilder->finish());
        }

        if (static_cast<size_t>(column) < pageLocations[rowGroup].size()) {
          auto OffsetIndexBuilder = Builder->getOffsetIndexBuilder(column);
          ASSERT_NO_THROW(
              OffsetIndexBuilder->addPage(pageLocations[rowGroup][column]));
          ASSERT_NO_THROW(OffsetIndexBuilder->finish(finalPosition));
        }
      }
    }
    ASSERT_NO_THROW(Builder->finish());

    auto sink = createOutputStream();
    Builder->writeTo(sink.get(), &pageIndexLocation_);
    PARQUET_ASSIGN_OR_THROW(buffer_, sink->Finish());

    ASSERT_EQ(
        static_cast<size_t>(numRowGroups),
        pageIndexLocation_.columnIndexLocation.size());
    ASSERT_EQ(
        static_cast<size_t>(numRowGroups),
        pageIndexLocation_.offsetIndexLocation.size());
    for (int rowGroup = 0; rowGroup < numRowGroups; ++rowGroup) {
      ASSERT_EQ(
          static_cast<size_t>(numColumns),
          pageIndexLocation_.columnIndexLocation[rowGroup].size());
      ASSERT_EQ(
          static_cast<size_t>(numColumns),
          pageIndexLocation_.offsetIndexLocation[rowGroup].size());
    }
  }

  void
  checkColumnIndex(int rowGroup, int column, const EncodedStatistics& stats) {
    auto columnIndex = readColumnIndex(rowGroup, column);
    ASSERT_NE(nullptr, columnIndex);
    ASSERT_EQ(size_t{1}, columnIndex->nullPages().size());
    ASSERT_EQ(stats.allNullValue, columnIndex->nullPages()[0]);
    ASSERT_EQ(stats.min(), columnIndex->encodedMinValues()[0]);
    ASSERT_EQ(stats.max(), columnIndex->encodedMaxValues()[0]);
    ASSERT_EQ(stats.hasNullCount, columnIndex->hasNullCounts());
    if (stats.hasNullCount) {
      ASSERT_EQ(stats.nullCount, columnIndex->nullCounts()[0]);
    }
  }

  void checkOffsetIndex(
      int rowGroup,
      int column,
      const PageLocation& expectedLocation,
      int64_t finalLocation) {
    auto offsetIndex = readOffsetIndex(rowGroup, column);
    ASSERT_NE(nullptr, offsetIndex);
    ASSERT_EQ(size_t{1}, offsetIndex->pageLocations().size());
    const auto& location = offsetIndex->pageLocations()[0];
    ASSERT_EQ(expectedLocation.offset + finalLocation, location.offset);
    ASSERT_EQ(expectedLocation.compressedPageSize, location.compressedPageSize);
    ASSERT_EQ(expectedLocation.firstRowIndex, location.firstRowIndex);
  }

 protected:
  std::unique_ptr<ColumnIndex> readColumnIndex(int rowGroup, int column) {
    auto location = pageIndexLocation_.columnIndexLocation[rowGroup][column];
    if (!location.has_value()) {
      return nullptr;
    }
    auto properties = defaultReaderProperties();
    return ColumnIndex::make(
        *schema_.column(column),
        buffer_->data() + location->offset,
        static_cast<uint32_t>(location->length),
        properties);
  }

  std::unique_ptr<OffsetIndex> readOffsetIndex(int rowGroup, int column) {
    auto location = pageIndexLocation_.offsetIndexLocation[rowGroup][column];
    if (!location.has_value()) {
      return nullptr;
    }
    auto properties = defaultReaderProperties();
    return OffsetIndex::make(
        buffer_->data() + location->offset,
        static_cast<uint32_t>(location->length),
        properties);
  }

  SchemaDescriptor schema_;
  std::shared_ptr<Buffer> buffer_;
  PageIndexLocation pageIndexLocation_;
};

TEST_F(PageIndexBuilderTest, SingleRowGroup) {
  schema::NodePtr root = schema::GroupNode::make(
      "schema",
      Repetition::kRepeated,
      {schema::byteArray("c1"),
       schema::byteArray("c2"),
       schema::byteArray("c3")});
  schema_.init(root);

  // Prepare page stats and page locations for single row group.
  // Note that the 3rd column does not have any stats and its page index is.
  // Disabled.
  const int numRowGroups = 1;
  const int numColumns = 3;
  const std::vector<std::vector<EncodedStatistics>> pageStats = {
      /*row_group_id=0*/
      {/*column_id=0*/ EncodedStatistics().setNullCount(0).setMin("a").setMax(
           "b"),
       /*column_id=1*/
       EncodedStatistics().setNullCount(0).setMin("A").setMax("B")}};
  const std::vector<std::vector<PageLocation>> pageLocations = {
      /*row_group_id=0*/
      {/*column_id=0*/ {/*offset=*/128,
                        /*compressed_page_size=*/512,
                        /*first_row_index=*/0},
       /*column_id=1*/
       {/*offset=*/1024,
        /*compressed_page_size=*/512,
        /*first_row_index=*/0}}};
  const int64_t finalPosition = 200;

  writePageIndexes(
      numRowGroups, numColumns, pageStats, pageLocations, finalPosition);

  // Verify that first two columns have good page indexes.
  for (int column = 0; column < 2; ++column) {
    checkColumnIndex(/*row_group=*/0, column, pageStats[0][column]);
    checkOffsetIndex(
        /*row_group=*/0, column, pageLocations[0][column], finalPosition);
  }

  // Verify the 3rd column does not have page indexes.
  ASSERT_EQ(nullptr, readColumnIndex(/*row_group=*/0, /*column=*/2));
  ASSERT_EQ(nullptr, readOffsetIndex(/*row_group=*/0, /*column=*/2));
}

TEST_F(PageIndexBuilderTest, TwoRowGroups) {
  schema::NodePtr root = schema::GroupNode::make(
      "schema",
      Repetition::kRepeated,
      {schema::byteArray("c1"), schema::byteArray("c2")});
  schema_.init(root);

  // Prepare page stats and page locations for two row groups.
  // Note that the 2nd column in the 2nd row group has corrupted stats.
  const int numRowGroups = 2;
  const int numColumns = 2;
  const std::vector<std::vector<EncodedStatistics>> pageStats = {
      /*row_group_id=0*/
      {/*column_id=0*/ EncodedStatistics().setMin("a").setMax("b"),
       /*column_id=1*/
       EncodedStatistics().setNullCount(0).setMin("A").setMax("B")},
      /*row_group_id=1*/
      {/*column_id=0*/ EncodedStatistics() /* corrupted stats */,
       /*column_id=1*/
       EncodedStatistics().setNullCount(0).setMin("bar").setMax("foo")}};
  const std::vector<std::vector<PageLocation>> pageLocations = {
      /*row_group_id=0*/
      {/*column_id=0*/ {/*offset=*/128,
                        /*compressed_page_size=*/512,
                        /*first_row_index=*/0},
       /*column_id=1*/
       {/*offset=*/1024,
        /*compressed_page_size=*/512,
        /*first_row_index=*/0}},
      /*row_group_id=0*/
      {/*column_id=0*/ {/*offset=*/128,
                        /*compressed_page_size=*/512,
                        /*first_row_index=*/0},
       /*column_id=1*/
       {/*offset=*/1024,
        /*compressed_page_size=*/512,
        /*first_row_index=*/0}}};
  const int64_t finalPosition = 200;

  writePageIndexes(
      numRowGroups, numColumns, pageStats, pageLocations, finalPosition);

  // Verify that all columns have good column indexes except the 2nd column in.
  // The 2nd row group.
  checkColumnIndex(/*row_group=*/0, /*column=*/0, pageStats[0][0]);
  checkColumnIndex(/*row_group=*/0, /*column=*/1, pageStats[0][1]);
  checkColumnIndex(/*row_group=*/1, /*column=*/1, pageStats[1][1]);
  ASSERT_EQ(nullptr, readColumnIndex(/*row_group=*/1, /*column=*/0));

  // Verify that two columns have good offset indexes.
  checkOffsetIndex(
      /*row_group=*/0, /*column=*/0, pageLocations[0][0], finalPosition);
  checkOffsetIndex(
      /*row_group=*/0, /*column=*/1, pageLocations[0][1], finalPosition);
  checkOffsetIndex(
      /*row_group=*/1, /*column=*/0, pageLocations[1][0], finalPosition);
  checkOffsetIndex(
      /*row_group=*/1, /*column=*/1, pageLocations[1][1], finalPosition);
}

} // namespace facebook::velox::parquet::arrow
