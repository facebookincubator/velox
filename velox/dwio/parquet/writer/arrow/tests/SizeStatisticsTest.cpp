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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "arrow/io/memory.h"
#include "arrow/table.h"
#include "arrow/testing/builder.h"
#include "arrow/testing/util.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/PageIndex.h"
#include "velox/dwio/parquet/writer/arrow/Properties.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/SizeStatistics.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/Writer.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/FileReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

namespace facebook::velox::parquet::arrow {

bool operator==(const SizeStatistics& lhs, const SizeStatistics& rhs) {
  return lhs.definitionLevelHistogram == rhs.definitionLevelHistogram &&
      lhs.repetitionLevelHistogram == rhs.repetitionLevelHistogram &&
      lhs.unencodedByteArrayDataBytes == rhs.unencodedByteArrayDataBytes;
}
struct PageSizeStatistics {
  std::vector<int64_t> definitionLevels;
  std::vector<int64_t> repetitionLevels;
  std::vector<int64_t> byteArrayBytes;

  bool operator==(const PageSizeStatistics& other) const {
    return definitionLevels == other.definitionLevels &&
        repetitionLevels == other.repetitionLevels &&
        byteArrayBytes == other.byteArrayBytes;
  }
};

namespace test {
TEST(SizeStatistics, updateLevelHistogram) {
  {
    // max_level = 1
    std::vector<int64_t> histogram(2, 0);
    updateLevelHistogram(std::vector<int16_t>{0, 1, 1, 1, 0}, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(2, 3));
    updateLevelHistogram(std::vector<int16_t>{1, 1, 0}, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(3, 5));
    updateLevelHistogram(std::vector<int16_t>{}, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(3, 5));
  }
  {
    // Cross the chunk boundary used by the max_level = 1 fast path.
    std::vector<int16_t> levels(1 << 14, 1);
    levels.push_back(0);
    std::vector<int64_t> histogram(2, 0);
    updateLevelHistogram(levels, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(1, 1 << 14));
  }
  {
    // max_level > 1
    std::vector<int64_t> histogram(3, 0);
    updateLevelHistogram(std::vector<int16_t>{0, 1, 2, 2, 0}, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(2, 1, 2));
    updateLevelHistogram(std::vector<int16_t>{1, 1, 0}, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(3, 3, 2));
    updateLevelHistogram(std::vector<int16_t>{}, histogram);
    EXPECT_THAT(histogram, ::testing::ElementsAre(3, 3, 2));
  }
}

TEST(SizeStatistics, thriftSerDe) {
  const std::vector<int64_t> kDefinitionLevels = {128, 64, 32, 16};
  const std::vector<int64_t> kRepetitionLevels = {100, 80, 60, 40, 20};
  constexpr int64_t kUnencodedByteArrayDataBytes = 1234;

  for (const auto& descriptor :
       {std::make_unique<ColumnDescriptor>(
            schema::int32("a"),
            /*maxDefinitionLevel=*/3,
            /*maxRepetitionLevel=*/4),
        std::make_unique<ColumnDescriptor>(
            schema::byteArray("a"),
            /*maxDefinitionLevel=*/3,
            /*maxRepetitionLevel=*/4)}) {
    auto sizeStatistics = SizeStatistics::make(descriptor.get());
    sizeStatistics->definitionLevelHistogram = kDefinitionLevels;
    sizeStatistics->repetitionLevelHistogram = kRepetitionLevels;
    if (descriptor->physicalType() == Type::kByteArray) {
      sizeStatistics->incrementUnencodedByteArrayDataBytes(
          kUnencodedByteArrayDataBytes);
    }

    const auto thriftStatistics = toThrift(*sizeStatistics);
    const auto restoredStatistics = fromThrift(thriftStatistics);
    EXPECT_EQ(restoredStatistics.definitionLevelHistogram, kDefinitionLevels);
    EXPECT_EQ(restoredStatistics.repetitionLevelHistogram, kRepetitionLevels);
    if (descriptor->physicalType() == Type::kByteArray) {
      EXPECT_TRUE(restoredStatistics.unencodedByteArrayDataBytes.has_value());
      EXPECT_EQ(
          *restoredStatistics.unencodedByteArrayDataBytes,
          kUnencodedByteArrayDataBytes);
    } else {
      EXPECT_FALSE(restoredStatistics.unencodedByteArrayDataBytes.has_value());
    }
  }
}

class SizeStatisticsRoundTripTest : public ::testing::Test {
 public:
  void writeFile(
      SizeStatisticsLevel level,
      const std::shared_ptr<::arrow::Table>& table,
      int64_t maxRowGroupLength,
      int64_t pageSize,
      int64_t writeBatchSize = DEFAULT_WRITE_BATCH_SIZE) {
    auto properties = WriterProperties::Builder()
                          .maxRowGroupLength(maxRowGroupLength)
                          ->dataPagesize(pageSize)
                          ->writeBatchSize(writeBatchSize)
                          ->enableWritePageIndex()
                          ->enableStatistics()
                          ->setSizeStatisticsLevel(level)
                          ->build();
    auto sink = createOutputStream();
    ASSERT_OK(
        arrow::writeTable(
            *table,
            ::arrow::default_memory_pool(),
            sink,
            maxRowGroupLength,
            std::move(properties)));
    ASSERT_OK_AND_ASSIGN(buffer_, sink->Finish());
  }

  void readSizeStatistics() {
    auto reader = ParquetFileReader::open(
        std::make_shared<::arrow::io::BufferReader>(buffer_));

    // Read row group size statistics in order.
    rowGroupStats_.clear();
    auto metadata = reader->metadata();
    for (int i = 0; i < metadata->numRowGroups(); ++i) {
      auto rowGroupMetadata = metadata->rowGroup(i);
      for (int j = 0; j < metadata->numColumns(); ++j) {
        auto columnMetadata = rowGroupMetadata->columnChunk(j);
        auto sizeStats = columnMetadata->sizeStatistics();
        rowGroupStats_.push_back(
            sizeStats == nullptr ? SizeStatistics{} : *sizeStats);
      }
    }

    // Read page size statistics in order.
    pageStats_.clear();
    auto pageIndexReader = reader->getPageIndexReader();
    ASSERT_NE(pageIndexReader, nullptr);

    for (int i = 0; i < metadata->numRowGroups(); ++i) {
      auto rowGroupIndexReader = pageIndexReader->rowGroup(i);
      ASSERT_NE(rowGroupIndexReader, nullptr);

      for (int j = 0; j < metadata->numColumns(); ++j) {
        PageSizeStatistics pageStats;

        auto columnIndex = rowGroupIndexReader->getColumnIndex(j);
        if (columnIndex != nullptr) {
          if (columnIndex->hasDefinitionLevelHistograms()) {
            pageStats.definitionLevels =
                columnIndex->definitionLevelHistograms();
          }
          if (columnIndex->hasRepetitionLevelHistograms()) {
            pageStats.repetitionLevels =
                columnIndex->repetitionLevelHistograms();
          }
        }

        auto offsetIndex = rowGroupIndexReader->getOffsetIndex(j);
        if (offsetIndex != nullptr) {
          pageStats.byteArrayBytes = offsetIndex->unencodedByteArrayDataBytes();
        }

        pageStats_.push_back(std::move(pageStats));
      }
    }
  }

  void readData() {
    auto reader = ParquetFileReader::open(
        std::make_shared<::arrow::io::BufferReader>(buffer_));
    auto metadata = reader->metadata();
    for (int i = 0; i < metadata->numRowGroups(); ++i) {
      const auto numRows = metadata->rowGroup(i)->numRows();
      auto rowGroupReader = reader->rowGroup(i);
      for (int column = 0; column < metadata->numColumns(); ++column) {
        auto columnReader = rowGroupReader->recordReader(column);
        EXPECT_EQ(columnReader->readRecords(numRows + 1), numRows);
      }
    }
  }

  void reset() {
    buffer_.reset();
  }

 protected:
  std::shared_ptr<::arrow::Buffer> buffer_;
  std::vector<SizeStatistics> rowGroupStats_;
  std::vector<PageSizeStatistics> pageStats_;
  inline static const SizeStatistics kEmptyRowGroupStats{};
  inline static const PageSizeStatistics kEmptyPageStats{};
};

TEST_F(SizeStatisticsRoundTripTest, enableSizeStats) {
  auto schema = ::arrow::schema({
      ::arrow::field("a", ::arrow::list(::arrow::list(::arrow::int32()))),
      ::arrow::field("b", ::arrow::list(::arrow::list(::arrow::utf8()))),
  });
  // First two rows will be in one row group, and the other two rows in another
  // row group.
  auto table = ::arrow::TableFromJSON(schema, {R"([
      [ [[1],[1,1],[1,1,1]], [["a"],["a","a"],["a","a","a"]] ],
      [ [[0,1,null]],        [["foo","bar",null]]            ],
      [ [],                  []                                  ],
      [ [[],[null],null],    [[],[null],null]                    ]
    ])"});

  for (const auto sizeStatsLevel :
       {SizeStatisticsLevel::None,
        SizeStatisticsLevel::ColumnChunk,
        SizeStatisticsLevel::PageAndColumnChunk}) {
    writeFile(sizeStatsLevel, table, /*maxRowGroupLength=*/2, /*pageSize=*/1);
    readSizeStatistics();
    if (sizeStatsLevel == SizeStatisticsLevel::None) {
      EXPECT_THAT(
          rowGroupStats_,
          ::testing::ElementsAre(
              kEmptyRowGroupStats,
              kEmptyRowGroupStats,
              kEmptyRowGroupStats,
              kEmptyRowGroupStats));
    } else {
      EXPECT_THAT(
          rowGroupStats_,
          ::testing::ElementsAre(
              SizeStatistics{{0, 0, 0, 0, 1, 8}, {2, 2, 5}, std::nullopt},
              SizeStatistics{{0, 0, 0, 0, 1, 8}, {2, 2, 5}, 12},
              SizeStatistics{{0, 1, 1, 1, 1, 0}, {2, 2, 0}, std::nullopt},
              SizeStatistics{{0, 1, 1, 1, 1, 0}, {2, 2, 0}, 0}));
    }
    if (sizeStatsLevel == SizeStatisticsLevel::PageAndColumnChunk) {
      EXPECT_THAT(
          pageStats_,
          ::testing::ElementsAre(
              PageSizeStatistics{
                  {0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 2}, {1, 2, 3, 1, 0, 2}, {}},
              PageSizeStatistics{
                  {0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 1, 2},
                  {1, 2, 3, 1, 0, 2},
                  {6, 6}},
              PageSizeStatistics{
                  {0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0}, {1, 0, 0, 1, 2, 0}, {}},
              PageSizeStatistics{
                  {0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0},
                  {1, 0, 0, 1, 2, 0},
                  {0, 0}}));
    } else {
      EXPECT_THAT(
          pageStats_,
          ::testing::ElementsAre(
              kEmptyPageStats,
              kEmptyPageStats,
              kEmptyPageStats,
              kEmptyPageStats));
    }
    reset();
  }
}

TEST_F(SizeStatisticsRoundTripTest, writeDictionaryArray) {
  auto schema = ::arrow::schema({::arrow::field(
      "a", ::arrow::dictionary(::arrow::int16(), ::arrow::utf8()))});
  writeFile(
      SizeStatisticsLevel::PageAndColumnChunk,
      ::arrow::TableFromJSON(
          schema, {R"([["aa"],["aaa"],[null],["a"],["aaa"],["a"]])"}),
      /*maxRowGroupLength=*/2,
      1);
  readSizeStatistics();
  EXPECT_THAT(
      rowGroupStats_,
      ::testing::ElementsAre(
          SizeStatistics{{0, 2}, {}, 5},
          SizeStatistics{{1, 1}, {}, 1},
          SizeStatistics{{0, 2}, {}, 4}));
  EXPECT_THAT(
      pageStats_,
      ::testing::ElementsAre(
          PageSizeStatistics{{0, 2}, {}, {5}},
          PageSizeStatistics{{1, 1}, {}, {1}},
          PageSizeStatistics{{0, 2}, {}, {4}}));
}

TEST_F(SizeStatisticsRoundTripTest, writePageInBatches) {
  // Rep/def level histograms are updates in batches of `write_batch_size`
  // levels inside a single page. Exercise the logic with more than one batch
  // per page.
  auto schema =
      ::arrow::schema({::arrow::field("a", ::arrow::list(::arrow::utf8()))});
  auto table = ::arrow::TableFromJSON(schema, {R"([
      [ [null,"a","ab"] ],
      [ null ],
      [ [] ],
      [ [null,"d","de"] ],
      [ ["g","gh",null] ],
      [ ["j","jk",null] ]
    ])"});
  for (int writeBatchSize : {100, 5, 4, 3, 2, 1}) {
    SCOPED_TRACE(writeBatchSize);
    writeFile(
        SizeStatisticsLevel::PageAndColumnChunk,
        table,
        /*maxRowGroupLength=*/1000,
        /*pageSize=*/1000,
        writeBatchSize);
    readSizeStatistics();
    EXPECT_THAT(
        rowGroupStats_,
        ::testing::ElementsAre(SizeStatistics{{1, 1, 4, 8}, {6, 8}, 12}));
    EXPECT_THAT(
        pageStats_,
        ::testing::ElementsAre(PageSizeStatistics{{1, 1, 4, 8}, {6, 8}, {12}}));
  }
}

TEST_F(SizeStatisticsRoundTripTest, largePage) {
  // When max_level is 1, the levels are summed in 2**30 chunks, exercise this
  // by testing with a 90000 rows table;
  auto schema = ::arrow::schema({::arrow::field("a", ::arrow::utf8())});
  auto seedBatch = ::arrow::RecordBatchFromJSON(schema, R"([
    [ "a" ],
    [ "bc" ],
    [ null ]
  ])");
  ASSERT_OK_AND_ASSIGN(
      auto table,
      ::arrow::Table::FromRecordBatches(
          ::arrow::RecordBatchVector(30000, seedBatch)));
  ASSERT_OK_AND_ASSIGN(table, table->CombineChunks());
  ASSERT_EQ(table->num_rows(), 90000);

  writeFile(
      SizeStatisticsLevel::PageAndColumnChunk,
      table,
      /*maxRowGroupLength=*/1 << 30,
      /*pageSize=*/1 << 30,
      /*writeBatchSize=*/50000);
  readSizeStatistics();
  EXPECT_THAT(
      rowGroupStats_,
      ::testing::ElementsAre(SizeStatistics{{30000, 60000}, {}, 90000}));
  EXPECT_THAT(
      pageStats_,
      ::testing::ElementsAre(PageSizeStatistics{{30000, 60000}, {}, {90000}}));
}

TEST_F(SizeStatisticsRoundTripTest, maxLevelZero) {
  auto schema = ::arrow::schema(
      {::arrow::field("a", ::arrow::utf8(), /*nullable=*/false)});
  writeFile(
      SizeStatisticsLevel::PageAndColumnChunk,
      ::arrow::TableFromJSON(schema, {R"([["foo"],["bar"]])"}),
      /*maxRowGroupLength=*/2,
      /*pageSize=*/1024);
  ASSERT_NO_FATAL_FAILURE(readSizeStatistics());
  ASSERT_NO_FATAL_FAILURE(readData());
  EXPECT_THAT(
      rowGroupStats_, ::testing::ElementsAre(SizeStatistics{{}, {}, 6}));
  EXPECT_THAT(
      pageStats_, ::testing::ElementsAre(PageSizeStatistics{{}, {}, {6}}));
}
} // namespace test
} // namespace facebook::velox::parquet::arrow
