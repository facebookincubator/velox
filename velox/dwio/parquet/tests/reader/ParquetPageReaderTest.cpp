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

#include "velox/dwio/parquet/reader/PageReader.h"

#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

class ParquetPageReaderTest : public ParquetTestBase {};

TEST_F(ParquetPageReaderTest, smallPage) {
  auto readFile =
      std::make_shared<LocalReadFile>(getExampleFilePath("small_page_header"));
  auto file = std::make_shared<ReadFileInputStream>(std::move(readFile));
  auto headerSize = file->getLength();
  auto inputStream = std::make_unique<SeekableFileInputStream>(
      std::move(file), 0, headerSize, *leafPool_, LogType::TEST);
  dwio::common::ColumnReaderStatistics stats;
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_GZIP,
      headerSize,
      stats);
  auto header = pageReader->readPageHeader();
  EXPECT_EQ(header.type, thrift::PageType::type::DATA_PAGE);
  EXPECT_EQ(header.uncompressed_page_size, 16950);
  EXPECT_EQ(header.compressed_page_size, 10759);
  EXPECT_EQ(header.data_page_header.num_values, 21738);

  // expectedMinValue: "aaaa...aaaa"
  std::string expectedMinValue(39, 'a');
  // expectedMaxValue: "zzzz...zzzz"
  std::string expectedMaxValue(49, 'z');
  auto minValue = header.data_page_header.statistics.min_value;
  auto maxValue = header.data_page_header.statistics.max_value;
  EXPECT_EQ(minValue, expectedMinValue);
  EXPECT_EQ(maxValue, expectedMaxValue);
  EXPECT_GT(stats.pageLoadTimeNs.sum(), 0);
}

TEST_F(ParquetPageReaderTest, largePage) {
  auto readFile =
      std::make_shared<LocalReadFile>(getExampleFilePath("large_page_header"));
  auto file = std::make_shared<ReadFileInputStream>(std::move(readFile));
  auto headerSize = file->getLength();
  auto inputStream = std::make_unique<SeekableFileInputStream>(
      std::move(file), 0, headerSize, *leafPool_, LogType::TEST);
  dwio::common::ColumnReaderStatistics stats;
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_GZIP,
      headerSize,
      stats);
  auto header = pageReader->readPageHeader();

  EXPECT_EQ(header.type, thrift::PageType::type::DATA_PAGE);
  EXPECT_EQ(header.uncompressed_page_size, 1050822);
  EXPECT_EQ(header.compressed_page_size, 66759);
  EXPECT_EQ(header.data_page_header.num_values, 970);

  // expectedMinValue: "aaaa...aaaa"
  std::string expectedMinValue(1295, 'a');
  // expectedMinValue: "zzzz...zzzz"
  std::string expectedMaxValue(2255, 'z');
  auto minValue = header.data_page_header.statistics.min_value;
  auto maxValue = header.data_page_header.statistics.max_value;
  EXPECT_EQ(minValue, expectedMinValue);
  EXPECT_EQ(maxValue, expectedMaxValue);
  EXPECT_GT(stats.pageLoadTimeNs.sum(), 0);
}

TEST_F(ParquetPageReaderTest, corruptedPageHeader) {
  auto readFile = std::make_shared<LocalReadFile>(
      getExampleFilePath("corrupted_page_header"));
  auto file = std::make_shared<ReadFileInputStream>(std::move(readFile));
  auto headerSize = file->getLength();
  auto inputStream = std::make_unique<SeekableFileInputStream>(
      std::move(file), 0, headerSize, *leafPool_, LogType::TEST);

  // In the corrupted_page_header, the min_value length is set incorrectly on
  // purpose. This is to simulate the situation where the Parquet Page Header is
  // corrupted. And an error is expected to be thrown.
  dwio::common::ColumnReaderStatistics stats;
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_GZIP,
      headerSize,
      stats);

  EXPECT_THROW(pageReader->readPageHeader(), VeloxException);
}

TEST(CompressionOptionsTest, testCompressionOptions) {
  auto options = getParquetDecompressionOptions(
      facebook::velox::common::CompressionKind_ZLIB);
  EXPECT_EQ(
      options.format.zlib.windowBits,
      dwio::common::compression::Compressor::PARQUET_ZLIB_WINDOW_BITS);
}

namespace {

// Helper to serialize a PageHeader using Thrift compact protocol.
std::string serializePageHeader(const thrift::PageHeader& header) {
  auto transport = std::make_shared<apache::thrift::transport::TMemoryBuffer>();
  apache::thrift::protocol::TCompactProtocolT<
      apache::thrift::transport::TMemoryBuffer>
      protocol(transport);
  header.write(&protocol);
  return transport->getBufferAsString();
}

// Helper to create a DATA_PAGE header with specified sizes.
thrift::PageHeader createDataPageV1Header(
    int32_t uncompressedSize,
    int32_t compressedSize,
    int32_t numValues) {
  thrift::PageHeader header;
  header.__set_type(thrift::PageType::DATA_PAGE);
  header.__set_uncompressed_page_size(uncompressedSize);
  header.__set_compressed_page_size(compressedSize);

  thrift::DataPageHeader dataHeader;
  dataHeader.__set_num_values(numValues);
  dataHeader.__set_encoding(thrift::Encoding::PLAIN);
  dataHeader.__set_definition_level_encoding(thrift::Encoding::RLE);
  dataHeader.__set_repetition_level_encoding(thrift::Encoding::RLE);
  header.__set_data_page_header(dataHeader);

  return header;
}

// Helper to create a DATA_PAGE_V2 header with specified sizes.
thrift::PageHeader createDataPageV2Header(
    int32_t uncompressedSize,
    int32_t compressedSize,
    int32_t numValues,
    int32_t definitionLevelsByteLength,
    int32_t repetitionLevelsByteLength) {
  thrift::PageHeader header;
  header.__set_type(thrift::PageType::DATA_PAGE_V2);
  header.__set_uncompressed_page_size(uncompressedSize);
  header.__set_compressed_page_size(compressedSize);

  thrift::DataPageHeaderV2 dataHeader;
  dataHeader.__set_num_values(numValues);
  dataHeader.__set_num_nulls(0);
  dataHeader.__set_num_rows(numValues);
  dataHeader.__set_encoding(thrift::Encoding::PLAIN);
  dataHeader.__set_definition_levels_byte_length(definitionLevelsByteLength);
  dataHeader.__set_repetition_levels_byte_length(repetitionLevelsByteLength);
  dataHeader.__set_is_compressed(false);
  header.__set_data_page_header_v2(dataHeader);

  return header;
}

} // namespace

// Test that prepareDataPageV1 rejects pages with defineLength exceeding page
// size. This guards against heap buffer overflow from corrupt Parquet files.
TEST_F(ParquetPageReaderTest, corruptDefineLengthV1) {
  // Create a DATA_PAGE header with small page size.
  constexpr int32_t kPageSize = 20;
  auto pageHeader = createDataPageV1Header(kPageSize, kPageSize, 100);
  std::string headerBytes = serializePageHeader(pageHeader);

  // Create corrupt page data where defineLength (first 4 bytes after
  // decompression for maxDefine > 0) is huge. Since compression is NONE, the
  // "decompressed" data is the raw page data.
  std::string pageData(kPageSize, '\0');
  // Set defineLength to a huge value (0x7FFFFFF0) that exceeds page size.
  uint32_t corruptDefineLength = 0x7FFFFFF0;
  memcpy(pageData.data(), &corruptDefineLength, sizeof(uint32_t));

  // Combine header and page data.
  std::string fullData = headerBytes + pageData;

  // Create an input stream from the crafted data.
  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  dwio::common::ColumnReaderStatistics stats;
  // Create PageReader with maxRepeat=0, maxDefine=1 (so defineLength is read)
  // and no compression (so page data is used directly).
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr,
      0,
      1);

  // Calling skip(1) triggers seekToPage() which calls prepareDataPageV1().
  // The bounds check should throw when defineLength exceeds page size.
  VELOX_ASSERT_THROW(
      pageReader->skip(1), "Definition level length 2147483632 exceeds");
}

// Test that prepareDataPageV1 rejects pages with repeatLength exceeding page
// size. This guards against heap buffer overflow from corrupt Parquet files.
TEST_F(ParquetPageReaderTest, corruptRepeatLengthV1) {
  // Create a DATA_PAGE header with small page size.
  constexpr int32_t kPageSize = 20;
  auto pageHeader = createDataPageV1Header(kPageSize, kPageSize, 100);
  std::string headerBytes = serializePageHeader(pageHeader);

  // Create corrupt page data where repeatLength (first 4 bytes after
  // decompression for maxRepeat > 0) is huge. Since compression is NONE, the
  // "decompressed" data is the raw page data.
  std::string pageData(kPageSize, '\0');
  // Set repeatLength to a huge value (0x7FFFFFF0) that exceeds page size.
  uint32_t corruptRepeatLength = 0x7FFFFFF0;
  memcpy(pageData.data(), &corruptRepeatLength, sizeof(uint32_t));

  // Combine header and page data.
  std::string fullData = headerBytes + pageData;

  // Create an input stream from the crafted data.
  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  dwio::common::ColumnReaderStatistics stats;
  // Create PageReader with maxRepeat=1 (so repeatLength is read), maxDefine=0,
  // and no compression (so page data is used directly).
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr,
      1,
      0);

  // Calling skip(1) triggers seekToPage() which calls prepareDataPageV1().
  // The bounds check should throw when repeatLength exceeds page size.
  VELOX_ASSERT_THROW(
      pageReader->skip(1), "Repetition level length 2147483632 exceeds");
}

// Test that prepareDataPageV2 rejects pages where repetition + definition
// level lengths exceed compressed page size.
TEST_F(ParquetPageReaderTest, corruptLevelLengthsV2) {
  // Create a DATA_PAGE_V2 header with small page size but huge level lengths.
  constexpr int32_t kPageSize = 20;
  // Set level lengths that exceed the page size.
  constexpr int32_t kCorruptRepeatLength = 0x7FFFFFF0;
  constexpr int32_t kCorruptDefineLength = 100;
  auto pageHeader = createDataPageV2Header(
      kPageSize, kPageSize, 100, kCorruptDefineLength, kCorruptRepeatLength);
  std::string headerBytes = serializePageHeader(pageHeader);

  // Create page data (content doesn't matter since validation should fail
  // before reading it).
  std::string pageData(kPageSize, '\0');

  // Combine header and page data.
  std::string fullData = headerBytes + pageData;

  // Create an input stream from the crafted data.
  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  dwio::common::ColumnReaderStatistics stats;
  // maxRepeat and maxDefine don't affect V2 validation since the lengths
  // are in the header, not the page data.
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr,
      1,
      1);

  // Calling skip(1) triggers seekToPage() which calls prepareDataPageV2().
  // The bounds check should throw when level lengths exceed page size.
  VELOX_ASSERT_THROW(
      pageReader->skip(1),
      "Repetition and definition level lengths (2147483632 + 100) exceed");
}

// Test that prepareDataPageV1 rejects pages that are too small to contain
// the repetition level length field (4 bytes).
TEST_F(ParquetPageReaderTest, insufficientBytesForRepeatLengthV1) {
  // Create a DATA_PAGE header with page size too small for the 4-byte
  // repetition level length field.
  constexpr int32_t kPageSize = 2; // Less than sizeof(int32_t)
  auto pageHeader = createDataPageV1Header(kPageSize, kPageSize, 100);
  std::string headerBytes = serializePageHeader(pageHeader);

  // Create minimal page data.
  std::string pageData(kPageSize, '\0');

  // Combine header and page data.
  std::string fullData = headerBytes + pageData;

  // Create an input stream from the crafted data.
  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  dwio::common::ColumnReaderStatistics stats;
  // PageReader with maxRepeat > 0 would try to read repeatLength but page
  // is too small.
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr,
      1,
      0);

  // Calling skip(1) triggers seekToPage() which calls prepareDataPageV1().
  // The bounds check should throw when page is too small to hold repeatLength.
  VELOX_ASSERT_THROW(
      pageReader->skip(1), "Insufficient bytes for repetition level length");
}

// Test that prepareDataPageV1 rejects pages where there are insufficient bytes
// remaining for the definition level length field after reading repetition
// levels.
TEST_F(ParquetPageReaderTest, insufficientBytesForDefineLengthV1) {
  // Create a DATA_PAGE header with page size that can hold repetition level
  // length (4 bytes) plus a small repeatLength value, but not enough remaining
  // for the definition level length field.
  constexpr int32_t kPageSize = 6; // 4 bytes for repeatLength field + 2 bytes
  auto pageHeader = createDataPageV1Header(kPageSize, kPageSize, 100);
  std::string headerBytes = serializePageHeader(pageHeader);

  // Create page data with a small repeatLength that leaves insufficient bytes
  // for defineLength.
  std::string pageData(kPageSize, '\0');
  // Set repeatLength to 0 (valid), leaving only 2 bytes which is insufficient
  // for the 4-byte defineLength field.
  uint32_t repeatLength = 0;
  memcpy(pageData.data(), &repeatLength, sizeof(uint32_t));

  // Combine header and page data.
  std::string fullData = headerBytes + pageData;

  // Create an input stream from the crafted data.
  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  dwio::common::ColumnReaderStatistics stats;
  // PageReader with both maxRepeat > 0 and maxDefine > 0 would read
  // repeatLength (0), advance past it, then try to read defineLength but
  // there are insufficient bytes remaining.
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr,
      1,
      1);

  // Calling skip(1) triggers seekToPage() which calls prepareDataPageV1().
  // The bounds check should throw when there are insufficient bytes for
  // defineLength.
  VELOX_ASSERT_THROW(
      pageReader->skip(1), "Insufficient bytes for definition level length");
}

// Test that prepareDataPageV2 rejects pages where only the repetition level
// length exceeds the page size (without combining with definition length).
TEST_F(ParquetPageReaderTest, corruptRepeatLengthOnlyV2) {
  // Create a DATA_PAGE_V2 header where just the repetition level length
  // exceeds the page size.
  constexpr int32_t kPageSize = 20;
  constexpr int32_t kCorruptRepeatLength = 0x7FFFFFF0;
  constexpr int32_t kDefineLength = 0; // No definition levels
  auto pageHeader = createDataPageV2Header(
      kPageSize, kPageSize, 100, kDefineLength, kCorruptRepeatLength);
  std::string headerBytes = serializePageHeader(pageHeader);

  // Create page data.
  std::string pageData(kPageSize, '\0');

  // Combine header and page data.
  std::string fullData = headerBytes + pageData;

  // Create an input stream from the crafted data.
  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  dwio::common::ColumnReaderStatistics stats;
  // maxRepeat and maxDefine don't affect V2 validation since the lengths
  // are in the header, not the page data.
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr,
      1,
      0);

  // Calling skip(1) triggers seekToPage() which calls prepareDataPageV2().
  // The bounds check should throw when repeatLength exceeds page size.
  VELOX_ASSERT_THROW(
      pageReader->skip(1),
      "Repetition and definition level lengths (2147483632 + 0) exceed");
}
