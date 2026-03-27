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

#include <thrift/lib/cpp2/protocol/Serializer.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/thrift/ParquetThrift.h"

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
  EXPECT_EQ(*header.type(), thrift::PageType::DATA_PAGE);
  EXPECT_EQ(*header.uncompressed_page_size(), 16950);
  EXPECT_EQ(*header.compressed_page_size(), 10759);
  EXPECT_EQ(*header.data_page_header()->num_values(), 21738);

  // expectedMinValue: "aaaa...aaaa"
  std::string expectedMinValue(39, 'a');
  // expectedMaxValue: "zzzz...zzzz"
  std::string expectedMaxValue(49, 'z');
  auto minValue = *header.data_page_header()->statistics()->min_value();
  auto maxValue = *header.data_page_header()->statistics()->max_value();
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

  EXPECT_EQ(*header.type(), thrift::PageType::DATA_PAGE);
  EXPECT_EQ(*header.uncompressed_page_size(), 1050822);
  EXPECT_EQ(*header.compressed_page_size(), 66759);
  EXPECT_EQ(*header.data_page_header()->num_values(), 970);

  // expectedMinValue: "aaaa...aaaa"
  std::string expectedMinValue(1295, 'a');
  // expectedMinValue: "zzzz...zzzz"
  std::string expectedMaxValue(2255, 'z');
  auto minValue = *header.data_page_header()->statistics()->min_value();
  auto maxValue = *header.data_page_header()->statistics()->max_value();
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
  return apache::thrift::CompactSerializer::serialize<std::string>(header);
}

// Helper to create a DATA_PAGE header with specified sizes.
thrift::PageHeader createDataPageV1Header(
    int32_t uncompressedSize,
    int32_t compressedSize,
    int32_t numValues) {
  thrift::PageHeader header;
  header.type() = thrift::PageType::DATA_PAGE;
  header.uncompressed_page_size() = uncompressedSize;
  header.compressed_page_size() = compressedSize;

  thrift::DataPageHeader dataHeader;
  dataHeader.num_values() = numValues;
  dataHeader.encoding() = thrift::Encoding::PLAIN;
  dataHeader.definition_level_encoding() = thrift::Encoding::RLE;
  dataHeader.repetition_level_encoding() = thrift::Encoding::RLE;
  header.data_page_header() = dataHeader;

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
  header.type() = thrift::PageType::DATA_PAGE_V2;
  header.uncompressed_page_size() = uncompressedSize;
  header.compressed_page_size() = compressedSize;

  thrift::DataPageHeaderV2 dataHeader;
  dataHeader.num_values() = numValues;
  dataHeader.num_nulls() = 0;
  dataHeader.num_rows() = numValues;
  dataHeader.encoding() = thrift::Encoding::PLAIN;
  dataHeader.definition_levels_byte_length() = definitionLevelsByteLength;
  dataHeader.repetition_levels_byte_length() = repetitionLevelsByteLength;
  dataHeader.is_compressed() = false;
  header.data_page_header_v2() = dataHeader;

  return header;
}

/// Helper to create a DICTIONARY_PAGE header with specified sizes.
thrift::PageHeader createDictionaryPageHeader(
    int32_t uncompressedSize,
    int32_t compressedSize,
    int32_t numValues) {
  thrift::PageHeader header;
  header.type() = thrift::PageType::DICTIONARY_PAGE;
  header.uncompressed_page_size() = uncompressedSize;
  header.compressed_page_size() = compressedSize;

  thrift::DictionaryPageHeader dictPageHeader;
  dictPageHeader.num_values() = numValues;
  dictPageHeader.encoding() = thrift::Encoding::PLAIN;
  header.dictionary_page_header() = dictPageHeader;

  return header;
}

} // namespace

// Test that prepareDictionary rejects FIXED_LEN_BYTE_ARRAY dictionary pages
// where the Parquet type length exceeds the Velox type length. This guards
// against heap buffer overflow from malicious Parquet files that have a patched
// precision (e.g. decimal128 with typeLength=16 but precision lowered to make
// Velox choose int64_t with cppSizeInBytes=8).
TEST_F(ParquetPageReaderTest, fixedLenByteArrayDictOverflow) {
  // Simulate the exploit: Parquet FIXED_LEN_BYTE_ARRAY with typeLength=16
  // but Velox sees SHORT_DECIMAL (precision=10) which is int64_t (8 bytes).
  // Without the check, prepareDictionary would allocate numValues*8 bytes
  // but memcpy numValues*16 bytes, causing a heap overflow.
  constexpr int32_t kNumDictValues = 4;
  constexpr int32_t kParquetTypeLength = 16;
  constexpr int32_t kDictPageSize = kNumDictValues * kParquetTypeLength;

  // Create a DICTIONARY_PAGE header.
  auto dictHeader =
      createDictionaryPageHeader(kDictPageSize, kDictPageSize, kNumDictValues);
  std::string dictHeaderBytes = serializePageHeader(dictHeader);

  // Dictionary page data (content doesn't matter, check fires before read).
  std::string dictPageData(kDictPageSize, '\0');

  // Create a DATA_PAGE header so seekToPage can find a data page after the
  // dictionary page.
  constexpr int32_t kDataPageSize = 8;
  thrift::PageHeader dataHeader;
  dataHeader.type() = thrift::PageType::DATA_PAGE;
  dataHeader.uncompressed_page_size() = kDataPageSize;
  dataHeader.compressed_page_size() = kDataPageSize;
  thrift::DataPageHeader dataPageHeader;
  dataPageHeader.num_values() = 1;
  dataPageHeader.encoding() = thrift::Encoding::RLE_DICTIONARY;
  dataPageHeader.definition_level_encoding() = thrift::Encoding::RLE;
  dataPageHeader.repetition_level_encoding() = thrift::Encoding::RLE;
  dataHeader.data_page_header() = dataPageHeader;

  std::string dataHeaderBytes = serializePageHeader(dataHeader);

  std::string dataPageData(kDataPageSize, '\0');

  // Combine: dict header + dict data + data header + data data.
  std::string fullData =
      dictHeaderBytes + dictPageData + dataHeaderBytes + dataPageData;

  auto inputStream = std::make_unique<SeekableArrayInputStream>(
      fullData.data(), fullData.size());

  // Construct ParquetTypeWithId: SHORT_DECIMAL (precision=10, cppSizeInBytes=8)
  // with parquetType=FIXED_LEN_BYTE_ARRAY and typeLength=16.
  auto fileType = std::make_shared<const ParquetTypeWithId>(
      DECIMAL(10, 2),
      std::vector<std::unique_ptr<dwio::common::TypeWithId>>{},
      /*id=*/0,
      /*maxId=*/0,
      /*column=*/0,
      "test_col",
      thrift::Type::FIXED_LEN_BYTE_ARRAY,
      std::nullopt,
      std::nullopt,
      /*maxRepeat=*/0,
      /*maxDefine=*/1,
      /*isOptional=*/true,
      /*isRepeated=*/false,
      /*precision=*/10,
      /*scale=*/2,
      /*typeLength=*/kParquetTypeLength);

  dwio::common::ColumnReaderStatistics stats;
  auto pageReader = std::make_unique<PageReader>(
      std::move(inputStream),
      *leafPool_,
      fileType,
      common::CompressionKind::CompressionKind_NONE,
      fullData.size(),
      stats,
      nullptr);

  // skip(1) triggers seekToPage() -> prepareDictionary().
  // The VELOX_CHECK_LE should fire because numParquetBytes (4*16=64) >
  // numVeloxBytes (4*8=32).
  VELOX_ASSERT_THROW(pageReader->skip(1), "");
}

// Example test demonstrating proper FBThrift dictionary page creation.
// This serves as a reference for converting any OSS-specific tests.
TEST_F(ParquetPageReaderTest, dictionaryPageExample) {
  constexpr int32_t kDictPageSize = 100;
  constexpr int32_t kNumDictValues = 10;

  // Create dictionary page header using FBThrift API
  auto dictHeader =
      createDictionaryPageHeader(kDictPageSize, kDictPageSize, kNumDictValues);

  // Verify the header was created correctly
  EXPECT_EQ(*dictHeader.type(), thrift::PageType::DICTIONARY_PAGE);
  EXPECT_EQ(*dictHeader.uncompressed_page_size(), kDictPageSize);
  EXPECT_EQ(*dictHeader.compressed_page_size(), kDictPageSize);
  EXPECT_EQ(*dictHeader.dictionary_page_header()->num_values(), kNumDictValues);
  EXPECT_EQ(
      *dictHeader.dictionary_page_header()->encoding(),
      thrift::Encoding::PLAIN);

  // Serialize using FBThrift CompactSerializer
  std::string serialized = serializePageHeader(dictHeader);
  EXPECT_GT(serialized.size(), 0);
}

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
