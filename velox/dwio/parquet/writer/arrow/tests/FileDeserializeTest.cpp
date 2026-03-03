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

// Adapted from Apache Arrow.

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/ThriftInternal.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

#include "arrow/testing/gtest_util.h"
#include "arrow/util/crc32.h"

namespace facebook::velox::parquet::arrow {
using namespace facebook::velox::common::testutil;
namespace {
void writeToFile(
    std::shared_ptr<TempFilePath> filePath,
    std::shared_ptr<arrow::Buffer> buffer) {
  auto localWriteFile =
      std::make_unique<LocalWriteFile>(filePath->getPath(), false, false);
  auto bufferReader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto bufferToString = bufferReader->buffer()->ToString();
  localWriteFile->append(bufferToString);
  localWriteFile->close();
}
} // namespace

using ::arrow::io::BufferReader;

// Adds page statistics occupying a certain amount of bytes (for testing very.
// large page headers)
template <typename H>
static inline void
addDummyStats(int statSize, H& header, bool fillAllStats = false) {
  std::vector<uint8_t> statBytes(statSize);
  // Some non-zero value.
  std::fill(statBytes.begin(), statBytes.end(), 1);
  header.statistics.__set_max(
      std::string(reinterpret_cast<const char*>(statBytes.data()), statSize));

  if (fillAllStats) {
    header.statistics.__set_min(
        std::string(reinterpret_cast<const char*>(statBytes.data()), statSize));
    header.statistics.__set_null_count(42);
    header.statistics.__set_distinct_count(1);
  }

  header.__isset.statistics = true;
}

template <typename H>
static inline void checkStatistics(
    const H& expected,
    const EncodedStatistics& actual) {
  if (expected.statistics.__isset.max) {
    ASSERT_EQ(expected.statistics.max, actual.max());
  }
  if (expected.statistics.__isset.min) {
    ASSERT_EQ(expected.statistics.min, actual.min());
  }
  if (expected.statistics.__isset.null_count) {
    ASSERT_EQ(expected.statistics.null_count, actual.nullCount);
  }
  if (expected.statistics.__isset.distinct_count) {
    ASSERT_EQ(expected.statistics.distinct_count, actual.distinctCount);
  }
}

static std::vector<Compression::type> getSupportedCodecTypes() {
  std::vector<Compression::type> codecTypes;

  codecTypes.push_back(Compression::SNAPPY);

#ifdef ARROW_WITH_BROTLI
  codecTypes.push_back(Compression::BROTLI);
#endif

  codecTypes.push_back(Compression::GZIP);

  codecTypes.push_back(Compression::LZ4);
  codecTypes.push_back(Compression::LZ4_HADOOP);

  codecTypes.push_back(Compression::ZSTD);
  return codecTypes;
}

class TestPageSerde : public ::testing::Test {
 public:
  void SetUp() {
    dataPageHeader_.encoding =
        facebook::velox::parquet::thrift::Encoding::PLAIN;
    dataPageHeader_.definition_level_encoding =
        facebook::velox::parquet::thrift::Encoding::RLE;
    dataPageHeader_.repetition_level_encoding =
        facebook::velox::parquet::thrift::Encoding::RLE;

    resetStream();
  }

  void initSerializedPageReader(
      int64_t numRows,
      Compression::type Codec = Compression::UNCOMPRESSED,
      const ReaderProperties& properties = ReaderProperties()) {
    endStream();

    auto stream = std::make_shared<::arrow::io::BufferReader>(outBuffer_);
    pageReader_ = PageReader::open(stream, numRows, Codec, properties);
  }

  void writeDataPageHeader(
      int maxSerializedLen = 1024,
      int32_t uncompressedSize = 0,
      int32_t compressedSize = 0,
      std::optional<int32_t> checksum = std::nullopt) {
    // Simplifying writing serialized data page headers which may or may not.
    // Have meaningful data associated with them.

    // Serialize the Page header.
    pageHeader_.__set_data_page_header(dataPageHeader_);
    pageHeader_.uncompressed_page_size = uncompressedSize;
    pageHeader_.compressed_page_size = compressedSize;
    pageHeader_.type = facebook::velox::parquet::thrift::PageType::DATA_PAGE;
    if (checksum.has_value()) {
      pageHeader_.__set_crc(checksum.value());
    }

    ThriftSerializer serializer;
    ASSERT_NO_THROW(serializer.serialize(&pageHeader_, outStream_.get()));
  }

  void writeDataPageHeaderV2(
      int maxSerializedLen = 1024,
      int32_t uncompressedSize = 0,
      int32_t compressedSize = 0,
      std::optional<int32_t> checksum = std::nullopt) {
    // Simplifying writing serialized data page V2 headers which may or may not.
    // Have meaningful data associated with them.

    // Serialize the Page header.
    pageHeader_.__set_data_page_header_v2(dataPageHeaderV2_);
    pageHeader_.uncompressed_page_size = uncompressedSize;
    pageHeader_.compressed_page_size = compressedSize;
    pageHeader_.type = facebook::velox::parquet::thrift::PageType::DATA_PAGE_V2;
    if (checksum.has_value()) {
      pageHeader_.__set_crc(checksum.value());
    }

    ThriftSerializer serializer;
    ASSERT_NO_THROW(serializer.serialize(&pageHeader_, outStream_.get()));
  }

  void writeDictionaryPageHeader(
      int32_t uncompressedSize = 0,
      int32_t compressedSize = 0,
      std::optional<int32_t> checksum = std::nullopt) {
    pageHeader_.__set_dictionary_page_header(dictionaryPageHeader_);
    pageHeader_.uncompressed_page_size = uncompressedSize;
    pageHeader_.compressed_page_size = compressedSize;
    pageHeader_.type =
        facebook::velox::parquet::thrift::PageType::DICTIONARY_PAGE;
    if (checksum.has_value()) {
      pageHeader_.__set_crc(checksum.value());
    }

    ThriftSerializer serializer;
    ASSERT_NO_THROW(serializer.serialize(&pageHeader_, outStream_.get()));
  }

  void writeIndexPageHeader(
      int32_t uncompressedSize = 0,
      int32_t compressedSize = 0) {
    pageHeader_.__set_index_page_header(indexPageHeader_);
    pageHeader_.uncompressed_page_size = uncompressedSize;
    pageHeader_.compressed_page_size = compressedSize;
    pageHeader_.type = facebook::velox::parquet::thrift::PageType::INDEX_PAGE;

    ThriftSerializer serializer;
    ASSERT_NO_THROW(serializer.serialize(&pageHeader_, outStream_.get()));
  }

  void resetStream() {
    outStream_ = createOutputStream();
  }

  void endStream() {
    PARQUET_ASSIGN_OR_THROW(outBuffer_, outStream_->Finish());
  }

  void testPageSerdeCrc(
      bool writeChecksum,
      bool writePageCorrupt,
      bool verificationChecksum,
      bool hasDictionary = false,
      bool writeDataPageV2 = false);

  void testPageCompressionRoundTrip(const std::vector<int>& pageSizes);

 protected:
  std::shared_ptr<::arrow::io::BufferOutputStream> outStream_;
  std::shared_ptr<Buffer> outBuffer_;

  std::unique_ptr<PageReader> pageReader_;
  facebook::velox::parquet::thrift::PageHeader pageHeader_;
  facebook::velox::parquet::thrift::DataPageHeader dataPageHeader_;
  facebook::velox::parquet::thrift::DataPageHeaderV2 dataPageHeaderV2_;
  facebook::velox::parquet::thrift::IndexPageHeader indexPageHeader_;
  facebook::velox::parquet::thrift::DictionaryPageHeader dictionaryPageHeader_;
};

void TestPageSerde::testPageSerdeCrc(
    bool writeChecksum,
    bool writePageCorrupt,
    bool verificationChecksum,
    bool hasDictionary,
    bool writeDataPageV2) {
  auto codecTypes = getSupportedCodecTypes();
  codecTypes.push_back(Compression::UNCOMPRESSED);
  const int32_t numRows = 32; // dummy value
  if (writeDataPageV2) {
    dataPageHeaderV2_.num_values = numRows;
  } else {
    dataPageHeader_.num_values = numRows;
  }
  dictionaryPageHeader_.num_values = numRows;

  const int numPages = 10;

  std::vector<std::vector<uint8_t>> fauxData;
  fauxData.resize(numPages);
  for (int i = 0; i < numPages; ++i) {
    // The pages keep getting larger.
    int pageSize = (i + 1) * 64;
    test::randomBytes(pageSize, 0, &fauxData[i]);
  }
  for (auto codecType : codecTypes) {
    auto Codec = getCodec(codecType);

    std::vector<uint8_t> buffer;
    for (int i = 0; i < numPages; ++i) {
      const uint8_t* data = fauxData[i].data();
      int dataSize = static_cast<int>(fauxData[i].size());
      int64_t actualSize;
      if (Codec == nullptr) {
        buffer = fauxData[i];
        actualSize = dataSize;
      } else {
        int64_t maxCompressedSize = Codec->maxCompressedLen(dataSize, data);
        buffer.resize(maxCompressedSize);

        ASSERT_OK_AND_ASSIGN(
            actualSize,
            Codec->compress(dataSize, data, maxCompressedSize, &buffer[0]));
      }
      std::optional<uint32_t> checksumOpt;
      if (writeChecksum) {
        uint32_t checksum =
            ::arrow::internal::crc32(/* prev */ 0, buffer.data(), actualSize);
        if (writePageCorrupt) {
          checksum += 1; // write a bad checksum
        }
        checksumOpt = checksum;
      }
      if (hasDictionary && i == 0) {
        ASSERT_NO_FATAL_FAILURE(writeDictionaryPageHeader(
            dataSize, static_cast<int32_t>(actualSize), checksumOpt));
      } else {
        if (writeDataPageV2) {
          ASSERT_NO_FATAL_FAILURE(writeDataPageHeaderV2(
              1024, dataSize, static_cast<int32_t>(actualSize), checksumOpt));
        } else {
          ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(
              1024, dataSize, static_cast<int32_t>(actualSize), checksumOpt));
        }
      }
      ASSERT_OK(outStream_->Write(buffer.data(), actualSize));
    }
    ReaderProperties ReaderProperties;
    ReaderProperties.setPageChecksumVerification(verificationChecksum);
    initSerializedPageReader(numRows * numPages, codecType, ReaderProperties);

    for (int i = 0; i < numPages; ++i) {
      if (writeChecksum && writePageCorrupt && verificationChecksum) {
        EXPECT_THROW_THAT(
            [&]() { pageReader_->nextPage(); },
            ParquetException,
            ::testing::Property(
                &ParquetException::what,
                ::testing::HasSubstr("CRC checksum verification failed")));
      } else {
        const auto page = pageReader_->nextPage();
        const int dataSize = static_cast<int>(fauxData[i].size());
        if (hasDictionary && i == 0) {
          ASSERT_EQ(PageType::kDictionaryPage, page->type());
          const auto dictPage = static_cast<const DictionaryPage*>(page.get());
          ASSERT_EQ(dataSize, dictPage->size());
          ASSERT_EQ(0, memcmp(fauxData[i].data(), dictPage->data(), dataSize));
        } else if (writeDataPageV2) {
          ASSERT_EQ(PageType::kDataPageV2, page->type());
          const auto dataPage = static_cast<const DataPageV2*>(page.get());
          ASSERT_EQ(dataSize, dataPage->size());
          ASSERT_EQ(0, memcmp(fauxData[i].data(), dataPage->data(), dataSize));
        } else {
          ASSERT_EQ(PageType::kDataPage, page->type());
          const auto dataPage = static_cast<const DataPageV1*>(page.get());
          ASSERT_EQ(dataSize, dataPage->size());
          ASSERT_EQ(0, memcmp(fauxData[i].data(), dataPage->data(), dataSize));
        }
      }
    }

    resetStream();
  }
}

void checkDataPageHeader(
    const facebook::velox::parquet::thrift::DataPageHeader& expected,
    const Page* page) {
  ASSERT_EQ(PageType::kDataPage, page->type());

  const DataPageV1* dataPage = static_cast<const DataPageV1*>(page);
  ASSERT_EQ(expected.num_values, dataPage->numValues());
  ASSERT_EQ(expected.encoding, dataPage->encoding());
  ASSERT_EQ(
      expected.definition_level_encoding, dataPage->definitionLevelEncoding());
  ASSERT_EQ(
      expected.repetition_level_encoding, dataPage->repetitionLevelEncoding());
  checkStatistics(expected, dataPage->statistics());
}

// Overload for DataPageV2 tests.
void checkDataPageHeader(
    const facebook::velox::parquet::thrift::DataPageHeaderV2& expected,
    const Page* page) {
  ASSERT_EQ(PageType::kDataPageV2, page->type());

  const DataPageV2* dataPage = static_cast<const DataPageV2*>(page);
  ASSERT_EQ(expected.num_values, dataPage->numValues());
  ASSERT_EQ(expected.num_nulls, dataPage->numNulls());
  ASSERT_EQ(expected.num_rows, dataPage->numRows());
  ASSERT_EQ(expected.encoding, dataPage->encoding());
  ASSERT_EQ(
      expected.definition_levels_byte_length,
      dataPage->definitionLevelsByteLength());
  ASSERT_EQ(
      expected.repetition_levels_byte_length,
      dataPage->repetitionLevelsByteLength());
  ASSERT_EQ(expected.is_compressed, dataPage->isCompressed());
  checkStatistics(expected, dataPage->statistics());
}

TEST_F(TestPageSerde, DataPageV1) {
  int statsSize = 512;
  const int32_t numRows = 4444;
  addDummyStats(statsSize, dataPageHeader_, /*fill_all_stats=*/true);
  dataPageHeader_.num_values = numRows;

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader());
  initSerializedPageReader(numRows);
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();
  ASSERT_NO_FATAL_FAILURE(
      checkDataPageHeader(dataPageHeader_, currentPage.get()));
}

// Templated test class to test page filtering for both.
// Facebook::velox::parquet::thrift::DataPageHeader and.
// Facebook::velox::parquet::thrift::DataPageHeaderV2.
template <typename T>
class PageFilterTest : public TestPageSerde {
 public:
  const int kNumPages = 10;
  void writeStream();
  void writePageWithoutStats();
  void checkNumRows(std::optional<int32_t> numRows, const T& header);

 protected:
  std::vector<T> dataPageHeaders_;
  int totalRows_ = 0;
};

template <>
void PageFilterTest<
    facebook::velox::parquet::thrift::DataPageHeader>::writeStream() {
  for (int i = 0; i < kNumPages; ++i) {
    // Vary the number of rows to produce different headers.
    int32_t numRows = i + 100;
    totalRows_ += numRows;
    int dataSize = i + 1024;
    this->dataPageHeader_.__set_num_values(numRows);
    this->dataPageHeader_.statistics.__set_min_value("A" + std::to_string(i));
    this->dataPageHeader_.statistics.__set_max_value("Z" + std::to_string(i));
    this->dataPageHeader_.statistics.__set_null_count(0);
    this->dataPageHeader_.statistics.__set_distinct_count(numRows);
    this->dataPageHeader_.__isset.statistics = true;
    ASSERT_NO_FATAL_FAILURE(
        this->writeDataPageHeader(1024, dataSize, dataSize));
    dataPageHeaders_.push_back(this->dataPageHeader_);
    // Also write data, to make sure we skip the data correctly.
    std::vector<uint8_t> fauxData(dataSize);
    ASSERT_OK(this->outStream_->Write(fauxData.data(), dataSize));
  }
  this->endStream();
}

template <>
void PageFilterTest<
    facebook::velox::parquet::thrift::DataPageHeaderV2>::writeStream() {
  for (int i = 0; i < kNumPages; ++i) {
    // Vary the number of rows to produce different headers.
    int32_t numRows = i + 100;
    totalRows_ += numRows;
    int dataSize = i + 1024;
    this->dataPageHeaderV2_.__set_num_values(numRows);
    this->dataPageHeaderV2_.__set_num_rows(numRows);
    this->dataPageHeaderV2_.statistics.__set_min_value("A" + std::to_string(i));
    this->dataPageHeaderV2_.statistics.__set_max_value("Z" + std::to_string(i));
    this->dataPageHeaderV2_.statistics.__set_null_count(0);
    this->dataPageHeaderV2_.statistics.__set_distinct_count(numRows);
    this->dataPageHeaderV2_.__isset.statistics = true;
    ASSERT_NO_FATAL_FAILURE(
        this->writeDataPageHeaderV2(1024, dataSize, dataSize));
    dataPageHeaders_.push_back(this->dataPageHeaderV2_);
    // Also write data, to make sure we skip the data correctly.
    std::vector<uint8_t> fauxData(dataSize);
    ASSERT_OK(this->outStream_->Write(fauxData.data(), dataSize));
  }
  this->endStream();
}

template <>
void PageFilterTest<
    facebook::velox::parquet::thrift::DataPageHeader>::writePageWithoutStats() {
  int32_t numRows = 100;
  totalRows_ += numRows;
  int dataSize = 1024;
  this->dataPageHeader_.__set_num_values(numRows);
  ASSERT_NO_FATAL_FAILURE(this->writeDataPageHeader(1024, dataSize, dataSize));
  dataPageHeaders_.push_back(this->dataPageHeader_);
  std::vector<uint8_t> fauxData(dataSize);
  ASSERT_OK(this->outStream_->Write(fauxData.data(), dataSize));
  this->endStream();
}

template <>
void PageFilterTest<facebook::velox::parquet::thrift::DataPageHeaderV2>::
    writePageWithoutStats() {
  int32_t numRows = 100;
  totalRows_ += numRows;
  int dataSize = 1024;
  this->dataPageHeaderV2_.__set_num_values(numRows);
  this->dataPageHeaderV2_.__set_num_rows(numRows);
  ASSERT_NO_FATAL_FAILURE(
      this->writeDataPageHeaderV2(1024, dataSize, dataSize));
  dataPageHeaders_.push_back(this->dataPageHeaderV2_);
  std::vector<uint8_t> fauxData(dataSize);
  ASSERT_OK(this->outStream_->Write(fauxData.data(), dataSize));
  this->endStream();
}

template <>
void PageFilterTest<facebook::velox::parquet::thrift::DataPageHeader>::
    checkNumRows(
        std::optional<int32_t> numRows,
        const facebook::velox::parquet::thrift::DataPageHeader& header) {
  ASSERT_EQ(numRows, std::nullopt);
}

template <>
void PageFilterTest<facebook::velox::parquet::thrift::DataPageHeaderV2>::
    checkNumRows(
        std::optional<int32_t> numRows,
        const facebook::velox::parquet::thrift::DataPageHeaderV2& header) {
  ASSERT_EQ(*numRows, header.num_rows);
}

using DataPageHeaderTypes = ::testing::Types<
    facebook::velox::parquet::thrift::DataPageHeader,
    facebook::velox::parquet::thrift::DataPageHeaderV2>;
TYPED_TEST_SUITE(PageFilterTest, DataPageHeaderTypes);

// Test that the returned encoded_statistics is nullptr when there are no.
// Statistics in the page header.
TYPED_TEST(PageFilterTest, TestPageWithoutStatistics) {
  this->writePageWithoutStats();

  auto stream = std::make_shared<::arrow::io::BufferReader>(this->outBuffer_);
  this->pageReader_ =
      PageReader::open(stream, this->totalRows_, Compression::UNCOMPRESSED);

  int numPages = 0;
  bool isStatsNull = false;
  auto readAllPages = [&](const DataPageStats& stats) -> bool {
    isStatsNull = stats.encodedStatistics == nullptr;
    ++numPages;
    return false;
  };

  this->pageReader_->setDataPageFilter(readAllPages);
  std::shared_ptr<Page> currentPage = this->pageReader_->nextPage();
  ASSERT_EQ(numPages, 1);
  ASSERT_EQ(isStatsNull, true);
  ASSERT_EQ(this->pageReader_->nextPage(), nullptr);
}

// Creates a number of pages and skips some of them with the page filter.
// Callback.
TYPED_TEST(PageFilterTest, TestPageFilterCallback) {
  this->writeStream();

  { // Read all pages.
    // Also check that the encoded statistics passed to the callback function.
    // Are right.
    auto stream = std::make_shared<::arrow::io::BufferReader>(this->outBuffer_);
    this->pageReader_ =
        PageReader::open(stream, this->totalRows_, Compression::UNCOMPRESSED);

    std::vector<EncodedStatistics> readStats;
    std::vector<int64_t> readNumValues;
    std::vector<std::optional<int32_t>> readNumRows;
    auto readAllPages = [&](const DataPageStats& stats) -> bool {
      VELOX_DCHECK_NOT_NULL(stats.encodedStatistics);
      readStats.push_back(*stats.encodedStatistics);
      readNumValues.push_back(stats.numValues);
      readNumRows.push_back(stats.numRows);
      return false;
    };

    this->pageReader_->setDataPageFilter(readAllPages);
    for (int i = 0; i < this->kNumPages; ++i) {
      std::shared_ptr<Page> currentPage = this->pageReader_->nextPage();
      ASSERT_NE(currentPage, nullptr);
      ASSERT_NO_FATAL_FAILURE(
          checkDataPageHeader(this->dataPageHeaders_[i], currentPage.get()));
      auto dataPage = static_cast<const DataPage*>(currentPage.get());
      const EncodedStatistics EncodedStatistics = dataPage->statistics();
      ASSERT_EQ(readStats[i].max(), EncodedStatistics.max());
      ASSERT_EQ(readStats[i].min(), EncodedStatistics.min());
      ASSERT_EQ(readStats[i].nullCount, EncodedStatistics.nullCount);
      ASSERT_EQ(readStats[i].distinctCount, EncodedStatistics.distinctCount);
      ASSERT_EQ(readNumValues[i], this->dataPageHeaders_[i].num_values);
      this->checkNumRows(readNumRows[i], this->dataPageHeaders_[i]);
    }
    ASSERT_EQ(this->pageReader_->nextPage(), nullptr);
  }

  { // Skip all pages.
    auto stream = std::make_shared<::arrow::io::BufferReader>(this->outBuffer_);
    this->pageReader_ =
        PageReader::open(stream, this->totalRows_, Compression::UNCOMPRESSED);

    auto skipAllPages = [](const DataPageStats& stats) -> bool { return true; };

    this->pageReader_->setDataPageFilter(skipAllPages);
    std::shared_ptr<Page> currentPage = this->pageReader_->nextPage();
    ASSERT_EQ(this->pageReader_->nextPage(), nullptr);
  }

  { // Skip every other page.
    auto stream = std::make_shared<::arrow::io::BufferReader>(this->outBuffer_);
    this->pageReader_ =
        PageReader::open(stream, this->totalRows_, Compression::UNCOMPRESSED);

    // Skip pages with even number of values.
    auto skipEvenPages = [](const DataPageStats& stats) -> bool {
      if (stats.numValues % 2 == 0)
        return true;
      return false;
    };

    this->pageReader_->setDataPageFilter(skipEvenPages);

    for (int i = 0; i < this->kNumPages; ++i) {
      // Only pages with odd number of values are read.
      if (i % 2 != 0) {
        std::shared_ptr<Page> currentPage = this->pageReader_->nextPage();
        ASSERT_NE(currentPage, nullptr);
        ASSERT_NO_FATAL_FAILURE(
            checkDataPageHeader(this->dataPageHeaders_[i], currentPage.get()));
      }
    }
    // We should have exhausted reading the pages by reading the odd pages only.
    ASSERT_EQ(this->pageReader_->nextPage(), nullptr);
  }
}

// Set the page filter more than once. The new filter should be effective.
// On the next NextPage() call.
TYPED_TEST(PageFilterTest, TestChangingPageFilter) {
  this->writeStream();

  auto stream = std::make_shared<::arrow::io::BufferReader>(this->outBuffer_);
  this->pageReader_ =
      PageReader::open(stream, this->totalRows_, Compression::UNCOMPRESSED);

  // This callback will always return false.
  auto readAllPages = [](const DataPageStats& stats) -> bool { return false; };
  this->pageReader_->setDataPageFilter(readAllPages);
  std::shared_ptr<Page> currentPage = this->pageReader_->nextPage();
  ASSERT_NE(currentPage, nullptr);
  ASSERT_NO_FATAL_FAILURE(
      checkDataPageHeader(this->dataPageHeaders_[0], currentPage.get()));

  // This callback will skip all pages.
  auto skipAllPages = [](const DataPageStats& stats) -> bool { return true; };
  this->pageReader_->setDataPageFilter(skipAllPages);
  ASSERT_EQ(this->pageReader_->nextPage(), nullptr);
}

// Test that we do not skip dictionary pages.
TEST_F(TestPageSerde, DoesNotFilterDictionaryPages) {
  int dataSize = 1024;
  std::vector<uint8_t> fauxData(dataSize);

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(1024, dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));

  ASSERT_NO_FATAL_FAILURE(writeDictionaryPageHeader(dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(1024, dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));
  endStream();

  // Try to read it back while asking for all data pages to be skipped.
  auto stream = std::make_shared<::arrow::io::BufferReader>(outBuffer_);
  pageReader_ = PageReader::open(stream, 100, Compression::UNCOMPRESSED);

  auto skipAllPages = [](const DataPageStats& stats) -> bool { return true; };

  pageReader_->setDataPageFilter(skipAllPages);
  // The first data page is skipped, so we are now at the dictionary page.
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();
  ASSERT_NE(currentPage, nullptr);
  ASSERT_EQ(currentPage->type(), PageType::kDictionaryPage);
  // The data page after dictionary page is skipped.
  ASSERT_EQ(pageReader_->nextPage(), nullptr);
}

// Tests that we successfully skip non-data pages.
TEST_F(TestPageSerde, SkipsNonDataPages) {
  int dataSize = 1024;
  std::vector<uint8_t> fauxData(dataSize);
  ASSERT_NO_FATAL_FAILURE(writeIndexPageHeader(dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(1024, dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));

  ASSERT_NO_FATAL_FAILURE(writeIndexPageHeader(dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));
  ASSERT_NO_FATAL_FAILURE(writeIndexPageHeader(dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(1024, dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));
  ASSERT_NO_FATAL_FAILURE(writeIndexPageHeader(dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));
  endStream();

  auto stream = std::make_shared<::arrow::io::BufferReader>(outBuffer_);
  pageReader_ = PageReader::open(stream, 100, Compression::UNCOMPRESSED);

  // Only the two data pages are returned.
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();
  ASSERT_EQ(currentPage->type(), PageType::kDataPage);
  currentPage = pageReader_->nextPage();
  ASSERT_EQ(currentPage->type(), PageType::kDataPage);
  ASSERT_EQ(pageReader_->nextPage(), nullptr);
}

TEST_F(TestPageSerde, DataPageV2) {
  int statsSize = 512;
  const int32_t numRows = 4444;
  addDummyStats(statsSize, dataPageHeaderV2_, /*fill_all_stats=*/true);
  dataPageHeaderV2_.num_values = numRows;

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeaderV2());
  initSerializedPageReader(numRows);
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();
  ASSERT_NO_FATAL_FAILURE(
      checkDataPageHeader(dataPageHeaderV2_, currentPage.get()));
}

TEST_F(TestPageSerde, TestLargePageHeaders) {
  int statsSize = 256 * 1024; // 256 KB
  addDummyStats(statsSize, dataPageHeader_);

  // Any number to verify metadata roundtrip.
  const int32_t numRows = 4141;
  dataPageHeader_.num_values = numRows;

  int maxHeaderSize = 512 * 1024; // 512 KB
  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(maxHeaderSize));

  ASSERT_OK_AND_ASSIGN(int64_t position, outStream_->Tell());
  ASSERT_GE(maxHeaderSize, position);

  // Check header size is between 256 KB to 16 MB.
  ASSERT_LE(statsSize, position);
  ASSERT_GE(kDefaultMaxPageHeaderSize, position);

  initSerializedPageReader(numRows);
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();
  ASSERT_NO_FATAL_FAILURE(
      checkDataPageHeader(dataPageHeader_, currentPage.get()));
}

TEST_F(TestPageSerde, TestFailLargePageHeaders) {
  const int32_t numRows = 1337; // dummy value

  int statsSize = 256 * 1024; // 256 KB
  addDummyStats(statsSize, dataPageHeader_);

  // Serialize the Page header.
  int maxHeaderSize = 512 * 1024; // 512 KB
  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(maxHeaderSize));
  ASSERT_OK_AND_ASSIGN(int64_t position, outStream_->Tell());
  ASSERT_GE(maxHeaderSize, position);

  int smallerMaxSize = 128 * 1024;
  ASSERT_LE(smallerMaxSize, position);
  initSerializedPageReader(numRows);

  // Set the max page header size to 128 KB, which is less than the current.
  // Header size.
  pageReader_->setMaxPageHeaderSize(smallerMaxSize);
  ASSERT_THROW(pageReader_->nextPage(), ParquetException);
}

void TestPageSerde::testPageCompressionRoundTrip(
    const std::vector<int>& pageSizes) {
  auto codecTypes = getSupportedCodecTypes();

  const int32_t numRows = 32; // dummy value
  dataPageHeader_.num_values = numRows;

  std::vector<std::vector<uint8_t>> fauxData;
  int numPages = static_cast<int>(pageSizes.size());
  fauxData.resize(numPages);
  for (int i = 0; i < numPages; ++i) {
    test::randomBytes(pageSizes[i], 0, &fauxData[i]);
  }
  for (auto codecType : codecTypes) {
    auto Codec = getCodec(codecType);

    std::vector<uint8_t> buffer;
    for (int i = 0; i < numPages; ++i) {
      const uint8_t* data = fauxData[i].data();
      int dataSize = static_cast<int>(fauxData[i].size());

      int64_t maxCompressedSize = Codec->maxCompressedLen(dataSize, data);
      buffer.resize(maxCompressedSize);

      int64_t actualSize;
      ASSERT_OK_AND_ASSIGN(
          actualSize,
          Codec->compress(dataSize, data, maxCompressedSize, &buffer[0]));

      ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(
          1024, dataSize, static_cast<int32_t>(actualSize)));
      ASSERT_OK(outStream_->Write(buffer.data(), actualSize));
    }

    initSerializedPageReader(numRows * numPages, codecType);

    std::shared_ptr<Page> page;
    const DataPageV1* dataPage;
    for (int i = 0; i < numPages; ++i) {
      int dataSize = static_cast<int>(fauxData[i].size());
      page = pageReader_->nextPage();
      dataPage = static_cast<const DataPageV1*>(page.get());
      ASSERT_EQ(dataSize, dataPage->size());
      ASSERT_EQ(0, memcmp(fauxData[i].data(), dataPage->data(), dataSize));
    }

    resetStream();
  }
}

TEST_F(TestPageSerde, Compression) {
  std::vector<int> pageSizes;
  pageSizes.reserve(10);
  for (int i = 0; i < 10; ++i) {
    // The pages keep getting larger.
    pageSizes.push_back((i + 1) * 64);
  }
  this->testPageCompressionRoundTrip(pageSizes);
}

TEST_F(TestPageSerde, PageSizeResetWhenRead) {
  // GH-35423: Parquet SerializedPageReader need to.
  // Reset the size after getting a smaller page.
  std::vector<int> pageSizes;
  pageSizes.reserve(10);
  for (int i = 0; i < 10; ++i) {
    // The pages keep getting smaller.
    pageSizes.push_back((10 - i) * 64);
  }
  this->testPageCompressionRoundTrip(pageSizes);
}

TEST_F(TestPageSerde, LZONotSupported) {
  // Must await PARQUET-530.
  int dataSize = 1024;
  std::vector<uint8_t> fauxData(dataSize);
  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader(1024, dataSize, dataSize));
  ASSERT_OK(outStream_->Write(fauxData.data(), dataSize));
  ASSERT_THROW(
      initSerializedPageReader(dataSize, Compression::LZO), ParquetException);
}

TEST_F(TestPageSerde, NoCrc) {
  int statsSize = 512;
  const int32_t numRows = 4444;
  addDummyStats(statsSize, dataPageHeader_, true);
  dataPageHeader_.num_values = numRows;

  ASSERT_NO_FATAL_FAILURE(writeDataPageHeader());
  ReaderProperties ReaderProperties;
  ReaderProperties.setPageChecksumVerification(true);
  initSerializedPageReader(
      numRows, Compression::UNCOMPRESSED, ReaderProperties);
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();
  ASSERT_NO_FATAL_FAILURE(
      checkDataPageHeader(dataPageHeader_, currentPage.get()));
}

TEST_F(TestPageSerde, NoCrcDict) {
  const int32_t numRows = 4444;
  dictionaryPageHeader_.num_values = numRows;

  ASSERT_NO_FATAL_FAILURE(writeDictionaryPageHeader());
  ReaderProperties ReaderProperties;
  ReaderProperties.setPageChecksumVerification(true);
  initSerializedPageReader(
      numRows, Compression::UNCOMPRESSED, ReaderProperties);
  std::shared_ptr<Page> currentPage = pageReader_->nextPage();

  ASSERT_EQ(PageType::kDictionaryPage, currentPage->type());

  const auto* dictPage = static_cast<const DictionaryPage*>(currentPage.get());
  EXPECT_EQ(numRows, dictPage->numValues());
}

TEST_F(TestPageSerde, CrcCheckSuccessful) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ false,
      /* verification_checksum */ true);
}

TEST_F(TestPageSerde, CrcCheckFail) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ true,
      /* verification_checksum */ true);
}

TEST_F(TestPageSerde, CrcCorruptNotChecked) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ true,
      /* verification_checksum */ false);
}

TEST_F(TestPageSerde, CrcCheckNonExistent) {
  this->testPageSerdeCrc(
      /* write_checksum */ false,
      /* write_page_corrupt */ false,
      /* verification_checksum */ true);
}

TEST_F(TestPageSerde, DictCrcCheckSuccessful) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ false,
      /* verification_checksum */ true,
      /* has_dictionary */ true);
}

TEST_F(TestPageSerde, DictCrcCheckFail) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ true,
      /* verification_checksum */ true,
      /* has_dictionary */ true);
}

TEST_F(TestPageSerde, DictCrcCorruptNotChecked) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ true,
      /* verification_checksum */ false,
      /* has_dictionary */ true);
}

TEST_F(TestPageSerde, DictCrcCheckNonExistent) {
  this->testPageSerdeCrc(
      /* write_checksum */ false,
      /* write_page_corrupt */ false,
      /* verification_checksum */ true,
      /* has_dictionary */ true);
}

TEST_F(TestPageSerde, DataPageV2CrcCheckSuccessful) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ false,
      /* verification_checksum */ true,
      /* has_dictionary */ false,
      /* write_data_page_v2 */ true);
}

TEST_F(TestPageSerde, DataPageV2CrcCheckFail) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ true,
      /* verification_checksum */ true,
      /* has_dictionary */ false,
      /* write_data_page_v2 */ true);
}

TEST_F(TestPageSerde, DataPageV2CrcCorruptNotChecked) {
  this->testPageSerdeCrc(
      /* write_checksum */ true,
      /* write_page_corrupt */ true,
      /* verification_checksum */ false,
      /* has_dictionary */ false,
      /* write_data_page_v2 */ true);
}

TEST_F(TestPageSerde, DataPageV2CrcCheckNonExistent) {
  this->testPageSerdeCrc(
      /* write_checksum */ false,
      /* write_page_corrupt */ false,
      /* verification_checksum */ true,
      /* has_dictionary */ false,
      /* write_data_page_v2 */ true);
}

// ----------------------------------------------------------------------.
// File structure tests.

class TestParquetFileReader : public ::testing::Test {
 public:
  void assertInvalidFileThrows(const std::shared_ptr<Buffer>& buffer) {
    auto reader = std::make_shared<BufferReader>(buffer);
    // Write the buffer to a temp file path.
    auto filePath = common::testutil::TempFilePath::create();
    writeToFile(filePath, buffer);
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
        memory::memoryManager()->addRootPool("MetadataTest");
    std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
        rootPool->addLeafChild("MetadataTest");
    dwio::common::ReaderOptions readerOptions{leafPool.get()};
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePath->getPath()),
        readerOptions.memoryPool());
    uint64_t fileSize = std::move(input)->getReadFile()->size();
    VELOX_ASSERT_THROW(
        std::make_unique<ParquetReader>(std::move(input), readerOptions),
        fmt::format("({} vs. 12) Parquet file is too small", fileSize));
  }
};

TEST_F(TestParquetFileReader, InvalidHeader) {
  const char* badHeader = "PAR2";

  auto buffer = Buffer::Wrap(badHeader, strlen(badHeader));
  ASSERT_NO_FATAL_FAILURE(assertInvalidFileThrows(buffer));
}

TEST_F(TestParquetFileReader, InvalidFooter) {
  // File is smaller than FOOTER_SIZE.
  const char* badFile = "PAR1PAR";
  auto buffer = Buffer::Wrap(badFile, strlen(badFile));
  ASSERT_NO_FATAL_FAILURE(assertInvalidFileThrows(buffer));

  // Magic number incorrect.
  const char* badFile2 = "PAR1PAR2";
  buffer = Buffer::Wrap(badFile2, strlen(badFile2));
  ASSERT_NO_FATAL_FAILURE(assertInvalidFileThrows(buffer));
}

TEST_F(TestParquetFileReader, IncompleteMetadata) {
  auto stream = createOutputStream();

  const char* magic = "PAR1";

  ASSERT_OK(
      stream->Write(reinterpret_cast<const uint8_t*>(magic), strlen(magic)));
  std::vector<uint8_t> bytes(10);
  ASSERT_OK(stream->Write(bytes.data(), bytes.size()));
  uint32_t metadataLen = 24;
  ASSERT_OK(stream->Write(
      reinterpret_cast<const uint8_t*>(&metadataLen), sizeof(uint32_t)));
  ASSERT_OK(
      stream->Write(reinterpret_cast<const uint8_t*>(magic), strlen(magic)));

  ASSERT_OK_AND_ASSIGN(auto buffer, stream->Finish());

  auto reader = std::make_shared<BufferReader>(buffer);
  // Write the buffer to a temp file path.
  auto filePath = TempFilePath::create();
  writeToFile(filePath, buffer);
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
      memory::memoryManager()->addRootPool("MetadataTest");
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
      rootPool->addLeafChild("MetadataTest");
  dwio::common::ReaderOptions readerOptions{leafPool.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePath->getPath()),
      readerOptions.memoryPool());

  uint64_t fileSize = std::move(input)->getReadFile()->size();
  std::unique_ptr<dwio::common::SeekableInputStream> inputStream;
  inputStream = std::move(input)->loadCompleteFile();
  std::vector<char> copy(fileSize);
  const char* bufferStart = nullptr;
  const char* bufferEnd = nullptr;
  dwio::common::readBytes(
      fileSize, inputStream.get(), copy.data(), bufferStart, bufferEnd);
  ASSERT_EQ(0, strncmp(copy.data() + fileSize - 4, "PAR1", 4));

  uint32_t footerLength;
  std::memcpy(&footerLength, copy.data() + fileSize - 8, sizeof(uint32_t));

  VELOX_ASSERT_THROW(
      std::make_unique<ParquetReader>(std::move(input), readerOptions),
      fmt::format("({} vs. {})", footerLength + 12, fileSize));
}

} // namespace facebook::velox::parquet::arrow
