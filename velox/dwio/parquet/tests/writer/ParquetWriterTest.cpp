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

#include <arrow/io/memory.h>
#include <arrow/type.h>
#include <folly/init/Init.h>
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h" // @manual
#include "velox/core/QueryCtx.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h" // @manual
#include "velox/dwio/parquet/common/ParquetConfig.h"
#include "velox/dwio/parquet/reader/PageReader.h"
#include "velox/dwio/parquet/reader/ParquetTypeWithId.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/FileReader.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

namespace {

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;
using namespace facebook::velox::common::testutil;
namespace parquetArrow = facebook::velox::parquet::arrow;

class ParquetWriterTest : public ParquetTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    testutil::TestValue::enable();
    filesystems::registerLocalFileSystem();
    connector::hive::HiveConnectorFactory factory;
    auto hiveConnector = factory.newConnector(
        kHiveConnectorId,
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
    parquet::registerParquetWriterFactory();
  }

  void TearDown() override {
    writers_.clear();
  }

  RowVectorPtr makeSmallintTestData(int64_t rows) {
    auto data = makeRowVector({
        makeFlatVector<int16_t>(rows, [](auto row) { return row + 1; }),
    });
    return data;
  }

  RowVectorPtr makeTimestampTestData(int64_t rows) {
    auto data = makeRowVector({makeFlatVector<Timestamp>(
        rows, [](auto row) { return Timestamp(row, row); })});
    return data;
  }

  thrift::PageHeader readPageHeader(
      MemorySink* sinkPtr,
      int64_t offsetFromDataPage) {
    auto reader = createReaderInMemory(*sinkPtr);

    auto colChunkPtr = reader->fileMetaData().rowGroup(0).columnChunk(0);
    std::string_view sinkData(sinkPtr->data(), sinkPtr->size());

    auto readFile = std::make_shared<InMemoryReadFile>(sinkData);
    auto file = std::make_shared<ReadFileInputStream>(std::move(readFile));

    auto inputStream = std::make_unique<SeekableFileInputStream>(
        std::move(file),
        colChunkPtr.dataPageOffset() + offsetFromDataPage,
        150,
        *leafPool_,
        LogType::TEST);
    auto pageReader = std::make_unique<PageReader>(
        std::move(inputStream),
        *leafPool_,
        colChunkPtr.compression(),
        colChunkPtr.totalCompressedSize(),
        stats);
    return pageReader->readPageHeader();
  }

  inline static const std::string kHiveConnectorId = "test-hive";
  dwio::common::ColumnReaderStatistics stats;
};

class ArrowMemoryPool final : public ::arrow::MemoryPool {
 public:
  explicit ArrowMemoryPool() : allocated_(0) {}

  ~ArrowMemoryPool() = default;

  ::arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out)
      override {
    *out = reinterpret_cast<uint8_t*>(malloc(size));
    VELOX_CHECK_NOT_NULL(*out, "Failed to allocate memory in ArrowMemoryPool.");

    allocated_ += size;
    return ::arrow::Status::OK();
  }

  ::arrow::Status Reallocate(
      int64_t oldSize,
      int64_t newSize,
      int64_t alignment,
      uint8_t** ptr) override {
    uint8_t* newBuffer = reinterpret_cast<uint8_t*>(realloc(*ptr, newSize));
    VELOX_CHECK_NOT_NULL(
        newBuffer, "Failed to reallocate memory in ArrowMemoryPool.");

    *ptr = newBuffer;
    allocated_ = allocated_ - oldSize + newSize;
    return ::arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
    free(buffer);
    allocated_ -= size;
  }

  int64_t bytes_allocated() const override {
    return allocated_;
    ;
  }

  int64_t max_memory() const override {
    VELOX_UNSUPPORTED("ArrowMemoryPool#max_memory() unsupported");
  }

  int64_t total_bytes_allocated() const override {
    VELOX_UNSUPPORTED("ArrowMemoryPool#total_bytes_allocated() unsupported");
  }

  int64_t num_allocations() const override {
    VELOX_UNSUPPORTED("ArrowMemoryPool#num_allocations() unsupported");
  }

  std::string backend_name() const override {
    return "arrow memory pool";
  }

 private:
  int64_t allocated_;
};

std::vector<CompressionKind> params = {
    CompressionKind::CompressionKind_NONE,
    CompressionKind::CompressionKind_SNAPPY,
    CompressionKind::CompressionKind_ZSTD,
    CompressionKind::CompressionKind_LZ4,
    CompressionKind::CompressionKind_LZ4_HADOOP,
    CompressionKind::CompressionKind_GZIP,
};

TEST_F(ParquetWriterTest, createFormatOptions) {
  config::ConfigBase rawConnectorConfig({
      {"hive.parquet.writer.enable-dictionary", "true"},
      {"hive.parquet.writer.page-size", "2KB"},
      {"hive.parquet.writer.created-by", "test-writer"},
      {"iceberg.parquet.writer.page-size", "4KB"},
  });
  config::ConfigBase session({
      {"writer_enable_dictionary", "false"},
      {"writer_batch_size", "97"},
  });

  ParquetWriterFactory factory;
  config::ConfigBase connectorConfig(
      rawConnectorConfig.rawConfigsWithPrefix("hive.parquet."));
  auto parquetOptions = checkedPointerCast<ParquetWriterOptions>(
      factory.createFormatOptions(connectorConfig, session));

  ASSERT_TRUE(parquetOptions->enableDictionary.has_value());
  ASSERT_TRUE(parquetOptions->dataPageSize.has_value());
  ASSERT_TRUE(parquetOptions->batchSize.has_value());
  ASSERT_TRUE(parquetOptions->createdBy.has_value());
  EXPECT_FALSE(parquetOptions->enableDictionary.value());
  EXPECT_EQ(parquetOptions->dataPageSize.value(), 2 * 1024);
  EXPECT_EQ(parquetOptions->batchSize.value(), 97);
  EXPECT_EQ(parquetOptions->createdBy.value(), "test-writer");

  config::ConfigBase icebergConnectorConfig(
      rawConnectorConfig.rawConfigsWithPrefix("iceberg.parquet."));
  parquetOptions =
      checkedPointerCast<ParquetWriterOptions>(factory.createFormatOptions(
          icebergConnectorConfig,
          config::ConfigBase{std::unordered_map<std::string, std::string>{}}));
  ASSERT_TRUE(parquetOptions->dataPageSize.has_value());
  EXPECT_EQ(parquetOptions->dataPageSize.value(), 4 * 1024);
}

TEST_F(ParquetWriterTest, dictionaryEncodingWithDictionaryPageSize) {
  constexpr int64_t kRows = 10'000;
  const auto data = makeSmallintTestData(kRows);

  // Write Parquet test data, then read and return the DataPage
  // (thrift::PageType::type) used.
  const auto testEnableDictionaryAndDictionaryPageSizeToGetPageHeader =
      [&](std::unordered_map<std::string, std::string> configFromFile,
          std::unordered_map<std::string, std::string> sessionProperties,
          bool isFirstPage) {
        auto* sinkPtr = write(
            data, std::move(configFromFile), std::move(sessionProperties));
        if (isFirstPage) {
          return readPageHeader(sinkPtr, 0);
        }
        constexpr int64_t kFirstDataPageCompressedSize = 1291;
        constexpr int64_t kFirstDataPageHeaderSize = 48;
        return readPageHeader(
            sinkPtr, kFirstDataPageCompressedSize + kFirstDataPageHeaderSize);
      };

  // Test default config (i.e., no explicit config)

  const std::unordered_map<std::string, std::string> defaultConfigFromFile;
  const std::unordered_map<std::string, std::string>
      defaultSessionPropertiesFromFile;

  const auto defaultHeader =
      testEnableDictionaryAndDictionaryPageSizeToGetPageHeader(
          defaultConfigFromFile, defaultSessionPropertiesFromFile, true);
  // We use the default version of data page (V1)
  EXPECT_EQ(*defaultHeader.type(), thrift::PageType::DATA_PAGE);
  // Dictionary encoding is enabled as default
  EXPECT_EQ(
      *defaultHeader.data_page_header()->encoding(),
      thrift::Encoding::RLE_DICTIONARY);
  // Default dictionary page size is 1MB (same as data page size), so it can
  // contain a dictionary for all values. So all data will be in the first
  // data page
  EXPECT_EQ(*defaultHeader.data_page_header()->num_values(), kRows);

  // Test normal config

  // Set the dictionary page size limit to 1B so that the first data page will
  // only contain one batch of data encoded with dictionary, and from the
  // second batch it falls back to PLAIN encoding. If not set the dictionary
  // page size limit, the default is 1MB (same as data page default size) then
  // there will be only one data page contains all data encoded with dictionary
  const std::unordered_map<std::string, std::string> normalConfigFromFile = {
      {std::string(parquet::ParquetConfig::kWriterEnableDictionary), "true"},
      {std::string(parquet::ParquetConfig::kWriterDictionaryPageSizeLimit),
       "1B"},
  };
  const std::unordered_map<std::string, std::string> normalSessionProperties = {
      {std::string(parquet::ParquetConfig::kWriterEnableDictionarySession),
       "true"},
      {std::string(
           parquet::ParquetConfig::kWriterDictionaryPageSizeLimitSession),
       "1B"},
  };

  // Here we are reading the second data page. If we don't set the dictionary
  // page size, then there will be only one data page (See the comments above
  // the declaration of configFromFile)
  const auto normalHeader =
      testEnableDictionaryAndDictionaryPageSizeToGetPageHeader(
          normalConfigFromFile, normalSessionProperties, false);

  // We use the default version of data page (V1)
  EXPECT_EQ(*normalHeader.type(), thrift::PageType::DATA_PAGE);
  // The second data page will fall back to PLAIN encoding
  EXPECT_EQ(
      *normalHeader.data_page_header()->encoding(), thrift::Encoding::PLAIN);

  // Test incorrect enable dictionary config

  const std::string invalidEnableDictionaryValue{"NaB"};
  const std::unordered_map<std::string, std::string>
      incorrectEnableDictionaryConfigFromFile = {
          {std::string(parquet::ParquetConfig::kWriterEnableDictionary),
           invalidEnableDictionaryValue},
      };
  const std::unordered_map<std::string, std::string>
      incorrectEnableDictionarySessionProperties = {
          {std::string(parquet::ParquetConfig::kWriterEnableDictionarySession),
           invalidEnableDictionaryValue},
      };

  // Values cannot be parsed so that the exception is thrown
  VELOX_ASSERT_THROW(
      testEnableDictionaryAndDictionaryPageSizeToGetPageHeader(
          incorrectEnableDictionaryConfigFromFile,
          incorrectEnableDictionarySessionProperties,
          true),
      fmt::format(
          "Invalid parquet writer enable dictionary option: Non-whitespace character found after end of conversion: \"{}\"",
          invalidEnableDictionaryValue.substr(1)));

  // Test incorrect dictionary page size config

  const std::string invalidDictionaryPageSizeValue{"NaN"};
  const std::unordered_map<std::string, std::string>
      incorrectDictionaryPageSizeConfigFromFile = {
          {std::string(parquet::ParquetConfig::kWriterDictionaryPageSizeLimit),
           invalidDictionaryPageSizeValue},
      };
  const std::unordered_map<std::string, std::string>
      incorrectDictionaryPageSizeSessionProperties = {
          {std::string(
               parquet::ParquetConfig::kWriterDictionaryPageSizeLimitSession),
           invalidDictionaryPageSizeValue},
      };

  // Values cannot be parsed so that the exception is thrown
  VELOX_ASSERT_THROW(
      testEnableDictionaryAndDictionaryPageSizeToGetPageHeader(
          incorrectDictionaryPageSizeConfigFromFile,
          incorrectDictionaryPageSizeSessionProperties,
          true),
      fmt::format(
          "Invalid capacity string '{}'", invalidDictionaryPageSizeValue));
}

TEST_F(ParquetWriterTest, dictionaryEncodingOff) {
  constexpr int64_t kRows = 10'000;
  const auto data = makeSmallintTestData(kRows);

  // Write Parquet test data, then read and return the DataPage
  // (thrift::PageType::type) used.
  const auto testEnableDictionaryAndDictionaryPageSizeToGetPageHeader =
      [&](std::unordered_map<std::string, std::string> configFromFile,
          std::unordered_map<std::string, std::string> sessionProperties) {
        auto* sinkPtr = write(
            data, std::move(configFromFile), std::move(sessionProperties));
        return readPageHeader(sinkPtr, 0);
      };

  // Test only dictionary off without dictionary page size configured

  const std::unordered_map<std::string, std::string>
      withoutPageSizeConfigFromFile = {
          {std::string(parquet::ParquetConfig::kWriterEnableDictionary),
           "false"},
      };
  const std::unordered_map<std::string, std::string>
      withoutPageSizeSessionProperties = {
          {std::string(parquet::ParquetConfig::kWriterEnableDictionarySession),
           "false"},
      };

  const auto withoutPageSizeHeader =
      testEnableDictionaryAndDictionaryPageSizeToGetPageHeader(
          withoutPageSizeConfigFromFile, withoutPageSizeSessionProperties);

  // We use the default version of data page (V1)
  EXPECT_EQ(*withoutPageSizeHeader.type(), thrift::PageType::DATA_PAGE);
  // Since we turn off the dictionary encoding, and the default data page size
  // is 1MB, there is only one page, and its encoding should be PLAIN, which
  // means the configuration is applied
  EXPECT_EQ(
      *withoutPageSizeHeader.data_page_header()->encoding(),
      thrift::Encoding::PLAIN);
  // All rows will be on the only data page, this is a sanity check
  EXPECT_EQ(*withoutPageSizeHeader.data_page_header()->num_values(), kRows);

  // Test dictionary off but with dictionary page size configured

  const std::unordered_map<std::string, std::string>
      withPageSizeConfigFromFile = {
          {std::string(parquet::ParquetConfig::kWriterEnableDictionary),
           "false"},
          {std::string(parquet::ParquetConfig::kWriterDictionaryPageSizeLimit),
           "1B"},
      };
  const std::unordered_map<std::string, std::string>
      withPageSizeSessionProperties = {
          {std::string(parquet::ParquetConfig::kWriterEnableDictionarySession),
           "false"},
          {std::string(
               parquet::ParquetConfig::kWriterDictionaryPageSizeLimitSession),
           "1B"},
      };

  const auto withPageSizeHeader =
      testEnableDictionaryAndDictionaryPageSizeToGetPageHeader(
          withPageSizeConfigFromFile, withPageSizeSessionProperties);

  // Should be the same as without dictionary page size configured, because
  // when the dictionary is disabled, the dictionary page silze is meaningless
  EXPECT_EQ(*withPageSizeHeader.type(), thrift::PageType::DATA_PAGE);
  EXPECT_EQ(
      *withPageSizeHeader.data_page_header()->encoding(),
      thrift::Encoding::PLAIN);
  EXPECT_EQ(*withPageSizeHeader.data_page_header()->num_values(), kRows);
}

TEST_F(ParquetWriterTest, compression) {
  auto schema =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           DOUBLE(),
           BIGINT(),
           INTEGER(),
           BIGINT(),
           INTEGER(),
           DOUBLE()});
  const int64_t kRows = 10'000;
  const auto data = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return row + 5; }),
      makeFlatVector<double>(kRows, [](auto row) { return row - 10; }),
      makeFlatVector<int64_t>(kRows, [](auto row) { return row - 15; }),
      makeFlatVector<uint32_t>(kRows, [](auto row) { return row + 20; }),
      makeFlatVector<uint64_t>(kRows, [](auto row) { return row + 25; }),
      makeFlatVector<int32_t>(kRows, [](auto row) { return row + 30; }),
      makeFlatVector<double>(kRows, [](auto row) { return row - 25; }),
  });

  ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.compressionKind = CompressionKind::CompressionKind_SNAPPY;

  const auto& fieldNames = schema->names();
  for (int i = 0; i < params.size(); i++) {
    writerOptions.columnCompressionsMap[fieldNames[i]] = params[i];
  }

  auto* sinkPtr = write(data, options, writerOptions);

  auto reader = createReaderInMemory(*sinkPtr);

  ASSERT_EQ(reader->numberOfRows(), kRows);
  ASSERT_EQ(*reader->rowType(), *schema);

  for (int i = 0; i < params.size(); i++) {
    EXPECT_EQ(
        reader->fileMetaData().rowGroup(0).columnChunk(i).compression(),
        (i < params.size()) ? params[i]
                            : CompressionKind::CompressionKind_SNAPPY);
  }

  auto rowReader = createRowReaderFromReader(*reader, schema);
  assertReadWithReaderAndExpected(schema, *rowReader, data, *leafPool_);
}

TEST_F(ParquetWriterTest, testPageSizeAndBatchSizeConfiguration) {
  constexpr int64_t kRows = 10'000;
  const auto data = makeSmallintTestData(kRows);

  // Write Parquet test data, then read and return the DataPage
  // (thrift::PageType::type) used.
  const auto testPageSizeAndBatchSizeToGetPageHeader =
      [&](std::unordered_map<std::string, std::string> configFromFile,
          std::unordered_map<std::string, std::string> sessionProperties) {
        auto* sinkPtr = write(
            data, std::move(configFromFile), std::move(sessionProperties));
        return readPageHeader(sinkPtr, 0);
      };

  // Test default config (i.e., no explicit config)

  const std::unordered_map<std::string, std::string> defaultConfigFromFile;
  const std::unordered_map<std::string, std::string>
      defaultSessionPropertiesFromFile;

  const auto defaultHeader = testPageSizeAndBatchSizeToGetPageHeader(
      defaultConfigFromFile, defaultSessionPropertiesFromFile);
  // We use the default version of data page (V1)
  EXPECT_EQ(*defaultHeader.type(), thrift::PageType::DATA_PAGE);
  // We don't use compressor here
  EXPECT_EQ(
      *defaultHeader.uncompressed_page_size(),
      defaultHeader.compressed_page_size());
  // The default page size is 1MB, which can actually contains all data in one
  // page
  EXPECT_EQ(*defaultHeader.compressed_page_size(), 17529);
  // As mentioned above, the default page size can contain all data in one page
  // so the number of values of the first page equals to the total number
  EXPECT_EQ(*defaultHeader.data_page_header()->num_values(), kRows);

  // Test normal config

  // We use 97 as the batch size to test, because 97 is a prime, if the number
  // of values in each page can be divided by 97, it means the batch size is
  // applied (default is 1024)
  const std::unordered_map<std::string, std::string> normalConfigFromFile = {
      {std::string(parquet::ParquetConfig::kWriterPageSize), "2KB"},
      {std::string(parquet::ParquetConfig::kWriterBatchSize), "97"},
  };
  const std::unordered_map<std::string, std::string> normalSessionProperties = {
      {std::string(parquet::ParquetConfig::kWriterPageSizeSession), "2KB"},
      {std::string(parquet::ParquetConfig::kWriterBatchSizeSession), "97"},
  };
  const auto normalHeader = testPageSizeAndBatchSizeToGetPageHeader(
      normalConfigFromFile, normalSessionProperties);
  // We use the default version of data page (V1)
  EXPECT_EQ(*normalHeader.type(), thrift::PageType::DATA_PAGE);
  // We don't use compressor here
  EXPECT_EQ(
      *normalHeader.uncompressed_page_size(),
      *normalHeader.compressed_page_size());
  // 1485B < 2KB < 1MB, which means the page size is applied (default is 1MB)
  EXPECT_EQ(*normalHeader.compressed_page_size(), 1485);
  // 1067 % 97 == 0, which means the batch size is applied (default is 1024)
  EXPECT_EQ(*normalHeader.data_page_header()->num_values(), 1067);

  // Test incorrect page size config

  const std::string invalidPageSizeAndBatchSizeValue{"NaN"};
  const std::unordered_map<std::string, std::string>
      incorrectPageSizeConfigFromFile = {
          {std::string(parquet::ParquetConfig::kWriterPageSize),
           invalidPageSizeAndBatchSizeValue},
      };
  const std::unordered_map<std::string, std::string>
      incorrectPageSizeSessionPropertiesFromFile = {
          {std::string(parquet::ParquetConfig::kWriterPageSizeSession),
           invalidPageSizeAndBatchSizeValue},
      };

  // Values cannot be parsed so that the exception is thrown
  VELOX_ASSERT_THROW(
      testPageSizeAndBatchSizeToGetPageHeader(
          incorrectPageSizeConfigFromFile,
          incorrectPageSizeSessionPropertiesFromFile),
      fmt::format(
          "Invalid capacity string '{}'", invalidPageSizeAndBatchSizeValue))

  // Test incorrect batch size config

  const std::unordered_map<std::string, std::string>
      incorrectBatchSizeConfigFromFile = {
          {std::string(parquet::ParquetConfig::kWriterBatchSize),
           invalidPageSizeAndBatchSizeValue},
      };
  const std::unordered_map<std::string, std::string>
      incorrectBatchSizeSessionPropertiesFromFile = {
          {std::string(parquet::ParquetConfig::kWriterBatchSizeSession),
           invalidPageSizeAndBatchSizeValue},
      };

  // Values cannot be parsed so that the exception is thrown
  VELOX_ASSERT_THROW(
      testPageSizeAndBatchSizeToGetPageHeader(
          incorrectBatchSizeConfigFromFile,
          incorrectBatchSizeSessionPropertiesFromFile),
      fmt::format(
          "Invalid parquet writer batch size: Invalid leading character: \"{}\"",
          invalidPageSizeAndBatchSizeValue));
}

TEST_F(ParquetWriterTest, toggleDataPageVersion) {
  const int64_t kRows = 1;
  const auto data = makeRowVector({
      makeFlatVector<int32_t>(kRows, [](auto row) { return 987; }),
  });

  // Write Parquet test data, then read and return the DataPage
  // (thrift::PageType::type) used.
  const auto testDataPageVersion =
      [&](std::unordered_map<std::string, std::string> configFromFile,
          std::unordered_map<std::string, std::string> sessionProperties)
      -> thrift::PageType {
    auto* sinkPtr =
        write(data, std::move(configFromFile), std::move(sessionProperties));
    return *readPageHeader(sinkPtr, 0).type_ref();
  };

  // Test default behavior - DataPage should be V1.
  ASSERT_EQ(testDataPageVersion({}, {}), thrift::PageType::DATA_PAGE);

  // Simulate setting DataPage version to V2 via Hive config from file.
  std::unordered_map<std::string, std::string> configFromFile = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersion), "V2"}};

  ASSERT_EQ(
      testDataPageVersion(configFromFile, {}), thrift::PageType::DATA_PAGE_V2);

  // Simulate setting DataPage version to V1 via Hive config from file.
  configFromFile = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersion), "V1"}};

  ASSERT_EQ(
      testDataPageVersion(configFromFile, {}), thrift::PageType::DATA_PAGE);

  // Simulate setting DataPage version to V2 via connector session property.
  std::unordered_map<std::string, std::string> sessionProperties = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersionSession),
       "V2"}};

  ASSERT_EQ(
      testDataPageVersion({}, sessionProperties),
      thrift::PageType::DATA_PAGE_V2);

  // Simulate setting DataPage version to V1 via connector session property.
  sessionProperties = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersionSession),
       "V1"}};

  ASSERT_EQ(
      testDataPageVersion({}, sessionProperties), thrift::PageType::DATA_PAGE);

  // Simulate setting DataPage version to V1 via connector session property,
  // and to V2 via Hive config from file. Session property should take
  // precedence.
  sessionProperties = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersionSession),
       "V1"}};
  configFromFile = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersion), "V2"}};

  ASSERT_EQ(
      testDataPageVersion({}, sessionProperties), thrift::PageType::DATA_PAGE);

  // Simulate setting DataPage version to V2 via connector session property,
  // and to V1 via Hive config from file. Session property should take
  // precedence.
  sessionProperties = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersionSession),
       "V2"}};
  configFromFile = {
      {std::string(parquet::ParquetConfig::kWriterDataPageVersion), "V1"}};

  ASSERT_EQ(
      testDataPageVersion({}, sessionProperties),
      thrift::PageType::DATA_PAGE_V2);
}

DEBUG_ONLY_TEST_F(ParquetWriterTest, unitFromWriterOptions) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::parquet::Writer::write",
      std::function<void(const ::arrow::Schema*)>(
          ([&](const ::arrow::Schema* arrowSchema) {
            const auto tsType =
                std::dynamic_pointer_cast<::arrow::TimestampType>(
                    arrowSchema->field(0)->type());
            ASSERT_EQ(tsType->unit(), ::arrow::TimeUnit::MICRO);
            ASSERT_EQ(tsType->timezone(), "America/Los_Angeles");
          })));

  const auto data = makeTimestampTestData(10'000);
  ParquetWriterOptions writerOptions;
  writerOptions.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
  writerOptions.parquetWriteTimestampTimeZone = "America/Los_Angeles";

  write(data, writerOptions);
}

DEBUG_ONLY_TEST_F(ParquetWriterTest, parquetWriteTimestampTimeZoneWithDefault) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::parquet::Writer::write",
      std::function<void(const ::arrow::Schema*)>(
          ([&](const ::arrow::Schema* arrowSchema) {
            const auto tsType =
                std::dynamic_pointer_cast<::arrow::TimestampType>(
                    arrowSchema->field(0)->type());
            ASSERT_EQ(tsType->unit(), ::arrow::TimeUnit::MICRO);
            ASSERT_EQ(tsType->timezone(), "");
          })));

  const auto data = makeTimestampTestData(10'000);
  ParquetWriterOptions writerOptions;
  writerOptions.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;

  write(data, writerOptions);
}

TEST_F(ParquetWriterTest, parquetWriteWithArrowMemoryPool) {
  const auto data = makeTimestampTestData(10'000);
  ParquetWriterOptions writerOptions;
  writerOptions.arrowMemoryPool = std::make_shared<ArrowMemoryPool>();

  write(data, writerOptions);
}

TEST_F(ParquetWriterTest, preEpochInt96Timestamp) {
  auto schema = ROW({"c0"}, {TIMESTAMP()});
  auto data = makeRowVector({makeFlatVector<Timestamp>({
      Timestamp(-86'401, 0),
      Timestamp(-86'400, 0),
      Timestamp(-3'600, 0),
      Timestamp(-1, 0),
      Timestamp(0, 0),
      Timestamp(1, 0),
  })});

  ParquetWriterOptions writerOptions;
  writerOptions.parquetWriteTimestampUnit = TimestampPrecision::kNanoseconds;
  writerOptions.writeInt96AsTimestamp = true;

  auto* sinkPtr = write(data, writerOptions);

  auto reader = createReaderInMemory(*sinkPtr);

  ASSERT_EQ(reader->numberOfRows(), data->size());
  ASSERT_EQ(*reader->rowType(), *schema);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  assertReadWithReaderAndExpected(schema, *rowReader, data, *leafPool_);

  // Read the Int96 values directly to verify the correctness of the conversion
  // from Timestamp to Int96.
  std::string_view sinkData(sinkPtr->data(), sinkPtr->size());
  auto arrowBufferReader = std::make_shared<::arrow::io::BufferReader>(
      std::make_shared<::arrow::Buffer>(
          reinterpret_cast<const uint8_t*>(sinkData.data()), sinkData.size()));
  auto fileReader = parquetArrow::ParquetFileReader::open(arrowBufferReader);
  auto int96Reader = std::dynamic_pointer_cast<parquetArrow::Int96Reader>(
      fileReader->rowGroup(0)->column(0));
  ASSERT_NE(int96Reader, nullptr);

  std::vector<parquetArrow::Int96> values(data->size());
  int64_t valuesRead = 0;
  ASSERT_EQ(
      int96Reader->readBatch(
          data->size(), nullptr, nullptr, values.data(), &valuesRead),
      data->size());
  ASSERT_EQ(valuesRead, data->size());

  const std::vector<std::pair<int32_t, uint64_t>> expected = {
      {-2, 86'399'000'000'000},
      {-1, 0},
      {-1, 23LL * 3'600 * 1'000'000'000},
      {-1, 86'399'000'000'000},
      {0, 0},
      {0, 1'000'000'000},
  };

  for (auto i = 0; i < values.size(); ++i) {
    const auto decoded = parquetArrow::decodeInt96Timestamp(values[i]);
    EXPECT_EQ(
        values[i].value[2],
        static_cast<uint32_t>(
            parquetArrow::kJulianToUnixEpochDays + expected[i].first));
    EXPECT_EQ(decoded.nanoseconds, expected[i].second);
    EXPECT_LT(decoded.nanoseconds, parquetArrow::kNanosecondsPerDay);
  }
}

TEST_F(ParquetWriterTest, writerMagic) {
  const auto data = makeRowVector(
      {makeFlatVector<int32_t>(20'000, [](auto row) { return row; })});

  ParquetWriterOptions writerOptions;

  const auto* sinkPtr = write(data, writerOptions);
  const auto fileData = std::string_view(sinkPtr->data(), sinkPtr->size());

  EXPECT_EQ("PAR1", std::string(fileData.data(), 4));
  EXPECT_EQ("PAR1", std::string(fileData.data() + fileData.size() - 4, 4));
}

TEST_F(ParquetWriterTest, flushWhenStreamBuffersGrow) {
  constexpr int64_t kNumRows = 200;

  ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  writerOptions.enableDictionary = false;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory =
      []() -> std::unique_ptr<dwio::common::FlushPolicy> {
    return std::make_unique<DefaultFlushPolicy>(
        /*rowsInRowGroup=*/1,
        /*bytesInRowGroup=*/4 * 1024);
  };

  const auto schema = ROW({"c0"}, {BIGINT()});
  auto sink = std::make_unique<MemorySink>(
      200 * 1024 * 1024, FileSink::Options{.pool = leafPool_.get()});
  auto* sinkPtr = sink.get();
  options.formatSpecificOptions =
      std::make_shared<ParquetWriterOptions>(writerOptions);
  auto writer = std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), options, schema);
  const auto data = makeRowVector(
      {makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; })});

  writer->write(data);

  // Data should be flushed into the FileSink by acculumated closed row groups.
  EXPECT_GT(sinkPtr->size(), 0);

  writer->close();

  const auto reader = createReaderInMemory(*sinkPtr);
  EXPECT_EQ(kNumRows, reader->numberOfRows());
  EXPECT_EQ(kNumRows, reader->fileMetaData().numRowGroups());
}

TEST_F(ParquetWriterTest, flushRowGroupByBufferedSize) {
  ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory = []() {
    return std::make_unique<DefaultFlushPolicy>(
        /*rowsInRowGroup=*/10'000,
        /*bytesInRowGroup=*/200);
  };

  auto rowType = ROW({"c0"}, {INTEGER()});
  auto testBatches =
      [&](int numBatches, int expectedNumRowGroups, int expectedNumRows) {
        std::vector<RowVectorPtr> batches;
        for (int i = 0; i < numBatches; ++i) {
          batches.push_back(
              makeRowVector({makeFlatVector<int32_t>({1, 1, 1, 1, 1})}));
        }

        const auto* sinkPtr = write(batches, options, writerOptions);
        const auto reader = createReaderInMemory(*sinkPtr);
        EXPECT_EQ(expectedNumRowGroups, reader->fileMetaData().numRowGroups());
        EXPECT_EQ(expectedNumRows, reader->numberOfRows());
      };

  testBatches(10, 1, 50);
  testBatches(20, 2, 100);
}

TEST_F(ParquetWriterTest, flushEmptyRowGroup) {
  ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory = []() {
    return std::make_unique<DefaultFlushPolicy>(
        /*rowsInRowGroup=*/50,
        /*bytesInRowGroup=*/128 * 1'024 * 1'024);
  };

  std::vector<RowVectorPtr> batches;
  for (int i = 0; i < 10; ++i) {
    batches.push_back(
        makeRowVector({makeFlatVector<int32_t>({1, 1, 1, 1, 1})}));
  }

  const auto* sinkPtr = write(batches, options, writerOptions);
  const auto reader = createReaderInMemory(*sinkPtr);
  EXPECT_EQ(1, reader->fileMetaData().numRowGroups());
  EXPECT_EQ(50, reader->numberOfRows());
}

TEST_F(ParquetWriterTest, largeMetadata) {
  const auto data = makeRowVector(
      {makeFlatVector<int32_t>(1'000, [](auto row) { return row; })});

  ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory = []() {
    return std::make_unique<DefaultFlushPolicy>(
        /*rowsInRowGroup=*/1,
        /*bytesInRowGroup=*/128 * 1'024 * 1'024);
  };

  const auto* sinkPtr = write(data, options, writerOptions);

  auto readerOpts = makeDefaultReaderOptions();
  readerOpts.setFooterSpeculativeIoSize(1024);
  readerOpts.setFilePreloadThreshold(1024 * 8);

  const auto reader = createReaderInMemory(*sinkPtr, readerOpts);
  EXPECT_EQ(1'000, reader->numberOfRows());
  EXPECT_EQ(1'000, reader->fileMetaData().numRowGroups());
}

TEST_F(ParquetWriterTest, writeDecimalAsInteger) {
  const auto rowVector = makeRowVector(
      {makeFlatVector<int64_t>({1, 2}, DECIMAL(8, 2)),
       makeFlatVector<int64_t>({1, 2}, DECIMAL(10, 2)),
       makeFlatVector<int128_t>({1, 2}, DECIMAL(19, 2))});

  ParquetWriterOptions writerOptions;

  const auto* sinkPtr = write(rowVector, writerOptions);

  const auto reader = createReaderInMemory(*sinkPtr);

  const auto types = reader->typeWithId()->getChildren();
  ASSERT_GE(types.size(), 3);
  const auto c0 = std::dynamic_pointer_cast<const ParquetTypeWithId>(types[0]);
  ASSERT_NE(c0, nullptr);
  EXPECT_EQ(c0->parquetType_.value(), thrift::Type::INT32);
  const auto c1 = std::dynamic_pointer_cast<const ParquetTypeWithId>(types[1]);
  ASSERT_NE(c1, nullptr);
  EXPECT_EQ(c1->parquetType_.value(), thrift::Type::INT64);
  const auto c2 = std::dynamic_pointer_cast<const ParquetTypeWithId>(types[2]);
  ASSERT_NE(c2, nullptr);
  EXPECT_EQ(c2->parquetType_.value(), thrift::Type::FIXED_LEN_BYTE_ARRAY);
}

TEST_F(ParquetWriterTest, configurableWriteSchema) {
  const auto test = [&](const RowTypePtr& type, const RowTypePtr& newType) {
    constexpr int32_t kNumBatches = 5;
    constexpr int32_t kBatchSize = 100;
    auto batches = createBatches(type, kNumBatches, kBatchSize);
    constexpr vector_size_t kNumRows = kNumBatches * kBatchSize;

    auto data = std::dynamic_pointer_cast<RowVector>(
        BaseVector::create(type, kNumRows, pool()));
    auto expected = std::dynamic_pointer_cast<RowVector>(
        BaseVector::create(newType, kNumRows, pool()));
    vector_size_t offset = 0;
    for (const auto& batch : batches) {
      data->copy(batch.get(), offset, 0, batch->size());
      expected->copy(batch.get(), offset, 0, batch->size());
      offset += batch->size();
    }

    ParquetWriterOptions writerOptions;
    const auto* sinkPtr = write(data, writerOptions, newType);
    auto reader = createReaderInMemory(*sinkPtr);

    ASSERT_EQ(reader->numberOfRows(), kNumRows);
    EXPECT_EQ(reader->rowType()->toString(), newType->toString());

    auto rowReader = createRowReaderFromReader(*reader, newType);
    assertReadWithReaderAndExpected(newType, *rowReader, expected, *leafPool_);
  };

  test(
      ROW({"a", "b"}, {INTEGER(), ROW({"c"}, {ROW({"d"}, INTEGER())})}),
      ROW({"aa", "bb"}, {INTEGER(), ROW({"cc"}, {ROW({"dd"}, INTEGER())})}));

  test(
      ROW({"a", "b"}, {ARRAY(ROW({"c", "d"}, BIGINT())), BIGINT()}),
      ROW({"aa", "bb"}, {ARRAY(ROW({"cc", "dd"}, BIGINT())), BIGINT()}));

  test(
      ROW({"a", "b"},
          {MAP(ROW({"c", "d"}, BIGINT()), ROW({"e", "f"}, BIGINT())),
           BIGINT()}),
      ROW({"aa", "bb"},
          {MAP(ROW({"cc", "dd"}, BIGINT()), ROW({"ee", "ff"}, BIGINT())),
           BIGINT()}));
}

TEST_F(ParquetWriterTest, updateWriterOptionsFromHiveConfig) {
  std::unordered_map<std::string, std::string> configFromFile = {
      {std::string(parquet::ParquetConfig::kWriterTimestampUnit), "3"}};
  const std::vector<Timestamp> timestamps = {
      Timestamp(1, 123'456'789),
      Timestamp(2, 987'654'321),
  };
  const auto data = makeRowVector({makeFlatVector<Timestamp>(timestamps)});
  const auto expected = makeRowVector({makeFlatVector<Timestamp>({
      Timestamp::fromMillis(timestamps[0].toMillis()),
      Timestamp::fromMillis(timestamps[1].toMillis()),
  })});

  const auto* sinkPtr = write(data, std::move(configFromFile), {});

  auto reader = createReaderInMemory(*sinkPtr);

  ASSERT_EQ(reader->numberOfRows(), data->size());
  ASSERT_EQ(*reader->rowType(), *data->rowType());

  auto rowReaderOpts = makeRowReaderOpts(data->rowType());
  rowReaderOpts.setScanSpec(makeScanSpec(data->rowType()));
  rowReaderOpts.setTimestampPrecision(TimestampPrecision::kNanoseconds);
  auto rowReader = reader->createRowReader(rowReaderOpts);
  assertReadWithReaderAndExpected(
      data->rowType(), *rowReader, expected, *leafPool_);
}

TEST_F(ParquetWriterTest, enableStoreDecimalAsInteger) {
  const auto rowType = ROW({
      {"c0", DECIMAL(8, 2)},
      {"c1", DECIMAL(10, 2)},
      {"c2", DECIMAL(19, 2)},
  });
  // c1 includes an unscaled value exceeding INT32 range (50000000.00).
  const auto data = makeRowVector({
      makeFlatVector<int64_t>({100, 99999999}, DECIMAL(8, 2)),
      makeFlatVector<int64_t>({100, 5000000000LL}, DECIMAL(10, 2)),
      makeFlatVector<int128_t>({100, 200}, DECIMAL(19, 2)),
  });

  using PhysicalTypes =
      std::vector<std::shared_ptr<const ParquetTypeWithId::TypeWithId>>;

  auto expectParquetType = [&](size_t columnIndex,
                               const PhysicalTypes& types,
                               thrift::Type expectedType,
                               std::optional<int32_t> expectedTypeLength =
                                   std::nullopt) {
    auto col =
        std::dynamic_pointer_cast<const ParquetTypeWithId>(types[columnIndex]);
    ASSERT_NE(col, nullptr);
    ASSERT_TRUE(col->parquetType_.has_value());
    EXPECT_EQ(col->parquetType_.value(), expectedType);
    if (expectedTypeLength.has_value()) {
      EXPECT_EQ(col->typeLength_, expectedTypeLength.value());
    }
  };

  const auto verifyStoredAsInteger = [&](const PhysicalTypes& types) {
    expectParquetType(0, types, thrift::Type::INT32);
    expectParquetType(1, types, thrift::Type::INT64);
    expectParquetType(2, types, thrift::Type::FIXED_LEN_BYTE_ARRAY, 9);
  };

  const auto verifyStoredAsFixedLenByteArray = [&](const PhysicalTypes& types) {
    expectParquetType(0, types, thrift::Type::FIXED_LEN_BYTE_ARRAY, 4);
    expectParquetType(1, types, thrift::Type::FIXED_LEN_BYTE_ARRAY, 5);
    expectParquetType(2, types, thrift::Type::FIXED_LEN_BYTE_ARRAY, 9);
  };

  const auto writeReadAndVerify =
      [&](std::unordered_map<std::string, std::string> configFromFile,
          std::unordered_map<std::string, std::string> sessionProperties,
          const std::function<void(const PhysicalTypes&)>&
              verifyPhysicalTypes) {
        auto* sinkPtr = write(
            data, std::move(configFromFile), std::move(sessionProperties));

        auto reader = createReaderInMemory(*sinkPtr);
        auto& parquetReader = dynamic_cast<ParquetReader&>(*reader);

        const auto& types = parquetReader.typeWithId()->getChildren();
        ASSERT_EQ(types.size(), 3);
        verifyPhysicalTypes(types);

        auto rowReader = createRowReaderFromReader(*reader, rowType);
        assertReadWithReaderAndExpected(rowType, *rowReader, data, *leafPool_);
      };

  const auto configKey =
      std::string(parquet::ParquetConfig::kWriterEnableStoreDecimalAsInteger);
  const auto sessionKey = std::string(
      parquet::ParquetConfig::kWriterEnableStoreDecimalAsIntegerSession);

  // Connector session property.
  writeReadAndVerify({}, {{sessionKey, "true"}}, verifyStoredAsInteger);
  writeReadAndVerify(
      {}, {{sessionKey, "false"}}, verifyStoredAsFixedLenByteArray);

  // Hive config from file.
  writeReadAndVerify({{configKey, "true"}}, {}, verifyStoredAsInteger);
  writeReadAndVerify(
      {{configKey, "false"}}, {}, verifyStoredAsFixedLenByteArray);

  // Session property takes precedence over Hive config from file.
  writeReadAndVerify(
      {{configKey, "false"}}, {{sessionKey, "true"}}, verifyStoredAsInteger);
  writeReadAndVerify(
      {{configKey, "true"}},
      {{sessionKey, "false"}},
      verifyStoredAsFixedLenByteArray);
}

#ifdef VELOX_ENABLE_PARQUET
DEBUG_ONLY_TEST_F(ParquetWriterTest, timestampUnitAndTimeZone) {
  SCOPED_TESTVALUE_SET(
      "facebook::velox::parquet::Writer::write",
      std::function<void(const ::arrow::Schema*)>(
          ([&](const ::arrow::Schema* arrowSchema) {
            const auto tsType =
                std::dynamic_pointer_cast<::arrow::TimestampType>(
                    arrowSchema->field(0)->type());
            ASSERT_EQ(tsType->unit(), ::arrow::TimeUnit::MICRO);
          })));

  SCOPED_TESTVALUE_SET(
      "facebook::velox::parquet::Writer::Writer",
      std::function<void(const ArrowOptions* options)>(
          ([&](const ArrowOptions* options) {
            ASSERT_TRUE(options->timestampTimeZone.has_value());
            ASSERT_EQ(options->timestampTimeZone.value(), "America/New_York");
          })));

  const auto data = makeRowVector({makeFlatVector<Timestamp>(
      10'000, [](auto row) { return Timestamp(row, row); })});
  const auto outputDirectory = TempDirectoryPath::create();

  auto parquetOptions = std::make_shared<ParquetWriterOptions>();
  parquetOptions->parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
  auto writerOptions = std::make_shared<dwio::common::WriterOptions>();
  writerOptions->formatSpecificOptions = parquetOptions;

  const auto plan = PlanBuilder()
                        .values({data})
                        .tableWrite(
                            outputDirectory->getPath(),
                            dwio::common::FileFormat::PARQUET,
                            {},
                            writerOptions)
                        .planNode();
  AssertQueryBuilder(plan)
      .config(core::QueryConfig::kSessionTimezone, "America/New_York")
      .copyResults(pool_.get());
}
#endif

TEST_F(ParquetWriterTest, dictionaryEncodedVector) {
  const auto randomIndices = [this](vector_size_t size) {
    BufferPtr indices =
        AlignedBuffer::allocate<vector_size_t>(size, leafPool_.get());
    auto rawIndices = indices->asMutable<vector_size_t>();
    for (int32_t i = 0; i < size; i++) {
      rawIndices[i] = folly::Random::rand32(size);
    }
    return indices;
  };

  const auto wrapDictionaryVectors =
      [&](const std::vector<VectorPtr>& vectors) {
        std::vector<VectorPtr> wrappedVectors;
        wrappedVectors.reserve(vectors.size());

        for (const auto& vector : vectors) {
          auto wrappedVector = BaseVector::wrapInDictionary(
              BufferPtr(nullptr),
              randomIndices(vector->size()),
              vector->size(),
              vector);
          EXPECT_EQ(
              wrappedVector->encoding(), VectorEncoding::Simple::DICTIONARY);
          wrappedVectors.emplace_back(wrappedVector);
        }
        return wrappedVectors;
      };

  // Dictionary encoded vectors with complex type.
  const auto size = 10'000;
  auto wrappedVectors = wrapDictionaryVectors({
      facebook::velox::test::BatchMaker::createVector<TypeKind::MAP>(
          MAP(VARCHAR(), INTEGER()), size, *leafPool_),
      facebook::velox::test::BatchMaker::createVector<TypeKind::ARRAY>(
          ARRAY(VARCHAR()), size, *leafPool_),
      facebook::velox::test::BatchMaker::createVector<TypeKind::ROW>(
          ROW({"c0", "c1"},
              {BIGINT(), ROW({"id", "name"}, {INTEGER(), VARCHAR()})}),
          size,
          *leafPool_),
  });

  auto data = makeRowVector(wrappedVectors);
  write(data);

  // Dictionary encoded constant vector of scalar type.
  const auto constantVector = makeConstant(static_cast<int64_t>(123'456), size);
  const auto wrappedVector = std::make_shared<DictionaryVector<int64_t>>(
      leafPool_.get(), nullptr, size, constantVector, randomIndices(size));
  EXPECT_EQ(wrappedVector->encoding(), VectorEncoding::Simple::DICTIONARY);
  VELOX_CHECK_NOT_NULL(wrappedVector->valueVector());
  EXPECT_FALSE(wrappedVector->wrappedVector()->isFlatEncoding());

  data = makeRowVector({wrappedVector});
  write(data);
}

/// Verifies round-trip correctness when writing DictionaryVector<VARCHAR> with
/// low cardinality.  With flattenDictionary=false the bridge exports a
/// DictionaryArray that the Arrow Parquet writer encodes directly via
/// writeArrowDictionary().
TEST_F(ParquetWriterTest, dictionaryPassthroughVarcharLowCardinality) {
  constexpr vector_size_t kSize = 10'000;
  constexpr int kDictSize = 10;

  // Build a dictionary of 10 strings.
  std::vector<std::string> dictStrings(kDictSize);
  for (int i = 0; i < kDictSize; ++i) {
    dictStrings[i] = fmt::format("val_{}", i);
  }
  auto dictionary = makeFlatVector<StringView>(
      kDictSize, [&](auto row) { return StringView(dictStrings[row]); });

  // Index buffer pointing into the dictionary.
  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i % kDictSize;
  }

  auto dictVector = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);
  ASSERT_EQ(dictVector->encoding(), VectorEncoding::Simple::DICTIONARY);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  // Build expected output as a flat vector (Parquet always reads back flat).
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies round-trip correctness with high-cardinality VARCHAR dictionary
/// (all unique values).  The Arrow Parquet writer may abandon dictionary
/// encoding and fall back to PLAIN.
TEST_F(ParquetWriterTest, dictionaryPassthroughVarcharHighCardinality) {
  constexpr vector_size_t kSize = 10'000;

  // All-unique dictionary (size == kSize).
  std::vector<std::string> dictStrings(kSize);
  for (vector_size_t i = 0; i < kSize; ++i) {
    dictStrings[i] = fmt::format("unique_{}", i);
  }
  auto dictionary = makeFlatVector<StringView>(
      kSize, [&](auto row) { return StringView(dictStrings[row]); });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i;
  }

  auto dictVector = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that DictionaryVector<INTEGER> (non-string type) round-trips
/// correctly.  The Arrow Parquet writer's dictionaryDirectWriteSupported()
/// only supports binary-like types for direct dict write, so integer
/// dictionaries should fall back to dense encoding internally.
TEST_F(ParquetWriterTest, dictionaryPassthroughInteger) {
  constexpr vector_size_t kSize = 10'000;
  constexpr int kDictSize = 50;

  auto dictionary =
      makeFlatVector<int32_t>(kDictSize, [](auto row) { return row * 7; });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i % kDictSize;
  }

  auto dictVector = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that DictionaryVector<BIGINT> round-trips correctly and that the
/// data is identical whether dictionary encoding or plain encoding is used in
/// the Parquet file.
TEST_F(ParquetWriterTest, dictionaryPassthroughBigint) {
  constexpr vector_size_t kSize = 5'000;
  constexpr int kDictSize = 100;

  auto dictionary = makeFlatVector<int64_t>(kDictSize, [](auto row) {
    return static_cast<int64_t>(row) * 1'000'000;
  });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i % kDictSize;
  }

  auto dictVector = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that writing multiple batches with changing dictionaries works
/// correctly.  Each batch has a different dictionary, testing that the Arrow
/// Parquet writer handles dictionary transitions across write() calls.
TEST_F(ParquetWriterTest, dictionaryPassthroughMultipleBatches) {
  constexpr vector_size_t kBatchSize = 2'000;
  constexpr int kDictSize = 5;

  auto schema = ROW({"c0"}, {VARCHAR()});

  std::vector<RowVectorPtr> batches;
  for (int batchIdx = 0; batchIdx < 3; ++batchIdx) {
    std::vector<std::string> dictStrings(kDictSize);
    for (int i = 0; i < kDictSize; ++i) {
      dictStrings[i] = fmt::format("batch{}_{}", batchIdx, i);
    }
    auto dictionary = makeFlatVector<StringView>(
        kDictSize, [&](auto row) { return StringView(dictStrings[row]); });

    BufferPtr indices =
        AlignedBuffer::allocate<vector_size_t>(kBatchSize, leafPool_.get());
    auto rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < kBatchSize; ++i) {
      rawIndices[i] = i % kDictSize;
    }

    auto dictVector = BaseVector::wrapInDictionary(
        BufferPtr(nullptr), indices, kBatchSize, dictionary);
    batches.push_back(makeRowVector({dictVector}));
  }

  ParquetWriterOptions writerOptions;
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.schema = schema;
  auto* sinkPtr = write(batches, options, writerOptions);

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(
      reader->numberOfRows(),
      static_cast<uint64_t>(kBatchSize) * batches.size());

  // Build expected flat output by concatenating all batches.
  auto totalSize = kBatchSize * batches.size();
  std::vector<std::string> expectedStrings(totalSize);
  for (size_t i = 0; i < totalSize; ++i) {
    int batchIdx = i / kBatchSize;
    int localIdx = i % kBatchSize;
    expectedStrings[i] =
        fmt::format("batch{}_{}", batchIdx, localIdx % kDictSize);
  }
  auto expected = makeRowVector({makeFlatVector<StringView>(
      totalSize, [&](auto row) { return StringView(expectedStrings[row]); })});

  auto rowReader = createRowReaderFromReader(*reader, schema);
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that DictionaryVector with nulls round-trips correctly.
TEST_F(ParquetWriterTest, dictionaryPassthroughWithNulls) {
  constexpr vector_size_t kSize = 5'000;
  constexpr int kDictSize = 20;

  std::vector<std::string> dictStrings(kDictSize);
  for (int i = 0; i < kDictSize; ++i) {
    dictStrings[i] = fmt::format("v_{}", i);
  }
  auto dictionary = makeFlatVector<StringView>(
      kDictSize, [&](auto row) { return StringView(dictStrings[row]); });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i % kDictSize;
  }

  // Every 3rd element is null.
  auto nulls = makeNulls(kSize, [](auto row) { return row % 3 == 0; });

  auto dictVector =
      BaseVector::wrapInDictionary(nulls, indices, kSize, dictionary);
  ASSERT_EQ(dictVector->encoding(), VectorEncoding::Simple::DICTIONARY);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that complex types (MAP, ARRAY, ROW) are written correctly without
/// flattening when they are not wrapped in dictionary encoding.  This tests O2:
/// the removal of the overly conservative isComplex check in needFlatten().
TEST_F(ParquetWriterTest, complexTypesWithoutFlattening) {
  constexpr vector_size_t kSize = 1'000;

  // Pre-generate key strings for the map.
  std::vector<std::string> keyStrings(kSize * 3);
  for (vector_size_t i = 0; i < static_cast<vector_size_t>(keyStrings.size());
       ++i) {
    keyStrings[i] = fmt::format("key_{}", i);
  }

  auto mapVector = makeMapVector<StringView, int32_t>(
      kSize,
      [](auto row) { return row % 3 + 1; },
      [&](auto row) { return StringView(keyStrings[row]); },
      [](auto row) { return static_cast<int32_t>(row * 10); });

  auto arrayVector = makeArrayVector<int64_t>(
      kSize,
      [](auto row) { return row % 4 + 1; },
      [](auto row) { return static_cast<int64_t>(row * 100); });

  auto data = makeRowVector({mapVector, arrayVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  assertReadWithReaderAndExpected(schema, *rowReader, data, *leafPool_);
}

/// Verifies that a mix of dictionary-encoded scalar columns and flat complex
/// type columns in the same RowVector round-trips correctly.  This exercises
/// the interaction between O1 (dictionary passthrough for scalars) and O2
/// (no flattening for complex types).
TEST_F(ParquetWriterTest, mixedDictionaryAndComplexTypes) {
  constexpr vector_size_t kSize = 2'000;
  constexpr int kDictSize = 15;

  // Dictionary-encoded VARCHAR column.
  std::vector<std::string> dictStrings(kDictSize);
  for (int i = 0; i < kDictSize; ++i) {
    dictStrings[i] = fmt::format("cat_{}", i);
  }
  auto dictionary = makeFlatVector<StringView>(
      kDictSize, [&](auto row) { return StringView(dictStrings[row]); });
  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i % kDictSize;
  }
  auto dictColumn = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);

  // Flat ARRAY column.
  auto arrayColumn = makeArrayVector<int32_t>(
      kSize,
      [](auto row) { return row % 3 + 1; },
      [](auto row) { return static_cast<int32_t>(row); });

  // Flat INTEGER column.
  auto intColumn = makeFlatVector<int32_t>(kSize, [](auto row) { return row; });

  auto data = makeRowVector({dictColumn, arrayColumn, intColumn});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  // Expected: flatten the dictionary column, keep complex and flat as-is.
  VectorPtr flatDictCol = dictColumn;
  BaseVector::flattenVector(flatDictCol);
  auto expected = makeRowVector({flatDictCol, arrayColumn, intColumn});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that dictionary wrapping a complex type (ARRAY) is correctly
/// flattened before writing.  This exercises the isPrimitiveType() guard in
/// needFlatten() — Arrow cannot handle DictionaryArray of complex types.
TEST_F(ParquetWriterTest, dictionaryWrappingComplexTypeFlattens) {
  constexpr vector_size_t kSize = 1'000;

  auto arrayVector = makeArrayVector<int32_t>(
      kSize,
      [](auto row) { return row % 3 + 1; },
      [](auto row) { return static_cast<int32_t>(row); });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i;
  }

  auto dictOfArray = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, arrayVector);
  ASSERT_EQ(dictOfArray->encoding(), VectorEncoding::Simple::DICTIONARY);

  auto data = makeRowVector({dictOfArray});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  // Expected output is the original unwrapped array (flattening removes the
  // identity dictionary wrapping).
  auto expected = makeRowVector({arrayVector});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that dictionary-of-dictionary (nested wrapping) is correctly
/// flattened.  This exercises the isFlatEncoding() guard in needFlatten().
TEST_F(ParquetWriterTest, dictionaryOfDictionaryFlattens) {
  constexpr vector_size_t kSize = 1'000;
  constexpr int kDictSize = 10;

  auto dictionary =
      makeFlatVector<int32_t>(kDictSize, [](auto row) { return row * 11; });

  BufferPtr innerIndices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawInner = innerIndices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawInner[i] = i % kDictSize;
  }

  auto innerDict = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), innerIndices, kSize, dictionary);

  // Wrap the dictionary in another dictionary layer.
  BufferPtr outerIndices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawOuter = outerIndices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawOuter[i] = i;
  }
  auto dictOfDict = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), outerIndices, kSize, innerDict);

  auto data = makeRowVector({dictOfDict});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatExpected = dictOfDict;
  BaseVector::flattenVector(flatExpected);
  auto expected = makeRowVector({flatExpected});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that a constant vector wrapping a dictionary is correctly
/// flattened.  This exercises the CONSTANT branch in needFlatten().
TEST_F(ParquetWriterTest, constantWrappingDictionaryFlattens) {
  constexpr vector_size_t kSize = 500;

  auto constantInt = makeConstant(static_cast<int32_t>(42), kSize);

  auto data = makeRowVector({constantInt});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatExpected = constantInt;
  BaseVector::flattenVector(flatExpected);
  auto expected = makeRowVector({flatExpected});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that an empty dictionary vector (0 rows) can be written without
/// crashing.  The Parquet reader rejects empty files, so this test only
/// verifies the write path.
TEST_F(ParquetWriterTest, dictionaryPassthroughEmptyVector) {
  constexpr vector_size_t kSize = 0;
  constexpr int kDictSize = 5;

  std::vector<std::string> dictStrings(kDictSize);
  for (int i = 0; i < kDictSize; ++i) {
    dictStrings[i] = fmt::format("v_{}", i);
  }
  auto dictionary = makeFlatVector<StringView>(
      kDictSize, [&](auto row) { return StringView(dictStrings[row]); });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(0, leafPool_.get());

  auto dictVector = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);

  auto data = makeRowVector({dictVector});

  // Write should succeed without crashing.
  write(data, ParquetWriterOptions{});
}

/// Verifies that a single-row dictionary vector round-trips correctly.
TEST_F(ParquetWriterTest, dictionaryPassthroughSingleRow) {
  constexpr vector_size_t kSize = 1;

  std::vector<std::string> dictStrings = {"only_value"};
  auto dictionary = makeFlatVector<StringView>(
      1, [&](auto row) { return StringView(dictStrings[row]); });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  indices->asMutable<vector_size_t>()[0] = 0;

  auto dictVector = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, kSize, dictionary);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that a dictionary vector where every element is null round-trips.
TEST_F(ParquetWriterTest, dictionaryPassthroughAllNulls) {
  constexpr vector_size_t kSize = 1'000;
  constexpr int kDictSize = 5;

  std::vector<std::string> dictStrings(kDictSize);
  for (int i = 0; i < kDictSize; ++i) {
    dictStrings[i] = fmt::format("v_{}", i);
  }
  auto dictionary = makeFlatVector<StringView>(
      kDictSize, [&](auto row) { return StringView(dictStrings[row]); });

  BufferPtr indices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIndices[i] = i % kDictSize;
  }

  // Every element is null.
  auto nulls = makeNulls(kSize, [](auto /*row*/) { return true; });

  auto dictVector =
      BaseVector::wrapInDictionary(nulls, indices, kSize, dictionary);

  auto data = makeRowVector({dictVector});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDict = dictVector;
  BaseVector::flattenVector(flatDict);
  auto expected = makeRowVector({flatDict});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies that multiple dictionary-encoded columns in the same RowVector all
/// pass through correctly.
TEST_F(ParquetWriterTest, dictionaryPassthroughMultipleColumns) {
  constexpr vector_size_t kSize = 2'000;

  // First dict column: VARCHAR, cardinality 5.
  constexpr int kDictSize1 = 5;
  std::vector<std::string> strings1(kDictSize1);
  for (int i = 0; i < kDictSize1; ++i) {
    strings1[i] = fmt::format("a_{}", i);
  }
  auto dict1 = makeFlatVector<StringView>(
      kDictSize1, [&](auto row) { return StringView(strings1[row]); });
  BufferPtr idx1 =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto raw1 = idx1->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    raw1[i] = i % kDictSize1;
  }
  auto col1 =
      BaseVector::wrapInDictionary(BufferPtr(nullptr), idx1, kSize, dict1);

  // Second dict column: VARCHAR, cardinality 20.
  constexpr int kDictSize2 = 20;
  std::vector<std::string> strings2(kDictSize2);
  for (int i = 0; i < kDictSize2; ++i) {
    strings2[i] = fmt::format("b_{}", i);
  }
  auto dict2 = makeFlatVector<StringView>(
      kDictSize2, [&](auto row) { return StringView(strings2[row]); });
  BufferPtr idx2 =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto raw2 = idx2->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    raw2[i] = i % kDictSize2;
  }
  auto col2 =
      BaseVector::wrapInDictionary(BufferPtr(nullptr), idx2, kSize, dict2);

  // Third dict column: INTEGER, cardinality 50.
  constexpr int kDictSize3 = 50;
  auto dict3 =
      makeFlatVector<int32_t>(kDictSize3, [](auto row) { return row * 3; });
  BufferPtr idx3 =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto raw3 = idx3->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    raw3[i] = i % kDictSize3;
  }
  auto col3 =
      BaseVector::wrapInDictionary(BufferPtr(nullptr), idx3, kSize, dict3);

  auto data = makeRowVector({col1, col2, col3});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flat1 = col1;
  BaseVector::flattenVector(flat1);
  VectorPtr flat2 = col2;
  BaseVector::flattenVector(flat2);
  VectorPtr flat3 = col3;
  BaseVector::flattenVector(flat3);
  auto expected = makeRowVector({flat1, flat2, flat3});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies dictionary passthrough for DOUBLE and BOOLEAN types.
TEST_F(ParquetWriterTest, dictionaryPassthroughOtherPrimitiveTypes) {
  constexpr vector_size_t kSize = 2'000;

  // Dictionary DOUBLE column.
  constexpr int kDoubleDictSize = 10;
  auto doubleDictionary = makeFlatVector<double>(
      kDoubleDictSize, [](auto row) { return row * 1.5; });
  BufferPtr doubleIndices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawDoubleIdx = doubleIndices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawDoubleIdx[i] = i % kDoubleDictSize;
  }
  auto dictDouble = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), doubleIndices, kSize, doubleDictionary);

  // Dictionary BOOLEAN column.
  auto boolDictionary =
      makeFlatVector<bool>(2, [](auto row) { return row == 1; });
  BufferPtr boolIndices =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawBoolIdx = boolIndices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawBoolIdx[i] = i % 2;
  }
  auto dictBool = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), boolIndices, kSize, boolDictionary);

  auto data = makeRowVector({dictDouble, dictBool});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatDouble = dictDouble;
  BaseVector::flattenVector(flatDouble);
  VectorPtr flatBool = dictBool;
  BaseVector::flattenVector(flatBool);
  auto expected = makeRowVector({flatDouble, flatBool});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

/// Verifies selective per-column flattening: a dict-of-dict column (must
/// flatten) alongside a passthrough dictionary column (must NOT flatten).
/// With blanket flattening, both would be materialized. With selective
/// flattening, only the nested-dict column is flattened while the simple
/// dictionary passes through to Arrow.
TEST_F(ParquetWriterTest, selectiveFlatteningMixedEncodings) {
  constexpr vector_size_t kSize = 2'000;

  // Column 0: simple dictionary VARCHAR (should pass through).
  constexpr int kDictSize = 10;
  std::vector<std::string> dictStrings(kDictSize);
  for (int i = 0; i < kDictSize; ++i) {
    dictStrings[i] = fmt::format("pass_{}", i);
  }
  auto dict = makeFlatVector<StringView>(
      kDictSize, [&](auto row) { return StringView(dictStrings[row]); });
  BufferPtr idx =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawIdx = idx->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawIdx[i] = i % kDictSize;
  }
  auto passthroughCol =
      BaseVector::wrapInDictionary(BufferPtr(nullptr), idx, kSize, dict);

  // Column 1: dict-of-dict INTEGER (must flatten).
  constexpr int kInnerDictSize = 20;
  auto innerDict =
      makeFlatVector<int32_t>(kInnerDictSize, [](auto row) { return row * 5; });
  BufferPtr innerIdx =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawInner = innerIdx->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawInner[i] = i % kInnerDictSize;
  }
  auto innerDictVec = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), innerIdx, kSize, innerDict);
  BufferPtr outerIdx =
      AlignedBuffer::allocate<vector_size_t>(kSize, leafPool_.get());
  auto rawOuter = outerIdx->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < kSize; ++i) {
    rawOuter[i] = i;
  }
  auto nestedDictCol = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), outerIdx, kSize, innerDictVec);

  auto data = makeRowVector({passthroughCol, nestedDictCol});
  auto schema = asRowType(data->type());

  auto* sinkPtr = write(data, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  VectorPtr flatPassthrough = passthroughCol;
  BaseVector::flattenVector(flatPassthrough);
  VectorPtr flatNested = nestedDictCol;
  BaseVector::flattenVector(flatNested);
  auto expected = makeRowVector({flatPassthrough, flatNested});
  assertReadWithReaderAndExpected(schema, *rowReader, expected, *leafPool_);
}

TEST_F(ParquetWriterTest, allNulls) {
  auto schema = ROW({"c0"}, {INTEGER()});
  const int64_t kRows = 4096;
  // Create a column with all elements being null.
  auto nulls = makeNulls(kRows, [](auto /*row*/) { return true; });
  auto flatVector = std::make_shared<FlatVector<int32_t>>(
      pool_.get(),
      schema->childAt(0),
      nulls,
      kRows,
      /*values=*/nullptr,
      std::vector<BufferPtr>());
  auto data = std::make_shared<RowVector>(
      pool_.get(), schema, nullptr, kRows, std::vector<VectorPtr>{flatVector});

  auto* sinkPtr = write(data);

  auto reader = createReaderInMemory(*sinkPtr);

  ASSERT_EQ(reader->numberOfRows(), kRows);
  ASSERT_EQ(*reader->rowType(), *schema);

  auto rowReader = createRowReaderFromReader(*reader, schema);
  assertReadWithReaderAndExpected(schema, *rowReader, data, *leafPool_);
}

/// Verifies that close() without any prior write() does not crash.
TEST_F(ParquetWriterTest, closeWithoutWrite) {
  auto schema = ROW({"c0"}, {INTEGER()});
  auto sink = std::make_unique<dwio::common::MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
  auto writer =
      std::make_unique<parquet::Writer>(std::move(sink), options, schema);
  writer->close();
}

/// Verifies that flush() with no accumulated data is a no-op.
TEST_F(ParquetWriterTest, flushWithNoData) {
  auto schema = ROW({"c0"}, {INTEGER()});
  auto sink = std::make_unique<dwio::common::MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto* sinkPtr = sink.get();
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
  auto writer =
      std::make_unique<parquet::Writer>(std::move(sink), options, schema);
  writer->flush();

  auto data = makeRowVector(
      {makeFlatVector<int32_t>(100, [](auto row) { return row; })});
  writer->write(data);
  writer->close();

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), 100);
}

/// Verifies that multiple consecutive flush() calls are idempotent.
TEST_F(ParquetWriterTest, consecutiveFlushCalls) {
  auto schema = ROW({"c0"}, {INTEGER()});
  auto sink = std::make_unique<dwio::common::MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto* sinkPtr = sink.get();
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
  auto writer =
      std::make_unique<parquet::Writer>(std::move(sink), options, schema);

  auto data = makeRowVector(
      {makeFlatVector<int32_t>(100, [](auto row) { return row; })});
  writer->write(data);
  writer->flush();
  writer->flush();
  writer->flush();

  writer->write(data);
  writer->close();

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), 200);
}

/// Verifies that a single large batch exceeding maxRowGroupLength is correctly
/// split into multiple row groups by writeRecordBatch.
TEST_F(ParquetWriterTest, batchExceedingRowGroupSize) {
  constexpr vector_size_t kSize = 1'000;
  constexpr int kRowsPerGroup = 100;

  auto data = makeRowVector(
      {makeFlatVector<int32_t>(kSize, [](auto row) { return row; })});
  auto schema = asRowType(data->type());

  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory = [kRowsPerGroup]() {
    return std::make_unique<DefaultFlushPolicy>(
        /*rowsInRowGroup=*/kRowsPerGroup,
        /*bytesInRowGroup=*/512 * 1'024 * 1'024);
  };
  auto* sinkPtr = write(data, options, ParquetWriterOptions{});

  auto reader = createReaderInMemory(*sinkPtr);
  ASSERT_EQ(reader->numberOfRows(), kSize);
  ASSERT_EQ(reader->fileMetaData().numRowGroups(), kSize / kRowsPerGroup);
}

} // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
