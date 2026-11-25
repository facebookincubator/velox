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

#include "velox/connectors/common/FileSplitReader.h"

#include <folly/executors/InlineExecutor.h>
#include <gtest/gtest.h>

#include "velox/common/Casts.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/common/FileConfig.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/orc/reader/OrcReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#endif

namespace facebook::velox::connector {
namespace {

using dwio::common::FileFormat;
using Properties = std::unordered_map<std::string, std::string>;

// Fails if a format factory receives format configs that still carry their
// format prefix.
class UnqualifiedConfigsNimbleReaderFactory
    : public dwio::common::ReaderFactory {
 public:
  UnqualifiedConfigsNimbleReaderFactory() : ReaderFactory(FileFormat::NIMBLE) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput>,
      const dwio::common::ReaderOptions&) override {
    VELOX_NYI();
  }

  std::shared_ptr<dwio::common::FormatSpecificOptions> createFormatOptions(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) const override {
    checkUnqualified(connectorConfig);
    checkUnqualified(session);
    return nullptr;
  }

 private:
  static void checkUnqualified(const config::ConfigBase& config) {
    for (const auto& [key, _] : config.rawConfigs()) {
      VELOX_CHECK_EQ(key.find('.'), std::string::npos, "{}", key);
    }
  }
};

// A partition key column of the given type.
class PartitionKeyHandle : public FileColumnHandle {
 public:
  PartitionKeyHandle(const std::string& name, const TypePtr& type)
      : FileColumnHandle(
            name,
            ColumnType::kPartitionKey,
            type,
            type,
            /*requiredSubfields=*/{},
            /*extractions=*/{},
            /*partitionDateValueDaysSinceEpoch=*/false,
            /*postProcessor=*/nullptr) {}

  folly::dynamic serialize() const override {
    VELOX_NYI();
  }
};

class FileSplitReaderTest : public testing::Test,
                            public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    dwrf::registerDwrfReaderFactory();
    orc::registerOrcReaderFactory();
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
#endif
    dwio::common::registerReaderFactory(
        std::make_shared<UnqualifiedConfigsNimbleReaderFactory>());
  }

  void TearDown() override {
    dwio::common::unregisterReaderFactory(FileFormat::NIMBLE);
#ifdef VELOX_ENABLE_PARQUET
    parquet::unregisterParquetReaderFactory();
#endif
    orc::unregisterOrcReaderFactory();
    dwrf::unregisterDwrfReaderFactory();
  }

  // A ConnectorQueryCtx and the session properties it points to.
  struct QueryContext {
    std::unique_ptr<config::ConfigBase> session;
    std::unique_ptr<ConnectorQueryCtx> ctx;
  };

  QueryContext makeQueryContext(
      Properties session = {},
      const std::string& sessionTimezone = "",
      bool adjustTimestampToTimezone = false) {
    QueryContext context;
    context.session = std::make_unique<config::ConfigBase>(std::move(session));
    context.ctx = ConnectorQueryCtx::Builder()
                      .operatorPool(pool())
                      .connectorPool(pool())
                      .sessionProperties(context.session.get())
                      .queryId("query.FileSplitReaderTest")
                      .taskId("task.FileSplitReaderTest")
                      .planNodeId("planNodeId.FileSplitReaderTest")
                      .sessionTimezone(sessionTimezone)
                      .adjustTimestampToTimezone(adjustTimestampToTimezone)
                      .build();
    return context;
  }

  static std::shared_ptr<const FileConfig> makeFileConfig(
      Properties config = {}) {
    return std::make_shared<FileConfig>(
        std::make_shared<config::ConfigBase>(std::move(config)), "hive.");
  }

  static std::shared_ptr<const FileConnectorSplit> makeSplit(
      FileFormat format,
      bool cacheable = true,
      std::optional<dwio::common::ColumnMappingMode> columnMappingMode =
          std::nullopt) {
    return std::make_shared<FileConnectorSplit>(
        "testConnectorId",
        "/tmp/testfile",
        format,
        /*_start=*/0,
        /*_length=*/std::numeric_limits<uint64_t>::max(),
        /*splitWeight=*/0,
        cacheable,
        /*infoColumns=*/std::nullopt,
        /*partitionKeys=*/
        std::unordered_map<std::string, std::optional<std::string>>{},
        columnMappingMode);
  }

  dwio::common::ReaderOptions newReaderOptions() {
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(ioStats_);
    options.setMetadataIoStats(ioStats_);
    return options;
  }

  // Returns the reader options configureReaderOptions() sets for 'split'.
  dwio::common::ReaderOptions configureReader(
      const std::shared_ptr<const FileConnectorSplit>& split,
      const Properties& session = {},
      const Properties& config = {},
      const Properties& tableParameters = {},
      const Properties& serdeParameters = {}) {
    auto context = makeQueryContext(session);
    auto options = newReaderOptions();
    FileSplitReader::configureReaderOptions(
        makeFileConfig(config),
        context.ctx.get(),
        split,
        /*fileSchema=*/nullptr,
        tableParameters,
        serdeParameters,
        options);
    return options;
  }

  // Returns the row reader options configureRowReaderOptions() sets for
  // 'split'.
  dwio::common::RowReaderOptions configureRowReader(
      const std::shared_ptr<const FileConnectorSplit>& split,
      const Properties& session = {},
      const Properties& config = {},
      const Properties& tableParameters = {},
      const Properties& serdeParameters = {}) {
    auto context = makeQueryContext(session);
    dwio::common::RowReaderOptions options;
    FileSplitReader::configureRowReaderOptions(
        makeFileConfig(config),
        context.ctx.get(),
        split,
        /*scanSpec=*/nullptr,
        /*metadataFilter=*/nullptr,
        /*requestedType=*/nullptr,
        /*ioExecutor=*/nullptr,
        tableParameters,
        serdeParameters,
        options);
    return options;
  }

  const std::shared_ptr<io::IoStatistics> ioStats_ =
      std::make_shared<io::IoStatistics>();
};

TEST_F(FileSplitReaderTest, readerOptionsFromConfig) {
  const auto options = configureReader(
      makeSplit(FileFormat::DWRF),
      /*session=*/{},
      {
          {FileConfig::kLoadQuantum, "321"},
          {FileConfig::kMaxCoalescedBytes, "129"},
          {FileConfig::kMaxCoalescedDistance, "513KB"},
          {FileConfig::kFileColumnNamesReadAsLowerCase, "true"},
          {FileConfig::kFilePreloadThreshold, "9999"},
          {FileConfig::kPrefetchRowGroups, "10"},
      });

  EXPECT_EQ(options.loadQuantum(), 321);
  EXPECT_EQ(options.maxCoalesceBytes(), 129);
  EXPECT_EQ(options.maxCoalesceDistance(), 513 << 10);
  EXPECT_TRUE(options.fileColumnNamesReadAsLowerCase());
  EXPECT_EQ(options.filePreloadThreshold(), 9999);
  EXPECT_EQ(options.prefetchRowGroups(), 10);
}

TEST_F(FileSplitReaderTest, readerOptionsFromSession) {
  const auto options = configureReader(
      makeSplit(FileFormat::NIMBLE),
      {
          {FileConfig::kMaxCoalescedDistanceSession, "3MB"},
          {FileConfig::kNimbleLazyFileStatsSession, "true"},
          {FileConfig::kNimbleDirectBufferedInputEnabledSession, "true"},
      });

  EXPECT_EQ(options.maxCoalesceDistance(), 3 << 20);
  EXPECT_TRUE(options.nimbleLazyFileStats());
  EXPECT_TRUE(options.nimbleDirectBufferedInputEnabled());
}

TEST_F(FileSplitReaderTest, fileSchemaAndTimezone) {
  const auto fileSchema = ROW({"a", "b"}, BIGINT());
  auto context = makeQueryContext(
      /*session=*/{},
      "America/Los_Angeles",
      /*adjustTimestampToTimezone=*/true);
  auto options = newReaderOptions();
  FileSplitReader::configureReaderOptions(
      makeFileConfig(),
      context.ctx.get(),
      makeSplit(FileFormat::DWRF),
      fileSchema,
      /*tableParameters=*/{},
      /*serdeParameters=*/{},
      options);

  EXPECT_EQ(options.fileSchema(), fileSchema);
  ASSERT_NE(options.sessionTimezone(), nullptr);
  EXPECT_EQ(options.sessionTimezone()->name(), "America/Los_Angeles");
  EXPECT_TRUE(options.adjustTimestampToTimezone());
}

// The session property decides when set. Otherwise the query config, passed
// through the ConnectorQueryCtx, does.
TEST_F(FileSplitReaderTest, selectiveNimbleReader) {
  const auto selectiveNimbleReaderEnabled = [&](const Properties& session,
                                                bool queryConfig) {
    auto context = makeQueryContext(session);
    context.ctx->setSelectiveNimbleReaderEnabled(queryConfig);
    auto options = newReaderOptions();
    FileSplitReader::configureReaderOptions(
        makeFileConfig(),
        context.ctx.get(),
        makeSplit(FileFormat::NIMBLE),
        /*fileSchema=*/nullptr,
        /*tableParameters=*/{},
        /*serdeParameters=*/{},
        options);
    return options.selectiveNimbleReaderEnabled();
  };

  const Properties disabled{
      {FileConfig::kSelectiveNimbleReaderEnabledSession, "false"}};
  EXPECT_FALSE(selectiveNimbleReaderEnabled(disabled, /*queryConfig=*/true));
  EXPECT_TRUE(selectiveNimbleReaderEnabled({}, /*queryConfig=*/true));
  EXPECT_FALSE(selectiveNimbleReaderEnabled({}, /*queryConfig=*/false));
}

TEST_F(FileSplitReaderTest, columnMappingMode) {
  const Properties useColumnNames{{FileConfig::kUseColumnNamesSession, "true"}};

  EXPECT_EQ(
      configureReader(makeSplit(FileFormat::ORC)).columnMappingMode(),
      dwio::common::ColumnMappingMode::kPosition);
  EXPECT_EQ(
      configureReader(makeSplit(FileFormat::ORC), useColumnNames)
          .columnMappingMode(),
      dwio::common::ColumnMappingMode::kName);

  // The split's mode overrides the session's.
  EXPECT_EQ(
      configureReader(
          makeSplit(
              FileFormat::ORC,
              /*cacheable=*/true,
              dwio::common::ColumnMappingMode::kPosition),
          useColumnNames)
          .columnMappingMode(),
      dwio::common::ColumnMappingMode::kPosition);

  VELOX_ASSERT_THROW(
      configureReader(makeSplit(
          FileFormat::ORC,
          /*cacheable=*/true,
          dwio::common::ColumnMappingMode::kParquetFieldId)),
      "not supported for file format orc");
}

TEST_F(FileSplitReaderTest, fileFormat) {
  EXPECT_EQ(
      configureReader(makeSplit(FileFormat::DWRF)).fileFormat(),
      FileFormat::DWRF);

  // A format without a reader factory gets no format-specific options.
  const auto json = configureReader(makeSplit(FileFormat::JSON));
  EXPECT_EQ(json.fileFormat(), FileFormat::JSON);
  EXPECT_EQ(json.formatSpecificOptions(), nullptr);

  auto context = makeQueryContext();
  auto options = newReaderOptions();
  options.setFileFormat(FileFormat::PARQUET);
  VELOX_ASSERT_THROW(
      FileSplitReader::configureReaderOptions(
          makeFileConfig(),
          context.ctx.get(),
          makeSplit(FileFormat::DWRF),
          /*fileSchema=*/nullptr,
          /*tableParameters=*/{},
          /*serdeParameters=*/{},
          options),
      "All splits of a data source must have the same file format");
}

TEST_F(FileSplitReaderTest, footerSpeculativeIoSizeByFormat) {
  const Properties session{
      {"nimble.unused", "1"},
      {"parquet.unused", "2"},
      {"parquet_footer_speculative_io_size", "7777"},
      {"parquet_footer_memory_tracking_threshold", "6666"},
      {"hive.parquet.footer_speculative_io_size", "8888"},
      {"unrelated.unused", "3"},
  };
  const Properties config{
      {FileConfig::kOrcFooterSpeculativeIoSize, "1111"},
      {FileConfig::kNimbleFooterSpeculativeIoSize, "3333"},
      {"parquet.footer-speculative-io-size", "9999"},
      {"parquet.footer-memory-tracking-threshold", "9999"},
      {"hive.parquet.footer-speculative-io-size", "2222"},
      {"hive.parquet.footer-memory-tracking-threshold", "5555"},
      {"iceberg.parquet.footer-speculative-io-size", "4444"},
  };
  const auto footerSize = [&](FileFormat format) {
    return configureReader(makeSplit(format), session, config)
        .footerSpeculativeIoSize();
  };

  // DWRF uses the ORC setting.
  EXPECT_EQ(footerSize(FileFormat::ORC), 1111);
  EXPECT_EQ(footerSize(FileFormat::DWRF), 1111);
  EXPECT_EQ(footerSize(FileFormat::NIMBLE), 3333);

#ifdef VELOX_ENABLE_PARQUET
  // Parquet keeps its footer settings in its format options. The session's
  // values win over the connector config's.
  const auto parquetOptions = checkedPointerCast<parquet::ParquetReaderOptions>(
      configureReader(makeSplit(FileFormat::PARQUET), session, config)
          .formatSpecificOptions());
  EXPECT_EQ(parquetOptions->footerSpeculativeIoSize, 7777);
  EXPECT_EQ(parquetOptions->footerMemoryTrackingThreshold(), 6666);
#endif
}

// Metadata and index caching need a cacheable split, so a split read on a
// node that doesn't keep its data doesn't fill the cache. Pinning applies to
// a reader's own cache and doesn't.
TEST_F(FileSplitReaderTest, caching) {
  const Properties allOn{
      {FileConfig::kCacheMetadataSession, "true"},
      {FileConfig::kCacheIndexSession, "true"},
      {FileConfig::kPinMetadataSession, "true"},
      {FileConfig::kPinIndexSession, "true"},
  };

  {
    const auto options = configureReader(
        makeSplit(FileFormat::NIMBLE, /*cacheable=*/true), allOn);
    EXPECT_TRUE(options.cacheable());
    EXPECT_TRUE(options.cacheMetadata());
    EXPECT_TRUE(options.cacheIndex());
    EXPECT_TRUE(options.pinMetadata());
    EXPECT_TRUE(options.pinIndex());
  }
  {
    const auto options = configureReader(
        makeSplit(FileFormat::NIMBLE, /*cacheable=*/false), allOn);
    EXPECT_FALSE(options.cacheable());
    EXPECT_FALSE(options.cacheMetadata());
    EXPECT_FALSE(options.cacheIndex());
    EXPECT_TRUE(options.pinMetadata());
    EXPECT_TRUE(options.pinIndex());
  }
  {
    const auto options =
        configureReader(makeSplit(FileFormat::NIMBLE, /*cacheable=*/true));
    EXPECT_TRUE(options.cacheable());
    EXPECT_FALSE(options.cacheMetadata());
    EXPECT_FALSE(options.cacheIndex());
    EXPECT_FALSE(options.pinMetadata());
    EXPECT_FALSE(options.pinIndex());
  }
}

TEST_F(FileSplitReaderTest, textSerDeOptions) {
  using dwio::common::SerDeOptions;
  using dwio::common::SerDeSeparator;
  using dwio::common::TableParameter;

  struct TestCase {
    std::string name;
    Properties serdeParameters;
    Properties tableParameters;
    SerDeOptions expected;
  };

  const auto withSeparator = [](SerDeSeparator separator, uint8_t value) {
    SerDeOptions options;
    options.separators[static_cast<size_t>(separator)] = value;
    return options;
  };
  const auto withEscape = [](uint8_t escapeChar) {
    SerDeOptions options;
    options.escapeChar = escapeChar;
    options.isEscaped = true;
    return options;
  };
  const auto withNullString = [](const std::string& nullString) {
    SerDeOptions options;
    options.nullString = nullString;
    return options;
  };

  const std::vector<TestCase> testCases{
      {"defaults", {}, {}, SerDeOptions{}},
      {"field delimiter",
       {{SerDeOptions::kFieldDelim, "\t"}},
       {},
       withSeparator(SerDeSeparator::FIELD_DELIM, '\t')},
      {"collection delimiter",
       {{SerDeOptions::kCollectionDelim, "="}},
       {},
       withSeparator(SerDeSeparator::COLLECTION_DELIM, '=')},
      {"map key delimiter",
       {{SerDeOptions::kMapKeyDelim, "&"}},
       {},
       withSeparator(SerDeSeparator::MAP_KEY_DELIM, '&')},
      {"null string from table parameters",
       {},
       {{TableParameter::kSerializationNullFormat, "x-x"}},
       withNullString("x-x")},
      {"empty null string",
       {},
       {{TableParameter::kSerializationNullFormat, ""}},
       withNullString("")},
      {"empty escape character means backslash",
       {{SerDeOptions::kEscapeChar, ""}},
       {},
       withEscape('\\')},
      {"escape character code",
       {{SerDeOptions::kEscapeChar, "38"}},
       {},
       withEscape('&')},
      {"escape character code out of range uses the first character",
       {{SerDeOptions::kEscapeChar, "381"}},
       {},
       withEscape('3')},
      {"escape string uses the first character",
       {{SerDeOptions::kEscapeChar, "7!"}},
       {},
       withEscape('7')},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    const auto actual = configureReader(
                            makeSplit(FileFormat::TEXT),
                            /*session=*/{},
                            /*config=*/{},
                            testCase.tableParameters,
                            testCase.serdeParameters)
                            .serDeOptions();
    EXPECT_EQ(actual.separators, testCase.expected.separators);
    EXPECT_EQ(actual.nullString, testCase.expected.nullString);
    EXPECT_EQ(actual.escapeChar, testCase.expected.escapeChar);
    EXPECT_EQ(actual.isEscaped, testCase.expected.isEscaped);
  }

  // Serde options apply only on the call that sets the file format.
  auto context = makeQueryContext();
  auto options = newReaderOptions();
  const auto configure = [&](const Properties& serdeParameters) {
    FileSplitReader::configureReaderOptions(
        makeFileConfig(),
        context.ctx.get(),
        makeSplit(FileFormat::TEXT),
        /*fileSchema=*/nullptr,
        /*tableParameters=*/{},
        serdeParameters,
        options);
  };
  configure({{SerDeOptions::kFieldDelim, "~"}});
  configure({{SerDeOptions::kFieldDelim, "$"}});
  EXPECT_EQ(
      options.serDeOptions()
          .separators[static_cast<size_t>(SerDeSeparator::FIELD_DELIM)],
      '~');
}

TEST_F(FileSplitReaderTest, rowReaderOptionsFromArguments) {
  auto context = makeQueryContext();
  const auto split = std::make_shared<FileConnectorSplit>(
      "testConnectorId",
      "/tmp/testfile",
      FileFormat::SST,
      /*start=*/100,
      /*length=*/5'000);
  const auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  const auto requestedType = ROW("c0", BIGINT());
  const Properties serdeParameters{{"key_col_indices", "0,1,2"}};
  folly::InlineExecutor executor;

  dwio::common::RowReaderOptions options;
  FileSplitReader::configureRowReaderOptions(
      makeFileConfig(),
      context.ctx.get(),
      split,
      scanSpec,
      /*metadataFilter=*/nullptr,
      requestedType,
      &executor,
      {{dwio::common::TableParameter::kSkipHeaderLineCount, "5"}},
      serdeParameters,
      options);

  EXPECT_EQ(options.scanSpec(), scanSpec);
  EXPECT_EQ(options.requestedType(), requestedType);
  EXPECT_EQ(options.ioExecutor(), &executor);
  EXPECT_EQ(options.offset(), 100);
  EXPECT_EQ(options.length(), 5'000);
  EXPECT_FALSE(options.splitRowRange().has_value());
  EXPECT_EQ(options.skipRows(), 5);
  EXPECT_EQ(options.serdeParameters(), serdeParameters);
}

TEST_F(FileSplitReaderTest, rowReaderOptionsFromSession) {
  const auto options = configureRowReader(
      makeSplit(FileFormat::NIMBLE),
      {
          {FileConfig::kReadTimestampUnitSession, "6"},
          {FileConfig::kParallelUnitLoadCountSession, "4"},
          {FileConfig::kPreserveFlatMapsInMemorySession, "true"},
          {FileConfig::kIndexEnabledSession, "true"},
          {FileConfig::kNimbleLazyColumnIoSession, "true"},
          {FileConfig::kReaderCollectColumnCpuMetricsSession, "true"},
          {FileConfig::kNimbleFsstDecodeRepeatedPayloadsSession, "true"},
          {FileConfig::kNimblePreserveDictionaryEncodingSession, "true"},
          {FileConfig::kNimbleDictionaryAwareReadsSession, "true"},
          {FileConfig::kNimbleScanFastPathSession, "true"},
      });

  EXPECT_EQ(options.timestampPrecision(), TimestampPrecision::kMicroseconds);
  EXPECT_EQ(options.parallelUnitLoadCount(), 4);
  EXPECT_TRUE(options.preserveFlatMapsInMemory());
  EXPECT_TRUE(options.indexEnabled());
  EXPECT_TRUE(options.lazyColumnIo());
  EXPECT_TRUE(options.collectColumnCpuMetrics());
  EXPECT_TRUE(options.nimbleFsstDecodeRepeatedPayloads());
  EXPECT_TRUE(options.nimblePreserveDictionaryEncoding());
  EXPECT_TRUE(options.nimbleDictionaryAwareReads());
  EXPECT_TRUE(options.nimbleScanFastPath());
}

TEST_F(FileSplitReaderTest, splitRowRange) {
  const dwio::common::SplitRowRange splitRowRange{
      .firstRow = 8'000, .numRows = 8'000};
  const auto configure = [&](FileFormat format) {
    auto split = std::make_shared<FileConnectorSplit>(
        "testConnectorId", "/tmp/testfile", format);
    split->splitRowRange = splitRowRange;
    return configureRowReader(split);
  };

  for (const auto format :
       {FileFormat::NIMBLE,
        FileFormat::DWRF,
        FileFormat::ORC,
        FileFormat::PARQUET,
        FileFormat::TEXT}) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    EXPECT_EQ(configure(format).splitRowRange(), splitRowRange);
  }
  VELOX_ASSERT_THROW(
      configure(FileFormat::JSON),
      "Split row ranges are only supported for Nimble, DWRF, ORC, Parquet and text files");
}

TEST_F(FileSplitReaderTest, partitionKeyFilters) {
  const auto fileType = ROW("c0", BIGINT());
  const std::unordered_map<std::string, FileColumnHandlePtr> handles{
      {"ds", std::make_shared<PartitionKeyHandle>("ds", VARCHAR())},
      {"hour", std::make_shared<PartitionKeyHandle>("hour", BIGINT())},
  };
  common::ScanSpec scanSpec("<root>");
  scanSpec.addField("c0", 0);
  scanSpec.addField("ds", 1)->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"2024-01-01"}, false));
  scanSpec.addField("hour", 2)->setFilter(
      std::make_unique<common::BigintRange>(0, 11, false));

  const auto test =
      [&](const std::unordered_map<std::string, std::optional<std::string>>&
              partitionKeys) {
        return FileSplitReader::testPartitionAndConstantFilters(
            scanSpec, *fileType, partitionKeys, handles, /*asLocalTime=*/false);
      };

  EXPECT_TRUE(test({{"ds", "2024-01-01"}, {"hour", "10"}}));
  EXPECT_FALSE(test({{"ds", "2024-02-15"}, {"hour", "10"}}));
  // Every filtered key is tested, not just the first.
  EXPECT_FALSE(test({{"ds", "2024-01-01"}, {"hour", "23"}}));
  // A NULL partition value fails a filter that rejects NULL.
  EXPECT_FALSE(test({{"ds", std::nullopt}, {"hour", "10"}}));
}

// A constant decides the filter even when the file carries the column. A NULL
// constant decides from testNull() alone.
TEST_F(FileSplitReaderTest, constantFilters) {
  const auto fileType = ROW("c0", BIGINT());
  const auto keepsFile = [&](const VectorPtr& constant,
                             std::shared_ptr<common::Filter> filter) {
    common::ScanSpec scanSpec("<root>");
    auto* child = scanSpec.addField("c0", 0);
    child->setConstantValue(constant);
    child->setFilter(std::move(filter));
    return FileSplitReader::testPartitionAndConstantFilters(
        scanSpec,
        *fileType,
        /*partitionKeys=*/{},
        /*partitionKeyHandles=*/{},
        /*asLocalTime=*/false);
  };

  const auto constant = makeConstant<int64_t>(1'000, 1);
  EXPECT_FALSE(
      keepsFile(constant, std::make_shared<common::BigintRange>(0, 99, false)));
  EXPECT_TRUE(keepsFile(
      constant, std::make_shared<common::BigintRange>(1'000, 1'000, false)));

  const auto nullConstant = makeNullConstant(TypeKind::BIGINT, 1);
  EXPECT_FALSE(keepsFile(nullConstant, std::make_shared<common::IsNotNull>()));
  EXPECT_TRUE(keepsFile(nullConstant, std::make_shared<common::IsNull>()));
}

// A column missing from the file, as after adding it to the table, reads as
// NULL.
TEST_F(FileSplitReaderTest, missingColumnFilters) {
  common::ScanSpec scanSpec("<root>");
  scanSpec.addField("c0", 0);
  scanSpec.addField("added", 1)
      ->setFilter(
          std::make_unique<common::BytesValues>(
              std::vector<std::string>{"someValue"}, false));

  EXPECT_FALSE(
      FileSplitReader::testPartitionAndConstantFilters(
          scanSpec,
          *ROW("c0", BIGINT()),
          /*partitionKeys=*/{},
          /*partitionKeyHandles=*/{},
          /*asLocalTime=*/false));
}

} // namespace
} // namespace facebook::velox::connector
