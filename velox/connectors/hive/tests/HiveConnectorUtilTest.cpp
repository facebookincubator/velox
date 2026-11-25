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

#include "velox/connectors/hive/HiveConnectorUtil.h"
#include <gtest/gtest.h>
#include "velox/common/Casts.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/common/Config.h"
#include "velox/dwio/orc/reader/OrcReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/expression/FieldReference.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

#include "velox/dwio/dwrf/writer/Writer.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#endif

namespace facebook::velox::connector {

using namespace dwio::common;

namespace {
class DummyNimbleReaderFactory : public dwio::common::ReaderFactory {
 public:
  explicit DummyNimbleReaderFactory(FileFormat format)
      : ReaderFactory(format) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput>,
      const dwio::common::ReaderOptions&) override {
    VELOX_NYI("Test reader factory could not create readers.");
  }

  std::shared_ptr<dwio::common::FormatSpecificOptions> createFormatOptions(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) const override {
    assertUnqualifiedFormatConfigs(connectorConfig);
    assertUnqualifiedFormatConfigs(session);
    return nullptr;
  }

 private:
  static void assertUnqualifiedFormatConfigs(const config::ConfigBase& config) {
    for (const auto& [key, _] : config.rawConfigs()) {
      VELOX_CHECK(
          key.find('.') == std::string::npos,
          "Unexpected qualified format config key: {}",
          key);
    }
  }
};
} // namespace

class HiveConnectorUtilTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    orc::registerOrcReaderFactory();
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
#endif
    dwio::common::registerReaderFactory(
        std::make_shared<DummyNimbleReaderFactory>(FileFormat::NIMBLE));
  }

  void TearDown() override {
    dwio::common::unregisterReaderFactory(FileFormat::NIMBLE);
#ifdef VELOX_ENABLE_PARQUET
    parquet::unregisterParquetReaderFactory();
#endif
    orc::unregisterOrcReaderFactory();
    HiveConnectorTestBase::TearDown();
  }

  static bool compareSerDeOptions(
      const SerDeOptions& l,
      const SerDeOptions& r) {
    return l.isEscaped == r.isEscaped && l.escapeChar == r.escapeChar &&
        l.lastColumnTakesRest == r.lastColumnTakesRest &&
        l.nullString == r.nullString && l.separators == r.separators;
  }

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
  std::shared_ptr<velox::io::IoStatistics> dataIoStats_ =
      std::make_shared<velox::io::IoStatistics>();
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_ =
      std::make_shared<velox::io::IoStatistics>();
};

TEST_F(HiveConnectorUtilTest, configureReaderOptions) {
  config::ConfigBase sessionProperties({});
  auto connectorQueryCtx = connector::ConnectorQueryCtx::Builder()
                               .operatorPool(pool_.get())
                               .connectorPool(pool_.get())
                               .sessionProperties(&sessionProperties)
                               .queryId("query.HiveConnectorUtilTest")
                               .taskId("task.HiveConnectorUtilTest")
                               .planNodeId("planNodeId.HiveConnectorUtilTest")
                               .build();
  auto hiveConfig =
      std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;
  const std::unordered_map<std::string, std::string> customSplitInfo;

  // Dynamic parameters.
  dwio::common::ReaderOptions readerOptions(pool_.get());
  readerOptions.setDataIoStats(dataIoStats_);
  readerOptions.setMetadataIoStats(metadataIoStats_);
  FileFormat fileFormat{FileFormat::DWRF};
  std::unordered_map<std::string, std::string> tableParameters;
  std::unordered_map<std::string, std::string> serdeParameters;
  SerDeOptions expectedSerDe;

  auto createTableHandle = [&]() {
    return std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        common::SubfieldFilters{},
        nullptr,
        nullptr,
        /*indexColumns=*/std::vector<std::string>{},
        tableParameters);
  };

  auto createSplit = [&]() {
    return std::make_shared<hive::HiveConnectorSplit>(
        "testConnectorId",
        "/tmp/",
        fileFormat,
        0UL,
        std::numeric_limits<uint64_t>::max(),
        partitionKeys,
        std::nullopt,
        customSplitInfo,
        nullptr,
        serdeParameters);
  };

  auto performConfigure = [&]() {
    auto tableHandle = createTableHandle();
    auto split = createSplit();
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
  };

  auto clearDynamicParameters = [&](FileFormat newFileFormat) {
    readerOptions = dwio::common::ReaderOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    fileFormat = newFileFormat;
    tableParameters.clear();
    serdeParameters.clear();
    expectedSerDe = SerDeOptions{};
  };

  auto checkColumnMappingMode = [&]() {
    const auto expectedMappingMode =
        hiveConfig->useColumnNames(&sessionProperties)
        ? dwio::common::ColumnMappingMode::kName
        : dwio::common::ColumnMappingMode::kPosition;
    EXPECT_EQ(readerOptions.columnMappingMode(), expectedMappingMode);
  };

  // Default.
  performConfigure();
  EXPECT_EQ(readerOptions.fileFormat(), fileFormat);
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));
  EXPECT_EQ(
      readerOptions.loadQuantum(), hiveConfig->loadQuantum(&sessionProperties));
  EXPECT_EQ(
      readerOptions.maxCoalesceBytes(),
      hiveConfig->maxCoalescedBytes(&sessionProperties));
  EXPECT_EQ(
      readerOptions.maxCoalesceDistance(),
      hiveConfig->maxCoalescedDistanceBytes(&sessionProperties));
  EXPECT_EQ(
      readerOptions.fileColumnNamesReadAsLowerCase(),
      hiveConfig->isFileColumnNamesReadAsLowerCase(&sessionProperties));
  checkColumnMappingMode();
  EXPECT_EQ(
      readerOptions.filePreloadThreshold(), hiveConfig->filePreloadThreshold());
  EXPECT_EQ(readerOptions.prefetchRowGroups(), hiveConfig->prefetchRowGroups());
  EXPECT_EQ(
      readerOptions.cacheMetadata(),
      hiveConfig->cacheMetadata(&sessionProperties));

  // Modify field delimiter and change the file format.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kFieldDelim] = '\t';
  expectedSerDe.separators[size_t(SerDeSeparator::FIELD_DELIM)] = '\t';
  performConfigure();
  EXPECT_EQ(readerOptions.fileFormat(), fileFormat);
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Modify collection delimiter.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kCollectionDelim] = '=';
  expectedSerDe.separators[size_t(SerDeSeparator::COLLECTION_DELIM)] = '=';
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Modify map key delimiter.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kMapKeyDelim] = '&';
  expectedSerDe.separators[size_t(SerDeSeparator::MAP_KEY_DELIM)] = '&';
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Modify null string.
  clearDynamicParameters(FileFormat::TEXT);
  tableParameters[TableParameter::kSerializationNullFormat] = "x-x";
  expectedSerDe.nullString = "x-x";
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Empty escape delim means default escape char.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kEscapeChar] = "";
  expectedSerDe.escapeChar = '\\';
  expectedSerDe.isEscaped = true;
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Convertible to byte escape char - use it.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kEscapeChar] = "38";
  expectedSerDe.escapeChar = '&';
  expectedSerDe.isEscaped = true;
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Overflow byte escape char - fall back to the 1st character of the string.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kEscapeChar] = "381";
  expectedSerDe.escapeChar = '3';
  expectedSerDe.isEscaped = true;
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Not convertible string - fall back to the 1st character of the string.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kEscapeChar] = "7!";
  expectedSerDe.escapeChar = '7';
  expectedSerDe.isEscaped = true;
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Modify all previous together.
  clearDynamicParameters(FileFormat::TEXT);
  serdeParameters[SerDeOptions::kFieldDelim] = '~';
  expectedSerDe.separators[size_t(SerDeSeparator::FIELD_DELIM)] = '~';
  serdeParameters[SerDeOptions::kCollectionDelim] = '$';
  expectedSerDe.separators[size_t(SerDeSeparator::COLLECTION_DELIM)] = '$';
  serdeParameters[SerDeOptions::kMapKeyDelim] = '*';
  expectedSerDe.separators[size_t(SerDeSeparator::MAP_KEY_DELIM)] = '*';
  serdeParameters[SerDeOptions::kEscapeChar] = '*';
  expectedSerDe.escapeChar = '*';
  expectedSerDe.isEscaped = true;
  tableParameters[TableParameter::kSerializationNullFormat] = "";
  expectedSerDe.nullString = "";
  performConfigure();
  EXPECT_TRUE(compareSerDeOptions(readerOptions.serDeOptions(), expectedSerDe));

  // Tests other custom reader options.
  clearDynamicParameters(FileFormat::TEXT);
  std::unordered_map<std::string, std::string> customHiveConfigProps;
  customHiveConfigProps[hive::HiveConfig::kLoadQuantum] = "321";
  customHiveConfigProps[hive::HiveConfig::kMaxCoalescedBytes] = "129";
  customHiveConfigProps[hive::HiveConfig::kMaxCoalescedDistance] = "513KB";
  customHiveConfigProps[hive::HiveConfig::kFileColumnNamesReadAsLowerCase] =
      "true";
  customHiveConfigProps["hive.use-column-names"] = "true";
  customHiveConfigProps[hive::HiveConfig::kFilePreloadThreshold] = "9999";
  customHiveConfigProps[hive::HiveConfig::kPrefetchRowGroups] = "10";
  customHiveConfigProps[hive::HiveConfig::kCacheMetadata] = "true";
  customHiveConfigProps[hive::HiveConfig::kOrcFooterSpeculativeIoSize] = "1111";
  hiveConfig = std::make_shared<hive::HiveConfig>(
      std::make_shared<config::ConfigBase>(std::move(customHiveConfigProps)));
  performConfigure();
  EXPECT_EQ(
      readerOptions.loadQuantum(), hiveConfig->loadQuantum(&sessionProperties));
  EXPECT_EQ(
      readerOptions.maxCoalesceBytes(),
      hiveConfig->maxCoalescedBytes(&sessionProperties));
  EXPECT_EQ(
      readerOptions.maxCoalesceDistance(),
      hiveConfig->maxCoalescedDistanceBytes(&sessionProperties));
  EXPECT_EQ(
      readerOptions.fileColumnNamesReadAsLowerCase(),
      hiveConfig->isFileColumnNamesReadAsLowerCase(&sessionProperties));
  EXPECT_EQ(
      readerOptions.footerSpeculativeIoSize(),
      hiveConfig->orcFooterSpeculativeIoSize(&sessionProperties));
  EXPECT_EQ(
      readerOptions.filePreloadThreshold(), hiveConfig->filePreloadThreshold());
  EXPECT_EQ(readerOptions.prefetchRowGroups(), hiveConfig->prefetchRowGroups());
  EXPECT_TRUE(readerOptions.cacheMetadata());
  clearDynamicParameters(FileFormat::ORC);
  performConfigure();
  EXPECT_EQ(
      readerOptions.maxCoalesceDistance(),
      hiveConfig->maxCoalescedDistanceBytes(&sessionProperties));
  checkColumnMappingMode();
  clearDynamicParameters(FileFormat::PARQUET);
  performConfigure();
  checkColumnMappingMode();
}

TEST_F(HiveConnectorUtilTest, footerSpeculativeIoSizeByFormat) {
  config::ConfigBase sessionProperties{
      std::unordered_map<std::string, std::string>{
          {"nimble.unused", "1"},
          {"parquet.unused", "2"},
          {"parquet_footer_speculative_io_size", "7777"},
          {"parquet_footer_memory_tracking_threshold", "6666"},
          {"hive.parquet.footer_speculative_io_size", "8888"},
          {"unrelated.unused", "3"},
      }};
  auto connectorQueryCtx = connector::ConnectorQueryCtx::Builder()
                               .operatorPool(pool_.get())
                               .connectorPool(pool_.get())
                               .sessionProperties(&sessionProperties)
                               .queryId("query.HiveConnectorUtilTest")
                               .taskId("task.HiveConnectorUtilTest")
                               .planNodeId("planNodeId.HiveConnectorUtilTest")
                               .build();

  std::unordered_map<std::string, std::string> customHiveConfigProps;
  customHiveConfigProps[hive::HiveConfig::kOrcFooterSpeculativeIoSize] = "1111";
#ifdef VELOX_ENABLE_PARQUET
  customHiveConfigProps["parquet.footer-speculative-io-size"] = "9999";
  customHiveConfigProps["parquet.footer-memory-tracking-threshold"] = "9999";
  customHiveConfigProps["hive.parquet.footer-speculative-io-size"] = "2222";
  customHiveConfigProps["hive.parquet.footer-memory-tracking-threshold"] =
      "5555";
  customHiveConfigProps["iceberg.parquet.footer-speculative-io-size"] = "4444";
#endif
  customHiveConfigProps[hive::HiveConfig::kNimbleFooterSpeculativeIoSize] =
      "3333";
  auto hiveConfig = std::make_shared<hive::HiveConfig>(
      std::make_shared<config::ConfigBase>(std::move(customHiveConfigProps)));

  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;
  const std::unordered_map<std::string, std::string> customSplitInfo;
  std::unordered_map<std::string, std::string> serdeParameters;

  auto createTableHandle = []() {
    return std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        common::SubfieldFilters{},
        nullptr,
        nullptr,
        std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{});
  };

  auto createSplit = [&](FileFormat format) {
    return std::make_shared<hive::HiveConnectorSplit>(
        "testConnectorId",
        "/tmp/",
        format,
        0UL,
        std::numeric_limits<uint64_t>::max(),
        partitionKeys,
        std::nullopt,
        customSplitInfo,
        nullptr,
        serdeParameters);
  };

  // Test ORC format.
  {
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    auto tableHandle = createTableHandle();
    auto split = createSplit(FileFormat::ORC);
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
    EXPECT_EQ(
        readerOptions.footerSpeculativeIoSize(),
        hiveConfig->orcFooterSpeculativeIoSize(&sessionProperties));
    EXPECT_EQ(readerOptions.footerSpeculativeIoSize(), 1111);
  }

  // Test DWRF format (uses ORC config).
  {
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    auto tableHandle = createTableHandle();
    auto split = createSplit(FileFormat::DWRF);
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
    EXPECT_EQ(
        readerOptions.footerSpeculativeIoSize(),
        hiveConfig->orcFooterSpeculativeIoSize(&sessionProperties));
    EXPECT_EQ(readerOptions.footerSpeculativeIoSize(), 1111);
  }

#ifdef VELOX_ENABLE_PARQUET
  // Test Parquet format.
  {
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    auto tableHandle = createTableHandle();
    auto split = createSplit(FileFormat::PARQUET);
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
    auto parquetOptions = checkedPointerCast<parquet::ParquetReaderOptions>(
        readerOptions.formatSpecificOptions());
    EXPECT_EQ(parquetOptions->footerSpeculativeIoSize, 7777);
    EXPECT_EQ(parquetOptions->footerMemoryTrackingThreshold(), 6666);
  }
#endif

  // Test Nimble format.
  {
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);
    auto tableHandle = createTableHandle();
    auto split = createSplit(FileFormat::NIMBLE);
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
    EXPECT_EQ(
        readerOptions.footerSpeculativeIoSize(),
        hiveConfig->nimbleFooterSpeculativeIoSize(&sessionProperties));
    EXPECT_EQ(readerOptions.footerSpeculativeIoSize(), 3333);
  }
}

TEST_F(HiveConnectorUtilTest, cacheMetadataSessionOverride) {
  // Verify default is off.
  dwio::common::ReaderOptions defaultOptions(pool_.get());
  defaultOptions.setDataIoStats(dataIoStats_);
  defaultOptions.setMetadataIoStats(metadataIoStats_);
  ASSERT_FALSE(defaultOptions.cacheMetadata());

  for (bool enabled : {true, false}) {
    SCOPED_TRACE(fmt::format("cacheMetadata={}", enabled));

    config::ConfigBase sessionProperties(
        std::unordered_map<std::string, std::string>{
            {hive::HiveConfig::kCacheMetadataSession,
             enabled ? "true" : "false"}});
    auto connectorQueryCtx = connector::ConnectorQueryCtx::Builder()
                                 .operatorPool(pool_.get())
                                 .connectorPool(pool_.get())
                                 .sessionProperties(&sessionProperties)
                                 .queryId("query.HiveConnectorUtilTest")
                                 .taskId("task.HiveConnectorUtilTest")
                                 .planNodeId("planNodeId.HiveConnectorUtilTest")
                                 .build();
    auto hiveConfig =
        std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);

    auto tableHandle = std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        common::SubfieldFilters{},
        nullptr,
        nullptr,
        std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{});
    auto split = std::make_shared<hive::HiveConnectorSplit>(
        "testConnectorId", "/tmp/", FileFormat::DWRF);
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
    ASSERT_EQ(readerOptions.cacheMetadata(), enabled);
  }
}

TEST_F(HiveConnectorUtilTest, cacheIndexSessionOverride) {
  dwio::common::ReaderOptions defaultOptions(pool_.get());
  defaultOptions.setDataIoStats(dataIoStats_);
  defaultOptions.setMetadataIoStats(metadataIoStats_);
  ASSERT_FALSE(defaultOptions.cacheIndex());

  for (bool enabled : {true, false}) {
    SCOPED_TRACE(fmt::format("cacheIndex={}", enabled));

    config::ConfigBase sessionProperties(
        std::unordered_map<std::string, std::string>{
            {hive::HiveConfig::kCacheIndexSession,
             enabled ? "true" : "false"}});
    auto connectorQueryCtx = connector::ConnectorQueryCtx::Builder()
                                 .operatorPool(pool_.get())
                                 .connectorPool(pool_.get())
                                 .sessionProperties(&sessionProperties)
                                 .queryId("query.HiveConnectorUtilTest")
                                 .taskId("task.HiveConnectorUtilTest")
                                 .planNodeId("planNodeId.HiveConnectorUtilTest")
                                 .build();
    auto hiveConfig =
        std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);

    auto tableHandle = std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        common::SubfieldFilters{},
        nullptr,
        nullptr,
        std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{});
    auto split = std::make_shared<hive::HiveConnectorSplit>(
        "testConnectorId", "/tmp/", FileFormat::DWRF);
    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        split,
        split->serdeParameters,
        readerOptions);
    ASSERT_EQ(readerOptions.cacheIndex(), enabled);
    ASSERT_EQ(readerOptions.pinIndex(), false);
  }
}

TEST_F(HiveConnectorUtilTest, cacheRetention) {
  struct {
    bool splitCacheable;
    bool expectedCacheable;

    std::string debugString() const {
      return fmt::format(
          "splitCacheable {}, expectedCacheable {}",
          splitCacheable,
          expectedCacheable);
    }
  } testSettings[] = {{false, false}, {true, true}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    config::ConfigBase sessionProperties({});
    auto hiveConfig =
        std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));

    auto connectorQueryCtx = connector::ConnectorQueryCtx::Builder()
                                 .operatorPool(pool_.get())
                                 .connectorPool(pool_.get())
                                 .sessionProperties(&sessionProperties)
                                 .queryId("query.HiveConnectorUtilTest")
                                 .taskId("task.HiveConnectorUtilTest")
                                 .planNodeId("planNodeId.HiveConnectorUtilTest")
                                 .build();

    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setDataIoStats(dataIoStats_);
    readerOptions.setMetadataIoStats(metadataIoStats_);

    auto tableHandle = std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        common::SubfieldFilters{},
        nullptr,
        nullptr,
        /*indexColumns=*/std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{});

    auto hiveSplit = std::make_shared<hive::HiveConnectorSplit>(
        "testConnectorId",
        "/tmp/",
        FileFormat::DWRF,
        0UL,
        std::numeric_limits<uint64_t>::max(),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        std::shared_ptr<std::string>{},
        std::unordered_map<std::string, std::string>{},
        0,
        testData.splitCacheable);

    configureReaderOptions(
        hiveConfig,
        connectorQueryCtx.get(),
        tableHandle,
        hiveSplit,
        hiveSplit->serdeParameters,
        readerOptions);

    ASSERT_EQ(readerOptions.cacheable(), testData.expectedCacheable);
  }
}

TEST_F(HiveConnectorUtilTest, configureSstRowReaderOptions) {
  dwio::common::RowReaderOptions rowReaderOpts;
  auto hiveSplit =
      std::make_shared<hive::HiveConnectorSplit>("", "", FileFormat::SST);
  hiveSplit->serdeParameters = {
      {"key_col_indices", "0,1,2"},
      {"value_col_indices", "4,5"},
  };
  configureRowReaderOptions(
      /*tableParameters=*/{},
      /*scanSpec=*/nullptr,
      /*metadataFilter=*/nullptr,
      /*rowType=*/nullptr,
      /*fileSplit=*/hiveSplit,
      /*serdeParameters=*/hiveSplit->serdeParameters,
      /*fileConfig=*/nullptr,
      /*sessionProperties=*/nullptr,
      /*ioExecutor=*/nullptr,
      /*rowReaderOptions=*/rowReaderOpts);

  EXPECT_EQ(rowReaderOpts.serdeParameters(), hiveSplit->serdeParameters);
}

TEST_F(HiveConnectorUtilTest, configureRowReaderOptionsFromConfig) {
  // Test default behavior (preserveFlatMapsInMemory = false)
  {
    auto hiveConfig =
        std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    config::ConfigBase sessionProperties({});

    dwio::common::RowReaderOptions rowReaderOpts;
    auto hiveSplit =
        std::make_shared<hive::HiveConnectorSplit>("", "", FileFormat::DWRF);

    configureRowReaderOptions(
        /*tableParameters=*/{},
        /*scanSpec=*/nullptr,
        /*metadataFilter=*/nullptr,
        /*rowType=*/nullptr,
        /*fileSplit=*/hiveSplit,
        /*serdeParameters=*/{},
        /*fileConfig=*/hiveConfig,
        /*sessionProperties=*/&sessionProperties,
        /*ioExecutor=*/nullptr,
        /*rowReaderOptions=*/rowReaderOpts);

    EXPECT_FALSE(rowReaderOpts.preserveFlatMapsInMemory());
  }

  // Test with config override (preserveFlatMapsInMemory = true)
  {
    std::unordered_map<std::string, std::string> configProps = {
        {hive::HiveConfig::kPreserveFlatMapsInMemory, "true"}};
    auto hiveConfig = std::make_shared<hive::HiveConfig>(
        std::make_shared<config::ConfigBase>(std::move(configProps)));
    config::ConfigBase sessionProperties({});

    dwio::common::RowReaderOptions rowReaderOpts;
    auto hiveSplit =
        std::make_shared<hive::HiveConnectorSplit>("", "", FileFormat::DWRF);

    configureRowReaderOptions(
        /*tableParameters=*/{},
        /*scanSpec=*/nullptr,
        /*metadataFilter=*/nullptr,
        /*rowType=*/nullptr,
        /*fileSplit=*/hiveSplit,
        /*serdeParameters=*/{},
        /*fileConfig=*/hiveConfig,
        /*sessionProperties=*/&sessionProperties,
        /*ioExecutor=*/nullptr,
        /*rowReaderOptions=*/rowReaderOpts);

    EXPECT_TRUE(rowReaderOpts.preserveFlatMapsInMemory());
  }

  // Test with session override (preserveFlatMapsInMemory = true)
  {
    auto hiveConfig =
        std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>()));
    std::unordered_map<std::string, std::string> sessionProps = {
        {hive::HiveConfig::kPreserveFlatMapsInMemorySession, "true"}};
    config::ConfigBase sessionProperties(std::move(sessionProps));

    dwio::common::RowReaderOptions rowReaderOpts;
    auto hiveSplit =
        std::make_shared<hive::HiveConnectorSplit>("", "", FileFormat::DWRF);

    configureRowReaderOptions(
        /*tableParameters=*/{},
        /*scanSpec=*/nullptr,
        /*metadataFilter=*/nullptr,
        /*rowType=*/nullptr,
        /*fileSplit=*/hiveSplit,
        /*serdeParameters=*/{},
        /*fileConfig=*/hiveConfig,
        /*sessionProperties=*/&sessionProperties,
        /*ioExecutor=*/nullptr,
        /*rowReaderOptions=*/rowReaderOpts);

    EXPECT_TRUE(rowReaderOpts.preserveFlatMapsInMemory());
  }

  // Test session override takes precedence over config
  {
    std::unordered_map<std::string, std::string> configProps = {
        {hive::HiveConfig::kPreserveFlatMapsInMemory, "false"}};
    auto hiveConfig = std::make_shared<hive::HiveConfig>(
        std::make_shared<config::ConfigBase>(std::move(configProps)));
    std::unordered_map<std::string, std::string> sessionProps = {
        {hive::HiveConfig::kPreserveFlatMapsInMemorySession, "true"}};
    config::ConfigBase sessionProperties(std::move(sessionProps));

    dwio::common::RowReaderOptions rowReaderOpts;
    auto hiveSplit =
        std::make_shared<hive::HiveConnectorSplit>("", "", FileFormat::DWRF);

    configureRowReaderOptions(
        /*tableParameters=*/{},
        /*scanSpec=*/nullptr,
        /*metadataFilter=*/nullptr,
        /*rowType=*/nullptr,
        /*fileSplit=*/hiveSplit,
        /*serdeParameters=*/{},
        /*fileConfig=*/hiveConfig,
        /*sessionProperties=*/&sessionProperties,
        /*ioExecutor=*/nullptr,
        /*rowReaderOptions=*/rowReaderOpts);

    EXPECT_TRUE(rowReaderOpts.preserveFlatMapsInMemory());
  }
}

} // namespace facebook::velox::connector
