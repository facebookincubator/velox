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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/expression/FieldReference.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

#include "velox/dwio/dwrf/writer/Writer.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/writer/Writer.h"
#endif

namespace facebook::velox::connector {

using namespace dwio::common;

namespace {
// Unsupported types for createPointFilter and createRangeFilter with test
// values.
struct UnsupportedFilterType {
  TypePtr type;
  variant value;
};

const std::vector<UnsupportedFilterType> kUnsupportedFilterTypes = {
    {TIMESTAMP(), variant(Timestamp(0, 0))},
    {ARRAY(BIGINT()), variant::array({variant::create<TypeKind::BIGINT>(1)})},
    {MAP(VARCHAR(), BIGINT()),
     variant::map({{variant("key"), variant::create<TypeKind::BIGINT>(1)}})},
    {ROW({{"a", BIGINT()}}),
     variant::row({variant::create<TypeKind::BIGINT>(1)})},
};
} // namespace

class HiveConnectorUtilTest : public exec::test::HiveConnectorTestBase {
 protected:
  static bool compareSerDeOptions(
      const SerDeOptions& l,
      const SerDeOptions& r) {
    return l.isEscaped == r.isEscaped && l.escapeChar == r.escapeChar &&
        l.lastColumnTakesRest == r.lastColumnTakesRest &&
        l.nullString == r.nullString && l.separators == r.separators;
  }

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

TEST_F(HiveConnectorUtilTest, configureReaderOptions) {
  config::ConfigBase sessionProperties({});
  auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
      pool_.get(),
      pool_.get(),
      &sessionProperties,
      nullptr,
      common::PrefixSortConfig(),
      nullptr,
      nullptr,
      "query.HiveConnectorUtilTest",
      "task.HiveConnectorUtilTest",
      "planNodeId.HiveConnectorUtilTest",
      0,
      "");
  auto hiveConfig =
      std::make_shared<hive::HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys;
  const std::unordered_map<std::string, std::string> customSplitInfo;

  // Dynamic parameters.
  dwio::common::ReaderOptions readerOptions(pool_.get());
  FileFormat fileFormat{FileFormat::DWRF};
  std::unordered_map<std::string, std::string> tableParameters;
  std::unordered_map<std::string, std::string> serdeParameters;
  SerDeOptions expectedSerDe;

  auto createTableHandle = [&]() {
    return std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        false,
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
        hiveConfig, connectorQueryCtx.get(), tableHandle, split, readerOptions);
  };

  auto clearDynamicParameters = [&](FileFormat newFileFormat) {
    readerOptions = dwio::common::ReaderOptions(pool_.get());
    fileFormat = newFileFormat;
    tableParameters.clear();
    serdeParameters.clear();
    expectedSerDe = SerDeOptions{};
  };

  auto checkUseColumnNamesForColumnMapping = [&]() {
    if (fileFormat == FileFormat::DWRF || fileFormat == FileFormat::ORC) {
      EXPECT_EQ(
          readerOptions.useColumnNamesForColumnMapping(),
          hiveConfig->isOrcUseColumnNames(&sessionProperties));
    } else if (fileFormat == FileFormat::PARQUET) {
      EXPECT_EQ(
          readerOptions.useColumnNamesForColumnMapping(),
          hiveConfig->isParquetUseColumnNames(&sessionProperties));
    }
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
  checkUseColumnNamesForColumnMapping();
  EXPECT_EQ(
      readerOptions.footerEstimatedSize(), hiveConfig->footerEstimatedSize());
  EXPECT_EQ(
      readerOptions.filePreloadThreshold(), hiveConfig->filePreloadThreshold());
  EXPECT_EQ(readerOptions.prefetchRowGroups(), hiveConfig->prefetchRowGroups());

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
  customHiveConfigProps[hive::HiveConfig::kOrcUseColumnNames] = "true";
  customHiveConfigProps[hive::HiveConfig::kFooterEstimatedSize] = "1111";
  customHiveConfigProps[hive::HiveConfig::kFilePreloadThreshold] = "9999";
  customHiveConfigProps[hive::HiveConfig::kPrefetchRowGroups] = "10";
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
      readerOptions.footerEstimatedSize(), hiveConfig->footerEstimatedSize());
  EXPECT_EQ(
      readerOptions.filePreloadThreshold(), hiveConfig->filePreloadThreshold());
  EXPECT_EQ(readerOptions.prefetchRowGroups(), hiveConfig->prefetchRowGroups());
  clearDynamicParameters(FileFormat::ORC);
  performConfigure();
  checkUseColumnNamesForColumnMapping();
  clearDynamicParameters(FileFormat::PARQUET);
  performConfigure();
  checkUseColumnNamesForColumnMapping();
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

    auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
        pool_.get(),
        pool_.get(),
        &sessionProperties,
        nullptr,
        common::PrefixSortConfig(),
        nullptr,
        nullptr,
        "query.HiveConnectorUtilTest",
        "task.HiveConnectorUtilTest",
        "planNodeId.HiveConnectorUtilTest",
        0,
        "");

    dwio::common::ReaderOptions readerOptions(pool_.get());

    auto tableHandle = std::make_shared<hive::HiveTableHandle>(
        "testConnectorId",
        "testTable",
        false,
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
      /*hiveSplit=*/hiveSplit,
      /*hiveConfig=*/nullptr,
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
        /*hiveSplit=*/hiveSplit,
        /*hiveConfig=*/hiveConfig,
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
        /*hiveSplit=*/hiveSplit,
        /*hiveConfig=*/hiveConfig,
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
        /*hiveSplit=*/hiveSplit,
        /*hiveConfig=*/hiveConfig,
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
        /*hiveSplit=*/hiveSplit,
        /*hiveConfig=*/hiveConfig,
        /*sessionProperties=*/&sessionProperties,
        /*ioExecutor=*/nullptr,
        /*rowReaderOptions=*/rowReaderOpts);

    EXPECT_TRUE(rowReaderOpts.preserveFlatMapsInMemory());
  }
}

TEST_F(HiveConnectorUtilTest, checkColumnHandleConsistent) {
  // Create two consistent column handles
  auto handle1 = std::make_shared<hive::HiveColumnHandle>(
      "col1", hive::HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
  auto handle2 = std::make_shared<hive::HiveColumnHandle>(
      "col1", hive::HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());

  // Should not throw for consistent handles
  EXPECT_NO_THROW(hive::checkColumnHandleConsistent(*handle1, *handle2));

  // Test inconsistent column type
  auto handlePartition = std::make_shared<hive::HiveColumnHandle>(
      "col1",
      hive::HiveColumnHandle::ColumnType::kPartitionKey,
      BIGINT(),
      BIGINT());
  VELOX_ASSERT_THROW(
      hive::checkColumnHandleConsistent(*handle1, *handlePartition),
      "Inconsistent column handle type: col1, expected Regular, got PartitionKey");

  // Test inconsistent data type
  auto handleVarchar = std::make_shared<hive::HiveColumnHandle>(
      "col1",
      hive::HiveColumnHandle::ColumnType::kRegular,
      VARCHAR(),
      VARCHAR());
  VELOX_ASSERT_THROW(
      hive::checkColumnHandleConsistent(*handle1, *handleVarchar),
      "Inconsistent column handle data type: col1, expected BIGINT, got VARCHAR");
}

TEST_F(HiveConnectorUtilTest, makeScanSpecWithIndexColumns) {
  // Data columns schema - all columns available in the file.
  const auto dataColumns = ROW(
      {{"a", BIGINT()},
       {"b", VARCHAR()},
       {"c", INTEGER()},
       {"d", DOUBLE()},
       {"e", ROW({{"x", INTEGER()}, {"y", VARCHAR()}})}});

  struct TestCase {
    std::string name;
    RowTypePtr rowType;
    folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
        outputSubfields;
    std::function<common::SubfieldFilters()> makeSubfieldFilters;
    std::vector<std::string> indexColumns;
    std::vector<std::string> expectedColumns;
    std::vector<std::string> unexpectedColumns;

    std::string debugString() const {
      return fmt::format(
          "name: {}, indexColumns: [{}], expectedColumns: [{}], unexpectedColumns: [{}]",
          name,
          folly::join(", ", indexColumns),
          folly::join(", ", expectedColumns),
          folly::join(", ", unexpectedColumns));
    }
  };

  // Subfields for nested column 'e'.
  const common::Subfield subfieldEx("e.x");
  const common::Subfield subfieldEy("e.y");

  const std::vector<TestCase> testCases = {
      {
          "Index columns not in output projection",
          ROW({"a", "b"}, {BIGINT(), VARCHAR()}),
          /*outputSubfields=*/{},
          /*makeSubfieldFilters=*/nullptr,
          /*indexColumns=*/{"c", "d"},
          /*expectedColumns=*/{"a", "b", "c", "d"},
          /*unexpectedColumns=*/{"e"},
      },
      {
          "Index column already in output projection",
          ROW({"a", "b"}, {BIGINT(), VARCHAR()}),
          /*outputSubfields=*/{},
          /*makeSubfieldFilters=*/nullptr,
          /*indexColumns=*/{"a", "c"},
          /*expectedColumns=*/{"a", "b", "c"},
          /*unexpectedColumns=*/{"d", "e"},
      },
      {
          "Empty index columns",
          ROW({"a", "b"}, {BIGINT(), VARCHAR()}),
          /*outputSubfields=*/{},
          /*makeSubfieldFilters=*/nullptr,
          /*indexColumns=*/{},
          /*expectedColumns=*/{"a", "b"},
          /*unexpectedColumns=*/{"c", "d", "e"},
      },
      {
          "Output subfield without index columns",
          ROW({"e"}, {ROW({{"x", INTEGER()}, {"y", VARCHAR()}})}),
          /*outputSubfields=*/{{"e", {&subfieldEx}}},
          /*makeSubfieldFilters=*/nullptr,
          /*indexColumns=*/{},
          /*expectedColumns=*/{"e"},
          /*unexpectedColumns=*/{"a", "b", "c", "d"},
      },
      {
          "Output subfield with different index column",
          ROW({"e"}, {ROW({{"x", INTEGER()}, {"y", VARCHAR()}})}),
          /*outputSubfields=*/{{"e", {&subfieldEx}}},
          /*makeSubfieldFilters=*/nullptr,
          /*indexColumns=*/{"c"},
          /*expectedColumns=*/{"e", "c"},
          /*unexpectedColumns=*/{"a", "b", "d"},
      },
      {
          "Output subfield with same parent as index column",
          ROW({"a", "e"},
              {BIGINT(), ROW({{"x", INTEGER()}, {"y", VARCHAR()}})}),
          /*outputSubfields=*/{{"e", {&subfieldEx}}},
          /*makeSubfieldFilters=*/nullptr,
          /*indexColumns=*/{"a", "e"},
          /*expectedColumns=*/{"a", "e"},
          /*unexpectedColumns=*/{"b", "c", "d"},
      },
      {
          "Subfield filter without index column",
          ROW({"a", "b"}, {BIGINT(), VARCHAR()}),
          /*outputSubfields=*/{},
          /*makeSubfieldFilters=*/
          []() {
            common::SubfieldFilters filters;
            filters.emplace(
                common::Subfield("c"), exec::greaterThanOrEqual(10));
            return filters;
          },
          /*indexColumns=*/{"d"},
          /*expectedColumns=*/{"a", "b", "c", "d"},
          /*unexpectedColumns=*/{"e"},
      },
      {
          "Subfield filter without index column",
          ROW({"a", "b"}, {BIGINT(), VARCHAR()}),
          /*outputSubfields=*/{},
          /*makeSubfieldFilters=*/
          []() {
            common::SubfieldFilters filters;
            filters.emplace(
                common::Subfield("c"), exec::greaterThanOrEqual(10));
            return filters;
          },
          /*indexColumns=*/{"c"},
          /*expectedColumns=*/{"a", "b", "c"},
          /*unexpectedColumns=*/{"e"},
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    common::SubfieldFilters subfieldFilters;
    if (testCase.makeSubfieldFilters) {
      subfieldFilters = testCase.makeSubfieldFilters();
    }

    auto scanSpec = hive::makeScanSpec(
        testCase.rowType,
        testCase.outputSubfields,
        subfieldFilters,
        testCase.indexColumns,
        dataColumns,
        /*partitionKeys=*/{},
        /*infoColumns=*/{},
        /*specialColumns=*/{},
        /*disableStatsBasedFilterReorder=*/false,
        pool_.get());

    for (const auto& col : testCase.expectedColumns) {
      EXPECT_NE(scanSpec->childByName(col), nullptr)
          << "Expected column " << col << " to be in scan spec";
    }
    for (const auto& col : testCase.unexpectedColumns) {
      EXPECT_EQ(scanSpec->childByName(col), nullptr)
          << "Unexpected column " << col << " should not be in scan spec";
    }
  }
}

TEST_F(HiveConnectorUtilTest, makeScanSpecWithIndexColumnsError) {
  // Test that makeScanSpec throws when index columns are set but dataColumns
  // is null.
  const auto rowType = ROW({"a", "b"}, {BIGINT(), VARCHAR()});

  VELOX_ASSERT_THROW(
      hive::makeScanSpec(
          rowType,
          /*outputSubfields=*/{},
          /*subfieldFilters=*/{},
          /*indexColumns=*/{"c"},
          /*dataColumns=*/nullptr,
          /*partitionKeys=*/{},
          /*infoColumns=*/{},
          /*specialColumns=*/{},
          /*disableStatsBasedFilterReorder=*/false,
          pool_.get()),
      "");
}

TEST_F(HiveConnectorUtilTest, shouldEagerlyMaterialize) {
  auto queryCtx = core::QueryCtx::create();
  auto execCtx = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx.get());

  auto compileExpression = [&](const std::string& expr,
                               const RowTypePtr& rowType) {
    auto untyped = parse::DuckSqlExpressionsParser().parseExpr(expr);
    auto typedExpr =
        core::Expressions::inferTypes(untyped, rowType, pool_.get());
    std::vector<core::TypedExprPtr> expressions = {typedExpr};
    return std::make_unique<exec::ExprSet>(
        std::move(expressions), execCtx.get());
  };

  const auto rowType = ROW({"a", "b", "c"}, {BIGINT(), BIGINT(), BIGINT()});

  // Test 1: OR expression doesn't evaluate arguments on non-increasing
  // selection, so should return true (eager materialization needed).
  {
    auto exprSet = compileExpression("a > 10 OR b > 20", rowType);
    auto& expr = *exprSet->exprs().front();
    for (const auto& field : expr.distinctFields()) {
      EXPECT_TRUE(hive::shouldEagerlyMaterialize(expr, *field));
    }
  }

  // Test 2: AND expression evaluates arguments on non-increasing selection.
  // Field used in simple conjunct (no conditionals) should not be eagerly
  // materialized.
  {
    auto exprSet = compileExpression("a > 10 AND b > 20", rowType);
    auto& expr = *exprSet->exprs().front();
    for (const auto& field : expr.distinctFields()) {
      EXPECT_FALSE(hive::shouldEagerlyMaterialize(expr, *field));
    }
  }

  // Test 3: AND expression with field used in IF conditional.
  // Field used in input with conditionals should be eagerly materialized.
  {
    auto exprSet =
        compileExpression("a > 10 AND if(b > 20, c < 30, c > 5)", rowType);
    auto& expr = *exprSet->exprs().front();
    for (const auto& field : expr.distinctFields()) {
      if (field->field() == "c" || field->field() == "b") {
        EXPECT_TRUE(hive::shouldEagerlyMaterialize(expr, *field));
      } else {
        EXPECT_FALSE(hive::shouldEagerlyMaterialize(expr, *field));
      }
    }
  }

  // Test 4: AND expression where field is used in simple conjunct,
  // not in conditional.
  {
    auto exprSet =
        compileExpression("a > 10 OR if(b > 20, c < 30, c < 5)", rowType);
    auto& expr = *exprSet->exprs().front();
    // Find field 'a' which is used in simple comparison, not in conditional.
    for (const auto& field : expr.distinctFields()) {
      EXPECT_TRUE(hive::shouldEagerlyMaterialize(expr, *field));
    }
  }
}

TEST_F(HiveConnectorUtilTest, createPointFilter) {
  // Test BIGINT point filter.
  {
    auto filter = hive::createPointFilter(
        BIGINT(), variant::create<TypeKind::BIGINT>(42));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testInt64(42));
    EXPECT_FALSE(filter->testInt64(41));
    EXPECT_FALSE(filter->testInt64(43));
    EXPECT_FALSE(filter->testNull());
  }

  // Test INTEGER point filter.
  {
    auto filter = hive::createPointFilter(INTEGER(), variant(100));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testInt64(100));
    EXPECT_FALSE(filter->testInt64(99));
    EXPECT_FALSE(filter->testNull());
  }

  // Test DOUBLE point filter.
  {
    auto filter = hive::createPointFilter(DOUBLE(), variant(3.14));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testDouble(3.14));
    EXPECT_FALSE(filter->testDouble(3.15));
    EXPECT_FALSE(filter->testNull());
  }

  // Test VARCHAR point filter.
  {
    auto filter = hive::createPointFilter(VARCHAR(), variant("hello"));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testBytes("hello", 5));
    EXPECT_FALSE(filter->testBytes("world", 5));
    EXPECT_FALSE(filter->testNull());
  }

  // Test BOOLEAN point filter.
  {
    auto filter = hive::createPointFilter(BOOLEAN(), variant(true));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testBool(true));
    EXPECT_FALSE(filter->testBool(false));
    EXPECT_FALSE(filter->testNull());
  }

  // Test null value throws.
  {
    VELOX_ASSERT_THROW(
        hive::createPointFilter(BIGINT(), variant::null(TypeKind::BIGINT)),
        "Value cannot be null");
  }

  // Test unsupported types throw.
  for (const auto& unsupported : kUnsupportedFilterTypes) {
    SCOPED_TRACE(
        fmt::format("Unsupported type: {}", unsupported.type->toString()));
    VELOX_ASSERT_THROW(
        hive::createPointFilter(unsupported.type, unsupported.value), "");
  }
}

TEST_F(HiveConnectorUtilTest, createRangeFilter) {
  // Test BIGINT range filter.
  {
    auto filter = hive::createRangeFilter(
        BIGINT(),
        variant::create<TypeKind::BIGINT>(10),
        variant::create<TypeKind::BIGINT>(20));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testInt64(10));
    EXPECT_TRUE(filter->testInt64(15));
    EXPECT_TRUE(filter->testInt64(20));
    EXPECT_FALSE(filter->testInt64(9));
    EXPECT_FALSE(filter->testInt64(21));
    EXPECT_FALSE(filter->testNull());
  }

  // Test INTEGER range filter.
  {
    auto filter = hive::createRangeFilter(INTEGER(), variant(0), variant(100));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testInt64(0));
    EXPECT_TRUE(filter->testInt64(50));
    EXPECT_TRUE(filter->testInt64(100));
    EXPECT_FALSE(filter->testInt64(-1));
    EXPECT_FALSE(filter->testInt64(101));
  }

  // Test DOUBLE range filter.
  {
    auto filter = hive::createRangeFilter(DOUBLE(), variant(1.0), variant(2.0));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testDouble(1.0));
    EXPECT_TRUE(filter->testDouble(1.5));
    EXPECT_TRUE(filter->testDouble(2.0));
    EXPECT_FALSE(filter->testDouble(0.9));
    EXPECT_FALSE(filter->testDouble(2.1));
  }

  // Test VARCHAR range filter.
  {
    auto filter =
        hive::createRangeFilter(VARCHAR(), variant("apple"), variant("banana"));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testBytes("apple", 5));
    EXPECT_TRUE(filter->testBytes("ball", 4));
    EXPECT_TRUE(filter->testBytes("banana", 6));
    EXPECT_FALSE(filter->testBytes("aaa", 3));
    EXPECT_FALSE(filter->testBytes("cherry", 6));
  }

  // Test lower bound only (upper unbounded) - BIGINT.
  {
    auto filter = hive::createRangeFilter(
        BIGINT(),
        variant::create<TypeKind::BIGINT>(10),
        variant::null(TypeKind::BIGINT));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testInt64(10));
    EXPECT_TRUE(filter->testInt64(100));
    EXPECT_TRUE(filter->testInt64(std::numeric_limits<int64_t>::max()));
    EXPECT_FALSE(filter->testInt64(9));
    EXPECT_FALSE(filter->testNull());
  }

  // Test upper bound only (lower unbounded) - BIGINT.
  {
    auto filter = hive::createRangeFilter(
        BIGINT(),
        variant::null(TypeKind::BIGINT),
        variant::create<TypeKind::BIGINT>(20));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testInt64(20));
    EXPECT_TRUE(filter->testInt64(0));
    EXPECT_TRUE(filter->testInt64(std::numeric_limits<int64_t>::min()));
    EXPECT_FALSE(filter->testInt64(21));
    EXPECT_FALSE(filter->testNull());
  }

  // Test lower bound only - DOUBLE.
  {
    auto filter = hive::createRangeFilter(
        DOUBLE(), variant(1.5), variant::null(TypeKind::DOUBLE));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testDouble(1.5));
    EXPECT_TRUE(filter->testDouble(100.0));
    EXPECT_FALSE(filter->testDouble(1.0));
  }

  // Test upper bound only - VARCHAR.
  {
    auto filter = hive::createRangeFilter(
        VARCHAR(), variant::null(TypeKind::VARCHAR), variant("banana"));
    ASSERT_NE(filter, nullptr);
    EXPECT_TRUE(filter->testBytes("banana", 6));
    EXPECT_TRUE(filter->testBytes("aaa", 3));
    EXPECT_FALSE(filter->testBytes("cherry", 6));
  }

  // Test both bounds null throws.
  {
    VELOX_ASSERT_THROW(
        hive::createRangeFilter(
            BIGINT(),
            variant::null(TypeKind::BIGINT),
            variant::null(TypeKind::BIGINT)),
        "At least one of lower or upper bound must be set");
  }

  // Test unsupported types throw.
  for (const auto& unsupported : kUnsupportedFilterTypes) {
    SCOPED_TRACE(
        fmt::format("Unsupported type: {}", unsupported.type->toString()));
    VELOX_ASSERT_THROW(
        hive::createRangeFilter(
            unsupported.type, unsupported.value, unsupported.value),
        "");
  }
}

} // namespace facebook::velox::connector
