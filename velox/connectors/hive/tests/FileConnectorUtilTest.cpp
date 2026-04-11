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

#include "velox/connectors/hive/FileConnectorUtil.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/type/Filter.h"

namespace facebook::velox::connector {

class FileConnectorUtilTest : public exec::test::HiveConnectorTestBase {
 protected:
  struct QueryCtxHolder {
    std::shared_ptr<config::ConfigBase> sessionProperties;
    std::unique_ptr<ConnectorQueryCtx> ctx;
  };

  QueryCtxHolder makeConnectorQueryCtx(
      std::unordered_map<std::string, std::string> sessionProps = {}) {
    QueryCtxHolder holder;
    holder.sessionProperties =
        std::make_shared<config::ConfigBase>(std::move(sessionProps), true);
    holder.ctx = std::make_unique<ConnectorQueryCtx>(
        pool_.get(),
        pool_.get(),
        holder.sessionProperties.get(),
        nullptr,
        common::PrefixSortConfig(),
        nullptr,
        nullptr,
        "query.FileConnectorUtilTest",
        "task.FileConnectorUtilTest",
        "planNodeId.FileConnectorUtilTest",
        0,
        "");
    return holder;
  }

  std::shared_ptr<const hive::FileConfig> makeFileConfig(
      std::unordered_map<std::string, std::string> props = {}) {
    return std::make_shared<hive::FileConfig>(
        std::make_shared<config::ConfigBase>(std::move(props)));
  }

  std::shared_ptr<const hive::FileConnectorSplit> makeSplit(
      dwio::common::FileFormat format = dwio::common::FileFormat::DWRF,
      const std::string& path = "/tmp/testfile") {
    return std::make_shared<hive::FileConnectorSplit>(
        "testConnectorId", path, format);
  }

  std::string writeDataFile(const RowVectorPtr& data) {
    auto path = exec::test::TempFilePath::create();
    auto filePath = path->getPath();
    tempPaths_.push_back(std::move(path));
    writeToFile(filePath, {data});
    return filePath;
  }

  std::unique_ptr<dwio::common::Reader> makeReader(const std::string& path) {
    dwio::common::ReaderOptions readerOpts{pool_.get()};
    readerOpts.setFileFormat(dwio::common::FileFormat::DWRF);
    auto readFile = std::make_shared<LocalReadFile>(path);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::move(readFile), readerOpts.memoryPool());
    return dwrf::DwrfReader::create(std::move(input), readerOpts);
  }

 private:
  std::vector<std::shared_ptr<exec::test::TempFilePath>> tempPaths_;
};

TEST_F(FileConnectorUtilTest, configureReaderOptions) {
  auto fileConfig = makeFileConfig();

  // Test with DWRF format.
  {
    auto holder = makeConnectorQueryCtx();
    auto split = makeSplit(dwio::common::FileFormat::DWRF);
    dwio::common::ReaderOptions readerOptions(pool_.get());
    hive::configureReaderOptions(
        fileConfig,
        holder.ctx.get(),
        /*fileSchema=*/nullptr,
        split,
        /*tableParameters=*/{},
        readerOptions);

    EXPECT_EQ(readerOptions.fileFormat(), dwio::common::FileFormat::DWRF);
    EXPECT_FALSE(readerOptions.fileColumnNamesReadAsLowerCase());
    EXPECT_FALSE(readerOptions.useColumnNamesForColumnMapping());
  }

  // Test with ORC format and useColumnNames enabled via session.
  {
    auto holder = makeConnectorQueryCtx(
        {{hive::FileConfig::kOrcUseColumnNamesSession, "true"}});
    auto split = makeSplit(dwio::common::FileFormat::ORC);
    dwio::common::ReaderOptions readerOptions(pool_.get());
    hive::configureReaderOptions(
        fileConfig,
        holder.ctx.get(),
        /*fileSchema=*/nullptr,
        split,
        /*tableParameters=*/{},
        readerOptions);

    EXPECT_EQ(readerOptions.fileFormat(), dwio::common::FileFormat::ORC);
    EXPECT_TRUE(readerOptions.useColumnNamesForColumnMapping());
  }

  // Test with Parquet format and useColumnNames enabled via session.
  {
    auto holder = makeConnectorQueryCtx(
        {{hive::FileConfig::kParquetUseColumnNamesSession, "true"}});
    auto split = makeSplit(dwio::common::FileFormat::PARQUET);
    dwio::common::ReaderOptions readerOptions(pool_.get());
    hive::configureReaderOptions(
        fileConfig,
        holder.ctx.get(),
        /*fileSchema=*/nullptr,
        split,
        /*tableParameters=*/{},
        readerOptions);

    EXPECT_EQ(readerOptions.fileFormat(), dwio::common::FileFormat::PARQUET);
    EXPECT_TRUE(readerOptions.useColumnNamesForColumnMapping());
  }

  // Test format mismatch throws.
  {
    auto holder = makeConnectorQueryCtx();
    auto split = makeSplit(dwio::common::FileFormat::DWRF);
    dwio::common::ReaderOptions readerOptions(pool_.get());
    readerOptions.setFileFormat(dwio::common::FileFormat::PARQUET);
    VELOX_ASSERT_THROW(
        hive::configureReaderOptions(
            fileConfig,
            holder.ctx.get(),
            /*fileSchema=*/nullptr,
            split,
            /*tableParameters=*/{},
            readerOptions),
        "received splits of different formats");
  }
}

TEST_F(FileConnectorUtilTest, configureRowReaderOptions) {
  auto holder = makeConnectorQueryCtx();
  auto fileConfig = makeFileConfig();
  auto split = makeSplit(dwio::common::FileFormat::DWRF);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  auto rowType = ROW({"c0"}, {BIGINT()});

  dwio::common::RowReaderOptions rowReaderOptions;
  hive::configureRowReaderOptions(
      /*tableParameters=*/{},
      scanSpec,
      /*metadataFilter=*/nullptr,
      rowType,
      split,
      fileConfig,
      holder.ctx->sessionProperties(),
      /*ioExecutor=*/nullptr,
      rowReaderOptions);

  EXPECT_EQ(rowReaderOptions.scanSpec(), scanSpec);
  EXPECT_EQ(rowReaderOptions.offset(), 0);
  EXPECT_EQ(rowReaderOptions.length(), std::numeric_limits<uint64_t>::max());
}

TEST_F(FileConnectorUtilTest, configureRowReaderOptionsSkipRows) {
  auto holder = makeConnectorQueryCtx();
  auto fileConfig = makeFileConfig();
  auto split = makeSplit(dwio::common::FileFormat::DWRF);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  auto rowType = ROW({"c0"}, {BIGINT()});

  std::unordered_map<std::string, std::string> tableParameters = {
      {dwio::common::TableParameter::kSkipHeaderLineCount, "5"},
  };

  dwio::common::RowReaderOptions rowReaderOptions;
  hive::configureRowReaderOptions(
      tableParameters,
      scanSpec,
      /*metadataFilter=*/nullptr,
      rowType,
      split,
      fileConfig,
      holder.ctx->sessionProperties(),
      /*ioExecutor=*/nullptr,
      rowReaderOptions);

  EXPECT_EQ(rowReaderOptions.skipRows(), 5);
}

TEST_F(FileConnectorUtilTest, configureRowReaderOptionsSplitRange) {
  auto holder = makeConnectorQueryCtx();
  auto fileConfig = makeFileConfig();
  auto split = std::make_shared<hive::FileConnectorSplit>(
      "testConnectorId",
      "/tmp/testfile",
      dwio::common::FileFormat::DWRF,
      /*start=*/100,
      /*length=*/5000);
  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  auto rowType = ROW({"c0"}, {BIGINT()});

  dwio::common::RowReaderOptions rowReaderOptions;
  hive::configureRowReaderOptions(
      /*tableParameters=*/{},
      scanSpec,
      /*metadataFilter=*/nullptr,
      rowType,
      split,
      fileConfig,
      holder.ctx->sessionProperties(),
      /*ioExecutor=*/nullptr,
      rowReaderOptions);

  EXPECT_EQ(rowReaderOptions.offset(), 100);
  EXPECT_EQ(rowReaderOptions.length(), 5000);
}

TEST_F(FileConnectorUtilTest, testFiltersNoFilters) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto filePath = writeDataFile(batch);
  auto reader = makeReader(filePath);

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField("c0", 0);

  EXPECT_TRUE(
      hive::testFilters(
          scanSpec.get(),
          reader.get(),
          filePath,
          /*partitionKey=*/{},
          /*partitionKeysHandle=*/{},
          /*asLocalTime=*/false));
}

TEST_F(FileConnectorUtilTest, testFiltersPartitionKeyPasses) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto filePath = writeDataFile(batch);
  auto reader = makeReader(filePath);

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField("c0", 0);
  auto* dsSpec = scanSpec->addField("ds", 1);
  dsSpec->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"2024-01-01"}, false));

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"ds", "2024-01-01"},
  };

  auto dsHandle = std::make_shared<hive::HiveColumnHandle>(
      "ds",
      hive::HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR());
  std::unordered_map<std::string, hive::FileColumnHandlePtr>
      partitionKeysHandle = {
          {"ds", dsHandle},
      };

  EXPECT_TRUE(
      hive::testFilters(
          scanSpec.get(),
          reader.get(),
          filePath,
          partitionKeys,
          partitionKeysHandle,
          /*asLocalTime=*/false));
}

TEST_F(FileConnectorUtilTest, testFiltersPartitionKeyFails) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto filePath = writeDataFile(batch);
  auto reader = makeReader(filePath);

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField("c0", 0);
  auto* dsSpec = scanSpec->addField("ds", 1);
  dsSpec->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"2024-01-01"}, false));

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"ds", "2024-02-15"},
  };

  auto dsHandle = std::make_shared<hive::HiveColumnHandle>(
      "ds",
      hive::HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR());
  std::unordered_map<std::string, hive::FileColumnHandlePtr>
      partitionKeysHandle = {
          {"ds", dsHandle},
      };

  EXPECT_FALSE(
      hive::testFilters(
          scanSpec.get(),
          reader.get(),
          filePath,
          partitionKeys,
          partitionKeysHandle,
          /*asLocalTime=*/false));
}

TEST_F(FileConnectorUtilTest, testFiltersNullPartitionKeyRejectsNotNull) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto filePath = writeDataFile(batch);
  auto reader = makeReader(filePath);

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField("c0", 0);
  auto* dsSpec = scanSpec->addField("ds", 1);
  dsSpec->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"2024-01-01"}, false));

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"ds", std::nullopt},
  };

  auto dsHandle = std::make_shared<hive::HiveColumnHandle>(
      "ds",
      hive::HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR());
  std::unordered_map<std::string, hive::FileColumnHandlePtr>
      partitionKeysHandle = {
          {"ds", dsHandle},
      };

  EXPECT_FALSE(
      hive::testFilters(
          scanSpec.get(),
          reader.get(),
          filePath,
          partitionKeys,
          partitionKeysHandle,
          /*asLocalTime=*/false));
}

TEST_F(FileConnectorUtilTest, testFiltersIntegerPartitionKey) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto filePath = writeDataFile(batch);
  auto reader = makeReader(filePath);

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField("c0", 0);
  auto* yearSpec = scanSpec->addField("year", 1);
  yearSpec->setFilter(std::make_unique<common::BigintRange>(2024, 2024, false));

  // Matching partition value.
  {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys =
        {{"year", "2024"}};
    auto yearHandle = std::make_shared<hive::HiveColumnHandle>(
        "year",
        hive::HiveColumnHandle::ColumnType::kPartitionKey,
        BIGINT(),
        BIGINT());
    std::unordered_map<std::string, hive::FileColumnHandlePtr>
        partitionKeysHandle = {{"year", yearHandle}};

    EXPECT_TRUE(
        hive::testFilters(
            scanSpec.get(),
            reader.get(),
            filePath,
            partitionKeys,
            partitionKeysHandle,
            /*asLocalTime=*/false));
  }

  // Non-matching partition value.
  {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys =
        {{"year", "2023"}};
    auto yearHandle = std::make_shared<hive::HiveColumnHandle>(
        "year",
        hive::HiveColumnHandle::ColumnType::kPartitionKey,
        BIGINT(),
        BIGINT());
    std::unordered_map<std::string, hive::FileColumnHandlePtr>
        partitionKeysHandle = {{"year", yearHandle}};

    EXPECT_FALSE(
        hive::testFilters(
            scanSpec.get(),
            reader.get(),
            filePath,
            partitionKeys,
            partitionKeysHandle,
            /*asLocalTime=*/false));
  }
}

TEST_F(FileConnectorUtilTest, testFiltersMissingColumn) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto filePath = writeDataFile(batch);
  auto reader = makeReader(filePath);

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addField("c0", 0);
  // Filter on a column that doesn't exist in the file and is not a partition
  // key. This simulates schema evolution where the column was added later.
  auto* newColSpec = scanSpec->addField("newCol", 1);
  // Filter that rejects null -- should cause the split to be skipped since
  // the column is missing (will be all nulls).
  newColSpec->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"someValue"}, false));

  EXPECT_FALSE(
      hive::testFilters(
          scanSpec.get(),
          reader.get(),
          filePath,
          /*partitionKey=*/{},
          /*partitionKeysHandle=*/{},
          /*asLocalTime=*/false));
}

} // namespace facebook::velox::connector
