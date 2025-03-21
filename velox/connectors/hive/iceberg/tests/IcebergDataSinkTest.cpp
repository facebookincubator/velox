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

#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include <gtest/gtest.h>
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <folly/init/Init.h>
#include "velox/common/base/Fs.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/writer/Writer.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#endif

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using namespace facebook::velox::common;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

class IcebergDataSinkTest : public exec::test::HiveConnectorTestBase,
                            public testing::WithParamInterface<std::tuple<
                                dwio::common::FileFormat,
                                std::optional<std::vector<std::string>>,
                                size_t>> {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
#endif
    Type::registerSerDe();

    rowType_ =
        ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
            {BIGINT(),
             INTEGER(),
             SMALLINT(),
             REAL(),
             DOUBLE(),
             VARCHAR(),
             BOOLEAN()});

    setupMemoryPools();
  }

  void TearDown() override {
    connectorQueryCtx_.reset();
    connectorPool_.reset();
    opPool_.reset();
    root_.reset();
    HiveConnectorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> createVectors(int vectorSize, int numVectors) {
    VectorFuzzer::Options options;
    options.vectorSize = vectorSize;
    VectorFuzzer fuzzer(options, pool());
    std::vector<RowVectorPtr> vectors;
    for (int i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    }
    return vectors;
  }

  void setupMemoryPools() {
    connectorQueryCtx_.reset();
    connectorPool_.reset();
    opPool_.reset();
    root_.reset();

    root_ = memory::memoryManager()->addRootPool(
        "IcebergDataSinkTest", 1L << 30, exec::MemoryReclaimer::create());
    opPool_ = root_->addLeafChild("operator");
    connectorPool_ =
        root_->addAggregateChild("connector", exec::MemoryReclaimer::create());

    connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
        opPool_.get(),
        connectorPool_.get(),
        connectorSessionProperties_.get(),
        nullptr,
        common::PrefixSortConfig(),
        nullptr,
        nullptr,
        "query.IcebergDataSinkTest",
        "task.IcebergDataSinkTest",
        "planNodeId.IcebergDataSinkTest",
        0,
        "");
  }

  std::shared_ptr<connector::hive::iceberg::IcebergInsertTableHandle>
  createIcebergInsertTableHandle(
      const RowTypePtr& outputRowType,
      const std::string& outputDirectoryPath,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF,
      const std::vector<std::string>& partitionedBy = {}) {
    std::vector<std::shared_ptr<const connector::hive::HiveColumnHandle>>
        columnHandles;
    int32_t numPartitionColumns{0};

    std::vector<std::string> columnNames = outputRowType->names();
    std::vector<TypePtr> columnTypes = outputRowType->children();

    for (int i = 0; i < columnNames.size(); ++i) {
      if (std::find(
              partitionedBy.cbegin(),
              partitionedBy.cend(),
              columnNames.at(i)) != partitionedBy.cend()) {
        ++numPartitionColumns;
        columnHandles.push_back(
            std::make_shared<connector::hive::HiveColumnHandle>(
                columnNames.at(i),
                connector::hive::HiveColumnHandle::ColumnType::kPartitionKey,
                columnTypes.at(i),
                columnTypes.at(i)));
      } else {
        columnHandles.push_back(
            std::make_shared<connector::hive::HiveColumnHandle>(
                columnNames.at(i),
                connector::hive::HiveColumnHandle::ColumnType::kRegular,
                columnTypes.at(i),
                columnTypes.at(i)));
      }
    }

    VELOX_CHECK_EQ(numPartitionColumns, partitionedBy.size());

    std::shared_ptr<const connector::hive::LocationHandle> locationHandle =
        makeLocationHandle(
            outputDirectoryPath,
            std::nullopt,
            connector::hive::LocationHandle::TableType::kNew);

    std::vector<std::shared_ptr<const VeloxIcebergNestedField>> columns;
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 0, "c0", BIGINT(), nullptr));
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 1, "c1", INTEGER(), nullptr));
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 2, "c2", SMALLINT(), nullptr));
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 3, "c3", REAL(), nullptr));
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 4, "c4", DOUBLE(), nullptr));
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 5, "c5", VARCHAR(), nullptr));
    columns.emplace_back(
        new VeloxIcebergNestedField(false, 6, "c6", BOOLEAN(), nullptr));

    std::shared_ptr<const VeloxIcebergSchema> schema =
        std::make_shared<VeloxIcebergSchema>(
            0,
            columns,
            std::unordered_map<std::string, std::int32_t>(),
            std::unordered_map<std::string, std::int32_t>(),
            std::vector<int32_t>());

    std::vector<std::string> fields;
    fields.reserve(partitionedBy.size());
    for (const auto& partition : partitionedBy) {
      fields.push_back(partition);
    }

    std::shared_ptr<const VeloxIcebergPartitionSpec> partitionSpec =
        std::make_shared<VeloxIcebergPartitionSpec>(0, schema, fields);

    return std::make_shared<connector::hive::iceberg::IcebergInsertTableHandle>(
        columnHandles,
        locationHandle,
        schema,
        partitionSpec,
        fileFormat,
        nullptr,
        CompressionKind::CompressionKind_ZSTD);
  }

  std::shared_ptr<IcebergDataSink> createIcebergDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF,
      const std::vector<std::string>& partitionedBy = {}) {
    return std::make_shared<IcebergDataSink>(
        rowType,
        createIcebergInsertTableHandle(
            rowType, outputDirectoryPath, fileFormat, partitionedBy),
        connectorQueryCtx_.get(),
        CommitStrategy::kNoCommit,
        connectorConfig_);
  }

  std::vector<std::string> listFiles(const std::string& dirPath) {
    std::vector<std::string> files;
    for (auto& dirEntry : fs::recursive_directory_iterator(dirPath)) {
      if (dirEntry.is_regular_file()) {
        files.push_back(dirEntry.path().string());
      }
    }
    return files;
  }

  std::shared_ptr<HiveIcebergSplit> makeIcebergConnectorSplit(
      const std::string& filePath,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF) {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
    std::unordered_map<std::string, std::string> customSplitInfo;
    customSplitInfo["table_format"] = "hive-iceberg";

    auto file = filesystems::getFileSystem(filePath, nullptr)
                    ->openFileForRead(filePath);
    const int64_t fileSize = file->size();

    return std::make_shared<HiveIcebergSplit>(
        kHiveConnectorId,
        filePath,
        fileFormat,
        0,
        fileSize,
        partitionKeys,
        std::nullopt,
        customSplitInfo,
        nullptr,
        /*cacheable=*/true,
        std::vector<IcebergDeleteFile>());
  }

  void verifyWrittenData(
      const std::string& dirPath,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF,
      int32_t numFiles = 1) {
    const std::vector<std::string> filePaths = listFiles(dirPath);
    ASSERT_EQ(filePaths.size(), numFiles);
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    std::for_each(filePaths.begin(), filePaths.end(), [&](auto filePath) {
      splits.push_back(makeIcebergConnectorSplit(filePath, fileFormat));
    });
    HiveConnectorTestBase::assertQuery(
        PlanBuilder().tableScan(rowType_).planNode(),
        splits,
        fmt::format("SELECT * FROM tmp"));
  }

  const std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();

  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  RowTypePtr rowType_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_ =
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>(),
          /*mutable=*/true);
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  std::shared_ptr<HiveConfig> connectorConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
};

TEST_P(IcebergDataSinkTest, testIcebergTableWrite) {
  const auto& [format, partitionKeys, expectedCommitTasks] = GetParam();
  const auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createIcebergDataSink(
      rowType_,
      outputDirectory->getPath(),
      format,
      partitionKeys.value_or(std::vector<std::string>()));

  auto stats = dataSink->stats();
  ASSERT_TRUE(stats.empty());

  const int numBatches = 10;
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  stats = dataSink->stats();
  if (format == dwio::common::FileFormat::DWRF) {
    ASSERT_FALSE(stats.empty());
    ASSERT_GT(stats.numWrittenBytes, 0);
    ASSERT_EQ(stats.numWrittenFiles, 0);
  }
  ASSERT_TRUE(dataSink->finish());

  const auto commitTasks = dataSink->close();
  stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_EQ(commitTasks.size(), expectedCommitTasks);

  createDuckDbTable(vectors);
  verifyWrittenData(
      fmt::format("{}/data", outputDirectory->getPath()),
      format,
      commitTasks.size());

  if (format == dwio::common::FileFormat::PARQUET) {
    dwio::common::ReaderOptions readerOpts{pool_.get()};
    const std::vector<std::string> filePaths =
        listFiles(outputDirectory->getPath());
    auto bufferedInput = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePaths[0]), readerOpts.memoryPool());

    auto reader =
        dwio::common::getReaderFactory(dwio::common::FileFormat::PARQUET)
            ->createReader(std::move(bufferedInput), readerOpts);
    auto parquetReader = dynamic_cast<parquet::ParquetReader&>(*reader.get());

    auto fileMeta = parquetReader.fileMetaData();
    EXPECT_EQ(fileMeta.numRowGroups(), 1);

    auto firstPartition = fileMeta.rowGroup(0).numRows();

    if (filePaths.size() > 1) {
      bufferedInput = std::make_unique<dwio::common::BufferedInput>(
          std::make_shared<LocalReadFile>(filePaths[1]),
          readerOpts.memoryPool());

      reader = dwio::common::getReaderFactory(dwio::common::FileFormat::PARQUET)
                   ->createReader(std::move(bufferedInput), readerOpts);
      parquetReader = dynamic_cast<parquet::ParquetReader&>(*reader.get());

      fileMeta = parquetReader.fileMetaData();
      EXPECT_EQ(fileMeta.numRowGroups(), 1);

      auto secondPartition = fileMeta.rowGroup(0).numRows();

      EXPECT_EQ(firstPartition + secondPartition, 5000);
    } else {
      EXPECT_EQ(firstPartition, 5000);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    IcebergDataSinkTests,
    IcebergDataSinkTest,
    testing::Values(
        std::make_tuple(dwio::common::FileFormat::DWRF, std::nullopt, 1),
        std::make_tuple(
            dwio::common::FileFormat::DWRF,
            std::make_optional(std::vector<std::string>{"c6"}),
            2)
#ifdef VELOX_ENABLE_PARQUET
            ,
        std::make_tuple(dwio::common::FileFormat::PARQUET, std::nullopt, 1),
        std::make_tuple(
            dwio::common::FileFormat::PARQUET,
            std::make_optional(std::vector<std::string>{"c6"}),
            2)
#endif
            ));

} // namespace
} // namespace facebook::velox::connector::hive::iceberg

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
