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

#include "velox/connectors/hive/iceberg/IcebergChangelogSplitReader.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/IcebergTableHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

class IcebergChangelogSplitReaderTest : public test::IcebergTestBase {
 protected:
  std::unique_ptr<IcebergChangelogSplitReader> makeChangelogSplitReader(
      const std::shared_ptr<HiveIcebergSplit>& split,
      const RowTypePtr& dataType,
      const RowTypePtr& changelogOutputType,
      ColumnHandleMap changelogColumnHandles,
      const common::SubfieldFilters* changelogFilters = nullptr) {
    auto sessionProperties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>());
    auto hiveConfig = std::make_shared<HiveConfig>(sessionProperties);
    fileHandleFactory_ = std::make_unique<FileHandleFactory>(
        std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(
            hiveConfig->numCacheFileHandles()),
        std::make_unique<FileHandleGenerator>(sessionProperties));

    ChangelogScanContext scanContext{
        std::make_shared<ColumnHandleMap>(makeColumnHandles(dataType)),
        dataType,
        makeScanSpec(
            dataType,
            /*outputSubfields=*/{},
            /*filters=*/{},
            dataType,
            /*partitionKeys=*/{},
            /*infoColumns=*/{},
            SpecialColumnNames{},
            /*reorderFiltersDisabled=*/false,
            pool())};
    return std::make_unique<IcebergChangelogSplitReader>(
        split,
        makeChangelogTableHandle(dataType),
        /*partitionKeys=*/nullptr,
        connectorQueryCtx_.get(),
        hiveConfig,
        scanContext,
        ioStats_,
        metadataIoStats_,
        connectorIoStats_,
        fileHandleFactory_.get(),
        /*executor=*/nullptr,
        changelogOutputType,
        std::move(changelogColumnHandles),
        changelogFilters);
  }

  /// Prepares 'reader' and drains all batches into a single RowVector.
  /// Returns nullptr if the split is empty (reader.emptySplit() == true).
  RowVectorPtr readAll(IcebergChangelogSplitReader& reader) {
    dwio::common::RuntimeStats runtimeStats;
    std::shared_ptr<random::RandomSkipTracker> randomSkip;
    reader.configureReaderOptions(randomSkip);
    reader.prepareSplit(nullptr, runtimeStats);
    if (reader.emptySplit()) {
      return nullptr;
    }

    std::vector<RowVectorPtr> batches;
    VectorPtr output;
    while (reader.next(1024, output) > 0) {
      batches.push_back(std::dynamic_pointer_cast<RowVector>(output));
    }

    if (batches.empty()) {
      return nullptr;
    }

    auto result = std::dynamic_pointer_cast<RowVector>(
        BaseVector::create(batches[0]->type(), 0, pool()));
    for (const auto& batch : batches) {
      result->append(batch.get());
    }
    return result;
  }

  /// Writes the standard test batches to a temp directory and returns the
  /// sole file path.
  std::string writeTestFile() {
    auto batches = makeTestBatches();
    outputDirectory_ = test::TempDirectoryPath::create();
    auto sink =
        createDataSinkAndAppendData(batches, outputDirectory_->getPath(), {});
    sink->close();
    return getOnlyDataFilePath(outputDirectory_->getPath());
  }

  std::shared_ptr<io::IoStatistics> ioStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<IoStats> connectorIoStats_{std::make_shared<IoStats>()};
  std::unique_ptr<FileHandleFactory> fileHandleFactory_;
  std::shared_ptr<test::TempDirectoryPath> outputDirectory_;
};

// Writes a file with two batches (200 rows) and verifies that every
// ChangelogOperation value produces the correct constant metadata columns
// alongside the original data rows.
TEST_F(IcebergChangelogSplitReaderTest, changelogSplitData) {
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});
  const std::vector<
      std::tuple<ChangelogOperation, std::string, int64_t, int64_t>>
      cases = {
          {ChangelogOperation::kInsert, "INSERT", 1, 1},
          {ChangelogOperation::kDelete, "DELETE", 1, 1},
          {ChangelogOperation::kUpdateBefore, "UPDATE_BEFORE", 1, 1},
          {ChangelogOperation::kUpdateAfter, "UPDATE_AFTER", 7, 12'345},
      };

  const auto filePath = writeTestFile();

  auto batches = makeTestBatches();
  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }

  for (
      const auto& [operationValue, expectedOperation, ordinalValue, snapshotIdValue] :
      cases) {
    auto reader = makeChangelogSplitReader(
        makeChangelogSplit(
            filePath, operationValue, ordinalValue, snapshotIdValue),
        dataType,
        makeChangelogOutputType(dataType),
        makeChangelogColumnHandles(dataType));

    auto result = readAll(*reader);
    ASSERT_NE(result, nullptr) << "op=" << expectedOperation;
    ASSERT_EQ(result->size(), expectedRows) << "op=" << expectedOperation;

    auto operation = result->childAt(0)->as<SimpleVector<StringView>>();
    auto ordinal = result->childAt(1)->as<SimpleVector<int64_t>>();
    auto snapshotId = result->childAt(2)->as<SimpleVector<int64_t>>();
    auto rowdata = result->childAt(3)->as<RowVector>();
    auto ids = rowdata->childAt(0)->as<SimpleVector<int64_t>>();
    auto names = rowdata->childAt(1)->as<SimpleVector<StringView>>();

    for (int32_t row = 0; row < result->size(); ++row) {
      ASSERT_EQ(operation->valueAt(row), StringView(expectedOperation));
      ASSERT_EQ(ordinal->valueAt(row), ordinalValue);
      ASSERT_EQ(snapshotId->valueAt(row), snapshotIdValue);
      ASSERT_EQ(ids->valueAt(row), row);
      const std::string expectedName = "name_" + std::to_string(row);
      ASSERT_EQ(names->valueAt(row), StringView(expectedName));
    }
  }
}

// Verifies that readerOutputType() returns the changelog schema
// (operation/ordinal/snapshotid/rowdata) and that the rowdata child carries
// the original data columns.
TEST_F(IcebergChangelogSplitReaderTest, readerOutputType) {
  auto dataType = ROW({{"x", BIGINT()}, {"y", INTEGER()}, {"z", VARCHAR()}});
  auto changelogOutputType = makeChangelogOutputType(dataType);
  auto batch = makeRowVector(
      {"x", "y", "z"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<std::string>({"a", "b", "c"})});

  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData({batch}, outputDirectory->getPath(), {});
  sink->close();

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(
          getOnlyDataFilePath(outputDirectory->getPath()),
          ChangelogOperation::kInsert,
          1,
          100),
      dataType,
      changelogOutputType,
      makeChangelogColumnHandles(dataType));

  ASSERT_EQ(*reader->readerOutputType(), *changelogOutputType);

  auto result = readAll(*reader);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 3);

  auto rowdata = result->childAt(3)->as<RowVector>();
  ASSERT_NE(rowdata, nullptr);
  ASSERT_EQ(rowdata->childrenSize(), 3);
  const auto& rowdataType =
      *std::dynamic_pointer_cast<const RowType>(rowdata->type());
  ASSERT_EQ(rowdataType.nameOf(0), "x");
  ASSERT_EQ(rowdataType.nameOf(1), "y");
  ASSERT_EQ(rowdataType.nameOf(2), "z");
}

// Constant-column filters that reject the split: snapshotid, operation,
// ordinal — each tested separately.
TEST_F(IcebergChangelogSplitReaderTest, filterRejectsEmpty) {
  const auto filePath = writeTestFile();
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});

  struct TestCase {
    std::string name;
    common::SubfieldFilters filters;
  };

  {
    // snapshotid rejects: split has snapshotid=100, filter requires 999.
    common::SubfieldFilters filters;
    filters[common::Subfield(std::string(kChangelogColSnapshotId))] =
        std::make_shared<common::BigintRange>(999, 999, false);
    auto reader = makeChangelogSplitReader(
        makeChangelogSplit(filePath, ChangelogOperation::kInsert, 1, 100),
        dataType,
        makeChangelogOutputType(dataType),
        makeChangelogColumnHandles(dataType),
        &filters);
    dwio::common::RuntimeStats runtimeStats;
    std::shared_ptr<random::RandomSkipTracker> randomSkip;
    reader->configureReaderOptions(randomSkip);
    reader->prepareSplit(nullptr, runtimeStats);
    ASSERT_TRUE(reader->emptySplit()) << "snapshotid reject";
  }

  {
    // operation rejects: split is INSERT, filter requires DELETE.
    const std::string_view deleteOp = kChangelogOpDelete;
    common::SubfieldFilters filters;
    filters[common::Subfield(std::string(kChangelogColOperation))] =
        std::make_shared<common::BytesValues>(
            std::vector<std::string>{std::string(deleteOp)}, false);
    auto reader = makeChangelogSplitReader(
        makeChangelogSplit(filePath, ChangelogOperation::kInsert, 1, 100),
        dataType,
        makeChangelogOutputType(dataType),
        makeChangelogColumnHandles(dataType),
        &filters);
    dwio::common::RuntimeStats runtimeStats;
    std::shared_ptr<random::RandomSkipTracker> randomSkip;
    reader->configureReaderOptions(randomSkip);
    reader->prepareSplit(nullptr, runtimeStats);
    ASSERT_TRUE(reader->emptySplit()) << "operation reject";
  }

  {
    // ordinal rejects: split has ordinal=1, filter requires [5, 10].
    common::SubfieldFilters filters;
    filters[common::Subfield(std::string(kChangelogColOrdinal))] =
        std::make_shared<common::BigintRange>(5, 10, false);
    auto reader = makeChangelogSplitReader(
        makeChangelogSplit(filePath, ChangelogOperation::kInsert, 1, 100),
        dataType,
        makeChangelogOutputType(dataType),
        makeChangelogColumnHandles(dataType),
        &filters);
    dwio::common::RuntimeStats runtimeStats;
    std::shared_ptr<random::RandomSkipTracker> randomSkip;
    reader->configureReaderOptions(randomSkip);
    reader->prepareSplit(nullptr, runtimeStats);
    ASSERT_TRUE(reader->emptySplit()) << "ordinal reject";
  }
}

// A snapshotid filter that matches the split constant passes through and all
// rows are returned.
TEST_F(IcebergChangelogSplitReaderTest, snapshotIdFilterAccepts) {
  const auto filePath = writeTestFile();
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});

  common::SubfieldFilters filters;
  filters[common::Subfield(std::string(kChangelogColSnapshotId))] =
      std::make_shared<common::BigintRange>(100, 100, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::kInsert, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  auto batches = makeTestBatches();
  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }

  auto result = readAll(*reader);
  ASSERT_FALSE(reader->emptySplit());
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows);
}

// An operation filter that matches the split constant passes through.
TEST_F(IcebergChangelogSplitReaderTest, operationFilterAccepts) {
  const auto filePath = writeTestFile();
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});

  const std::string_view insertOp = kChangelogOpInsert;
  common::SubfieldFilters filters;
  filters[common::Subfield(std::string(kChangelogColOperation))] =
      std::make_shared<common::BytesValues>(
          std::vector<std::string>{std::string(insertOp)}, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::kInsert, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  auto result = readAll(*reader);
  ASSERT_FALSE(reader->emptySplit());
  ASSERT_NE(result, nullptr);
  ASSERT_GT(result->size(), 0);
}

// An ordinal filter that matches the split constant passes through.
TEST_F(IcebergChangelogSplitReaderTest, ordinalFilterAccepts) {
  const auto filePath = writeTestFile();
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});

  common::SubfieldFilters filters;
  filters[common::Subfield(std::string(kChangelogColOrdinal))] =
      std::make_shared<common::BigintRange>(1, 5, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::kInsert, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  auto result = readAll(*reader);
  ASSERT_FALSE(reader->emptySplit());
  ASSERT_NE(result, nullptr);
  ASSERT_GT(result->size(), 0);
}

// next() drains all rows across multiple internal batches.
TEST_F(IcebergChangelogSplitReaderTest, nextDrainsAllBatches) {
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});
  const auto filePath = writeTestFile();

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::kDelete, 3, 42),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType));

  auto batches = makeTestBatches();
  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }

  auto result = readAll(*reader);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows);
}

// prepareSplit() throws when changelogSplitInfo is absent from the split.
TEST_F(IcebergChangelogSplitReaderTest, missingChangelogSplitInfoThrows) {
  const auto filePath = writeTestFile();
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});

  // Build a split without changelogSplitInfo.
  auto splits = makeIcebergSplits(filePath);
  ASSERT_EQ(splits.size(), 1);
  auto rawSplit = std::dynamic_pointer_cast<HiveIcebergSplit>(splits.front());
  ASSERT_NE(rawSplit, nullptr);
  ASSERT_FALSE(rawSplit->changelogSplitInfo.has_value());

  auto reader = makeChangelogSplitReader(
      rawSplit,
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType));

  dwio::common::RuntimeStats runtimeStats;
  std::shared_ptr<random::RandomSkipTracker> randomSkip;
  reader->configureReaderOptions(randomSkip);
  VELOX_ASSERT_THROW(
      reader->prepareSplit(nullptr, runtimeStats),
      "HiveIcebergSplit missing changelogSplitInfo for changelog query");
}

// prepareSplit() throws when the split carries delete files.
TEST_F(IcebergChangelogSplitReaderTest, deleteFilesThrows) {
  const auto filePath = writeTestFile();
  auto dataType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});

  // Build a changelog split that also has a (dummy) delete file attached.
  IcebergDeleteFile dummyDelete{
      FileContent::kPositionalDeletes,
      filePath,
      dwio::common::FileFormat::PARQUET,
      /*recordCount=*/0,
      /*fileSizeInBytes=*/0};

  auto split = IcebergSplitBuilder(filePath)
                   .connectorId(test::kIcebergConnectorId)
                   .fileFormat(fileFormat_)
                   .deleteFiles({dummyDelete})
                   .changelogSplitInfo(
                       ChangelogSplitInfo{ChangelogOperation::kInsert, 1, 100})
                   .build();

  auto reader = makeChangelogSplitReader(
      split,
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType));

  dwio::common::RuntimeStats runtimeStats;
  std::shared_ptr<random::RandomSkipTracker> randomSkip;
  reader->configureReaderOptions(randomSkip);
  VELOX_ASSERT_THROW(
      reader->prepareSplit(nullptr, runtimeStats),
      "Changelog splits do not support delete files");
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
