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
            {},
            {},
            dataType,
            {},
            {},
            SpecialColumnNames{},
            false,
            pool())};
    return std::make_unique<IcebergChangelogSplitReader>(
        split,
        makeChangelogTableHandle(dataType),
        nullptr,
        connectorQueryCtx_.get(),
        hiveConfig,
        scanContext,
        ioStats_,
        metadataIoStats_,
        connectorIoStats_,
        fileHandleFactory_.get(),
        nullptr,
        changelogOutputType,
        std::move(changelogColumnHandles),
        changelogFilters);
  }

  /// Prepares 'reader', drains all batches, concatenates them into a single
  /// RowVector, and returns it. Returns nullptr if the split is empty.
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

  std::shared_ptr<io::IoStatistics> ioStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::shared_ptr<IoStats> connectorIoStats_{std::make_shared<IoStats>()};
  std::unique_ptr<FileHandleFactory> fileHandleFactory_;
};

// Writes a file with two batches (200 rows) and verifies that every
// ChangelogOperation value produces the correct constant metadata columns
// alongside the original data rows.
TEST_F(IcebergChangelogSplitReaderTest, changelogSplitData) {
  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  const std::vector<
      std::tuple<ChangelogOperation, std::string, int64_t, int64_t>>
      cases = {
          {ChangelogOperation::INSERT, "INSERT", 1, 1},
          {ChangelogOperation::DELETE, "DELETE", 1, 1},
          {ChangelogOperation::UPDATE_BEFORE, "UPDATE_BEFORE", 1, 1},
          {ChangelogOperation::UPDATE_AFTER, "UPDATE_AFTER", 7, 12345},
      };

  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }

  const auto filePath = getOnlyDataFilePath(outputDirectory->getPath());

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
  auto dataType = ROW({"x", "y", "z"}, {BIGINT(), INTEGER(), VARCHAR()});
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
          ChangelogOperation::INSERT,
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

// A snapshotid filter that does not match the split constant marks the split
// empty, so readAll() returns nullptr.
TEST_F(IcebergChangelogSplitReaderTest, snapshotIdFilterRejects) {
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  const auto filePath = getOnlyDataFilePath(outputDirectory->getPath());
  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

  common::SubfieldFilters filters;
  filters[common::Subfield("snapshotid")] =
      std::make_shared<common::BigintRange>(999, 999, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::INSERT, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  ASSERT_EQ(readAll(*reader), nullptr);
}

// An operation filter that does not match the split's operation marks the
// split empty.
TEST_F(IcebergChangelogSplitReaderTest, operationFilterRejects) {
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  const auto filePath = getOnlyDataFilePath(outputDirectory->getPath());
  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

  // Split has operation=INSERT; filter requires DELETE.
  common::SubfieldFilters filters;
  const std::string_view deleteOp = kChangelogOpDelete;
  filters[common::Subfield(std::string(kChangelogColOperation))] =
      std::make_shared<common::BytesValues>(
          std::vector<std::string>{std::string(deleteOp)}, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::INSERT, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  ASSERT_EQ(readAll(*reader), nullptr);
}

// An ordinal filter that does not match the split's ordinal marks the split
// empty.
TEST_F(IcebergChangelogSplitReaderTest, ordinalFilterRejects) {
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  const auto filePath = getOnlyDataFilePath(outputDirectory->getPath());
  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

  // Split has ordinal=1; filter requires ordinal in [5, 10].
  common::SubfieldFilters filters;
  filters[common::Subfield(std::string(kChangelogColOrdinal))] =
      std::make_shared<common::BigintRange>(5, 10, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::INSERT, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  ASSERT_EQ(readAll(*reader), nullptr);
}

// A snapshotid filter that matches the split constant passes through and all
// rows are returned.
TEST_F(IcebergChangelogSplitReaderTest, snapshotIdFilterAccepts) {
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  const auto filePath = getOnlyDataFilePath(outputDirectory->getPath());
  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

  common::SubfieldFilters filters;
  filters[common::Subfield("snapshotid")] =
      std::make_shared<common::BigintRange>(100, 100, false);

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(filePath, ChangelogOperation::INSERT, 1, 100),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType),
      &filters);

  int32_t expectedRows = 0;
  for (const auto& batch : batches) {
    expectedRows += batch->size();
  }

  auto result = readAll(*reader);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), expectedRows);
}

// next() drains all rows across multiple internal batches.
TEST_F(IcebergChangelogSplitReaderTest, nextDrainsAllBatches) {
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});
  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(
          getOnlyDataFilePath(outputDirectory->getPath()),
          ChangelogOperation::DELETE,
          3,
          42),
      dataType,
      makeChangelogOutputType(dataType),
      makeChangelogColumnHandles(dataType));

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
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  const auto filePath = getOnlyDataFilePath(outputDirectory->getPath());
  auto dataType = ROW({"id", "name"}, {BIGINT(), VARCHAR()});

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

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
