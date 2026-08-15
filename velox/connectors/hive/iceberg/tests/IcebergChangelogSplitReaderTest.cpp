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
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

class IcebergChangelogSplitReaderIntegrationTest
    : public test::IcebergTestBase {
 protected:
  std::vector<RowVectorPtr> makeTestBatches() {
    std::vector<RowVectorPtr> batches;
    for (int32_t batch = 0; batch < 2; ++batch) {
      auto idVector = makeFlatVector<int64_t>(100, [batch](auto row) {
        return static_cast<int64_t>(batch * 100 + row);
      });
      auto nameVector = makeFlatVector<std::string>(100, [batch](auto row) {
        return "name_" + std::to_string(batch * 100 + row);
      });
      batches.push_back(makeRowVector({"id", "name"}, {idVector, nameVector}));
    }
    return batches;
  }

  std::string getOnlyDataFilePath(const std::string& directory) {
    auto files = listFiles(directory);
    VELOX_CHECK_EQ(files.size(), 1);
    return files.front();
  }

  std::shared_ptr<HiveIcebergSplit> makeChangelogSplit(
      const std::string& filePath,
      ChangelogOperation operation,
      int64_t ordinal,
      int64_t snapshotId) {
    auto changelogInfo =
        std::make_shared<ChangelogSplitInfo>(operation, ordinal, snapshotId);
    auto splits = makeIcebergSplits(filePath);
    VELOX_CHECK_EQ(splits.size(), 1);

    auto split = std::dynamic_pointer_cast<HiveIcebergSplit>(splits.front());
    VELOX_CHECK_NOT_NULL(split);
    split->changelogSplitInfo = std::move(changelogInfo);
    return split;
  }

  RowTypePtr makeChangelogOutputType(const RowTypePtr& dataType) {
    return ROW(
        {"operation", "ordinal", "snapshotid", "rowdata"},
        {VARCHAR(), BIGINT(), BIGINT(), dataType});
  }

  ColumnHandleMap makeChangelogColumnHandles(const RowTypePtr& dataType) {
    ColumnHandleMap handles;
    handles["operation"] = std::make_shared<HiveColumnHandle>(
        "operation",
        HiveColumnHandle::ColumnType::kRegular,
        VARCHAR(),
        VARCHAR());
    handles["ordinal"] = std::make_shared<HiveColumnHandle>(
        "ordinal", HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
    handles["snapshotid"] = std::make_shared<HiveColumnHandle>(
        "snapshotid",
        HiveColumnHandle::ColumnType::kRegular,
        BIGINT(),
        BIGINT());
    handles["rowdata"] = std::make_shared<HiveColumnHandle>(
        "rowdata", HiveColumnHandle::ColumnType::kRegular, dataType, dataType);
    return handles;
  }

  std::shared_ptr<HiveTableHandle> makeChangelogTableHandle(
      const RowTypePtr& dataType) {
    return std::make_shared<HiveTableHandle>(
        test::kIcebergConnectorId,
        "test_table",
        common::SubfieldFilters{},
        nullptr,
        dataType,
        std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{},
        std::vector<HiveColumnHandlePtr>{},
        1.0,
        "",
        true,
        makeColumnHandles(dataType));
  }

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

    return std::make_unique<IcebergChangelogSplitReader>(
        split,
        makeChangelogTableHandle(dataType),
        nullptr,
        connectorQueryCtx_.get(),
        hiveConfig,
        dataType,
        ioStats_,
        metadataIoStats_,
        connectorIoStats_,
        fileHandleFactory_.get(),
        nullptr,
        makeScanSpec(
            dataType,
            {},
            {},
            dataType,
            {},
            {},
            SpecialColumnNames{},
            false,
            pool()),
        std::make_shared<ColumnHandleMap>(makeColumnHandles(dataType)),
        changelogOutputType,
        std::move(changelogColumnHandles),
        changelogFilters);
  }

  RowVectorPtr readAll(IcebergChangelogSplitReader& reader) {
    dwio::common::RuntimeStatistics runtimeStats;
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

TEST_F(IcebergChangelogSplitReaderIntegrationTest, testChangelogSplitData) {
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
    ASSERT_EQ(result->size(), 200) << "op=" << expectedOperation;

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

TEST_F(IcebergChangelogSplitReaderIntegrationTest, testReaderOutputType) {
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

  const auto& reported = reader->readerOutputType();
  ASSERT_EQ(reported->size(), 4);
  ASSERT_EQ(reported->nameOf(0), "operation");
  ASSERT_EQ(reported->childAt(0)->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(reported->nameOf(1), "ordinal");
  ASSERT_EQ(reported->childAt(1)->kind(), TypeKind::BIGINT);
  ASSERT_EQ(reported->nameOf(2), "snapshotid");
  ASSERT_EQ(reported->childAt(2)->kind(), TypeKind::BIGINT);
  ASSERT_EQ(reported->nameOf(3), "rowdata");
  ASSERT_EQ(reported->childAt(3)->kind(), TypeKind::ROW);

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

TEST_F(IcebergChangelogSplitReaderIntegrationTest, testFiltersRejects) {
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

  auto result = readAll(*reader);
  ASSERT_EQ(result, nullptr);
}

TEST_F(IcebergChangelogSplitReaderIntegrationTest, testFilterAccepts) {
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

  auto result = readAll(*reader);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 200);
}

TEST_F(IcebergChangelogSplitReaderIntegrationTest, nextDrainsAllBatches) {
  auto batches = makeTestBatches();
  auto outputDirectory = test::TempDirectoryPath::create();
  auto sink =
      createDataSinkAndAppendData(batches, outputDirectory->getPath(), {});
  sink->close();

  auto reader = makeChangelogSplitReader(
      makeChangelogSplit(
          getOnlyDataFilePath(outputDirectory->getPath()),
          ChangelogOperation::DELETE,
          3,
          42),
      ROW({"id", "name"}, {BIGINT(), VARCHAR()}),
      makeChangelogOutputType(ROW({"id", "name"}, {BIGINT(), VARCHAR()})),
      makeChangelogColumnHandles(ROW({"id", "name"}, {BIGINT(), VARCHAR()})));

  auto result = readAll(*reader);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 200);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
