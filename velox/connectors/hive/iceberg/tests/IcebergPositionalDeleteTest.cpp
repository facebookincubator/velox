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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <algorithm>
#include <numeric>

#include <folly/Singleton.h>
#include <folly/lang/Bits.h>

#include "velox/common/encode/Base64.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/dwrf/reader/ReaderBase.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using TempFilePath = common::testutil::TempFilePath;

class IcebergPositionalDeleteTest : public test::IcebergTestBase {
 protected:
  struct DataFileSpec {
    std::string name;
    std::vector<int64_t> rowGroupSizes;
  };

  struct DeleteRowGroup {
    std::string dataFileName;
    std::vector<int64_t> positions;
  };

  struct DeleteFileSpec {
    std::string name;
    std::vector<DeleteRowGroup> rowGroups;
  };

  static constexpr int32_t kRowCount = 20000;

  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    fileFormat_ = dwio::common::FileFormat::DWRF;
  }

  std::shared_ptr<TempFilePath> writeBigintDataFile(
      const std::vector<int64_t>& values) {
    auto dataFilePath = TempFilePath::create();
    writeToFile(
        dataFilePath->getPath(),
        {makeRowVector({makeFlatVector<int64_t>(values)})});
    return dataFilePath;
  }

  void assertSingleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositions,
      const std::vector<int64_t>& expectedValues) {
    assertPositionalDeletes(
        {{"data_file_1", {10000, 10000}}},
        {{"delete_file_1", {{"data_file_1", deletePositions}}}},
        expectedValues);
  }

  void assertMultipleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositions,
      const std::vector<int64_t>& expectedValues) {
    assertPositionalDeletes(
        {
            {"data_file_0", {5}},
            {"data_file_1", {10}},
            {"data_file_2", {5}},
        },
        {{"delete_file_1", {{"data_file_1", deletePositions}}}},
        expectedValues);
  }

  void assertSingleBaseFileMultipleDeleteFiles(
      const std::vector<std::vector<int64_t>>& deletePositionLists,
      const std::vector<int64_t>& expectedValues) {
    std::vector<DeleteFileSpec> deleteFiles;
    for (int32_t i = 0; i < deletePositionLists.size(); ++i) {
      deleteFiles.push_back(
          {fmt::format("delete_file_{}", i),
           {{"data_file_1", deletePositionLists[i]}}});
    }

    assertPositionalDeletes(
        {{"data_file_1", {20}}}, deleteFiles, expectedValues);
  }

  void assertMultipleSplits(
      const std::vector<int64_t>& deletePositions,
      int32_t fileCount,
      int32_t numPrefetchSplits,
      const std::vector<int64_t>& expectedValues,
      int rowCountPerFile = kRowCount,
      int32_t splitCountPerFile = 1) {
    std::vector<DataFileSpec> dataFiles;
    std::vector<DeleteFileSpec> deleteFiles;
    for (int32_t i = 0; i < fileCount; ++i) {
      const auto dataFileName = fmt::format("data_file_{}", i);
      dataFiles.push_back({dataFileName, {rowCountPerFile}});
      deleteFiles.push_back(
          {fmt::format("delete_file_{}", i),
           {{dataFileName, deletePositions}}});
    }

    assertPositionalDeletes(
        dataFiles,
        deleteFiles,
        expectedValues,
        numPrefetchSplits,
        splitCountPerFile);
  }

  static std::vector<int64_t> sequence(int64_t begin, int64_t end) {
    std::vector<int64_t> values(end - begin);
    std::iota(values.begin(), values.end(), begin);
    return values;
  }

  static std::vector<int64_t> concat(
      std::initializer_list<std::vector<int64_t>> ranges) {
    std::vector<int64_t> values;
    for (const auto& range : ranges) {
      values.insert(values.end(), range.begin(), range.end());
    }
    return values;
  }

  // Writes the requested data files and delete files, then verifies the table
  // scan output against explicit expected values.
  void assertPositionalDeletes(
      const std::vector<DataFileSpec>& dataFiles,
      const std::vector<DeleteFileSpec>& deleteFiles,
      const std::vector<int64_t>& expectedValues,
      int32_t numPrefetchSplits = 0,
      int32_t splitCount = 1) {
    auto dataFilePaths = writeDataFiles(dataFiles);
    auto deleteFilePaths = writePositionDeleteFiles(deleteFiles, dataFilePaths);

    std::vector<std::shared_ptr<ConnectorSplit>> splits;

    for (const auto& dataFile : dataFiles) {
      const auto& baseFilePath = dataFilePaths.at(dataFile.name)->getPath();
      std::vector<IcebergDeleteFile> matchingDeleteFiles;

      for (const auto& deleteFile : deleteFiles) {
        const auto hasDeletesForDataFile = std::any_of(
            deleteFile.rowGroups.begin(),
            deleteFile.rowGroups.end(),
            [&](const auto& rowGroup) {
              return rowGroup.dataFileName == dataFile.name;
            });
        if (!hasDeletesForDataFile) {
          continue;
        }

        const auto deleteFilePath =
            deleteFilePaths.at(deleteFile.name).second->getPath();
        matchingDeleteFiles.emplace_back(
            FileContent::kPositionalDeletes,
            deleteFilePath,
            fileFormat_,
            deleteFilePaths.at(deleteFile.name).first,
            getFileSize(deleteFilePath));
      }

      auto icebergSplits =
          makeIcebergSplits(baseFilePath, matchingDeleteFiles, {}, splitCount);
      splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
    }

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(ROW({"c0"}, {BIGINT()}))
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .config(core::QueryConfig::kMaxSplitPreloadPerDriver, numPrefetchSplits)
        .splits(splits)
        .assertResults(
            makeRowVector({makeFlatVector<int64_t>(expectedValues)}));
  }

  IcebergDeleteFile makePositionalDeleteFile(
      const std::string& baseFilePath,
      const std::vector<int64_t>& deletePositions,
      const std::shared_ptr<TempFilePath>& deleteFilePath,
      int64_t dataSequenceNumber = 0,
      bool includePositionUpperBound = false) {
    writeToFile(
        deleteFilePath->getPath(),
        {makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {
                makeFlatVector<std::string>(
                    static_cast<vector_size_t>(deletePositions.size()),
                    [&](vector_size_t) { return baseFilePath; }),
                makeFlatVector<int64_t>(deletePositions),
            })});

    std::unordered_map<int32_t, std::string> upperBounds;
    if (includePositionUpperBound && !deletePositions.empty()) {
      const uint64_t upperBound = static_cast<uint64_t>(
          *std::max_element(deletePositions.begin(), deletePositions.end()));
      const auto upperBoundLE = folly::Endian::little(upperBound);
      upperBounds[posColumn_->id] = encoding::Base64::encode(
          std::string_view(
              reinterpret_cast<const char*>(&upperBoundLE),
              sizeof(upperBoundLE)));
    }

    return IcebergDeleteFile(
        FileContent::kPositionalDeletes,
        deleteFilePath->getPath(),
        fileFormat_,
        static_cast<int64_t>(deletePositions.size()),
        getFileSize(deleteFilePath->getPath()),
        {},
        {},
        upperBounds,
        dataSequenceNumber);
  }

#ifdef VELOX_ENABLE_PARQUET
  std::vector<std::shared_ptr<ConnectorSplit>> createParquetDeleteFileAndSplits(
      const std::string& path,
      const std::vector<int64_t>& deletePositions,
      int32_t deletedPositionSize,
      const std::shared_ptr<TempFilePath>& deleteFilePath) {
    writeToFile(
        deleteFilePath->getPath(),
        {makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {
                makeFlatVector<std::string>(
                    static_cast<vector_size_t>(deletedPositionSize),
                    [&](vector_size_t) { return path; }),
                makeFlatVector<int64_t>(deletePositions),
            })});

    IcebergDeleteFile icebergDeleteFile(
        FileContent::kPositionalDeletes,
        deleteFilePath->getPath(),
        fileFormat_,
        deletedPositionSize,
        getFileSize(deleteFilePath->getPath()));
    auto file =
        filesystems::getFileSystem(path, nullptr)->openFileForRead(path);

    return {IcebergSplitBuilder(path)
                .connectorId(test::kIcebergConnectorId)
                .fileFormat(dwio::common::FileFormat::PARQUET)
                .length(file->size())
                .deleteFiles({icebergDeleteFile})
                .build()};
  }
#endif

  void assertDeleteSequenceScenario(
      int64_t dataSequenceNumber,
      int64_t deleteSequenceNumber,
      const std::vector<int64_t>& expectedValues) {
    auto dataFilePath = writeBigintDataFile({0, 1, 2, 3, 4});
    auto deleteFilePath = TempFilePath::create();

    auto deleteFile = makePositionalDeleteFile(
        dataFilePath->getPath(), {1, 3}, deleteFilePath, deleteSequenceNumber);

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(ROW({"c0"}, {BIGINT()}))
                    .endTableScan()
                    .planNode();
    auto expected = makeRowVector({makeFlatVector<int64_t>(expectedValues)});
    exec::test::AssertQueryBuilder(plan)
        .splits({makeIcebergSplitWithInfoColumns(
            dataFilePath->getPath(), {}, {deleteFile}, dataSequenceNumber)})
        .assertResults({expected});
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      const std::vector<DataFileSpec>& dataFiles) {
    std::unordered_map<std::string, std::shared_ptr<TempFilePath>>
        dataFilePaths;
    int64_t startingValue = 0;

    for (const auto& dataFile : dataFiles) {
      dataFilePaths[dataFile.name] = TempFilePath::create();
      auto dataVectors = makeVectors(dataFile.rowGroupSizes, startingValue);
      writeToFile(dataFilePaths[dataFile.name]->getPath(), dataVectors);
    }

    return dataFilePaths;
  }

  std::unordered_map<
      std::string,
      std::pair<int64_t, std::shared_ptr<TempFilePath>>>
  writePositionDeleteFiles(
      const std::vector<DeleteFileSpec>& deleteFiles,
      const std::unordered_map<std::string, std::shared_ptr<TempFilePath>>&
          baseFilePaths) {
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths;
    deleteFilePaths.reserve(deleteFiles.size());

    for (const auto& deleteFile : deleteFiles) {
      auto deleteFilePath = TempFilePath::create();

      std::vector<RowVectorPtr> deleteFileVectors;
      int64_t totalPositionsInDeleteFile = 0;

      for (const auto& deleteFileRowGroup : deleteFile.rowGroups) {
        const auto& baseFilePath =
            baseFilePaths.at(deleteFileRowGroup.dataFileName)->getPath();

        deleteFileVectors.push_back(makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {
                makeFlatVector<std::string>(
                    static_cast<vector_size_t>(
                        deleteFileRowGroup.positions.size()),
                    [&](vector_size_t) { return baseFilePath; }),
                makeFlatVector<int64_t>(deleteFileRowGroup.positions),
            }));
        totalPositionsInDeleteFile +=
            static_cast<int64_t>(deleteFileRowGroup.positions.size());
      }

      writeToFile(deleteFilePath->getPath(), deleteFileVectors);
      deleteFilePaths[deleteFile.name] =
          std::make_pair(totalPositionsInDeleteFile, deleteFilePath);
    }

    return deleteFilePaths;
  }

  std::vector<RowVectorPtr> makeVectors(
      const std::vector<int64_t>& vectorSizes,
      int64_t& startingValue) {
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(vectorSizes.size());

    for (auto vectorSize : vectorSizes) {
      auto data = sequence(startingValue, startingValue + vectorSize);
      vectors.push_back(makeRowVector({makeFlatVector<int64_t>(data)}));
      startingValue += vectorSize;
    }

    return vectors;
  }

  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();
};

// Verifies one data file with one positional delete file.
TEST_F(IcebergPositionalDeleteTest, singleBaseFileSinglePositionalDeleteFile) {
  assertSingleBaseFileSingleDeleteFile({0, 1, 2, 3}, sequence(4, kRowCount));
  assertSingleBaseFileSingleDeleteFile(
      {0, 9999, 10000, 19999},
      concat({sequence(1, 9999), sequence(10001, 19999)}));
  assertSingleBaseFileSingleDeleteFile(
      {10000, 10002, 19999},
      concat({sequence(0, 10000), {10001}, sequence(10003, 19999)}));
  assertSingleBaseFileSingleDeleteFile({}, sequence(0, kRowCount));
  assertSingleBaseFileSingleDeleteFile(sequence(0, kRowCount), {});
  assertSingleBaseFileSingleDeleteFile({20000, 29999}, sequence(0, kRowCount));
}

// Verifies that deletes for one base file do not affect neighboring files.
TEST_F(
    IcebergPositionalDeleteTest,
    multipleBaseFilesSinglePositionalDeleteFile) {
  assertMultipleBaseFileSingleDeleteFile(
      {0, 3, 9}, {0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19});
  assertMultipleBaseFileSingleDeleteFile({}, sequence(0, 20));
  assertMultipleBaseFileSingleDeleteFile(
      sequence(0, 10), concat({sequence(0, 5), sequence(15, 20)}));
}

// Verifies one base data file with multiple positional delete files.
TEST_F(
    IcebergPositionalDeleteTest,
    singleBaseFileMultiplePositionalDeleteFiles) {
  assertSingleBaseFileMultipleDeleteFiles(
      {{1}, {2}, {3}, {4}}, concat({{0}, sequence(5, 20)}));
  assertSingleBaseFileMultipleDeleteFiles(
      {{0, 1, 2}, {2, 3, 4}, {25}}, sequence(5, 20));
  assertSingleBaseFileMultipleDeleteFiles({{}, {}}, sequence(0, 20));
  assertSingleBaseFileMultipleDeleteFiles(
      {sequence(0, 20), sequence(0, 20)}, {});
}

// Verifies multiple base files and delete files with unaligned row-group
// boundaries.
TEST_F(
    IcebergPositionalDeleteTest,
    multipleBaseFileMultiplePositionalDeleteFiles) {
  std::vector<DataFileSpec> dataFiles = {
      {"data_file_1", {3, 2}},
      {"data_file_2", {2, 3}},
  };

  assertPositionalDeletes(
      dataFiles,
      {{"delete_file_1", {{"data_file_1", {0, 1}}, {"data_file_2", {2, 4}}}}},
      {2, 3, 4, 5, 6, 8});

  assertPositionalDeletes(
      dataFiles,
      {
          {"delete_file_1", {{"data_file_1", {0, 3}}, {"data_file_2", {1}}}},
          {"delete_file_2",
           {{"data_file_1", {1, 3, 4}}, {"data_file_2", {3, 4}}}},
      },
      {2, 5, 7});
}

TEST_F(IcebergPositionalDeleteTest, positionalDeletesMultipleSplits) {
  assertMultipleSplits(
      {1, 2, 3, 4},
      3,
      5,
      concat({{0}, sequence(5, 21), sequence(25, 41), sequence(45, 60)}),
      20);
  assertMultipleSplits(
      {1, 2, 3, 4},
      3,
      0,
      concat({{0}, sequence(5, 21), sequence(25, 41), sequence(45, 60)}),
      20);
  assertMultipleSplits(
      {0, 19}, 2, 3, concat({sequence(1, 19), sequence(21, 39)}), 20);
  assertMultipleSplits({}, 2, 3, sequence(0, 40), 20);
  assertMultipleSplits(
      {1, 2, 3, 4}, 1, 5, concat({{0}, sequence(5, 30)}), 30, 3);

  assertPositionalDeletes(
      {
          {"data_file_0", {5}},
          {"data_file_1", {10, 10}},
          {"data_file_2", {5}},
      },
      {{"delete_file_1", {{"data_file_1", {0, 10, 19}}}}},
      concat(
          {sequence(0, 5),
           sequence(6, 15),
           sequence(16, 24),
           sequence(25, 30)}),
      0,
      3);

  assertMultipleSplits(
      {10, 19, 20}, 1, 0, concat({sequence(0, 10), sequence(11, 19)}), 20, 3);
}

// Test that positional delete files are skipped when their position upper bound
// is before the split offset. When a delete file's upperBound is less than the
// split's starting row, all deletes in that file are not relevant to this
// split.
TEST_F(IcebergPositionalDeleteTest, skipDeleteFileByPositionUpperBound) {
  // Create a data file with 100 rows (2 stripes of 50 rows each).
  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {
          makeRowVector({makeFlatVector<int64_t>(sequence(0, 50))}),
          makeRowVector({makeFlatVector<int64_t>(sequence(50, 100))}),
      },
      std::make_shared<dwrf::Config>(),
      []() {
        return std::make_unique<dwrf::LambdaFlushPolicy>([]() { return true; });
      });

  // Create a delete file targeting positions 0, 1, 2.
  auto deleteFilePath = TempFilePath::create();
  auto deleteFile = makePositionalDeleteFile(
      dataFilePath->getPath(), {0, 1, 2}, deleteFilePath, 0, true);

  // Create a split that starts at the middle of the file. The split offset
  // will be greater than the delete file's upper bound (2), so the delete
  // file should be skipped completely.
  auto filePtr = filesystems::getFileSystem(dataFilePath->getPath(), nullptr)
                     ->openFileForRead(dataFilePath->getPath());
  const uint64_t fileSize = filePtr->size();
  std::shared_ptr<ReadFile> file = std::move(filePtr);
  dwio::common::ReaderOptions readerOptions{pool()};
  auto input = std::make_unique<dwio::common::BufferedInput>(file, *pool());
  auto reader =
      std::make_unique<dwrf::ReaderBase>(readerOptions, std::move(input));
  reader->loadCache();
  ASSERT_GE(reader->footer().stripesSize(), 2);
  const uint64_t splitStart = reader->footer().stripes(1).offset();
  std::shared_ptr<ConnectorSplit> split =
      IcebergSplitBuilder(dataFilePath->getPath())
          .connectorId(test::kIcebergConnectorId)
          .fileFormat(fileFormat_)
          .start(splitStart)
          .length(fileSize - splitStart)
          .deleteFiles({deleteFile})
          .build();

  // The second half of the file should be returned with no rows deleted.
  auto expected = makeRowVector({makeFlatVector<int64_t>(sequence(50, 100))});
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan).splits({split}).assertResults(
      {expected});
}
#ifdef VELOX_ENABLE_PARQUET
TEST_F(IcebergPositionalDeleteTest, positionalDeleteFileWithRowGroupFilter) {
  // This file contains three row groups. The remaining filter prunes the
  // middle row group, which verifies that position deletes still align with
  // the correct file offsets.
  auto path = facebook::velox::test::getDataFilePath(
      "velox/connectors/hive/iceberg/tests", "examples/three_groups.parquet");
  const int32_t deletedPositionSize = 100;
  std::vector<int64_t> deletePositionsVec(deletedPositionSize);
  std::iota(deletePositionsVec.begin(), deletePositionsVec.end(), 100);
  auto deleteFilePath = TempFilePath::create();

  assertQuery(
      exec::test::PlanBuilder()
          .startTableScan(test::kIcebergConnectorId)
          .outputType(ROW({"id"}, {BIGINT()}))
          .remainingFilter("id >= 100")
          .endTableScan()
          .planNode(),
      createParquetDeleteFileAndSplits(
          path, deletePositionsVec, deletedPositionSize, deleteFilePath),
      "SELECT i AS id FROM range(100, 300) AS t(i)",
      0);
}
#endif

TEST_F(IcebergPositionalDeleteTest, positionalDeleteSequenceNumberApplied) {
  // Sequence number filtering tests for positional deletes (Diff 2).
  // Per the Iceberg V2+ spec, a positional delete file should only apply to
  // data files whose dataSequenceNumber is strictly less than the delete
  // file's.
  assertDeleteSequenceScenario(
      /*dataSequenceNumber=*/5, /*deleteSequenceNumber=*/10, {0, 2, 4});
}

TEST_F(IcebergPositionalDeleteTest, positionalDeleteSequenceNumberSkipped) {
  assertDeleteSequenceScenario(
      /*dataSequenceNumber=*/10, /*deleteSequenceNumber=*/5, {0, 1, 2, 3, 4});
}

// Verifies that same-snapshot positional deletes apply (deleteSeqNum ==
// dataSeqNum). Per the Iceberg spec, positional deletes in the same snapshot
// SHOULD apply, so the skip condition uses strict < (not <=).
TEST_F(
    IcebergPositionalDeleteTest,
    positionalDeleteSequenceNumberEqualApplied) {
  // Same-snapshot delete applied: positions 1 and 3 deleted -> [0, 2, 4].
  assertDeleteSequenceScenario(
      /*dataSequenceNumber=*/5, /*deleteSequenceNumber=*/5, {0, 2, 4});
}

TEST_F(
    IcebergPositionalDeleteTest,
    positionalDeleteSequenceNumberZeroDisablesFilter) {
  // SeqNum=0 disables filtering -> delete applied: [0, 2, 4].
  assertDeleteSequenceScenario(
      /*dataSequenceNumber=*/100, /*deleteSequenceNumber=*/0, {0, 2, 4});
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
