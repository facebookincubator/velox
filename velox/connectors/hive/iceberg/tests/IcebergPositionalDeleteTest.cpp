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
#include <map>
#include <numeric>
#include <random>

#include <folly/Random.h>
#include <folly/Singleton.h>
#include <folly/lang/Bits.h>

#include "velox/common/encode/Base64.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/dwrf/reader/ReaderBase.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using TempFilePath = common::testutil::TempFilePath;

class IcebergPositionalDeleteTest : public test::IcebergTestBase {
 protected:
  using RowGroupSizesForFiles = std::map<std::string, std::vector<int64_t>>;
  using DeleteFilesForBaseDataFiles = std::unordered_map<
      std::string,
      std::multimap<std::string, std::vector<int64_t>>>;

  static constexpr int32_t rowCount = 20000;

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
      const std::vector<int64_t>& deletePositionsVec) {
    assertPositionalDeletes(
        {{"data_file_1", {10000, 10000}}},
        {{"delete_file_1", {{"data_file_1", deletePositionsVec}}}},
        0);
  }

  void assertMultipleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositionsVec) {
    assertPositionalDeletes(
        {
            {"data_file_0", {500}},
            {"data_file_1", {10000, 10000}},
            {"data_file_2", {500}},
        },
        {{"delete_file_1", {{"data_file_1", deletePositionsVec}}}},
        0);
  }

  void assertSingleBaseFileMultipleDeleteFiles(
      const std::vector<std::vector<int64_t>>& deletePositionsVecs) {
    DeleteFilesForBaseDataFiles deleteFilesForBaseDatafiles;
    for (int32_t i = 0; i < deletePositionsVecs.size(); ++i) {
      deleteFilesForBaseDatafiles[fmt::format("delete_file_{}", i)] = {
          {"data_file_1", deletePositionsVecs[i]}};
    }

    assertPositionalDeletes(
        {{"data_file_1", {10000, 10000}}}, deleteFilesForBaseDatafiles, 0);
  }

  void assertMultipleSplits(
      const std::vector<int64_t>& deletePositions,
      int32_t fileCount,
      int32_t numPrefetchSplits,
      int rowCountPerFile = rowCount,
      int32_t splitCountPerFile = 1) {
    RowGroupSizesForFiles rowGroupSizesForFiles;
    DeleteFilesForBaseDataFiles deleteFilesForBaseDatafiles;
    for (int32_t i = 0; i < fileCount; ++i) {
      const auto dataFileName = fmt::format("data_file_{}", i);
      rowGroupSizesForFiles[dataFileName] = {rowCountPerFile};
      deleteFilesForBaseDatafiles[fmt::format("delete_file_{}", i)] = {
          {dataFileName, deletePositions}};
    }

    assertPositionalDeletes(
        rowGroupSizesForFiles,
        deleteFilesForBaseDatafiles,
        numPrefetchSplits,
        splitCountPerFile);
  }

  std::vector<int64_t> makeRandomIncreasingValues(int64_t begin, int64_t end) {
    VELOX_CHECK(begin < end);

    std::mt19937 gen{0};
    std::vector<int64_t> values;
    values.reserve(end - begin);
    for (int64_t i = begin; i < end; ++i) {
      if (folly::Random::rand32(0, 10, gen) > 8) {
        values.push_back(i);
      }
    }
    return values;
  }

  static std::vector<int64_t> makeContinuousIncreasingValues(
      int64_t begin,
      int64_t end) {
    std::vector<int64_t> values(end - begin);
    std::iota(values.begin(), values.end(), begin);
    return values;
  }

  /// @rowGroupSizesForFiles The key is the file name, and the value is a vector
  /// of RowGroup sizes
  /// @deleteFilesForBaseDatafiles The key is the delete file name, and the
  /// value contains the information about the content of this delete file.
  /// e.g. {
  ///         "delete_file_1",
  ///         {
  ///             {"data_file_1", {1, 2, 3}},
  ///             {"data_file_1", {4, 5, 6}},
  ///             {"data_file_2", {0, 2, 4}}
  ///         }
  ///     }
  /// represents one delete file called delete_file_1, which contains delete
  /// positions for data_file_1 and data_file_2. THere are 3 RowGroups in this
  /// delete file, the first two contain positions for data_file_1, and the last
  /// contain positions for data_file_2
  void assertPositionalDeletes(
      const RowGroupSizesForFiles& rowGroupSizesForFiles,
      const DeleteFilesForBaseDataFiles& deleteFilesForBaseDatafiles,
      int32_t numPrefetchSplits = 0,
      int32_t splitCount = 1) {
    auto dataFilePaths = writeDataFiles(rowGroupSizesForFiles);
    auto deleteFilePaths =
        writePositionDeleteFiles(deleteFilesForBaseDatafiles, dataFilePaths);

    std::vector<std::shared_ptr<ConnectorSplit>> splits;

    for (const auto& dataFile : dataFilePaths) {
      const auto& baseFileName = dataFile.first;
      const auto& baseFilePath = dataFile.second->getPath();
      std::vector<IcebergDeleteFile> deleteFiles;

      for (const auto& deleteFile : deleteFilesForBaseDatafiles) {
        const auto& deleteFileName = deleteFile.first;
        const auto& deleteFileContent = deleteFile.second;

        if (!deleteFileContent.contains(baseFileName)) {
          continue;
        }

        const auto deleteFilePath =
            deleteFilePaths[deleteFileName].second->getPath();
        deleteFiles.emplace_back(
            FileContent::kPositionalDeletes,
            deleteFilePath,
            fileFormat_,
            deleteFilePaths[deleteFileName].first,
            getFileSize(deleteFilePath));
      }

      auto icebergSplits =
          makeIcebergSplits(baseFilePath, deleteFiles, {}, splitCount);
      splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
    }

    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(ROW({"c0"}, {BIGINT()}))
                    .endTableScan()
                    .planNode();
    auto task = assertQuery(
        plan,
        splits,
        getDuckDBQuery(rowGroupSizesForFiles, deleteFilesForBaseDatafiles),
        numPrefetchSplits);

    const auto planStats = exec::toPlanStats(task->taskStats());
    const auto it = planStats.find(plan->id());
    ASSERT_TRUE(it != planStats.end());
    ASSERT_TRUE(it->second.peakMemoryBytes > 0);
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
      const std::vector<int64_t>& deletePositionsVec,
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
                makeFlatVector<int64_t>(deletePositionsVec),
            })});

    IcebergDeleteFile icebergDeleteFile(
        FileContent::kPositionalDeletes,
        deleteFilePath->getPath(),
        fileFormat_,
        deletedPositionSize,
        getFileSize(deleteFilePath->getPath()));
    auto file =
        filesystems::getFileSystem(path, nullptr)->openFileForRead(path);

    return {std::make_shared<HiveIcebergSplit>(
        test::kIcebergConnectorId,
        path,
        dwio::common::FileFormat::PARQUET,
        0,
        file->size(),
        std::unordered_map<std::string, std::optional<std::string>>{},
        std::nullopt,
        std::unordered_map<std::string, std::string>{},
        nullptr,
        /*cacheable=*/true,
        std::vector<IcebergDeleteFile>{icebergDeleteFile})};
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

    auto expected = makeRowVector({makeFlatVector<int64_t>(expectedValues)});
    assertTableScan(
        ROW({"c0"}, {BIGINT()}),
        {makeIcebergSplitWithInfoColumns(
            dataFilePath->getPath(), {}, {deleteFile}, dataSequenceNumber)},
        {expected});
  }

 private:
  std::map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      const RowGroupSizesForFiles& rowGroupSizesForFiles) {
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths;

    std::vector<RowVectorPtr> dataVectorsJoined;
    dataVectorsJoined.reserve(rowGroupSizesForFiles.size());

    int64_t startingValue = 0;
    for (auto& dataFile : rowGroupSizesForFiles) {
      dataFilePaths[dataFile.first] = TempFilePath::create();
      // We make the values continuously increasing even across base data files.
      // This makes constructing DuckDB queries easier.
      auto dataVectors = makeVectors(dataFile.second, startingValue);
      writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors);
      dataVectorsJoined.insert(
          dataVectorsJoined.end(), dataVectors.begin(), dataVectors.end());
    }

    createDuckDbTable(dataVectorsJoined);
    return dataFilePaths;
  }

  std::unordered_map<
      std::string,
      std::pair<int64_t, std::shared_ptr<TempFilePath>>>
  writePositionDeleteFiles(
      const DeleteFilesForBaseDataFiles& deleteFilesForBaseDatafiles,
      std::map<std::string, std::shared_ptr<TempFilePath>> baseFilePaths) {
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths;
    deleteFilePaths.reserve(deleteFilesForBaseDatafiles.size());

    for (const auto& deleteFile : deleteFilesForBaseDatafiles) {
      const auto& deleteFileName = deleteFile.first;
      const auto& deleteFileContent = deleteFile.second;
      auto deleteFilePath = TempFilePath::create();

      std::vector<RowVectorPtr> deleteFileVectors;
      int64_t totalPositionsInDeleteFile = 0;

      for (const auto& deleteFileRowGroup : deleteFileContent) {
        const auto& baseFileName = deleteFileRowGroup.first;
        const auto& positionsInRowGroup = deleteFileRowGroup.second;
        const auto& baseFilePath = baseFilePaths[baseFileName]->getPath();

        deleteFileVectors.push_back(makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {
                makeFlatVector<std::string>(
                    static_cast<vector_size_t>(positionsInRowGroup.size()),
                    [&](vector_size_t) { return baseFilePath; }),
                makeFlatVector<int64_t>(positionsInRowGroup),
            }));
        totalPositionsInDeleteFile +=
            static_cast<int64_t>(positionsInRowGroup.size());
      }

      writeToFile(deleteFilePath->getPath(), deleteFileVectors);
      deleteFilePaths[deleteFileName] =
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
      auto data = makeContinuousIncreasingValues(
          startingValue, startingValue + vectorSize);
      vectors.push_back(makeRowVector({makeFlatVector<int64_t>(data)}));
      startingValue += vectorSize;
    }

    return vectors;
  }

  std::string getDuckDBQuery(
      const RowGroupSizesForFiles& rowGroupSizesForFiles,
      const DeleteFilesForBaseDataFiles& deleteFilesForBaseDatafiles) {
    int64_t totalNumRowsInAllBaseFiles = 0;
    std::map<std::string, int64_t> baseFileSizes;
    for (const auto& rowGroupSizesInFile : rowGroupSizesForFiles) {
      // Sum up the row counts in all RowGroups in each base file.
      baseFileSizes[rowGroupSizesInFile.first] += std::accumulate(
          rowGroupSizesInFile.second.begin(),
          rowGroupSizesInFile.second.end(),
          0LL);
      totalNumRowsInAllBaseFiles += baseFileSizes[rowGroupSizesInFile.first];
    }

    std::map<std::string, std::vector<std::vector<int64_t>>>
        deletePosVectorsForAllBaseFiles;
    for (const auto& deleteFile : deleteFilesForBaseDatafiles) {
      // Group the delete vectors by baseFileName.
      for (const auto& rowGroup : deleteFile.second) {
        deletePosVectorsForAllBaseFiles[rowGroup.first].push_back(
            rowGroup.second);
      }
    }

    std::map<std::string, std::vector<int64_t>>
        flattenedDeletePosVectorsForAllBaseFiles;
    int64_t totalNumDeletePositions = 0;
    for (const auto& deleteVectorsForBaseFile :
         deletePosVectorsForAllBaseFiles) {
      // Flatten and deduplicate the delete position vectors in
      // deletePosVectorsForAllBaseFiles from previous step, and count the total
      // number of distinct delete positions for all base files.
      auto deletePositionVector = flattenAndDedup(
          deleteVectorsForBaseFile.second,
          baseFileSizes[deleteVectorsForBaseFile.first]);
      flattenedDeletePosVectorsForAllBaseFiles[deleteVectorsForBaseFile.first] =
          deletePositionVector;
      totalNumDeletePositions +=
          static_cast<int64_t>(deletePositionVector.size());
    }

    if (totalNumDeletePositions == 0) {
      return "SELECT * FROM tmp";
    }

    if (totalNumDeletePositions >= totalNumRowsInAllBaseFiles) {
      return "SELECT * FROM tmp WHERE 1 = 0";
    }

    std::vector<int64_t> allDeleteValues;
    int64_t numRowsInPreviousBaseFiles = 0;
    // Now build the DuckDB query by converting delete positions in all base
    // files into c0 values.
    for (const auto& baseFileSize : baseFileSizes) {
      auto deletePositions =
          flattenedDeletePosVectorsForAllBaseFiles[baseFileSize.first];

      if (numRowsInPreviousBaseFiles > 0) {
        for (int64_t& deleteValue : deletePositions) {
          deleteValue += numRowsInPreviousBaseFiles;
        }
      }

      allDeleteValues.insert(
          allDeleteValues.end(),
          deletePositions.begin(),
          deletePositions.end());
      numRowsInPreviousBaseFiles += baseFileSize.second;
    }

    return fmt::format(
        "SELECT * FROM tmp WHERE c0 NOT IN ({})",
        folly::join(", ", allDeleteValues));
  }

  std::vector<int64_t> flattenAndDedup(
      const std::vector<std::vector<int64_t>>& deletePositionVectors,
      int64_t baseFileSize) {
    std::vector<int64_t> deletePositionVector;
    for (const auto& vec : deletePositionVectors) {
      for (auto pos : vec) {
        if (pos >= 0 && pos < baseFileSize) {
          deletePositionVector.push_back(pos);
        }
      }
    }

    std::sort(deletePositionVector.begin(), deletePositionVector.end());
    deletePositionVector.erase(
        std::unique(deletePositionVector.begin(), deletePositionVector.end()),
        deletePositionVector.end());
    return deletePositionVector;
  }

  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();
};

/// This test creates one single data file and one delete file. The parameter
/// passed to assertSingleBaseFileSingleDeleteFile is the delete positions.
TEST_F(IcebergPositionalDeleteTest, singleBaseFileSinglePositionalDeleteFile) {
  // Delete the first and last row in each batch (10000 rows per batch).
  assertSingleBaseFileSingleDeleteFile({{0, 1, 2, 3}});
  assertSingleBaseFileSingleDeleteFile({{0, 9999, 10000, 19999}});
  // Delete several rows in the second batch (10000 rows per batch).
  assertSingleBaseFileSingleDeleteFile({{10000, 10002, 19999}});
  // Delete random rows.
  assertSingleBaseFileSingleDeleteFile({makeRandomIncreasingValues(0, 20000)});
  // Delete 0 rows.
  assertSingleBaseFileSingleDeleteFile({});
  // Delete all rows.
  assertSingleBaseFileSingleDeleteFile(
      makeContinuousIncreasingValues(0, 20000));
  // Delete rows that don't exist.
  assertSingleBaseFileSingleDeleteFile({{20000, 29999}});
}

/// This test creates 3 base data files, only the middle one has corresponding
/// delete positions. The parameter passed to
/// assertSingleBaseFileSingleDeleteFile is the delete positions for the middle
/// base file.
TEST_F(
    IcebergPositionalDeleteTest,
    multipleBaseFilesSinglePositionalDeleteFile) {
  assertMultipleBaseFileSingleDeleteFile({0, 1, 2, 3});
  assertMultipleBaseFileSingleDeleteFile({0, 9999, 10000, 19999});
  assertMultipleBaseFileSingleDeleteFile({10000, 10002, 19999});
  assertMultipleBaseFileSingleDeleteFile({10000, 10002, 19999});
  assertMultipleBaseFileSingleDeleteFile(
      makeRandomIncreasingValues(0, rowCount));
  assertMultipleBaseFileSingleDeleteFile({});
  assertMultipleBaseFileSingleDeleteFile(
      makeContinuousIncreasingValues(0, rowCount));
}

/// This test creates one base data file/split with multiple delete files. The
/// parameter passed to assertSingleBaseFileMultipleDeleteFiles is the vector of
/// delete files. Each leaf vector represents the delete positions in that
/// delete file.
TEST_F(
    IcebergPositionalDeleteTest,
    singleBaseFileMultiplePositionalDeleteFiles) {
  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertSingleBaseFileMultipleDeleteFiles({{1}, {2}, {3}, {4}});
  // Delete the first and last row in each batch (10000 rows per batch).
  assertSingleBaseFileMultipleDeleteFiles({{0}, {9999}, {10000}, {19999}});
  assertSingleBaseFileMultipleDeleteFiles({{500, 21000}});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeRandomIncreasingValues(0, 10000),
       makeRandomIncreasingValues(10000, 20000),
       makeRandomIncreasingValues(5000, 15000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeContinuousIncreasingValues(0, 10000),
       makeContinuousIncreasingValues(10000, 20000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeContinuousIncreasingValues(0, 10000),
       makeContinuousIncreasingValues(10000, 20000),
       makeRandomIncreasingValues(5000, 15000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeContinuousIncreasingValues(0, 20000),
       makeContinuousIncreasingValues(0, 20000)});
  assertSingleBaseFileMultipleDeleteFiles(
      {makeRandomIncreasingValues(0, 20000),
       {},
       makeRandomIncreasingValues(5000, 15000)});
  assertSingleBaseFileMultipleDeleteFiles({{}, {}});
}

/// This test creates 2 base data files, and 1 or 2 delete files, with
/// unaligned RowGroup boundaries.
TEST_F(
    IcebergPositionalDeleteTest,
    multipleBaseFileMultiplePositionalDeleteFiles) {
  // Create two data files, each with two RowGroups.
  RowGroupSizesForFiles rowGroupSizesForFiles = {
      {"data_file_1", {100, 85}},
      {"data_file_2", {99, 1}},
  };
  DeleteFilesForBaseDataFiles deleteFilesForBaseDatafiles;

  // Delete 3 rows from the first RowGroup in data_file_1.
  deleteFilesForBaseDatafiles["delete_file_1"] = {{"data_file_1", {0, 1, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete 3 rows from the second RowGroup in data_file_1.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {100, 101, 184}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete random rows from both RowGroups in data_file_1.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete all rows in data_file_1.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeContinuousIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete non-existent rows from data_file_1.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(186, 300)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  // Delete several rows from both RowGroups in both data files.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 100, 102, 184}}, {"data_file_2", {1, 98, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  // The delete file delete_file_1 contains 3 RowGroups itself, with the first
  // rows deleting some repeating rows in data_file_1, and the last rows
  // deleting some repeating rows in data_file_2.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}},
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", {1, 3, 5, 7}},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  // delete_file_2 contains non-overlapping delete rows for each data file in
  // each RowGroup.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}}, {"data_file_2", {1, 3, 5, 7}}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", {98, 99, 100, 101, 184}},
      {"data_file_2", {3, 5, 7, 9}},
      {"data_file_2", {98, 99, 100}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  // Two delete files each containing overlapping delete rows for both data
  // files.
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", makeRandomIncreasingValues(10, 120)},
      {"data_file_2", makeRandomIncreasingValues(50, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
}

TEST_F(IcebergPositionalDeleteTest, positionalDeletesMultipleSplits) {
  assertMultipleSplits({1, 2, 3, 4}, 10, 5);
  assertMultipleSplits({1, 2, 3, 4}, 10, 0);
  assertMultipleSplits({1, 2, 3, 4}, 10, 10);
  assertMultipleSplits({0, 9999, 10000, 19999}, 10, 3);
  assertMultipleSplits(makeRandomIncreasingValues(0, 20000), 10, 3);
  assertMultipleSplits(makeContinuousIncreasingValues(0, 20000), 10, 3);
  assertMultipleSplits({}, 10, 3);
  assertMultipleSplits({1, 2, 3, 4}, 10, 5, 30000, 3);

  assertPositionalDeletes(
      {
          {"data_file_0", {500}},
          {"data_file_1", {10000, 10000}},
          {"data_file_2", {500}},
      },
      {{"delete_file_1",
        {{"data_file_1", makeRandomIncreasingValues(0, 20000)}}}},
      0,
      3);

  assertMultipleSplits({1000, 9000, 20000}, 1, 0, 20000, 3);
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
          makeRowVector(
              {makeFlatVector<int64_t>(makeContinuousIncreasingValues(0, 50))}),
          makeRowVector({makeFlatVector<int64_t>(
              makeContinuousIncreasingValues(50, 100))}),
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
  auto split = std::make_shared<HiveIcebergSplit>(
      test::kIcebergConnectorId,
      dataFilePath->getPath(),
      fileFormat_,
      splitStart,
      fileSize - splitStart,
      std::unordered_map<std::string, std::optional<std::string>>{},
      std::nullopt,
      std::unordered_map<std::string, std::string>{},
      nullptr,
      /*cacheable=*/true,
      std::vector<IcebergDeleteFile>{deleteFile});

  // The second half of the file should be returned with no rows deleted.
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>(makeContinuousIncreasingValues(50, 100))});
  assertTableScan(ROW({"c0"}, {BIGINT()}), {split}, {expected});
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
          .startTableScan()
          .connectorId(test::kIcebergConnectorId)
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
  // Same-snapshot delete applied: positions 1 and 3 deleted → [0, 2, 4].
  assertDeleteSequenceScenario(
      /*dataSequenceNumber=*/5, /*deleteSequenceNumber=*/5, {0, 2, 4});
}

TEST_F(
    IcebergPositionalDeleteTest,
    positionalDeleteSequenceNumberZeroDisablesFilter) {
  // SeqNum=0 disables filtering → delete applied: [0, 2, 4].
  assertDeleteSequenceScenario(
      /*dataSequenceNumber=*/100, /*deleteSequenceNumber=*/0, {0, 2, 4});
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
