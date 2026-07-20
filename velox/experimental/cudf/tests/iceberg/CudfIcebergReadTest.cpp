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

/// Basic end-to-end read tests for the cudf Iceberg connector.
/// Deletion vector and equality delete tests live in their own files
/// (CudfDeletionVectorReaderTest.cpp and CudfEqualityDeleteFileReaderTest.cpp).

#include "velox/experimental/cudf/tests/iceberg/CudfIcebergTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Timestamp.h"
#include "velox/type/TimestampConversion.h"

#include <folly/Random.h>
#include <folly/String.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::connector::hive::iceberg;
using facebook::velox::common::testutil::TempFilePath;
using facebook::velox::connector::hive::HiveColumnHandle;

namespace facebook::velox::cudf_velox::exec::test {

class CudfIcebergReadTest : public CudfIcebergTestBase {
 protected:
  static constexpr int rowCount = 20000;

  static std::vector<int64_t> makeContinuousIncreasingValues(
      int64_t begin,
      int64_t end) {
    std::vector<int64_t> values;
    values.resize(end - begin);
    std::iota(values.begin(), values.end(), begin);
    return values;
  }

  std::vector<int64_t> makeRandomIncreasingValues(int64_t begin, int64_t end) {
    VELOX_CHECK(begin < end);
    std::mt19937 gen{0};
    std::vector<int64_t> values;
    values.reserve(end - begin);
    for (int i = begin; i < end; i++) {
      if (folly::Random::rand32(0, 10, gen) > 8) {
        values.push_back(i);
      }
    }
    return values;
  }

  /// @rowGroupSizesForFiles The key is the file name, and the value is a vector
  /// of RowGroup sizes.
  /// @deleteFilesForBaseDatafiles The key is the delete file name, and the
  /// value contains the information about the content of this delete file.
  void assertPositionalDeletes(
      const std::map<std::string, std::vector<int64_t>>& rowGroupSizesForFiles,
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles,
      int32_t numPrefetchSplits = 0,
      int32_t splitCount = 1) {
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths =
        writeDataFiles(rowGroupSizesForFiles);
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths = writePositionDeleteFiles(
            deleteFilesForBaseDatafiles, dataFilePaths);

    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;

    for (const auto& dataFile : dataFilePaths) {
      std::string baseFileName = dataFile.first;
      std::string baseFilePath = dataFile.second->getPath();

      std::vector<IcebergDeleteFile> deleteFiles;

      for (auto const& deleteFile : deleteFilesForBaseDatafiles) {
        std::string deleteFileName = deleteFile.first;
        std::multimap<std::string, std::vector<int64_t>> deleteFileContent =
            deleteFile.second;

        if (deleteFileContent.count(baseFileName) != 0) {
          auto deleteFilePath =
              deleteFilePaths[deleteFileName].second->getPath();
          IcebergDeleteFile icebergDeleteFile(
              FileContent::kPositionalDeletes,
              deleteFilePath,
              dwio::common::FileFormat::DWRF,
              deleteFilePaths[deleteFileName].first,
              getFileSize(deleteFilePath));
          deleteFiles.push_back(icebergDeleteFile);
        }
      }

      auto icebergSplits =
          makeIcebergSplits(baseFilePath, deleteFiles, {}, splitCount);
      splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
    }

    std::string duckdbSql =
        getDuckDBQuery(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
    auto plan = PlanBuilder()
                    .startTableScan()
                    .connectorId(kCudfIcebergConnectorId)
                    .outputType(ROW({"c0"}, {BIGINT()}))
                    .endTableScan()
                    .planNode();
    auto task = assertQuery(plan, splits, duckdbSql, numPrefetchSplits);

    auto planStats = toPlanStats(task->taskStats());
    auto it = planStats.find(plan->id());
    ASSERT_TRUE(it != planStats.end());
    // TODO (mh): enable once we start to track gpu memory
    // ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  }

  void assertSingleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositionsVec) {
    std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles = {
        {"data_file_1", {10000, 10000}}};
    std::unordered_map<
        std::string,
        std::multimap<std::string, std::vector<int64_t>>>
        deleteFilesForBaseDatafiles = {
            {"delete_file_1", {{"data_file_1", deletePositionsVec}}}};
    assertPositionalDeletes(
        rowGroupSizesForFiles, deleteFilesForBaseDatafiles, 0);
  }

  void assertMultipleBaseFileSingleDeleteFile(
      const std::vector<int64_t>& deletePositionsVec) {
    int64_t previousFileRowCount = 500;
    int64_t afterFileRowCount = 500;
    assertPositionalDeletes(
        {
            {"data_file_0", {previousFileRowCount}},
            {"data_file_1", {10000, 10000}},
            {"data_file_2", {afterFileRowCount}},
        },
        {{"delete_file_1", {{"data_file_1", deletePositionsVec}}}},
        0);
  }

  void assertSingleBaseFileMultipleDeleteFiles(
      const std::vector<std::vector<int64_t>>& deletePositionsVecs) {
    std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles = {
        {"data_file_1", {10000, 10000}}};
    std::unordered_map<
        std::string,
        std::multimap<std::string, std::vector<int64_t>>>
        deleteFilesForBaseDatafiles;
    for (int i = 0; i < deletePositionsVecs.size(); i++) {
      std::string deleteFileName = fmt::format("delete_file_{}", i);
      deleteFilesForBaseDatafiles[deleteFileName] = {
          {"data_file_1", deletePositionsVecs[i]}};
    }
    assertPositionalDeletes(
        rowGroupSizesForFiles, deleteFilesForBaseDatafiles, 0);
  }

  void assertMultipleSplits(
      const std::vector<int64_t>& deletePositions,
      int32_t fileCount,
      int32_t numPrefetchSplits,
      int rowCountPerFile = rowCount,
      int32_t splitCountPerFile = 1) {
    std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles;
    for (int32_t i = 0; i < fileCount; i++) {
      std::string dataFileName = fmt::format("data_file_{}", i);
      rowGroupSizesForFiles[dataFileName] = {rowCountPerFile};
    }
    std::unordered_map<
        std::string,
        std::multimap<std::string, std::vector<int64_t>>>
        deleteFilesForBaseDatafiles;
    for (int i = 0; i < fileCount; i++) {
      std::string deleteFileName = fmt::format("delete_file_{}", i);
      deleteFilesForBaseDatafiles[deleteFileName] = {
          {fmt::format("data_file_{}", i), deletePositions}};
    }
    assertPositionalDeletes(
        rowGroupSizesForFiles,
        deleteFilesForBaseDatafiles,
        numPrefetchSplits,
        splitCountPerFile);
  }

 private:
  std::map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles) {
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths;
    std::vector<RowVectorPtr> dataVectorsJoined;
    dataVectorsJoined.reserve(rowGroupSizesForFiles.size());

    int64_t startingValue = 0;
    for (auto& dataFile : rowGroupSizesForFiles) {
      dataFilePaths[dataFile.first] = TempFilePath::create();

      std::vector<RowVectorPtr> dataVectors =
          makeVectors(dataFile.second, startingValue);
      writeToFile(dataFilePaths[dataFile.first]->getPath(), dataVectors);

      for (int i = 0; i < dataVectors.size(); i++) {
        dataVectorsJoined.push_back(dataVectors[i]);
      }
    }

    createDuckDbTable(dataVectorsJoined);
    return dataFilePaths;
  }

  std::unordered_map<
      std::string,
      std::pair<int64_t, std::shared_ptr<TempFilePath>>>
  writePositionDeleteFiles(
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles,
      std::map<std::string, std::shared_ptr<TempFilePath>> baseFilePaths) {
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths;
    deleteFilePaths.reserve(deleteFilesForBaseDatafiles.size());

    for (auto& deleteFile : deleteFilesForBaseDatafiles) {
      auto deleteFileName = deleteFile.first;
      auto deleteFileContent = deleteFile.second;
      auto deleteFilePath = TempFilePath::create();

      std::vector<RowVectorPtr> deleteFileVectors;
      int64_t totalPositionsInDeleteFile = 0;

      for (auto& deleteFileRowGroup : deleteFileContent) {
        auto baseFileName = deleteFileRowGroup.first;
        auto baseFilePath = baseFilePaths[baseFileName]->getPath();
        auto positionsInRowGroup = deleteFileRowGroup.second;

        auto filePathVector = makeFlatVector<std::string>(
            static_cast<vector_size_t>(positionsInRowGroup.size()),
            [&](vector_size_t) { return baseFilePath; });
        auto deletePosVector = makeFlatVector<int64_t>(positionsInRowGroup);

        RowVectorPtr deleteFileVector = makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {filePathVector, deletePosVector});

        deleteFileVectors.push_back(deleteFileVector);
        totalPositionsInDeleteFile += positionsInRowGroup.size();
      }

      writeDeleteFile(
          DeleteFileFormat::DWRF, deleteFilePath->getPath(), deleteFileVectors);

      deleteFilePaths[deleteFileName] =
          std::make_pair(totalPositionsInDeleteFile, deleteFilePath);
    }

    return deleteFilePaths;
  }

  std::vector<RowVectorPtr> makeVectors(
      std::vector<int64_t> vectorSizes,
      int64_t& startingValue) {
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(vectorSizes.size());
    for (int j = 0; j < vectorSizes.size(); j++) {
      auto data = makeContinuousIncreasingValues(
          startingValue, startingValue + vectorSizes[j]);
      vectors.push_back(makeRowVector({makeFlatVector<int64_t>(data)}));
      startingValue += vectorSizes[j];
    }
    return vectors;
  }

  std::string getDuckDBQuery(
      const std::map<std::string, std::vector<int64_t>>& rowGroupSizesForFiles,
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles) {
    int64_t totalNumRowsInAllBaseFiles = 0;
    std::map<std::string, int64_t> baseFileSizes;
    for (auto rowGroupSizesInFile : rowGroupSizesForFiles) {
      baseFileSizes[rowGroupSizesInFile.first] += std::accumulate(
          rowGroupSizesInFile.second.begin(),
          rowGroupSizesInFile.second.end(),
          0LL);
      totalNumRowsInAllBaseFiles += baseFileSizes[rowGroupSizesInFile.first];
    }

    std::map<std::string, std::vector<std::vector<int64_t>>>
        deletePosVectorsForAllBaseFiles;
    for (auto& deleteFile : deleteFilesForBaseDatafiles) {
      auto deleteFileContent = deleteFile.second;
      for (auto& rowGroup : deleteFileContent) {
        auto baseFileName = rowGroup.first;
        deletePosVectorsForAllBaseFiles[baseFileName].push_back(
            rowGroup.second);
      }
    }

    std::map<std::string, std::vector<int64_t>>
        flattenedDeletePosVectorsForAllBaseFiles;
    int64_t totalNumDeletePositions = 0;
    for (auto& deleteVectorsForBaseFile : deletePosVectorsForAllBaseFiles) {
      auto baseFileName = deleteVectorsForBaseFile.first;
      auto deletePositionVectors = deleteVectorsForBaseFile.second;
      std::vector<int64_t> deletePositionVector =
          flattenAndDedup(deletePositionVectors, baseFileSizes[baseFileName]);
      flattenedDeletePosVectorsForAllBaseFiles[baseFileName] =
          deletePositionVector;
      totalNumDeletePositions += deletePositionVector.size();
    }

    if (totalNumDeletePositions == 0) {
      return "SELECT * FROM tmp";
    }
    if (totalNumDeletePositions >= totalNumRowsInAllBaseFiles) {
      return "SELECT * FROM tmp WHERE 1 = 0";
    }

    std::vector<int64_t> allDeleteValues;
    int64_t numRowsInPreviousBaseFiles = 0;
    for (auto& baseFileSize : baseFileSizes) {
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
    for (auto& vec : deletePositionVectors) {
      for (auto pos : vec) {
        if (pos >= 0 && pos < baseFileSize) {
          deletePositionVector.push_back(pos);
        }
      }
    }
    std::sort(deletePositionVector.begin(), deletePositionVector.end());
    auto last =
        std::unique(deletePositionVector.begin(), deletePositionVector.end());
    deletePositionVector.erase(last, deletePositionVector.end());
    return deletePositionVector;
  }

  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();
};

/// Basic read without any deletes.
TEST_F(CudfIcebergReadTest, basicRead) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfIcebergConnectorId,
          "iceberg_table",
          facebook::velox::common::SubfieldFilters{},
          nullptr,
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read with multiple columns.
TEST_F(CudfIcebergReadTest, multiColumn) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto data = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  });
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read a larger file to verify chunked reading works.
TEST_F(CudfIcebergReadTest, largerFile) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto values = makeContinuousIncreasingValues(0, 10000);
  auto data = makeRowVector({makeFlatVector<int64_t>(values)});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, splits, "SELECT * FROM tmp", 0);
}

/// Read with multiple data files (multiple splits).
TEST_F(CudfIcebergReadTest, multipleSplits) {
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto splits1 = makeIcebergSplits(filePath1->getPath());
  auto splits2 = makeIcebergSplits(filePath2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      allSplits;
  allSplits.insert(allSplits.end(), splits1.begin(), splits1.end());
  allSplits.insert(allSplits.end(), splits2.begin(), splits2.end());

  createDuckDbTable({data1, data2});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, allSplits, "SELECT * FROM tmp", 0);
}

/// All  missing (schema evolution) columns
TEST_F(CudfIcebergReadTest, allSchemaEvolutionColumns) {
  auto dataVector = makeRowVector(
      {"old_col"},
      {
          makeFlatVector<int64_t>({100, 200, 300}),
      });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  auto newRowType = ROW({"new_col"}, {BIGINT()});
  auto expected = makeRowVector(
      {"new_col"},
      {
          makeNullConstant(TypeKind::BIGINT, 3),
      });

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(newRowType)
                  .dataColumns(newRowType)
                  .endTableScan()
                  .planNode();

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({expected});

  // Filtering a schema-evolution column that is NULL for every row in an old
  // file must return an empty result (Iceberg NULL semantics).
  auto filteredPlan = PlanBuilder()
                          .startTableScan()
                          .connectorId(kCudfIcebergConnectorId)
                          .outputType(newRowType)
                          .dataColumns(newRowType)
                          .subfieldFilter("new_col = 1")
                          .endTableScan()
                          .planNode();

  auto filteredEmptyExpected = makeRowVector(
      {"new_col"},
      {
          makeFlatVector<int64_t>(std::vector<int64_t>{}),
      });

  AssertQueryBuilder(filteredPlan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({filteredEmptyExpected});

  // Filter-only evolution col (not in output) with a projected file column.
  auto mixedType = ROW({"old_col", "new_col"}, {BIGINT(), BIGINT()});
  auto fileOnlyOutput = ROW({"old_col"}, {BIGINT()});
  auto filterOnlyPlan = PlanBuilder()
                            .startTableScan()
                            .connectorId(kCudfIcebergConnectorId)
                            .outputType(fileOnlyOutput)
                            .dataColumns(mixedType)
                            .subfieldFilter("new_col = 1")
                            .endTableScan()
                            .planNode();

  AssertQueryBuilder(filterOnlyPlan)
      .splits(makeIcebergSplits(dataFilePath->getPath()))
      .assertResults({makeRowVector(
          {"old_col"}, {makeFlatVector<int64_t>(std::vector<int64_t>{})})});
}

/// Column alias: output column `a` -> physical file column `c0`
TEST_F(CudfIcebergReadTest, columnAliasUsesPhysicalFileName) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({10, 20, 30}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto fileType = ROW({"c0"}, {BIGINT()});
  auto outputType = ROW({"a"}, {BIGINT()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(fileType)
                  .columnAliases({{"a", "c0"}})
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"a"},
      {
          makeFlatVector<int64_t>({10, 20, 30}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath()))
      .assertResults({expected});
}

/// A nonempty data file in a NULL partition must return one NULL partition
/// value per row
TEST_F(CudfIcebergReadTest, nullPartitionColumn) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", std::nullopt}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"c0", "country"},
      {
          data->childAt(0),
          makeNullConstant(TypeKind::VARCHAR, 3),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});
}

/// A projection containing only injected columns. e.g., partition column, leads
/// to no columns being read from the data file and the output is synthesized
/// from the injected columns only
TEST_F(CudfIcebergReadTest, partitionOnlyProjection) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"country"}, {VARCHAR()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"country"},
      {
          makeFlatVector<std::string>({"US", "US", "US"}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});

  // Filter `country = 'CA'` excludes the partition value 'US', so the output is
  // an empty table. Reader is bypassed due to no physical columns to read.
  auto filteredNoMatchPlan = PlanBuilder()
                                 .startTableScan()
                                 .connectorId(kCudfIcebergConnectorId)
                                 .outputType(outputType)
                                 .dataColumns(tableType)
                                 .assignments(assignments)
                                 .subfieldFilter("country = 'CA'")
                                 .endTableScan()
                                 .planNode();

  auto emptyExpected =
      makeRowVector({"country"}, {makeFlatVector<std::string>({})});

  AssertQueryBuilder(filteredNoMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({emptyExpected});

  // A matching subfield filter keeps all rows.
  auto filteredMatchPlan = PlanBuilder()
                               .startTableScan()
                               .connectorId(kCudfIcebergConnectorId)
                               .outputType(outputType)
                               .dataColumns(tableType)
                               .assignments(assignments)
                               .subfieldFilter("country = 'US'")
                               .endTableScan()
                               .planNode();

  AssertQueryBuilder(filteredMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});

  // Mixed physical + injected projection with a non-matching filter on the
  // injected partition column must return no rows. The filter cannot be pushed
  // into the parquet reader because `country` is not a file column.
  auto mixedAssignments = assignments;
  mixedAssignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});

  auto mixedPlan = PlanBuilder()
                       .startTableScan()
                       .connectorId(kCudfIcebergConnectorId)
                       .outputType(tableType)
                       .dataColumns(tableType)
                       .assignments(mixedAssignments)
                       .subfieldFilter("country = 'CA'")
                       .endTableScan()
                       .planNode();

  auto mixedEmptyExpected = makeRowVector(
      {"c0", "country"},
      {
          makeFlatVector<int64_t>(std::vector<int64_t>{}),
          makeFlatVector<std::string>(std::vector<std::string>{}),
      });

  AssertQueryBuilder(mixedPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({mixedEmptyExpected});

  // Matching filter on the injected column with a physical column present.
  auto mixedMatchPlan = PlanBuilder()
                            .startTableScan()
                            .connectorId(kCudfIcebergConnectorId)
                            .outputType(tableType)
                            .dataColumns(tableType)
                            .assignments(mixedAssignments)
                            .subfieldFilter("country = 'US'")
                            .endTableScan()
                            .planNode();

  auto mixedMatchExpected = makeRowVector(
      {"c0", "country"},
      {
          data->childAt(0),
          makeFlatVector<std::string>({"US", "US", "US"}),
      });

  AssertQueryBuilder(mixedMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({mixedMatchExpected});

  // Filter-only partition (not in output): inject, filter, then strip.
  auto fileOnlyAssignments = facebook::velox::connector::ColumnHandleMap{};
  fileOnlyAssignments["c0"] = mixedAssignments["c0"];
  auto fileOnlyOutput = ROW({"c0"}, {BIGINT()});

  auto filterOnlyMatchPlan = PlanBuilder()
                                 .startTableScan()
                                 .connectorId(kCudfIcebergConnectorId)
                                 .outputType(fileOnlyOutput)
                                 .dataColumns(tableType)
                                 .assignments(fileOnlyAssignments)
                                 .subfieldFilter("country = 'US'")
                                 .endTableScan()
                                 .planNode();

  AssertQueryBuilder(filterOnlyMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({makeRowVector({"c0"}, {data->childAt(0)})});

  auto filterOnlyNoMatchPlan = PlanBuilder()
                                   .startTableScan()
                                   .connectorId(kCudfIcebergConnectorId)
                                   .outputType(fileOnlyOutput)
                                   .dataColumns(tableType)
                                   .assignments(fileOnlyAssignments)
                                   .subfieldFilter("country = 'CA'")
                                   .endTableScan()
                                   .planNode();

  AssertQueryBuilder(filterOnlyNoMatchPlan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({makeRowVector(
          {"c0"}, {makeFlatVector<int64_t>(std::vector<int64_t>{})})});

  // Experimental hybrid reader must also support injected-only projections.
  AssertQueryBuilder(plan)
      .connectorSessionProperty(
          kCudfIcebergConnectorId,
          cudf_velox::connector::hive::CudfHiveConfig::
              kUseExperimentalCudfReaderSession,
          "true")
      .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
      .assertResults({expected});
}

/// A remaining filter on an unprojected column (`c1`) adds `c1` to the
/// projection for post-scan filtering and removal
TEST_F(CudfIcebergReadTest, unprojectedRemainingFilterColumn) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto dataType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c0"}, {BIGINT()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(dataType)
                  .remainingFilter("c1 % 2 = 0")
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({20, 40}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath()))
      .assertResults({expected});
}

/// Remaining filter referencing an injected filter-only partition column and an
/// unprojected file column.
TEST_F(CudfIcebergReadTest, remainingFilterOnInjectedColumn) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType =
      ROW({"c0", "c1", "country"}, {BIGINT(), BIGINT(), VARCHAR()});
  auto outputType = ROW({"c0"}, {BIGINT()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["c1"] = std::make_shared<HiveColumnHandle>(
      "c1",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .remainingFilter("country = 'CA' OR c1 % 2 = 0")
                  .endTableScan()
                  .planNode();

  // Partition 'US': `country = 'CA'` is false, so the filter reduces to
  // `c1 % 2 = 0`, keeping c1={2,4} -> c0={20,40}.
  std::unordered_map<std::string, std::optional<std::string>> usPartition = {
      {"country", "US"}};
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, usPartition))
      .assertResults(
          {makeRowVector({"c0"}, {makeFlatVector<int64_t>({20, 40})})});

  // Partition 'CA': `country = 'CA'` is true for every row, so all rows pass.
  std::unordered_map<std::string, std::optional<std::string>> caPartition = {
      {"country", "CA"}};
  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {}, caPartition))
      .assertResults(
          {makeRowVector({"c0"}, {makeFlatVector<int64_t>({10, 20, 30, 40})})});
}

/// Injected-only projection with positional deletes: surviving row count must
/// reflect deletes (via rowCountOverride), then deferred partition filter.
TEST_F(CudfIcebergReadTest, partitionOnlyProjectionWithPositionalDeletes) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto deleteFilePath = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return dataFile->getPath(); }),
              makeFlatVector<int64_t>({0, 2}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()));

  auto tableType = ROW({"c0", "country"}, {BIGINT(), VARCHAR()});
  auto outputType = ROW({"country"}, {VARCHAR()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["country"] = std::make_shared<HiveColumnHandle>(
      "country",
      HiveColumnHandle::ColumnType::kPartitionKey,
      VARCHAR(),
      VARCHAR(),
      std::vector<common::Subfield>{});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"}};

  // Delete rows 0 and 2 → 2 surviving "US" rows.
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .subfieldFilter("country = 'US'")
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"country"},
      {
          makeFlatVector<std::string>({"US", "US"}),
      });

  AssertQueryBuilder(plan)
      .splits(
          makeIcebergSplits(dataFile->getPath(), {deleteFile}, partitionKeys))
      .assertResults({expected});
}

// A timezone-less TIMESTAMP partition value is
// interpreted in the default timezone and converted to UTC when the Hive config
// is true.
TEST_F(CudfIcebergReadTest, timestampPartitionHonorsLocalTimeSetting) {
  auto data = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({1, 2, 3}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  auto tableType = ROW({"c0", "pkey"}, {BIGINT(), TIMESTAMP()});

  facebook::velox::connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0",
      HiveColumnHandle::ColumnType::kRegular,
      BIGINT(),
      BIGINT(),
      std::vector<common::Subfield>{});
  assignments["pkey"] = std::make_shared<HiveColumnHandle>(
      "pkey",
      HiveColumnHandle::ColumnType::kPartitionKey,
      TIMESTAMP(),
      TIMESTAMP(),
      std::vector<common::Subfield>{});

  const std::string partitionValue = "2023-10-14 07:00:00";
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"pkey", partitionValue}};

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableType)
                  .dataColumns(tableType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto utcTimestamp =
      util::fromTimestampString(
          StringView(partitionValue), util::TimestampParseMode::kPrestoCast)
          .value();

  auto localTimestamp = utcTimestamp;
  localTimestamp.toGMT(Timestamp::defaultTimezone());

  auto assertValue = [&](bool readAsLocalTime, Timestamp expectedTimestamp) {
    auto expected = makeRowVector(
        {"c0", "pkey"},
        {
            data->childAt(0),
            makeFlatVector<Timestamp>(
                {expectedTimestamp, expectedTimestamp, expectedTimestamp}),
        });

    AssertQueryBuilder(plan)
        .connectorSessionProperty(
            kCudfIcebergConnectorId,
            facebook::velox::connector::hive::HiveConfig::
                kReadTimestampPartitionValueAsLocalTimeSession,
            readAsLocalTime ? "true" : "false")
        .splits(makeIcebergSplits(dataFile->getPath(), {}, partitionKeys))
        .assertResults({expected});
  };

  // Interpret the partition string as already being in UTC.
  assertValue(false, utcTimestamp);

  // Interpret the partition string as local time and convert it to UTC.
  assertValue(true, localTimestamp);
}

/// A remaining filter is allowed with positional deletes
TEST_F(CudfIcebergReadTest, remainingFilterAndPositionalDeletes) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>({10, 20, 30, 40}),
          makeFlatVector<int64_t>({1, 2, 3, 4}),
      });

  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  // Positional delete file removing row 1 (c0=20, c1=2).
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto deleteVector = makeRowVector(
      {pathColumn->name, posColumn->name},
      {
          makeFlatVector<std::string>(
              1, [&](vector_size_t) { return dataFile->getPath(); }),
          makeFlatVector<int64_t>({1}),
      });
  writeDeleteFile(
      DeleteFileFormat::DWRF, deleteFilePath->getPath(), {deleteVector});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(deleteFilePath->getPath()));

  auto dataType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c0"}, {BIGINT()});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(dataType)
                  .remainingFilter("c1 % 2 = 0")
                  .endTableScan()
                  .planNode();

  // Row 1 (c1=2) is deleted; of the remaining c1={1,3,4}, `c1 % 2 = 0` keeps
  // c1=4, i.e. c0=40.
  auto expected = makeRowVector(
      {"c0"},
      {
          makeFlatVector<int64_t>({40}),
      });

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {deleteFile}))
      .assertResults({expected});
}

// Subfield filter + positional deletes: filter is deferred until after
// deletes so file positions stay valid.
TEST_F(CudfIcebergReadTest, subfieldFilterAndPositionalDeletes) {
  auto data = makeRowVector({makeFlatVector<int64_t>({10, 20, 30, 40})});
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  // Positional delete file removing row 0 (value 10).
  auto deleteFilePath = TempFilePath::create();
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto deleteVector = makeRowVector(
      {pathColumn->name, posColumn->name},
      {
          makeFlatVector<std::string>(
              1, [&](vector_size_t) { return dataFile->getPath(); }),
          makeFlatVector<int64_t>({0}),
      });
  writeDeleteFile(
      DeleteFileFormat::DWRF, deleteFilePath->getPath(), {deleteVector});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(deleteFilePath->getPath()));

  // `c0 < 30` is a simple range filter pushed as a subfield filter. With
  // positional deletes present it is deferred: deletes run first on full
  // file positions, then the filter. Rows: {10,20,30,40} → delete 10 →
  // {20,30,40} → filter → {20}.
  auto dataType = ROW({"c0"}, {BIGINT()});
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(dataType)
                  .dataColumns(dataType)
                  .subfieldFilter("c0 < 30")
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({20})});

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {deleteFile}))
      .assertResults({expected});
}

// A subfield filter is allowed with equality deletes
TEST_F(CudfIcebergReadTest, subfieldFilterAndEqualityDeletes) {
  auto data = makeRowVector({makeFlatVector<int64_t>({10, 20, 30, 40})});
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), data);

  // Equality delete file removing rows where c0=20.
  auto deleteData = makeRowVector({"c0"}, {makeFlatVector<int64_t>({20})});
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::DWRF, eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile deleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto dataType = ROW({"c0"}, {BIGINT()});
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(dataType)
                  .dataColumns(dataType)
                  .subfieldFilter("c0 < 35")
                  .endTableScan()
                  .planNode();

  // `c0 < 35` keeps {10, 20, 30}; the equality delete then removes c0=20,
  // leaving {10, 30}.
  auto expected = makeRowVector({makeFlatVector<int64_t>({10, 30})});

  AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(dataFile->getPath(), {deleteFile}))
      .assertResults({expected});
}

TEST_F(CudfIcebergReadTest, singleBaseFileSinglePositionalDeleteFile) {
  assertSingleBaseFileSingleDeleteFile({{0, 1, 2, 3}});
  assertSingleBaseFileSingleDeleteFile({{0, 9999, 10000, 19999}});
  assertSingleBaseFileSingleDeleteFile({{10000, 10002, 19999}});
  assertSingleBaseFileSingleDeleteFile({makeRandomIncreasingValues(0, 20000)});
  assertSingleBaseFileSingleDeleteFile({});
  assertSingleBaseFileSingleDeleteFile(
      {makeContinuousIncreasingValues(0, 20000)});
  assertSingleBaseFileSingleDeleteFile({{20000, 29999}});
}

TEST_F(CudfIcebergReadTest, multipleBaseFilesSinglePositionalDeleteFile) {
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

TEST_F(CudfIcebergReadTest, singleBaseFileMultiplePositionalDeleteFiles) {
  assertSingleBaseFileMultipleDeleteFiles({{1}, {2}, {3}, {4}});
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

TEST_F(CudfIcebergReadTest, multipleBaseFileMultiplePositionalDeleteFiles) {
  std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles;
  std::unordered_map<
      std::string,
      std::multimap<std::string, std::vector<int64_t>>>
      deleteFilesForBaseDatafiles;

  rowGroupSizesForFiles["data_file_1"] = {100, 85};
  rowGroupSizesForFiles["data_file_2"] = {99, 1};

  deleteFilesForBaseDatafiles["delete_file_1"] = {{"data_file_1", {0, 1, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {100, 101, 184}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeContinuousIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(186, 300)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 100, 102, 184}}, {"data_file_2", {1, 98, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}},
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", {1, 3, 5, 7}},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}}, {"data_file_2", {1, 3, 5, 7}}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", {98, 99, 100, 101, 184}},
      {"data_file_2", {3, 5, 7, 9}},
      {"data_file_2", {98, 99, 100}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", makeRandomIncreasingValues(10, 120)},
      {"data_file_2", makeRandomIncreasingValues(50, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
}

// TODO(mh): This test assumes the file is chunked across multiple splits.
#if 0
TEST_F(CudfIcebergReadTest, positionalDeletesMultipleSplits) {

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
#endif

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberApplied) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/10);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/5);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberSkipped) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/10);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

// Subfield filter + positional delete skipped by sequence-number resolution:
// no delete readers remain, so the filter can still be pushed into parquet.
TEST_F(CudfIcebergReadTest, subfieldFilterWithSkippedPositionalDelete) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  // Positional delete with dataSequenceNumber (5) < the data file's sequence
  // number (10), so it is skipped by sequence-number resolution.
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/10);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .subfieldFilter("c0 < 4")
                  .endTableScan()
                  .planNode();

  // The delete is skipped, so `c0 < 4` simply keeps {0, 1, 2, 3}.
  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

// Subfield filter + positional delete for a different data file. Filter is
// deferred until cuDF PR #23077 merges.
TEST_F(
    CudfIcebergReadTest,
    subfieldFilterWithPositionalDeleteForDifferentFile) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto baseFilePath = dataFilePath->getPath();
  auto otherDataFilePath = TempFilePath::create();

  // The delete file's positions reference a different data file, so its
  // file_path column statistics exclude `baseFilePath` and `testFilters` prunes
  // the reader during construction.
  auto deleteFilePath = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2,
                  [&](vector_size_t) { return otherDataFilePath->getPath(); }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  // Equal sequence numbers make the positional delete applicable, ensuring it
  // reaches file-path pruning inside `PositionalDeleteFileReader`.
  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/10);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/10);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .subfieldFilter("c0 < 4")
                  .endTableScan()
                  .planNode();

  // The positional delete file contains no entries for this data file, so
  // `c0 < 4` simply keeps {0, 1, 2, 3}.
  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberEqualApplied) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/5);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, positionalDeleteSequenceNumberZeroDisablesFilter) {
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto dataFilePath = TempFilePath::create();
  writeToFile(
      dataFilePath->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})})});

  auto deleteFilePath = TempFilePath::create();
  auto baseFilePath = dataFilePath->getPath();
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      deleteFilePath->getPath(),
      {makeRowVector(
          {pathColumn->name, posColumn->name},
          {
              makeFlatVector<std::string>(
                  2, [&](vector_size_t) { return baseFilePath; }),
              makeFlatVector<int64_t>({1, 3}),
          })});

  IcebergDeleteFile deleteFile(
      FileContent::kPositionalDeletes,
      deleteFilePath->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(deleteFilePath->getPath()),
      {},
      {},
      {},
      /*dataSequenceNumber=*/0);

  auto splits = makeIcebergSplits(
      baseFilePath,
      {deleteFile},
      {},
      1,
      /*dataSequenceNumber=*/100);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 2, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Edge-case tests from Sirius project

/// Combined positional + equality deletes on the same data file.
/// Verifies that positional deletes (position-based) and equality deletes
/// (value-based) compose correctly when applied sequentially.
TEST_F(CudfIcebergReadTest, combinedPositionalAndEqualityDeletes) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Positional delete: remove rows at positions 1 and 3 (c0=1, c0=3)
  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      2, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({1, 3});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posIcebergDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(posDeleteFile->getPath()));

  // Equality delete: remove rows where c0={5, 7}
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({5, 7}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {posIcebergDelete, eqIcebergDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // Positional removes pos 1,3 (c0=1,3). Equality removes c0=5,7.
  // Surviving: 0, 2, 4, 6, 8, 9
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 2, 4, 6, 8, 9}),
      makeFlatVector<int64_t>({10, 12, 14, 16, 18, 19}),
  });

  assertEqualResults({expected}, {result});
}

/// Non-projected equality delete key column.
/// Query selects only c1, but equality delete key is c0.
/// The reader must force-project c0 for filtering then strip it from output.
TEST_F(CudfIcebergReadTest, nonProjectedDeleteKeyColumn) {
  auto fullType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto outputType = ROW({"c1"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  // Equality delete on c0 (NOT in the output projection)
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqIcebergDelete});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(outputType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  // c0=2,4 deleted -> c1=20,40 removed
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Insert-delete-insert interleaving: data written after a delete (higher
/// sequence number) should NOT be affected by that delete.
TEST_F(CudfIcebergReadTest, insertDeleteInsertInterleaving) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  // File 1: original data (seq=1)
  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  // File 2: data written AFTER the delete (seq=3)
  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({200, 400}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  // Equality delete at seq=2: delete c0=2
  auto eqDeleteData = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(
      DeleteFileFormat::PARQUET, eqDeleteFile->getPath(), {eqDeleteData});

  IcebergDeleteFile eqIcebergDelete(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  // File 1: data seq=1, delete seq=2 -> delete APPLIES (2 > 1)
  auto splits1 = makeIcebergSplits(
      dataFile1->getPath(), {eqIcebergDelete}, {}, 1, /*dataSeq=*/1);
  // File 2: data seq=3, delete seq=2 -> delete SKIPPED (2 < 3)
  auto splits2 = makeIcebergSplits(
      dataFile2->getPath(), {eqIcebergDelete}, {}, 1, /*dataSeq=*/3);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      allSplits;
  allSplits.insert(allSplits.end(), splits1.begin(), splits1.end());
  allSplits.insert(allSplits.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(allSplits).copyResults(pool());

  // File1 loses c0=2 (delete applies), file2 keeps c0=2 (delete skipped)
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 2, 4}),
      makeFlatVector<int64_t>({10, 30, 200, 400}),
  });

  assertEqualResults({expected}, {result});
}

TEST_F(CudfIcebergReadTest, schemaEvolutionRemoveColumn) {
  auto oldRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), INTEGER(), VARCHAR()});
  auto newRowType = ROW({"c0", "c2"}, {BIGINT(), VARCHAR()});

  // Write data file with old schema (c0, c1, c2).
  auto dataVector = makeRowVector(
      oldRowType->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50}),
          makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
      });

  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  auto icebergSplits = makeIcebergSplits(dataFilePath->getPath());

  // Expected result: c0 and c2 have values, c1 is not present.
  auto expected = makeRowVector(
      newRowType->names(),
      {
          dataVector->childAt(0),
          dataVector->childAt(2),
      });

  // Read with new schema (c0 and c2 only, c1 removed).
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(newRowType)
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan).splits(icebergSplits).assertResults({expected});
}

TEST_F(CudfIcebergReadTest, schemaEvolutionAddColumns) {
  auto oldRowType = ROW({"c0"}, {BIGINT()});
  auto newRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), INTEGER(), VARCHAR()});

  // Write data file with old schema (only c0).
  auto dataVector = makeRowVector({
      makeFlatVector<int64_t>({100, 200, 300}),
  });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);
  auto icebergSplits = makeIcebergSplits(dataFilePath->getPath());

  // Expected result: c0 has values, c1 and c2 are NULL.
  auto expected = makeRowVector({
      dataVector->childAt(0),
      makeNullConstant(TypeKind::INTEGER, 3),
      makeNullConstant(TypeKind::VARCHAR, 3),
  });

  // Read with new schema (c0, c1, and c2).
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(newRowType)
                  .dataColumns(newRowType)
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan).splits(icebergSplits).assertResults({expected});
}

// Hive-migrated partition columns: values come from the partitionKeys map,
// not the data file, and are injected by adaptColumns.
TEST_F(CudfIcebergReadTest, partitionColumnsFromHive) {
  auto fileRowType = ROW({"c0", "c1"}, {BIGINT(), INTEGER()});
  auto tableRowType =
      ROW({"c0", "c1", "region", "year"},
          {BIGINT(), INTEGER(), VARCHAR(), INTEGER()});

  // Write data file with only non-partition columns (c0, c1).
  auto dataVector = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({10, 20, 30}),
  });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  // Set partition keys for region and year.
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  partitionKeys["region"] = "US";
  partitionKeys["year"] = "2025";

  auto icebergSplits =
      makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);

  // Build column handles marking partition columns.
  facebook::velox::connector::ColumnHandleMap assignments;
  for (uint32_t i = 0; i < tableRowType->size(); ++i) {
    const auto& name = tableRowType->nameOf(i);
    auto columnType = (i >= fileRowType->size())
        ? HiveColumnHandle::ColumnType::kPartitionKey
        : HiveColumnHandle::ColumnType::kRegular;
    assignments[name] = std::make_shared<HiveColumnHandle>(
        name,
        columnType,
        tableRowType->childAt(i),
        tableRowType->childAt(i),
        std::vector<common::Subfield>{});
  }

  // Expected result: c0 and c1 from file, region and year from partition keys.
  auto expected = makeRowVector(
      tableRowType->names(),
      {
          dataVector->childAt(0),
          dataVector->childAt(1),
          makeFlatVector<std::string>({"US", "US", "US"}),
          makeFlatVector<int32_t>({2025, 2025, 2025}),
      });

  // Read with table schema including partition columns.
  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableRowType)
                  .dataColumns(tableRowType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();
  AssertQueryBuilder(plan).splits(icebergSplits).assertResults({expected});
}

// Test reading a DATE identity partition column. DATE partition values arrive
// in two encodings: Hive-migrated tables store a date string ("2025-06-05")
// while Iceberg-native tables store days-since-epoch ("20244"). Both must
// decode to the same DATE result.
TEST_F(CudfIcebergReadTest, partitionColumnsDate) {
  auto fileRowType = ROW({"c0"}, {BIGINT()});
  auto tableRowType = ROW({"c0", "partitiondate"}, {BIGINT(), DATE()});

  // Write a data file with only the non-partition column c0.
  auto dataVector = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFilePath = TempFilePath::create();
  writeToFile(dataFilePath->getPath(), dataVector);

  // 2025-06-05 is 20244 days since the Unix epoch.
  const int32_t kDays = DATE()->toDays("2025-06-05");

  // Build column handles marking partitiondate as a partition key.
  facebook::velox::connector::ColumnHandleMap assignments;
  for (uint32_t i = 0; i < tableRowType->size(); ++i) {
    const auto& name = tableRowType->nameOf(i);
    auto columnType = (i >= fileRowType->size())
        ? HiveColumnHandle::ColumnType::kPartitionKey
        : HiveColumnHandle::ColumnType::kRegular;
    assignments[name] = std::make_shared<HiveColumnHandle>(
        name,
        columnType,
        tableRowType->childAt(i),
        tableRowType->childAt(i),
        std::vector<common::Subfield>{});
  }

  auto expected = makeRowVector(
      tableRowType->names(),
      {
          dataVector->childAt(0),
          makeFlatVector<int32_t>({kDays, kDays, kDays}, DATE()),
      });

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(tableRowType)
                  .dataColumns(tableRowType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  // Encoding 1: Hive-migrated date string.
  {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
    partitionKeys["partitiondate"] = "2025-06-05";
    auto splits = makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);
    AssertQueryBuilder(plan).splits(splits).assertResults({expected});
  }

  // Encoding 2: Iceberg-native days-since-epoch.
  {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
    partitionKeys["partitiondate"] = folly::to<std::string>(kDays);
    auto splits = makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);
    AssertQueryBuilder(plan).splits(splits).assertResults({expected});
  }
}

} // namespace facebook::velox::cudf_velox::exec::test
