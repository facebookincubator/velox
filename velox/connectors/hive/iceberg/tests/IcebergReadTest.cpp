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

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/Singleton.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::test;

namespace facebook::velox::connector::hive::iceberg {

class HiveIcebergTest : public HiveConnectorTestBase {
 public:
  void assertPositionalDeletes(
      const std::vector<std::vector<int64_t>>& deleteRowsVec,
      bool multipleBaseFiles = false) {
    assertPositionalDeletesInternal(
        deleteRowsVec, getQuery(deleteRowsVec), multipleBaseFiles, 1, 0);
  }

  void assertPositionalDeletes(
      const std::vector<std::vector<int64_t>>& deleteRowsVec,
      std::string duckdbSql,
      bool multipleBaseFiles = false) {
    assertPositionalDeletesInternal(
        deleteRowsVec, duckdbSql, multipleBaseFiles, 1, 0);
  }

  void assertPositionalDeletes(
      const std::vector<std::vector<int64_t>>& deleteRowsVec,
      std::string duckdbSql,
      int32_t splitCount,
      int32_t numPrefetchSplits) {
    assertPositionalDeletesInternal(
        deleteRowsVec, duckdbSql, true, splitCount, numPrefetchSplits);
  }

  std::vector<int64_t> makeRandomDeleteRows(int32_t maxRowNumber) {
    std::mt19937 gen{0};
    std::vector<int64_t> deleteRows;
    for (int i = 0; i < maxRowNumber; i++) {
      if (folly::Random::rand32(0, 10, gen) > 8) {
        deleteRows.push_back(i);
      }
    }
    return deleteRows;
  }

  std::vector<int64_t> makeSequenceRows(int32_t numRows, int8_t repeat = 1) {
    VELOX_CHECK_GT(repeat, 0);

    auto maxValue = std::ceil((double)numRows / repeat);
    std::vector<int64_t> values;
    values.reserve(numRows);
    for (int32_t i = 0; i < maxValue; i++) {
      for (int8_t j = 0; j < repeat; j++) {
        values.push_back(i);
      }
    }
    values.resize(numRows);
    return values;
  }

  std::string getQuery(const std::vector<std::vector<int64_t>>& deleteRowsVec) {
    return "SELECT * FROM tmp WHERE c0 NOT IN (" +
        makeNotInList(deleteRowsVec) + ")";
  }

  const static int rowCount = 20000;

 private:
  void assertPositionalDeletesInternal(
      const std::vector<std::vector<int64_t>>& deleteRowsVec,
      std::string duckdbSql,
      bool multipleBaseFiles,
      int32_t splitCount,
      int32_t numPrefetchSplits) {
    auto dataFilePaths = writeDataFiles(rowCount, 1, splitCount);
    std::vector<std::shared_ptr<ConnectorSplit>> splits;
    // Keep the reference to the deleteFilePath, otherwise the corresponding
    // file will be deleted.
    std::vector<std::shared_ptr<TempFilePath>> deleteFilePaths;
    for (const auto& dataFilePath : dataFilePaths) {
      std::mt19937 gen{0};
      int64_t numDeleteRowsBefore =
          multipleBaseFiles ? folly::Random::rand32(0, 1000, gen) : 0;
      int64_t numDeleteRowsAfter =
          multipleBaseFiles ? folly::Random::rand32(0, 1000, gen) : 0;
      std::vector<IcebergDeleteFile> deleteFiles;
      deleteFiles.reserve(deleteRowsVec.size());
      for (auto const& deleteRows : deleteRowsVec) {
        std::shared_ptr<TempFilePath> deleteFilePath = writePositionDeleteFile(
            dataFilePath->getPath(),
            deleteRows,
            numDeleteRowsBefore,
            numDeleteRowsAfter);
        auto path = deleteFilePath->getPath();
        IcebergDeleteFile deleteFile(
            FileContent::kPositionalDeletes,
            deleteFilePath->getPath(),
            fileFomat_,
            deleteRows.size() + numDeleteRowsBefore + numDeleteRowsAfter,
            testing::internal::GetFileSize(std::fopen(path.c_str(), "r")));
        deleteFilePaths.emplace_back(deleteFilePath);
        deleteFiles.emplace_back(deleteFile);
      }

      splits.emplace_back(
          makeIcebergSplit(dataFilePath->getPath(), deleteFiles));
    }

    auto plan = tableScanNode();
    auto task = HiveConnectorTestBase::assertQuery(
        plan, splits, duckdbSql, numPrefetchSplits);

    auto planStats = toPlanStats(task->taskStats());
    auto scanNodeId = plan->id();
    auto it = planStats.find(scanNodeId);
    ASSERT_TRUE(it != planStats.end());
    ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  }

  std::shared_ptr<ConnectorSplit> makeIcebergSplit(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {}) {
    std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
    std::unordered_map<std::string, std::string> customSplitInfo;
    customSplitInfo["table_format"] = "hive-iceberg";

    auto file = filesystems::getFileSystem(dataFilePath, nullptr)
                    ->openFileForRead(dataFilePath);
    const int64_t fileSize = file->size();

    return std::make_shared<HiveIcebergSplit>(
        kHiveConnectorId,
        dataFilePath,
        fileFomat_,
        0,
        fileSize,
        partitionKeys,
        std::nullopt,
        customSplitInfo,
        nullptr,
        deleteFiles);
  }

  std::vector<RowVectorPtr>
  makeVectors(int32_t count, int32_t rowsPerVector, int32_t numColumns = 1) {
    std::vector<TypePtr> types(numColumns, BIGINT());
    std::vector<std::string> names;
    for (int j = 0; j < numColumns; j++) {
      names.push_back(fmt::format("c{}", j));
    }

    std::vector<RowVectorPtr> rowVectors;
    for (int i = 0; i < count; i++) {
      std::vector<VectorPtr> vectors;

      // Create the column values like below:
      // c0 c1 c2
      //  0  0  0
      //  1  0  0
      //  2  1  0
      //  3  1  1
      //  4  2  1
      //  5  2  1
      //  6  3  2
      // ...
      // In the first column c0, the values are continuously increasing and not
      // repeating. In the second column c1, the values are continuously
      // increasing and each value repeats once. And so on.
      for (int j = 0; j < numColumns; j++) {
        auto data = makeSequenceRows(rowsPerVector, j + 1);
        vectors.push_back(vectorMaker_.flatVector<int64_t>(data));
      }

      rowVectors.push_back(makeRowVector(names, vectors));
    }

    rowType_ = std::make_shared<RowType>(std::move(names), std::move(types));

    return rowVectors;
  }

  std::vector<std::shared_ptr<TempFilePath>> writeDataFiles(
      uint64_t numRows,
      int32_t numColumns = 1,
      int32_t splitCount = 1) {
    auto dataVectors = makeVectors(splitCount, numRows, numColumns);
    VELOX_CHECK_EQ(dataVectors.size(), splitCount);

    std::vector<std::shared_ptr<TempFilePath>> dataFilePaths;
    dataFilePaths.reserve(splitCount);
    for (auto i = 0; i < splitCount; i++) {
      dataFilePaths.emplace_back(TempFilePath::create());
      writeToFile(dataFilePaths.back()->getPath(), dataVectors[i]);
    }

    createDuckDbTable(dataVectors);
    return dataFilePaths;
  }

  std::shared_ptr<TempFilePath> writePositionDeleteFile(
      const std::string& dataFilePath,
      const std::vector<int64_t>& deleteRows,
      int64_t numRowsBefore = 0,
      int64_t numRowsAfter = 0) {
    // if containsMultipleDataFiles == true, we will write rows for other base
    // files before and after the target base file
    uint32_t numDeleteRows = numRowsBefore + deleteRows.size() + numRowsAfter;

    std::string dataFilePathBefore = dataFilePath + "_before";
    std::string dataFilePathAfter = dataFilePath + "_after";

    auto filePathVector =
        vectorMaker_.flatVector<StringView>(numDeleteRows, [&](auto row) {
          if (row < numRowsBefore) {
            return StringView(dataFilePathBefore);
          } else if (
              row >= numRowsBefore && row < deleteRows.size() + numRowsBefore) {
            return StringView(dataFilePath);
          } else if (
              row >= deleteRows.size() + numRowsBefore && row < numDeleteRows) {
            return StringView(dataFilePathAfter);
          } else {
            return StringView();
          }
        });

    std::vector<int64_t> deleteRowsVec;
    deleteRowsVec.reserve(numDeleteRows);

    if (numRowsBefore > 0) {
      auto rowsBefore = makeSequenceRows(numRowsBefore);
      deleteRowsVec.insert(
          deleteRowsVec.end(), rowsBefore.begin(), rowsBefore.end());
    }
    deleteRowsVec.insert(
        deleteRowsVec.end(), deleteRows.begin(), deleteRows.end());
    if (numRowsAfter > 0) {
      auto rowsAfter = makeSequenceRows(numRowsAfter);
      deleteRowsVec.insert(
          deleteRowsVec.end(), rowsAfter.begin(), rowsAfter.end());
    }

    auto deletePositionsVector =
        vectorMaker_.flatVector<int64_t>(deleteRowsVec);
    RowVectorPtr deleteFileVectors = makeRowVector(
        {pathColumn_->name, posColumn_->name},
        {filePathVector, deletePositionsVector});

    auto deleteFilePath = TempFilePath::create();
    writeToFile(deleteFilePath->getPath(), deleteFileVectors);

    return deleteFilePath;
  }

  std::string makeNotInList(
      const std::vector<std::vector<int64_t>>& deleteRowsVec) {
    std::vector<int64_t> deleteRows;
    size_t totalSize = 0;
    for (const auto& subVec : deleteRowsVec) {
      totalSize += subVec.size();
    }
    deleteRows.reserve(totalSize);
    for (const auto& subVec : deleteRowsVec) {
      deleteRows.insert(deleteRows.end(), subVec.begin(), subVec.end());
    }

    if (deleteRows.empty()) {
      return "";
    }

    return std::accumulate(
        deleteRows.begin() + 1,
        deleteRows.end(),
        std::to_string(deleteRows[0]),
        [](const std::string& a, int64_t b) {
          return a + ", " + std::to_string(b);
        });
  }

  core::PlanNodePtr tableScanNode() {
    return PlanBuilder(pool_.get()).tableScan(rowType_).planNode();
  }

  dwio::common::FileFormat fileFomat_{dwio::common::FileFormat::DWRF};
  RowTypePtr rowType_{ROW({"c0"}, {BIGINT()})};
  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();
};

TEST_F(HiveIcebergTest, positionalDeletesSingleBaseFile) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertPositionalDeletes({{0, 1, 2, 3}});
  // Delete the first and last row in each batch (10000 rows per batch)
  assertPositionalDeletes({{0, 9999, 10000, 19999}});
  // Delete several rows in the second batch (10000 rows per batch)
  assertPositionalDeletes({{10000, 10002, 19999}});
  // Delete random rows
  assertPositionalDeletes({makeRandomDeleteRows(rowCount)});
  // Delete 0 rows
  assertPositionalDeletes({}, "SELECT * FROM tmp", false);
  // Delete all rows
  assertPositionalDeletes(
      {makeSequenceRows(rowCount)}, "SELECT * FROM tmp WHERE 1 = 0", false);
  // Delete rows that don't exist
  assertPositionalDeletes({{20000, 29999}});
}

// The positional delete file contains rows from multiple base files
TEST_F(HiveIcebergTest, positionalDeletesMultipleBaseFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertPositionalDeletes({{0, 1, 2, 3}}, true);
  // Delete the first and last row in each batch (10000 rows per batch)
  assertPositionalDeletes({{0, 9999, 10000, 19999}}, true);
  // Delete several rows in the second batch (10000 rows per batch)
  assertPositionalDeletes({{10000, 10002, 19999}}, true);
  // Delete random rows
  assertPositionalDeletes({makeRandomDeleteRows(rowCount)}, true);
  // Delete 0 rows
  assertPositionalDeletes({}, "SELECT * FROM tmp", true);
  // Delete all rows
  assertPositionalDeletes(
      {makeSequenceRows(rowCount)}, "SELECT * FROM tmp WHERE 1 = 0", true);
  // Delete rows that don't exist
  assertPositionalDeletes({{20000, 29999}}, true);
}

// The base file has multiple delete files.
TEST_F(HiveIcebergTest, baseFileMultiplePositionalDeletes) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertPositionalDeletes({{1}, {2}, {3}, {4}});
  // Delete the first and last row in each batch (10000 rows per batch).
  assertPositionalDeletes({{0}, {9999}, {10000}, {19999}});
}

TEST_F(HiveIcebergTest, positionalDeletesMultipleSplits) {
  folly::SingletonVault::singleton()->registrationComplete();
  constexpr int32_t splitCount = 50;
  constexpr int32_t numPrefetchSplits = 10;
  std::vector<std::vector<int64_t>> deletedRows = {{1}, {2}, {3, 4}};
  assertPositionalDeletes(
      deletedRows, getQuery(deletedRows), splitCount, numPrefetchSplits);
}

} // namespace facebook::velox::connector::hive::iceberg
