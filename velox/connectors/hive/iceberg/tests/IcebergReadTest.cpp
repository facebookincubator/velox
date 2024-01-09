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
      const std::vector<int64_t>& deleteRows,
      bool multipleBaseFiles = false) {
    std::string duckDbSql = "SELECT * FROM tmp";
    if (!deleteRows.empty()) {
      duckDbSql +=
          fmt::format(" WHERE c1 NOT IN ({})", makeNotInList(deleteRows));
    }

    assertPositionalDeletes(deleteRows, duckDbSql, multipleBaseFiles);
  }

  void assertPositionalDeletes(
      const std::vector<int64_t>& deleteRows,
      std::string duckDbSql,
      bool multipleBaseFiles = false) {
    std::shared_ptr<TempFilePath> dataFilePath = writeDataFile(1, kRowCount);

    std::mt19937 gen{0};
    int64_t numDeleteRowsBefore =
        multipleBaseFiles ? folly::Random::rand32(0, 1000, gen) : 0;
    int64_t numDeleteRowsAfter =
        multipleBaseFiles ? folly::Random::rand32(0, 1000, gen) : 0;
    std::shared_ptr<TempFilePath> deleteFilePath = writePositionDeleteFile(
        dataFilePath->path,
        deleteRows,
        numDeleteRowsBefore,
        numDeleteRowsAfter);

    IcebergDeleteFile deleteFile(
        FileContent::kPositionalDeletes,
        deleteFilePath->path,
        fileFomat_,
        deleteRows.size() + numDeleteRowsBefore + numDeleteRowsAfter,
        testing::internal::GetFileSize(
            std::fopen(deleteFilePath->path.c_str(), "r")));

    auto icebergSplit = makeIcebergSplit(dataFilePath->path, {deleteFile});

    auto plan = tableScanNode(rowType_);
    auto task = OperatorTestBase::assertQuery(plan, {icebergSplit}, duckDbSql);

    auto planStats = toPlanStats(task->taskStats());
    auto scanNodeId = plan->id();
    auto it = planStats.find(scanNodeId);
    ASSERT_TRUE(it != planStats.end());
    ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  }

  void assertEqualityDeletes(
      const std::vector<std::vector<int64_t>>& equalityDeleteVector,
      const std::vector<int32_t>& equalityFieldIds,
      std::string duckDbSql = "") {
    // We will create data vectors with numColumns number of columns that is the
    // max field Id in equalityFieldIds
    int32_t numDataColumns =
        *std::max_element(equalityFieldIds.begin(), equalityFieldIds.end());

    VELOX_CHECK_GT(numDataColumns, 0);
    VELOX_CHECK_GE(numDataColumns, equalityDeleteVector.size());
    VELOX_CHECK_GT(equalityDeleteVector.size(), 0);

    auto numDeletedValues = equalityDeleteVector[0].size();

    VELOX_CHECK_LE(equalityFieldIds.size(), numDataColumns);

    std::shared_ptr<TempFilePath> dataFilePath =
        writeDataFile(numDataColumns, kRowCount);

    std::shared_ptr<TempFilePath> deleteFilePath =
        writeEqualityDeleteFile(equalityDeleteVector);
    IcebergDeleteFile deleteFile(
        FileContent::kEqualityDeletes,
        deleteFilePath->path,
        fileFomat_,
        equalityDeleteVector[0].size(),
        testing::internal::GetFileSize(
            std::fopen(deleteFilePath->path.c_str(), "r")),
        equalityFieldIds);

    auto icebergSplit = makeIcebergSplit(dataFilePath->path, {deleteFile});

    std::string predicates =
        makePredicates(equalityDeleteVector, equalityFieldIds);

    // Select all columns
    duckDbSql = "SELECT * FROM tmp";
    if (numDeletedValues > 0) {
      duckDbSql += fmt::format(" WHERE {}", predicates);
    }

    assertEqualityDeletes(icebergSplit, rowType_, duckDbSql);

    // Select a column that's not in the filter columns
    if (numDataColumns > 1 && equalityDeleteVector.size() < numDataColumns) {
      //      if (inputDuckDbSql.empty()) {
      std::string duckDbSql = "SELECT c1 FROM tmp";
      if (numDeletedValues > 0) {
        duckDbSql += fmt::format(" WHERE {}", predicates);
      }

      std::vector<std::string> names({"c1"});
      std::vector<TypePtr> types(1, BIGINT());
      assertEqualityDeletes(
          icebergSplit,
          std::make_shared<RowType>(std::move(names), std::move(types)),
          duckDbSql);
    }
  }

  void assertEqualityDeletes(
      std::shared_ptr<connector::ConnectorSplit> split,
      RowTypePtr outputRowType,
      const std::string& duckDbSql) {
    auto plan = tableScanNode(outputRowType);
    auto task = OperatorTestBase::assertQuery(plan, {split}, duckDbSql);

    auto planStats = toPlanStats(task->taskStats());
    auto scanNodeId = plan->id();
    auto it = planStats.find(scanNodeId);
    ASSERT_TRUE(it != planStats.end());
    ASSERT_TRUE(it->second.peakMemoryBytes > 0);
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

  const static int64_t kRowCount = 20000;

 private:
  std::shared_ptr<connector::ConnectorSplit> makeIcebergSplit(
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

  std::vector<RowVectorPtr> makeVectors(
      int32_t numColumns,
      int32_t rowsPerVector) {
    std::vector<VectorPtr> vectors;
    std::vector<std::string> names;

    // c0 c1 c2
    //  0  0  0
    //  1  0  0
    //  2  1  0
    //  3  1  1
    //  4  2  1
    //  5  2  1
    //  6  3  2
    for (int i = 0; i < numColumns; i++) {
      auto data = makeSequenceRows(rowsPerVector, i + 1);
      vectors.push_back(vectorMaker_.flatVector<int64_t>(data));
      names.push_back(fmt::format("c{}", i + 1));
    }

    std::vector<RowVectorPtr> rowVectors;
    rowVectors.push_back(makeRowVector(names, vectors));

    std::vector<TypePtr> types(numColumns, BIGINT());
    rowType_ = std::make_shared<RowType>(std::move(names), std::move(types));
    return rowVectors;
  }

  std::shared_ptr<TempFilePath> writeDataFile(
      uint64_t numColumns,
      uint64_t numRows) {
    auto dataVectors = makeVectors(numColumns, numRows);

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->path, dataVectors);
    createDuckDbTable(dataVectors);
    return dataFilePath;
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
    writeToFile(deleteFilePath->path, deleteFileVectors);

    return deleteFilePath;
  }

  std::shared_ptr<TempFilePath> writeEqualityDeleteFile(
      const std::vector<std::vector<int64_t>>& equalityDeleteVector) {
    std::vector<std::string> names;
    std::vector<VectorPtr> vectors;
    for (int i = 0; i < equalityDeleteVector.size(); i++) {
      names.push_back(fmt::format("c{}", i + 1));
      vectors.push_back(
          vectorMaker_.flatVector<int64_t>(equalityDeleteVector[i]));
    }

    RowVectorPtr deleteFileVectors = makeRowVector(names, vectors);

    auto deleteFilePath = TempFilePath::create();
    writeToFile(deleteFilePath->path, deleteFileVectors);

    return deleteFilePath;
  }

  std::string makeNotInList(const std::vector<int64_t>& deleteRows) {
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

  std::string makePredicates(
      const std::vector<std::vector<int64_t>>& equalityDeleteVector,
      const std::vector<int32_t>& equalityFieldIds) {
    std::string predicates("");
    int32_t numDataColumns =
        *std::max_element(equalityFieldIds.begin(), equalityFieldIds.end());

    VELOX_CHECK_GT(numDataColumns, 0);
    VELOX_CHECK_GE(numDataColumns, equalityDeleteVector.size());
    VELOX_CHECK_GT(equalityDeleteVector.size(), 0);

    auto numDeletedValues = equalityDeleteVector[0].size();

    if (numDeletedValues == 0) {
      return predicates;
    }

    // If all values for a column are deleted, just return an always-false
    // predicate
    for (auto i = 0; i < equalityDeleteVector.size(); i++) {
      auto equalityFieldId = equalityFieldIds[i];
      auto deleteValues = equalityDeleteVector[i];

      auto lastIter = std::unique(deleteValues.begin(), deleteValues.end());
      auto numDistinctValues = lastIter - deleteValues.begin();
      auto minValue = *std::min_element(deleteValues.begin(), lastIter);
      auto maxValue = *std::max_element(deleteValues.begin(), lastIter);
      if (maxValue - minValue + 1 == numDistinctValues &&
          maxValue == (kRowCount - 1) / equalityFieldId) {
        return "1 = 0";
      }
    }

    if (equalityDeleteVector.size() == 1) {
      std::string name = fmt::format("c{}", equalityFieldIds[0]);
      predicates = fmt::format(
          "{} NOT IN ({})", name, makeNotInList(equalityDeleteVector[0]));
    } else {
      for (int i = 0; i < numDeletedValues; i++) {
        std::string oneRow("");
        for (int j = 0; j < equalityFieldIds.size(); j++) {
          std::string name = fmt::format("c{}", equalityFieldIds[j]);
          std::string predicate =
              fmt::format("({} <> {})", name, equalityDeleteVector[j][i]);

          oneRow = oneRow == "" ? predicate
                                : fmt::format("({} OR {})", oneRow, predicate);
        }

        predicates = predicates == ""
            ? oneRow
            : fmt::format("{} AND {}", predicates, oneRow);
      }
    }
    return predicates;
  }

  std::shared_ptr<exec::Task> assertQuery(
      const core::PlanNodePtr& plan,
      std::shared_ptr<TempFilePath> dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles,
      const std::string& duckDbSql) {
    auto icebergSplit = makeIcebergSplit(dataFilePath->path, deleteFiles);
    return OperatorTestBase::assertQuery(plan, {icebergSplit}, duckDbSql);
  }

  core::PlanNodePtr tableScanNode(RowTypePtr outputRowType) {
    return PlanBuilder(pool_.get()).tableScan(outputRowType).planNode();
  }

 private:
  dwio::common::FileFormat fileFomat_{dwio::common::FileFormat::DWRF};
  RowTypePtr rowType_;
  RowTypePtr outputRowType_;
  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();
};

// The positional delete file contains rows from multiple base files
TEST_F(HiveIcebergTest, positionalDeletesSingleBaseFile) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertPositionalDeletes({0, 1, 2, 3});
  // Delete the first and last row in each batch (10000 rows per batch)
  assertPositionalDeletes({0, 9999, 10000, 19999});
  // Delete several rows in the second batch (10000 rows per batch)
  assertPositionalDeletes({10000, 10002, 19999});
  // Delete random rows
  assertPositionalDeletes(makeRandomDeleteRows(kRowCount));
  // Delete 0 rows
  assertPositionalDeletes({}, "SELECT * FROM tmp", false);
  // Delete all rows
  assertPositionalDeletes(
      makeSequenceRows(kRowCount), "SELECT * FROM tmp WHERE 1 = 0", false);
  // Delete rows that don't exist
  assertPositionalDeletes({20000, 29999});
}

// The positional delete file contains rows from multiple base files
TEST_F(HiveIcebergTest, positionalDeletesMultipleBaseFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertPositionalDeletes({0, 1, 2, 3}, true);
  // Delete the first and last row in each batch (10000 rows per batch)
  assertPositionalDeletes({0, 9999, 10000, 19999}, true);
  // Delete several rows in the second batch (10000 rows per batch)
  assertPositionalDeletes({10000, 10002, 19999}, true);
  // Delete random rows
  assertPositionalDeletes(makeRandomDeleteRows(kRowCount), true);
  // Delete 0 rows
  assertPositionalDeletes({}, "SELECT * FROM tmp", true);
  // Delete all rows
  assertPositionalDeletes(
      makeSequenceRows(kRowCount), "SELECT * FROM tmp WHERE 1 = 0", true);
  // Delete rows that don't exist
  assertPositionalDeletes({20000, 29999}, true);
}

// Delete values from a single column file
TEST_F(HiveIcebergTest, equalityDeletesSingleFileColumn1) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertEqualityDeletes({{0, 1, 2, 3}}, {1});
  // Delete the first and last row in each batch (10000 rows per batch)
  assertEqualityDeletes({{0, 9999, 10000, 19999}}, {1});
  // Delete several rows in the second batch (10000 rows per batch)
  assertEqualityDeletes({{10000, 10002, 19999}}, {1});
  // Delete random rows
  assertEqualityDeletes({makeRandomDeleteRows(kRowCount)}, {1});
  // Delete 0 rows
  assertEqualityDeletes({{}}, {1});
  // Delete all rows
  assertEqualityDeletes({makeSequenceRows(kRowCount)}, {1});
  // Delete rows that don't exist
  assertEqualityDeletes({{20000, 29999}}, {1});
}

// Delete values from the second column in a 2-column file
//
//    c1    c2
//    0     0
//    1     0
//    2     1
//    3     1
//    4     2
//  ...    ...
//  19999 9999
TEST_F(HiveIcebergTest, equalityDeletesSingleFileColumn2) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete values 0, 1, 2, 3 from the second column
  assertEqualityDeletes({{0, 1, 2, 3}}, {2});
  // Delete the smallest value 0 and the largest value 9999 from the second
  // column, which has the range [0, 9999]
  assertEqualityDeletes({{0, 9999}}, {2});
  // Delete non-existent values from the second column
  assertEqualityDeletes({{10000, 10002, 19999}}, {2});
  // Delete random rows from the second column
  assertEqualityDeletes({makeRandomDeleteRows(kRowCount)}, {2});
  //     Delete 0 values
  assertEqualityDeletes({{}}, {2});
  // Delete all values
  assertEqualityDeletes({makeSequenceRows(kRowCount / 2)}, {2});
}

// Delete values from 2 columns with the following data:
//
//    c1    c2
//    0     0
//    1     0
//    2     1
//    3     1
//    4     2
//  ...    ...
//  19999 9999
TEST_F(HiveIcebergTest, equalityDeletesSingleFileMultipleColumns) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete rows 0, 1
  assertEqualityDeletes({{0, 1}, {0, 0}}, {1, 2});
  // Delete rows 0, 2, 4, 6
  assertEqualityDeletes({{0, 2, 4, 6}, {0, 1, 2, 3}}, {1, 2});
  // Delete the last row
  assertEqualityDeletes({{19999}, {9999}}, {1, 2});
  // Delete non-existent values
  assertEqualityDeletes({{20000, 30000}, {10000, 1500}}, {1, 2});
  // Delete 0 values
  assertEqualityDeletes({{}, {}}, {1, 2}, "SELECT * FROM tmp");
  // Delete all values
  assertEqualityDeletes(
      {makeSequenceRows(kRowCount), makeSequenceRows(kRowCount, 2)},
      {1, 2},
      "SELECT * FROM tmp WHERE 1 = 0");
}

} // namespace facebook::velox::connector::hive::iceberg
