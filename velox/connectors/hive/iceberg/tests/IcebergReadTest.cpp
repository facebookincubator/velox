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

#include "velox/common/base/tests/GTestUtils.h"
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

class IcebergReadPositionalDeleteTest : public HiveConnectorTestBase {
 public:
  IcebergReadPositionalDeleteTest()
      : config_{std::make_shared<facebook::velox::dwrf::Config>()} {
    // Make the writers flush per batch so that we can create non-aligned
    // RowGroups between the base data files and delete files
    flushPolicyFactory_ = []() {
      return std::make_unique<dwrf::LambdaFlushPolicy>([]() { return true; });
    };
  }

  /// Create 1 base data file data_file_1 with 2 RowGroups of 10000 rows each.
  /// Also create 1 delete file delete_file_1 which contains delete positions
  /// for data_file_1.
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

  /// Create 3 base data files, where the first file data_file_0 has 500 rows,
  /// the second file data_file_1 contains 2 RowGroups of 10000 rows each, and
  /// the third file data_file_2 contains 500 rows. It creates 1 positional
  /// delete file delete_file_1, which contains delete positions for
  /// data_file_1.
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

  /// Create 1 base data file data_file_1 with 2 RowGroups of 10000 rows each.
  /// Create multiple delete files with name data_file_1, data_file_2, and so on
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
      int rowCountPerFile = rowCount_,
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
      const std::map<std::string, std::vector<int64_t>>& rowGroupSizesForFiles,
      const std::unordered_map<
          std::string,
          std::multimap<std::string, std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles,
      int32_t numPrefetchSplits = 0,
      int32_t splitCount = 1) {
    // Keep the reference to the deleteFilePath, otherwise the corresponding
    // file will be deleted.
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths =
        writeDataFiles(rowGroupSizesForFiles);
    std::unordered_map<
        std::string,
        std::pair<int64_t, std::shared_ptr<TempFilePath>>>
        deleteFilePaths = writePositionDeleteFiles(
            deleteFilesForBaseDatafiles, dataFilePaths);

    std::vector<std::shared_ptr<ConnectorSplit>> splits;

    for (const auto& dataFile : dataFilePaths) {
      std::string baseFileName = dataFile.first;
      std::string baseFilePath = dataFile.second->getPath();

      std::vector<IcebergDeleteFile> deleteFiles;

      for (auto const& deleteFile : deleteFilesForBaseDatafiles) {
        std::string deleteFileName = deleteFile.first;
        std::multimap<std::string, std::vector<int64_t>> deleteFileContent =
            deleteFile.second;

        if (deleteFileContent.count(baseFileName) != 0) {
          // If this delete file contains rows for the target base file, then
          // add it to the split
          auto deleteFilePath =
              deleteFilePaths[deleteFileName].second->getPath();
          IcebergDeleteFile icebergDeleteFile(
              FileContent::kPositionalDeletes,
              deleteFilePath,
              fileFormat_,
              deleteFilePaths[deleteFileName].first,
              testing::internal::GetFileSize(
                  std::fopen(deleteFilePath.c_str(), "r")));
          deleteFiles.push_back(icebergDeleteFile);
        }
      }

      auto icebergSplits =
          makeIcebergSplits(baseFilePath, deleteFiles, {}, splitCount);
      splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
    }

    std::string duckdbSql =
        getDuckDBQuery(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
    auto plan = tableScanNode(rowType_);
    auto task = HiveConnectorTestBase::assertQuery(
        plan, splits, duckdbSql, numPrefetchSplits);

    auto planStats = toPlanStats(task->taskStats());
    auto scanNodeId = plan->id();
    auto it = planStats.find(scanNodeId);
    ASSERT_TRUE(it != planStats.end());
    ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  }

  template <TypeKind KIND>
  void assertEqualityDeletes(
      const std::unordered_map<
          int8_t,
          std::vector<std::vector<typename TypeTraits<KIND>::NativeType>>>&
          equalityDeleteVectorMap,
      const std::unordered_map<int8_t, std::vector<int32_t>>&
          equalityFieldIdsMap,
      std::string duckDbSql = "",
      std::vector<RowVectorPtr> dataVectors = {}) {
    VELOX_CHECK_EQ(equalityDeleteVectorMap.size(), equalityFieldIdsMap.size());
    // We will create data vectors with numColumns number of columns that is the
    // max field ID in equalityFieldIds
    int32_t numDataColumns = 0;

    for (auto it = equalityFieldIdsMap.begin(); it != equalityFieldIdsMap.end();
         ++it) {
      auto equalityFieldIds = it->second;
      auto currentMax =
          *std::max_element(equalityFieldIds.begin(), equalityFieldIds.end());
      numDataColumns = std::max(numDataColumns, currentMax);
    }

    VELOX_CHECK_GT(numDataColumns, 0);
    VELOX_CHECK_GE(numDataColumns, equalityDeleteVectorMap.size());
    VELOX_CHECK_GT(equalityDeleteVectorMap.size(), 0);

    VELOX_CHECK_LE(equalityFieldIdsMap.size(), numDataColumns);

    std::shared_ptr<TempFilePath> dataFilePath =
        writeDataFiles<KIND>(rowCount_, numDataColumns, 1, dataVectors)[0];

    std::vector<IcebergDeleteFile> deleteFiles;
    std::string predicates = "";
    unsigned long numDeletedValues = 0;

    std::vector<std::shared_ptr<TempFilePath>> deleteFilePaths;
    for (auto it = equalityFieldIdsMap.begin();
         it != equalityFieldIdsMap.end();) {
      auto equalityFieldIds = it->second;
      auto equalityDeleteVector = equalityDeleteVectorMap.at(it->first);
      VELOX_CHECK_GT(equalityDeleteVector.size(), 0);
      numDeletedValues =
          std::max(numDeletedValues, equalityDeleteVector[0].size());
      deleteFilePaths.push_back(
          writeEqualityDeleteFile<KIND>(equalityDeleteVector));
      IcebergDeleteFile deleteFile(
          FileContent::kEqualityDeletes,
          deleteFilePaths.back()->getPath(),
          fileFormat_,
          equalityDeleteVector[0].size(),
          testing::internal::GetFileSize(
              std::fopen(deleteFilePaths.back()->getPath().c_str(), "r")),
          equalityFieldIds);
      deleteFiles.push_back(deleteFile);
      predicates +=
          makePredicates<KIND>(equalityDeleteVector, equalityFieldIds);
      ++it;
      if (it != equalityFieldIdsMap.end()) {
        predicates += " AND ";
      }
    }

    // The default split count is 1.
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    // If the caller passed in a query, use that.
    if (duckDbSql == "") {
      // Select all columns
      duckDbSql = "SELECT * FROM tmp ";
      if (numDeletedValues > 0) {
        duckDbSql += fmt::format("WHERE {}", predicates);
      }
    }

    assertEqualityDeletes(
        icebergSplits.back(),
        !dataVectors.empty() ? asRowType(dataVectors[0]->type()) : rowType_,
        duckDbSql);

    // Select a column that's not in the filter columns
    if (numDataColumns > 1 &&
        equalityDeleteVectorMap.at(0).size() < numDataColumns) {
      std::string duckDbQuery = "SELECT c0 FROM tmp";
      if (numDeletedValues > 0) {
        duckDbQuery += fmt::format(" WHERE {}", predicates);
      }

      std::vector<std::string> names({"c0"});
      std::vector<TypePtr> types(1, createScalarType<KIND>());
      assertEqualityDeletes(
          icebergSplits.back(),
          std::make_shared<RowType>(std::move(names), std::move(types)),
          duckDbQuery);
    }
  }

  template <typename T>
  std::vector<T> makeSequenceValues(int32_t numRows, int8_t repeat = 1) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    VELOX_CHECK_GT(repeat, 0);

    auto maxValue = std::ceil(static_cast<double>(numRows) / repeat);
    std::vector<T> values;
    values.reserve(numRows);
    for (int32_t i = 0; i < maxValue; i++) {
      for (int8_t j = 0; j < repeat; j++) {
        values.push_back(static_cast<T>(i));
      }
    }
    values.resize(numRows);
    return values;
  }

  std::vector<int64_t> makeRandomDeleteValues(int32_t maxRowNumber) {
    std::mt19937 gen{0};
    std::vector<int64_t> deleteRows;
    for (int i = 0; i < maxRowNumber; i++) {
      if (folly::Random::rand32(0, 10, gen) > 8) {
        deleteRows.push_back(i);
      }
    }
    return deleteRows;
  }

 private:
  /// Input is like <"deleteFile1", <"dataFile1", {pos_RG1, pos_RG2,..}>,
  /// <"dataFile2", {pos_RG1, pos_RG2,..}>
  std::unordered_map<
      std::string,
      std::pair<int64_t, std::shared_ptr<TempFilePath>>>
  writePositionDeleteFiles(
      const std::unordered_map<
          std::string, // delete file name
          std::multimap<
              std::string,
              std::vector<int64_t>>>&
          deleteFilesForBaseDatafiles, // <base file name, delete position
                                       // vector for all RowGroups>
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
            [&](vector_size_t row) { return baseFilePath; });
        auto deletePosVector = makeFlatVector<int64_t>(positionsInRowGroup);

        RowVectorPtr deleteFileVector = makeRowVector(
            {pathColumn_->name, posColumn_->name},
            {filePathVector, deletePosVector});

        deleteFileVectors.push_back(deleteFileVector);
        totalPositionsInDeleteFile += positionsInRowGroup.size();
      }

      writeToFile(
          deleteFilePath->getPath(),
          deleteFileVectors,
          config_,
          flushPolicyFactory_);

      deleteFilePaths[deleteFileName] =
          std::make_pair(totalPositionsInDeleteFile, deleteFilePath);
    }

    return deleteFilePaths;
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
      // Sum up the row counts in all RowGroups in each base file
      baseFileSizes[rowGroupSizesInFile.first] += std::accumulate(
          rowGroupSizesInFile.second.begin(),
          rowGroupSizesInFile.second.end(),
          0LL);
      totalNumRowsInAllBaseFiles += baseFileSizes[rowGroupSizesInFile.first];
    }

    // Group the delete vectors by baseFileName
    std::map<std::string, std::vector<std::vector<int64_t>>>
        deletePosVectorsForAllBaseFiles;
    for (auto deleteFile : deleteFilesForBaseDatafiles) {
      auto deleteFileContent = deleteFile.second;
      for (auto rowGroup : deleteFileContent) {
        auto baseFileName = rowGroup.first;
        deletePosVectorsForAllBaseFiles[baseFileName].push_back(
            rowGroup.second);
      }
    }
    // Flatten and deduplicate the delete position vectors in
    // deletePosVectorsForAllBaseFiles from previous step, and count the total
    // number of distinct delete positions for all base files
    std::map<std::string, std::vector<int64_t>>
        flattenedDeletePosVectorsForAllBaseFiles;
    int64_t totalNumDeletePositions = 0;
    for (auto deleteVectorsForBaseFile : deletePosVectorsForAllBaseFiles) {
      auto baseFileName = deleteVectorsForBaseFile.first;
      auto deletePositionVectors = deleteVectorsForBaseFile.second;
      std::vector<int64_t> deletePositionVector =
          flattenAndDedup(deletePositionVectors, baseFileSizes[baseFileName]);
      flattenedDeletePosVectorsForAllBaseFiles[baseFileName] =
          deletePositionVector;
      totalNumDeletePositions += deletePositionVector.size();
    }

    // Now build the DuckDB queries
    if (totalNumDeletePositions == 0) {
      return "SELECT * FROM tmp";
    } else if (totalNumDeletePositions >= totalNumRowsInAllBaseFiles) {
      return "SELECT * FROM tmp WHERE 1 = 0";
    } else {
      // Convert the delete positions in all base files into column values
      std::vector<int64_t> allDeleteValues;

      int64_t numRowsInPreviousBaseFiles = 0;
      for (auto baseFileSize : baseFileSizes) {
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
          makeNotInList<TypeKind::BIGINT>(allDeleteValues));
    }
  }

  std::vector<int64_t> flattenAndDedup(
      const std::vector<std::vector<int64_t>>& deletePositionVectors,
      int64_t baseFileSize) {
    std::vector<int64_t> deletePositionVector;
    for (auto vec : deletePositionVectors) {
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

  template <TypeKind KIND>
  std::string makeNotInList(
      const std::vector<typename TypeTraits<KIND>::NativeType>& deleteValues) {
    using T = typename TypeTraits<KIND>::NativeType;
    if (deleteValues.empty()) {
      return "";
    }

    if constexpr (KIND == TypeKind::BOOLEAN) {
      VELOX_FAIL("Unsupported Type : {}", TypeTraits<KIND>::name);
    } else if constexpr (
        KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      // For VARCHAR, wrap values in single quotes
      return std::accumulate(
          deleteValues.begin() + 1,
          deleteValues.end(),
          fmt::format("'{}'", to<std::string>(deleteValues[0])),
          [](const std::string& a, const T& b) {
            return a + fmt::format(", '{}'", to<std::string>(b));
          });
    } else if (std::is_integral_v<T> || std::is_floating_point_v<T>) {
      return std::accumulate(
          deleteValues.begin() + 1,
          deleteValues.end(),
          to<std::string>(deleteValues[0]),
          [](const std::string& a, const T& b) {
            return a + ", " + to<std::string>(b);
          });
    } else {
      VELOX_FAIL("Unsupported Type : {}", TypeTraits<KIND>::name);
    }
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputRowType) {
    return PlanBuilder(pool_.get()).tableScan(outputRowType).planNode();
  }

  template <TypeKind KIND>
  std::string makeTypedPredicate(
      const std::string& columnName,
      const typename TypeTraits<KIND>::NativeType& value) {
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      return fmt::format("({} <> '{}')", columnName, value);
    } else if constexpr (
        KIND == TypeKind::TINYINT || KIND == TypeKind::SMALLINT ||
        KIND == TypeKind::INTEGER || KIND == TypeKind::BIGINT) {
      return fmt::format("({} <> {})", columnName, value);
    } else {
      VELOX_FAIL("Unsupported predicate type : {}", TypeTraits<KIND>::name);
    }
  }

  template <TypeKind KIND>
  std::string makePredicates(
      const std::vector<std::vector<typename TypeTraits<KIND>::NativeType>>&
          equalityDeleteVector,
      const std::vector<int32_t>& equalityFieldIds) {
    using T = typename TypeTraits<KIND>::NativeType;

    std::string predicates;
    int32_t numDataColumns =
        *std::max_element(equalityFieldIds.begin(), equalityFieldIds.end());

    VELOX_CHECK_GT(numDataColumns, 0);
    VELOX_CHECK_GE(numDataColumns, equalityDeleteVector.size());
    VELOX_CHECK_GT(equalityDeleteVector.size(), 0);

    auto numDeletedValues = equalityDeleteVector[0].size();

    if (numDeletedValues == 0) {
      return predicates;
    }

    // Check if all values for a column are deleted
    for (auto i = 0; i < equalityDeleteVector.size(); i++) {
      auto equalityFieldId = equalityFieldIds[i];
      auto deleteValues = equalityDeleteVector[i];

      // Make a copy to find unique values
      auto uniqueValues = deleteValues;
      std::sort(uniqueValues.begin(), uniqueValues.end());
      auto lastIter = std::unique(uniqueValues.begin(), uniqueValues.end());
      auto numDistinctValues = std::distance(uniqueValues.begin(), lastIter);

      // For column with field ID n, the max value is (rowCount_-1)/(n)
      // because values repeat n times
      if (numDistinctValues > 0 && equalityFieldId > 0) {
        auto maxPossibleValue = (rowCount_ - 1) / equalityFieldId;
        if (numDistinctValues > maxPossibleValue) {
          return "1 = 0";
        }
      }
    }

    if (equalityDeleteVector.size() == 1) {
      std::string name = fmt::format("c{}", equalityFieldIds[0] - 1);
      predicates = fmt::format(
          "{} NOT IN ({})",
          name,
          makeNotInList<KIND>({equalityDeleteVector[0]}));
    } else {
      for (int i = 0; i < numDeletedValues; i++) {
        std::string oneRow;
        for (int j = 0; j < equalityFieldIds.size(); j++) {
          std::string const name = fmt::format("c{}", equalityFieldIds[j] - 1);
          std::string predicate =
              makeTypedPredicate<KIND>(name, equalityDeleteVector[j][i]);

          oneRow = oneRow.empty()
              ? predicate
              : fmt::format("({} OR {})", oneRow, predicate);
        }

        predicates = predicates.empty()
            ? oneRow
            : fmt::format("{} AND {}", predicates, oneRow);
      }
    }
    return predicates;
  }

  std::shared_ptr<IcebergMetadataColumn> pathColumn_ =
      IcebergMetadataColumn::icebergDeleteFilePathColumn();
  std::shared_ptr<IcebergMetadataColumn> posColumn_ =
      IcebergMetadataColumn::icebergDeletePosColumn();

 protected:
  RowTypePtr rowType_{ROW({"c0"}, {BIGINT()})};
  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};
  static constexpr int rowCount_ = 20000;
  std::shared_ptr<dwrf::Config> config_;
  std::function<std::unique_ptr<dwrf::DWRFFlushPolicy>()> flushPolicyFactory_;

  std::vector<std::shared_ptr<ConnectorSplit>> makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      const uint32_t splitCount = 1) {
    std::unordered_map<std::string, std::string> customSplitInfo;
    customSplitInfo["table_format"] = "hive-iceberg";

    auto file = filesystems::getFileSystem(dataFilePath, nullptr)
                    ->openFileForRead(dataFilePath);
    const int64_t fileSize = file->size();
    std::vector<std::shared_ptr<ConnectorSplit>> splits;
    const uint64_t splitSize = std::floor((fileSize) / splitCount);

    for (int i = 0; i < splitCount; ++i) {
      splits.emplace_back(std::make_shared<HiveIcebergSplit>(
          kHiveConnectorId,
          dataFilePath,
          fileFormat_,
          i * splitSize,
          splitSize,
          partitionKeys,
          std::nullopt,
          customSplitInfo,
          nullptr,
          /*cacheable=*/true,
          deleteFiles));
    }

    return splits;
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

  template <TypeKind KIND>
  std::shared_ptr<TempFilePath> writeEqualityDeleteFile(
      const std::vector<std::vector<typename TypeTraits<KIND>::NativeType>>&
          equalityDeleteVector) {
    using T = typename TypeTraits<KIND>::NativeType;
    std::vector<std::string> names;
    std::vector<VectorPtr> vectors;
    for (int i = 0; i < equalityDeleteVector.size(); i++) {
      names.push_back(fmt::format("c{}", i));
      vectors.push_back(makeFlatVector<T>(equalityDeleteVector[i]));
    }

    RowVectorPtr const deleteFileVectors = makeRowVector(names, vectors);

    auto deleteFilePath = TempFilePath::create();
    writeToFile(deleteFilePath->getPath(), deleteFileVectors);

    return deleteFilePath;
  }

  /// Creates data files for Iceberg testing with simple row/column
  /// specifications.
  ///
  /// This function generates test data files with the specified data type and
  /// structure. It creates columnar data with predictable patterns for testing
  /// purposes.
  ///
  /// @tparam KIND The data type for columns (default: BIGINT)
  /// @param numRows Total number of rows per split/file
  /// @param numColumns Number of columns in each file (default: 1)
  /// @param splitCount Number of separate files to create (default: 1)
  /// @param dataVectors Pre-created data vectors; if empty, generates new ones
  /// @return Vector of file paths to the created data files
  ///
  /// @note Generated data follows patterns:
  ///   - First column (c0): continuously increasing values [0, 1, 2, ...]
  ///   - Second column (c1): values repeat once [0, 0, 1, 1, 2, 2, ...]
  ///   - Third column (c2): values repeat twice [0, 0, 0, 0, 1, 1, 1, 1, ...]
  ///   - And so on for additional columns
  template <TypeKind KIND = TypeKind::BIGINT>
  std::vector<std::shared_ptr<TempFilePath>> writeDataFiles(
      uint64_t numRows,
      int32_t numColumns = 1,
      int32_t splitCount = 1,
      std::vector<RowVectorPtr> dataVectors = {}) {
    if (dataVectors.empty()) {
      dataVectors = makeVectors<KIND>(splitCount, numRows, numColumns);
    }
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

  /// Creates named data files with custom row group structures for complex
  /// testing.
  ///
  /// This function generates multiple named data files, each with specific row
  /// group sizes. Values are continuously increasing across all files to
  /// simplify DuckDB query construction for test validation.
  ///
  /// @param rowGroupSizesForFiles Map of file names to vectors of row group
  /// sizes
  ///   Key: file name (e.g., "data_file_1")
  ///   Value: vector of row group sizes (e.g., {1000, 500} for 2 row groups)
  /// @return Map of file names to their corresponding file paths
  ///
  /// @example
  ///   Input: {
  ///     "data_file_1": {1000, 500},
  ///     "data_file_2": {750}
  ///   }
  ///   Creates:
  ///   - data_file_1: 1500 rows (1000 + 500) with values [0-1499]
  ///   - data_file_2: 750 rows with values [1500-2249]
  ///
  /// @note This is primarily used for testing positional deletes and complex
  ///       row group boundary scenarios in Iceberg format.
  std::map<std::string, std::shared_ptr<TempFilePath>> writeDataFiles(
      const std::map<std::string, std::vector<int64_t>>&
          rowGroupSizesForFiles) {
    std::map<std::string, std::shared_ptr<TempFilePath>> dataFilePaths;
    std::vector<RowVectorPtr> dataVectorsJoined;
    dataVectorsJoined.reserve(rowGroupSizesForFiles.size());

    int64_t startingValue = 0;
    for (const auto& dataFile : rowGroupSizesForFiles) {
      dataFilePaths[dataFile.first] = TempFilePath::create();

      std::vector<RowVectorPtr> dataVectors;
      dataVectors.reserve(dataFile.second.size());

      for (int64_t size : dataFile.second) {
        std::vector<int64_t> data;
        data.reserve(size);
        for (int64_t i = 0; i < size; ++i) {
          data.push_back(startingValue + i);
        }

        VectorPtr c0 = makeFlatVector<int64_t>(data);
        dataVectors.push_back(makeRowVector({"c0"}, {c0}));
        startingValue += size;
      }

      writeToFile(
          dataFilePaths[dataFile.first]->getPath(),
          dataVectors,
          config_,
          flushPolicyFactory_);

      for (const auto& vector : dataVectors) {
        dataVectorsJoined.push_back(vector);
      }
    }

    createDuckDbTable(dataVectorsJoined);
    return dataFilePaths;
  }

  template <TypeKind KIND>
  std::vector<RowVectorPtr>
  makeVectors(int32_t count, int32_t rowsPerVector, int32_t numColumns = 1) {
    using T = typename TypeTraits<KIND>::NativeType;

    // Sample strings for VARCHAR data generation
    const std::vector<std::string> sampleStrings = {
        "apple",     "banana",     "cherry",    "date",       "elderberry",
        "fig",       "grape",      "honeydew",  "kiwi",       "lemon",
        "mango",     "nectarine",  "orange",    "papaya",     "quince",
        "raspberry", "strawberry", "tangerine", "watermelon", "zucchini"};

    std::vector<TypePtr> types;
    for (int j = 0; j < numColumns; j++) {
      types.push_back(createScalarType<KIND>());
    }
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
        VectorPtr columnVector;
        if constexpr (KIND == TypeKind::VARCHAR) {
          // For VARCHAR, use sample strings with sequence-based indexing
          auto intData = makeSequenceValues<int64_t>(rowsPerVector, j + 1);
          auto stringVector = BaseVector::create<FlatVector<StringView>>(
              VARCHAR(), rowsPerVector, pool_.get());

          for (int idx = 0; idx < rowsPerVector; ++idx) {
            auto stringIndex = intData[idx] % sampleStrings.size();
            const std::string& selectedString = sampleStrings[stringIndex];
            stringVector->set(idx, StringView(selectedString));
          }
          columnVector = stringVector;
        } else if constexpr (KIND == TypeKind::VARBINARY) {
          auto intData = makeSequenceValues<int64_t>(rowsPerVector, j + 1);
          auto binaryVector = BaseVector::create<FlatVector<StringView>>(
              VARBINARY(), rowsPerVector, pool_.get());

          for (int idx = 0; idx < rowsPerVector; ++idx) {
            auto stringIndex = intData[idx] % sampleStrings.size();
            const std::string& baseString = sampleStrings[stringIndex];

            std::string binaryStr;
            for (char c : baseString) {
              binaryStr += static_cast<unsigned char>(c);
            }
            binaryVector->set(idx, StringView(binaryStr));
          }
          columnVector = binaryVector;
        } else if constexpr (std::is_integral_v<T>) {
          auto data = makeSequenceValues<typename TypeTraits<KIND>::NativeType>(
              rowsPerVector, j + 1);
          columnVector = vectorMaker_.flatVector<T>(data);
        } else if constexpr (std::is_floating_point_v<T>) {
          auto intData = makeSequenceValues<int64_t>(rowsPerVector, j + 1);
          std::vector<T> floatData;
          floatData.reserve(intData.size());
          for (auto val : intData) {
            floatData.push_back(static_cast<T>(val) + 0.5f);
          }
          columnVector = vectorMaker_.flatVector<T>(floatData);
        } else {
          VELOX_FAIL(
              "Unsupported type for makeVectors: {}", TypeTraits<KIND>::name);
        }
        vectors.push_back(columnVector);
      }

      rowVectors.push_back(makeRowVector(names, vectors));
    }

    rowType_ = std::make_shared<RowType>(std::move(names), std::move(types));

    return rowVectors;
  }
};

/// This test creates one single data file and one delete file. The parameter
/// passed to assertSingleBaseFileSinglePositionalDelete is the delete
/// positions.
TEST_F(
    IcebergReadPositionalDeleteTest,
    singleBaseFileSinglePositionalDeleteFile) {
  folly::SingletonVault::singleton()->registrationComplete();

  assertSingleBaseFileSingleDeleteFile({{0, 1, 2, 3}});
  // Delete the first and last row in each batch (10000 rows per batch)
  assertSingleBaseFileSingleDeleteFile({{0, 9999, 10000, 19999}});
  // Delete several rows in the second batch (10000 rows per batch)
  assertSingleBaseFileSingleDeleteFile({{10000, 10002, 19999}});
  // Delete random rows
  assertSingleBaseFileSingleDeleteFile({makeRandomIncreasingValues(0, 20000)});
  // Delete 0 rows
  assertSingleBaseFileSingleDeleteFile({});
  // Delete all rows
  assertSingleBaseFileSingleDeleteFile({makeSequenceValues<int64_t>(20000)});
  // Delete rows that don't exist
  assertSingleBaseFileSingleDeleteFile({{20000, 29999}});
}

/// This test creates 3 base data files, only the middle one has corresponding
/// delete positions. The parameter passed to
/// assertSingleBaseFileSingleDeleteFile is the delete positions.for the middle
/// base file.
TEST_F(
    IcebergReadPositionalDeleteTest,
    multipleBaseFilesSinglePositionalDeleteFile) {
  folly::SingletonVault::singleton()->registrationComplete();

  assertMultipleBaseFileSingleDeleteFile({0, 1, 2, 3});
  assertMultipleBaseFileSingleDeleteFile({0, 9999, 10000, 19999});
  assertMultipleBaseFileSingleDeleteFile({10000, 10002, 19999});
  assertMultipleBaseFileSingleDeleteFile({10000, 10002, 19999});
  assertMultipleBaseFileSingleDeleteFile(
      makeRandomIncreasingValues(0, rowCount_));
  assertMultipleBaseFileSingleDeleteFile({});
  assertMultipleBaseFileSingleDeleteFile(
      makeSequenceValues<int64_t>(rowCount_));
}

/// This test creates one base data file/split with multiple delete files. The
/// parameter passed to assertSingleBaseFileMultipleDeleteFiles is the vector of
/// delete files. Each leaf vector represents the delete positions in that
/// delete file.
TEST_F(
    IcebergReadPositionalDeleteTest,
    singleBaseFileMultiplePositionalDeleteFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Delete row 0, 1, 2, 3 from the first batch out of two.
  assertSingleBaseFileMultipleDeleteFiles({{1}, {2}, {3}, {4}});
  // Delete the first and last row in each batch (10000 rows per batch).
  assertSingleBaseFileMultipleDeleteFiles({{0}, {9999}, {10000}, {19999}});

  assertSingleBaseFileMultipleDeleteFiles({{500, 21000}});

  assertSingleBaseFileMultipleDeleteFiles(
      {makeRandomIncreasingValues(0, 10000),
       makeRandomIncreasingValues(10000, 20000),
       makeRandomIncreasingValues(5000, 15000)});

  auto firstHalf = makeSequenceValues<int64_t>(10000);
  auto secondHalf = makeSequenceValues<int64_t>(10000);
  for (int i = 0; i < secondHalf.size(); i++) {
    secondHalf[i] += 10000;
  }
  assertSingleBaseFileMultipleDeleteFiles({firstHalf, secondHalf});

  assertSingleBaseFileMultipleDeleteFiles(
      {firstHalf, secondHalf, makeRandomIncreasingValues(5000, 15000)});

  auto allRows = makeSequenceValues<int64_t>(20000);
  assertSingleBaseFileMultipleDeleteFiles({allRows, allRows});

  assertSingleBaseFileMultipleDeleteFiles(
      {makeRandomIncreasingValues(0, 20000),
       {},
       makeRandomIncreasingValues(5000, 15000)});

  assertSingleBaseFileMultipleDeleteFiles({{}, {}});
}

/// This test creates 2 base data files, and 1 or 2 delete files, with unaligned
/// RowGroup boundaries
TEST_F(
    IcebergReadPositionalDeleteTest,
    multipleBaseFileMultiplePositionalDeleteFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  std::map<std::string, std::vector<int64_t>> rowGroupSizesForFiles;
  std::unordered_map<
      std::string,
      std::multimap<std::string, std::vector<int64_t>>>
      deleteFilesForBaseDatafiles;

  // Create two data files, each with two RowGroups
  rowGroupSizesForFiles["data_file_1"] = {100, 85};
  rowGroupSizesForFiles["data_file_2"] = {99, 1};

  // Delete 3 rows from the first RowGroup in data_file_1
  deleteFilesForBaseDatafiles["delete_file_1"] = {{"data_file_1", {0, 1, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete 3 rows from the second RowGroup in data_file_1
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {100, 101, 184}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete random rows from the both RowGroups in data_file_1
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete all rows in data_file_1
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeSequenceValues<int64_t>(185)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
  //
  // Delete non-existent rows from data_file_1
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(186, 300)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Delete several rows from both RowGroups in both data files
  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 100, 102, 184}}, {"data_file_2", {1, 98, 99}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // The delete file delete_file_1 contains 3 RowGroups itself, with the first 3
  // deleting some repeating rows in data_file_1, and the last 2 RowGroups
  // deleting some  repeating rows in data_file_2
  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}},
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", {1, 3, 5, 7}},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // delete_file_2 contains non-overlapping delete rows for each data files in
  // each RowGroup
  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", {0, 1, 2, 3}}, {"data_file_2", {1, 3, 5, 7}}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", {1, 2, 3, 4}},
      {"data_file_1", {98, 99, 100, 101, 184}},
      {"data_file_2", {3, 5, 7, 9}},
      {"data_file_2", {98, 99, 100}}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);

  // Two delete files each containing overlapping delete rows for both data
  // files
  deleteFilesForBaseDatafiles.clear();
  deleteFilesForBaseDatafiles["delete_file_1"] = {
      {"data_file_1", makeRandomIncreasingValues(0, 185)},
      {"data_file_2", makeRandomIncreasingValues(0, 100)}};
  deleteFilesForBaseDatafiles["delete_file_2"] = {
      {"data_file_1", makeRandomIncreasingValues(10, 120)},
      {"data_file_2", makeRandomIncreasingValues(50, 100)}};
  assertPositionalDeletes(rowGroupSizesForFiles, deleteFilesForBaseDatafiles);
}

TEST_F(IcebergReadPositionalDeleteTest, positionalDeletesMultipleSplits) {
  folly::SingletonVault::singleton()->registrationComplete();

  assertMultipleSplits({1, 2, 3, 4}, 10, 5);
  assertMultipleSplits({1, 2, 3, 4}, 10, 0);
  assertMultipleSplits({1, 2, 3, 4}, 10, 10);
  assertMultipleSplits({0, 9999, 10000, 19999}, 10, 3);
  assertMultipleSplits(makeRandomIncreasingValues(0, 20000), 10, 3);
  assertMultipleSplits(makeSequenceValues<int64_t>(20000), 10, 3);
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

  // Include only upper bound(which is exclusive) in delete positions for the
  // second 10k batch of rows.
  assertMultipleSplits({1000, 9000, 20000}, 1, 0, 20000, 3);
}

TEST_F(IcebergReadPositionalDeleteTest, testPartitionedRead) {
  RowTypePtr rowType{ROW({"c0", "ds"}, {BIGINT(), DateType::get()})};
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys;
  // Iceberg API sets partition values for dates to daysSinceEpoch, so
  // in velox, we do not need to convert it to days.
  // Test query on two partitions ds=17627(2018-04-06), ds=17628(2018-04-07)
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  std::vector<std::shared_ptr<TempFilePath>> dataFilePaths;
  for (int i = 0; i <= 1; ++i) {
    std::vector<RowVectorPtr> dataVectors;
    int32_t daysSinceEpoch = 17627 + i;
    VectorPtr c0 = makeFlatVector<int64_t>((std::vector<int64_t>){i});
    VectorPtr ds =
        makeFlatVector<int32_t>((std::vector<int32_t>){daysSinceEpoch});
    dataVectors.push_back(makeRowVector({"c0", "ds"}, {c0, ds}));

    auto dataFilePath = TempFilePath::create();
    dataFilePaths.push_back(dataFilePath);
    writeToFile(
        dataFilePath->getPath(), dataVectors, config_, flushPolicyFactory_);
    partitionKeys["ds"] = std::to_string(daysSinceEpoch);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), {}, partitionKeys);
    splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
  }

  connector::ColumnHandleMap assignments;
  assignments.insert(
      {"c0",
       std::make_shared<HiveColumnHandle>(
           "c0",
           HiveColumnHandle::ColumnType::kRegular,
           rowType->childAt(0),
           rowType->childAt(0))});

  std::vector<common::Subfield> requiredSubFields;
  HiveColumnHandle::ColumnParseParameters columnParseParameters;
  columnParseParameters.partitionDateValueFormat =
      HiveColumnHandle::ColumnParseParameters::kDaysSinceEpoch;
  assignments.insert(
      {"ds",
       std::make_shared<HiveColumnHandle>(
           "ds",
           HiveColumnHandle::ColumnType::kPartitionKey,
           rowType->childAt(1),
           rowType->childAt(1),
           std::move(requiredSubFields),
           columnParseParameters)});

  auto plan = PlanBuilder(pool_.get())
                  .tableScan(rowType, {}, "", nullptr, assignments)
                  .planNode();

  HiveConnectorTestBase::assertQuery(
      plan,
      splits,
      "SELECT * FROM (VALUES (0, '2018-04-06'), (1, '2018-04-07'))",
      0);

  // Test filter on non-partitioned non-date column
  std::vector<std::string> nonPartitionFilters = {"c0 = 1"};
  plan = PlanBuilder(pool_.get())
             .tableScan(rowType, nonPartitionFilters, "", nullptr, assignments)
             .planNode();

  HiveConnectorTestBase::assertQuery(plan, splits, "SELECT 1, '2018-04-07'");

  // Test filter on non-partitioned date column
  std::vector<std::string> filters = {"ds = date'2018-04-06'"};
  plan = PlanBuilder(pool_.get()).tableScan(rowType, filters).planNode();

  splits.clear();
  for (auto& dataFilePath : dataFilePaths) {
    auto icebergSplits = makeIcebergSplits(dataFilePath->getPath());
    splits.insert(splits.end(), icebergSplits.begin(), icebergSplits.end());
  }

  HiveConnectorTestBase::assertQuery(plan, splits, "SELECT 0, '2018-04-06'");
}

class IcebergReadEqualityDeleteTest : public IcebergReadPositionalDeleteTest {};

TEST_F(IcebergReadEqualityDeleteTest, testSubFieldEqualityDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Write the base file
  std::shared_ptr<TempFilePath> dataFilePath = TempFilePath::create();
  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {"c_bigint", "c_row"},
      {makeFlatVector<int64_t>(20, [](auto row) { return row + 1; }),
       makeRowVector(
           {"c0", "c1", "c2"},
           {makeFlatVector<int64_t>(20, [](auto row) { return row + 1; }),
            makeFlatVector<int64_t>(20, [](auto row) { return row + 1; }),
            makeFlatVector<int64_t>(20, [](auto row) { return row + 1; })})})};
  int32_t numDataColumns = 1;
  dataFilePath = writeDataFiles<TypeKind::BIGINT>(
      rowCount_, numDataColumns, 1, dataVectors)[0];

  // Write the delete file. Equality delete field is c_row.c1
  std::vector<IcebergDeleteFile> deleteFiles;
  // Delete rows {0, 1} from c_row.c1, whose schema Id is 4
  std::vector<RowVectorPtr> deleteDataVectors = {makeRowVector(
      {"c1"}, {makeFlatVector<int64_t>(2, [](auto row) { return row + 1; })})};

  std::vector<std::shared_ptr<TempFilePath>> deleteFilePaths;
  auto equalityFieldIds = std::vector<int32_t>({4});
  auto deleteFilePath = TempFilePath::create();
  writeToFile(deleteFilePath->getPath(), deleteDataVectors.back());
  deleteFilePaths.push_back(deleteFilePath);
  IcebergDeleteFile deleteFile(
      FileContent::kEqualityDeletes,
      deleteFilePaths.back()->getPath(),
      fileFormat_,
      2,
      testing::internal::GetFileSize(
          std::fopen(deleteFilePaths.back()->getPath().c_str(), "r")),
      equalityFieldIds);
  deleteFiles.push_back(deleteFile);

  auto icebergSplits = makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

  // Select both c_bigint and c_row column columns
  std::string duckDbSql = "SELECT * FROM tmp WHERE c_row.c0 not in (1, 2)";
  assertEqualityDeletes(
      icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);

  // SELECT only c_bigint column
  duckDbSql = "SELECT c_bigint FROM tmp WHERE c_row.c0 not in (1, 2)";
  assertEqualityDeletes(
      icebergSplits.back(), ROW({"c_bigint"}, {BIGINT()}), duckDbSql);
}

TEST_F(IcebergReadEqualityDeleteTest, equalityDeletesMixedTypesInt64Varchar) {
  folly::SingletonVault::singleton()->registrationComplete();

  std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
  equalityFieldIdsMap.insert({0, {1, 2}});

  // Create data vectors with int64_t and varchar columns
  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
       makeFlatVector<StringView>(
           {"apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
            "honeydew",
            "kiwi",
            "lemon"})})};

  // Test 1: Delete first and last rows
  {
    std::unordered_map<int8_t, std::vector<std::vector<int64_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        stringDeleteMap;

    intDeleteMap.insert({0, {{0, 9}}});
    stringDeleteMap.insert({1, {{"apple", "lemon"}}});

    // Write int64_t delete file
    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int64_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    // Write varchar delete file
    auto stringDeleteFilePath = TempFilePath::create();
    RowVectorPtr stringDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(stringDeleteMap.at(1)[0])});
    writeToFile(stringDeleteFilePath->getPath(), stringDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        2,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        stringDeleteFilePath->getPath(),
        fileFormat_,
        2,
        testing::internal::GetFileSize(
            std::fopen(stringDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql =
        "SELECT * FROM tmp WHERE c0 NOT IN (0, 9) AND c1 NOT IN ('apple', 'lemon')";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  // Test 2: Delete random rows
  {
    std::unordered_map<int8_t, std::vector<std::vector<int64_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        stringDeleteMap;

    intDeleteMap.insert({0, {{1, 3, 5, 7}}});
    stringDeleteMap.insert({1, {{"banana", "date", "fig", "honeydew"}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int64_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto stringDeleteFilePath = TempFilePath::create();
    RowVectorPtr stringDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(stringDeleteMap.at(1)[0])});
    writeToFile(stringDeleteFilePath->getPath(), stringDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        4,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        stringDeleteFilePath->getPath(),
        fileFormat_,
        4,
        testing::internal::GetFileSize(
            std::fopen(stringDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql =
        "SELECT * FROM tmp WHERE c0 NOT IN (1, 3, 5, 7) AND c1 NOT IN ('banana', 'date', 'fig', 'honeydew')";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  // Test 3: Delete all rows
  {
    std::unordered_map<int8_t, std::vector<std::vector<int64_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        stringDeleteMap;

    intDeleteMap.insert({0, {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}});
    stringDeleteMap.insert(
        {1,
         {{"apple",
           "banana",
           "cherry",
           "date",
           "elderberry",
           "fig",
           "grape",
           "honeydew",
           "kiwi",
           "lemon"}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int64_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto stringDeleteFilePath = TempFilePath::create();
    RowVectorPtr stringDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(stringDeleteMap.at(1)[0])});
    writeToFile(stringDeleteFilePath->getPath(), stringDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        10,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        stringDeleteFilePath->getPath(),
        fileFormat_,
        10,
        testing::internal::GetFileSize(
            std::fopen(stringDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql = "SELECT * FROM tmp WHERE 1 = 0";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  // Test 4: Delete none
  {
    std::unordered_map<int8_t, std::vector<std::vector<int64_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        stringDeleteMap;

    intDeleteMap.insert({0, {{}}});
    stringDeleteMap.insert({1, {{}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int64_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto stringDeleteFilePath = TempFilePath::create();
    RowVectorPtr stringDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(stringDeleteMap.at(1)[0])});
    writeToFile(stringDeleteFilePath->getPath(), stringDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        0,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        stringDeleteFilePath->getPath(),
        fileFormat_,
        0,
        testing::internal::GetFileSize(
            std::fopen(stringDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql = "SELECT * FROM tmp";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }
}

TEST_F(
    IcebergReadEqualityDeleteTest,
    equalityDeletesMixedTypesTinyintVarbinary) {
  folly::SingletonVault::singleton()->registrationComplete();

  std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
  equalityFieldIdsMap.insert({0, {1, 2}});

  // Create data vectors with int8_t and varbinary columns
  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int8_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
       makeFlatVector<StringView>(
           {"\x01\x02",
            "\x03\x04",
            "\x05\x06",
            "\x07\x08",
            "\x09\x0A",
            "\x0B\x0C",
            "\x0D\x0E",
            "\x0F\x10",
            "\x11\x12",
            "\x13\x14"})})};

  // Test 1: Delete first and last rows
  {
    std::unordered_map<int8_t, std::vector<std::vector<int8_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        binaryDeleteMap;

    intDeleteMap.insert({0, {{0, 9}}});
    binaryDeleteMap.insert({1, {{"\x01\x02", "\x13\x14"}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int8_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto binaryDeleteFilePath = TempFilePath::create();
    RowVectorPtr binaryDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(binaryDeleteMap.at(1)[0])});
    writeToFile(binaryDeleteFilePath->getPath(), binaryDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        2,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        binaryDeleteFilePath->getPath(),
        fileFormat_,
        2,
        testing::internal::GetFileSize(
            std::fopen(binaryDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql =
        "SELECT * FROM tmp WHERE c0 NOT IN (0, 9) AND hex(c1) NOT IN ('0102', '1314')";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  // Test 2: Delete random rows
  {
    std::unordered_map<int8_t, std::vector<std::vector<int8_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        binaryDeleteMap;

    intDeleteMap.insert({0, {{1, 3, 5, 7}}});
    binaryDeleteMap.insert(
        {1, {{"\x03\x04", "\x07\x08", "\x0B\x0C", "\x0F\x10"}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int8_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto binaryDeleteFilePath = TempFilePath::create();
    RowVectorPtr binaryDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(binaryDeleteMap.at(1)[0])});
    writeToFile(binaryDeleteFilePath->getPath(), binaryDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        4,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        binaryDeleteFilePath->getPath(),
        fileFormat_,
        4,
        testing::internal::GetFileSize(
            std::fopen(binaryDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql =
        "SELECT * FROM tmp WHERE c0 NOT IN (1, 3, 5, 7) AND hex(c1) NOT IN ('0304', '0708', '0B0C', '0F10')";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  // Test 3: Delete all rows
  {
    std::unordered_map<int8_t, std::vector<std::vector<int8_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        binaryDeleteMap;

    intDeleteMap.insert({0, {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}});
    binaryDeleteMap.insert(
        {1,
         {{"\x01\x02",
           "\x03\x04",
           "\x05\x06",
           "\x07\x08",
           "\x09\x0A",
           "\x0B\x0C",
           "\x0D\x0E",
           "\x0F\x10",
           "\x11\x12",
           "\x13\x14"}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int8_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto binaryDeleteFilePath = TempFilePath::create();
    RowVectorPtr binaryDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(binaryDeleteMap.at(1)[0])});
    writeToFile(binaryDeleteFilePath->getPath(), binaryDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        10,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        binaryDeleteFilePath->getPath(),
        fileFormat_,
        10,
        testing::internal::GetFileSize(
            std::fopen(binaryDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql = "SELECT * FROM tmp WHERE 1 = 0";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  // Test 4: Delete none
  {
    std::unordered_map<int8_t, std::vector<std::vector<int8_t>>> intDeleteMap;
    std::unordered_map<int8_t, std::vector<std::vector<StringView>>>
        binaryDeleteMap;

    intDeleteMap.insert({0, {{}}});
    binaryDeleteMap.insert({1, {{}}});

    auto intDeleteFilePath = TempFilePath::create();
    RowVectorPtr intDeleteVector =
        makeRowVector({"c0"}, {makeFlatVector<int8_t>(intDeleteMap.at(0)[0])});
    writeToFile(intDeleteFilePath->getPath(), intDeleteVector);

    auto binaryDeleteFilePath = TempFilePath::create();
    RowVectorPtr binaryDeleteVector = makeRowVector(
        {"c1"}, {makeFlatVector<StringView>(binaryDeleteMap.at(1)[0])});
    writeToFile(binaryDeleteFilePath->getPath(), binaryDeleteVector);

    std::vector<IcebergDeleteFile> deleteFiles;
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        intDeleteFilePath->getPath(),
        fileFormat_,
        0,
        testing::internal::GetFileSize(
            std::fopen(intDeleteFilePath->getPath().c_str(), "r")),
        {1}));
    deleteFiles.push_back(IcebergDeleteFile(
        FileContent::kEqualityDeletes,
        binaryDeleteFilePath->getPath(),
        fileFormat_,
        0,
        testing::internal::GetFileSize(
            std::fopen(binaryDeleteFilePath->getPath().c_str(), "r")),
        {2}));

    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);
    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    createDuckDbTable(dataVectors);
    std::string duckDbSql = "SELECT * FROM tmp";
    assertEqualityDeletes(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }
}

class IcebergReadEqualityDeletesParameterizedTest
    : public IcebergReadPositionalDeleteTest,
      public testing::WithParamInterface<TypeKind> {
 public:
  template <TypeKind KIND>
  void testSingleColumnEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();
    using T = typename TypeTraits<KIND>::NativeType;

    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1}});

    // Test 1: Delete first and last rows
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 1);
      auto flatVector =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      std::vector<StringView> deleteValues = {
          flatVector->valueAt(0), flatVector->valueAt(rowCount_ - 1)};
      equalityDeleteVectorMap.insert({0, {deleteValues}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert(
          {0, {{static_cast<T>(0), static_cast<T>(rowCount_ - 1)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 2: Delete none (empty delete vector)
    equalityDeleteVectorMap.clear();
    equalityDeleteVectorMap.insert({0, {{}}});
    assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

    // Test 3: Delete all rows
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 1);
      auto flatVector =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      std::vector<StringView> deleteValues;
      deleteValues.reserve(rowCount_);
      for (int i = 0; i < rowCount_; ++i) {
        deleteValues.push_back(flatVector->valueAt(i));
      }
      equalityDeleteVectorMap.insert({0, {deleteValues}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert({0, {makeSequenceValues<T>(rowCount_)}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 4: Delete random rows
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 1);
      auto flatVector =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto randomIndices = makeRandomDeleteValues(rowCount_);
      std::vector<StringView> deleteValues;
      for (auto idx : randomIndices) {
        deleteValues.push_back(flatVector->valueAt(idx));
      }
      equalityDeleteVectorMap.insert({0, {deleteValues}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      std::vector<T> deleteValues;
      auto randomIndices = makeRandomDeleteValues(rowCount_);
      for (auto idx : randomIndices) {
        deleteValues.push_back(static_cast<T>(idx));
      }
      equalityDeleteVectorMap.insert({0, {deleteValues}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 5: Delete rows that don't exist
    equalityDeleteVectorMap.clear();
    if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert(
          {0, {{static_cast<T>(rowCount_), static_cast<T>(rowCount_ + 1)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    } else if constexpr (
        KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      std::vector<StringView> deleteValues = {
          StringView("nonexistent1"), StringView("nonexistent2")};
      equalityDeleteVectorMap.insert({0, {deleteValues}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }
  }

  template <TypeKind KIND>
  void testTwoColumnEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();

    using T = typename TypeTraits<KIND>::NativeType;
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1, 2}});

    // Test 1: Delete specific row pairs
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();

      // Delete first two row pairs
      std::vector<StringView> deleteValuesCol0 = {
          col0->valueAt(0), col0->valueAt(1)};
      std::vector<StringView> deleteValuesCol1 = {
          col1->valueAt(0), col1->valueAt(1)};

      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      // Delete rows where (c0=0 AND c1=0) and (c0=2 AND c1=1)
      equalityDeleteVectorMap.insert(
          {0,
           {{static_cast<T>(0), static_cast<T>(2)},
            {static_cast<T>(0), static_cast<T>(1)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 2: Delete none (empty delete vectors)
    equalityDeleteVectorMap.clear();
    equalityDeleteVectorMap.insert({0, {{}, {}}});
    assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

    // Test 3: Delete all rows
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();

      std::vector<StringView> deleteValuesCol0;
      std::vector<StringView> deleteValuesCol1;
      deleteValuesCol0.reserve(rowCount_);
      deleteValuesCol1.reserve(rowCount_);

      for (int i = 0; i < rowCount_; i++) {
        deleteValuesCol0.push_back(col0->valueAt(i));
        deleteValuesCol1.push_back(col1->valueAt(i));
      }

      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0",
          dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert(
          {0,
           {makeSequenceValues<T>(rowCount_, 1),
            makeSequenceValues<T>(rowCount_, 2)}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0");
    }

    // Test 4: Delete random row pairs
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();

      auto randomIndices = makeRandomDeleteValues(rowCount_ / 2);
      std::vector<StringView> deleteValuesCol0;
      std::vector<StringView> deleteValuesCol1;

      for (auto idx : randomIndices) {
        deleteValuesCol0.push_back(col0->valueAt(idx));
        deleteValuesCol1.push_back(col1->valueAt(idx));
      }

      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      auto randomIndices = makeRandomDeleteValues(rowCount_ / 2);
      std::vector<T> deleteValuesCol0;
      std::vector<T> deleteValuesCol1;

      for (auto idx : randomIndices) {
        deleteValuesCol0.push_back(static_cast<T>(idx));
        deleteValuesCol1.push_back(static_cast<T>(idx / 2));
      }

      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 5: Delete non-existent row pairs
    equalityDeleteVectorMap.clear();
    if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert(
          {0,
           {{static_cast<T>(rowCount_), static_cast<T>(rowCount_ + 1)},
            {static_cast<T>(rowCount_), static_cast<T>(rowCount_ + 1)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    } else if constexpr (
        KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      equalityDeleteVectorMap.insert(
          {0,
           {{StringView("nonexistent1"), StringView("nonexistent2")},
            {StringView("nonexistent3"), StringView("nonexistent4")}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }
  }

  template <TypeKind KIND>
  void testSingleFileMultipleColumnsEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();

    using T = typename TypeTraits<KIND>::NativeType;
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;

    equalityFieldIdsMap.insert({0, {1, 2}});

    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      // Delete rows 0, 1
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();
      std::vector<StringView> deleteValuesCol0 = {
          col0->valueAt(0), col0->valueAt(1)};
      std::vector<StringView> deleteValuesCol1 = {
          col1->valueAt(0), col1->valueAt(1)};
      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);

      // Delete rows 0, 2, 4, 6
      equalityDeleteVectorMap.clear();
      dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      col0 = dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      col1 = dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();
      deleteValuesCol0 = {
          col0->valueAt(0),
          col0->valueAt(2),
          col0->valueAt(4),
          col0->valueAt(6)};
      deleteValuesCol1 = {
          col1->valueAt(0),
          col1->valueAt(2),
          col1->valueAt(4),
          col1->valueAt(6)};
      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);

      // Delete the last row
      equalityDeleteVectorMap.clear();
      dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      col0 = dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      col1 = dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();
      deleteValuesCol0 = {col0->valueAt(rowCount_ - 1)};
      deleteValuesCol1 = {col1->valueAt(rowCount_ - 1)};
      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);

      // Delete non-existent values
      equalityDeleteVectorMap.clear();
      dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      deleteValuesCol0 = {
          StringView("nonexistent1"), StringView("nonexistent2")};
      deleteValuesCol1 = {
          StringView("nonexistent3"), StringView("nonexistent4")};
      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);

      // Delete 0 values
      equalityDeleteVectorMap.clear();
      dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      equalityDeleteVectorMap.insert({0, {{}, {}}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);

      // Delete all values
      equalityDeleteVectorMap.clear();
      dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      col0 = dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      col1 = dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();
      deleteValuesCol0.clear();
      deleteValuesCol1.clear();
      deleteValuesCol0.reserve(rowCount_);
      deleteValuesCol1.reserve(rowCount_);
      for (int i = 0; i < rowCount_; i++) {
        deleteValuesCol0.push_back(col0->valueAt(i));
        deleteValuesCol1.push_back(col1->valueAt(i));
      }
      equalityDeleteVectorMap.insert({0, {deleteValuesCol0, deleteValuesCol1}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0",
          dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      // Delete rows 0, 1
      equalityDeleteVectorMap.insert(
          {0,
           {{static_cast<T>(0), static_cast<T>(1)},
            {static_cast<T>(0), static_cast<T>(0)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

      // Delete rows 0, 2, 4, 6
      equalityDeleteVectorMap.clear();
      equalityDeleteVectorMap.insert(
          {0,
           {{static_cast<T>(0),
             static_cast<T>(2),
             static_cast<T>(4),
             static_cast<T>(6)},
            {static_cast<T>(0),
             static_cast<T>(1),
             static_cast<T>(2),
             static_cast<T>(3)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

      // Delete the last row
      equalityDeleteVectorMap.clear();
      equalityDeleteVectorMap.insert(
          {0,
           {{static_cast<T>(rowCount_ - 1)},
            {static_cast<T>((rowCount_ - 1) / 2)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

      // Delete non-existent values
      equalityDeleteVectorMap.clear();
      equalityDeleteVectorMap.insert(
          {0,
           {{static_cast<T>(rowCount_), static_cast<T>(rowCount_ + 1000)},
            {static_cast<T>(rowCount_ / 2), static_cast<T>(1500)}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

      // Delete 0 values
      equalityDeleteVectorMap.clear();
      equalityDeleteVectorMap.insert({0, {{}, {}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

      // Delete all values
      equalityDeleteVectorMap.clear();
      equalityDeleteVectorMap.insert(
          {0,
           {makeSequenceValues<T>(rowCount_),
            makeSequenceValues<T>(rowCount_, 2)}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0");
    }
  }

  template <TypeKind KIND>
  void testMultipleFileMultipleColumnEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();

    using T = typename TypeTraits<KIND>::NativeType;
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;

    // Test 1: Delete rows {0, 1} from c0, {2, 3} from c1, with two equality
    // delete files
    equalityFieldIdsMap.clear();
    equalityDeleteVectorMap.clear();
    equalityFieldIdsMap.insert({{0, {1}}, {1, {2}}});

    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();

      // Delete file 0: delete values from column 0 (indices 0, 1)
      std::vector<StringView> deleteValuesCol0 = {
          col0->valueAt(0), col0->valueAt(1)};

      // Delete file 1: delete values from column 1 (indices 2, 3)
      std::vector<StringView> deleteValuesCol1 = {
          col1->valueAt(2), col1->valueAt(3)};

      equalityDeleteVectorMap.insert(
          {{0, {deleteValuesCol0}}, {1, {deleteValuesCol1}}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      // Delete file 0: delete rows where c0 in {0, 1}
      // Delete file 1: delete rows where c1 in {2, 3} (which corresponds to
      // rows 4, 5, 6, 7)
      equalityDeleteVectorMap.insert(
          {{0, {{static_cast<T>(0), static_cast<T>(1)}}},
           {1, {{static_cast<T>(2), static_cast<T>(3)}}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 2: Delete no values (empty delete vectors for both files)
    equalityDeleteVectorMap.clear();
    equalityDeleteVectorMap.insert({{0, {{}}}, {1, {{}}}});
    assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);

    // Test 3: Delete all values
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();

      // Collect all unique values from column 0
      std::set<std::string> uniqueCol0Values;
      std::set<std::string> uniqueCol1Values;
      for (int i = 0; i < rowCount_; i++) {
        uniqueCol0Values.insert(std::string(col0->valueAt(i)));
        uniqueCol1Values.insert(std::string(col1->valueAt(i)));
      }

      std::vector<StringView> deleteValuesCol0;
      std::vector<StringView> deleteValuesCol1;

      for (const auto& value : uniqueCol0Values) {
        deleteValuesCol0.push_back(StringView(value));
      }
      for (const auto& value : uniqueCol1Values) {
        deleteValuesCol1.push_back(StringView(value));
      }

      equalityDeleteVectorMap.insert(
          {{0, {deleteValuesCol0}}, {1, {deleteValuesCol1}}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0",
          dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      // Delete all unique values from both columns
      equalityDeleteVectorMap.insert(
          {{0, {makeSequenceValues<T>(rowCount_)}},
           {1, {makeSequenceValues<T>(rowCount_, 2)}}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0");
    }

    // Test 4: Delete overlapping values from both files
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto dataVectors = makeVectors<KIND>(1, rowCount_, 2);
      auto col0 =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();
      auto col1 =
          dataVectors[0]->childAt(1)->template as<FlatVector<StringView>>();

      // Delete file 0: delete some values from column 0
      std::vector<StringView> deleteValuesCol0 = {
          col0->valueAt(0), col0->valueAt(2), col0->valueAt(4)};

      // Delete file 1: delete overlapping values from column 1
      std::vector<StringView> deleteValuesCol1 = {
          col1->valueAt(1), col1->valueAt(2), col1->valueAt(5)};

      equalityDeleteVectorMap.insert(
          {{0, {deleteValuesCol0}}, {1, {deleteValuesCol1}}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      // Delete file 0: delete some values from column 0
      // Delete file 1: delete some overlapping values from column 1
      auto randomIndices1 = makeRandomDeleteValues(rowCount_ / 4);
      auto randomIndices2 = makeRandomDeleteValues(rowCount_ / 4);

      std::vector<T> deleteValuesCol0;
      std::vector<T> deleteValuesCol1;

      for (auto idx : randomIndices1) {
        deleteValuesCol0.push_back(static_cast<T>(idx));
      }

      for (auto idx : randomIndices2) {
        deleteValuesCol1.push_back(static_cast<T>(idx / 2));
      }

      equalityDeleteVectorMap.insert(
          {{0, {deleteValuesCol0}}, {1, {deleteValuesCol1}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }

    // Test 5: Delete non-existent values from both files
    equalityDeleteVectorMap.clear();
    if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert(
          {{0, {{static_cast<T>(rowCount_), static_cast<T>(rowCount_ + 1)}}},
           {1,
            {{static_cast<T>(rowCount_ + 2), static_cast<T>(rowCount_ + 3)}}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    } else if constexpr (
        KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      equalityDeleteVectorMap.insert(
          {{0, {{StringView("nonexistent1"), StringView("nonexistent2")}}},
           {1, {{StringView("nonexistent3"), StringView("nonexistent4")}}}});
      assertEqualityDeletes<KIND>(equalityDeleteVectorMap, equalityFieldIdsMap);
    }
  }
};

TEST_P(
    IcebergReadEqualityDeletesParameterizedTest,
    singleColumnEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(testSingleColumnEqualityDeletes, typeKind);
}

TEST_P(IcebergReadEqualityDeletesParameterizedTest, twoColumnEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(testTwoColumnEqualityDeletes, typeKind);
}

TEST_P(
    IcebergReadEqualityDeletesParameterizedTest,
    singleFileMultipleColumnsEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      testSingleFileMultipleColumnsEqualityDeletes, typeKind);
}

TEST_P(
    IcebergReadEqualityDeletesParameterizedTest,
    multipleFileEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      testMultipleFileMultipleColumnEqualityDeletes, typeKind);
}

INSTANTIATE_TEST_SUITE_P(
    AllTypes,
    IcebergReadEqualityDeletesParameterizedTest,
    testing::Values(
        TypeKind::TINYINT,
        TypeKind::SMALLINT,
        TypeKind::INTEGER,
        TypeKind::BIGINT,
        TypeKind::VARCHAR,
        TypeKind::VARBINARY),
    [](const testing::TestParamInfo<TypeKind>& info) {
      return mapTypeKindToName(info.param);
    });

} // namespace facebook::velox::connector::hive::iceberg
