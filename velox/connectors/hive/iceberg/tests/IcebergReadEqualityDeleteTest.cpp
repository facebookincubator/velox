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
#include "velox/connectors/hive/iceberg/tests/IcebergReadTestBase.h"
#include "velox/exec/PlanNodeStats.h"

namespace facebook::velox::connector::hive::iceberg {

class IcebergReadEqualityDeleteTest : public IcebergReadTestBase {
 protected:
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

    std::vector<connector::hive::iceberg::IcebergDeleteFile> deleteFiles;
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

  /// Generate test data vectors with configurable null patterns
  template <TypeKind KIND>
  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      int32_t numColumns = 1,
      bool allNulls = false,
      bool partialNull = false) {
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

        if (allNulls) {
          // Use allNullFlatVector for all-null columns
          columnVector = vectorMaker_.allNullFlatVector<T>(rowsPerVector);
        } else if constexpr (KIND == TypeKind::VARCHAR) {
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

        // Apply partial nulls by randomly setting some positions to null
        if (partialNull && !allNulls) {
          std::mt19937 gen(42); // Fixed seed for reproducibility
          std::uniform_real_distribution<> dis(0.0, 1.0);
          const double nullProbability = 0.2;

          for (vector_size_t idx = 0; idx < rowsPerVector; ++idx) {
            if (dis(gen) < nullProbability) {
              columnVector->setNull(idx, true);
            }
          }
        }

        vectors.push_back(columnVector);
      }

      rowVectors.push_back(makeRowVector(names, vectors));
    }

    rowType_ = std::make_shared<RowType>(std::move(names), std::move(types));

    return rowVectors;
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

 private:
  /// Write equality delete file with typed data
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

  /// Create typed predicate string for DuckDB queries
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

  /// Generate DuckDB predicates for equality delete testing
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
          "({} IS NULL OR {} NOT IN ({}))",
          name,
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
};

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

TEST_F(
    IcebergReadEqualityDeleteTest,
    equalityDeletesFloatAndDoubleThrowsError) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Test for float (REAL)
  {
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<float>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({{0, {1}}, {1, {2}}});
    equalityDeleteVectorMap.insert({{0, {{0, 1}}}, {1, {{2, 3}}}});
    VELOX_ASSERT_THROW(
        assertEqualityDeletes<TypeKind::REAL>(
            equalityDeleteVectorMap, equalityFieldIdsMap),
        "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: c1 : REAL");
  }

  // Test for float (REAL) - Delete all
  {
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<float>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1}});
    std::vector<float> allValues;
    for (int i = 0; i < rowCount_; ++i) {
      allValues.push_back(static_cast<double>(i));
    }
    equalityDeleteVectorMap.insert({0, {allValues}});
    VELOX_ASSERT_THROW(
        assertEqualityDeletes<TypeKind::REAL>(
            equalityDeleteVectorMap, equalityFieldIdsMap),
        "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: c0 : REAL");
  }

  // Test for double (DOUBLE)
  {
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<double>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({{0, {1}}, {1, {2}}});
    equalityDeleteVectorMap.insert({{0, {{0, 1}}}, {1, {{2, 3}}}});
    VELOX_ASSERT_THROW(
        assertEqualityDeletes<TypeKind::DOUBLE>(
            equalityDeleteVectorMap, equalityFieldIdsMap),
        "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: c1 : DOUBLE");
  }

  // Test for double (DOUBLE) - Delete all
  {
    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<double>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1}});
    std::vector<double> allValues;
    for (int i = 0; i < rowCount_; ++i) {
      allValues.push_back(static_cast<double>(i));
    }
    equalityDeleteVectorMap.insert({0, {allValues}});
    VELOX_ASSERT_THROW(
        assertEqualityDeletes<TypeKind::DOUBLE>(
            equalityDeleteVectorMap, equalityFieldIdsMap),
        "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: c0 : DOUBLE");
  }
}

TEST_F(IcebergReadEqualityDeleteTest, equalityDeletesShortDecimal) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Use DECIMAL(6, 2) for short decimal (precision 6, scale 2)
  auto decimalType = DECIMAL(6, 2);
  std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
  std::unordered_map<int8_t, std::vector<std::vector<int64_t>>>
      equalityDeleteVectorMap;
  equalityFieldIdsMap.insert({0, {1}});

  // Values: 123456 (represents 1234.56), 789012 (represents 7890.12)
  equalityDeleteVectorMap.insert({0, {{123456, 789012}}});
  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {"c0"},
      {makeFlatVector<int64_t>(
          {123456, 789012, 345678, 901234, 567890}, decimalType)})};

  assertEqualityDeletes<TypeKind::BIGINT>(
      equalityDeleteVectorMap,
      equalityFieldIdsMap,
      "SELECT * FROM tmp WHERE c0 NOT IN (1234.56, 7890.12)",
      dataVectors);

  // Delete all
  equalityDeleteVectorMap.clear();
  equalityDeleteVectorMap.insert(
      {0, {{123456, 789012, 345678, 901234, 567890}}});

  assertEqualityDeletes<TypeKind::BIGINT>(
      equalityDeleteVectorMap,
      equalityFieldIdsMap,
      "SELECT * FROM tmp WHERE 1 = 0",
      dataVectors);

  // Delete none
  equalityDeleteVectorMap.clear();
  equalityDeleteVectorMap.insert({0, {{}}});
  assertEqualityDeletes<TypeKind::BIGINT>(
      equalityDeleteVectorMap,
      equalityFieldIdsMap,
      "SELECT * FROM tmp",
      dataVectors);
}

TEST_F(IcebergReadEqualityDeleteTest, equalityDeletesLongDecimal) {
  folly::SingletonVault::singleton()->registrationComplete();

  // Use DECIMAL(25, 5) for long decimal (precision 25, scale 5)
  auto decimalType = DECIMAL(25, 5);
  std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
  std::unordered_map<int8_t, std::vector<std::vector<int128_t>>>
      equalityDeleteVectorMap;
  equalityFieldIdsMap.insert({0, {1}});

  // Values: 123456789012345 (represents 1234567.89012), 987654321098765
  // (represents 9876543.21098)
  equalityDeleteVectorMap.insert(
      {0, {{int128_t(123456789012345), int128_t(987654321098765)}}});
  std::vector<RowVectorPtr> dataVectors = {makeRowVector(
      {"c0"},
      {makeFlatVector<int128_t>(
          {(123456789012345),
           (987654321098765),
           (111111111111111),
           (222222222222222),
           (333333333333333)},
          decimalType)})};

  VELOX_ASSERT_THROW(
      assertEqualityDeletes<TypeKind::HUGEINT>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE c0 NOT IN (123456789012345, 987654321098765)",
          dataVectors),
      "Decimal is not supported for DWRF.");

  // Delete all
  equalityDeleteVectorMap.clear();
  equalityDeleteVectorMap.insert(
      {0,
       {{123456789012345,
         987654321098765,
         111111111111111,
         222222222222222,
         333333333333333}}});

  VELOX_ASSERT_THROW(
      assertEqualityDeletes<TypeKind::HUGEINT>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp WHERE 1 = 0",
          dataVectors),
      "Decimal is not supported for DWRF.");

  // Delete none
  equalityDeleteVectorMap.clear();
  equalityDeleteVectorMap.insert({0, {{}}});
  VELOX_ASSERT_THROW(
      assertEqualityDeletes<TypeKind::HUGEINT>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp",
          dataVectors),
      "Decimal is not supported for DWRF.");
}

class IcebergReadEqualityDeletesParameterizedTest
    : public IcebergReadEqualityDeleteTest,
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

  template <TypeKind KIND>
  void testAllNullsEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();
    using T = typename TypeTraits<KIND>::NativeType;

    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1}});

    // Test with all-null data vectors
    auto dataVectors = makeVectors<KIND>(1, rowCount_, 1, true, false);

    // Since all values are null, delete operations should not match anything
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      std::vector<StringView> deleteValues = {
          StringView("apple"), StringView("banana")};
      equalityDeleteVectorMap.insert({0, {deleteValues}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp",
          dataVectors);
    } else if constexpr (std::is_integral_v<T>) {
      equalityDeleteVectorMap.insert(
          {0, {{static_cast<T>(0), static_cast<T>(1)}}});
      assertEqualityDeletes<KIND>(
          equalityDeleteVectorMap,
          equalityFieldIdsMap,
          "SELECT * FROM tmp",
          dataVectors);
    }
  }

  template <TypeKind KIND>
  void testPartialNullsEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();
    using T = typename TypeTraits<KIND>::NativeType;

    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1}});

    // Test with partial-null data vectors (20% nulls)
    auto dataVectors = makeVectors<KIND>(1, rowCount_, 1, false, true);

    // Delete some actual values that should exist in the non-null positions
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto flatVector =
          dataVectors[0]->childAt(0)->template as<FlatVector<StringView>>();

      // Find a few non-null values to delete
      std::vector<StringView> deleteValues;
      for (vector_size_t i = 0; i < rowCount_ && deleteValues.size() < 5; ++i) {
        if (!flatVector->isNullAt(i)) {
          deleteValues.push_back(flatVector->valueAt(i));
        }
      }

      if (!deleteValues.empty()) {
        equalityDeleteVectorMap.insert({0, {deleteValues}});
        assertEqualityDeletes<KIND>(
            equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
      }
    } else if constexpr (std::is_integral_v<T>) {
      auto flatVector =
          dataVectors[0]->childAt(0)->template as<FlatVector<T>>();

      // Find a few non-null values to delete
      std::vector<T> deleteValues;
      for (vector_size_t i = 0; i < rowCount_ && deleteValues.size() < 5; ++i) {
        if (!flatVector->isNullAt(i)) {
          deleteValues.push_back(flatVector->valueAt(i));
        }
      }

      if (!deleteValues.empty()) {
        equalityDeleteVectorMap.insert({0, {deleteValues}});
        assertEqualityDeletes<KIND>(
            equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
      }
    }
  }

  template <TypeKind KIND>
  void testMixedNullTypesEqualityDeletes() {
    folly::SingletonVault::singleton()->registrationComplete();
    using T = typename TypeTraits<KIND>::NativeType;

    std::unordered_map<int8_t, std::vector<int32_t>> equalityFieldIdsMap;
    std::unordered_map<int8_t, std::vector<std::vector<T>>>
        equalityDeleteVectorMap;
    equalityFieldIdsMap.insert({0, {1, 2}});

    // Test with different null patterns for each column
    // Column 0: all nulls, Column 1: partial nulls
    auto allNullVectors = makeVectors<KIND>(1, rowCount_, 1, true, false);
    auto partialNullVectors = makeVectors<KIND>(1, rowCount_, 1, false, true);

    // Combine the vectors to create a mixed scenario
    std::vector<VectorPtr> combinedVectors = {
        allNullVectors[0]->childAt(0), partialNullVectors[0]->childAt(0)};

    std::vector<std::string> names = {"c0", "c1"};
    auto combinedRowVector = makeRowVector(names, combinedVectors);
    std::vector<RowVectorPtr> dataVectors = {combinedRowVector};

    // Delete operations should only affect the partial null column
    equalityDeleteVectorMap.clear();
    if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
      auto col1FlatVector = partialNullVectors[0]
                                ->childAt(0)
                                ->template as<FlatVector<StringView>>();

      // Find non-null values from column 1 to delete
      std::vector<StringView> deleteValuesCol0; // Empty for all-null column
      std::vector<StringView> deleteValuesCol1;

      for (vector_size_t i = 0; i < rowCount_ && deleteValuesCol1.size() < 3;
           ++i) {
        if (!col1FlatVector->isNullAt(i)) {
          deleteValuesCol1.push_back(col1FlatVector->valueAt(i));
        }
      }

      if (!deleteValuesCol1.empty()) {
        equalityDeleteVectorMap.insert(
            {0, {deleteValuesCol0, deleteValuesCol1}});
        assertEqualityDeletes<KIND>(
            equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
      }
    } else if constexpr (std::is_integral_v<T>) {
      auto col1FlatVector =
          partialNullVectors[0]->childAt(0)->template as<FlatVector<T>>();

      // Find non-null values from column 1 to delete
      std::vector<T> deleteValuesCol0; // Empty for all-null column
      std::vector<T> deleteValuesCol1;

      for (vector_size_t i = 0; i < rowCount_ && deleteValuesCol1.size() < 3;
           ++i) {
        if (!col1FlatVector->isNullAt(i)) {
          deleteValuesCol1.push_back(col1FlatVector->valueAt(i));
        }
      }

      if (!deleteValuesCol1.empty()) {
        equalityDeleteVectorMap.insert(
            {0, {deleteValuesCol0, deleteValuesCol1}});
        assertEqualityDeletes<KIND>(
            equalityDeleteVectorMap, equalityFieldIdsMap, "", dataVectors);
      }
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

TEST_P(IcebergReadEqualityDeletesParameterizedTest, allNullsEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(testAllNullsEqualityDeletes, typeKind);
}

TEST_P(
    IcebergReadEqualityDeletesParameterizedTest,
    partialNullsEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(testPartialNullsEqualityDeletes, typeKind);
}

TEST_P(
    IcebergReadEqualityDeletesParameterizedTest,
    mixedNullTypesEqualityDeletes) {
  TypeKind typeKind = GetParam();
  VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      testMixedNullTypesEqualityDeletes, typeKind);
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
