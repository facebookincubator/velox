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
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::connector::hive::iceberg {

struct TestParams {
  std::vector<TypeKind> columnTypes;
  std::vector<NullParam> nullParamForData;
};

// Helper function to get a string representation of NullParam for test names
std::string nullParamToString(NullParam param) {
  switch (param) {
    case NullParam::kNoNulls:
      return "NoNulls";
    case NullParam::kPartialNulls:
      return "PartialNulls";
    case NullParam::kAllNulls:
      return "AllNulls";
    default:
      return "Unknown";
  }
}

// Helper function to get a string representation of TestParams for test names
std::string testParamsToString(const TestParams& params) {
  std::string result;
  for (size_t i = 0; i < params.columnTypes.size(); ++i) {
    result += mapTypeKindToName(params.columnTypes[i]) +
        nullParamToString(params.nullParamForData[i]);
  }
  return result;
}

class IcebergReadEqualityDeleteTest
    : public IcebergTestBase,
      public testing::WithParamInterface<TestParams> {
  std::shared_ptr<TempFilePath> writeEqualityDeleteFile(
      const std::vector<RowVectorPtr>& deleteVectors) {
    VELOX_CHECK_GT(deleteVectors.size(), 0);

    // Combine all delete vectors into one
    auto deleteFilePath = TempFilePath::create();
    writeToFile(deleteFilePath->getPath(), deleteVectors);

    return deleteFilePath;
  }

  std::string makeTypePredicate(
      const std::string& columnName,
      TypeKind columnType,
      const std::string& valueStr) {
    switch (columnType) {
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        return fmt::format(
            "({} IS NULL OR {} <> '{}')", columnName, columnName, valueStr);
      case TypeKind::TINYINT:
      case TypeKind::SMALLINT:
      case TypeKind::INTEGER:
      case TypeKind::BIGINT:
      case TypeKind::REAL:
      case TypeKind::DOUBLE:
        return fmt::format(
            "({} IS NULL OR {} <> {})", columnName, columnName, valueStr);
      default:
        VELOX_FAIL(
            "Unsupported predicate type: {}", mapTypeKindToName(columnType));
    }
  }

  /// Generates equality-delete value column for the provided indices.
  ///
  /// Extracts values from \p dataColumn at the specified \p indices for the
  /// given \p kind, and appends a single matching FlatVector to
  /// \p deleteVectorColumns. The randomness, if any, is determined by how
  /// \p indices is produced by the caller.
  ///
  /// Supported kinds: TINYINT, SMALLINT, INTEGER, BIGINT, VARCHAR, VARBINARY.
  /// Throws for unsupported kinds.
  ///
  /// \param kind The TypeKind of the source column.
  /// \param dataColumn The source vector to read values from; must match \p
  /// kind.
  /// \param indices Row indices whose values are used to form the delete
  /// column.
  /// \param deleteVectorColumns Output vector where the constructed delete
  /// column is appended.
  void generateRandomDeleteValues(
      TypeKind kind,
      const VectorPtr& dataColumn,
      const std::vector<int64_t>& indices,
      std::vector<VectorPtr>& deleteVectorColumns) {
    switch (kind) {
      case TypeKind::TINYINT:
        generateDeleteColumns<TypeKind::TINYINT>(
            dataColumn, indices, deleteVectorColumns);
        break;
      case TypeKind::SMALLINT:
        generateDeleteColumns<TypeKind::SMALLINT>(
            dataColumn, indices, deleteVectorColumns);
        break;
      case TypeKind::INTEGER:
        generateDeleteColumns<TypeKind::INTEGER>(
            dataColumn, indices, deleteVectorColumns);
        break;
      case TypeKind::BIGINT:
        generateDeleteColumns<TypeKind::BIGINT>(
            dataColumn, indices, deleteVectorColumns);
        break;
      case TypeKind::VARCHAR:
        generateDeleteColumns<TypeKind::VARCHAR>(
            dataColumn, indices, deleteVectorColumns);
        break;
      case TypeKind::VARBINARY:
        generateDeleteColumns<TypeKind::VARBINARY>(
            dataColumn, indices, deleteVectorColumns);
        break;
      default:
        VELOX_FAIL("Unsupported type: {}", mapTypeKindToName(kind));
    }
  }

  /// \brief Generates delete columns for a given type KIND.
  ///
  /// This templated helper function extracts values from the provided
  /// data column at the specified indices and appends them as a new
  /// FlatVector to the deleteVectorColumns vector.
  ///
  /// @tparam KIND The TypeKind of the column.
  /// @param dataColumn The source vector containing data.
  /// @param indices The indices of rows to be deleted.
  /// @param deleteVectorColumns The vector to which the delete column is
  /// appended.
  template <TypeKind KIND>
  void generateDeleteColumns(
      const VectorPtr& dataColumn,
      const std::vector<int64_t>& indices,
      std::vector<VectorPtr>& deleteVectorColumns) {
    using T = TypeTraits<KIND>::NativeType;
    auto flatVector = dataColumn->as<FlatVector<T>>();

    std::vector<T> deleteValues;
    deleteValues.reserve(indices.size());

    for (auto idx : indices) {
      deleteValues.push_back(flatVector->valueAt(idx));
    }

    deleteVectorColumns.push_back(makeFlatVector<T>(deleteValues));
  }

 public:
  void assertEqualityDeletes(
      const std::vector<TypeKind>& columnTypes,
      const std::vector<NullParam>& nullParams,
      const std::vector<RowVectorPtr>& deleteVectors,
      const std::vector<int32_t>& equalityFieldIds,
      std::string duckDbSql = "",
      std::vector<RowVectorPtr> dataVectors = {}) {
    folly::SingletonVault::singleton()->registrationComplete();

    if (dataVectors.empty()) {
      dataVectors = makeVectors(1, rowCount_, columnTypes, nullParams);
    }

    // Write data file
    auto dataFilePath = TempFilePath::create();
    writeToFile(dataFilePath->getPath(), dataVectors);

    // Create DuckDB table for comparison
    createDuckDbTable(dataVectors);

    // Write delete file
    auto deleteFilePath = writeEqualityDeleteFile(deleteVectors);

    // Create Iceberg delete file info
    std::vector<connector::hive::iceberg::IcebergDeleteFile> deleteFiles;
    int64_t deleteFileSize = 0;
    for (auto& deleteVec : deleteVectors) {
      deleteFileSize += deleteVec->size();
    }

    IcebergDeleteFile deleteFile(
        FileContent::kEqualityDeletes,
        deleteFilePath->getPath(),
        fileFormat_,
        deleteFileSize,
        testing::internal::GetFileSize(
            std::fopen(deleteFilePath->getPath().c_str(), "r")),
        equalityFieldIds);
    deleteFiles.push_back(deleteFile);

    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    // Generate DuckDB query
    if (duckDbSql == "") {
      duckDbSql = "SELECT * FROM tmp ";
      if (deleteFileSize > 0) {
        std::string predicates =
            makeTypePredicates(deleteVectors, equalityFieldIds, columnTypes);
        if (!predicates.empty()) {
          duckDbSql += fmt::format("WHERE {}", predicates);
        }
      }
    }

    assertQuery(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);
  }

  void assertQuery(
      std::shared_ptr<connector::ConnectorSplit> split,
      const RowTypePtr& outputRowType,
      const std::string& duckDbSql) {
    auto plan = tableScanNode(outputRowType);
    auto task = OperatorTestBase::assertQuery(plan, {split}, duckDbSql);

    auto planStats = toPlanStats(task->taskStats());
    auto scanNodeId = plan->id();
    auto it = planStats.find(scanNodeId);
    ASSERT_TRUE(it != planStats.end());
    ASSERT_TRUE(it->second.peakMemoryBytes > 0);
  }

  void testSubFieldEqualityDelete() {
    TestParams params = GetParam();

    // Skip non-BIGINT types and non-NoNulls configurations for this test
    if (params.columnTypes.size() > 1 ||
        (params.columnTypes[0] != TypeKind::BIGINT ||
         params.nullParamForData[0] != NullParam::kNoNulls)) {
      GTEST_SKIP()
          << "This testcase is only tested against single BIGINT column with no nulls";
    }

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
              makeFlatVector<int64_t>(
                  20, [](auto row) { return row + 1; })})})};
    int32_t numDataColumns = 1;
    IcebergTestBase::WriteDataFilesConfig config;
    config.numRows = rowCount_;
    config.numColumns = numDataColumns;
    config.splitCount = 1;
    config.dataVectors = dataVectors;
    config.useConfigAndFlushPolicy = false;
    auto dataFilePaths = writeDataFiles(config);
    dataFilePath = dataFilePaths["data_file_0"];

    // Write the delete file. Equality delete field is c_row.c1
    std::vector<IcebergDeleteFile> deleteFiles;
    // Delete rows {0, 1} from c_row.c1, whose schema ID is 4
    std::vector<RowVectorPtr> deleteDataVectors = {makeRowVector(
        {"c1"},
        {makeFlatVector<int64_t>(2, [](auto row) { return row + 1; })})};

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

    auto icebergSplits =
        makeIcebergSplits(dataFilePath->getPath(), deleteFiles);

    // Select both c_bigint and c_row column columns
    std::string duckDbSql = "SELECT * FROM tmp WHERE c_row.c0 not in (1, 2)";
    assertQuery(
        icebergSplits.back(), asRowType(dataVectors[0]->type()), duckDbSql);

    // SELECT only c_bigint column
    duckDbSql = "SELECT c_bigint FROM tmp WHERE c_row.c0 not in (1, 2)";
    assertQuery(icebergSplits.back(), ROW({"c_bigint"}, {BIGINT()}), duckDbSql);
  }

  void testFloatAndDoubleThrowsError() {
    TestParams params = GetParam();

    if (!(params.columnTypes.size() == 1 &&
          (params.columnTypes[0] == TypeKind::REAL ||
           params.columnTypes[0] == TypeKind::DOUBLE) &&
          params.nullParamForData[0] == NullParam::kNoNulls)) {
      GTEST_SKIP()
          << "This testcase is only tested against single REAL or DOUBLE column with no nulls";
    }

    // Create delete vectors using makeVectors
    std::vector<RowVectorPtr> deleteVectors =
        makeVectors(1, 2, params.columnTypes, params.nullParamForData);

    std::vector<int32_t> equalityFieldIds;
    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      equalityFieldIds.push_back(static_cast<int32_t>(i + 1));
    }

    std::string expectedErrorMessage;
    if (params.columnTypes[0] == TypeKind::REAL) {
      expectedErrorMessage =
          "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: c0 : REAL";
      VELOX_ASSERT_THROW(
          assertEqualityDeletes(
              params.columnTypes,
              params.nullParamForData,
              deleteVectors,
              equalityFieldIds),
          expectedErrorMessage);
    } else {
      expectedErrorMessage =
          "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: c0 : DOUBLE";
      VELOX_ASSERT_THROW(
          assertEqualityDeletes(
              params.columnTypes,
              params.nullParamForData,
              deleteVectors,
              equalityFieldIds),
          expectedErrorMessage);
    }
  }

  void testDeleteFirstAndLastRows() {
    TestParams params = GetParam();

    if (params.columnTypes[0] == TypeKind::REAL ||
        params.columnTypes[0] == TypeKind::DOUBLE ||
        params.columnTypes[0] == TypeKind::HUGEINT) {
      GTEST_SKIP()
          << "Skipping unsupported types (REAL, DOUBLE, HUGEINT) for testDeleteFirstAndLastRows";
    }

    folly::SingletonVault::singleton()->registrationComplete();

    // Create test data using makeVectors
    std::vector<RowVectorPtr> dataVectors =
        makeVectors(1, rowCount_, params.columnTypes, params.nullParamForData);

    // Create delete vectors with first and last row values
    std::vector<VectorPtr> deleteVectorColumns;
    std::vector<std::string> columnNames;

    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      columnNames.push_back(fmt::format("c{}", i));
      auto dataColumn = dataVectors[0]->childAt(i);

      switch (params.columnTypes[i]) {
        case TypeKind::TINYINT: {
          auto flatVector = dataColumn->as<FlatVector<int8_t>>();
          std::vector<int8_t> deleteValues = {
              flatVector->valueAt(0), flatVector->valueAt(rowCount_ - 1)};
          deleteVectorColumns.push_back(makeFlatVector<int8_t>(deleteValues));
          break;
        }
        case TypeKind::SMALLINT: {
          auto flatVector = dataColumn->as<FlatVector<int16_t>>();
          std::vector<int16_t> deleteValues = {
              flatVector->valueAt(0), flatVector->valueAt(rowCount_ - 1)};
          deleteVectorColumns.push_back(makeFlatVector<int16_t>(deleteValues));
          break;
        }
        case TypeKind::INTEGER: {
          auto flatVector = dataColumn->as<FlatVector<int32_t>>();
          std::vector<int32_t> deleteValues = {
              flatVector->valueAt(0), flatVector->valueAt(rowCount_ - 1)};
          deleteVectorColumns.push_back(makeFlatVector<int32_t>(deleteValues));
          break;
        }
        case TypeKind::BIGINT: {
          auto flatVector = dataColumn->as<FlatVector<int64_t>>();
          std::vector<int64_t> deleteValues = {
              flatVector->valueAt(0), flatVector->valueAt(rowCount_ - 1)};
          deleteVectorColumns.push_back(makeFlatVector<int64_t>(deleteValues));
          break;
        }
        case TypeKind::VARCHAR:
        case TypeKind::VARBINARY: {
          auto flatVector = dataColumn->as<FlatVector<StringView>>();
          std::vector<StringView> deleteValues = {
              flatVector->valueAt(0), flatVector->valueAt(rowCount_ - 1)};
          deleteVectorColumns.push_back(
              makeFlatVector<StringView>(deleteValues));
          break;
        }
        default:
          VELOX_FAIL(
              "Unsupported type for testDeleteFirstAndLastRows: {}",
              mapTypeKindToName(params.columnTypes[i]));
      }
    }

    std::vector<RowVectorPtr> deleteVectors = {
        makeRowVector(columnNames, deleteVectorColumns)};

    // Create equality field IDs (all columns)
    std::vector<int32_t> equalityFieldIds;
    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      equalityFieldIds.push_back(static_cast<int32_t>(i + 1));
    }

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds);
  }

  void testDeleteRandomRows() {
    TestParams params = GetParam();

    // Skip floating point types for this test
    for (auto columnType : params.columnTypes) {
      if (columnType == TypeKind::REAL || columnType == TypeKind::DOUBLE ||
          columnType == TypeKind::HUGEINT) {
        GTEST_SKIP()
            << "Skipping unsupported types (REAL, DOUBLE, HUGEINT) for testDeleteRandomRows";
      }
    }

    folly::SingletonVault::singleton()->registrationComplete();

    // Create test data using makeVectors
    std::vector<RowVectorPtr> dataVectors =
        makeVectors(1, rowCount_, params.columnTypes, params.nullParamForData);

    // Generate random indices to delete
    auto randomIndices = makeRandomDeleteValues(rowCount_);

    // Create delete vectors with random row values
    std::vector<VectorPtr> deleteVectorColumns;
    std::vector<std::string> columnNames;

    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      columnNames.push_back(fmt::format("c{}", i));
      auto dataColumn = dataVectors[0]->childAt(i);

      generateRandomDeleteValues(
          params.columnTypes[i],
          dataColumn,
          randomIndices,
          deleteVectorColumns);
    }

    std::vector<RowVectorPtr> deleteVectors = {
        makeRowVector(columnNames, deleteVectorColumns)};

    // Create equality field IDs (all columns)
    std::vector<int32_t> equalityFieldIds;
    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      equalityFieldIds.push_back(static_cast<int32_t>(i + 1));
    }

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds);
  }

  void testDeleteAllRows() {
    TestParams params = GetParam();

    // Skip floating point types for this test
    for (auto columnType : params.columnTypes) {
      if (columnType == TypeKind::REAL || columnType == TypeKind::DOUBLE ||
          columnType == TypeKind::HUGEINT) {
        GTEST_SKIP()
            << "Skipping unsupported types (REAL, DOUBLE, HUGEINT) for testDeleteAllRows";
      }
    }

    folly::SingletonVault::singleton()->registrationComplete();

    // Create delete vectors with all actual values from data (not preserving
    // null pattern) The delete vectors should contain the actual data values,
    // not nulls
    std::vector<RowVectorPtr> deleteVectors = makeVectors(
        1,
        rowCount_,
        params.columnTypes,
        std::vector<NullParam>(params.columnTypes.size(), NullParam::kNoNulls));

    // Create equality field IDs (all columns)
    std::vector<int32_t> equalityFieldIds;
    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      equalityFieldIds.push_back(static_cast<int32_t>(i + 1));
    }

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds);
  }

  void testDeleteNoRows() {
    TestParams params = GetParam();

    // Skip floating point types for this test
    for (auto columnType : params.columnTypes) {
      if (columnType == TypeKind::REAL || columnType == TypeKind::DOUBLE ||
          columnType == TypeKind::HUGEINT) {
        GTEST_SKIP()
            << "Skipping unsupported types (REAL, DOUBLE, HUGEINT) for testDeleteNoRows";
        return;
      }
    }

    folly::SingletonVault::singleton()->registrationComplete();

    // Create empty delete vectors (no rows to delete)
    std::vector<VectorPtr> deleteVectorColumns;
    std::vector<std::string> columnNames;

    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      columnNames.push_back(fmt::format("c{}", i));

      switch (params.columnTypes[i]) {
        case TypeKind::TINYINT:
          deleteVectorColumns.push_back(
              makeFlatVector<int8_t>(std::vector<int8_t>{}));
          break;
        case TypeKind::SMALLINT:
          deleteVectorColumns.push_back(
              makeFlatVector<int16_t>(std::vector<int16_t>{}));
          break;
        case TypeKind::INTEGER:
          deleteVectorColumns.push_back(
              makeFlatVector<int32_t>(std::vector<int32_t>{}));
          break;
        case TypeKind::BIGINT:
          deleteVectorColumns.push_back(
              makeFlatVector<int64_t>(std::vector<int64_t>{}));
          break;
        case TypeKind::VARCHAR:
        case TypeKind::VARBINARY:
          deleteVectorColumns.push_back(
              makeFlatVector<StringView>(std::vector<StringView>{}));
          break;
        default:
          VELOX_FAIL(
              "Unsupported type for testDeleteNoRows: {}",
              mapTypeKindToName(params.columnTypes[i]));
      }
    }

    std::vector<RowVectorPtr> deleteVectors = {
        makeRowVector(columnNames, deleteVectorColumns)};

    // Create equality field IDs (all columns)
    std::vector<int32_t> equalityFieldIds;
    for (size_t i = 0; i < params.columnTypes.size(); ++i) {
      equalityFieldIds.push_back(static_cast<int32_t>(i + 1));
    }

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds);
  }

  void testShortDecimal() {
    TestParams params = GetParam();

    if (!(params.columnTypes.size() == 1 &&
          params.columnTypes[0] == TypeKind::BIGINT &&
          params.nullParamForData[0] == NullParam::kNoNulls)) {
      GTEST_SKIP()
          << "This testcase is only tested against BIGINT for short decimal";
    }

    folly::SingletonVault::singleton()->registrationComplete();

    // Use DECIMAL(6, 2) for short decimal (precision 6, scale 2)
    auto decimalType = DECIMAL(6, 2);
    std::vector<int32_t> equalityFieldIds = {1};

    // Test 1: Delete first and last short decimal values
    std::vector<RowVectorPtr> shortDecimalDataVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int64_t>(
            {123456, 789012, 345678, 901234, 567890}, decimalType)})};

    std::vector<RowVectorPtr> deleteVectors = {makeRowVector(
        {"c0"}, {makeFlatVector<int64_t>({123456, 789012}, decimalType)})};

    // Create DuckDB table for comparison
    createDuckDbTable(shortDecimalDataVectors);

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds,
        "SELECT * FROM tmp WHERE c0 NOT IN (1234.56, 7890.12)",
        shortDecimalDataVectors);

    // Test 2: Delete all short decimal values
    deleteVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int64_t>(
            {123456, 789012, 345678, 901234, 567890}, decimalType)})};

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds,
        "SELECT * FROM tmp WHERE 1 = 0",
        shortDecimalDataVectors);

    // Test 3: Delete none (empty short decimal delete vector)
    deleteVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int64_t>(std::vector<int64_t>{}, decimalType)})};

    assertEqualityDeletes(
        params.columnTypes,
        params.nullParamForData,
        deleteVectors,
        equalityFieldIds,
        "SELECT * FROM tmp",
        shortDecimalDataVectors);
  }

  void testLongDecimal() {
    TestParams params = GetParam();

    if (!(params.columnTypes.size() == 1 &&
          params.columnTypes[0] == TypeKind::HUGEINT &&
          params.nullParamForData[0] == NullParam::kNoNulls)) {
      GTEST_SKIP()
          << "This testcase is only tested against HUGEINT for long decimal";
    }

    folly::SingletonVault::singleton()->registrationComplete();

    // Use DECIMAL(25, 5) for long decimal (precision 25, scale 5)
    auto decimalType = DECIMAL(25, 5);
    std::vector<int32_t> equalityFieldIds = {1};

    // Test 1: Delete first two long decimal values
    // Values: 123456789012345 (represents 1234567890.12345), 987654321098765
    // (represents 9876543210.98765)
    std::vector<RowVectorPtr> longDecimalDataVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int128_t>(
            {int128_t(123456789012345),
             int128_t(987654321098765),
             int128_t(111111111111111),
             int128_t(222222222222222),
             int128_t(333333333333333)},
            decimalType)})};

    std::vector<RowVectorPtr> deleteVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int128_t>(
            {int128_t(123456789012345), int128_t(987654321098765)},
            decimalType)})};

    // Create DuckDB table for comparison
    createDuckDbTable(longDecimalDataVectors);

    VELOX_ASSERT_THROW(
        assertEqualityDeletes(
            params.columnTypes,
            params.nullParamForData,
            deleteVectors,
            equalityFieldIds,
            "SELECT * FROM tmp WHERE c0 NOT IN (1234567890.12345, 9876543210.98765)"),
        "Decimal is not supported for DWRF.");

    // Test 2: Delete all long decimal values
    deleteVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int128_t>(
            {int128_t(123456789012345),
             int128_t(987654321098765),
             int128_t(111111111111111),
             int128_t(222222222222222),
             int128_t(333333333333333)},
            decimalType)})};

    VELOX_ASSERT_THROW(
        assertEqualityDeletes(
            params.columnTypes,
            params.nullParamForData,
            deleteVectors,
            equalityFieldIds,
            "SELECT * FROM tmp WHERE 1 = 0"),
        "Decimal is not supported for DWRF.");

    // Test 3: Delete none (empty long decimal delete vector)
    deleteVectors = {makeRowVector(
        {"c0"},
        {makeFlatVector<int128_t>(std::vector<int128_t>{}, decimalType)})};

    VELOX_ASSERT_THROW(
        assertEqualityDeletes(
            params.columnTypes,
            params.nullParamForData,
            deleteVectors,
            equalityFieldIds,
            "SELECT * FROM tmp"),
        "Decimal is not supported for DWRF.");
  }
};

TEST_P(IcebergReadEqualityDeleteTest, testSubFieldEqualityDelete) {
  testSubFieldEqualityDelete();
}

TEST_P(IcebergReadEqualityDeleteTest, floatAndDoubleThrowsError) {
  testFloatAndDoubleThrowsError();
}

TEST_P(IcebergReadEqualityDeleteTest, deleteFirstAndLastRows) {
  testDeleteFirstAndLastRows();
}

TEST_P(IcebergReadEqualityDeleteTest, deleteRandomRows) {
  testDeleteRandomRows();
}

TEST_P(IcebergReadEqualityDeleteTest, deleteAllRows) {
  testDeleteAllRows();
}

TEST_P(IcebergReadEqualityDeleteTest, deleteNoRows) {
  testDeleteNoRows();
}

TEST_P(IcebergReadEqualityDeleteTest, shortDecimal) {
  testShortDecimal();
}

TEST_P(IcebergReadEqualityDeleteTest, LongDecimal) {
  testLongDecimal();
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    AllTest,
    IcebergReadEqualityDeleteTest,
    testing::Values(
        // Single row tests - No nulls
        TestParams{{TypeKind::TINYINT}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::SMALLINT}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::INTEGER}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::BIGINT}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::VARCHAR}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::VARBINARY}, {NullParam::kNoNulls}},

        // Single row tests - Partial nulls
        TestParams{{TypeKind::TINYINT}, {NullParam::kPartialNulls}},
        TestParams{{TypeKind::SMALLINT}, {NullParam::kPartialNulls}},
        TestParams{{TypeKind::INTEGER}, {NullParam::kPartialNulls}},
        TestParams{{TypeKind::BIGINT}, {NullParam::kPartialNulls}},
        TestParams{{TypeKind::VARCHAR}, {NullParam::kPartialNulls}},
        TestParams{{TypeKind::VARBINARY}, {NullParam::kPartialNulls}},

        // Single row tests - All nulls
        TestParams{{TypeKind::TINYINT}, {NullParam::kAllNulls}},
        TestParams{{TypeKind::SMALLINT}, {NullParam::kAllNulls}},
        TestParams{{TypeKind::INTEGER}, {NullParam::kAllNulls}},
        TestParams{{TypeKind::BIGINT}, {NullParam::kAllNulls}},
        TestParams{{TypeKind::VARCHAR}, {NullParam::kAllNulls}},
        TestParams{{TypeKind::VARBINARY}, {NullParam::kAllNulls}},

        // Failure testcase
        TestParams{{TypeKind::REAL}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::DOUBLE}, {NullParam::kNoNulls}},
        TestParams{{TypeKind::HUGEINT}, {NullParam::kNoNulls}},

        // Multiple row tests
        TestParams{
            {TypeKind::TINYINT, TypeKind::TINYINT},
            {NullParam::kNoNulls, NullParam::kNoNulls}},
        TestParams{
            {TypeKind::SMALLINT, TypeKind::SMALLINT},
            {NullParam::kPartialNulls, NullParam::kNoNulls}},
        TestParams{
            {TypeKind::INTEGER, TypeKind::INTEGER},
            {NullParam::kAllNulls, NullParam::kPartialNulls}},
        TestParams{
            {TypeKind::BIGINT, TypeKind::BIGINT},
            {NullParam::kNoNulls, NullParam::kNoNulls}},
        TestParams{
            {TypeKind::VARCHAR, TypeKind::VARCHAR},
            {NullParam::kPartialNulls, NullParam::kAllNulls}},
        TestParams{
            {TypeKind::VARBINARY, TypeKind::VARBINARY},
            {NullParam::kAllNulls, NullParam::kNoNulls}},

        // Mixed row type tests
        TestParams{
            {TypeKind::TINYINT, TypeKind::SMALLINT},
            {NullParam::kNoNulls, NullParam::kNoNulls}},
        TestParams{
            {TypeKind::SMALLINT, TypeKind::VARCHAR},
            {NullParam::kPartialNulls, NullParam::kNoNulls}},
        TestParams{
            {TypeKind::INTEGER, TypeKind::VARBINARY},
            {NullParam::kAllNulls, NullParam::kPartialNulls}},
        TestParams{
            {TypeKind::BIGINT, TypeKind::VARBINARY},
            {NullParam::kNoNulls, NullParam::kNoNulls}},

        // Three column mixed type tests
        TestParams{
            {TypeKind::INTEGER, TypeKind::VARCHAR, TypeKind::BIGINT},
            {NullParam::kNoNulls,
             NullParam::kPartialNulls,
             NullParam::kNoNulls}}),
    [](const testing::TestParamInfo<TestParams>& info) {
      return testParamsToString(info.param);
    });

} // namespace facebook::velox::connector::hive::iceberg
