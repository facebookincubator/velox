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

#include <exec/tests/utils/AssertQueryBuilder.h>
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::connector::hive::iceberg::test {

class IcebergSortOrderTest : public IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    rowType_ = ROW(
        {"c_int",
         "c_bigint",
         "c_varchar",
         "c_date",
         "c_decimal",
         "c_varbinary"},
        {INTEGER(), BIGINT(), VARCHAR(), DATE(), DECIMAL(18, 3), VARBINARY()});
  }

  RowTypePtr rowType_;

  // Verify data in the file is sorted according to the specified sort columns.
  void verifySortOrder(
      const std::string& dataPath,
    const std::vector<std::shared_ptr<const IcebergSortingColumn>>& sortedBy) {
    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_FALSE(splits.empty()) << "No data files found in " << dataPath;

    // Create a projection that selects all columns.
    std::vector<std::string> allColumns;
    for (auto i = 0; i < rowType_->size(); ++i) {
      allColumns.push_back(rowType_->nameOf(i));
    }

    auto plan =
        PlanBuilder().startTableScan().connectorId(test::kIcebergConnectorId)
        .outputType(rowType_).endTableScan().project(allColumns).planNode();
    auto result =
        AssertQueryBuilder(plan).splits(splits).copyResults(opPool_.get());

    ASSERT_GT(result->size(), 0) << "No rows found in the data file";

    // For each sort column, verify the data is sorted.
    for (const auto& sortExpr : sortedBy) {
      std::string columnName = sortExpr.get()->sortColumn();
      bool isAscending = sortExpr.get()->sortOrder().isAscending();
      bool isNullsFirst = sortExpr.get()->sortOrder().isNullsFirst();

      int32_t columnIndex = -1;
      for (auto i = 0; i < rowType_->size(); ++i) {
        if (rowType_->nameOf(i) == columnName) {
          columnIndex = i;
          break;
        }
      }
      ASSERT_NE(columnIndex, -1)
          << "Column " << columnName << " not found in row type";

      auto columnVector = result->childAt(columnIndex);
      bool hasNulls = false;
      bool hasNonNulls = false;
      vector_size_t firstNonNullIndex = 0;
      vector_size_t lastNullIndex = 0;

      for (auto i = 0; i < columnVector->size(); ++i) {
        if (columnVector->isNullAt(i)) {
          hasNulls = true;
          lastNullIndex = i;
        } else {
          if (!hasNonNulls) {
            firstNonNullIndex = i;
            hasNonNulls = true;
          }
        }
      }

      if (hasNulls && hasNonNulls) {
        if (isNullsFirst) {
          ASSERT_LT(lastNullIndex, firstNonNullIndex)
              << "NULL values should come before non-NULL values when NULLS FIRST is specified";
        } else {
          ASSERT_GT(lastNullIndex, firstNonNullIndex)
              << "NULL values should come after non-NULL values when NULLS LAST is specified";
        }
      }

      DecodedVector decoded;
      SelectivityVector rows(columnVector->size());
      decoded.decode(*columnVector, rows);

      for (auto i = 1; i < columnVector->size(); ++i) {
        // Skip if either current or previous is null.
        if (columnVector->isNullAt(i) || columnVector->isNullAt(i - 1)) {
          continue;
        }

        // Compare values based on type.
        int32_t comparison = 0;
        switch (auto kind = rowType_->childAt(columnIndex)->kind()) {
          case TypeKind::INTEGER: {
            auto prev = decoded.valueAt<int32_t>(i - 1);
            auto curr = decoded.valueAt<int32_t>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::BIGINT: {
            auto prev = decoded.valueAt<int64_t>(i - 1);
            auto curr = decoded.valueAt<int64_t>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::VARCHAR: {
            auto prev = decoded.valueAt<StringView>(i - 1);
            auto curr = decoded.valueAt<StringView>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::VARBINARY: {
            auto prev = decoded.valueAt<StringView>(i - 1);
            auto curr = decoded.valueAt<StringView>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::HUGEINT: {
            if (rowType_->childAt(columnIndex)->isLongDecimal()) {
              auto prev = decoded.valueAt<int128_t>(i - 1);
              auto curr = decoded.valueAt<int128_t>(i);
              comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            }
            break;
          }
          default:
            ASSERT_TRUE(false)
                << "Unsupported column type for sorting verification: " << kind;
        }

        if (isAscending) {
          ASSERT_LE(comparison, 0)
              << "Data not sorted in ascending order at row " << i
              << " for column " << columnName;
        } else {
          ASSERT_GE(comparison, 0)
              << "Data not sorted in descending order at row " << i
              << " for column " << columnName;
        }

        // If values are equal, continue to next row.
        if (comparison == 0) {
          continue;
        }
        break;
      }
    }
  }

  // Verify that data is sorted according to multiple sort columns.
  void verifyMultiColumnSortOrder(
      const std::string& dataPath,
    const std::vector<std::shared_ptr<const IcebergSortingColumn>>& sortedBy) {
    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_FALSE(splits.empty()) << "No data files found in " << dataPath;
    std::vector<std::string> allColumns;
    for (auto i = 0; i < rowType_->size(); ++i) {
      allColumns.push_back(rowType_->nameOf(i));
    }
    auto plan =
        PlanBuilder().startTableScan().connectorId(test::kIcebergConnectorId)
        .outputType(rowType_).endTableScan().project(allColumns).planNode();
    auto result =
        AssertQueryBuilder(plan).splits(splits).copyResults(opPool_.get());

    ASSERT_GT(result->size(), 0) << "No rows found in the data file";

    std::vector<std::string> columnNames;
    std::vector<bool> isAscending;
    std::vector<bool> isNullsFirst;
    std::vector<int32_t> columnIndices;

    for (const auto& sortExpr : sortedBy) {
      std::string columnName = sortExpr.get()->sortColumn();
      bool ascending = sortExpr.get()->sortOrder().isAscending();
      bool nullsFirst = sortExpr.get()->sortOrder().isNullsFirst();

      int32_t columnIndex = -1;
      for (auto i = 0; i < rowType_->size(); ++i) {
        if (rowType_->nameOf(i) == columnName) {
          columnIndex = i;
          break;
        }
      }
      ASSERT_NE(columnIndex, -1)
          << "Column " << columnName << " not found in row type";

      columnNames.push_back(columnName);
      isAscending.push_back(ascending);
      isNullsFirst.push_back(nullsFirst);
      columnIndices.push_back(columnIndex);
    }

    // Verify the sort order row by row.
    for (auto i = 1; i < result->size(); ++i) {
      // Compare row i-1 with row i using all sort columns in order.
      for (size_t colIdx = 0; colIdx < columnIndices.size(); ++colIdx) {
        int32_t columnIndex = columnIndices[colIdx];
        auto columnVector = result->childAt(columnIndex);
        bool ascending = isAscending[colIdx];
        bool nullsFirst = isNullsFirst[colIdx];
        bool prevIsNull = columnVector->isNullAt(i - 1);
        bool currIsNull = columnVector->isNullAt(i);

        if (prevIsNull && currIsNull) {
          // Both null, continue to next column.
          continue;
        } else if (prevIsNull) {
          // Previous is null, current is not.
          ASSERT_TRUE(nullsFirst)
              << "NULL values should come last at row " << (i - 1)
              << " for column " << columnNames[colIdx] << " in " << dataPath;
          break;
        } else if (currIsNull) {
          // Current is null, previous is not.
          ASSERT_FALSE(nullsFirst)
              << "NULL values should come first at row " << i << " for column "
              << columnNames[colIdx] << " in " << dataPath;
          break;
        }

        // Both values are non-null, compare them.
        DecodedVector decoded;
        SelectivityVector rows(columnVector->size());
        decoded.decode(*columnVector, rows);

        int32_t comparison = 0;
        switch (auto kind = rowType_->childAt(columnIndex)->kind()) {
          case TypeKind::INTEGER: {
            auto prev = decoded.valueAt<int32_t>(i - 1);
            auto curr = decoded.valueAt<int32_t>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::BIGINT: {
            auto prev = decoded.valueAt<int64_t>(i - 1);
            auto curr = decoded.valueAt<int64_t>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::VARCHAR: {
            auto prev = decoded.valueAt<StringView>(i - 1);
            auto curr = decoded.valueAt<StringView>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::VARBINARY: {
            auto prev = decoded.valueAt<StringView>(i - 1);
            auto curr = decoded.valueAt<StringView>(i);
            comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            break;
          }
          case TypeKind::HUGEINT: {
            if (rowType_->childAt(columnIndex)->isLongDecimal()) {
              auto prev = decoded.valueAt<int128_t>(i - 1);
              auto curr = decoded.valueAt<int128_t>(i);
              comparison = prev < curr ? -1 : (prev > curr ? 1 : 0);
            }
            break;
          }
          default:
            ASSERT_TRUE(false)
                << "Unsupported column type for sorting verification: " << kind;
        }

        if (comparison != 0) {
          if (ascending) {
            ASSERT_LE(comparison, 0)
                << "Data not sorted in ascending order at row " << i
                << " for column " << columnNames[colIdx] << " in " << dataPath
                << ". Previous value: " << columnVector->toString(i - 1)
                << ", Current value: " << columnVector->toString(i);
          } else {
            ASSERT_GE(comparison, 0)
                << "Data not sorted in descending order at row " << i
                << " for column " << columnNames[colIdx] << " in " << dataPath
                << ". Previous value: " << columnVector->toString(i - 1)
                << ", Current value: " << columnVector->toString(i);
          }
          // Found definitive ordering, no need to check further columns.
          break;
        }
        // If values are equal, continue to next column.
      }
      // Rows can be equal across all sort columns.
    }
  }

  static constexpr auto numBatches = 10;
  static constexpr auto rowsPerBatch = 1'000;

  void testSorting(
    const std::vector<std::shared_ptr<const IcebergSortingColumn>>& sortedBy,
      double nullRatio = 0.0) {
    std::vector<RowVectorPtr> vectors =
        createTestData(rowType_, numBatches, rowsPerBatch, nullRatio);
    auto outputDirectory = TempDirectoryPath::create();

    auto dataSink = createDataSinkWithSorting(
        rowType_, outputDirectory->getPath(), {}, sortedBy);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    dataSink->close();
    if (sortedBy.size() == 1) {
      verifySortOrder(outputDirectory->getPath(), sortedBy);
    } else {
      verifyMultiColumnSortOrder(outputDirectory->getPath(), sortedBy);
    }
  }

  void testSortingWithPartitioning(
      const std::vector<PartitionField>& partitionTransforms,
      const std::vector<std::shared_ptr<const IcebergSortingColumn>>& sortedBy,
      const double nullRatio = 0.0) {
    std::vector<RowVectorPtr> vectors =
        createTestData(rowType_, numBatches, rowsPerBatch, nullRatio);
    const auto outputDirectory = TempDirectoryPath::create();

    auto dataSink = createDataSinkWithSorting(
        rowType_,
        outputDirectory->getPath(),
        partitionTransforms,
        sortedBy);

    for (const auto& vector : vectors) {
      dataSink->appendData(vector);
    }

    ASSERT_TRUE(dataSink->finish());
    dataSink->close();

    // For partitioned data, we need to find all partition directories.
    std::vector<std::string> partitionDirs;
    std::function<void(const std::string&)> findLeafDataDirs =
        [&partitionDirs, &findLeafDataDirs](const std::string& dir) {
          bool hasSubDirs = false;

          for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) {
              hasSubDirs = true;
              findLeafDataDirs(entry.path().string());
            }
          }
          if (!hasSubDirs) {
            partitionDirs.push_back(dir);
          }
        };

    // Start the recursive search from the data directory.
    if (std::filesystem::exists(outputDirectory->getPath())) {
      findLeafDataDirs(outputDirectory->getPath());
    }
    if (partitionDirs.empty()) {
      partitionDirs.push_back(outputDirectory->getPath());
    }

    // Verify each partition directory has properly sorted data.
    ASSERT_FALSE(partitionDirs.empty()) << "No partition directories found";
    for (const auto& partitionDir : partitionDirs) {
      if (sortedBy.size() == 1) {
        verifySortOrder(partitionDir, sortedBy);
      } else {
        verifyMultiColumnSortOrder(partitionDir, sortedBy);
      }
    }
  }
};

static std::shared_ptr<const IcebergSortingColumn> sc(
    const std::string& column,
    bool ascending = true,
    bool nullsFirst = false) {
  return std::make_shared<const IcebergSortingColumn>(
      column, core::SortOrder(ascending, nullsFirst));
}

TEST_F(IcebergSortOrderTest, singleColumnSortDefault) {
  testSorting({sc("c_int")});
  testSorting({sc("c_bigint")});
  testSorting({sc("c_varchar")});
  testSorting({sc("c_date")});
  testSorting({sc("c_decimal")});
  testSorting({sc("c_varbinary")});
}

TEST_F(IcebergSortOrderTest, singleColumnSortDesc) {
  testSorting({sc("c_int", false)});
  testSorting({sc("c_bigint", false)});
  testSorting({sc("c_varchar", false)});
  testSorting({sc("c_date", false)});
  testSorting({sc("c_decimal", false)});
  testSorting({sc("c_varbinary", false)});
}

TEST_F(IcebergSortOrderTest, nullOrderingFirst) {
  testSorting({sc("c_int", true, true)}, 0.2);
  testSorting({sc("c_bigint", true, true)}, 0.2);
  testSorting({sc("c_varchar", true, true)}, 0.3);
  testSorting({sc("c_date", true, true)}, 0.3);
  testSorting({sc("c_decimal", true, true)}, 0.2);
  testSorting({sc("c_varbinary", true, true)}, 0.2);

  testSorting({sc("c_varbinary", false, true)}, 0.2);
  testSorting({sc("c_int", false, true)}, 0.2);
  testSorting({sc("c_bigint", false, true)}, 0.2);
  testSorting({sc("c_varchar", false, true)}, 0.3);
  testSorting({sc("c_date", false, true)}, 0.3);
  testSorting({sc("c_decimal", false, true)}, 0.2);
}

TEST_F(IcebergSortOrderTest, nullOrderingLast) {
  testSorting({sc("c_int", true, false)}, 0.2);
  testSorting({sc("c_bigint", true, false)}, 0.2);
  testSorting({sc("c_varchar", true, false)}, 0.2);
  testSorting({sc("c_date", true, false)}, 0.2);
  testSorting({sc("c_decimal", true, false)}, 0.2);
  testSorting({sc("c_varbinary", true, false)}, 0.2);

  testSorting({sc("c_varbinary", false, false)}, 0.2);
  testSorting({sc("c_int", false, false)}, 0.2);
  testSorting({sc("c_bigint", false, false)}, 0.2);
  testSorting({sc("c_varchar", false, false)}, 0.2);
  testSorting({sc("c_date", false, false)}, 0.2);
  testSorting({sc("c_decimal", false, false)}, 0.2);
}

TEST_F(IcebergSortOrderTest, multiColumnSort) {
  testSorting({sc("c_int"), sc("c_bigint", false)});
  testSorting({sc("c_int"), sc("c_bigint")});
  testSorting({sc("c_int", false), sc("c_bigint", false)});
  testSorting({sc("c_int", false), sc("c_bigint")});

  testSorting({sc("c_int"), sc("c_varchar", false)});
  testSorting({sc("c_int"), sc("c_varchar")});
  testSorting({sc("c_int", false), sc("c_varchar", false)});
  testSorting({sc("c_int", false), sc("c_varchar")});

  testSorting({sc("c_varchar"), sc("c_date", false)});
  testSorting({sc("c_varchar"), sc("c_date")});
  testSorting({sc("c_varchar", false), sc("c_date", false)});
  testSorting({sc("c_varchar", false), sc("c_date")});

  testSorting({sc("c_int"), sc("c_decimal", false)});
  testSorting({sc("c_decimal"), sc("c_varbinary", false)});
  testSorting({sc("c_varbinary"), sc("c_decimal", false)});
}

TEST_F(IcebergSortOrderTest, multiColumnSortWithNull) {
  testSorting({sc("c_int"), sc("c_bigint"), sc("c_varchar")}, 0.2);
  testSorting({sc("c_int"), sc("c_bigint", false, false)}, 0.4);
  testSorting(
      {sc("c_int", true, true),
       sc("c_bigint", false, false),
       sc("c_varchar", true, true)},
      0.2);
  testSorting(
      {sc("c_int", true, true), sc("c_bigint"), sc("c_varchar", true, true)},
      0.2);
  testSorting(
      {sc("c_int", true, false), sc("c_bigint"), sc("c_varchar", true, true)},
      0.2);
  testSorting(
      {sc("c_int", false, false), sc("c_bigint"), sc("c_varchar", true, false)},
      0.2);
  testSorting(
      {sc("c_int", true, true), sc("c_bigint", false), sc("c_varchar", true, true)},
      0.2);
  testSorting(
      {sc("c_int", true, true),
       sc("c_bigint", false, false),
       sc("c_varchar", true, true)},
      0.2);
  testSorting(
      {sc("c_int", true, true), sc("c_bigint", false), sc("c_varchar", true, false)},
      0.2);
  testSorting(
      {sc("c_int", true, true),
       sc("c_bigint", false, false),
       sc("c_varchar", false, true)},
      0.2);

  testSorting(
      {sc("c_int", true, true),
       sc("c_decimal", false, false),
       sc("c_varbinary", true, true)},
      0.2);
}

TEST_F(IcebergSortOrderTest, sortWithSinglePartitioning) {
  testSortingWithPartitioning({{3, TransformType::kBucket, 5}}, {sc("c_int")});
  testSortingWithPartitioning({{0, TransformType::kBucket, 7}}, {sc("c_varchar")});
}

TEST_F(IcebergSortOrderTest, sortWithPartitioningOnSameColumn) {
  testSortingWithPartitioning({{3, TransformType::kBucket, 5}}, {sc("c_date")});
  testSortingWithPartitioning({{0, TransformType::kBucket, 7}}, {sc("c_int")});
  testSortingWithPartitioning({{2, TransformType::kBucket, 4}}, {sc("c_varchar", false)});
}

TEST_F(IcebergSortOrderTest, sortWithMultiPartitioning) {
  testSortingWithPartitioning(
      {{3, TransformType::kBucket, 3}, {2, TransformType::kBucket, 4}},
      {sc("c_int"), sc("c_bigint", false)});

  testSortingWithPartitioning(
      {{2, TransformType::kTruncate, 1}},
      {sc("c_int"), sc("c_bigint", false)});
}

TEST_F(IcebergSortOrderTest, sortWithPartitioningAndNulls) {
  testSortingWithPartitioning(
      {{0, TransformType::kBucket, 8}},
      {sc("c_int", true, true), sc("c_bigint", false, false)},
      0.2);

  testSortingWithPartitioning(
      {{2, TransformType::kBucket, 8}},
      {sc("c_varchar", true, true), sc("c_int", false, false)},
      0.2);

  testSortingWithPartitioning(
      {{4, TransformType::kBucket, 8}},
      {sc("c_decimal", true, true), sc("c_int", false, false)},
      0.3);

  testSortingWithPartitioning(
      {{5, TransformType::kBucket, 8}},
      {sc("c_varbinary", true, true), sc("c_int", false, false)},
      0.3);
}

} // namespace facebook::velox::connector::hive::iceberg::test