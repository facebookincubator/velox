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

#include <gtest/gtest.h>

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

#ifdef VELOX_ENABLE_PARQUET

class IcebergParquetStatsTest : public test::IcebergTestBase {
 protected:
  // Write data and get all stats (for partitioned tables).
  std::vector<std::shared_ptr<dwio::common::IcebergDataFileStatistics>>
  writeDataAndGetAllStats(
      const RowVectorPtr& data,
      const std::vector<test::PartitionField>& partitionFields = {}) {
    const auto outputDir = exec::test::TempDirectoryPath::create();
    auto dataSink = createDataSinkAndAppendData(
        {data}, outputDir->getPath(), partitionFields);
    auto commitTasks = dataSink->close();
    EXPECT_FALSE(commitTasks.empty());
    return dataSink->dataFileStats();
  }

  // Decode and extract typed value from base64 encoded bounds.
  template <typename T>
  static std::pair<T, T> decodeBounds(
      const std::shared_ptr<dwio::common::IcebergDataFileStatistics>& stats,
      int32_t fieldId) {
    auto decode = [](const std::string& base64Encoded) {
      const std::string decoded = encoding::Base64::decode(base64Encoded);
      T value;
      std::memcpy(&value, decoded.data(), sizeof(T));
      return value;
    };

    const auto& columnStats = stats->columnStats.at(fieldId);
    VELOX_CHECK(columnStats.lowerBound.has_value());
    VELOX_CHECK(columnStats.upperBound.has_value());
    return {
        decode(columnStats.lowerBound.value()),
        decode(columnStats.upperBound.value()),
    };
  }

  // Verify basic statistics (record count, value counts, null counts).
  static void verifyBasicStats(
      const std::shared_ptr<dwio::common::IcebergDataFileStatistics>& stats,
      int64_t expectedRecords,
      const std::unordered_map<int32_t, int64_t>& expectedValueCounts,
      const std::unordered_map<int32_t, int64_t>& expectedNullCounts) {
    EXPECT_EQ(stats->numRecords, expectedRecords);

    for (const auto& [fieldId, count] : expectedValueCounts) {
      ASSERT_TRUE(stats->columnStats.contains(fieldId));
      EXPECT_EQ(stats->columnStats.at(fieldId).valueCount, count);
    }

    if (!expectedNullCounts.empty()) {
      for (const auto& [fieldId, count] : expectedNullCounts) {
        ASSERT_TRUE(stats->columnStats.contains(fieldId));
        EXPECT_EQ(stats->columnStats.at(fieldId).nullValueCount, count);
      }
    }
  }

  // Verify bounds exist for given field IDs.
  static void verifyBoundsExist(
      const std::shared_ptr<dwio::common::IcebergDataFileStatistics>& stats,
      const std::vector<int32_t>& fieldIds) {
    for (const int32_t fieldId : fieldIds) {
      ASSERT_TRUE(stats->columnStats.contains(fieldId));
      const auto& columnStats = stats->columnStats.at(fieldId);
      ASSERT_TRUE(columnStats.lowerBound.has_value());
      ASSERT_TRUE(columnStats.upperBound.has_value());
      EXPECT_FALSE(columnStats.lowerBound.value().empty());
      EXPECT_FALSE(columnStats.upperBound.value().empty());
    }
  }

  // Verify bounds do not exist for given field IDs.
  static void verifyBoundsNotExist(
      const std::shared_ptr<dwio::common::IcebergDataFileStatistics>& stats,
      const std::vector<int32_t>& fieldIds) {
    for (const int32_t fieldId : fieldIds) {
      if (stats->columnStats.contains(fieldId)) {
        const auto& columnStats = stats->columnStats.at(fieldId);
        ASSERT_FALSE(columnStats.lowerBound.has_value());
        ASSERT_FALSE(columnStats.upperBound.has_value());
      }
    }
  }
};

TEST_F(IcebergParquetStatsTest, mixedNull) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedIntNulls = 34;
  constexpr int32_t intColId = 1;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int32_t>(
          size, [](vector_size_t row) { return row * 10; }, nullEvery(3))}));
  verifyBasicStats(
      stats[0], size, {{intColId, size}}, {{intColId, expectedIntNulls}});
  verifyBoundsExist(stats[0], {intColId});

  const auto& [minVal, maxVal] = decodeBounds<int32_t>(stats[0], intColId);
  EXPECT_EQ(minVal, 10);
  EXPECT_EQ(maxVal, 980);
}

TEST_F(IcebergParquetStatsTest, bigint) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 25;
  constexpr int32_t bigintColId = 1;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int64_t>(
          size,
          [](vector_size_t row) { return row * 1'000'000'000LL; },
          nullEvery(4))}));
  verifyBasicStats(
      stats[0], size, {{bigintColId, size}}, {{bigintColId, expectedNulls}});
  verifyBoundsExist(stats[0], {bigintColId});

  const auto& [minVal, maxVal] = decodeBounds<int64_t>(stats[0], bigintColId);
  EXPECT_EQ(minVal, 1'000'000'000LL);
  EXPECT_EQ(maxVal, 99'000'000'000LL);
}

TEST_F(IcebergParquetStatsTest, decimal) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 20;
  constexpr int32_t decimalColId = 1;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int128_t>(
          size,
          [](vector_size_t row) { return HugeInt::build(row, row * 123); },
          nullEvery(5),
          DECIMAL(38, 3))}));
  verifyBasicStats(
      stats[0], size, {{decimalColId, size}}, {{decimalColId, expectedNulls}});
  verifyBoundsExist(stats[0], {decimalColId});
}

TEST_F(IcebergParquetStatsTest, varchar) {
  constexpr vector_size_t size = 100;
  constexpr int32_t varcharColId = 1;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<std::string>(
          size,
          [](vector_size_t row) {
            return "Customer#00000" + std::to_string(row) + "_" +
                std::string(row % 10, 'a');
          },
          nullEvery(6))}));

  constexpr int32_t expectedNulls = 17;
  verifyBasicStats(
      stats[0], size, {{varcharColId, size}}, {{varcharColId, expectedNulls}});
  verifyBoundsExist(stats[0], {varcharColId});

  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(varcharColId).lowerBound.value()),
      "Customer#0000010");
  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(varcharColId).upperBound.value()),
      "Customer#000009`");
}

TEST_F(IcebergParquetStatsTest, varbinary) {
  constexpr vector_size_t size = 100;
  constexpr int32_t varbinaryColId = 1;

  auto rowVector = makeRowVector({makeFlatVector<std::string>(
      size,
      [](vector_size_t row) {
        std::string value(17, 11);
        value[0] = static_cast<char>(row % 256);
        value[1] = static_cast<char>((row * 3) % 256);
        value[2] = static_cast<char>((row * 7) % 256);
        value[3] = static_cast<char>((row * 11) % 256);
        return value;
      },
      nullEvery(5),
      VARBINARY())});

  const auto& stats = writeDataAndGetAllStats(rowVector);
  constexpr int32_t expectedNulls = 20;
  verifyBasicStats(
      stats[0],
      size,
      {{varbinaryColId, size}},
      {{varbinaryColId, expectedNulls}});
  verifyBoundsExist(stats[0], {varbinaryColId});
}

TEST_F(IcebergParquetStatsTest, varbinaryWithTransform) {
  const auto& fileStats = writeDataAndGetAllStats(
      makeRowVector({makeFlatVector<std::string>(
          {"01020304",
           "05060708",
           "090A0B0C",
           "0D0E0F10",
           "11121314",
           "15161718",
           "191A1B1C",
           "1D1E1F20",
           "21222324",
           "25262728"},
          VARBINARY())}),
      {{0, TransformType::kBucket, 4}});
  ASSERT_EQ(fileStats.size(), 3);
  const auto& stats = fileStats[0];
  EXPECT_EQ(stats->numRecords, 5);
  constexpr int32_t varbinaryColId = 1;
  EXPECT_EQ(stats->columnStats.at(varbinaryColId).valueCount, 5);
}

TEST_F(IcebergParquetStatsTest, multipleDataTypes) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t bigintColId = 2;
  constexpr int32_t decimalColId = 3;
  constexpr int32_t varcharColId = 4;
  constexpr int32_t varbinaryColId = 5;

  constexpr int32_t expectedIntNulls = 34;
  constexpr int32_t expectedBigintNulls = 25;
  constexpr int32_t expectedDecimalNulls = 20;
  constexpr int32_t expectedVarcharNulls = 17;
  constexpr int32_t expectedVarbinaryNulls = 15;

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(
           size, [](vector_size_t row) { return row * 10; }, nullEvery(3)),
       makeFlatVector<int64_t>(
           size,
           [](vector_size_t row) { return row * 1'000'000'000LL; },
           nullEvery(4)),
       makeFlatVector<int128_t>(
           size,
           [](vector_size_t row) { return HugeInt::build(row, row * 12'345); },
           nullEvery(5),
           DECIMAL(38, 3)),
       makeFlatVector<std::string>(
           size,
           [](vector_size_t row) { return "str_" + std::to_string(row); },
           nullEvery(6)),
       makeFlatVector<std::string>(
           size,
           [](vector_size_t row) {
             std::string value(4, 0);
             value[0] = static_cast<char>(row % 256);
             value[1] = static_cast<char>((row * 3) % 256);
             value[2] = static_cast<char>((row * 7) % 256);
             value[3] = static_cast<char>((row * 11) % 256);
             return value;
           },
           nullEvery(7),
           VARBINARY())});
  const auto& stats = writeDataAndGetAllStats(rowVector);

  verifyBasicStats(
      stats[0],
      size,
      {
          {intColId, size},
          {bigintColId, size},
          {decimalColId, size},
          {varcharColId, size},
          {varbinaryColId, size},
      },
      {
          {intColId, expectedIntNulls},
          {bigintColId, expectedBigintNulls},
          {decimalColId, expectedDecimalNulls},
          {varcharColId, expectedVarcharNulls},
          {varbinaryColId, expectedVarbinaryNulls},
      });

  verifyBoundsExist(
      stats[0],
      {intColId, bigintColId, decimalColId, varcharColId, varbinaryColId});
}

TEST_F(IcebergParquetStatsTest, date) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 20;
  constexpr int32_t dateColId = 1;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int32_t>(
          size,
          [](vector_size_t row) { return 18262 + row; },
          nullEvery(5),
          DATE())}));
  verifyBasicStats(
      stats[0], size, {{dateColId, size}}, {{dateColId, expectedNulls}});
  verifyBoundsExist(stats[0], {dateColId});

  const auto& [minVal, maxVal] = decodeBounds<int32_t>(stats[0], dateColId);
  EXPECT_EQ(minVal, 18263);
  EXPECT_EQ(maxVal, 18262 + 99);
}

TEST_F(IcebergParquetStatsTest, boolean) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 10;
  constexpr int32_t boolColId = 1;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<bool>(
          size,
          [](vector_size_t row) { return row % 2 == 1; },
          nullEvery(10),
          BOOLEAN())}));
  verifyBasicStats(
      stats[0], size, {{boolColId, size}}, {{boolColId, expectedNulls}});
  verifyBoundsExist(stats[0], {boolColId});

  // For boolean, the lower bound should be false (0) and upper bound should be
  // true (1) if both values are present.
  const auto& [minVal, maxVal] = decodeBounds<bool>(stats[0], boolColId);
  EXPECT_FALSE(minVal);
  EXPECT_TRUE(maxVal);
}

TEST_F(IcebergParquetStatsTest, empty) {
  constexpr vector_size_t size = 0;

  const auto& stats = writeDataAndGetAllStats(makeRowVector(
      {makeFlatVector<int32_t>(0), makeFlatVector<StringView>(0)}));
  EXPECT_EQ(stats[0]->numRecords, size);
  ASSERT_TRUE(stats[0]->columnStats.empty());
}

TEST_F(IcebergParquetStatsTest, nullValues) {
  constexpr vector_size_t size = 100;

  const auto& stats = writeDataAndGetAllStats(makeRowVector(
      {makeNullConstant(TypeKind::INTEGER, size),
       makeNullConstant(TypeKind::VARCHAR, size)}));
  EXPECT_EQ(stats[0]->numRecords, size);
  ASSERT_EQ(stats[0]->columnStats.at(1).nullValueCount, size);
  // Do not collect lower and upper bounds for NULLs.
  for (const auto& [fieldId, columnStats] : stats[0]->columnStats) {
    ASSERT_FALSE(columnStats.lowerBound.has_value());
    ASSERT_FALSE(columnStats.upperBound.has_value());
  }
}

TEST_F(IcebergParquetStatsTest, real) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 20;
  constexpr int32_t realColId = 1;
  int32_t expectedNaNs = 0;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<float>(
          size,
          [&expectedNaNs](vector_size_t row) {
            if (row % 6 == 0) {
              expectedNaNs++;
              return std::numeric_limits<float>::quiet_NaN();
            }
            return row * 1.5f;
          },
          nullEvery(5),
          REAL())}));
  verifyBasicStats(
      stats[0], size, {{realColId, size}}, {{realColId, expectedNulls}});

  EXPECT_EQ(
      stats[0]->columnStats.at(realColId).nanValueCount.value_or(0),
      expectedNaNs);
  verifyBoundsExist(stats[0], {realColId});
  const auto& [minVal, maxVal] = decodeBounds<float>(stats[0], realColId);
  EXPECT_FLOAT_EQ(minVal, 1.5f);
  EXPECT_FLOAT_EQ(maxVal, 148.5f);
}

TEST_F(IcebergParquetStatsTest, double) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 15;
  constexpr int32_t doubleColId = 1;
  int32_t expectedNaNs = 0;

  auto rowVector = makeRowVector({makeFlatVector<double>(
      size,
      [&expectedNaNs](vector_size_t row) {
        if (row % 3 == 0) {
          expectedNaNs++;
          return std::numeric_limits<double>::quiet_NaN();
        }
        if (row % 4 == 0) {
          return std::numeric_limits<double>::infinity();
        }
        if (row % 5 == 0) {
          return -std::numeric_limits<double>::infinity();
        }
        return row * 2.5;
      },
      nullEvery(7),
      DOUBLE())});

  const auto& stats = writeDataAndGetAllStats(rowVector);
  verifyBasicStats(
      stats[0], size, {{doubleColId, size}}, {{doubleColId, expectedNulls}});

  EXPECT_EQ(
      stats[0]->columnStats.at(doubleColId).nanValueCount.value_or(0),
      expectedNaNs);

  verifyBoundsExist(stats[0], {doubleColId});

  // Verify bounds are set correctly and NaN/infinity values don't affect
  // min/max incorrectly.
  const auto& [minVal, maxVal] = decodeBounds<double>(stats[0], doubleColId);
  EXPECT_DOUBLE_EQ(minVal, -std::numeric_limits<double>::infinity())
      << "Lower bound should be -infinity";
  EXPECT_DOUBLE_EQ(maxVal, std::numeric_limits<double>::infinity())
      << "Upper bound should be infinity";
}

TEST_F(IcebergParquetStatsTest, mixedDoubleFloat) {
  constexpr vector_size_t size = 6;

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(size, [](vector_size_t row) { return 1; }),
       makeFlatVector<float>(
           size,
           [](vector_size_t row) {
             return -std::numeric_limits<float>::infinity();
           }),
       makeFlatVector<double>(
           size,
           [](vector_size_t row) {
             return std::numeric_limits<double>::infinity();
           }),
       makeFlatVector<double>(size, [](vector_size_t row) {
         switch (row) {
           case 0:
             return 1.23;
           case 1:
             return -1.23;
           case 2:
             return std::numeric_limits<double>::infinity();
           case 3:
             return 2.23;
           case 4:
             return -std::numeric_limits<double>::infinity();
           default:
             return -2.23;
         }
       })});

  const auto& stats = writeDataAndGetAllStats(rowVector);
  constexpr int32_t doubleColId = 4;
  verifyBasicStats(stats[0], size, {{doubleColId, size}}, {{doubleColId, 0}});
  const auto& [minVal, maxVal] = decodeBounds<double>(stats[0], doubleColId);
  EXPECT_DOUBLE_EQ(minVal, -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(maxVal, std::numeric_limits<double>::infinity());

  constexpr int32_t floatColId = 2;
  const auto& [minFloatVal, maxFloatVal] =
      decodeBounds<float>(stats[0], floatColId);
  EXPECT_FLOAT_EQ(minFloatVal, -std::numeric_limits<float>::infinity());
  EXPECT_FLOAT_EQ(maxFloatVal, -std::numeric_limits<float>::infinity());
}

TEST_F(IcebergParquetStatsTest, NaN) {
  constexpr vector_size_t size = 1'000;
  constexpr int32_t expectedNulls = 500;
  constexpr int32_t doubleColId = 1;
  int32_t expectedNaNs = 0;

  const auto& stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<double>(
          size,
          [&expectedNaNs](vector_size_t /*row*/) {
            expectedNaNs++;
            return std::numeric_limits<double>::quiet_NaN();
          },
          nullEvery(2),
          DOUBLE())}));
  verifyBasicStats(
      stats[0], size, {{doubleColId, size}}, {{doubleColId, expectedNulls}});

  EXPECT_EQ(
      stats[0]->columnStats.at(doubleColId).nanValueCount.value_or(0),
      expectedNaNs);
  // Do not collect bounds for NULLs and NaNs.
  for (const auto& [fieldId, columnStats] : stats[0]->columnStats) {
    ASSERT_FALSE(columnStats.lowerBound.has_value());
    ASSERT_FALSE(columnStats.upperBound.has_value());
  }
}

TEST_F(IcebergParquetStatsTest, partitionedTable) {
  std::vector<test::PartitionField> partitionTransforms = {
      {0, TransformType::kBucket, 4},
      {1, TransformType::kDay, std::nullopt},
      {2, TransformType::kTruncate, 2},
  };

  constexpr vector_size_t size = 100;

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; }),
       makeFlatVector<int32_t>(
           size,
           [](vector_size_t row) { return 18262 + (row % 5); },
           nullptr,
           DATE()),
       makeFlatVector<std::string>(size, [](vector_size_t row) {
         return fmt::format("str{}", row % 10);
       })});

  const auto& fileStats =
      writeDataAndGetAllStats(rowVector, partitionTransforms);

  EXPECT_GT(fileStats.size(), 1)
      << "Expected multiple files due to partitioning";

  for (const auto& stats : fileStats) {
    EXPECT_GT(stats->numRecords, 0);
    ASSERT_FALSE(stats->columnStats.empty());

    constexpr int32_t intColId = 1;
    constexpr int32_t dateColId = 2;
    constexpr int32_t varcharColId = 3;
    EXPECT_EQ(stats->columnStats.at(intColId).valueCount, stats->numRecords);
    EXPECT_EQ(stats->columnStats.at(dateColId).valueCount, stats->numRecords);
    EXPECT_EQ(
        stats->columnStats.at(varcharColId).valueCount, stats->numRecords);

    for (const auto fieldId : {intColId, dateColId, varcharColId}) {
      const auto& columnStats = stats->columnStats.at(fieldId);
      ASSERT_TRUE(columnStats.lowerBound.has_value());
      ASSERT_TRUE(columnStats.upperBound.has_value());
      EXPECT_FALSE(columnStats.lowerBound.value().empty());
      EXPECT_FALSE(columnStats.upperBound.value().empty());
    }
  }

  // Verify total record count across all partitions.
  int64_t totalRecords = 0;
  for (const auto& stats : fileStats) {
    totalRecords += stats->numRecords;
  }
  EXPECT_EQ(totalRecords, size);
}

TEST_F(IcebergParquetStatsTest, multiplePartitionTransforms) {
  std::vector<test::PartitionField> partitionTransforms = {
      {0, TransformType::kBucket, 2},
      {1, TransformType::kYear, std::nullopt},
      {2, TransformType::kTruncate, 3},
      {3, TransformType::kIdentity, std::nullopt}};

  constexpr vector_size_t size = 100;

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(
           size, [](vector_size_t row) { return row * 10; }),
       makeFlatVector<int32_t>(
           size,
           [](vector_size_t row) { return 18262 + (row * 100); },
           nullptr,
           DATE()),
       makeFlatVector<std::string>(
           size,
           [](vector_size_t row) {
             return fmt::format("prefix{}_value", row % 5);
           }),
       makeFlatVector<int64_t>(
           size, [](vector_size_t row) { return (row % 3) * 1'000; })});

  const auto& fileStats =
      writeDataAndGetAllStats(rowVector, partitionTransforms);
  EXPECT_GT(fileStats.size(), 1)
      << "Expected multiple files due to partitioning";
  // Check each file's stats.
  for (const auto& stats : fileStats) {
    EXPECT_GT(stats->numRecords, 0);
    constexpr int32_t intColId = 1;
    constexpr int32_t dateColId = 2;
    constexpr int32_t bigintColId = 4;

    if (stats->columnStats.contains(intColId)) {
      const auto& [minVal, maxVal] = decodeBounds<int32_t>(stats, intColId);
      EXPECT_LE(minVal, maxVal)
          << "Lower bound should be <= upper bound for int column";
    }

    if (stats->columnStats.contains(dateColId)) {
      const auto& [minVal, maxVal] = decodeBounds<int32_t>(stats, dateColId);
      EXPECT_LE(minVal, maxVal)
          << "Lower bound should be <= upper bound for date column";
    }

    if (stats->columnStats.contains(bigintColId)) {
      const auto& [minVal, maxVal] = decodeBounds<int64_t>(stats, bigintColId);
      EXPECT_LE(minVal, maxVal)
          << "Lower bound should be <= upper bound for bigint column";
    }
  }
  int64_t totalRecords = 0;
  for (const auto& stats : fileStats) {
    totalRecords += stats->numRecords;
  }
  EXPECT_EQ(totalRecords, size);
}

TEST_F(IcebergParquetStatsTest, partitionedTableWithNulls) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedIntNulls = 20;
  constexpr int32_t expectedDateNulls = 15;
  constexpr int32_t expectedVarcharNulls = 10;

  std::vector<test::PartitionField> partitionTransforms = {
      {0, TransformType::kIdentity, std::nullopt},
      {1, TransformType::kMonth, std::nullopt},
      {2, TransformType::kTruncate, 2}};
  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(
           size,
           [](vector_size_t row) { return row % 10; },
           nullEvery(5),
           INTEGER()),
       makeFlatVector<int32_t>(
           size,
           [](vector_size_t row) { return 18262 + (row % 3) * 30; },
           nullEvery(7),
           DATE()),
       makeFlatVector<std::string>(
           size,
           [](vector_size_t row) { return fmt::format("val{}", row % 5); },
           nullEvery(11))});
  const auto& fileStats =
      writeDataAndGetAllStats(rowVector, partitionTransforms);
  int32_t totalIntNulls = 0;
  int32_t totalDateNulls = 0;
  int32_t totalVarcharNulls = 0;
  int32_t totalRecords = 0;

  constexpr int32_t intColId = 1;
  constexpr int32_t dateColId = 2;
  constexpr int32_t varcharColId = 3;

  for (const auto& stats : fileStats) {
    totalRecords += stats->numRecords;
    // Add null counts if present.
    if (stats->columnStats.contains(intColId)) {
      totalIntNulls += stats->columnStats.at(intColId).nullValueCount;
    }

    if (stats->columnStats.contains(dateColId)) {
      totalDateNulls += stats->columnStats.at(dateColId).nullValueCount;
    }

    if (stats->columnStats.contains(varcharColId)) {
      totalVarcharNulls += stats->columnStats.at(varcharColId).nullValueCount;
    }
  }

  // Verify total counts match expected.
  EXPECT_EQ(totalRecords, size);
  EXPECT_EQ(totalIntNulls, expectedIntNulls);
  EXPECT_EQ(totalDateNulls, expectedDateNulls);
  EXPECT_EQ(totalVarcharNulls, expectedVarcharNulls);
}

TEST_F(IcebergParquetStatsTest, mapType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t mapValueColId = 3; // Map value field ID.

  std::vector<std::optional<
      std::vector<std::pair<int32_t, std::optional<std::string>>>>>
      mapData;
  for (auto i = 0; i < size; ++i) {
    std::vector<std::pair<int32_t, std::optional<std::string>>> mapRow;
    for (auto j = 0; j < 5; ++j) {
      mapRow.emplace_back(j, fmt::format("value_{}", i * 5 + j));
    }
    mapData.push_back(std::move(mapRow));
  }

  const auto& stats = writeDataAndGetAllStats(makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
      makeNullableMapVector<int32_t, std::string>(mapData),
  }));
  verifyBasicStats(stats[0], size, {{intColId, size}}, {{intColId, 0}});

  EXPECT_EQ(stats[0]->columnStats.at(mapValueColId).valueCount, size * 5);
  // Map values have stats but no bounds (skipBounds=true for maps).
  verifyBoundsNotExist(stats[0], {mapValueColId});
}

TEST_F(IcebergParquetStatsTest, arrayType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t arrayElementColId = 3; // Array element field ID.

  std::vector<std::vector<std::optional<std::string>>> arrayData;
  for (auto i = 0; i < size; ++i) {
    std::vector<std::optional<std::string>> arrayRow;
    for (auto j = 0; j < 3; ++j) {
      arrayRow.emplace_back(fmt::format("item_{}", i * 3 + j));
    }
    arrayData.push_back(std::move(arrayRow));
  }

  const auto& stats = writeDataAndGetAllStats(makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
       makeNullableArrayVector<std::string>(arrayData)}));
  verifyBasicStats(stats[0], size, {{intColId, size}}, {{intColId, 0}});

  EXPECT_EQ(stats[0]->columnStats.at(arrayElementColId).valueCount, size * 3);
  // Array elements have stats but no bounds (skipBounds=true for arrays).
  verifyBoundsNotExist(stats[0], {arrayElementColId});
}

// Test statistics collection for nested struct fields.
// Field ID assignment:
//   int_col: 1
//   struct_col: 2 (parent, no stats)
//     first_level_id: 3
//     first_level_name: 4
//     nested_struct: 5 (parent, no stats)
//       second_level_id: 6
//       second_level_name: 7
// Statistics collected for leaf fields: [1, 3, 4, 6, 7]
TEST_F(IcebergParquetStatsTest, structType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t firstLevelIdColId = 3;
  constexpr int32_t secondLevelIdColId = 6;
  constexpr int32_t secondLevelNameColId = 7;

  auto firstLevelId = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row % size; }, nullEvery(5));

  auto firstLevelName = makeFlatVector<std::string>(
      size,
      [](vector_size_t row) { return fmt::format("name_{}", row * 10); },
      nullEvery(7));

  auto secondLevelId = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row * size; }, nullEvery(6));

  auto secondLevelName = makeFlatVector<std::string>(
      size,
      [](vector_size_t row) { return fmt::format("nested_{}", row * 100); },
      nullEvery(8));

  auto nestedStruct = makeRowVector({secondLevelId, secondLevelName});
  auto structVector =
      makeRowVector({firstLevelId, firstLevelName, nestedStruct});
  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
       structVector});

  const auto& stats = writeDataAndGetAllStats(rowVector);
  EXPECT_EQ(stats[0]->numRecords, size);
  EXPECT_EQ(stats[0]->columnStats.size(), 5);

  verifyBasicStats(
      stats[0],
      size,
      {{intColId, size}, {firstLevelIdColId, size}, {secondLevelIdColId, size}},
      {{intColId, 0}, {firstLevelIdColId, 20}});

  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(secondLevelNameColId).lowerBound.value()),
      "nested_100");
  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(secondLevelNameColId).upperBound.value()),
      "nested_9900");
}

#endif

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
