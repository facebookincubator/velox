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

#include <folly/Conv.h>
#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergDataFileStatistics.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Reconstructs IcebergDataFileStatistics from the JSON metrics that the sink
// emits in each commit task. Mirrors the helper in IcebergParquetStatsTest so
// the DWRF stats path is exercised end-to-end through commitMessage().
class IcebergDwrfStatsTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    dwrf::registerDwrfReaderFactory();
    dwrf::registerDwrfWriterFactory();
    fileFormat_ = dwio::common::FileFormat::DWRF;
  }

  static IcebergDataFileStatisticsPtr statsFromMetrics(
      const folly::dynamic& metrics) {
    VELOX_CHECK(metrics.isObject());
    VELOX_CHECK(metrics.count("recordCount") > 0);
    auto stats = std::make_shared<IcebergDataFileStatistics>();
    stats->numRecords = metrics["recordCount"].asInt();

    auto setIntField = [&](const folly::dynamic& map, auto setter) {
      if (!map.isObject()) {
        return;
      }
      for (const auto& item : map.items()) {
        const auto fieldId = folly::to<int32_t>(item.first.asString());
        auto& column = stats->columnStats[fieldId];
        setter(column, item.second);
      }
    };

    setIntField(metrics["columnSizes"], [](auto& column, const auto& value) {
      column.columnSize = value.asInt();
    });
    setIntField(metrics["valueCounts"], [](auto& column, const auto& value) {
      column.valueCount = value.asInt();
    });
    setIntField(
        metrics["nullValueCounts"], [](auto& column, const auto& value) {
          column.nullValueCount = value.asInt();
        });

    const auto& lowerBounds = metrics["lowerBounds"];
    if (lowerBounds.isObject()) {
      for (const auto& item : lowerBounds.items()) {
        const auto fieldId = folly::to<int32_t>(item.first.asString());
        stats->columnStats[fieldId].lowerBound = item.second.asString();
      }
    }
    const auto& upperBounds = metrics["upperBounds"];
    if (upperBounds.isObject()) {
      for (const auto& item : upperBounds.items()) {
        const auto fieldId = folly::to<int32_t>(item.first.asString());
        stats->columnStats[fieldId].upperBound = item.second.asString();
      }
    }

    return stats;
  }

  static std::vector<IcebergDataFileStatisticsPtr> statsFromCommitTasks(
      const std::vector<std::string>& commitTasks) {
    std::vector<IcebergDataFileStatisticsPtr> stats;
    stats.reserve(commitTasks.size());
    for (const auto& task : commitTasks) {
      auto taskJson = folly::parseJson(task);
      VELOX_CHECK(taskJson.isObject());
      VELOX_CHECK(taskJson.count("metrics") > 0);
      stats.emplace_back(statsFromMetrics(taskJson["metrics"]));
    }
    return stats;
  }

  std::vector<IcebergDataFileStatisticsPtr> writeDataAndGetAllStats(
      const RowVectorPtr& data,
      const std::vector<test::PartitionField>& partitionFields = {}) {
    const auto outputDir = TempDirectoryPath::create();
    auto dataSink = createDataSinkAndAppendData(
        {data}, outputDir->getPath(), partitionFields);
    auto commitTasks = dataSink->close();
    EXPECT_FALSE(commitTasks.empty());
    return statsFromCommitTasks(commitTasks);
  }

  // Decodes a little-endian fixed-width bound back to a typed value.
  template <typename T>
  static std::pair<T, T> decodeBounds(
      const IcebergDataFileStatisticsPtr& stats,
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
        decode(columnStats.upperBound.value())};
  }

  static void verifyBasicStats(
      const IcebergDataFileStatisticsPtr& stats,
      int64_t expectedRecords,
      const std::unordered_map<int32_t, int64_t>& expectedValueCounts,
      const std::unordered_map<int32_t, int64_t>& expectedNullCounts) {
    EXPECT_EQ(stats->numRecords, expectedRecords);
    for (const auto& [fieldId, count] : expectedValueCounts) {
      ASSERT_TRUE(stats->columnStats.contains(fieldId));
      EXPECT_EQ(stats->columnStats.at(fieldId).valueCount, count);
    }
    for (const auto& [fieldId, count] : expectedNullCounts) {
      ASSERT_TRUE(stats->columnStats.contains(fieldId));
      EXPECT_EQ(stats->columnStats.at(fieldId).nullValueCount, count);
    }
  }

  static void verifyBoundsExist(
      const IcebergDataFileStatisticsPtr& stats,
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

  static void verifyBoundsNotExist(
      const IcebergDataFileStatisticsPtr& stats,
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

TEST_F(IcebergDwrfStatsTest, integer) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 34;
  constexpr int32_t intColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int32_t>(
          size, [](vector_size_t row) { return row * 10; }, nullEvery(3))}));
  verifyBasicStats(
      stats[0], size, {{intColId, size}}, {{intColId, expectedNulls}});
  verifyBoundsExist(stats[0], {intColId});

  const auto [minVal, maxVal] = decodeBounds<int32_t>(stats[0], intColId);
  EXPECT_EQ(minVal, 10);
  EXPECT_EQ(maxVal, 980);
}

TEST_F(IcebergDwrfStatsTest, smallint) {
  constexpr vector_size_t size = 100;
  constexpr int32_t smallintColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int16_t>(
          size,
          [](vector_size_t row) { return static_cast<int16_t>(row); },
          nullEvery(7))}));
  verifyBoundsExist(stats[0], {smallintColId});

  // SMALLINT serializes to Iceberg int (4-byte little-endian).
  const auto [minVal, maxVal] = decodeBounds<int32_t>(stats[0], smallintColId);
  EXPECT_EQ(minVal, 1);
  EXPECT_EQ(maxVal, 99);
}

TEST_F(IcebergDwrfStatsTest, bigint) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 25;
  constexpr int32_t bigintColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int64_t>(
          size,
          [](vector_size_t row) { return row * 1'000'000'000LL; },
          nullEvery(4))}));
  verifyBasicStats(
      stats[0], size, {{bigintColId, size}}, {{bigintColId, expectedNulls}});
  verifyBoundsExist(stats[0], {bigintColId});

  const auto [minVal, maxVal] = decodeBounds<int64_t>(stats[0], bigintColId);
  EXPECT_EQ(minVal, 1'000'000'000LL);
  EXPECT_EQ(maxVal, 99'000'000'000LL);
}

TEST_F(IcebergDwrfStatsTest, real) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 20;
  constexpr int32_t realColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<float>(
          size,
          [](vector_size_t row) { return static_cast<float>(row) * 1.5f; },
          nullEvery(5),
          REAL())}));
  verifyBasicStats(
      stats[0], size, {{realColId, size}}, {{realColId, expectedNulls}});
  verifyBoundsExist(stats[0], {realColId});

  const auto [minVal, maxVal] = decodeBounds<float>(stats[0], realColId);
  // Row 0 is null (nullEvery(5)); smallest non-null is row 1 -> 1.5.
  EXPECT_FLOAT_EQ(minVal, 1.5f);
  EXPECT_FLOAT_EQ(maxVal, 99 * 1.5f);
}

TEST_F(IcebergDwrfStatsTest, doubleType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 15;
  constexpr int32_t doubleColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<double>(
          size,
          [](vector_size_t row) { return row * 2.5; },
          nullEvery(7),
          DOUBLE())}));
  verifyBasicStats(
      stats[0], size, {{doubleColId, size}}, {{doubleColId, expectedNulls}});
  verifyBoundsExist(stats[0], {doubleColId});

  const auto [minVal, maxVal] = decodeBounds<double>(stats[0], doubleColId);
  // Row 0 is null; smallest non-null is row 1 -> 2.5.
  EXPECT_DOUBLE_EQ(minVal, 2.5);
  EXPECT_DOUBLE_EQ(maxVal, 99 * 2.5);
}

TEST_F(IcebergDwrfStatsTest, varchar) {
  constexpr vector_size_t size = 100;
  constexpr int32_t varcharColId = 1;
  constexpr int32_t expectedNulls = 17;

  // Keep values <= 16 codepoints so bounds are not truncated.
  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<std::string>(
          size,
          [](vector_size_t row) { return fmt::format("key_{:06}", row); },
          nullEvery(6))}));
  verifyBasicStats(
      stats[0], size, {{varcharColId, size}}, {{varcharColId, expectedNulls}});
  verifyBoundsExist(stats[0], {varcharColId});

  // Row 0 is null; smallest non-null is row 1 -> "key_000001"; max is row 99.
  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(varcharColId).lowerBound.value()),
      "key_000001");
  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(varcharColId).upperBound.value()),
      "key_000099");
}

TEST_F(IcebergDwrfStatsTest, date) {
  constexpr vector_size_t size = 100;
  constexpr int32_t expectedNulls = 20;
  constexpr int32_t dateColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int32_t>(
          size,
          [](vector_size_t row) { return 18262 + row; },
          nullEvery(5),
          DATE())}));
  verifyBasicStats(
      stats[0], size, {{dateColId, size}}, {{dateColId, expectedNulls}});
  verifyBoundsExist(stats[0], {dateColId});

  const auto [minVal, maxVal] = decodeBounds<int32_t>(stats[0], dateColId);
  // Row 0 is null; smallest non-null is row 1 -> 18263.
  EXPECT_EQ(minVal, 18263);
  EXPECT_EQ(maxVal, 18262 + 99);
}

// DWRF does not expose scalar min/max for TIMESTAMP through
// buildColumnStatisticsFromProto, so counts are populated but bounds are not.
TEST_F(IcebergDwrfStatsTest, timestampHasCountsButNoBounds) {
  constexpr vector_size_t size = 100;
  constexpr int32_t timestampColId = 1;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<Timestamp>(
          size,
          [](vector_size_t row) { return Timestamp(row, 0); },
          nullEvery(5))}));
  EXPECT_EQ(stats[0]->numRecords, size);
  EXPECT_EQ(stats[0]->columnStats.at(timestampColId).valueCount, size);
  EXPECT_EQ(stats[0]->columnStats.at(timestampColId).nullValueCount, 20);
  verifyBoundsNotExist(stats[0], {timestampColId});
}

// Short decimal is stored by the DWRF writer as an integer column; bounds are
// serialized as Iceberg big-endian two's-complement of the unscaled value.
TEST_F(IcebergDwrfStatsTest, shortDecimal) {
  constexpr vector_size_t size = 100;
  constexpr int32_t decimalColId = 1;
  constexpr int32_t expectedNulls = 20;

  const auto stats =
      writeDataAndGetAllStats(makeRowVector({makeFlatVector<int64_t>(
          size,
          [](vector_size_t row) { return row * 100LL; },
          nullEvery(5),
          DECIMAL(10, 2))}));
  verifyBasicStats(
      stats[0], size, {{decimalColId, size}}, {{decimalColId, expectedNulls}});
  verifyBoundsExist(stats[0], {decimalColId});
}

TEST_F(IcebergDwrfStatsTest, multipleDataTypes) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t bigintColId = 2;
  constexpr int32_t realColId = 3;
  constexpr int32_t varcharColId = 4;

  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(
           size, [](vector_size_t row) { return row * 10; }, nullEvery(3)),
       makeFlatVector<int64_t>(
           size,
           [](vector_size_t row) { return row * 1'000'000'000LL; },
           nullEvery(4)),
       makeFlatVector<float>(
           size,
           [](vector_size_t row) { return static_cast<float>(row) * 1.5f; },
           nullEvery(5),
           REAL()),
       makeFlatVector<std::string>(
           size,
           [](vector_size_t row) { return "str_" + std::to_string(row); },
           nullEvery(6))});
  const auto stats = writeDataAndGetAllStats(rowVector);

  verifyBasicStats(
      stats[0],
      size,
      {
          {intColId, size},
          {bigintColId, size},
          {realColId, size},
          {varcharColId, size},
      },
      {
          {intColId, 34},
          {bigintColId, 25},
          {realColId, 20},
          {varcharColId, 17},
      });
  verifyBoundsExist(stats[0], {intColId, bigintColId, realColId, varcharColId});
}

TEST_F(IcebergDwrfStatsTest, nullValues) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;

  const auto stats = writeDataAndGetAllStats(makeRowVector(
      {makeNullConstant(TypeKind::INTEGER, size),
       makeNullConstant(TypeKind::VARCHAR, size)}));
  EXPECT_EQ(stats[0]->numRecords, size);
  EXPECT_EQ(stats[0]->columnStats.at(intColId).nullValueCount, size);
  // No bounds for all-null columns.
  for (const auto& [fieldId, columnStats] : stats[0]->columnStats) {
    ASSERT_FALSE(columnStats.lowerBound.has_value());
    ASSERT_FALSE(columnStats.upperBound.has_value());
  }
}

// DWRF writes a file footer even when no rows are appended (unlike Parquet,
// which produces no file). The collector must report numRecords = 0 from the
// footer and emit no bounds.
TEST_F(IcebergDwrfStatsTest, emptyFile) {
  const auto outputDir = TempDirectoryPath::create();
  auto dataSink = createDataSinkAndAppendData(
      {makeRowVector(
          {makeFlatVector<int32_t>(0), makeFlatVector<StringView>(0)})},
      outputDir->getPath());
  auto commitTasks = dataSink->close();

  for (const auto& stats : statsFromCommitTasks(commitTasks)) {
    EXPECT_EQ(stats->numRecords, 0);
    for (const auto& [fieldId, columnStats] : stats->columnStats) {
      EXPECT_FALSE(columnStats.lowerBound.has_value());
      EXPECT_FALSE(columnStats.upperBound.has_value());
    }
  }
}

// Map values carry counts but no bounds (skipBounds=true for repeated types).
TEST_F(IcebergDwrfStatsTest, mapType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t mapValueColId = 3;

  std::vector<std::optional<
      std::vector<std::pair<int32_t, std::optional<std::string>>>>>
      mapData;
  for (auto i = 0; i < size; ++i) {
    std::vector<std::pair<int32_t, std::optional<std::string>>> mapRow;
    mapRow.reserve(5);
    for (auto j = 0; j < 5; ++j) {
      mapRow.emplace_back(j, fmt::format("value_{}", i * 5 + j));
    }
    mapData.emplace_back(std::move(mapRow));
  }

  const auto stats = writeDataAndGetAllStats(makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
       makeNullableMapVector<int32_t, std::string>(mapData)}));
  verifyBasicStats(stats[0], size, {{intColId, size}}, {{intColId, 0}});

  EXPECT_EQ(stats[0]->columnStats.at(mapValueColId).valueCount, size * 5);
  verifyBoundsNotExist(stats[0], {mapValueColId});
}

// Array elements carry counts but no bounds (skipBounds=true for repeated
// types).
TEST_F(IcebergDwrfStatsTest, arrayType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t arrayElementColId = 3;

  std::vector<std::vector<std::optional<std::string>>> arrayData;
  for (auto i = 0; i < size; ++i) {
    std::vector<std::optional<std::string>> arrayRow;
    arrayRow.reserve(3);
    for (auto j = 0; j < 3; ++j) {
      arrayRow.emplace_back(fmt::format("item_{}", i * 3 + j));
    }
    arrayData.emplace_back(std::move(arrayRow));
  }

  const auto stats = writeDataAndGetAllStats(makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
       makeNullableArrayVector<std::string>(arrayData)}));
  verifyBasicStats(stats[0], size, {{intColId, size}}, {{intColId, 0}});

  EXPECT_EQ(stats[0]->columnStats.at(arrayElementColId).valueCount, size * 3);
  verifyBoundsNotExist(stats[0], {arrayElementColId});
}

// Nested struct leaf fields carry bounds matched by field id.
// Field ids (depth-first from 1):
//   int_col: 1
//   struct_col: 2 (parent, no bounds)
//     first_level_id: 3
//     first_level_name: 4
//     nested_struct: 5 (parent, no bounds)
//       second_level_id: 6
//       second_level_name: 7
TEST_F(IcebergDwrfStatsTest, structType) {
  constexpr vector_size_t size = 100;
  constexpr int32_t intColId = 1;
  constexpr int32_t firstLevelIdColId = 3;
  constexpr int32_t secondLevelIdColId = 6;
  constexpr int32_t secondLevelNameColId = 7;

  auto firstLevelId = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row; }, nullEvery(5));
  auto firstLevelName = makeFlatVector<std::string>(
      size,
      [](vector_size_t row) { return fmt::format("name_{:04}", row); },
      nullEvery(7));
  auto secondLevelId = makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row * 2; }, nullEvery(6));
  auto secondLevelName = makeFlatVector<std::string>(
      size,
      [](vector_size_t row) { return fmt::format("nested_{:04}", row); },
      nullEvery(8));

  auto nestedStruct = makeRowVector({secondLevelId, secondLevelName});
  auto structVector =
      makeRowVector({firstLevelId, firstLevelName, nestedStruct});
  auto rowVector = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return row * 10; }),
       structVector});

  const auto stats = writeDataAndGetAllStats(rowVector);
  EXPECT_EQ(stats[0]->numRecords, size);

  verifyBoundsExist(
      stats[0],
      {intColId, firstLevelIdColId, secondLevelIdColId, secondLevelNameColId});

  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(secondLevelNameColId).lowerBound.value()),
      "nested_0001");
  EXPECT_EQ(
      encoding::Base64::decode(
          stats[0]->columnStats.at(secondLevelNameColId).upperBound.value()),
      "nested_0099");
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
