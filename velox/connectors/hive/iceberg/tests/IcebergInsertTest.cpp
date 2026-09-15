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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/IcebergConnector.h"
#include "velox/connectors/hive/iceberg/IcebergFieldMetadata.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/BaseVector.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::connector::hive::iceberg {
namespace {

#ifdef VELOX_ENABLE_PARQUET

class IcebergInsertTest : public test::IcebergTestBase {
 protected:
  void test(const RowTypePtr& rowType, double nullRatio = 0.0) {
    const auto outputDirectory = TempDirectoryPath::create();
    const auto dataPath = outputDirectory->getPath();
    constexpr int32_t numBatches = 10;
    constexpr int32_t vectorSize = 5'000;
    const auto vectors =
        createTestData(rowType, numBatches, vectorSize, nullRatio);
    const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
    const auto commitTasks = dataSink->close();

    auto splits = createSplitsForDirectory(dataPath);
    ASSERT_EQ(splits.size(), commitTasks.size());
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }

  void icebergAddInputWithDefault(
      const RowTypePtr& rowType,
      const std::string& outputPath,
      const std::vector<std::string>& insertedColumns,
      const std::unordered_map<std::string, std::string>& writeDefaults,
      const std::vector<RowVectorPtr>& rows) {
    auto handle = makeWriteDefaultHandle(
        rowType, outputPath, insertedColumns, writeDefaults);
    writeThroughHandle(rowType, handle, rows);
  }

  // Scans the Iceberg table using the given splits and asserts that the result
  // matches @p expected.
  void readFromIcebergTable(
      const RowTypePtr& rowType,
      const std::vector<std::shared_ptr<connector::ConnectorSplit>>& splits,
      const RowVectorPtr& expected) {
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(
        {expected});
  }

  IcebergInsertTableHandlePtr makeWriteDefaultHandle(
      const RowTypePtr& rowType,
      const std::string& outputPath,
      const std::vector<std::string>& insertedColumns,
      const std::unordered_map<std::string, std::string>& writeDefaults) {
    std::vector<IcebergColumnHandlePtr> columnHandles;
    columnHandles.reserve(rowType->size());
    for (auto i = 0; i < rowType->size(); ++i) {
      const auto& name = rowType->nameOf(i);
      const auto& type = rowType->childAt(i);
      const parquet::ParquetFieldId field{static_cast<int32_t>(i + 1), {}};
      const auto it = writeDefaults.find(name);
      const auto writeDefault = it != writeDefaults.end()
          ? std::make_optional(it->second)
          : std::nullopt;
      columnHandles.push_back(
          std::make_shared<const IcebergColumnHandle>(
              name,
              FileColumnHandle::ColumnType::kRegular,
              type,
              field,
              /*requiredSubfields=*/std::vector<common::Subfield>{},
              /*initialDefaultValue=*/std::nullopt,
              /*icebergMetadata=*/IcebergFieldMetadata{},
              /*postProcessor=*/std::function<void(VectorPtr&)>{},
              writeDefault));
    }
    auto locationHandle = std::make_shared<LocationHandle>(
        outputPath, outputPath, LocationHandle::TableType::kNew);
    return std::make_shared<const IcebergInsertTableHandle>(
        columnHandles,
        locationHandle,
        fileFormat_,
        /*partitionSpec=*/nullptr,
        common::CompressionKind::CompressionKind_ZSTD,
        /*serdeParameters=*/std::unordered_map<std::string, std::string>{},
        IcebergInsertTableHandle::WriteKind::kData,
        /*existingDeletionVectors=*/
        std::unordered_map<
            std::string,
            IcebergInsertTableHandle::ExistingDeletionVector>{},
        std::make_shared<const IcebergFileNameGenerator>(),
        insertedColumns);
  }

  // Writes @p vectors through a sink built from @p handle, asserts that the
  // sink finishes cleanly, and returns the commit tasks.
  std::vector<std::string> writeThroughHandle(
      const RowTypePtr& rowType,
      const IcebergInsertTableHandlePtr& handle,
      const std::vector<RowVectorPtr>& batches) {
    auto sink = std::make_shared<IcebergDataSink>(
        rowType,
        handle,
        connectorQueryCtx_.get(),
        CommitStrategy::kNoCommit,
        std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{})),
        std::make_shared<IcebergConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {IcebergConfig::kFunctionPrefixConfig,
                 IcebergConfig::kDefaultFunctionPrefix}})));
    for (const auto& batch : batches) {
      sink->appendData(batch);
    }
    EXPECT_TRUE(sink->finish());
    auto commitTasks = sink->close();
    EXPECT_FALSE(commitTasks.empty());
    return commitTasks;
  }
};

TEST_F(IcebergInsertTest, basic) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           BOOLEAN(),
           REAL(),
           DECIMAL(18, 5),
           VARCHAR(),
           VARBINARY(),
           DATE(),
           TIMESTAMP(),
           ROW({"id", "name"}, {INTEGER(), VARCHAR()})});
  test(rowType, 0.2);
}

TEST_F(IcebergInsertTest, mapAndArray) {
  auto rowType =
      ROW({"c1", "c2"}, {MAP(INTEGER(), VARCHAR()), ARRAY(VARCHAR())});
  test(rowType);
}

TEST_F(IcebergInsertTest, bigDecimal) {
  auto rowType = ROW({"c1"}, {DECIMAL(38, 5)});
  fileFormat_ = dwio::common::FileFormat::PARQUET;
  test(rowType);
}

TEST_F(IcebergInsertTest, singleColumnPartition) {
  struct TestCase {
    std::string name;
    TypePtr type;
  };

  std::vector<TestCase> testCases = {
      {"c1", BIGINT()},
      {"c2", INTEGER()},
      {"c3", SMALLINT()},
      {"c4", DECIMAL(18, 5)},
      {"c5", BOOLEAN()},
      {"c6", VARCHAR()},
      {"c7", DATE()},
      {"c8", TIMESTAMP()}};

  for (const auto& testCase : testCases) {
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;
    auto rowType = ROW({testCase.name}, {testCase.type});

    const auto vectors = createTestData(rowType, numBatches, vectorSize, 0.5);
    std::vector<test::PartitionField> partitionTransforms = {
        {0, TransformType::kIdentity, std::nullopt}};
    const auto dataSink = createDataSinkAndAppendData(
        vectors, outputDirectory->getPath(), partitionTransforms);
    const auto commitTasks = dataSink->close();
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_GT(commitTasks.size(), 0);
    ASSERT_EQ(splits.size(), commitTasks.size());

    for (const auto& task : commitTasks) {
      auto taskJson = folly::parseJson(task);
      ASSERT_TRUE(taskJson.count("partitionDataJson") > 0);
    }

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
}

TEST_F(IcebergInsertTest, partitionNullColumn) {
  struct TestCase {
    std::string name;
    TypePtr type;
  };

  std::vector<TestCase> testCases = {
      {"c1", BIGINT()},
      {"c2", INTEGER()},
      {"c3", SMALLINT()},
      {"c4", DECIMAL(18, 5)},
      {"c5", BOOLEAN()},
      {"c6", VARCHAR()},
      {"c7", DATE()},
      {"c8", TIMESTAMP()}};

  for (const auto& testCase : testCases) {
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 100;
    auto rowType = ROW({testCase.name}, {testCase.type});
    // nullRatio = 1.0
    const auto vectors = createTestData(rowType, numBatches, vectorSize, 1.0);

    std::vector<test::PartitionField> partitionTransforms = {
        {0, TransformType::kIdentity, std::nullopt}};
    const auto dataSink = createDataSinkAndAppendData(
        vectors, outputDirectory->getPath(), partitionTransforms);

    const auto commitTasks = dataSink->close();
    ASSERT_EQ(1, commitTasks.size());
    auto taskJson = folly::parseJson(commitTasks.at(0));
    ASSERT_EQ(1, taskJson.count("partitionDataJson"));
    auto partitionData =
        folly::parseJson(taskJson["partitionDataJson"].asString());
    ASSERT_EQ(1, partitionData.count("partitionValues"));
    auto partitionValues = partitionData["partitionValues"];
    ASSERT_TRUE(partitionValues.isArray());
    ASSERT_TRUE(partitionValues[0].isNull());

    auto files = listFiles(outputDirectory->getPath());
    ASSERT_EQ(files.size(), 1);

    for (const auto& file : files) {
      auto partitionKeys = extractPartitionKeys(file);
      ASSERT_EQ(partitionKeys.size(), 1);
      ASSERT_TRUE(partitionKeys.contains(testCase.name));
      ASSERT_FALSE(partitionKeys.at(testCase.name).has_value());
    }
  }
}

TEST_F(IcebergInsertTest, partitionMultiColumns) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4"},
          {
              BIGINT(),
              INTEGER(),
              SMALLINT(),
              DECIMAL(18, 5),
          });
  std::vector<std::vector<int32_t>> columnCombinations = {
      {0, 1}, // BIGINT, INTEGER.
      {2, 1}, // SMALLINT, INTEGER.
      {2, 3}, // SMALLINT, DECIMAL.
      {0, 2, 1} // BIGINT, SMALLINT, INTEGER.
  };

  for (const auto& combination : columnCombinations) {
    const auto outputDirectory = TempDirectoryPath::create();
    constexpr int32_t numBatches = 2;
    constexpr int32_t vectorSize = 50;

    std::vector<RowVectorPtr> vectors;
    vectors.reserve(numBatches);
    for (int32_t batch = 0; batch < numBatches; ++batch) {
      vectors.push_back(makeRowVector(
          rowType->names(),
          {
              makeFlatVector<int64_t>(
                  vectorSize, [](auto row) { return row * 100; }),
              makeFlatVector<int32_t>(
                  vectorSize, [](auto row) { return row * 10; }),
              makeFlatVector<int16_t>(vectorSize, [](auto row) { return row; }),
              makeFlatVector<int64_t>(
                  vectorSize,
                  [](auto row) { return (row * 1000); },
                  nullptr,
                  DECIMAL(18, 5)),
          }));
    }

    std::vector<test::PartitionField> partitionTransforms;
    for (auto colIndex : combination) {
      partitionTransforms.push_back(
          {colIndex, TransformType::kIdentity, std::nullopt});
    }

    const auto dataSink = createDataSinkAndAppendData(
        vectors, outputDirectory->getPath(), partitionTransforms);

    const auto commitTasks = dataSink->close();
    auto splits = createSplitsForDirectory(outputDirectory->getPath());

    ASSERT_EQ(commitTasks.size(), vectorSize);
    ASSERT_EQ(splits.size(), commitTasks.size());

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
}

TEST_F(IcebergInsertTest, maxTargetFileSizeRotation) {
  constexpr int32_t kNumBatches = 10;
  constexpr vector_size_t kRowsPerBatch = 100;
  constexpr int32_t kPayloadSize = 96;

  // Generate fixed-size, per-row-varying strings for predictable size
  // accounting without relying on fuzzed VARCHAR lengths.
  auto makePayload = [](int64_t value) {
    std::string payload;
    payload.reserve(kPayloadSize);
    for (auto i = 0; i < kPayloadSize; ++i) {
      payload.push_back(static_cast<char>('a' + ((value + i) % 26)));
    }
    return payload;
  };

  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  std::vector<RowVectorPtr> vectors;
  vectors.reserve(kNumBatches);
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    const auto batchOffset = batch * kRowsPerBatch;
    vectors.push_back(makeRowVector(
        rowType->names(),
        {
            makeFlatVector<int64_t>(
                kRowsPerBatch,
                [batchOffset](auto row) { return batchOffset + row; }),
            makeFlatVector<std::string>(
                kRowsPerBatch,
                [&, batchOffset](auto row) {
                  return makePayload(batchOffset + row);
                }),
        }));
  }

  auto writeAndRead = [&](const std::string& maxTargetFileSize) {
    setConnectorSessionProperty(
        HiveConfig::kParquetMaxTargetFileSizeSession, maxTargetFileSize);

    const auto outputDirectory = TempDirectoryPath::create();
    const auto outputPath = outputDirectory->getPath();
    const auto dataSink = createDataSinkAndAppendData(vectors, outputPath);
    const auto commitTasks = dataSink->close();
    const auto files = listFiles(outputPath);
    EXPECT_EQ(files.size(), commitTasks.size());

    auto splits = createSplitsForDirectory(outputPath);
    auto plan = exec::test::PlanBuilder()
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);

    return files.size();
  };

  ASSERT_EQ(writeAndRead("1KB"), kNumBatches);
  ASSERT_EQ(writeAndRead("10MB"), 1);
}

// Explicit NULL (column in insertedColumns) and omitted column (not in
// insertedColumns) written to separate files. The explicit NULL must be
// preserved; the omitted column must receive the write-default.
TEST_F(IcebergInsertTest, explicitNullPreservedOmittedGetsWriteDefault) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "status"}, {BIGINT(), VARCHAR()});

  // Rows 2–3: status explicitly set to NULL (column in insertedColumns).
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id", "status"},
      {{"status", "ACTIVE"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3}),
           makeNullableFlatVector<std::string>(
               {std::nullopt, std::nullopt})})});

  // Rows 4–5: status omitted → write-default 'ACTIVE' must be materialised.
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      {{"status", "ACTIVE"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({4, 5}),
           makeNullableFlatVector<std::string>(
               {std::nullopt, std::nullopt})})});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 2U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3, 4, 5}),
           makeNullableFlatVector<std::string>(
               {std::nullopt, std::nullopt, "ACTIVE", "ACTIVE"})}));
}

// Multiple default columns: only 'id' is in insertedColumns for row 2 (all
// defaults applied); 'id' and 'country' for row 3 (priority and is_enabled
// get their defaults, 'country' keeps the explicit value 'UK').
TEST_F(IcebergInsertTest, multipleWriteDefaultColumns) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType =
      ROW({"id", "country", "priority", "is_enabled"},
          {BIGINT(), VARCHAR(), INTEGER(), BOOLEAN()});

  const std::unordered_map<std::string, std::string> defaults = {
      {"country", "US"}, {"priority", "10"}, {"is_enabled", "true"}};

  // Rows 2–3: all default columns omitted.
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      defaults,
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3}),
           makeNullableFlatVector<std::string>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<bool>({std::nullopt, std::nullopt})})});

  // Rows 4–5: country explicitly set; priority and is_enabled omitted.
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id", "country"},
      defaults,
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({4, 5}),
           makeFlatVector<std::string>({"UK", "FR"}),
           makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<bool>({std::nullopt, std::nullopt})})});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 2U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3, 4, 5}),
           makeFlatVector<std::string>({"US", "US", "UK", "FR"}),
           makeFlatVector<int32_t>({10, 10, 10, 10}),
           makeFlatVector<bool>({true, true, true, true})}));
}

// All write-default types are materialised when their columns are omitted.
// Covers: VARCHAR, DOUBLE, BIGINT, INTEGER, BOOLEAN (plain scalars),
// DATE, TIMESTAMP, and DECIMAL (decimal literal string) in a single
// write + read.
// DATE and TIMESTAMP use the ISO forms the coordinator emits for defaults,
// in UTC — not the numeric encoding partition values use. The coordinator
// always writes six fractional digits on a TIMESTAMP, so 'ts' carries them.
// DECIMAL uses "99.99" parsed by DecimalUtil::castFromString → 9999.
TEST_F(IcebergInsertTest, writeDefaultAllTypes) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType =
      ROW({"id",
           "name",
           "score",
           "count",
           "priority",
           "active",
           "dt",
           "ts",
           "price"},
          {BIGINT(),
           VARCHAR(),
           DOUBLE(),
           BIGINT(),
           INTEGER(),
           BOOLEAN(),
           DATE(),
           TIMESTAMP(),
           DECIMAL(10, 2)});

  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      {{"name", "Unknown"},
       {"score", "0.0"},
       {"count", "0"},
       {"priority", "5"},
       {"active", "false"},
       // DATE: ISO date string parsed as 19737 days since epoch.
       {"dt", "2024-01-15"},
       // TIMESTAMP: ISO datetime string parsed as UTC, with the six
       // fractional digits the coordinator always emits.
       {"ts", "2024-01-15 10:30:00.000000"},
       // DECIMAL(10,2): "99.99" → scaled integer 9999.
       {"price", "99.99"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3}),
           makeNullableFlatVector<std::string>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<double>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<bool>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<int32_t>(
               {std::nullopt, std::nullopt}, DATE()),
           makeNullableFlatVector<Timestamp>({std::nullopt, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {std::nullopt, std::nullopt}, DECIMAL(10, 2))})});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3}),
           makeFlatVector<std::string>({"Unknown", "Unknown"}),
           makeFlatVector<double>({0.0, 0.0}),
           makeFlatVector<int64_t>({0, 0}),
           makeFlatVector<int32_t>({5, 5}),
           makeFlatVector<bool>({false, false}),
           makeFlatVector<int32_t>({19737, 19737}, DATE()),
           makeFlatVector<Timestamp>(
               {Timestamp(1705314600, 0), Timestamp(1705314600, 0)}),
           makeFlatVector<int64_t>({9999, 9999}, DECIMAL(10, 2))}));
}

// Simulates ALTER TABLE … SET DEFAULT: two consecutive INSERTs each use a
// different write-default for 'country' (first 'IN', then 'US').  Reading
// both files back must return the value that was current at write time.
TEST_F(IcebergInsertTest, changedWriteDefaultAcrossInserts) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType =
      ROW({"id", "amount", "country"}, {BIGINT(), DOUBLE(), VARCHAR()});

  auto omittedCountryRow = [&](int64_t id, double amount) {
    return makeRowVector(
        rowType->names(),
        {makeFlatVector<int64_t>({id}),
         makeFlatVector<double>({amount}),
         makeNullableFlatVector<std::string>({std::nullopt})});
  };

  // INSERT 1: write-default = 'IN', two rows.
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id", "amount"},
      {{"country", "IN"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3}),
           makeFlatVector<double>({200.0, 201.0}),
           makeNullableFlatVector<std::string>(
               {std::nullopt, std::nullopt})})});

  // INSERT 2: write-default changed to 'US' (ALTER TABLE … SET DEFAULT 'US').
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id", "amount"},
      {{"country", "US"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({4, 5}),
           makeFlatVector<double>({300.0, 301.0}),
           makeNullableFlatVector<std::string>(
               {std::nullopt, std::nullopt})})});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 2U);
  // Rows 2–3 written with default 'IN', rows 4–5 with 'US'.
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({2, 3, 4, 5}),
           makeFlatVector<double>({200.0, 201.0, 300.0, 301.0}),
           makeFlatVector<std::string>({"IN", "IN", "US", "US"})}));
}

// An omitted column arrives as a null ConstantVector — the natural encoding
// the planner produces. The sink must substitute the write-default correctly
// regardless of the input's null encoding.
TEST_F(IcebergInsertTest, writeDefaultWithConstantNullInput) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "status"}, {BIGINT(), VARCHAR()});
  const auto handle = makeWriteDefaultHandle(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      {{"status", "ACTIVE"}});

  // 128 rows so the batch exercises more than one uint64_t word of null bits.
  const vector_size_t batchSize = 128;
  auto ids =
      makeFlatVector<int64_t>(batchSize, [](vector_size_t i) { return i + 1; });
  auto statuses = BaseVector::createNullConstant(VARCHAR(), batchSize, pool());

  auto batch = std::make_shared<RowVector>(
      pool(),
      rowType,
      /*nulls=*/nullptr,
      batchSize,
      std::vector<VectorPtr>{ids, statuses});

  writeThroughHandle(rowType, handle, {batch});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>(
               batchSize, [](vector_size_t i) { return i + 1; }),
           makeFlatVector<std::string>(
               batchSize, [](vector_size_t) { return "ACTIVE"; })}));
}

// An omitted column produced by IcebergMergeSink::makeInsertBatch arrives as a
// DictionaryVector wrapping a flat all-null base (wrapInDictionary with
// nulls=nullptr over each input child). The DictionaryVector has no
// wrapper-level null buffer, so isNullAt() follows indices into the base.
// The sink must substitute the write-default correctly for this encoding too.
TEST_F(IcebergInsertTest, writeDefaultWithDictionaryNullInput) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "status"}, {BIGINT(), VARCHAR()});
  const auto handle = makeWriteDefaultHandle(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      {{"status", "ACTIVE"}});

  const vector_size_t batchSize = 4;
  auto ids = makeFlatVector<int64_t>({10, 20, 30, 40});

  // wrapInDictionary returns the base unchanged when it is a ConstantVector
  // and nulls==nullptr, so use a FlatVector base to actually get a
  // DictionaryVector encoding.
  auto flatNullBase = makeNullableFlatVector<std::string>(
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  auto indices = makeIndices(batchSize, [](vector_size_t i) { return i; });
  auto dictStatuses = BaseVector::wrapInDictionary(
      /*nulls=*/nullptr, indices, batchSize, flatNullBase);
  ASSERT_EQ(dictStatuses->encoding(), VectorEncoding::Simple::DICTIONARY);

  auto batch = std::make_shared<RowVector>(
      pool(),
      rowType,
      /*nulls=*/nullptr,
      batchSize,
      std::vector<VectorPtr>{ids, dictStatuses});

  writeThroughHandle(rowType, handle, {batch});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({10, 20, 30, 40}),
           makeFlatVector<std::string>(
               {"ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"})}));
}

// A non-null value in a write-default column is a planner bug: the column
// was classified as omitted at plan time so no row should ever carry a value
// for it. The sink must surface this immediately via VELOX_CHECK.
TEST_F(IcebergInsertTest, writeDefaultNonNullInputThrows) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "status"}, {BIGINT(), VARCHAR()});
  const auto handle = makeWriteDefaultHandle(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      {{"status", "ACTIVE"}});

  // 'status' has a non-null value in row 1 — contradicts its omitted
  // classification. The sink must throw.
  auto batch = makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>({1, 2}),
       makeNullableFlatVector<std::string>({"EXPLICIT", std::nullopt})});

  VELOX_ASSERT_THROW(
      writeThroughHandle(rowType, handle, {batch}),
      "Non-null value found at row 0 in write-default column 'status'");
}

// A partitioned table where a non-partition column has a write-default.
// Rows go to different partition buckets; the defaulted column must be
// materialised correctly in every partition file.
TEST_F(IcebergInsertTest, writeDefaultPartitionedTable) {
  const auto outputDir = TempDirectoryPath::create();
  // Table: id BIGINT (partition), country VARCHAR (write-default='US')
  const auto rowType = ROW({"id", "country"}, {BIGINT(), VARCHAR()});

  // Build a partitioned handle: 'id' is the identity-partition column,
  // 'country' is a regular column with a write-default.
  std::vector<IcebergColumnHandlePtr> columnHandles;
  columnHandles.push_back(
      std::make_shared<const IcebergColumnHandle>(
          "id",
          FileColumnHandle::ColumnType::kPartitionKey,
          BIGINT(),
          parquet::ParquetFieldId{1, {}}));
  columnHandles.push_back(
      std::make_shared<const IcebergColumnHandle>(
          "country",
          FileColumnHandle::ColumnType::kRegular,
          VARCHAR(),
          parquet::ParquetFieldId{2, {}},
          /*requiredSubfields=*/std::vector<common::Subfield>{},
          /*initialDefaultValue=*/std::nullopt,
          /*icebergMetadata=*/IcebergFieldMetadata{},
          /*postProcessor=*/std::function<void(VectorPtr&)>{},
          /*writeDefaultValue=*/"US"));

  const std::vector<IcebergPartitionSpec::Field> fields = {
      {"id", BIGINT(), TransformType::kIdentity, std::nullopt}};
  auto partitionSpec = std::make_shared<IcebergPartitionSpec>(1, fields);

  auto locationHandle = std::make_shared<LocationHandle>(
      outputDir->getPath(),
      outputDir->getPath(),
      LocationHandle::TableType::kNew);

  auto handle = std::make_shared<const IcebergInsertTableHandle>(
      columnHandles,
      locationHandle,
      fileFormat_,
      partitionSpec,
      common::CompressionKind::CompressionKind_ZSTD,
      /*serdeParameters=*/std::unordered_map<std::string, std::string>{},
      IcebergInsertTableHandle::WriteKind::kData,
      /*existingDeletionVectors=*/
      std::unordered_map<
          std::string,
          IcebergInsertTableHandle::ExistingDeletionVector>{},
      std::make_shared<const IcebergFileNameGenerator>(),
      /*insertedColumns=*/std::vector<std::string>{"id"});

  // Two rows with different 'id' values → two partition directories.
  // 'country' is omitted → write-default 'US' must appear in both.
  auto batch = makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>({1, 2}),
       makeNullableFlatVector<std::string>({std::nullopt, std::nullopt})});

  writeThroughHandle(rowType, handle, {batch});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 2U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<std::string>({"US", "US"})}));
}

// A partially-null column that is in insertedColumns must have its NULLs
// preserved — the all-null invariant check must NOT fire for it, and neither
// must the write-default substitution.
TEST_F(IcebergInsertTest, partiallyNullInsertedColumnPreservesNulls) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "status"}, {BIGINT(), VARCHAR()});

  // 'status' IS in insertedColumns, so it never enters writeDefaultColumns_.
  // A partially-null batch (some rows NULL, some non-NULL) must be written
  // as-is without substitution or assertion failure.
  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id", "status"},
      {{"status", "ACTIVE"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeNullableFlatVector<std::string>(
               {"PENDING", std::nullopt, "DONE"})})});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeNullableFlatVector<std::string>(
               {"PENDING", std::nullopt, "DONE"})}));
}

// The engine lowercases INSERT statement column names while column handle
// names keep the Iceberg schema's spelling, so a mixed-case column named
// explicitly in the INSERT must not be treated as omitted.
TEST_F(IcebergInsertTest, writeDefaultMixedCaseColumnName) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "Status"}, {BIGINT(), VARCHAR()});

  icebergAddInputWithDefault(
      rowType,
      outputDir->getPath(),
      // The engine sends lowercased statement names; the schema spells the
      // column 'Status'. Both the supplied value and the explicit NULL must
      // survive.
      /*insertedColumns=*/{"id", "status"},
      {{"Status", "ACTIVE"}},
      {makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2}),
           makeNullableFlatVector<std::string>({"PENDING", std::nullopt})})});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2}),
           makeNullableFlatVector<std::string>({"PENDING", std::nullopt})}));
}

// toString() includes writeDefaultValue when set.
TEST_F(IcebergInsertTest, columnHandleToStringIncludesWriteDefault) {
  auto col = std::make_shared<IcebergColumnHandle>(
      "country",
      FileColumnHandle::ColumnType::kRegular,
      VARCHAR(),
      parquet::ParquetFieldId{3, {}},
      /*requiredSubfields=*/std::vector<common::Subfield>{},
      /*initialDefaultValue=*/std::nullopt,
      /*icebergMetadata=*/IcebergFieldMetadata{},
      /*postProcessor=*/std::function<void(VectorPtr&)>{},
      /*writeDefaultValue=*/"US");
  EXPECT_NE(col->toString().find("writeDefaultValue: US"), std::string::npos);
}

// A write-default on a partition column decides which partition file a row
// lands in. Substitution happens before HiveDataSink computes partition IDs,
// so a defaulted partition key must route rows to the correct partition.
TEST_F(IcebergInsertTest, writeDefaultOnPartitionColumn) {
  const auto outputDir = TempDirectoryPath::create();
  // Table: region VARCHAR (identity-partition, write-default='APAC'),
  //        value BIGINT (inserted explicitly).
  const auto rowType = ROW({"region", "value"}, {VARCHAR(), BIGINT()});

  std::vector<IcebergColumnHandlePtr> columnHandles;
  columnHandles.push_back(
      std::make_shared<const IcebergColumnHandle>(
          "region",
          FileColumnHandle::ColumnType::kPartitionKey,
          VARCHAR(),
          parquet::ParquetFieldId{1, {}},
          /*requiredSubfields=*/std::vector<common::Subfield>{},
          /*initialDefaultValue=*/std::nullopt,
          /*icebergMetadata=*/IcebergFieldMetadata{},
          /*postProcessor=*/std::function<void(VectorPtr&)>{},
          /*writeDefaultValue=*/"APAC"));
  columnHandles.push_back(
      std::make_shared<const IcebergColumnHandle>(
          "value",
          FileColumnHandle::ColumnType::kRegular,
          BIGINT(),
          parquet::ParquetFieldId{2, {}}));

  const std::vector<IcebergPartitionSpec::Field> fields = {
      {"region", VARCHAR(), TransformType::kIdentity, std::nullopt}};
  auto partitionSpec = std::make_shared<IcebergPartitionSpec>(1, fields);

  auto locationHandle = std::make_shared<LocationHandle>(
      outputDir->getPath(),
      outputDir->getPath(),
      LocationHandle::TableType::kNew);

  auto handle = std::make_shared<const IcebergInsertTableHandle>(
      columnHandles,
      locationHandle,
      fileFormat_,
      partitionSpec,
      common::CompressionKind::CompressionKind_ZSTD,
      /*serdeParameters=*/std::unordered_map<std::string, std::string>{},
      IcebergInsertTableHandle::WriteKind::kData,
      /*existingDeletionVectors=*/
      std::unordered_map<
          std::string,
          IcebergInsertTableHandle::ExistingDeletionVector>{},
      std::make_shared<const IcebergFileNameGenerator>(),
      /*insertedColumns=*/std::vector<std::string>{"value"});

  // 'region' is omitted → write-default 'APAC' must determine the partition.
  // All rows land in the same partition directory (region=APAC).
  auto batch = makeRowVector(
      rowType->names(),
      {makeNullableFlatVector<std::string>({std::nullopt, std::nullopt}),
       makeFlatVector<int64_t>({10, 20})});

  writeThroughHandle(rowType, handle, {batch});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  // Both rows go to the same partition → exactly one data file.
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<std::string>({"APAC", "APAC"}),
           makeFlatVector<int64_t>({10, 20})}));
}

// Multiple batches per sink: the pre-computed constant vector must be
// re-wrapped correctly for each batch, including batches of different sizes.
TEST_F(IcebergInsertTest, writeDefaultMultipleBatches) {
  const auto outputDir = TempDirectoryPath::create();
  const auto rowType = ROW({"id", "label"}, {BIGINT(), VARCHAR()});

  const auto handle = makeWriteDefaultHandle(
      rowType,
      outputDir->getPath(),
      /*insertedColumns=*/{"id"},
      {{"label", "DEFAULT"}});

  // Three batches of different sizes to exercise re-wrapping.
  auto batch1 = makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>({1}),
       makeNullableFlatVector<std::string>({std::nullopt})});
  auto batch2 = makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>({2, 3, 4}),
       makeNullableFlatVector<std::string>(
           {std::nullopt, std::nullopt, std::nullopt})});
  auto batch3 = makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>({5, 6}),
       makeNullableFlatVector<std::string>({std::nullopt, std::nullopt})});

  writeThroughHandle(rowType, handle, {batch1, batch2, batch3});

  auto splits = createSplitsForDirectory(outputDir->getPath());
  ASSERT_EQ(splits.size(), 1U);
  readFromIcebergTable(
      rowType,
      splits,
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
           makeFlatVector<std::string>(
               {"DEFAULT",
                "DEFAULT",
                "DEFAULT",
                "DEFAULT",
                "DEFAULT",
                "DEFAULT"})}));
}

#endif

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
