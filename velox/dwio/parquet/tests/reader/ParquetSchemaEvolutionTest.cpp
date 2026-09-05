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
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/parquet/RegisterParquetReader.h" // @manual
#include "velox/dwio/parquet/common/ParquetConfig.h"
#include "velox/dwio/parquet/reader/ParquetReader.h" // @manual=//velox/connectors/hive:velox_hive_connector_parquet
#include "velox/dwio/parquet/writer/Writer.h" // @manual
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h" // @manual
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "velox/connectors/hive/HiveConfig.h" // @manual=//velox/connectors/hive:velox_hive_connector_parquet

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::parquet;
using namespace facebook::velox::test;

// This suite keeps Parquet schema evolution coverage in one place.
//
// Coverage matrix:
//
// Axis                  Covered cases
// --------------------  ------------------------------------------------------
// Baseline behavior     Existing primitive and complex-type schema matching.
// Missing-field shape   Missing top-level fields, missing nested fields, all
//                       nested fields missing, empty nested structs, and
//                       case-sensitive field names.
// Config                'nullStructIfAllFieldsMissing' enabled and disabled.
// Mapping mode          Name mapping and position/index mapping.
// Struct depth          Direct ROW children and structs below ARRAY/MAP fields.
// First physical child  ARRAY, MAP, and ROW as the first child used for
//                       repetition/definition levels.
// ARRAY/MAP nesting     Structs inside ARRAY elements and MAP values.
// Split count           Single-split scans and a reused ScanSpec across two
//                       splits.
// Row-group count       Not varied; these cases focus on state reconciliation
//                       between file splits, not between row groups in the same
//                       file.
// Partition precedence  Partition constant takes precedence over a same-name
//                       file column.
class ParquetSchemaEvolutionTest : public HiveConnectorTestBase {
 protected:
  static std::string parquetSessionProperty(std::string_view key) {
    return dwio::common::formatConfigPrefix(
               dwio::common::FileFormat::PARQUET, "_") +
        std::string(key);
  }

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath,
      const std::optional<
          std::unordered_map<std::string, std::optional<std::string>>>&
          partitionKeys = std::nullopt) {
    return makeHiveConnectorSplits(
        filePath, 1, dwio::common::FileFormat::PARQUET, partitionKeys)[0];
  }

  void writeToParquetFile(
      const std::string& path,
      const std::vector<RowVectorPtr>& data,
      ParquetWriterOptions options = {}) {
    VELOX_CHECK_GT(data.size(), 0);

    dwio::common::WriterOptions writerOptions;
    auto writeFile = std::make_unique<LocalWriteFile>(path, true, false);
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), path);
    auto childPool =
        rootPool_->addAggregateChild("ParquetSchemaEvolutionTest.Writer");
    writerOptions.memoryPool = childPool.get();
    writerOptions.formatSpecificOptions =
        std::make_shared<ParquetWriterOptions>(std::move(options));
    auto writer = std::make_unique<Writer>(
        std::move(sink), writerOptions, asRowType(data[0]->type()));

    for (const auto& vector : data) {
      writer->write(vector);
    }
    writer->close();
  }

  RowVectorPtr singleRow(
      const RowTypePtr& rowType,
      std::vector<VectorPtr> children) {
    return makeRowVector(rowType->names(), std::move(children));
  }

  VectorPtr nullConstant(const TypePtr& type) {
    return BaseVector::createNullConstant(type, 1, pool());
  }

  std::shared_ptr<Task> assertSelectUseColumnNames(
      const std::string& filePath,
      const RowTypePtr& outputType,
      const RowVectorPtr& expected,
      const std::string& nullStructIfAllFieldsMissing,
      const std::string& remainingFilter = "") {
    const auto plan =
        PlanBuilder().tableScan(outputType, {}, remainingFilter).planNode();
    return AssertQueryBuilder(plan)
        .connectorSessionProperty(
            kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
        .connectorSessionProperty(
            kHiveConnectorId,
            parquetSessionProperty(
                ParquetConfig::kNullStructIfAllFieldsMissingSession),
            nullStructIfAllFieldsMissing)
        .split(makeSplit(filePath))
        .assertResults(expected);
  }

  std::shared_ptr<Task> assertEmptySelectUseColumnNames(
      const std::string& filePath,
      const RowTypePtr& outputType,
      const std::string& nullStructIfAllFieldsMissing,
      const std::string& remainingFilter) {
    const auto plan =
        PlanBuilder().tableScan(outputType, {}, remainingFilter).planNode();
    return AssertQueryBuilder(plan)
        .connectorSessionProperty(
            kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
        .connectorSessionProperty(
            kHiveConnectorId,
            parquetSessionProperty(
                ParquetConfig::kNullStructIfAllFieldsMissingSession),
            nullStructIfAllFieldsMissing)
        .split(makeSplit(filePath))
        .assertEmptyResults();
  }
};

TEST_F(ParquetSchemaEvolutionTest, schemaMatchWithComplexTypes) {
  vector_size_t kSize = 100;
  auto valuesVector = makeRowVector(
      {"aa", "bb"},
      {makeFlatVector<int64_t>(kSize * 4, [](auto row) { return row; }),
       makeFlatVector<int32_t>(kSize * 4, [](auto row) { return row; })});
  auto keysVector =
      makeFlatVector<int64_t>(kSize * 4, [](auto row) { return row % 4; });
  std::vector<vector_size_t> offsets;
  for (auto i = 0; i < kSize; i++) {
    offsets.push_back(i * 4);
  }
  auto mapVector = makeMapVector(offsets, keysVector, valuesVector);
  auto arrayVector = makeArrayVector(offsets, valuesVector);
  auto primitiveVector = makeFlatVector(offsets);

  RowVectorPtr dataFileVectors =
      makeRowVector({"p", "m", "a"}, {primitiveVector, mapVector, arrayVector});

  const std::shared_ptr<TempDirectoryPath> dataFileFolder =
      TempDirectoryPath::create();
  auto filePath = dataFileFolder->getPath() + "/" + "nested_data.parquet";
  ParquetWriterOptions options;
  options.writeInt96AsTimestamp = false;
  writeToParquetFile(filePath, {dataFileVectors}, options);

  // Create a row type with columns having different names than in the file.
  auto structType = ROW({"aa1", "bb1"}, {BIGINT(), INTEGER()});
  auto rowType =
      ROW({"p1", "m1", "a1"},
          {{INTEGER(), MAP(BIGINT(), structType), ARRAY(structType)}});

  auto op =
      PlanBuilder()
          .startTableScan()
          .outputType(rowType)
          .dataColumns(rowType)
          .endTableScan()
          .project({"p1", "m1[0].aa1", "m1[1].bb1", "a1[1].aa1", "a1[2].bb1"})
          .planNode();

  // Position mapping reads renamed primitive, MAP, and ARRAY columns by
  // their physical file order.
  auto result =
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool());

  ASSERT_EQ(result->size(), kSize);
  auto rows = result->as<RowVector>();
  ASSERT_TRUE(rows);
  ASSERT_EQ(rows->childrenSize(), 5);

  assertEqualVectors(rows->childAt(0), primitiveVector);

  auto expected1 =
      makeFlatVector<int64_t>(kSize, [](auto row) { return row * 4; });
  assertEqualVectors(rows->childAt(1), expected1);
  assertEqualVectors(rows->childAt(3), expected1);

  auto expected2 =
      makeFlatVector<int>(kSize, [](auto row) { return row * 4 + 1; });
  assertEqualVectors(rows->childAt(2), expected2);
  assertEqualVectors(rows->childAt(4), expected2);

  // Name mapping treats the renamed primitive and nested fields as
  // missing and returns nulls.
  result = AssertQueryBuilder(op)
               .connectorSessionProperty(
                   kHiveConnectorId, FileConfig::kUseColumnNamesSession, "true")
               .split(makeSplit(filePath))
               .copyResults(pool());
  rows = result->as<RowVector>();
  auto nullBigIntVector = makeFlatVector<int64_t>(
      kSize, [](auto row) { return row; }, [](auto row) { return true; });
  auto nullIntVector = makeFlatVector<int>(
      kSize, [](auto row) { return row; }, [](auto row) { return true; });
  for (const auto index : std::vector<int>({0, 2, 4})) {
    assertEqualVectors(rows->childAt(index), nullIntVector);
  }
  for (const auto index : std::vector<int>({1, 3})) {
    assertEqualVectors(rows->childAt(index), nullBigIntVector);
  }
}

// Covers primitive top-level schema evolution for position and name mapping.
TEST_F(ParquetSchemaEvolutionTest, schemaMatch) {
  vector_size_t kSize = 100;
  RowVectorPtr dataFileVectors = makeRowVector(
      {"c1", "c2"},
      {makeFlatVector<int64_t>(kSize, [](auto row) { return row; }),
       makeFlatVector<int64_t>(kSize, [](auto row) { return row * 4; })});

  const std::shared_ptr<TempDirectoryPath> dataFileFolder =
      TempDirectoryPath::create();
  auto filePath = dataFileFolder->getPath() + "/" + "data.parquet";
  ParquetWriterOptions options;
  options.writeInt96AsTimestamp = false;
  writeToParquetFile(filePath, {dataFileVectors}, options);

  auto rowType = ROW({"c2", "c3"}, {BIGINT(), BIGINT()});
  auto op = PlanBuilder()
                .startTableScan()
                .outputType(rowType)
                .dataColumns(rowType)
                .endTableScan()
                .planNode();

  // Position mapping reads renamed top-level columns by file order.
  auto result =
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool());
  auto rows = result->as<RowVector>();

  assertEqualVectors(rows->childAt(0), dataFileVectors->childAt(0));
  assertEqualVectors(rows->childAt(1), dataFileVectors->childAt(1));

  // Position mapping surfaces a type mismatch when the renamed column is
  // read with an incompatible type.
  auto rowType1 = ROW({"c2", "c3"}, {BIGINT(), VARCHAR()});
  op = PlanBuilder()
           .startTableScan()
           .outputType(rowType1)
           .dataColumns(rowType1)
           .endTableScan()
           .planNode();
  EXPECT_THROW(
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool()),
      VeloxRuntimeError);

  // Name mapping reads the matching column by name and returns null for a
  // missing requested column.
  op = PlanBuilder()
           .startTableScan()
           .outputType(rowType1)
           .dataColumns(rowType1)
           .endTableScan()
           .planNode();

  result = AssertQueryBuilder(op)
               .connectorSessionProperty(
                   kHiveConnectorId, FileConfig::kUseColumnNamesSession, "true")
               .split(makeSplit(filePath))
               .copyResults(pool());

  rows = result->as<RowVector>();
  auto nullVector = makeFlatVector<std::string>(
      kSize, [](auto row) { return "row"; }, [](auto row) { return true; });
  assertEqualVectors(rows->childAt(0), dataFileVectors->childAt(1));
  assertEqualVectors(rows->childAt(1), nullVector);

  // Position mapping rejects a requested type that conflicts with the
  // first physical column.
  rowType = ROW({"c1", "c2"}, {{REAL(), BIGINT()}});
  op = PlanBuilder()
           .startTableScan()
           .outputType(rowType)
           .dataColumns(rowType)
           .endTableScan()
           .project({"c1"})
           .planNode();

  EXPECT_THROW(
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool()),
      VeloxRuntimeError);

  // Removing a requested column still reads the remaining column by
  // position.
  rowType = ROW({"c1"}, {{BIGINT()}});
  op = PlanBuilder()
           .startTableScan()
           .outputType(rowType)
           .dataColumns(rowType)
           .endTableScan()
           .project({"c1"})
           .planNode();

  result =
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool());
  rows = result->as<RowVector>();
  assertEqualVectors(rows->childAt(0), dataFileVectors->childAt(0));

  // Adding a requested column returns null for the missing column.
  rowType = ROW({"c1", "c2", "c3"}, {{BIGINT(), BIGINT(), VARCHAR()}});
  op = PlanBuilder()
           .startTableScan()
           .outputType(rowType)
           .dataColumns(rowType)
           .endTableScan()
           .project({"c1", "c2", "c3"})
           .planNode();

  result =
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool());
  rows = result->as<RowVector>();
  assertEqualVectors(rows->childAt(0), dataFileVectors->childAt(0));
  assertEqualVectors(rows->childAt(1), dataFileVectors->childAt(1));
  assertEqualVectors(rows->childAt(2), nullVector);

  // Removing one column and adding another fails by position because the
  // remaining physical column has an incompatible type.
  rowType = ROW({"c1", "c3"}, {{BIGINT(), VARCHAR()}});
  op = PlanBuilder()
           .startTableScan()
           .outputType(rowType)
           .dataColumns(rowType)
           .endTableScan()
           .project({"c3"})
           .planNode();

  EXPECT_THROW(
      AssertQueryBuilder(op).split(makeSplit(filePath)).copyResults(pool()),
      VeloxRuntimeError);

  // The same remove-and-add shape is safe by name: the added column is missing
  // and returns null.
  result = AssertQueryBuilder(op)
               .connectorSessionProperty(
                   kHiveConnectorId, FileConfig::kUseColumnNamesSession, "true")
               .split(makeSplit(filePath))
               .copyResults(pool());
  rows = result->as<RowVector>();
  assertEqualVectors(rows->childAt(0), nullVector);
}

// Covers direct ROW children under name mapping while enumerating the
// missing-field shapes across both nullStructIfAllFieldsMissing values.
TEST_F(ParquetSchemaEvolutionTest, structMatchByNameNullStruct) {
  const auto id = makeFlatVector<int64_t>({2});
  const auto name = makeRowVector(
      {"first", "last"},
      {
          makeFlatVector<std::string>({"Janet"}),
          makeFlatVector<std::string>({"Jones"}),
      });
  const auto address = makeFlatVector<std::string>({"567 Maple Drive"});
  auto vector = makeRowVector({"id", "name", "address"}, {id, name, address});

  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  const auto assertForConfig =
      [&](const std::string& nullStructIfAllFieldsMissing) {
        const bool nullStruct = nullStructIfAllFieldsMissing == "true";

        // One nested field and one top-level field are missing, while
        // existing nested siblings still read by name.
        auto rowType = ROW(
            {"id", "name", "email"},
            {BIGINT(), ROW({"first", "middle", "last"}, VARCHAR()), VARCHAR()});
        assertSelectUseColumnNames(
            file->getPath(),
            rowType,
            singleRow(
                rowType,
                {
                    id,
                    makeRowVector(
                        {"first", "middle", "last"},
                        {
                            makeFlatVector<std::string>({"Janet"}),
                            makeNullableFlatVector<std::string>({std::nullopt}),
                            makeFlatVector<std::string>({"Jones"}),
                        }),
                    nullConstant(VARCHAR()),
                }),
            nullStructIfAllFieldsMissing);
        // A filter on the missing nested field rejects all rows for both
        // null-struct config values.
        assertEmptySelectUseColumnNames(
            file->getPath(),
            rowType,
            nullStructIfAllFieldsMissing,
            "not(is_null(name.middle))");

        // All requested nested fields are missing after a struct-field
        // rename.
        rowType =
            ROW({"id", "name", "address"},
                {BIGINT(), ROW({"a", "b"}, VARCHAR()), VARCHAR()});
        assertSelectUseColumnNames(
            file->getPath(),
            rowType,
            singleRow(
                rowType,
                {
                    id,
                    nullStruct ? nullConstant(rowType->childAt(1))
                               : makeRowVector(
                                     {"a", "b"},
                                     {makeNullableFlatVector<std::string>(
                                          {std::nullopt}),
                                      makeNullableFlatVector<std::string>(
                                          {std::nullopt})}),
                    address,
                }),
            nullStructIfAllFieldsMissing);
        // Filtering the all-missing renamed struct follows the configured
        // null representation.
        assertEmptySelectUseColumnNames(
            file->getPath(),
            rowType,
            nullStructIfAllFieldsMissing,
            nullStruct ? "not(is_null(name))" : "not(is_null(name.a))");

        // Deleting all but one requested nested field leaves the single
        // field missing.
        rowType =
            ROW({"id", "name", "address"},
                {BIGINT(), ROW("full", VARCHAR()), VARCHAR()});
        assertSelectUseColumnNames(
            file->getPath(),
            rowType,
            singleRow(
                rowType,
                {
                    id,
                    nullStruct ? nullConstant(rowType->childAt(1))
                               : makeRowVector(
                                     {"full"},
                                     {makeNullableFlatVector<std::string>(
                                         {std::nullopt})}),
                    address,
                }),
            nullStructIfAllFieldsMissing);
        // A filter on the deleted single nested field rejects all rows.
        assertEmptySelectUseColumnNames(
            file->getPath(),
            rowType,
            nullStructIfAllFieldsMissing,
            "not(is_null(name.full))");

        // Requesting an empty nested struct uses either a null struct or
        // an empty row according to the config.
        rowType =
            ROW({"id", "name", "address"}, {BIGINT(), ROW({}, {}), VARCHAR()});
        assertSelectUseColumnNames(
            file->getPath(),
            rowType,
            singleRow(
                rowType,
                {
                    id,
                    nullStruct ? nullConstant(rowType->childAt(1))
                               : makeRowVector(ROW({}, {}), 1),
                    address,
                }),
            nullStructIfAllFieldsMissing);
      };

  for (const auto& nullStructIfAllFieldsMissing :
       std::vector<std::string>{"true", "false"}) {
    // Run the direct-row missing-field shapes with null structs
    // both enabled and disabled.
    assertForConfig(nullStructIfAllFieldsMissing);
  }

  vector = makeRowVector(
      {"id", "name", "address"},
      {id,
       makeRowVector(
           {"FIRST", "LAST"},
           {
               makeFlatVector<std::string>({"Janet"}),
               makeFlatVector<std::string>({"Jones"}),
           }),
       address});
  const auto upperCaseFile = TempFilePath::create();
  writeToParquetFile(upperCaseFile->getPath(), {vector});

  for (const auto& nullStructIfAllFieldsMissing :
       std::vector<std::string>{"true", "false"}) {
    // Case-sensitive name matching makes every requested nested field
    // missing when the file has uppercase child names.
    const bool nullStruct = nullStructIfAllFieldsMissing == "true";
    const auto rowType =
        ROW({"id", "name", "address"},
            {BIGINT(), ROW({"first", "middle", "last"}, VARCHAR()), VARCHAR()});
    assertSelectUseColumnNames(
        upperCaseFile->getPath(),
        rowType,
        singleRow(
            rowType,
            {
                id,
                nullStruct ? nullConstant(rowType->childAt(1))
                           : makeRowVector(
                                 {"first", "middle", "last"},
                                 {
                                     makeNullableFlatVector<std::string>(
                                         {std::nullopt}),
                                     makeNullableFlatVector<std::string>(
                                         {std::nullopt}),
                                     makeNullableFlatVector<std::string>(
                                         {std::nullopt}),
                                 }),
                address,
            }),
        nullStructIfAllFieldsMissing);
  }

  // Lower-casing file column names restores matching for uppercase nested
  // field names and leaves only the genuinely missing field null.
  const auto rowType =
      ROW({"id", "name", "address"},
          {BIGINT(), ROW({"first", "middle", "last"}, VARCHAR()), VARCHAR()});
  auto plan = PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
  AssertQueryBuilder(plan)
      .connectorSessionProperty(
          kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
      .connectorSessionProperty(
          kHiveConnectorId,
          HiveConfig::kFileColumnNamesReadAsLowerCaseSession,
          "true")
      .split(makeSplit(upperCaseFile->getPath()))
      .assertResults(singleRow(
          rowType,
          {
              id,
              makeRowVector(
                  {"first", "middle", "last"},
                  {
                      makeFlatVector<std::string>({"Janet"}),
                      makeNullableFlatVector<std::string>({std::nullopt}),
                      makeFlatVector<std::string>({"Jones"}),
                  }),
              address,
          }));
}

// Covers rep/def sourcing when every requested field is missing and the first
// physical child is itself complex. ARRAY, MAP, and ROW are each sampled
// explicitly.
TEST_F(ParquetSchemaEvolutionTest, structMatchByNameRepDefSource) {
  const auto assertAllFieldsMissing = [&](const RowVectorPtr& vector) {
    const auto file = TempFilePath::create();
    writeToParquetFile(file->getPath(), {vector});

    const auto rowType =
        ROW({"id", "name"}, {BIGINT(), ROW("middle", VARCHAR())});
    const auto plan = PlanBuilder().tableScan(rowType, {}, "").planNode();
    AssertQueryBuilder(plan)
        .connectorSessionProperty(
            kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
        .connectorSessionProperty(
            kHiveConnectorId,
            parquetSessionProperty(
                ParquetConfig::kNullStructIfAllFieldsMissingSession),
            "false")
        .split(makeSplit(file->getPath()))
        .assertResults(singleRow(
            rowType,
            {
                makeFlatVector<int64_t>({2}),
                makeRowVector(
                    {"middle"},
                    {makeNullableFlatVector<std::string>({std::nullopt})}),
            }));
  };

  // The first physical child used for rep/def sourcing is ARRAY.
  assertAllFieldsMissing(makeRowVector(
      {"id", "name"},
      {
          makeFlatVector<int64_t>({2}),
          makeRowVector(
              {"phones"},
              {makeArrayVector<StringView>({{"123-4567", "234-5678"}})}),
      }));
  // The first physical child used for rep/def sourcing is MAP.
  assertAllFieldsMissing(makeRowVector(
      {"id", "name"},
      {
          makeFlatVector<int64_t>({2}),
          makeRowVector(
              {"phones"},
              {makeMapVector<StringView, StringView>(
                  {{{StringView("home"), StringView("123-4567")}}})}),
      }));
  // The first physical child used for rep/def sourcing is ROW.
  assertAllFieldsMissing(makeRowVector(
      {"id", "name"},
      {
          makeFlatVector<int64_t>({2}),
          makeRowVector(
              {"parts"},
              {makeRowVector(
                  {"first"}, {makeFlatVector<std::string>({"Janet"})})}),
      }));
}

// Covers missing struct fields inside ARRAY elements and MAP values. Rep/def
// levels are hardest to preserve here because the missing struct is not a
// direct child of the top-level row.
TEST_F(ParquetSchemaEvolutionTest, structMatchByNameArrayAndMapElements) {
  std::vector<vector_size_t> offsets = {0};
  const auto values = makeRowVector(
      {"first", "last"},
      {
          makeFlatVector<std::string>({"Janet", "John"}),
          makeFlatVector<std::string>({"Jones", "Smith"}),
      });

  const auto vector = makeRowVector(
      {"names", "lookup"},
      {
          // The all-missing struct is an ARRAY element.
          makeArrayVector(offsets, values),
          // The all-missing struct is a MAP value.
          makeMapVector(offsets, makeFlatVector<int64_t>({1, 2}), values),
      });
  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  const auto elementType = ROW("middle", VARCHAR());
  const auto rowType = ROW(
      {"names", "lookup"}, {ARRAY(elementType), MAP(BIGINT(), elementType)});
  const auto plan = PlanBuilder().tableScan(rowType, {}, "").planNode();
  AssertQueryBuilder(plan)
      .connectorSessionProperty(
          kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
      .connectorSessionProperty(
          kHiveConnectorId,
          parquetSessionProperty(
              ParquetConfig::kNullStructIfAllFieldsMissingSession),
          "false")
      .split(makeSplit(file->getPath()))
      .assertResults(singleRow(
          rowType,
          {
              makeArrayVector(
                  offsets,
                  makeRowVector(
                      {"middle"},
                      {makeNullableFlatVector<std::string>(
                          {std::nullopt, std::nullopt})})),
              makeMapVector(
                  offsets,
                  makeFlatVector<int64_t>({1, 2}),
                  makeRowVector(
                      {"middle"},
                      {makeNullableFlatVector<std::string>(
                          {std::nullopt, std::nullopt})})),
          }));
}

// Covers reuse of nested ScanSpecs for structs inside ARRAY elements and MAP
// values. The first split sets each struct to null; the second split brings the
// requested field back and must clear the stale struct-level constant.
TEST_F(
    ParquetSchemaEvolutionTest,
    structMatchByNameArrayAndMapElementsMixedSplits) {
  const std::vector<vector_size_t> fileOffsets = {0};
  const auto missingValues = makeRowVector(
      {"first", "last"},
      {
          makeFlatVector<std::string>({"Janet", "John"}),
          makeFlatVector<std::string>({"Jones", "Smith"}),
      });
  const auto firstFileVector = makeRowVector(
      {"names", "lookup"},
      {
          // Every requested field in the ARRAY element struct is missing.
          makeArrayVector(fileOffsets, missingValues),
          // Every requested field in the MAP value struct is missing.
          makeMapVector(
              fileOffsets, makeFlatVector<int64_t>({1, 2}), missingValues),
      });

  const auto presentValues = makeRowVector(
      {"middle"}, {makeFlatVector<std::string>({"Anne", "Michael"})});
  const auto secondFileVector = makeRowVector(
      {"names", "lookup"},
      {
          // The requested ARRAY element field is present in the second split.
          makeArrayVector(fileOffsets, presentValues),
          // The requested MAP value field is present in the second split.
          makeMapVector(
              fileOffsets, makeFlatVector<int64_t>({1, 2}), presentValues),
      });

  const auto firstFile = TempFilePath::create();
  writeToParquetFile(firstFile->getPath(), {firstFileVector});
  const auto secondFile = TempFilePath::create();
  writeToParquetFile(secondFile->getPath(), {secondFileVector});

  const auto elementType = ROW("middle", VARCHAR());
  const auto outputType = ROW(
      {"names", "lookup"}, {ARRAY(elementType), MAP(BIGINT(), elementType)});
  const std::vector<vector_size_t> expectedOffsets = {0, 2};
  const auto expectedValues = makeRowVector(
      {"middle"},
      {makeNullableFlatVector<std::string>(
          {std::nullopt,
           std::nullopt,
           std::optional<std::string>{"Anne"},
           std::optional<std::string>{"Michael"}})},
      [](auto row) { return row < 2; });
  const auto expected = makeRowVector(
      outputType->names(),
      {
          makeArrayVector(expectedOffsets, expectedValues),
          makeMapVector(
              expectedOffsets,
              makeFlatVector<int64_t>({1, 2, 1, 2}),
              expectedValues),
      });

  const auto plan = PlanBuilder().tableScan(outputType).planNode();
  AssertQueryBuilder(plan)
      .maxDrivers(1)
      .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "0")
      .connectorSessionProperty(
          kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
      .connectorSessionProperty(
          kHiveConnectorId,
          parquetSessionProperty(
              ParquetConfig::kNullStructIfAllFieldsMissingSession),
          "true")
      .splits(
          {makeSplit(firstFile->getPath()), makeSplit(secondFile->getPath())})
      .assertResults(expected);
}

// Covers reuse of one ScanSpec across two files with
// 'nullStructIfAllFieldsMissing' enabled. The first split sets a struct-level
// null constant for 'name'; the second split brings 'name.middle' back and must
// clear that stale constant.
TEST_F(ParquetSchemaEvolutionTest, structMatchByNameMixedSplits) {
  // The first split has the requested nested field fully missing and sets
  // a struct-level null constant.
  const auto firstFileVector = makeRowVector(
      {"id", "name"},
      {makeFlatVector<int64_t>({1}),
       makeRowVector(
           {"first", "last"},
           {
               makeFlatVector<std::string>({"Janet"}),
               makeFlatVector<std::string>({"Jones"}),
           })});

  // The second split has the requested nested field present and must
  // clear the reused struct-level constant.
  const auto secondFileVector = makeRowVector(
      {"id", "name"},
      {makeFlatVector<int64_t>({2}),
       makeRowVector(
           {"middle"},
           {
               makeFlatVector<std::string>({"Middle"}),
           })});

  const auto firstFile = TempFilePath::create();
  writeToParquetFile(firstFile->getPath(), {firstFileVector});
  const auto secondFile = TempFilePath::create();
  writeToParquetFile(secondFile->getPath(), {secondFileVector});

  const auto outputType =
      ROW({"id", "name"}, {BIGINT(), ROW("middle", VARCHAR())});

  const auto expected = makeRowVector(
      outputType->names(),
      {
          makeFlatVector<int64_t>({1, 2}),
          makeRowVector(
              {"middle"},
              {makeNullableFlatVector<std::string>(
                  {std::nullopt, std::optional<std::string>{"Middle"}})},
              [](auto row) { return row == 0; }),
      });

  const auto plan = PlanBuilder().tableScan(outputType).planNode();
  AssertQueryBuilder(plan)
      .maxDrivers(1)
      .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "0")
      .connectorSessionProperty(
          kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
      .connectorSessionProperty(
          kHiveConnectorId,
          parquetSessionProperty(
              ParquetConfig::kNullStructIfAllFieldsMissingSession),
          "true")
      .splits(
          {makeSplit(firstFile->getPath()), makeSplit(secondFile->getPath())})
      .assertResults(expected);
}

// Covers precedence between same-name partition keys and physical file columns
// under name mapping.
TEST_F(ParquetSchemaEvolutionTest, partitionPrecedenceInNameMapping) {
  // The file contains a regular column with the same name as the
  // partition key.
  const auto fileVector = makeRowVector(
      {"i", "p", "j"},
      {
          makeFlatVector<int64_t>({1, 1}),
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({1, 1}),
      });

  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {fileVector});

  const auto outputType = ROW({"i", "p", "j"}, BIGINT());
  auto assignments = allRegularColumns(outputType);
  assignments["p"] = partitionKey("p", BIGINT());

  auto plan = PlanBuilder()
                  .tableScan(outputType, {}, "", nullptr, assignments)
                  .planNode();
  // The partition assignment wins over the same-name physical column.
  AssertQueryBuilder(plan)
      .connectorSessionProperty(
          kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
      .split(makeSplit(
          file->getPath(),
          std::unordered_map<std::string, std::optional<std::string>>{{
              "p",
              "1",
          }}))
      .assertResults(makeRowVector(
          outputType->names(),
          {
              makeFlatVector<int64_t>({1, 1}),
              makeFlatVector<int64_t>({1, 1}),
              makeFlatVector<int64_t>({1, 1}),
          }));
}

// Covers existing position-mapping behavior for struct children.
TEST_F(ParquetSchemaEvolutionTest, structMatchByIndex) {
  const auto id = makeFlatVector<int64_t>({2});
  const auto name = makeRowVector(
      {"first", "last"},
      {
          makeFlatVector<std::string>({"Janet"}),
          makeFlatVector<std::string>({"Jones"}),
      });
  const auto address = makeFlatVector<std::string>({"567 Maple Drive"});
  const auto vector =
      makeRowVector({"id", "name", "address"}, {id, name, address});

  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  // Adding a requested nested field appends null after position-mapped
  // siblings.
  auto rowType =
      ROW({"id", "name", "address"},
          {BIGINT(), ROW({"first", "middle", "last"}, VARCHAR()), VARCHAR()});
  auto plan = PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
  AssertQueryBuilder(plan)
      .split(makeSplit(file->getPath()))
      .assertResults(singleRow(
          rowType,
          {
              id,
              makeRowVector(
                  {"first", "middle", "last"},
                  {
                      makeFlatVector<std::string>({"Janet"}),
                      makeFlatVector<std::string>({"Jones"}),
                      makeNullableFlatVector<std::string>({std::nullopt}),
                  }),
              address,
          }));

  // Renaming nested fields does not affect position mapping.
  rowType =
      ROW({"id", "name", "address"},
          {BIGINT(), ROW({"a", "b"}, VARCHAR()), VARCHAR()});
  plan = PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
  AssertQueryBuilder(plan)
      .split(makeSplit(file->getPath()))
      .assertResults(singleRow(
          rowType,
          {
              id,
              makeRowVector(
                  {"a", "b"},
                  {
                      makeFlatVector<std::string>({"Janet"}),
                      makeFlatVector<std::string>({"Jones"}),
                  }),
              address,
          }));

  // Deleting requested nested fields reads the first physical child by
  // position.
  rowType = ROW(
      {"id", "name", "address"}, {BIGINT(), ROW("full", VARCHAR()), VARCHAR()});
  plan = PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
  AssertQueryBuilder(plan)
      .split(makeSplit(file->getPath()))
      .assertResults(singleRow(
          rowType,
          {
              id,
              makeRowVector({"full"}, {makeFlatVector<std::string>({"Janet"})}),
              address,
          }));
}
