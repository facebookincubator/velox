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
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/common/tests/utils/DataFiles.h" // @manual
#include "velox/dwio/parquet/RegisterParquetReader.h" // @manual
#include "velox/dwio/parquet/common/ParquetConfig.h"
#include "velox/dwio/parquet/reader/ParquetReader.h" // @manual=//velox/connectors/hive:velox_hive_connector_parquet
#include "velox/dwio/parquet/tests/reader/ParquetTableScanTestBase.h"
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
using namespace facebook::velox::parquet::test;
using namespace facebook::velox::test;

// This suite covers Parquet schema evolution for primitive and complex types.
// Missing-field behavior is split into focused tests.
//
// Test                    Missing-field shape
// ----------------------  -----------------------------------
// partialFieldsMissing    One nested and one top-level field
// allFieldsMissing        Every requested nested field
// onlyFieldMissing        The only requested nested field
// emptyStruct             An empty requested nested struct
// caseSensitiveNames      All fields due to case mismatch
// lowercaseNames          One field after lower-case matching
// requiredStruct          All fields in a required struct
//
// Cross-cutting coverage:
//
// Axis                  Covered cases
// --------------------  ------------------------------------------------------
// Baseline schemas      Primitive and complex types in schemaMatch and
//                       schemaMatchWithComplexTypes.
// Mapping mode          Name mapping and position/index mapping.
// Struct depth          Direct ROW children, nested ROW children, ARRAY
//                       elements, and MAP values.
// Rep/def source path   Scalar leaves and leaves below ARRAY, MAP, and ROW.
// Split count           Single files and a reused ScanSpec across two files.
// Row-group count       Fixed at one; state reconciliation is tested between
//                       files, not row groups within one file.
// Predicates            Filters on partially and completely missing fields.
// Name casing           Case-sensitive matching and lower-cased file names.
// File nullability      Optional and required structs.
// Partition precedence  A partition constant overrides a same-name file
//                       column.
class ParquetSchemaEvolutionTest : public ParquetTableScanTestBase {
 protected:
  RowVectorPtr singleRow(
      const RowTypePtr& rowType,
      std::vector<VectorPtr> children) {
    return makeRowVector(rowType->names(), std::move(children));
  }

  VectorPtr nullConstant(const TypePtr& type) {
    return BaseVector::createNullConstant(type, 1, pool());
  }

  RowVectorPtr makePersonVector(
      const std::vector<std::string>& nameFieldNames) {
    return makeRowVector(
        {"id", "name", "address"},
        {
            makeFlatVector<int64_t>({2}),
            makeRowVector(
                nameFieldNames,
                {
                    makeFlatVector<std::string>({"Janet"}),
                    makeFlatVector<std::string>({"Jones"}),
                }),
            makeFlatVector<std::string>({"567 Maple Drive"}),
        });
  }

  AssertQueryBuilder selectUseColumnNames(
      const std::string& filePath,
      const RowTypePtr& outputType,
      bool nullStructIfAllFieldsMissing,
      const std::string& remainingFilter) {
    const auto plan =
        PlanBuilder().tableScan(outputType, {}, remainingFilter).planNode();
    AssertQueryBuilder builder(plan);
    builder
        .connectorSessionProperty(
            kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
        .connectorSessionProperty(
            kHiveConnectorId,
            parquetSessionProperty(
                ParquetConfig::kNullStructIfAllFieldsMissingSession),
            nullStructIfAllFieldsMissing ? "true" : "false")
        .split(makeSplit(filePath));
    return builder;
  }

  std::shared_ptr<Task> assertSelectUseColumnNames(
      const std::string& filePath,
      const RowTypePtr& outputType,
      const RowVectorPtr& expected,
      bool nullStructIfAllFieldsMissing,
      const std::string& remainingFilter = "") {
    return selectUseColumnNames(
               filePath,
               outputType,
               nullStructIfAllFieldsMissing,
               remainingFilter)
        .assertResults(expected);
  }

  std::shared_ptr<Task> assertEmptySelectUseColumnNames(
      const std::string& filePath,
      const RowTypePtr& outputType,
      bool nullStructIfAllFieldsMissing,
      const std::string& remainingFilter) {
    return selectUseColumnNames(
               filePath,
               outputType,
               nullStructIfAllFieldsMissing,
               remainingFilter)
        .assertEmptyResults();
  }

  std::shared_ptr<Task> assertMixedSplitsUseColumnNames(
      const std::string& firstFilePath,
      const std::string& secondFilePath,
      const RowTypePtr& outputType,
      const RowVectorPtr& expected) {
    const auto plan = PlanBuilder().tableScan(outputType).planNode();
    return AssertQueryBuilder(plan)
        .maxDrivers(1)
        .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "0")
        .connectorSessionProperty(
            kHiveConnectorId, HiveConfig::kUseColumnNamesSession, "true")
        .connectorSessionProperty(
            kHiveConnectorId,
            parquetSessionProperty(
                ParquetConfig::kNullStructIfAllFieldsMissingSession),
            "true")
        .splits({makeSplit(firstFilePath), makeSplit(secondFilePath)})
        .assertResults(expected);
  }
};

TEST_F(ParquetSchemaEvolutionTest, schemaMatchWithComplexTypes) {
  const vector_size_t kSize = 100;
  const auto valuesVector = makeRowVector(
      {"aa", "bb"},
      {makeFlatVector<int64_t>(kSize * 4, [](auto row) { return row; }),
       makeFlatVector<int32_t>(kSize * 4, [](auto row) { return row; })});
  const auto keysVector =
      makeFlatVector<int64_t>(kSize * 4, [](auto row) { return row % 4; });
  std::vector<vector_size_t> offsets;
  for (auto i = 0; i < kSize; i++) {
    offsets.push_back(i * 4);
  }
  const auto mapVector = makeMapVector(offsets, keysVector, valuesVector);
  const auto arrayVector = makeArrayVector(offsets, valuesVector);
  const auto primitiveVector = makeFlatVector(offsets);

  const auto dataFileVectors =
      makeRowVector({"p", "m", "a"}, {primitiveVector, mapVector, arrayVector});

  const std::shared_ptr<TempDirectoryPath> dataFileFolder =
      TempDirectoryPath::create();
  const auto filePath = dataFileFolder->getPath() + "/nested_data.parquet";
  ParquetWriterOptions options;
  options.writeInt96AsTimestamp = false;
  writeToParquetFile(filePath, {dataFileVectors}, options);

  // Create a row type with columns having different names than in the file.
  const auto structType = ROW({{"aa1", BIGINT()}, {"bb1", INTEGER()}});
  const auto rowType = ROW(
      {{"p1", INTEGER()},
       {"m1", MAP(BIGINT(), structType)},
       {"a1", ARRAY(structType)}});

  const auto plan =
      PlanBuilder()
          .tableScan(rowType, {}, "", rowType)
          .project({"p1", "m1[0].aa1", "m1[1].bb1", "a1[1].aa1", "a1[2].bb1"})
          .planNode();

  // Position mapping reads renamed primitive, MAP, and ARRAY columns by
  // their physical file order.
  {
    const auto bigintValues =
        makeFlatVector<int64_t>(kSize, [](auto row) { return row * 4; });
    const auto integerValues =
        makeFlatVector<int32_t>(kSize, [](auto row) { return row * 4 + 1; });
    const auto expected = makeRowVector(
        plan->outputType()->names(),
        {primitiveVector,
         bigintValues,
         integerValues,
         bigintValues,
         integerValues});
    const auto result =
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool());
    assertEqualVectors(expected, result);
  }

  // Name mapping treats the renamed primitive and nested fields as
  // missing and returns nulls.
  {
    const auto nullBigint = makeFlatVector<int64_t>(
        kSize, [](auto row) { return row; }, [](auto row) { return true; });
    const auto nullInteger = makeFlatVector<int32_t>(
        kSize, [](auto row) { return row; }, [](auto row) { return true; });
    const auto expected = makeRowVector(
        plan->outputType()->names(),
        {nullInteger, nullBigint, nullInteger, nullBigint, nullInteger});
    const auto result =
        AssertQueryBuilder(plan)
            .connectorSessionProperty(
                kHiveConnectorId, FileConfig::kUseColumnNamesSession, "true")
            .split(makeSplit(filePath))
            .copyResults(pool());
    assertEqualVectors(expected, result);
  }
}

// Covers primitive top-level schema evolution for position and name mapping.
TEST_F(ParquetSchemaEvolutionTest, schemaMatch) {
  const vector_size_t kSize = 100;
  const auto dataFileVectors = makeRowVector(
      {"c1", "c2"},
      {makeFlatVector<int64_t>(kSize, [](auto row) { return row; }),
       makeFlatVector<int64_t>(kSize, [](auto row) { return row * 4; })});

  const std::shared_ptr<TempDirectoryPath> dataFileFolder =
      TempDirectoryPath::create();
  const auto filePath = dataFileFolder->getPath() + "/data.parquet";
  ParquetWriterOptions options;
  options.writeInt96AsTimestamp = false;
  writeToParquetFile(filePath, {dataFileVectors}, options);

  // Position mapping reads renamed top-level columns by file order.
  {
    const auto rowType = ROW({"c2", "c3"}, BIGINT());
    const auto plan =
        PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
    const auto expected =
        makeRowVector(rowType->names(), dataFileVectors->children());
    const auto result =
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool());
    assertEqualVectors(expected, result);
  }

  // Position mapping surfaces a type mismatch when the renamed column is
  // read with an incompatible type.
  const auto nullVector = makeFlatVector<std::string>(
      kSize, [](auto row) { return "row"; }, [](auto row) { return true; });
  {
    const auto rowType = ROW({{"c2", BIGINT()}, {"c3", VARCHAR()}});
    const auto plan =
        PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool()),
        "Converted type BIGINT is not allowed for requested type VARCHAR for file column 'c2'");

    // Name mapping reads the matching column by name and returns null for a
    // missing requested column.
    const auto expected = makeRowVector(
        rowType->names(), {dataFileVectors->childAt(1), nullVector});
    const auto result =
        AssertQueryBuilder(plan)
            .connectorSessionProperty(
                kHiveConnectorId, FileConfig::kUseColumnNamesSession, "true")
            .split(makeSplit(filePath))
            .copyResults(pool());
    assertEqualVectors(expected, result);
  }

  // Position mapping rejects a requested type that conflicts with the
  // first physical column.
  {
    const auto rowType = ROW({{"c1", REAL()}, {"c2", BIGINT()}});
    const auto plan = PlanBuilder()
                          .tableScan(rowType, {}, "", rowType)
                          .project({"c1"})
                          .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool()),
        "Converted type BIGINT is not allowed for requested type REAL for file column 'c1'");
  }

  // Removing a requested column still reads the remaining column by
  // position.
  {
    const auto rowType = ROW("c1", BIGINT());
    const auto plan = PlanBuilder()
                          .tableScan(rowType, {}, "", rowType)
                          .project({"c1"})
                          .planNode();
    const auto expected = makeRowVector(
        plan->outputType()->names(), {dataFileVectors->childAt(0)});
    const auto result =
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool());
    assertEqualVectors(expected, result);
  }

  // Adding a requested column returns null for the missing column.
  {
    const auto rowType =
        ROW({{"c1", BIGINT()}, {"c2", BIGINT()}, {"c3", VARCHAR()}});
    const auto plan = PlanBuilder()
                          .tableScan(rowType, {}, "", rowType)
                          .project({"c1", "c2", "c3"})
                          .planNode();
    const auto expected = makeRowVector(
        plan->outputType()->names(),
        {dataFileVectors->childAt(0), dataFileVectors->childAt(1), nullVector});
    const auto result =
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool());
    assertEqualVectors(expected, result);
  }

  // Removing one column and adding another fails by position because the
  // remaining physical column has an incompatible type.
  {
    const auto rowType = ROW({{"c1", BIGINT()}, {"c3", VARCHAR()}});
    const auto plan = PlanBuilder()
                          .tableScan(rowType, {}, "", rowType)
                          .project({"c3"})
                          .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan).split(makeSplit(filePath)).copyResults(pool()),
        "Converted type BIGINT is not allowed for requested type VARCHAR for file column 'c2'");

    // The same remove-and-add shape is safe by name: the added column is
    // missing and returns null.
    const auto expected =
        makeRowVector(plan->outputType()->names(), {nullVector});
    const auto result =
        AssertQueryBuilder(plan)
            .connectorSessionProperty(
                kHiveConnectorId, FileConfig::kUseColumnNamesSession, "true")
            .split(makeSplit(filePath))
            .copyResults(pool());
    assertEqualVectors(expected, result);
  }
}

// Reads present nested fields and synthesizes nulls for one missing nested
// field and one missing top-level field under both config values.
TEST_F(ParquetSchemaEvolutionTest, partialFieldsMissing) {
  const auto vector = makePersonVector({"first", "last"});
  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  const auto rowType = ROW(
      {{"id", BIGINT()},
       {"name", ROW({"first", "middle", "last"}, VARCHAR())},
       {"email", VARCHAR()}});
  const auto expected = singleRow(
      rowType,
      {
          vector->childAt(0),
          makeRowVector(
              {"first", "middle", "last"},
              {
                  makeFlatVector<std::string>({"Janet"}),
                  makeNullableFlatVector<std::string>({std::nullopt}),
                  makeFlatVector<std::string>({"Jones"}),
              }),
          nullConstant(VARCHAR()),
      });
  for (const bool nullStruct : {true, false}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "nullStructIfAllFieldsMissing=" << (nullStruct ? "true" : "false"));
    assertSelectUseColumnNames(file->getPath(), rowType, expected, nullStruct);

    // A filter on the missing nested field rejects all rows.
    assertEmptySelectUseColumnNames(
        file->getPath(), rowType, nullStruct, "not(is_null(name.middle))");
  }
}

// Returns an all-null nested struct, or a null struct when configured, after
// every requested nested field is renamed.
TEST_F(ParquetSchemaEvolutionTest, allFieldsMissing) {
  const auto vector = makePersonVector({"first", "last"});
  const auto id = vector->childAt(0);
  const auto address = vector->childAt(2);
  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  const auto rowType = ROW(
      {{"id", BIGINT()},
       {"name", ROW({"a", "b"}, VARCHAR())},
       {"address", VARCHAR()}});
  for (const bool nullStruct : {true, false}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "nullStructIfAllFieldsMissing=" << (nullStruct ? "true" : "false"));
    assertSelectUseColumnNames(
        file->getPath(),
        rowType,
        singleRow(
            rowType,
            {
                id,
                nullStruct
                    ? nullConstant(rowType->childAt(1))
                    : makeRowVector(
                          {"a", "b"},
                          {makeNullableFlatVector<std::string>({std::nullopt}),
                           makeNullableFlatVector<std::string>(
                               {std::nullopt})}),
                address,
            }),
        nullStruct);

    // Filter the representation selected by the config.
    assertEmptySelectUseColumnNames(
        file->getPath(),
        rowType,
        nullStruct,
        nullStruct ? "not(is_null(name))" : "not(is_null(name.a))");
  }
}

// Covers a nested struct whose only requested field is missing.
TEST_F(ParquetSchemaEvolutionTest, onlyFieldMissing) {
  const auto vector = makePersonVector({"first", "last"});
  const auto id = vector->childAt(0);
  const auto address = vector->childAt(2);
  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  const auto rowType = ROW(
      {{"id", BIGINT()},
       {"name", ROW("full", VARCHAR())},
       {"address", VARCHAR()}});
  for (const bool nullStruct : {true, false}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "nullStructIfAllFieldsMissing=" << (nullStruct ? "true" : "false"));
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
        nullStruct);

    assertEmptySelectUseColumnNames(
        file->getPath(), rowType, nullStruct, "not(is_null(name.full))");
  }
}

// Returns either a null struct or an empty row when the requested nested
// struct has no fields.
TEST_F(ParquetSchemaEvolutionTest, emptyStruct) {
  const auto vector = makePersonVector({"first", "last"});
  const auto id = vector->childAt(0);
  const auto address = vector->childAt(2);
  const auto file = TempFilePath::create();
  writeToParquetFile(file->getPath(), {vector});

  const auto emptyType = ROW({}, {});
  const auto rowType =
      ROW({{"id", BIGINT()}, {"name", emptyType}, {"address", VARCHAR()}});
  for (const bool nullStruct : {true, false}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "nullStructIfAllFieldsMissing=" << (nullStruct ? "true" : "false"));
    assertSelectUseColumnNames(
        file->getPath(),
        rowType,
        singleRow(
            rowType,
            {
                id,
                nullStruct ? nullConstant(emptyType)
                           : makeRowVector(emptyType, 1),
                address,
            }),
        nullStruct);
  }
}

// Treats uppercase physical nested fields as missing under case-sensitive
// name matching.
TEST_F(ParquetSchemaEvolutionTest, caseSensitiveNames) {
  const auto upperCaseVector = makePersonVector({"FIRST", "LAST"});
  const auto id = upperCaseVector->childAt(0);
  const auto address = upperCaseVector->childAt(2);
  const auto upperCaseFile = TempFilePath::create();
  writeToParquetFile(upperCaseFile->getPath(), {upperCaseVector});

  const auto rowType = ROW(
      {{"id", BIGINT()},
       {"name", ROW({"first", "middle", "last"}, VARCHAR())},
       {"address", VARCHAR()}});
  for (const bool nullStruct : {true, false}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "nullStructIfAllFieldsMissing=" << (nullStruct ? "true" : "false"));
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
        nullStruct);
  }
}

// Lower-cases physical field names before matching, leaving only the genuinely
// absent nested field null.
TEST_F(ParquetSchemaEvolutionTest, lowercaseNames) {
  const auto upperCaseVector = makePersonVector({"FIRST", "LAST"});
  const auto upperCaseFile = TempFilePath::create();
  writeToParquetFile(upperCaseFile->getPath(), {upperCaseVector});

  const auto rowType = ROW(
      {{"id", BIGINT()},
       {"name", ROW({"first", "middle", "last"}, VARCHAR())},
       {"address", VARCHAR()}});
  const auto plan =
      PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
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
              upperCaseVector->childAt(0),
              makeRowVector(
                  {"first", "middle", "last"},
                  {
                      makeFlatVector<std::string>({"Janet"}),
                      makeNullableFlatVector<std::string>({std::nullopt}),
                      makeFlatVector<std::string>({"Jones"}),
                  }),
              upperCaseVector->childAt(2),
          }));
}

// Returns a null required struct when nullStructIfAllFieldsMissing is enabled
// and a non-null struct with null children when it is disabled.
TEST_F(ParquetSchemaEvolutionTest, requiredStruct) {
  const auto filePath = getDataFilePath(
      "velox/dwio/parquet/tests/reader",
      "../examples/proto-struct-with-array.parquet");
  const auto requiredMessageType = ROW("missing", INTEGER());
  const auto rowType = ROW("requiredMessage", requiredMessageType);

  for (const bool nullStruct : {true, false}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "nullStructIfAllFieldsMissing=" << (nullStruct ? "true" : "false"));
    const auto requiredMessage = nullStruct
        ? nullConstant(requiredMessageType)
        : singleRow(
              requiredMessageType,
              {makeNullableFlatVector<int32_t>({std::nullopt})});
    assertSelectUseColumnNames(
        filePath, rowType, singleRow(rowType, {requiredMessage}), nullStruct);
  }
}

// Covers rep/def sourcing when every requested field is missing and the first
// physical child is itself complex. ARRAY, MAP, and ROW are each sampled
// explicitly.
TEST_F(ParquetSchemaEvolutionTest, repDefSource) {
  const auto assertAllFieldsMissing = [&](const RowVectorPtr& vector) {
    const auto file = TempFilePath::create();
    writeToParquetFile(file->getPath(), {vector});

    const auto rowType =
        ROW({{"id", BIGINT()}, {"name", ROW("middle", VARCHAR())}});
    assertSelectUseColumnNames(
        file->getPath(),
        rowType,
        singleRow(
            rowType,
            {
                makeFlatVector<int64_t>({2}),
                makeRowVector(
                    {"middle"},
                    {makeNullableFlatVector<std::string>({std::nullopt})}),
            }),
        false);
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
TEST_F(ParquetSchemaEvolutionTest, arrayAndMapElements) {
  const std::vector<vector_size_t> offsets = {0};
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
      {{"names", ARRAY(elementType)}, {"lookup", MAP(BIGINT(), elementType)}});
  assertSelectUseColumnNames(
      file->getPath(),
      rowType,
      singleRow(
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
          }),
      false);
}

// Covers reuse of nested ScanSpecs for structs inside ARRAY elements and MAP
// values. The first split sets each struct to null; the second split brings the
// requested field back and must clear the stale struct-level constant.
TEST_F(ParquetSchemaEvolutionTest, arrayAndMapMixedSplits) {
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
      {{"names", ARRAY(elementType)}, {"lookup", MAP(BIGINT(), elementType)}});
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

  assertMixedSplitsUseColumnNames(
      firstFile->getPath(), secondFile->getPath(), outputType, expected);
}

// Covers reuse of one ScanSpec across two files with
// 'nullStructIfAllFieldsMissing' enabled. The first split sets a struct-level
// null constant for 'name'; the second split brings 'name.middle' back and must
// clear that stale constant.
TEST_F(ParquetSchemaEvolutionTest, mixedSplits) {
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
      ROW({{"id", BIGINT()}, {"name", ROW("middle", VARCHAR())}});

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

  assertMixedSplitsUseColumnNames(
      firstFile->getPath(), secondFile->getPath(), outputType, expected);
}

// Covers a reused ScanSpec for a struct nested directly inside another struct.
// A null synthesized for the first split must not hide a field that returns in
// the second split.
TEST_F(ParquetSchemaEvolutionTest, nestedMixedSplits) {
  const auto firstFileVector = makeRowVector(
      {"id", "outer"},
      {makeFlatVector<int64_t>({1}),
       makeRowVector(
           {"inner"},
           {makeRowVector({"old"}, {makeFlatVector<std::string>({"Old"})})})});
  const auto secondFileVector = makeRowVector(
      {"id", "outer"},
      {makeFlatVector<int64_t>({2}),
       makeRowVector(
           {"inner"},
           {makeRowVector(
               {"a"}, {makeFlatVector<std::string>({"Present"})})})});

  const auto firstFile = TempFilePath::create();
  writeToParquetFile(firstFile->getPath(), {firstFileVector});
  const auto secondFile = TempFilePath::create();
  writeToParquetFile(secondFile->getPath(), {secondFileVector});

  const auto innerType = ROW("a", VARCHAR());
  const auto outputType =
      ROW({{"id", BIGINT()}, {"outer", ROW("inner", innerType)}});
  const auto expected = makeRowVector(
      outputType->names(),
      {makeFlatVector<int64_t>({1, 2}),
       makeRowVector(
           {"inner"},
           {makeRowVector(
               {"a"},
               {makeNullableFlatVector<std::string>(
                   {std::nullopt, std::optional<std::string>{"Present"}})},
               [](auto row) { return row == 0; })})});

  assertMixedSplitsUseColumnNames(
      firstFile->getPath(), secondFile->getPath(), outputType, expected);
}

// Covers precedence between same-name partition keys and physical file columns
// under name mapping.
TEST_F(ParquetSchemaEvolutionTest, partitionPrecedence) {
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

  const auto plan = PlanBuilder()
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
TEST_F(ParquetSchemaEvolutionTest, matchByIndex) {
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
  {
    const auto rowType = ROW(
        {{"id", BIGINT()},
         {"name", ROW({"first", "middle", "last"}, VARCHAR())},
         {"address", VARCHAR()}});
    const auto plan =
        PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
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
  }

  // Renaming nested fields does not affect position mapping.
  {
    const auto rowType = ROW(
        {{"id", BIGINT()},
         {"name", ROW({"a", "b"}, VARCHAR())},
         {"address", VARCHAR()}});
    const auto plan =
        PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
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
  }

  // Deleting requested nested fields reads the first physical child by
  // position.
  {
    const auto rowType = ROW(
        {{"id", BIGINT()},
         {"name", ROW("full", VARCHAR())},
         {"address", VARCHAR()}});
    const auto plan =
        PlanBuilder().tableScan(rowType, {}, "", rowType).planNode();
    AssertQueryBuilder(plan)
        .split(makeSplit(file->getPath()))
        .assertResults(singleRow(
            rowType,
            {
                id,
                makeRowVector(
                    {"full"}, {makeFlatVector<std::string>({"Janet"})}),
                address,
            }));
  }
}
