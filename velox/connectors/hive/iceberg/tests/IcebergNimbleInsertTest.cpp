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

#include "dwio/nimble/velox/reader/fb/NimbleReader.h"
#include "dwio/nimble/velox/writer/fb/NimbleWriter.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::common::testutil;

namespace facebook::velox::connector::hive::iceberg {
namespace {

// Iceberg field-id column resolution under schema evolution only runs in the
// batch NimbleReader, not the selective reader. The connector session property
// "selective_nimble_reader_enabled" defaults to true, so the field-id read
// queries below set it to false to exercise the batch path.
constexpr const char* kSelectiveNimbleReaderEnabled{
    "selective_nimble_reader_enabled"};

// End-to-end tests for writing and reading Iceberg tables using the NIMBLE
// file format, mirroring IcebergDwrfInsertTest. Exercises the full write path
// (IcebergDataSink -> NIMBLE writer, which stamps iceberg.id attributes via
// NimbleWriterOptionsAdapter) and the full read path (IcebergSplitReader ->
// batch NimbleReader, which resolves columns by Iceberg field id and null-
// fills added columns).
class IcebergNimbleInsertTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    nimble::registerNimbleReaderFactory();
    nimble::registerNimbleWriterFactory();
    fileFormat_ = dwio::common::FileFormat::NIMBLE;
  }

  void TearDown() override {
    nimble::unregisterNimbleReaderFactory();
    nimble::unregisterNimbleWriterFactory();
    IcebergTestBase::TearDown();
  }

  // Write test data using NIMBLE format, then read it back and verify results.
  void test(const RowTypePtr& rowType, double nullRatio) {
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
                    .startTableScan()
                    .connectorId(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(vectors);
  }
};

// No schema evolution: write and read the same schema. Sanity check for the
// NIMBLE connector write+read path and iceberg.id stamping.
TEST_F(IcebergNimbleInsertTest, roundTrip) {
  auto rowType =
      ROW({"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"},
          {BIGINT(),
           INTEGER(),
           SMALLINT(),
           BOOLEAN(),
           REAL(),
           VARCHAR(),
           VARBINARY(),
           DOUBLE()});
  test(rowType, 0.2);
}

// Builds an Iceberg column handle with an explicit field-id tree, used to drive
// reads whose requested schema differs from the written schema.
std::shared_ptr<const connector::ColumnHandle> makeIcebergColumnHandle(
    const std::string& name,
    const TypePtr& type,
    const parquet::ParquetFieldId& field) {
  return std::make_shared<const IcebergColumnHandle>(
      name, FileColumnHandle::ColumnType::kRegular, type, field);
}

// Schema evolution: a top-level column is renamed (c -> c2, id 3) and
// reordered, another is dropped from the projection (b, id 2), and a new column
// is added (d, id 9) that is absent from the file. Field-id resolution must
// read c2/a from the file by id and null-fill d.
TEST_F(IcebergNimbleInsertTest, fieldIdRenameReorderDropAdd) {
  auto writeVector = makeRowVector(
      {"a", "b", "c"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<std::string>({"x", "y", "z"})});

  const auto dir = TempDirectoryPath::create();
  const auto path = dir->getPath();
  createDataSinkAndAppendData({writeVector}, path)->close();
  auto splits = createSplitsForDirectory(path);

  // Written field ids (assigned depth-first from 1): a=1, b=2, c=3.
  auto readSchema = ROW({"c2", "a", "d"}, {VARCHAR(), BIGINT(), INTEGER()});
  std::
      unordered_map<std::string, std::shared_ptr<const connector::ColumnHandle>>
          assignments{
              {"c2", makeIcebergColumnHandle("c2", VARCHAR(), {3, {}})},
              {"a", makeIcebergColumnHandle("a", BIGINT(), {1, {}})},
              {"d", makeIcebergColumnHandle("d", INTEGER(), {9, {}})}};

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(readSchema)
                  .dataColumns(readSchema)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"c2", "a", "d"},
      {makeFlatVector<std::string>({"x", "y", "z"}),
       makeFlatVector<int64_t>({1, 2, 3}),
       makeNullConstant(TypeKind::INTEGER, 3)});
  exec::test::AssertQueryBuilder(plan)
      .connectorSessionProperty(
          test::kIcebergConnectorId, kSelectiveNimbleReaderEnabled, "false")
      .splits(splits)
      .assertResults({expected});
}

// Schema evolution: column c (id 1) is dropped and a new column with the same
// name c (id 9) is added. The stale file column must NOT bind to the new c by
// name; the new c must read as null. The resolver renames the retained-dropped
// file column to a non-colliding sentinel so the connector's by-name null-fill
// correctly null-fills the re-added column instead of binding to stale data.
TEST_F(IcebergNimbleInsertTest, fieldIdDropReaddSameName) {
  auto writeVector = makeRowVector({"c"}, {makeFlatVector<int32_t>({1, 2, 3})});

  const auto dir = TempDirectoryPath::create();
  const auto path = dir->getPath();
  createDataSinkAndAppendData({writeVector}, path)->close();
  auto splits = createSplitsForDirectory(path);

  // Written field id: c=1. The re-added c has id 9.
  auto readSchema = ROW({"c"}, {INTEGER()});
  std::
      unordered_map<std::string, std::shared_ptr<const connector::ColumnHandle>>
          assignments{{"c", makeIcebergColumnHandle("c", INTEGER(), {9, {}})}};

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(readSchema)
                  .dataColumns(readSchema)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected =
      makeRowVector({"c"}, {makeNullConstant(TypeKind::INTEGER, 3)});
  exec::test::AssertQueryBuilder(plan)
      .connectorSessionProperty(
          test::kIcebergConnectorId, kSelectiveNimbleReaderEnabled, "false")
      .splits(splits)
      .assertResults({expected});
}

// Schema evolution inside a struct: a nested field is dropped (y) and the
// remaining nested fields are reordered (z before x). Field-id resolution must
// match nested columns by id, not position. The batch NimbleReader reshapes
// nested struct columns to the requested (table) field order/selection (see
// projectStructToSpec in NimbleReader.cpp), so the struct comes back as the
// requested ROW<z,x> rather than the file-order ROW<x,y,z>.
TEST_F(IcebergNimbleInsertTest, fieldIdNestedStructReorderDrop) {
  auto writeVector = makeRowVector(
      {"s"},
      {makeRowVector(
          {"x", "y", "z"},
          {makeFlatVector<int32_t>({1, 2, 3}),
           makeFlatVector<int32_t>({4, 5, 6}),
           makeFlatVector<int32_t>({7, 8, 9})})});

  const auto dir = TempDirectoryPath::create();
  const auto path = dir->getPath();
  createDataSinkAndAppendData({writeVector}, path)->close();
  auto splits = createSplitsForDirectory(path);

  // Written field ids (depth-first): s=1, x=2, y=3, z=4.
  auto readSchema = ROW({"s"}, {ROW({"z", "x"}, {INTEGER(), INTEGER()})});
  std::
      unordered_map<std::string, std::shared_ptr<const connector::ColumnHandle>>
          assignments{
              {"s",
               makeIcebergColumnHandle(
                   "s",
                   ROW({"z", "x"}, {INTEGER(), INTEGER()}),
                   {1, {{4, {}}, {2, {}}}})}};

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(test::kIcebergConnectorId)
                  .outputType(readSchema)
                  .dataColumns(readSchema)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeRowVector(
          {"z", "x"},
          {makeFlatVector<int32_t>({7, 8, 9}),
           makeFlatVector<int32_t>({1, 2, 3})})});
  exec::test::AssertQueryBuilder(plan)
      .connectorSessionProperty(
          test::kIcebergConnectorId, kSelectiveNimbleReaderEnabled, "false")
      .splits(splits)
      .assertResults({expected});
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
