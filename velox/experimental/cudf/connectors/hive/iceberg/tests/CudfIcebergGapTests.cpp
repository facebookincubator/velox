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

#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfDeletionVectorTestUtils.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfIcebergTestBase.h"

#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gtest/gtest.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector::hive::iceberg;
using facebook::velox::common::testutil::TempFilePath;
using namespace facebook::velox::cudf_velox::iceberg::test;

namespace facebook::velox::cudf_velox::exec::test {

class CudfIcebergGapTests : public CudfIcebergTestBase {};

/// Multiple equality delete files at different sequence numbers targeting
/// overlapping values.
TEST_F(CudfIcebergGapTests, multipleDeletesAtDifferentSequenceNumbers) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 5}),
      makeFlatVector<int64_t>({200, 500}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto data3 = makeRowVector({
      makeFlatVector<int64_t>({2, 6}),
      makeFlatVector<int64_t>({2000, 6000}),
  });
  auto dataFile3 = TempFilePath::create();
  writeToFile(dataFile3->getPath(), data3);

  auto del1 = makeRowVector({makeFlatVector<int64_t>({2})});
  auto delFile1 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, delFile1->getPath(), {del1});
  IcebergDeleteFile icebergDel1(
      FileContent::kEqualityDeletes,
      delFile1->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(delFile1->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto del2 = makeRowVector({makeFlatVector<int64_t>({2})});
  auto delFile2 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, delFile2->getPath(), {del2});
  IcebergDeleteFile icebergDel2(
      FileContent::kEqualityDeletes,
      delFile2->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(delFile2->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/4);

  auto splits1 = makeIcebergSplits(
      dataFile1->getPath(), {icebergDel1, icebergDel2}, {}, 1, 1);
  auto splits2 = makeIcebergSplits(
      dataFile2->getPath(), {icebergDel1, icebergDel2}, {}, 1, 3);
  auto splits3 = makeIcebergSplits(
      dataFile3->getPath(), {icebergDel1, icebergDel2}, {}, 1, 5);

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());
  all.insert(all.end(), splits3.begin(), splits3.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5, 2, 6}),
      makeFlatVector<int64_t>({10, 30, 500, 2000, 6000}),
  });

  assertEqualResults({expected}, {result});
}

/// Mixed positional + equality + deletion vector interleaving.
TEST_F(CudfIcebergGapTests, positionalAndEqualityWithSequenceNumbers) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70, 80}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      2, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({0, 7});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(posDeleteFile->getPath()),
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({30, 60})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits =
      makeIcebergSplits(dataFile->getPath(), {posDelete, eqDelete}, {}, 1, 1);

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 40, 50, 70}),
      makeFlatVector<int64_t>({2, 4, 5, 7}),
  });

  assertEqualResults({expected}, {result});
}

/// Multi-column equality delete with overlapping but non-matching values.
TEST_F(CudfIcebergGapTests, multiColumnPartialMatchDoesNotDelete) {
  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 2, 1}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto eqDel = makeRowVector({
      makeFlatVector<int64_t>({2}),
      makeFlatVector<int64_t>({20}),
  });
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 2, 1}),
      makeFlatVector<int64_t>({10, 30, 40, 50}),
      makeFlatVector<int64_t>({100, 300, 400, 500}),
  });

  assertEqualResults({expected}, {result});
}

/// Equality delete where the delete value doesn't exist in any data file.
TEST_F(CudfIcebergGapTests, equalityDeleteNoMatchAcrossFiles) {
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({999})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile1->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
  });

  assertEqualResults({expected}, {result});
}

/// Hive partitioned Iceberg table: data file in a partition directory,
/// query selects both data columns and the partition column.
TEST_F(CudfIcebergGapTests, hivePartitionedTable) {
  auto fullType = ROW({"c0", "c1", "country"}, {BIGINT(), BIGINT(), VARCHAR()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"country", "US"},
  };

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      /*deleteFiles=*/{},
      partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(ROW({"c0", "c1"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<std::string>({"US", "US", "US"}),
  });

  assertEqualResults({expected}, {result});
}

/// Hive partitioned table with equality deletes.
TEST_F(CudfIcebergGapTests, hivePartitionWithEqualityDelete) {
  auto fullType = ROW({"c0", "c1", "region"}, {BIGINT(), BIGINT(), VARCHAR()});
  auto dataColumns = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"region", "APAC"},
  };

  auto splits =
      makeIcebergSplits(dataFile->getPath(), {eqDelete}, partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(dataColumns)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 30, 40, 50}),
      makeFlatVector<std::string>({"APAC", "APAC", "APAC", "APAC"}),
  });

  assertEqualResults({expected}, {result});
}

/// Schema evolution: file 1 has [c0, c1], file 2 has [c0, c1, c2].
TEST_F(CudfIcebergGapTests, schemaEvolutionAddedColumn) {
  auto fullType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<int64_t>({10, 20}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({3, 4}),
      makeFlatVector<int64_t>({30, 40}),
      makeFlatVector<int64_t>({300, 400}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto splits1 = makeIcebergSplits(dataFile1->getPath());
  auto splits2 = makeIcebergSplits(dataFile2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeFlatVector<int64_t>({10, 20, 30, 40}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 300, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// Schema evolution with equality delete.
TEST_F(CudfIcebergGapTests, schemaEvolutionWithEqualityDelete) {
  auto fullType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto data1 = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({200, 400}),
      makeFlatVector<int64_t>({2000, 4000}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile1->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 4}),
      makeFlatVector<int64_t>({10, 30, 400}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 4000}),
  });

  assertEqualResults({expected}, {result});
}

/// NULL equality matching: per Iceberg spec, NULL == NULL is TRUE for
/// equality deletes.
TEST_F(CudfIcebergGapTests, equalityDeleteNullMatchesNull) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto eqDel = makeRowVector({
      makeNullableFlatVector<int64_t>({std::nullopt}),
  });
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Multiple equality delete files with DIFFERENT key columns targeting
/// the same data file.
TEST_F(CudfIcebergGapTests, multipleEqualityDeletesDifferentKeyColumns) {
  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto eqDel1 = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile1 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile1->getPath(), {eqDel1});

  IcebergDeleteFile eqDelete1(
      FileContent::kEqualityDeletes,
      eqDelFile1->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile1->getPath()),
      /*equalityFieldIds=*/{1});

  auto eqDel2 = makeRowVector({"c1"}, {makeFlatVector<int64_t>({40})});
  auto eqDelFile2 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile2->getPath(), {eqDel2});

  IcebergDeleteFile eqDelete2(
      FileContent::kEqualityDeletes,
      eqDelFile2->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile2->getPath()),
      /*equalityFieldIds=*/{2});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete1, eqDelete2});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
      makeFlatVector<int64_t>({100, 300, 500}),
  });

  assertEqualResults({expected}, {result});
}

/// All rows deleted by equality deletes — verify that reading continues
/// past an empty chunk.
TEST_F(CudfIcebergGapTests, allRowsDeletedContinuesReading) {
  auto rowType = ROW({"c0"}, {BIGINT()});

  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto dataFile1 = TempFilePath::create();
  writeToFile(dataFile1->getPath(), data1);

  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), data2);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile1->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({5, 6}),
  });

  assertEqualResults({expected}, {result});
}

/// Equality delete file with extra non-key columns (for CDC).
TEST_F(CudfIcebergGapTests, equalityDeleteWithExtraNonKeyColumns) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto eqDel = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
      makeFlatVector<int64_t>({999, 888}),
  });
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Schema evolution: column added in the MIDDLE of the schema.
TEST_F(CudfIcebergGapTests, schemaEvolutionColumnAddedInMiddle) {
  auto fullType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});

  auto data1 = makeRowVector(
      {"c0", "c2"},
      {
          makeFlatVector<int64_t>({1, 2}),
          makeFlatVector<int64_t>({100, 200}),
      });
  auto dataFile1 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, dataFile1->getPath(), {data1});

  auto data2 = makeRowVector(
      {"c0", "c1", "c2"},
      {
          makeFlatVector<int64_t>({3, 4}),
          makeFlatVector<int64_t>({30, 40}),
          makeFlatVector<int64_t>({300, 400}),
      });
  auto dataFile2 = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, dataFile2->getPath(), {data2});

  auto splits1 = makeIcebergSplits(dataFile1->getPath());
  auto splits2 = makeIcebergSplits(dataFile2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(fullType)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 30, 40}),
      makeFlatVector<int64_t>({100, 200, 300, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// Empty data file (0 rows) with delete files attached.
TEST_F(CudfIcebergGapTests, emptyDataFileWithDeletes) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto emptyData = makeRowVector({
      makeFlatVector<int64_t>({}),
      makeFlatVector<int64_t>({}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), emptyData);

  auto realData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile2 = TempFilePath::create();
  writeToFile(dataFile2->getPath(), realData);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({2})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits1 = makeIcebergSplits(dataFile->getPath(), {eqDelete});
  auto splits2 = makeIcebergSplits(dataFile2->getPath(), {eqDelete});
  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>> all;
  all.insert(all.end(), splits1.begin(), splits1.end());
  all.insert(all.end(), splits2.begin(), splits2.end());

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(all).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3}),
      makeFlatVector<int64_t>({10, 30}),
  });

  assertEqualResults({expected}, {result});
}

/// Partition column with INT32 type.
TEST_F(CudfIcebergGapTests, partitionColumnInt32Type) {
  auto fullType = ROW({"c0", "year"}, {BIGINT(), INTEGER()});
  auto dataColumns = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"year", "2025"},
  };

  auto splits = makeIcebergSplits(dataFile->getPath(), {}, partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(dataColumns)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({2025, 2025, 2025}),
  });

  assertEqualResults({expected}, {result});
}

/// Partition column with INT64 type.
TEST_F(CudfIcebergGapTests, partitionColumnInt64Type) {
  auto fullType = ROW({"c0", "timestamp_ms"}, {BIGINT(), BIGINT()});
  auto dataColumns = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  std::unordered_map<std::string, std::optional<std::string>> partitionKeys = {
      {"timestamp_ms", "1700000000000"},
  };

  auto splits = makeIcebergSplits(dataFile->getPath(), {}, partitionKeys);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(fullType)
                  .dataColumns(dataColumns)
                  .endTableScan()
                  .planNode();

  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<int64_t>(
          {1700000000000LL, 1700000000000LL, 1700000000000LL}),
  });

  assertEqualResults({expected}, {result});
}

/// Deletion vector combined with equality deletes on the same data file.
TEST_F(CudfIcebergGapTests, deletionVectorPlusEqualityDelete) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60, 70, 80}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto bitmapData = serializeRoaringBitmapNoRun<int64_t>({0, 7});
  auto dvFile = writeDvFile(bitmapData);
  auto dvDelete = makeDvDeleteFile(
      dvFile->getPath(), bitmapData.size(), 2, 0, {}, /*dataSequenceNumber=*/2);

  auto eqDel = makeRowVector({makeFlatVector<int64_t>({30, 60})});
  auto eqDelFile = TempFilePath::create();
  writeDeleteFile(DeleteFileFormat::PARQUET, eqDelFile->getPath(), {eqDel});

  IcebergDeleteFile eqDelete(
      FileContent::kEqualityDeletes,
      eqDelFile->getPath(),
      dwio::common::FileFormat::PARQUET,
      2,
      getFileSize(eqDelFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {dvDelete, eqDelete}, {}, 1, /*dataSeq=*/1);

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 40, 50, 70}),
      makeFlatVector<int64_t>({2, 4, 5, 7}),
  });

  assertEqualResults({expected}, {result});
}

/// Deletion vector combined with positional deletes (V2 + V3 coexistence).
TEST_F(CudfIcebergGapTests, deletionVectorPlusPositionalDelete) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto bitmapData = serializeRoaringBitmapNoRun<int64_t>({0, 5});
  auto dvFile = writeDvFile(bitmapData);
  auto dvDelete = makeDvDeleteFile(
      dvFile->getPath(), bitmapData.size(), 2, 0, {}, /*dataSequenceNumber=*/2);

  auto pathColumn = IcebergMetadataColumn::icebergDeleteFilePathColumn();
  auto posColumn = IcebergMetadataColumn::icebergDeletePosColumn();
  auto posDeleteFile = TempFilePath::create();
  auto filePathVec = makeFlatVector<std::string>(
      1, [&](vector_size_t) { return dataFile->getPath(); });
  auto posVec = makeFlatVector<int64_t>({2});
  auto posDeleteVector =
      makeRowVector({pathColumn->name, posColumn->name}, {filePathVec, posVec});
  writeDeleteFile(
      DeleteFileFormat::DWRF,
      posDeleteFile->getPath(),
      std::vector<RowVectorPtr>{posDeleteVector});

  IcebergDeleteFile posDelete(
      FileContent::kPositionalDeletes,
      posDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(posDeleteFile->getPath()),
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto splits = makeIcebergSplits(
      dataFile->getPath(), {dvDelete, posDelete}, {}, 1, /*dataSeq=*/1);

  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({20, 40, 50}),
      makeFlatVector<int64_t>({2, 4, 5}),
  });

  assertEqualResults({expected}, {result});
}

} // namespace facebook::velox::cudf_velox::exec::test
