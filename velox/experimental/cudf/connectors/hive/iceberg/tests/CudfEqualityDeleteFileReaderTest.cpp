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

/// End-to-end tests for the cudf Iceberg connector's handling of equality
/// delete files. Ported from the upstream EqualityDeleteFileReaderTest.
///
/// Data files are written as Parquet (via cudf writer) while equality delete
/// files are written as DWRF (via the upstream velox::dwrf::Writer) since
/// they are read by the upstream Velox EqualityDeleteFileReader, not cudf.

#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfIcebergTestBase.h"

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/Singleton.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector::hive::iceberg;

namespace facebook::velox::cudf_velox::exec::test {

class CudfEqualityDeleteFileReaderTest : public CudfIcebergTestBase {};

/// Basic single-column equality delete.
/// (Ported from upstream EqualityDeleteFileReaderTest::basicSingleColumnDelete)
TEST_F(CudfEqualityDeleteFileReaderTest, basicSingleColumnDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({3, 7}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 4, 5, 6, 8, 9}),
      makeFlatVector<int64_t>({10, 11, 12, 14, 15, 16, 18, 19}),
  });

  assertEqualResults({expected}, {result});
}

/// Multi-column equality deletes (both columns must match).
/// (Ported from upstream EqualityDeleteFileReaderTest::multiColumnDelete)
TEST_F(CudfEqualityDeleteFileReaderTest, multiColumnDelete) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({10, 20, 30, 10, 20}),
      makeFlatVector<int64_t>({100, 200, 300, 400, 500}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int32_t>({2, 5, 1}),
      makeFlatVector<int32_t>({20, 20, 20}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1, 2});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 3, 4}),
      makeFlatVector<int32_t>({10, 30, 10}),
      makeFlatVector<int64_t>({100, 300, 400}),
  });

  assertEqualResults({expected}, {result});
}

/// When no rows match, all rows survive.
/// (Ported from upstream EqualityDeleteFileReaderTest::noMatchingDeletes)
TEST_F(CudfEqualityDeleteFileReaderTest, noMatchingDeletes) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({100, 200}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });

  assertEqualResults({expected}, {result});
}

/// All rows deleted.
/// (Ported from upstream EqualityDeleteFileReaderTest::allRowsDeleted)
TEST_F(CudfEqualityDeleteFileReaderTest, allRowsDeleted) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1});

  auto splits = makeIcebergSplits(dataFile->getPath(), {icebergDeleteFile});
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  EXPECT_EQ(result->size(), 0);
}

/// Equality deletes with higher sequence number should apply.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberDeleteApplies)
TEST_F(CudfEqualityDeleteFileReaderTest, sequenceNumberDeleteApplies) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({2, 4}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      2,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/3);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 5}),
      makeFlatVector<int64_t>({10, 30, 50}),
  });

  assertEqualResults({expected}, {result});
}

/// Equality deletes with lower sequence number should be skipped.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberDeleteSkipped)
TEST_F(CudfEqualityDeleteFileReaderTest, sequenceNumberDeleteSkipped) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/2);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int64_t>({10, 20, 30}),
  });

  assertEqualResults({expected}, {result});
}

/// Equal sequence numbers should also skip.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberEqualSkipped)
TEST_F(CudfEqualityDeleteFileReaderTest, sequenceNumberEqualSkipped) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      3,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/5);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });

  assertEqualResults({expected}, {result});
}

/// Sequence number 0 means legacy/unassigned — always apply.
/// (Ported from upstream
/// EqualityDeleteFileReaderTest::sequenceNumberZeroAlwaysApplies)
TEST_F(CudfEqualityDeleteFileReaderTest, sequenceNumberZeroAlwaysApplies) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile = TempFilePath::create();
  writeDeleteFile(eqDeleteFile->getPath(), {deleteData});

  IcebergDeleteFile icebergDeleteFile(
      FileContent::kEqualityDeletes,
      eqDeleteFile->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/10);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3}),
  });

  assertEqualResults({expected}, {result});
}

/// Mixed sequence numbers: only delete files with higher seqNum apply.
/// (Ported from upstream EqualityDeleteFileReaderTest::mixedSequenceNumbers)
TEST_F(CudfEqualityDeleteFileReaderTest, mixedSequenceNumbers) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});

  auto baseData = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
  });
  auto dataFile = TempFilePath::create();
  writeToFile(dataFile->getPath(), baseData);

  auto deleteData1 = makeRowVector({
      makeFlatVector<int64_t>({2}),
  });
  auto eqDeleteFile1 = TempFilePath::create();
  writeDeleteFile(eqDeleteFile1->getPath(), {deleteData1});
  IcebergDeleteFile icebergDeleteFile1(
      FileContent::kEqualityDeletes,
      eqDeleteFile1->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile1->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/10);

  auto deleteData2 = makeRowVector({
      makeFlatVector<int64_t>({4}),
  });
  auto eqDeleteFile2 = TempFilePath::create();
  writeDeleteFile(eqDeleteFile2->getPath(), {deleteData2});
  IcebergDeleteFile icebergDeleteFile2(
      FileContent::kEqualityDeletes,
      eqDeleteFile2->getPath(),
      dwio::common::FileFormat::DWRF,
      1,
      getFileSize(eqDeleteFile2->getPath()),
      /*equalityFieldIds=*/{1},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/3);

  auto splits = makeIcebergSplits(
      dataFile->getPath(),
      {icebergDeleteFile1, icebergDeleteFile2},
      /*partitionKeys=*/{},
      /*splitCount=*/1,
      /*dataSequenceNumber=*/5);
  auto plan = makeTableScanPlan(rowType);
  auto result = AssertQueryBuilder(plan).splits(splits).copyResults(pool());

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 30, 40, 50}),
  });

  assertEqualResults({expected}, {result});
}

} // namespace facebook::velox::cudf_velox::exec::test
