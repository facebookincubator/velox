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

#include <folly/Singleton.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/tests/CudfIcebergTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::connector::hive::iceberg;

namespace facebook::velox::cudf_velox::exec::test {

class CudfIcebergReadTest : public CudfIcebergTestBase {
 protected:
  static std::vector<int64_t> makeContinuousIncreasingValues(
      int64_t begin,
      int64_t end) {
    std::vector<int64_t> values;
    values.resize(end - begin);
    std::iota(values.begin(), values.end(), begin);
    return values;
  }
};

/// Basic read without any deletes - verifies the connector can read plain
/// parquet files through the Iceberg path.
TEST_F(CudfIcebergReadTest, basicRead) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfIcebergConnectorId,
          "iceberg_table",
          facebook::velox::common::SubfieldFilters{},
          nullptr,
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read with multiple columns.
TEST_F(CudfIcebergReadTest, multiColumn) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto data = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  });

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({1.1, 2.2, 3.3}),
  });
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read a larger file to verify chunked reading works.
TEST_F(CudfIcebergReadTest, largerFile) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});
  auto values = makeContinuousIncreasingValues(0, 10000);
  auto data = makeRowVector({makeFlatVector<int64_t>(values)});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, splits, "SELECT * FROM tmp", 0);
}

/// Read with a deletion vector that deletes specific rows.
TEST_F(CudfIcebergReadTest, deletionVector) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});
  auto values = makeContinuousIncreasingValues(0, 100);
  auto data = makeRowVector({makeFlatVector<int64_t>(values)});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  // Create a deletion vector file containing roaring bitmap with positions
  // 10, 20, 30, 40, 50 deleted. We use the Velox DeletionVectorWriter or
  // simply create a binary roaring bitmap.
  //
  // Construct a Roaring64 bitmap blob for positions {10,20,30,40,50}.
  // The cudf experimental reader expects 64-bit roaring bitmap (portable) format.
  auto dvFilePath = TempFilePath::create();
  {
    // Build a Roaring64 bitmap for positions {10, 20, 30, 40, 50}.
    // The cudf experimental reader expects 64-bit roaring bitmap format:
    //   [uint64: num_buckets]
    //   For each bucket:
    //     [uint32: bucket_key (upper 32 bits)]
    //     [standard 32-bit roaring bitmap (portable format)]
    //
    // All our positions are < 2^16, so they fall in one bucket with key=0.
    std::vector<uint8_t> blob;
    auto appendLE = [&blob](auto val) {
      val = folly::Endian::little(val);
      blob.insert(
          blob.end(),
          reinterpret_cast<uint8_t*>(&val),
          reinterpret_cast<uint8_t*>(&val) + sizeof(val));
    };

    // Roaring64 header: 1 bucket.
    appendLE(static_cast<uint64_t>(1));
    // Bucket key: 0 (upper 32 bits of 64-bit values).
    appendLE(static_cast<uint32_t>(0));

    // Now the standard 32-bit roaring bitmap for the lower 32 bits.
    // Cookie: 12346 (SERIAL_COOKIE_NO_RUNCONTAINER).
    appendLE(static_cast<uint32_t>(12346));
    // Number of containers: 1.
    appendLE(static_cast<uint32_t>(1));
    // Descriptive header: container key=0, cardinality-1=4.
    appendLE(static_cast<uint16_t>(0));
    appendLE(static_cast<uint16_t>(4));
    // Offset header: byte offset from start of 32-bit bitmap to container data.
    // 32-bit bitmap layout: cookie(4) + numContainers(4) + desc(4) + offset(4) = 16.
    appendLE(static_cast<uint32_t>(16));
    // Array container: 5 uint16 values.
    for (uint16_t pos : {10, 20, 30, 40, 50}) {
      appendLE(pos);
    }

    // Write blob to file. TempFilePath::create() already creates the file,
    // so we need to allow overwriting.
    auto fs = filesystems::getFileSystem(dvFilePath->getPath(), nullptr);
    filesystems::FileOptions writeOptions;
    writeOptions.shouldThrowOnFileAlreadyExists = false;
    auto writeFile =
        fs->openFileForWrite(dvFilePath->getPath(), writeOptions);
    writeFile->append(
        std::string(reinterpret_cast<char*>(blob.data()), blob.size()));
    writeFile->close();
  }

  // Create IcebergDeleteFile for the DV.
  IcebergDeleteFile dvDeleteFile(
      FileContent::kDeletionVector,
      dvFilePath->getPath(),
      dwio::common::FileFormat::PARQUET,
      5,
      filesystems::getFileSystem(dvFilePath->getPath(), nullptr)
          ->openFileForRead(dvFilePath->getPath())
          ->size());

  auto splits = makeIcebergSplits(filePath->getPath(), {dvDeleteFile});

  // Build expected result: all rows except 10, 20, 30, 40, 50.
  std::vector<int64_t> expectedValues;
  for (int64_t i = 0; i < 100; ++i) {
    if (i != 10 && i != 20 && i != 30 && i != 40 && i != 50) {
      expectedValues.push_back(i);
    }
  }
  auto expected =
      makeRowVector({makeFlatVector<int64_t>(expectedValues)});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read with an empty split (no delete files).
TEST_F(CudfIcebergReadTest, noDeleteFiles) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeRowVector({makeFlatVector<int64_t>({100, 200, 300})});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  auto splits = makeIcebergSplits(filePath->getPath());

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto expected =
      makeRowVector({makeFlatVector<int64_t>({100, 200, 300})});
  AssertQueryBuilder(plan).splits(splits).assertResults({expected});
}

/// Read with multiple data files (multiple splits).
TEST_F(CudfIcebergReadTest, multipleSplits) {
  folly::SingletonVault::singleton()->registrationComplete();

  auto rowType = ROW({"c0"}, {BIGINT()});

  auto data1 = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto data2 = makeRowVector({makeFlatVector<int64_t>({4, 5, 6})});

  auto filePath1 = TempFilePath::create();
  auto filePath2 = TempFilePath::create();
  writeToFile(filePath1->getPath(), data1);
  writeToFile(filePath2->getPath(), data2);

  auto splits1 = makeIcebergSplits(filePath1->getPath());
  auto splits2 = makeIcebergSplits(filePath2->getPath());

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
      allSplits;
  allSplits.insert(allSplits.end(), splits1.begin(), splits1.end());
  allSplits.insert(allSplits.end(), splits2.begin(), splits2.end());

  createDuckDbTable({data1, data2});

  auto plan = PlanBuilder()
                  .startTableScan()
                  .connectorId(kCudfIcebergConnectorId)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, allSplits, "SELECT * FROM tmp", 0);
}

} // namespace facebook::velox::cudf_velox::exec::test
