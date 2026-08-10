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

#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <numeric>

#include <folly/Singleton.h>

#include "velox/dwio/common/FileSink.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using TempFilePath = common::testutil::TempFilePath;
using TempDirectoryPath = common::testutil::TempDirectoryPath;

// End-to-end reproduction of GitHub issue #18126: querying an Iceberg V3 table
// whose rows are removed by a Spark-written deletion vector (a Puffin
// deletion-vector-v1 blob) failed with "Deletion-vector-v1 CRC-32 mismatch"
// because the reader validated the trailing CRC-32 with the wrong (un-inverted)
// algorithm. This exercises the full TableScan path -- data file + Puffin DV ->
// scan -> deleted rows filtered out -- against a spec-compliant DV frame that
// carries the standard IEEE CRC-32 Iceberg/Spark writes.
class IcebergDeletionVectorReadTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    fileFormat_ = dwio::common::FileFormat::DWRF;
    dwio::common::LocalFileSink::registerFactory();
  }

  // Writes a spec-compliant Puffin deletion-vector-v1 file (PFA1 frame + magic
  // + portable roaring bitmap + standard CRC-32) for 'deletedPositions',
  // mirroring what Spark/Iceberg emits, and returns an IcebergDeleteFile
  // pointing at the DV blob inside it. The returned TempDirectoryPath keeps the
  // Puffin file alive for the duration of the scan.
  std::pair<std::shared_ptr<TempDirectoryPath>, IcebergDeleteFile>
  writeDeletionVector(
      const std::string& referencedDataFile,
      const std::vector<int64_t>& deletedPositions) {
    DeletionVectorWriter writer;
    writer.addDeletedPositions(deletedPositions);
    const auto blob = writer.serialize();

    // LocalFileSink refuses to open an existing file, so write the Puffin into
    // a fresh path under a temp directory rather than a pre-created TempFile.
    auto puffinDir = TempDirectoryPath::create();
    const std::string puffinPath = puffinDir->getPath() + "/deletes.puffin";
    auto sink = dwio::common::FileSink::create(
        "file:" + puffinPath, {.pool = pool_.get()});
    const auto [blobOffset, blobLength] = writePuffinFile(
        *sink,
        *pool_,
        blob,
        referencedDataFile,
        static_cast<int64_t>(deletedPositions.size()));
    sink->close();

    IcebergDeleteFile dvFile(
        FileContent::kDeletionVector,
        puffinPath,
        fileFormat_,
        /*recordCount=*/deletedPositions.size(),
        /*fileSizeInBytes=*/getFileSize(puffinPath),
        /*equalityFieldIds=*/{},
        /*lowerBounds=*/{},
        /*upperBounds=*/{},
        /*dataSequenceNumber=*/0,
        /*contentOffset=*/static_cast<int64_t>(blobOffset),
        /*contentLength=*/static_cast<int64_t>(blobLength),
        /*referencedDataFile=*/referencedDataFile);
    return {puffinDir, dvFile};
  }
};

TEST_F(IcebergDeletionVectorReadTest, deletionVectorFiltersDeletedRowsInScan) {
  // Data file: 10 rows whose value equals their row position (0..9).
  std::vector<int64_t> values(10);
  std::iota(values.begin(), values.end(), 0);
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(), {makeRowVector({makeFlatVector<int64_t>(values)})});

  // Delete positions 8 and 9 (mirrors the issue's DELETE of ids 8, 9).
  auto [puffinDir, dvFile] = writeDeletionVector(dataFile->getPath(), {8, 9});

  auto splits = makeIcebergSplits(dataFile->getPath(), {dvFile});

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();

  // Positions 8 and 9 are deleted, so values 0..7 remain.
  auto expected =
      makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7})});
  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
