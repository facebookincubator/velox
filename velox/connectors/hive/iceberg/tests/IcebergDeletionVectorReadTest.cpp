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
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <numeric>

#include <folly/Singleton.h>

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/dwrf/reader/ReaderBase.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
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

// A deletion vector removes rows from the scan output, so an output row's
// index no longer matches its position in the data file. _row_id must still
// report file-absolute positions.
//
// The row-number column that carries those positions used to be injected only
// for V2 positional delete files, so a split whose only delete file was a V3
// deletion vector fell back to deriving positions from the output index. A
// DELETE with no WHERE clause hit this: with no filter to force the row-number
// column on, the rewritten deletion vector recorded 0..N-1 instead of the true
// positions and left the trailing rows of the file undeleted.
TEST_F(IcebergDeletionVectorReadTest, rowIdIsFileAbsoluteWithDeletionVector) {
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      {makeRowVector({makeFlatVector<int64_t>({10, 20, 30, 40, 50})})});

  auto [puffinDir, dvFile] = writeDeletionVector(dataFile->getPath(), {1, 3});

  const std::unordered_map<std::string, std::string> infoColumns{
      {IcebergMetadataColumn::kFirstRowIdInfoColumn, "200"},
      {IcebergMetadataColumn::kDataSequenceNumberInfoColumn, "42"}};

  const std::vector<std::string> outputNames{
      "c0", "_row_id", "_last_updated_sequence_number"};
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(
                      ROW({outputNames[0], outputNames[1], outputNames[2]},
                          {BIGINT(), BIGINT(), BIGINT()}))
                  .dataColumns(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();

  // Positions 1 and 3 are deleted, so the surviving rows keep file positions
  // 0, 2 and 4 and _row_id is firstRowId plus those positions.
  auto expected = makeRowVector(
      outputNames,
      {
          makeFlatVector<int64_t>({10, 30, 50}),
          makeFlatVector<int64_t>({200, 202, 204}),
          makeFlatVector<int64_t>({42, 42, 42}),
      });
  exec::test::AssertQueryBuilder(plan)
      .splits({makeIcebergSplitWithInfoColumns(
          dataFile->getPath(), infoColumns, {dvFile})})
      .assertResults(expected);
}

// Deletion-vector positions are absolute row ordinals in the base data file,
// while a split's 'start'/'length' are a byte range over that file. This
// pins down the conversion between those two coordinate systems for a split
// that begins at a nonzero byte offset.
//
// IcebergSplitReader resolves the two by asking the format reader, not by
// arithmetic on the byte offset: after creating the row reader it sets
// 'splitOffset_ = baseRowReader_->nextRowNumber()', which for DWRF/ORC is the
// cumulative row count of the preceding stripes (row groups for Parquet), and
// each batch then recovers its absolute range as
// 'splitOffset_ + baseReadOffset_'. A DV position preceding the split must be
// ignored rather than shifted onto a row this split actually reads.
TEST_F(IcebergDeletionVectorReadTest, deletionVectorAppliesToNonZeroByteSplit) {
  // Two DWRF stripes of five rows each: values 0..4 then 5..9, where each
  // value equals its absolute row position.
  auto dataFile = TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      {
          makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4})}),
          makeRowVector({makeFlatVector<int64_t>({5, 6, 7, 8, 9})}),
      },
      std::make_shared<dwrf::Config>(),
      []() {
        return std::make_unique<dwrf::LambdaFlushPolicy>([]() { return true; });
      });

  auto readFile = filesystems::getFileSystem(dataFile->getPath(), nullptr)
                      ->openFileForRead(dataFile->getPath());
  const uint64_t fileSize = readFile->size();
  dwio::common::ReaderOptions readerOptions{pool()};
  auto reader = std::make_unique<dwrf::ReaderBase>(
      readerOptions,
      std::make_unique<dwio::common::BufferedInput>(
          std::shared_ptr<ReadFile>(std::move(readFile)), *pool()));
  reader->loadCache();

  // The premise of the test: the second stripe really does start at a nonzero
  // byte offset and at absolute row 5, so the byte offset and the row offset
  // are different numbers and cannot be confused for one another.
  ASSERT_EQ(reader->footer().stripesSize(), 2);
  const uint64_t secondStripeByteOffset = reader->footer().stripes(1).offset();
  ASSERT_GT(secondStripeByteOffset, 0);
  ASSERT_EQ(reader->footer().stripes(0).numberOfRows(), 5);

  // Absolute data-file positions: 1 lives in the first stripe and is outside
  // this split, 7 is the third row of the second stripe.
  auto [puffinDir, dvFile] = writeDeletionVector(dataFile->getPath(), {1, 7});

  auto split = IcebergSplitBuilder(dataFile->getPath())
                   .connectorId(test::kIcebergConnectorId)
                   .fileFormat(fileFormat_)
                   .start(secondStripeByteOffset)
                   .length(fileSize - secondStripeByteOffset)
                   .deleteFiles({dvFile})
                   .build();

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"c0"}, {BIGINT()}))
                  .endTableScan()
                  .planNode();

  // Position 7 is removed from the second stripe. Position 1 precedes the
  // split and must not delete anything; in particular it must not be
  // misread as the second row of this split (value 6).
  auto expected = makeRowVector({makeFlatVector<int64_t>({5, 6, 8, 9})});
  exec::test::AssertQueryBuilder(plan).splits({split}).assertResults(expected);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
