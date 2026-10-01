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

#include <gtest/gtest.h>
#include "velox/common/file/LocalFile.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/delta/DeltaSplitReader.h"
#include "velox/connectors/hive/delta/HiveDeltaSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::delta;

namespace {

class DeltaSplitReaderTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
    // Install the Delta dispatch in HiveSplitReader's factory registry so a
    // Hive split carrying customSplitInfo["table_format"] = "hive-delta" is
    // routed to DeltaSplitReader. Idempotent.
    registerHiveDeltaSplitReader();
  }

  // Writes a single-batch Parquet file to a fresh temp path and returns it.
  std::shared_ptr<common::testutil::TempFilePath> writeParquetFile(
      const RowVectorPtr& data) {
    auto file = common::testutil::TempFilePath::create();
    auto writeFile = std::make_unique<LocalWriteFile>(
        file->getPath(),
        /*shouldCreate=*/true,
        /*shouldThrowOnFileAlreadyExists=*/false);
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), file->getPath());
    dwio::common::WriterOptions writerOptions;
    writerOptions.memoryPool = rootPool_.get();
    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink), writerOptions, asRowType(data->type()));
    writer->write(data);
    writer->close();
    return file;
  }

  // Builds a HiveDeltaSplit for 'filePath' with the given partition values.
  // Stamps customSplitInfo["table_format"]="hive-delta" so HiveSplitReader's
  // factory registry dispatches to DeltaSplitReader.
  std::shared_ptr<HiveDeltaSplit> makeDeltaSplit(
      const std::string& filePath,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys) {
    return std::make_shared<HiveDeltaSplit>(
        exec::test::kHiveConnectorId,
        filePath,
        dwio::common::FileFormat::PARQUET,
        /*start=*/0,
        /*length=*/std::numeric_limits<uint64_t>::max(),
        partitionKeys,
        /*tableBucketNumber=*/std::nullopt,
        /*customSplitInfo=*/
        std::unordered_map<std::string, std::string>{
            {"table_format", "hive-delta"}});
  }

  // Builds a column-handle map matching 'rowType' with the columns at
  // 'partitionIndices' marked as partition keys and the rest regular.
  ColumnHandleMap makeAssignments(
      const RowTypePtr& rowType,
      const std::unordered_set<int>& partitionIndices) {
    ColumnHandleMap assignments;
    for (int32_t i = 0; i < rowType->size(); ++i) {
      const auto& name = rowType->nameOf(i);
      const auto& type = rowType->childAt(i);
      if (partitionIndices.contains(i)) {
        assignments[name] = partitionKey(name, type);
      } else {
        assignments[name] = regularColumn(name, type);
      }
    }
    return assignments;
  }
};

// Two consecutive Delta splits run through a single driver. One file is
// missing a data column that the next file has, and partition values change
// across splits. Verifies that DeltaSplitReader::adaptColumns correctly
// resets the per-ScanSpec constant/non-constant state between splits — the
// invariant covered by scanSpec_->resetCachedValues(false) at the end of
// adaptColumns.
//
// Addresses vuule's review comment on PR #17703 (velox/connectors/hive/delta/
// tests/CMakeLists.txt:16):
//   "Could we add an actual DeltaSplitReader test with two consecutive
//    splits and one driver? A missing-to-present column and changing
//    partition values would cover the per-split state here."
TEST_F(DeltaSplitReaderTest, twoSplitsMissingToPresentColumn) {
  // Logical table schema: (id BIGINT, val BIGINT, part VARCHAR).
  auto outputType =
      ROW({"id", "val", "part"}, {BIGINT(), BIGINT(), VARCHAR()});

  // Split A's file has only (id); 'val' is missing and must materialize as
  // null (Delta schema evolution). Partition 'part' = "A".
  auto fileA = writeParquetFile(
      makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})}));

  // Split B's file has (id, val); 'val' is read from the file. Partition
  // 'part' = "B". The adaptColumns pass for split B must clear the constant
  // that was installed on 'val' by split A, otherwise B's val reads would
  // return null constants instead of file values.
  auto fileB = writeParquetFile(
      makeRowVector(
          {"id", "val"},
          {makeFlatVector<int64_t>({10, 20, 30}),
           makeFlatVector<int64_t>({100, 200, 300})}));

  auto splitA = makeDeltaSplit(
      fileA->getPath(),
      {{"part", std::optional<std::string>{"A"}}});
  auto splitB = makeDeltaSplit(
      fileB->getPath(),
      {{"part", std::optional<std::string>{"B"}}});

  // Expected: 6 rows — three from A (val NULL, part="A"), three from B
  // (val 100/200/300, part="B").
  auto expected = makeRowVector(
      outputType->names(),
      {
          makeFlatVector<int64_t>({1, 2, 3, 10, 20, 30}),
          makeNullableFlatVector<int64_t>(
              {std::nullopt,
               std::nullopt,
               std::nullopt,
               100,
               200,
               300}),
          makeFlatVector<std::string>({"A", "A", "A", "B", "B", "B"}),
      });

  auto assignments = makeAssignments(outputType, /*partitionIndices=*/{2});
  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .outputType(outputType)
                  .dataColumns(outputType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  // maxDrivers(1) forces both splits through a single driver so the
  // ScanSpec is reused. Without the resetCachedValues call at the end of
  // adaptColumns, split B would either read null for 'val' or trip a stale
  // hasFilter_ check.
  exec::test::AssertQueryBuilder(plan)
      .maxDrivers(1)
      .split(splitA)
      .split(splitB)
      .assertResults(expected);
}

} // namespace
