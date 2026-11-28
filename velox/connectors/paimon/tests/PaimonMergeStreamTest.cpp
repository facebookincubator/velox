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
#include "velox/connectors/paimon/PaimonMergeStream.h"

#include <fmt/ranges.h>
#include <folly/lang/Bits.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <fstream>
#include <numeric>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/connectors/common/FileDataSource.h"
#include "velox/connectors/hive/iceberg/DeletionVectorWriter.h"
#include "velox/connectors/paimon/PaimonConfig.h"
#include "velox/connectors/paimon/PaimonSplitReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::paimon {
namespace {

const std::string kConnectorId{"test-paimon"};

// Table handle of the merge reads, which read the merge columns of a file with
// no filters, as PaimonSplitReader::openMergeStreams() builds it.
class MergeTableHandle final : public connector::FileTableHandle {
 public:
  MergeTableHandle(
      const connector::FileTableHandle& tableHandle,
      RowTypePtr mergeType)
      : connector::FileTableHandle(
            tableHandle.connectorId(),
            tableHandle.tableName(),
            /*subfieldFilters=*/{},
            /*remainingFilter=*/nullptr,
            std::move(mergeType),
            tableHandle.tableParameters(),
            /*sampleRate=*/1.0,
            tableHandle.dbName()) {}

  std::vector<connector::FileColumnHandlePtr> filterColumnHandles()
      const override {
    return {};
  }
};

class PaimonMergeStreamTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    queryCtx_ = core::QueryCtx::create();
    connectorQueryCtx_ =
        ConnectorQueryCtx::Builder()
            .operatorPool(pool())
            .connectorPool(rootPool_.get())
            .sessionProperties(sessionProperties_.get())
            .expressionEvaluator(
                std::make_unique<exec::SimpleExpressionEvaluator>(
                    queryCtx_.get(), pool()))
            .queryId("PaimonMergeStreamTest")
            .taskId("PaimonMergeStreamTest")
            .planNodeId("scan")
            .build();
  }

  void TearDown() override {
    connectorQueryCtx_.reset();
    queryCtx_.reset();
    HiveConnectorTestBase::TearDown();
  }

  // Returns the handle of a primary-key table of 'rowType'.
  static std::shared_ptr<PaimonTableHandle> makeTableHandle(
      const RowTypePtr& rowType,
      const std::vector<std::string>& primaryKeys,
      const std::vector<std::string>& partitionKeys = {},
      std::unordered_map<std::string, std::string> options = {}) {
    std::vector<int32_t> fieldIds(rowType->size());
    std::iota(fieldIds.begin(), fieldIds.end(), 0);
    return std::make_shared<PaimonTableHandle>(
        kConnectorId,
        "paimon_table",
        /*schemaId=*/0,
        rowType,
        std::move(fieldIds),
        PaimonTableType::kPrimaryKey,
        partitionKeys,
        std::move(options),
        common::SubfieldFilters{},
        /*remainingFilter=*/nullptr,
        std::vector<connector::FileColumnHandlePtr>{},
        /*sampleRate=*/1.0,
        /*dbName=*/"",
        primaryKeys);
  }

  // Returns the layout of the merge columns of 'tableHandle' followed by
  // 'valueColumns'.
  static std::shared_ptr<const PaimonMergeLayout> makeLayout(
      const PaimonTableHandle& tableHandle,
      const RowTypePtr& valueColumns) {
    return std::make_shared<const PaimonMergeLayout>(PaimonMergeLayout::create(
        tableHandle, valueColumns, /*lowerCaseSystemColumns=*/false));
  }

  // Writes 'rows' to a DWRF file in the layout of Paimon primary-key files,
  // [_KEY_<key> for each of 'keys', _SEQUENCE_NUMBER, _VALUE_KIND, table
  // columns], and returns its data file. Rows are inserts unless 'rowKinds'
  // says otherwise.
  PaimonDataFile writeFile(
      const std::vector<std::string>& keys,
      const RowVectorPtr& rows,
      const std::vector<int64_t>& sequenceNumbers,
      const std::vector<PaimonRowKind>& rowKinds = {}) {
    VELOX_CHECK_EQ(sequenceNumbers.size(), rows->size());
    std::vector<int8_t> valueKinds(
        rows->size(), static_cast<int8_t>(PaimonRowKind::kInsert));
    if (!rowKinds.empty()) {
      VELOX_CHECK_EQ(rowKinds.size(), rows->size());
      std::transform(
          rowKinds.begin(), rowKinds.end(), valueKinds.begin(), [](auto kind) {
            return static_cast<int8_t>(kind);
          });
    }
    std::vector<std::string> names;
    std::vector<VectorPtr> columns;
    for (const auto& key : keys) {
      names.push_back(fmt::format("_KEY_{}", key));
      columns.push_back(rows->childAt(key));
    }
    names.emplace_back("_SEQUENCE_NUMBER");
    columns.push_back(makeFlatVector<int64_t>(sequenceNumbers));
    names.emplace_back("_VALUE_KIND");
    columns.push_back(makeFlatVector<int8_t>(valueKinds));
    const auto& rowType = rows->rowType();
    names.insert(names.end(), rowType->names().begin(), rowType->names().end());
    columns.insert(
        columns.end(), rows->children().begin(), rows->children().end());
    auto file = writeToFile(makeRowVector(names, columns));

    PaimonDataFile dataFile;
    dataFile.path = file->getPath();
    dataFile.fileFormat = dwio::common::FileFormat::DWRF;
    dataFile.rowCount = rows->size();
    dataFile.minSequenceNumber =
        *std::min_element(sequenceNumbers.begin(), sequenceNumbers.end());
    dataFile.maxSequenceNumber =
        *std::max_element(sequenceNumbers.begin(), sequenceNumbers.end());
    files_.push_back(std::move(file));
    return dataFile;
  }

  // Writes a deletion vector of 'positions' into an index file as Paimon Java
  // does: a version byte, then a version 2 frame of [length][magic][Roaring64
  // bitmap][CRC]. Returns its deletion file.
  PaimonDeletionFile writeDeletionVector(
      const std::vector<int64_t>& positions) {
    hive::iceberg::DeletionVectorWriter writer;
    writer.addDeletedPositions(positions);
    const auto bitmap = writer.serialize();
    const auto appendInt = [](std::string& data, uint32_t value, bool big) {
      value = big ? folly::Endian::big(value) : folly::Endian::little(value);
      data.append(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    std::string data(1, '\1');
    const auto offset = data.size();
    const auto frameLength =
        static_cast<uint32_t>(sizeof(uint32_t) + bitmap.size());
    appendInt(data, frameLength, /*big=*/true);
    appendInt(data, 1'681'511'377, /*big=*/false);
    data += bitmap;
    // Paimon Java skips the CRC on read, so leave it zero.
    appendInt(data, 0, /*big=*/true);
    auto file = exec::test::TempFilePath::create();
    {
      std::ofstream out(file->getPath(), std::ios::binary);
      out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    PaimonDeletionFile deletionFile(
        file->getPath(),
        offset,
        frameLength + 8,
        writer.numDistinctPositions());
    files_.push_back(std::move(file));
    return deletionFile;
  }

  // Opens the merge stream of 'dataFile', the file at 'fileIndex' of a split,
  // which reads the columns of 'layout' 'batchSize' rows at a time, as
  // PaimonSplitReader::openMergeStreams() does. 'dataFile' must outlive the
  // stream.
  std::unique_ptr<PaimonMergeStream> openStream(
      size_t fileIndex,
      const PaimonDataFile& dataFile,
      const PaimonTableHandle& tableHandle,
      const std::shared_ptr<const PaimonMergeLayout>& layout,
      uint64_t batchSize) {
    const PaimonConnectorSplit split(
        kConnectorId,
        /*snapshotId=*/1,
        PaimonTableType::kPrimaryKey,
        {dataFile},
        /*partitionKeys=*/{},
        /*tableBucketNumber=*/std::nullopt,
        /*rawConvertible=*/false);
    auto reader = connector::FileSplitReader::create(
        PaimonSplitReader::makeFileConnectorSplit(split, dataFile),
        std::make_shared<MergeTableHandle>(tableHandle, layout->type),
        &partitionKeys_,
        connectorQueryCtx_.get(),
        fileConfig_,
        layout->type,
        ioStats_,
        metadataIoStats_,
        connectorIoStats_,
        &fileHandleFactory_,
        /*ioExecutor=*/nullptr,
        connector::FileDataSource::makeScanSpec(
            layout->type,
            /*outputSubfields=*/{},
            /*subfieldFilters=*/{},
            layout->type,
            /*partitionKeys=*/{},
            /*infoColumns=*/{},
            /*specialColumns=*/{},
            /*disableStatsBasedFilterReorder=*/false,
            pool()));
    reader->initialize(/*randomSkip=*/nullptr, /*remainingFilterColumns=*/{});
    reader->prepareSplit(/*metadataFilter=*/nullptr, runtimeStats_);
    auto deletionVectorReader = dataFile.deletionFile.has_value()
        ? std::make_unique<PaimonDeletionVectorReader>(
              *dataFile.deletionFile,
              pool(),
              fileConfig_->config(),
              /*tokenProvider=*/nullptr)
        : nullptr;
    return std::make_unique<PaimonMergeStream>(
        fileIndex,
        dataFile,
        std::move(reader),
        std::move(deletionVectorReader),
        layout,
        batchSize,
        pool());
  }

  const std::shared_ptr<config::ConfigBase> sessionProperties_{
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{})};
  const std::shared_ptr<const PaimonConfig> fileConfig_{
      std::make_shared<PaimonConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}))};
  const std::unordered_map<std::string, connector::FileColumnHandlePtr>
      partitionKeys_;
  const std::shared_ptr<io::IoStatistics> ioStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<IoStats> connectorIoStats_{std::make_shared<IoStats>()};

  FileHandleFactory fileHandleFactory_{
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(0),
      std::make_unique<FileHandleGenerator>()};
  dwio::common::RuntimeStats runtimeStats_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  // Data and deletion files, which must outlive the streams that read them.
  std::vector<std::shared_ptr<exec::test::TempFilePath>> files_;
};

// The layout holds the merge columns, which are the primary keys that are not
// partition columns, the sequence fields, _SEQUENCE_NUMBER and _VALUE_KIND, and
// then the value columns. All before _VALUE_KIND order rows.
TEST_F(PaimonMergeStreamTest, layout) {
  const auto rowType = ROW(
      {{"dt", VARCHAR()},
       {"id", BIGINT()},
       {"ts", BIGINT()},
       {"total", BIGINT()},
       {"name", VARCHAR()}});
  const auto tableHandle = makeTableHandle(
      rowType,
      /*primaryKeys=*/{"dt", "id"},
      /*partitionKeys=*/{"dt"},
      {{"sequence.field", "ts"}, {"sequence.field.sort-order", "descending"}});
  {
    const auto layout = PaimonMergeLayout::create(
        *tableHandle,
        ROW({{"total", BIGINT()}, {"name", VARCHAR()}}),
        /*lowerCaseSystemColumns=*/false);
    EXPECT_EQ(
        *layout.type,
        *ROW(
            {{"id", BIGINT()},
             {"ts", BIGINT()},
             {"_SEQUENCE_NUMBER", BIGINT()},
             {"_VALUE_KIND", TINYINT()},
             {"total", BIGINT()},
             {"name", VARCHAR()}}));
    EXPECT_EQ(layout.numKeys, 1u);
    EXPECT_EQ(layout.sequenceNumberChannel(), 2u);
    EXPECT_EQ(layout.valueKindChannel(), 3u);
    ASSERT_EQ(layout.compareFlags.size(), 3u);
    EXPECT_TRUE(layout.compareFlags[0].ascending);
    EXPECT_FALSE(layout.compareFlags[1].ascending);
    EXPECT_TRUE(layout.compareFlags[2].ascending);
    for (const auto& flags : layout.compareFlags) {
      EXPECT_TRUE(flags.nullsFirst);
    }
  }
  {
    const auto layout = PaimonMergeLayout::create(
        *tableHandle, ROW({}, {}), /*lowerCaseSystemColumns=*/true);
    EXPECT_THAT(
        layout.type->names(),
        testing::ElementsAre("id", "ts", "_sequence_number", "_value_kind"));
  }

  // A key or a sequence field is already a merge column.
  VELOX_ASSERT_THROW(
      PaimonMergeLayout::create(
          *tableHandle,
          ROW({{"id", BIGINT()}}),
          /*lowerCaseSystemColumns=*/false),
      "Duplicate Paimon merge column: id");
  VELOX_ASSERT_THROW(
      PaimonMergeLayout::create(
          *tableHandle,
          ROW({{"ts", BIGINT()}}),
          /*lowerCaseSystemColumns=*/false),
      "Duplicate Paimon merge column: ts");
}

// A stream returns the rows of its file in file order, reading batches as
// needed and skipping the rows its deletion vector deletes.
TEST_F(PaimonMergeStreamTest, readRows) {
  const auto rowType = ROW({{"id", BIGINT()}, {"total", BIGINT()}});
  const auto tableHandle = makeTableHandle(rowType, {"id"});
  const auto layout = makeLayout(*tableHandle, ROW({{"total", BIGINT()}}));
  const auto totalChannel = layout->type->getChildIdx("total");
  const auto rows = makeRowVector(
      {"id", "total"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int64_t>({10, 20, 30, 40, 50})});

  struct TestParam {
    std::vector<int64_t> deletedRows;
    std::vector<uint64_t> expectedFileRows;
    std::vector<int64_t> expectedTotals;

    std::string debugString() const {
      return fmt::format("deletedRows [{}]", fmt::join(deletedRows, ", "));
    }
  };
  // Batches of 2 rows: deleting rows 2 and 3 deletes the whole second batch.
  const std::vector<TestParam> testSettings{
      {{}, {0, 1, 2, 3, 4}, {10, 20, 30, 40, 50}},
      {{0, 2, 3}, {1, 4}, {20, 50}},
      {{0, 1, 2, 3, 4}, {}, {}},
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto dataFile = writeFile({"id"}, rows, {1, 2, 3, 4, 5});
    if (!testData.deletedRows.empty()) {
      dataFile.deletionFile = writeDeletionVector(testData.deletedRows);
    }
    auto stream =
        openStream(0, dataFile, *tableHandle, layout, /*batchSize=*/2);
    std::vector<uint64_t> fileRows;
    std::vector<int64_t> totals;
    while (stream->hasData()) {
      fileRows.push_back(stream->fileRow());
      totals.push_back(stream->batch()
                           ->childAt(totalChannel)
                           ->asChecked<SimpleVector<int64_t>>()
                           ->valueAt(stream->row()));
      stream->pop();
    }
    EXPECT_EQ(fileRows, testData.expectedFileRows);
    EXPECT_EQ(totals, testData.expectedTotals);
  }
}

// hasSameKey() compares every key column of the current row with a row of
// another stream, and no other column.
TEST_F(PaimonMergeStreamTest, hasSameKey) {
  const auto rowType =
      ROW({{"id", BIGINT()}, {"part", VARCHAR()}, {"total", BIGINT()}});
  const std::vector<std::string> keys{"id", "part"};
  const auto tableHandle = makeTableHandle(rowType, keys);
  const auto layout = makeLayout(*tableHandle, ROW({{"total", BIGINT()}}));
  // Key (1, b) differs from (1, a) in its second column only, and from (2, b)
  // in its first. The rows of key (1, a) differ in sequence number and total.
  const auto olderFile = writeFile(
      keys,
      makeRowVector(
          {"id", "part", "total"},
          {makeFlatVector<int64_t>({1, 1, 2}),
           makeFlatVector<std::string>({"a", "b", "b"}),
           makeFlatVector<int64_t>({10, 11, 20})}),
      {1, 2, 3});
  const auto newerFile = writeFile(
      keys,
      makeRowVector(
          {"id", "part", "total"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<std::string>({"a", "b"}),
           makeFlatVector<int64_t>({12, 21})}),
      {4, 5});
  auto older = openStream(0, olderFile, *tableHandle, layout, /*batchSize=*/10);
  auto newer = openStream(1, newerFile, *tableHandle, layout, /*batchSize=*/10);
  const auto& olderBatch = *older->batch();

  // 'newer' is at key (1, a).
  EXPECT_TRUE(newer->hasSameKey(olderBatch, 0));
  EXPECT_FALSE(newer->hasSameKey(olderBatch, 1));
  EXPECT_FALSE(newer->hasSameKey(olderBatch, 2));

  newer->pop();
  // 'newer' is at key (2, b).
  EXPECT_FALSE(newer->hasSameKey(olderBatch, 0));
  EXPECT_FALSE(newer->hasSameKey(olderBatch, 1));
  EXPECT_TRUE(newer->hasSameKey(olderBatch, 2));
}

// rowKind() returns the _VALUE_KIND of the current row, and isRetract() is true
// for DELETE and UPDATE_BEFORE rows.
TEST_F(PaimonMergeStreamTest, rowKind) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto tableHandle = makeTableHandle(rowType, {"id"});
  const auto layout = makeLayout(*tableHandle, ROW({}, {}));
  const std::vector<PaimonRowKind> rowKinds{
      PaimonRowKind::kInsert,
      PaimonRowKind::kUpdateBefore,
      PaimonRowKind::kUpdateAfter,
      PaimonRowKind::kDelete};
  const auto dataFile = writeFile(
      {"id"},
      makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3, 4})}),
      {1, 2, 3, 4},
      rowKinds);
  auto stream = openStream(0, dataFile, *tableHandle, layout, /*batchSize=*/10);
  for (const auto kind : rowKinds) {
    SCOPED_TRACE(fmt::format("{}", kind));
    ASSERT_TRUE(stream->hasData());
    EXPECT_EQ(stream->rowKind(), kind);
    EXPECT_EQ(
        stream->isRetract(),
        kind == PaimonRowKind::kUpdateBefore || kind == PaimonRowKind::kDelete);
    stream->pop();
  }
  EXPECT_FALSE(stream->hasData());
}

// Rows order by key, then by sequence fields and then by sequence number. Rows
// that tie order by file index.
TEST_F(PaimonMergeStreamTest, compare) {
  {
    const auto rowType = ROW({{"id", BIGINT()}});
    const auto tableHandle = makeTableHandle(rowType, {"id"});
    const auto layout = makeLayout(*tableHandle, ROW({}, {}));
    const auto ids = makeFlatVector<int64_t>({1, 2});
    const auto olderFile =
        writeFile({"id"}, makeRowVector({"id"}, {ids}), {5, 5});
    const auto newerFile =
        writeFile({"id"}, makeRowVector({"id"}, {ids}), {6, 5});
    auto older =
        openStream(0, olderFile, *tableHandle, layout, /*batchSize=*/10);
    auto newer =
        openStream(1, newerFile, *tableHandle, layout, /*batchSize=*/10);

    // Key 1 with sequence numbers 5 and 6.
    EXPECT_LT(older->compare(*newer), 0);
    EXPECT_GT(newer->compare(*older), 0);

    older->pop();
    // Key 2 against key 1.
    EXPECT_GT(older->compare(*newer), 0);
    EXPECT_LT(newer->compare(*older), 0);

    newer->pop();
    // Key 2 with sequence number 5 in both files.
    EXPECT_LT(older->compare(*newer), 0);
    EXPECT_GT(newer->compare(*older), 0);
  }

  // A descending sequence field orders the rows of a key before their
  // sequence numbers do.
  {
    const auto rowType = ROW({{"id", BIGINT()}, {"ts", BIGINT()}});
    const auto tableHandle = makeTableHandle(
        rowType,
        {"id"},
        /*partitionKeys=*/{},
        {{"sequence.field", "ts"},
         {"sequence.field.sort-order", "descending"}});
    const auto layout = makeLayout(*tableHandle, ROW({}, {}));
    const auto olderFile = writeFile(
        {"id"},
        makeRowVector(
            {"id", "ts"},
            {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({10})}),
        {1});
    const auto newerFile = writeFile(
        {"id"},
        makeRowVector(
            {"id", "ts"},
            {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({20})}),
        {2});
    auto older =
        openStream(0, olderFile, *tableHandle, layout, /*batchSize=*/10);
    auto newer =
        openStream(1, newerFile, *tableHandle, layout, /*batchSize=*/10);
    EXPECT_GT(older->compare(*newer), 0);
    EXPECT_LT(newer->compare(*older), 0);
  }
}

} // namespace
} // namespace facebook::velox::connector::paimon
