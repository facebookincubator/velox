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

#include "velox/connectors/hive/iceberg/IcebergMergeSink.h"

#include <folly/json.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::connector::hive::iceberg::test {
namespace {

using IMS = IcebergMergeSink;

// Two target table columns: id BIGINT and name VARCHAR.
// The row_id ROW carries (file_path VARCHAR, pos BIGINT) — the inner two
// fields the DV sub-sink actually consumes. Extra trailing Iceberg row id
// fields (spec_id, partition_data) are intentionally omitted to keep the
// test small; the merge sink tolerates them when present.
const RowTypePtr kRowIdType = ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()});

// Composite-sink input schema as produced by Layer 1's
// IcebergMergeProcessor: target cols, then operation, then row_id, then
// insert_from_update. createDataSink's kMerge case relies on this exact
// layout.
const RowTypePtr kMergeInputType =
    ROW({"id", "name", "operation", "row_id", "insert_from_update"},
        {BIGINT(), VARCHAR(), TINYINT(), kRowIdType, TINYINT()});

class IcebergMergeSinkTest : public IcebergTestBase {
 protected:
  void SetUp() override {
    IcebergTestBase::SetUp();
    // IcebergTestBase only registers the Parquet writer factory when its
    // own translation unit was compiled with VELOX_ENABLE_PARQUET. The
    // composite sink relies on the Parquet writer (via the inner
    // IcebergDataSink), so register the factory unconditionally here.
    // Idempotent: registerWriterFactory tolerates already-registered
    // formats.
    parquet::registerParquetWriterFactory();
  }

  IcebergInsertTableHandlePtr makeMergeHandle(const std::string& outputDir) {
    // Construct the target column handles for (id, name). Field IDs match
    // a fresh Iceberg V1 schema (1, 2).
    std::vector<IcebergColumnHandlePtr> columnHandles;
    columnHandles.emplace_back(
        std::make_shared<const IcebergColumnHandle>(
            "id",
            FileColumnHandle::ColumnType::kRegular,
            BIGINT(),
            parquet::ParquetFieldId{1, {}}));
    columnHandles.emplace_back(
        std::make_shared<const IcebergColumnHandle>(
            "name",
            FileColumnHandle::ColumnType::kRegular,
            VARCHAR(),
            parquet::ParquetFieldId{2, {}}));

    auto locationHandle = std::make_shared<LocationHandle>(
        outputDir, outputDir, LocationHandle::TableType::kNew);

    return std::make_shared<const IcebergInsertTableHandle>(
        /*inputColumns=*/std::move(columnHandles),
        locationHandle,
        /*tableStorageFormat=*/dwio::common::FileFormat::PARQUET,
        /*partitionSpec=*/nullptr,
        /*compressionKind=*/common::CompressionKind::CompressionKind_ZSTD,
        /*serdeParameters=*/std::unordered_map<std::string, std::string>{},
        IcebergInsertTableHandle::WriteKind::kMerge);
  }

  std::unique_ptr<IcebergMergeSink> makeSink(const std::string& outputDir) {
    auto handle = makeMergeHandle(outputDir);
    return std::make_unique<IcebergMergeSink>(
        kMergeInputType,
        handle,
        connectorQueryCtx_.get(),
        CommitStrategy::kNoCommit,
        getHiveConfig(),
        getIcebergConfig(),
        /*targetColumnChannels=*/std::vector<column_index_t>{0, 1},
        /*operationChannel=*/2,
        /*rowIdChannel=*/3);
  }

  // Builds a RowVector matching kMergeInputType from parallel vectors. Use
  // std::nullopt for null cells. operation bytes are passed directly so
  // tests can include invalid bytes to exercise error paths.
  RowVectorPtr makeInput(
      const std::vector<std::optional<int64_t>>& ids,
      const std::vector<std::optional<std::string>>& names,
      const std::vector<int8_t>& operations,
      const std::vector<std::optional<std::string>>& filePaths,
      const std::vector<std::optional<int64_t>>& positions,
      const std::vector<int8_t>& insertFromUpdate) {
    const auto numRows = operations.size();
    auto* pool = opPool_.get();
    velox::test::VectorMaker maker(pool);

    auto idCol = maker.flatVectorNullable<int64_t>(ids);
    auto nameCol = maker.flatVectorNullable<StringView>(toStringViews(names));
    auto opCol = maker.flatVector<int8_t>(operations);

    auto filePathCol =
        maker.flatVectorNullable<StringView>(toStringViews(filePaths));
    auto posCol = maker.flatVectorNullable<int64_t>(positions);
    auto rowIdCol = std::make_shared<RowVector>(
        pool,
        kRowIdType,
        /*nulls=*/nullptr,
        numRows,
        std::vector<VectorPtr>{filePathCol, posCol});

    auto insertFromUpdateCol = maker.flatVector<int8_t>(insertFromUpdate);

    return std::make_shared<RowVector>(
        pool,
        kMergeInputType,
        /*nulls=*/nullptr,
        numRows,
        std::vector<VectorPtr>{
            idCol, nameCol, opCol, rowIdCol, insertFromUpdateCol});
  }

  // Returns the hive/iceberg configs from the base fixture via friend-free
  // accessors built on top of IcebergTestBase's setup. We re-read them
  // from a fresh ConfigBase to match what the base creates internally.
  std::shared_ptr<const HiveConfig> getHiveConfig() {
    return std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>()));
  }

  IcebergConfigPtr getIcebergConfig() {
    return std::make_shared<IcebergConfig>(std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {IcebergConfig::kFunctionPrefixConfig,
             IcebergConfig::kDefaultFunctionPrefix}}));
  }

  // Lookup keys for commit-message JSON entries.
  static constexpr const char* kContentKey = "content";
  static constexpr const char* kFileFormatKey = "fileFormat";

  static std::vector<std::optional<StringView>> toStringViews(
      const std::vector<std::optional<std::string>>& values) {
    std::vector<std::optional<StringView>> result;
    result.reserve(values.size());
    for (const auto& v : values) {
      if (v.has_value()) {
        result.emplace_back(StringView(*v));
      } else {
        result.emplace_back(std::nullopt);
      }
    }
    return result;
  }

  // Returns the count of commit messages whose "content" field equals
  // `content` (e.g. "DATA" for data files, "POSITION_DELETES" for puffin
  // delete files). Tests use this to verify which sub-sink produced each
  // message after `close()`.
  static size_t countContent(
      const std::vector<std::string>& messages,
      const std::string& content) {
    size_t count = 0;
    for (const auto& msg : messages) {
      const auto parsed = folly::parseJson(msg);
      if (parsed.count(kContentKey) > 0 &&
          parsed[kContentKey].asString() == content) {
        ++count;
      }
    }
    return count;
  }
};

TEST_F(IcebergMergeSinkTest, insertOnlyBatchProducesOnlyDataFiles) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  // 3 INSERTs; row_id and insert_from_update are null/0 (ignored on INSERT
  // by the data sub-sink, which only sees the target columns).
  auto input = makeInput(
      /*ids=*/{1, 2, 3},
      /*names=*/{{std::string("a")}, {std::string("b")}, {std::string("c")}},
      /*operations=*/
      {IMS::kInsertOperationNumber,
       IMS::kInsertOperationNumber,
       IMS::kInsertOperationNumber},
      /*filePaths=*/{std::nullopt, std::nullopt, std::nullopt},
      /*positions=*/{std::nullopt, std::nullopt, std::nullopt},
      /*insertFromUpdate=*/{0, 0, 0});

  sink->appendData(input);
  EXPECT_TRUE(sink->finish());
  auto messages = sink->close();

  // Exactly one data file (single partition) and zero puffin files.
  EXPECT_EQ(countContent(messages, "DATA"), 1u);
  EXPECT_EQ(countContent(messages, "POSITION_DELETES"), 0u);
}

TEST_F(IcebergMergeSinkTest, deleteOnlyBatchProducesOnlyPuffinFiles) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  const std::string dataFile = tempDir->getPath() + "/d.parquet";
  // 3 DELETEs against one referenced data file; target cols are null
  // (ignored on DELETE).
  auto input = makeInput(
      /*ids=*/{std::nullopt, std::nullopt, std::nullopt},
      /*names=*/{std::nullopt, std::nullopt, std::nullopt},
      /*operations=*/
      {IMS::kDeleteOperationNumber,
       IMS::kDeleteOperationNumber,
       IMS::kDeleteOperationNumber},
      /*filePaths=*/{{dataFile}, {dataFile}, {dataFile}},
      /*positions=*/{10, 20, 30},
      /*insertFromUpdate=*/{0, 0, 0});

  sink->appendData(input);
  EXPECT_TRUE(sink->finish());
  auto messages = sink->close();

  // Zero data files and exactly one puffin file (one referenced data file).
  EXPECT_EQ(countContent(messages, "DATA"), 0u);
  EXPECT_EQ(countContent(messages, "POSITION_DELETES"), 1u);
}

TEST_F(IcebergMergeSinkTest, mixedBatchProducesBothKinds) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  const std::string dataFile = tempDir->getPath() + "/m.parquet";
  // 2 INSERTs interleaved with 2 DELETEs.
  auto input = makeInput(
      /*ids=*/{1, std::nullopt, 2, std::nullopt},
      /*names=*/
      {{std::string("a")}, std::nullopt, {std::string("b")}, std::nullopt},
      /*operations=*/
      {IMS::kInsertOperationNumber,
       IMS::kDeleteOperationNumber,
       IMS::kInsertOperationNumber,
       IMS::kDeleteOperationNumber},
      /*filePaths=*/
      {std::nullopt, {dataFile}, std::nullopt, {dataFile}},
      /*positions=*/{std::nullopt, 5, std::nullopt, 6},
      /*insertFromUpdate=*/{0, 0, 0, 0});

  sink->appendData(input);
  EXPECT_TRUE(sink->finish());
  auto messages = sink->close();

  EXPECT_EQ(countContent(messages, "DATA"), 1u);
  EXPECT_EQ(countContent(messages, "POSITION_DELETES"), 1u);
}

TEST_F(IcebergMergeSinkTest, emptyBatchProducesNoCommitMessages) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  // No appendData call. finish() must still be safe and close() returns no
  // messages (neither sub-sink saw any data).
  EXPECT_TRUE(sink->finish());
  auto messages = sink->close();

  EXPECT_EQ(countContent(messages, "DATA"), 0u);
  EXPECT_EQ(countContent(messages, "POSITION_DELETES"), 0u);
}

TEST_F(IcebergMergeSinkTest, updateOperationByteIsRejected) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  // UPDATE (3) must have been fanned out by Layer 1 before reaching here.
  // The sink must reject it explicitly so a missing fan-out is caught
  // early rather than producing silently wrong data.
  auto input = makeInput(
      /*ids=*/{1},
      /*names=*/{{std::string("x")}},
      /*operations=*/{static_cast<int8_t>(3)},
      /*filePaths=*/{std::nullopt},
      /*positions=*/{std::nullopt},
      /*insertFromUpdate=*/{0});

  VELOX_ASSERT_USER_THROW(
      sink->appendData(input),
      "IcebergMergeSink only accepts INSERT (1) and DELETE (2)");
}

TEST_F(IcebergMergeSinkTest, defaultCaseByteIsRejected) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  auto input = makeInput(
      /*ids=*/{1},
      /*names=*/{{std::string("x")}},
      /*operations=*/{static_cast<int8_t>(-1)},
      /*filePaths=*/{std::nullopt},
      /*positions=*/{std::nullopt},
      /*insertFromUpdate=*/{0});

  VELOX_ASSERT_USER_THROW(
      sink->appendData(input),
      "IcebergMergeSink only accepts INSERT (1) and DELETE (2)");
}

TEST_F(IcebergMergeSinkTest, abortPropagatesToBothSubsinks) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  // Feed one INSERT + one DELETE, then abort. abort() must propagate to both
  // sub-sinks so they discard pending state and reject any subsequent
  // appendData().
  const std::string dataFile = tempDir->getPath() + "/a.parquet";
  auto input = makeInput(
      /*ids=*/{1, std::nullopt},
      /*names=*/{{std::string("a")}, std::nullopt},
      /*operations=*/
      {IMS::kInsertOperationNumber, IMS::kDeleteOperationNumber},
      /*filePaths=*/{std::nullopt, {dataFile}},
      /*positions=*/{std::nullopt, 1},
      /*insertFromUpdate=*/{0, 0});

  sink->appendData(input);
  sink->abort();

  // After abort, subsequent appendData must throw.
  VELOX_ASSERT_USER_THROW(
      sink->appendData(input), "appendData() called after abort()");
}

TEST_F(IcebergMergeSinkTest, statsAggregateAcrossSubsinks) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  const std::string dataFile = tempDir->getPath() + "/s.parquet";
  auto input = makeInput(
      /*ids=*/{1, std::nullopt},
      /*names=*/{{std::string("a")}, std::nullopt},
      /*operations=*/
      {IMS::kInsertOperationNumber, IMS::kDeleteOperationNumber},
      /*filePaths=*/{std::nullopt, {dataFile}},
      /*positions=*/{std::nullopt, 7},
      /*insertFromUpdate=*/{0, 0});

  sink->appendData(input);
  EXPECT_TRUE(sink->finish());
  sink->close();

  const auto stats = sink->stats();
  // Two files: one data file + one puffin. Bytes > 0 because both writers
  // produced non-trivial output.
  EXPECT_EQ(stats.numWrittenFiles, 2u);
  EXPECT_GT(stats.numWrittenBytes, 0u);
}

TEST_F(IcebergMergeSinkTest, closeIsIdempotent) {
  auto tempDir = TempDirectoryPath::create();
  auto sink = makeSink(tempDir->getPath());

  // Two close() calls should return identical commit-message vectors. The
  // composite's underlying sub-sinks are individually idempotent on close;
  // verify the composite preserves that contract.
  const std::string dataFile = tempDir->getPath() + "/i.parquet";
  auto input = makeInput(
      /*ids=*/{1, std::nullopt},
      /*names=*/{{std::string("a")}, std::nullopt},
      /*operations=*/
      {IMS::kInsertOperationNumber, IMS::kDeleteOperationNumber},
      /*filePaths=*/{std::nullopt, {dataFile}},
      /*positions=*/{std::nullopt, 3},
      /*insertFromUpdate=*/{0, 0});

  sink->appendData(input);
  EXPECT_TRUE(sink->finish());
  // First close().
  auto first = sink->close();
  // Second close() must return an identical commit-message vector without
  // re-closing the sub-sinks.
  auto second = sink->close();
  EXPECT_EQ(second, first);
  EXPECT_EQ(countContent(first, "DATA"), 1u);
  EXPECT_EQ(countContent(first, "POSITION_DELETES"), 1u);
}

// Verifies that the sink derives its inner data sub-sink's column names
// from the IcebergInsertTableHandle's input columns rather than from the
// source RowVector names. The inputType here uses positional column names
// ("c0", "c1") while the handle declares iceberg-schema names ("id",
// "name") — the sink must still construct successfully and write the
// expected data file. This guards against upstream rename drift breaking
// the writer's name-based binding.
TEST_F(IcebergMergeSinkTest, dataInputTypeNamesComeFromHandleNotSource) {
  auto tempDir = TempDirectoryPath::create();
  auto handle = makeMergeHandle(tempDir->getPath());

  // Source RowVector uses positional names instead of iceberg-schema names.
  const RowTypePtr positionalInputType =
      ROW({"c0", "c1", "operation", "row_id", "insert_from_update"},
          {BIGINT(), VARCHAR(), TINYINT(), kRowIdType, TINYINT()});

  auto sink = std::make_unique<IcebergMergeSink>(
      positionalInputType,
      handle,
      connectorQueryCtx_.get(),
      CommitStrategy::kNoCommit,
      getHiveConfig(),
      getIcebergConfig(),
      /*targetColumnChannels=*/std::vector<column_index_t>{0, 1},
      /*operationChannel=*/2,
      /*rowIdChannel=*/3);

  // Build an input vector with the same positional schema; the sink should
  // still successfully write the insert batch by matching by channel
  // rather than by source-page name.
  auto* pool = opPool_.get();
  velox::test::VectorMaker maker(pool);
  auto idCol = maker.flatVector<int64_t>(std::vector<int64_t>{1, 2});
  auto nameCol = maker.flatVector<StringView>(
      std::vector<StringView>{StringView("a"), StringView("b")});
  auto opCol = maker.flatVector<int8_t>(std::vector<int8_t>{
      IMS::kInsertOperationNumber, IMS::kInsertOperationNumber});
  auto filePathCol = maker.flatVectorNullable<StringView>(
      std::vector<std::optional<StringView>>{std::nullopt, std::nullopt});
  auto posCol = maker.flatVectorNullable<int64_t>(
      std::vector<std::optional<int64_t>>{std::nullopt, std::nullopt});
  auto rowIdCol = std::make_shared<RowVector>(
      pool,
      kRowIdType,
      /*nulls=*/nullptr,
      2,
      std::vector<VectorPtr>{filePathCol, posCol});
  auto insertFromUpdateCol =
      maker.flatVector<int8_t>(std::vector<int8_t>{0, 0});
  auto input = std::make_shared<RowVector>(
      pool,
      positionalInputType,
      /*nulls=*/nullptr,
      2,
      std::vector<VectorPtr>{
          idCol, nameCol, opCol, rowIdCol, insertFromUpdateCol});

  sink->appendData(input);
  EXPECT_TRUE(sink->finish());
  auto messages = sink->close();
  EXPECT_EQ(countContent(messages, "DATA"), 1u);
}

// VELOX_ENABLE_PARQUET guard removed: rely on the iceberg_connector
// target's unconditional Parquet writer dependency.

} // namespace
} // namespace facebook::velox::connector::hive::iceberg::test
