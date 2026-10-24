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
#include "velox/connectors/hive/paimon/PaimonConnector.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <atomic>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/process/ThreadDebugInfo.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/orc/reader/OrcReader.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/vector/DecodedVector.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#endif

#ifdef VELOX_ENABLE_NIMBLE
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#endif

namespace facebook::velox::connector::hive::paimon {
namespace {

static const std::string kPaimonConnectorId = "test-paimon";

class TableValidationFileSystem : public tests::utils::FaultyFileSystem {
 public:
  TableValidationFileSystem() : FaultyFileSystem(nullptr) {}

  std::string_view extractPath(std::string_view path) const override {
    return path.substr(std::string_view("paimon-table-test:").size());
  }

  std::unique_ptr<ReadFile> openFileForRead(
      std::string_view path,
      const filesystems::FileOptions& options) override {
    ++openedFiles;
    return FaultyFileSystem::openFileForRead(path, options);
  }

  std::atomic<int32_t> openedFiles{0};
};

class PaimonConnectorTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    Type::registerSerDe();
    common::Filter::registerSerDe();
    core::ITypedExpr::registerSerDe();
    core::PlanNode::registerSerDe();
    PaimonConnector::registerSerDe();
    orc::registerOrcReaderFactory();
    auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    auto connector =
        PaimonConnectorFactory().newConnector(kPaimonConnectorId, config);
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
#ifdef VELOX_ENABLE_NIMBLE
    nimble::registerSelectiveNimbleReaderFactory();
#endif
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
#endif
  }

  ConnectorQueryCtx::Builder queryContextBuilder(core::QueryCtx* queryCtx) {
    ConnectorQueryCtx::Builder builder;
    builder.operatorPool(pool())
        .connectorPool(rootPool_.get())
        .expressionEvaluator(
            std::make_unique<exec::SimpleExpressionEvaluator>(queryCtx, pool()))
        .queryId("PaimonConnectorTest")
        .taskId("PaimonConnectorTest")
        .planNodeId("scan");
    return builder;
  }

  void TearDown() override {
    orc::unregisterOrcReaderFactory();
#ifdef VELOX_ENABLE_NIMBLE
    nimble::unregisterSelectiveNimbleReaderFactory();
#endif
#ifdef VELOX_ENABLE_PARQUET
    parquet::unregisterParquetReaderFactory();
#endif
    ConnectorRegistry::global().erase(kPaimonConnectorId);
    HiveConnectorTestBase::TearDown();
  }

  static std::vector<dwio::common::FileFormat> fileFormats() {
    std::vector<dwio::common::FileFormat> formats{
        dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
    formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
#ifdef VELOX_ENABLE_PARQUET
    formats.push_back(dwio::common::FileFormat::PARQUET);
#endif
    return formats;
  }

  void writeFile(
      const std::string& path,
      const RowVectorPtr& input,
      dwio::common::FileFormat format) {
    if (format == dwio::common::FileFormat::DWRF) {
      writeToFile(path, input);
      return;
    }
#ifdef VELOX_ENABLE_NIMBLE
    if (format == dwio::common::FileFormat::NIMBLE) {
      nimble::Writer writer(
          asRowType(input->type()),
          std::make_unique<LocalWriteFile>(path, false, false),
          *pool(),
          nimble::WriterOptions{});
      writer.write(input);
      writer.close();
      return;
    }
#endif
#ifdef VELOX_ENABLE_PARQUET
    if (format == dwio::common::FileFormat::PARQUET) {
      dwio::common::WriterOptions options;
      options.memoryPool = rootPool_.get();
      parquet::Writer writer(
          std::make_unique<dwio::common::WriteFileSink>(
              std::make_unique<LocalWriteFile>(path, false, false), path),
          options,
          asRowType(input->type()));
      writer.write(input);
      writer.close();
      return;
    }
#endif
    VELOX_FAIL("Unsupported test file format");
  }

  static const std::shared_ptr<TableValidationFileSystem>&
  validationFileSystem() {
    static const auto kFileSystem =
        std::make_shared<TableValidationFileSystem>();
    static const bool kRegistered = [] {
      filesystems::registerFileSystem(
          [](std::string_view path) {
            return path.starts_with("paimon-table-test:");
          },
          [](std::shared_ptr<const config::ConfigBase>, std::string_view) {
            return kFileSystem;
          });
      return true;
    }();
    VELOX_CHECK(kRegistered);
    return kFileSystem;
  }

  void testTakeoverRuntimeStats(int fileCount, bool existingStats) {
    auto queryCtx = core::QueryCtx::create();
    auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    // Keep the context and all other borrowed inputs alive to isolate the
    // ownership of runtimeStats during DataSource takeover.
    auto context = ConnectorQueryCtx::Builder()
                       .operatorPool(pool())
                       .connectorPool(rootPool_.get())
                       .sessionProperties(config.get())
                       .expressionEvaluator(
                           std::make_unique<exec::SimpleExpressionEvaluator>(
                               queryCtx.get(), pool()))
                       .queryId("PaimonConnectorTest")
                       .taskId("PaimonConnectorTest")
                       .planNodeId("scan")
                       .build();
    const auto type = ROW({{"c0", BIGINT()}});
    auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
    auto table = makePaimonTableHandle(type);
    auto columns = makePaimonColumnHandles(type);
    auto makeSource = [&] {
      return connector->createDataSource(type, table, columns, context.get());
    };
    auto processedFiles = [](DataSource& source) {
      auto stats = source.getRuntimeStats();
      auto it = stats.find("processedSplits");
      return it == stats.end() ? 0 : it->second.sum;
    };
    for (auto format : fileFormats()) {
      SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
      const int firstFile = existingStats ? 1 : 0;
      auto files = makeFilePaths(firstFile + fileCount);
      std::vector<std::string> paths;
      std::vector<RowVectorPtr> inputs;
      for (int i = 0; i < files.size(); ++i) {
        paths.push_back(files[i]->getPath());
        inputs.push_back(
            makeRowVector({makeFlatVector<int64_t>({10 * i + 1, 10 * i + 2})}));
        writeFile(paths.back(), inputs.back(), format);
      }
      auto source = makeSource();
      auto readFiles = [&](int begin, int count) {
        ContinueFuture future = ContinueFuture::makeEmpty();
        for (int i = begin; i < begin + count; ++i) {
          auto result = source->next(100, future);
          ASSERT_TRUE(result.has_value());
          ASSERT_FALSE(future.valid());
          ASSERT_TRUE(*result);
          test::assertEqualVectors(inputs[i], *result);
        }
        auto result = source->next(100, future);
        ASSERT_TRUE(result.has_value());
        ASSERT_FALSE(future.valid());
        EXPECT_EQ(*result, nullptr);
      };
      if (existingStats) {
        source->addSplit(makePaimonSplit(
            {paths.front()}, PaimonTableType::kAppendOnly, format));
        readFiles(0, 1);
      }
      ASSERT_EQ(processedFiles(*source), firstFile);
      auto preloaded = makeSource();
      preloaded->addSplit(makePaimonSplit(
          std::vector<std::string>(paths.begin() + firstFile, paths.end()),
          PaimonTableType::kAppendOnly,
          format));
      // Paimon opens even the first file lazily. No file preparation has
      // occurred when the donor is destroyed by setFromDataSource().
      ASSERT_EQ(processedFiles(*preloaded), 0);
      source->setFromDataSource(std::move(preloaded));
      ASSERT_EQ(preloaded, nullptr);
      EXPECT_EQ(processedFiles(*source), firstFile);
      readFiles(firstFile, fileCount);
      EXPECT_EQ(processedFiles(*source), firstFile + fileCount);
      EXPECT_EQ(source->getCompletedRows(), 2 * (firstFile + fileCount));
    }
  }

  static std::vector<int32_t> fieldIds(const RowTypePtr& rowType) {
    std::vector<int32_t> ids;
    for (column_index_t i = 0; i < rowType->size(); ++i) {
      ids.push_back(i + 10);
    }
    return ids;
  }

  static std::shared_ptr<PaimonTableHandle> makePaimonTableHandle(
      const RowTypePtr& rowType,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      const std::vector<std::string>& partitionNames = {},
      std::vector<FileColumnHandlePtr> filterColumns = {},
      std::unordered_map<std::string, std::string> options = {}) {
    return std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        17,
        rowType,
        fieldIds(rowType),
        PaimonTableType::kAppendOnly,
        partitionNames,
        std::move(options),
        std::move(subfieldFilters),
        remainingFilter,
        std::move(filterColumns));
  }

  /// Creates column assignments for all columns as regular columns.
  static connector::ColumnHandleMap makePaimonColumnHandles(
      const RowTypePtr& rowType) {
    connector::ColumnHandleMap assignments;
    assignments.reserve(rowType->size());
    for (uint32_t i = 0; i < rowType->size(); ++i) {
      const auto& name = rowType->nameOf(i);
      assignments[name] = std::make_shared<HiveColumnHandle>(
          name,
          HiveColumnHandle::ColumnType::kRegular,
          rowType->childAt(i),
          rowType->childAt(i));
    }
    return assignments;
  }

  /// Builds a table scan plan using the Paimon connector.
  core::PlanNodePtr makePaimonScanPlan(
      const RowTypePtr& outputType,
      const RowTypePtr& dataColumns = nullptr,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr) {
    const auto rowType = dataColumns ? dataColumns : outputType;
    const auto tableHandle = makePaimonTableHandle(
        rowType, std::move(subfieldFilters), remainingFilter);
    auto assignments =
        makePaimonColumnHandles(dataColumns ? dataColumns : outputType);
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(tableHandle)
        .assignments(assignments)
        .endTableScan()
        .planNode();
  }

  /// Creates a PaimonConnectorSplit from file paths.
  std::shared_ptr<PaimonConnectorSplit> makePaimonSplit(
      const std::vector<std::string>& filePaths,
      PaimonTableType tableType = PaimonTableType::kAppendOnly,
      dwio::common::FileFormat format = dwio::common::FileFormat::DWRF,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      bool rawConvertible = true) {
    PaimonConnectorSplitBuilder builder(
        kPaimonConnectorId, /*snapshotId=*/1, tableType, format);
    for (const auto& filePath : filePaths) {
      builder.addFile(filePath, /*fileSize=*/0);
    }
    for (const auto& [key, value] : partitionKeys) {
      builder.partitionKey(key, value);
    }
    builder.rawConvertible(rawConvertible);
    return builder.build();
  }
};

TEST_F(PaimonConnectorTest, connectorRegistration) {
  auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  ASSERT_NE(connector, nullptr);
  ASSERT_NE(connector->connectorConfig(), nullptr);
  EXPECT_FALSE(connector->supportsSplitPreload());
  EXPECT_FALSE(connector->supportsIndexLookup());
  VELOX_ASSERT_THROW(
      connector->createDataSink(
          ROW({}), nullptr, nullptr, CommitStrategy::kNoCommit),
      "Paimon connector does not support writes");
  VELOX_ASSERT_THROW(
      connector->createIndexSource(ROW({}), {}, ROW({}), nullptr, {}, nullptr),
      "Paimon connector does not support index lookup");
}

TEST_F(PaimonConnectorTest, connectorFactory) {
  PaimonConnectorFactory factory;
  EXPECT_EQ(
      std::string(PaimonConnectorFactory::kPaimonConnectorName), "paimon");

  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {HiveConfig::kEnableFileHandleCache, "true"},
          {HiveConfig::kNumCacheFileHandles, "500"}});

  auto connector = factory.newConnector("test-paimon-2", config);
  ASSERT_NE(connector, nullptr);

  HiveConfig hiveConfig(connector->connectorConfig());
  EXPECT_TRUE(hiveConfig.isFileHandleCacheEnabled());
  EXPECT_EQ(hiveConfig.numCacheFileHandles(), 500);
}

// E2E test: read a single file from an append-only Paimon split.
TEST_F(PaimonConnectorTest, takeoverRuntimeStatsBeforeFirstFile) {
  testTakeoverRuntimeStats(1, false);
}

TEST_F(PaimonConnectorTest, takeoverRuntimeStatsAcrossFiles) {
  testTakeoverRuntimeStats(2, false);
}

TEST_F(PaimonConnectorTest, takeoverRuntimeStatsAccumulate) {
  testTakeoverRuntimeStats(2, true);
}

TEST_F(PaimonConnectorTest, appendOnlySingleFile) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto vectors = makeVectors(rowType, 1, 100);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit({filePaths[0]->getPath()});
  auto plan = makePaimonScanPlan(rowType);

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(vectors);
}

TEST_F(PaimonConnectorTest, ordinaryOrcTableScan) {
  const auto rowType = ROW({{"int1", INTEGER()}, {"string1", VARCHAR()}});
  const auto plan = exec::test::PlanBuilder(
                        makePaimonScanPlan(rowType),
                        std::make_shared<core::PlanNodeIdGenerator>(1))
                        .limit(0, 3, false)
                        .planNode();
  const auto path = velox::test::getDataFilePath(
      "velox/dwio/orc/test", "examples/TestOrcFile.columnProjection.orc");
  const auto expected = makeRowVector(
      {"int1", "string1"},
      {makeFlatVector<int32_t>({-1155869325, 431529176, 1761283695}),
       makeFlatVector<std::string>(
           {"bb2c72394b1ab9f8", "e6c5459001105f17", "7f2fc3e9dacc5ecd"})});
  exec::test::AssertQueryBuilder(plan)
      .split(makePaimonSplit(
          {path}, PaimonTableType::kAppendOnly, dwio::common::FileFormat::ORC))
      .assertResults(expected);
}

// E2E test: read multiple files from a single append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlyMultipleFiles) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto vectors1 = makeVectors(rowType, 1, 100);
  auto vectors2 = makeVectors(rowType, 1, 50);

  auto filePaths = makeFilePaths(2);
  writeToFile(filePaths[0]->getPath(), vectors1);
  writeToFile(filePaths[1]->getPath(), vectors2);

  auto split =
      makePaimonSplit({filePaths[0]->getPath(), filePaths[1]->getPath()});
  auto plan = makePaimonScanPlan(rowType);

  // Expected result is all rows from both files.
  std::vector<RowVectorPtr> expected;
  expected.insert(expected.end(), vectors1.begin(), vectors1.end());
  expected.insert(expected.end(), vectors2.begin(), vectors2.end());

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(expected);
}

TEST_F(PaimonConnectorTest, tableSchemaTypesAndBatchOwnership) {
  const auto input = makeRowVector(
      {"flag", "number", "id", "score", "text"},
      {makeNullableFlatVector<bool>({true, false, std::nullopt, true}),
       makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({11, 12, 13, 14}),
       makeNullableFlatVector<double>({1.5, 2.5, std::nullopt, 4.5}),
       makeNullableFlatVector<std::string>(
           {"alpha", "", std::nullopt, "omega"})});
  const auto rowType = asRowType(input->type());
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto table = makePaimonTableHandle(rowType);
  const auto plan = makePaimonScanPlan(rowType);
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
    }
    auto split = makePaimonSplit(
        {files[0]->getPath(), files[1]->getPath()},
        PaimonTableType::kAppendOnly,
        format);
    split = PaimonConnectorSplit::create(split->serialize());
    exec::test::AssertQueryBuilder(copy).split(split).assertResults(
        {input, input});
    const auto countPlan = exec::test::PlanBuilder()
                               .startTableScan()
                               .connectorId(kPaimonConnectorId)
                               .outputType(ROW({}))
                               .tableHandle(table)
                               .assignments({})
                               .endTableScan()
                               .singleAggregation({}, {"count(1)"})
                               .planNode();
    exec::test::AssertQueryBuilder(countPlan).split(split).assertSingleResult(
        int64_t{8});

    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(split);
    ContinueFuture future = ContinueFuture::makeEmpty();
    std::vector<RowVectorPtr> heldOutputs;
    std::vector<RowVectorPtr> expectedOutputs;
    while (true) {
      auto result = source->next(1, future);
      ASSERT_TRUE(result.has_value());
      ASSERT_FALSE(future.valid());
      if (!*result) {
        break;
      }
      for (auto& child : (*result)->children()) {
        child->loadedVector();
      }
      auto expected = BaseVector::create(rowType, (*result)->size(), pool());
      expected->copy(result->get(), 0, 0, (*result)->size());
      expectedOutputs.push_back(std::static_pointer_cast<RowVector>(expected));
      heldOutputs.push_back(std::move(*result));
    }
    EXPECT_EQ(source->getCompletedRows(), 8);
    const auto stats = source->getRuntimeStats();
    source.reset();
    for (size_t i = 0; i < heldOutputs.size(); ++i) {
      test::assertEqualVectors(expectedOutputs[i], heldOutputs[i]);
    }
    EXPECT_EQ(stats.at("processedSplits").sum, 2);
  }
}

// Verify that empty splits (no data files) are rejected by
// PaimonConnectorSplit.
TEST_F(PaimonConnectorTest, rejectsEmptySplit) {
  VELOX_ASSERT_THROW(
      makePaimonSplit({}), "PaimonConnectorSplit requires non-empty dataFiles");
}

// E2E test: read with partition keys from an append-only Paimon split.
TEST_F(PaimonConnectorTest, appendOnlyWithPartitionKeys) {
  // Data file columns (not including partition column).
  auto dataRowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(dataRowType, 1, 100);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  // The output includes both the data column and the partition column.
  auto outputType = ROW({"c0", "p0"}, {BIGINT(), VARCHAR()});
  auto tableHandle = makePaimonTableHandle(
      outputType, /*subfieldFilters=*/{}, /*remainingFilter=*/nullptr, {"p0"});

  connector::ColumnHandleMap assignments;
  assignments["c0"] = std::make_shared<HiveColumnHandle>(
      "c0", HiveColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
  assignments["p0"] = std::make_shared<HiveColumnHandle>(
      "p0", HiveColumnHandle::ColumnType::kPartitionKey, VARCHAR(), VARCHAR());

  auto plan = exec::test::PlanBuilder()
                  .startTableScan()
                  .connectorId(kPaimonConnectorId)
                  .outputType(outputType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      {{"p0", std::optional<std::string>("2024-01-01")}});

  // Build expected output with the partition value filled in.
  auto expectedC0 = vectors[0]->childAt(0);
  auto numRows = vectors[0]->size();
  auto expectedP0 =
      BaseVector::createConstant(VARCHAR(), "2024-01-01", numRows, pool());

  auto expected = makeRowVector({"c0", "p0"}, {expectedC0, expectedP0});

  exec::test::AssertQueryBuilder(plan).split(split).assertResults({expected});
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  exec::test::AssertQueryBuilder(copy).split(split).assertResults({expected});
}

TEST_F(PaimonConnectorTest, physicalPartitionColumnsUseSplitConstants) {
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto table = makePaimonTableHandle(rowType, {}, nullptr, {"p"});
  const ColumnHandleMap assignments{
      {"id",
       std::make_shared<HiveColumnHandle>(
           "id", FileColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT())},
      {"p",
       std::make_shared<HiveColumnHandle>(
           "p",
           FileColumnHandle::ColumnType::kPartitionKey,
           VARCHAR(),
           VARCHAR())}};
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(table)
                        .assignments(assignments)
                        .endTableScan()
                        .planNode();
  const auto input = makeRowVector(
      {"id", "p"},
      {makeFlatVector<int64_t>({1, 2}),
       makeNullableFlatVector<std::string>({"physical-copy", std::nullopt})});
  const auto expected = makeRowVector(
      {"id", "p"},
      {input->childAt(0),
       makeFlatVector<std::string>({"split-value", "split-value"})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    exec::test::AssertQueryBuilder(plan)
        .split(makePaimonSplit(
            {files[0]->getPath()},
            PaimonTableType::kAppendOnly,
            format,
            {{"p", "split-value"}}))
        .assertResults(expected);
  }
}

TEST_F(PaimonConnectorTest, tableHandleFiltersAndProjection) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<int64_t>({0, 11, 2, 13, 15})});
  const auto remainingFilter = parseExpr("value % 2 = 1", rowType);
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(2, 4, false));
  auto tableHandle = makePaimonTableHandle(
      rowType,
      std::move(filters),
      remainingFilter,
      {},
      std::vector<FileColumnHandlePtr>{std::make_shared<HiveColumnHandle>(
          "value",
          FileColumnHandle::ColumnType::kRegular,
          BIGINT(),
          BIGINT())});
  const auto outputType = ROW({{"selected_id", BIGINT()}});
  connector::ColumnHandleMap assignments{
      {"selected_id",
       std::make_shared<HiveColumnHandle>(
           "id", FileColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT())}};
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(outputType)
                        .tableHandle(tableHandle)
                        .assignments(assignments)
                        .endTableScan()
                        .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto expected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({2, 4})});
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto paths = makeFilePaths(1);
    writeFile(paths[0]->getPath(), input, format);
    const auto split = makePaimonSplit(
        {paths[0]->getPath()}, PaimonTableType::kAppendOnly, format);
    exec::test::AssertQueryBuilder(plan).split(split).assertResults(expected);
    exec::test::AssertQueryBuilder(copy).split(split).assertResults(expected);
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          outputType, tableHandle, assignments, context.get());
    source->addSplit(split);
    // TableScan supplies cumulative static and dynamic conditions.
    const common::BigintRange dynamicFilter(4, 5, false);
    source->addDynamicFilter(
        0,
        source->getFilters()
            ->at(common::Subfield("id"))
            ->mergeWith(&dynamicFilter));
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source->next(100, future);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(*result);
    test::assertEqualVectors(
        makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({4})}),
        *result);
  }
}

TEST_F(PaimonConnectorTest, fileScanCapabilitiesWithPaimonTable) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({0, 1, 2, 3}),
       makeFlatVector<int64_t>({11, 12, 13, 14})});
  std::unordered_map<std::string, std::string> options;
  options["write-buffer-size"] = "256 mb";
  options["scan.mode"] = "latest";
  options["bucket"] = "8";
  options["file.format"] = "avro";
  // A native table option is not a File reader's skip-header instruction.
  options["skip.header.line.count"] = "1";
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"), std::make_unique<common::BoolValue>(true, false));
  const auto filterColumn = std::make_shared<HiveColumnHandle>(
      "value", FileColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
  const auto table = makePaimonTableHandle(
      rowType,
      std::move(filters),
      parseExpr("value % 2 = 1", rowType),
      {},
      {filterColumn, filterColumn},
      std::move(options));
  const auto outputType = ROW({{"selected_id", BIGINT()}});
  const auto makePlan = [&](bool postProcess) {
    std::function<void(VectorPtr&)> processor;
    if (postProcess) {
      processor = [&](VectorPtr& column) {
        const DecodedVector decoded(*column);
        column =
            makeFlatVector<int64_t>(column->size(), [&](vector_size_t row) {
              return decoded.valueAt<int64_t>(row) + 100;
            });
      };
    }
    std::vector<common::Subfield> requiredSubfields;
    requiredSubfields.emplace_back("id");
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(table)
        .assignments(
            {{"selected_id",
              std::make_shared<HiveColumnHandle>(
                  "id",
                  FileColumnHandle::ColumnType::kRegular,
                  BIGINT(),
                  BIGINT(),
                  std::move(requiredSubfields),
                  std::vector<NamedExtraction>{},
                  HiveColumnHandle::ColumnParseParameters{},
                  std::move(processor))}})
        .endTableScan()
        .planNode();
  };
  const auto plan = makePlan(false);
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto processedPlan = makePlan(true);
  const auto expected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({2, 2})});
  const auto processedExpected =
      makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({102, 102})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    std::vector<std::string> paths;
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
      paths.push_back(file->getPath());
    }
    const auto split =
        makePaimonSplit(paths, PaimonTableType::kAppendOnly, format);
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan).split(split).assertResults(expected);
    }
    exec::test::AssertQueryBuilder(processedPlan)
        .maxDrivers(1)
        .split(split)
        .assertResults(processedExpected);
  }

  // This local scan keeps the literal name in the Subfield path elements.
  const auto literalType = ROW({{"a.b", BIGINT()}});
  const auto literalInput =
      makeRowVector({"a.b"}, {makeFlatVector<int64_t>({1, 2})});
  std::vector<common::Subfield> requiredSubfields;
  requiredSubfields.push_back(std::move(*common::Subfield::create("a.b")));
  const auto literalPlan = exec::test::PlanBuilder()
                               .startTableScan()
                               .connectorId(kPaimonConnectorId)
                               .outputType(literalType)
                               .tableHandle(makePaimonTableHandle(literalType))
                               .assignments(
                                   {{"a.b",
                                     std::make_shared<HiveColumnHandle>(
                                         "a.b",
                                         FileColumnHandle::ColumnType::kRegular,
                                         BIGINT(),
                                         BIGINT(),
                                         std::move(requiredSubfields))}})
                               .endTableScan()
                               .planNode();
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), literalInput, format);
    exec::test::AssertQueryBuilder(literalPlan)
        .split(makePaimonSplit(
            {files[0]->getPath()}, PaimonTableType::kAppendOnly, format))
        .assertResults(literalInput);
  }
}

TEST_F(PaimonConnectorTest, partitionFilterOnlyConstants) {
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({7, 8})});
  const auto partitionColumn = std::make_shared<HiveColumnHandle>(
      "p", FileColumnHandle::ColumnType::kPartitionKey, VARCHAR(), VARCHAR());
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    for (const std::optional<std::string>& partition :
         {std::optional<std::string>{}, std::optional<std::string>{""}}) {
      const auto table = makePaimonTableHandle(
          rowType,
          common::SubfieldFilters{},
          parseExpr("p is null or length(p) = 0", rowType),
          {"p"},
          std::vector<FileColumnHandlePtr>{partitionColumn});
      const auto plan =
          exec::test::PlanBuilder()
              .startTableScan()
              .connectorId(kPaimonConnectorId)
              .outputType(ROW({{"id", BIGINT()}}))
              .tableHandle(table)
              .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
              .endTableScan()
              .planNode();
      const auto split = makePaimonSplit(
          {files[0]->getPath()},
          PaimonTableType::kAppendOnly,
          format,
          {{"p", partition}});
      exec::test::AssertQueryBuilder(plan).split(split).assertResults(input);
      const auto copy =
          ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
      exec::test::AssertQueryBuilder(copy).split(split).assertResults(input);
    }
    common::SubfieldFilters filters;
    filters.emplace(common::Subfield("p"), std::make_unique<common::IsNull>());
    const auto table = makePaimonTableHandle(
        rowType,
        std::move(filters),
        nullptr,
        {"p"},
        std::vector<FileColumnHandlePtr>{partitionColumn});
    const auto plan =
        exec::test::PlanBuilder()
            .startTableScan()
            .connectorId(kPaimonConnectorId)
            .outputType(ROW({{"id", BIGINT()}}))
            .tableHandle(table)
            .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
            .endTableScan()
            .planNode();
    const auto nullSplit = makePaimonSplit(
        {files[0]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", std::nullopt}});
    const auto emptyStringSplit = makePaimonSplit(
        {files[0]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", ""}});
    exec::test::AssertQueryBuilder(plan).split(nullSplit).assertResults(input);
    exec::test::AssertQueryBuilder(plan)
        .split(emptyStringSplit)
        .assertEmptyResults();
  }
}

TEST_F(PaimonConnectorTest, fileExpressionPipelineWithLambda) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto input = makeRowVector(
      {"id", "value"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30})});
  // The lambda captures 'value', which is needed by the filter but is not
  // included in the scan output.
  const auto table = makePaimonTableHandle(
      rowType, {}, parseExpr("any_match(array[20], x -> x = value)", rowType));
  const auto plan =
      exec::test::PlanBuilder()
          .startTableScan()
          .connectorId(kPaimonConnectorId)
          .outputType(ROW({{"id", BIGINT()}}))
          .tableHandle(table)
          .assignments(makePaimonColumnHandles(ROW({{"id", BIGINT()}})))
          .endTableScan()
          .planNode();
  const auto copy =
      ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
  const auto expected =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({2, 2})});
  for (auto format : fileFormats()) {
    const auto files = makeFilePaths(2);
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
    }
    const auto split = makePaimonSplit(
        {files[0]->getPath(), files[1]->getPath()},
        PaimonTableType::kAppendOnly,
        format);
    for (const auto& scan : {plan, copy}) {
      exec::test::AssertQueryBuilder(scan).split(split).assertResults(expected);
    }
  }
}

TEST_F(PaimonConnectorTest, dynamicFilterAndEmptyFiles) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"a", BIGINT()}, {"b", DOUBLE()}});
  const auto table = makePaimonTableHandle(rowType);
  const auto input = makeRowVector(
      {"a", "b"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<double>({1.5, 2.5, 3.5})});
  const auto empty = makeRowVector(
      {"a", "b"}, {makeFlatVector<int64_t>({}), makeFlatVector<double>({})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(4);
    auto firstEmpty = files[0]->getPath();
    auto nextEmpty = files[2]->getPath();
#ifdef VELOX_ENABLE_PARQUET
    if (format == dwio::common::FileFormat::PARQUET) {
      firstEmpty = velox::test::getDataFilePath(
          "velox/dwio/parquet/tests", "examples/empty.parquet");
      nextEmpty = firstEmpty;
    } else
#endif
    {
      writeFile(firstEmpty, empty, format);
      writeFile(nextEmpty, empty, format);
    }
    writeFile(files[1]->getPath(), input, format);
    writeFile(files[3]->getPath(), input, format);
    auto source = ConnectorRegistry::tryGet(kPaimonConnectorId)
                      ->createDataSource(
                          rowType,
                          table,
                          makePaimonColumnHandles(rowType),
                          context.get());
    source->addSplit(makePaimonSplit(
        {firstEmpty, files[1]->getPath(), nextEmpty, files[3]->getPath()},
        PaimonTableType::kAppendOnly,
        format));
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto first = source->next(100, future);
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(*first);
    test::assertEqualVectors(input, *first);
    source->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(2, 2, false));
    auto second = source->next(100, future);
    ASSERT_TRUE(second.has_value());
    ASSERT_TRUE(*second);
    test::assertEqualVectors(
        makeRowVector(
            {"a", "b"},
            {makeFlatVector<int64_t>({2}), makeFlatVector<double>({2.5})}),
        *second);
    auto end = source->next(100, future);
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(*end, nullptr);
    EXPECT_EQ(source->getCompletedRows(), 6);
    const auto stats = source->getRuntimeStats();
    EXPECT_EQ(stats.at("processedSplits").sum, 2);
    EXPECT_EQ(
        source->getRuntimeStats().at("processedSplits").sum,
        stats.at("processedSplits").sum);
  }
}

TEST_F(PaimonConnectorTest, tableValidationBeforeFileOpen) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;

  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto tableHandle =
      makePaimonTableHandle(rowType, {}, nullptr, /*partitionNames=*/{"p"});
  connector::ColumnHandleMap assignments{
      {"id",
       std::make_shared<HiveColumnHandle>(
           "id", FileColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT())},
      {"p",
       std::make_shared<HiveColumnHandle>(
           "p",
           FileColumnHandle::ColumnType::kPartitionKey,
           VARCHAR(),
           VARCHAR())}};
  auto source =
      ConnectorRegistry::tryGet(kPaimonConnectorId)
          ->createDataSource(
              ROW({{"id", BIGINT()}}), tableHandle, assignments, context.get());
  const auto files = makeFilePaths(1);
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2})});
  writeToFile(files[0]->getPath(), input);
  const std::string path = "paimon-table-test:" + files[0]->getPath();
  const auto makeSplit = [&](PaimonTableType type, const auto& partitions) {
    return makePaimonSplit(
        {path}, type, dwio::common::FileFormat::DWRF, partitions);
  };
  const std::unordered_map<std::string, std::optional<std::string>> good{
      {"p", std::nullopt}};
  VELOX_ASSERT_THROW(
      source->addSplit(makeSplit(PaimonTableType::kPrimaryKey, good)),
      "Paimon split table type does not match table handle");
  EXPECT_EQ(fileSystem->openedFiles, 0);
  for (const auto& change : {"dv", "changelog"}) {
    SCOPED_TRACE(change);
    auto wire = makeSplit(PaimonTableType::kAppendOnly, good)->serialize();
    if (std::string_view(change) == "dv") {
      wire["dataFiles"][0]["deletionFile"] =
          PaimonDeletionFile{"/dv", 0, 1, 1}.serialize();
    } else {
      wire["dataFiles"][0]["fileType"] = "CHANGELOG";
    }
    VELOX_ASSERT_USER_THROW(
        source->addSplit(PaimonConnectorSplit::create(wire)), "Paimon reader");
    EXPECT_EQ(fileSystem->openedFiles, 0);
  }

  VELOX_ASSERT_USER_THROW(
      source->addSplit(makePaimonSplit(
          {path},
          PaimonTableType::kAppendOnly,
          dwio::common::FileFormat::DWRF,
          good,
          false)),
      "Paimon same-schema append reader requires rawConvertible=true");
  EXPECT_EQ(fileSystem->openedFiles, 0);

  auto dataSplit = makeSplit(PaimonTableType::kAppendOnly, good)->serialize();
  dataSplit["dataFiles"][0]["level"] = 1;
  source->addSplit(PaimonConnectorSplit::create(dataSplit));
  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(*result);
  test::assertEqualVectors(input, *result);
  EXPECT_GT(fileSystem->openedFiles, 0);
  auto end = source->next(100, future);
  ASSERT_TRUE(end.has_value());
  EXPECT_EQ(*end, nullptr);

  // Partition strings are converted when the scan binds a constant.
  const auto numericTable = makePaimonTableHandle(
      ROW({{"id", BIGINT()}, {"p", INTEGER()}}), {}, nullptr, {"p"});
  const auto numericColumn = std::make_shared<HiveColumnHandle>(
      "p", FileColumnHandle::ColumnType::kPartitionKey, INTEGER(), INTEGER());
  auto numericSource = ConnectorRegistry::tryGet(kPaimonConnectorId)
                           ->createDataSource(
                               ROW({{"p", INTEGER()}}),
                               numericTable,
                               {{"p", numericColumn}},
                               context.get());
  const auto numericFiles = makeFilePaths(1);
  writeToFile(numericFiles[0]->getPath(), input);
  const std::string numericPath =
      "paimon-table-test:" + numericFiles[0]->getPath();
  const auto openedFiles = fileSystem->openedFiles.load();
  numericSource->addSplit(makePaimonSplit(
      {numericPath},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      std::unordered_map<std::string, std::optional<std::string>>{
          {"p", "not-an-integer"}}));
  EXPECT_EQ(fileSystem->openedFiles, openedFiles);
  EXPECT_THROW(numericSource->next(100, future), std::exception);
  EXPECT_GT(fileSystem->openedFiles, openedFiles);
}

TEST_F(PaimonConnectorTest, queryValidationBeforeFileOpen) {
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  const auto table = makePaimonTableHandle(rowType, {}, nullptr, {"p"});
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  const auto idColumns = makePaimonColumnHandles(ROW({{"id", BIGINT()}}));
  VELOX_ASSERT_THROW(
      connector->createDataSource(
          ROW({{"missing", BIGINT()}}), table, idColumns, context.get()),
      "ColumnHandle is missing for output column: missing");
  VELOX_ASSERT_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}),
          table,
          {{"id", idColumns.at("id")}, {"alias", idColumns.at("id")}},
          context.get()),
      "Cannot map from same table column to different outputs in table scan; a project node should be used instead: id");
  const auto requiredColumn = [&](const common::Subfield& path) {
    std::vector<common::Subfield> subfields;
    subfields.push_back(path.clone());
    return std::make_shared<HiveColumnHandle>(
        "id",
        FileColumnHandle::ColumnType::kRegular,
        BIGINT(),
        BIGINT(),
        std::move(subfields));
  };
  for (const auto& path : {"missing", "p"}) {
    SCOPED_TRACE(path);
    VELOX_ASSERT_USER_THROW(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}),
            table,
            {{"id", requiredColumn(common::Subfield(path))}},
            context.get()),
        "Required subfield does not match column name");
  }
  EXPECT_NE(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}),
          table,
          {{"id", requiredColumn(common::Subfield("id"))}},
          context.get()),
      nullptr);
  {
    auto wire = table->serialize();
    wire["tableType"] = "PRIMARY_KEY";
    const auto unsupported =
        ISerializable::deserialize<PaimonTableHandle>(wire);
    VELOX_ASSERT_USER_THROW(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}), unsupported, idColumns, context.get()),
        "Paimon same-schema append reader");
  }
  const auto widerType = ROW({{"id", BIGINT()}, {"unused", VARBINARY()}});
  const auto widerTable = makePaimonTableHandle(widerType);
  EXPECT_NE(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}), widerTable, idColumns, context.get()),
      nullptr);
  const auto binaryType = ROW({{"unused", VARBINARY()}});
  EXPECT_NE(
      connector->createDataSource(
          binaryType,
          widerTable,
          makePaimonColumnHandles(binaryType),
          context.get()),
      nullptr);
  const auto foreignTable = std::make_shared<PaimonTableHandle>(
      "other-connector",
      table->name(),
      table->schemaId(),
      table->rowType(),
      table->fieldIds(),
      table->tableType(),
      table->partitionKeys(),
      table->options());
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}), foreignTable, idColumns, context.get()),
      "Paimon table handle connector does not match connector instance");
  for (const auto& condition : {"p is null", "length(p) = 0"}) {
    SCOPED_TRACE(condition);
    const auto filteredTable = makePaimonTableHandle(
        rowType, {}, parseExpr(condition, rowType), {"p"});
    const ColumnHandleMap columns{
        {"id", idColumns.at("id")},
        {"p",
         std::make_shared<HiveColumnHandle>(
             "p",
             FileColumnHandle::ColumnType::kPartitionKey,
             VARCHAR(),
             VARCHAR())}};
    EXPECT_NE(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}), filteredTable, columns, context.get()),
        nullptr);
    const auto filterColumnTable = makePaimonTableHandle(
        rowType,
        common::SubfieldFilters{},
        parseExpr(condition, rowType),
        {"p"},
        std::vector<FileColumnHandlePtr>{
            std::static_pointer_cast<const FileColumnHandle>(columns.at("p"))});
    EXPECT_NE(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}),
            filterColumnTable,
            idColumns,
            context.get()),
        nullptr);
  }
  for (const auto* option :
       {FileConfig::kIgnoreMissingFilesSession,
        FileConfig::kFileColumnNamesReadAsLowerCaseSession}) {
    SCOPED_TRACE(option);
    const auto session = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{{option, "true"}});
    auto optionContext = queryContextBuilder(queryCtx.get())
                             .sessionProperties(session.get())
                             .build();
    EXPECT_NE(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}), table, idColumns, optionContext.get()),
        nullptr);
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

TEST_F(PaimonConnectorTest, rejectsWrongFormatAndMissingFiles) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto plan = makePaimonScanPlan(ROW({{"id", BIGINT()}}), rowType);
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    const auto input = makeRowVector(
        {"id", "value"},
        {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({2})});
    writeFile(files[0]->getPath(), input, format);
    const auto wrongFormat = format == dwio::common::FileFormat::DWRF
        ? dwio::common::FileFormat::PARQUET
        : dwio::common::FileFormat::DWRF;
    EXPECT_THROW(
        exec::test::AssertQueryBuilder(plan)
            .split(makePaimonSplit(
                {files[0]->getPath()},
                PaimonTableType::kAppendOnly,
                wrongFormat))
            .copyResults(pool()),
        std::exception);
  }
  const auto missing = makePaimonSplit({"/missing-paimon-file"});
  EXPECT_THROW(
      exec::test::AssertQueryBuilder(plan).split(missing).copyResults(pool()),
      std::exception);
}

TEST_F(PaimonConnectorTest, ordinaryColumnsIgnoreFeatureOptions) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const std::vector<std::unordered_map<std::string, std::string>> optionSets{
      {},
      {{"deletion-vectors.enabled", "true"}},
      {{"data-evolution.enabled", "true"}, {"row-tracking.enabled", "true"}},
      {{"data-file.thin-mode", "true"}},
      {{"row-tracking.enabled", "true"}},
      {{"row-tracking.enabled", "false"}}};
  const auto expected =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({4, 5})});
  const auto physical = makeRowVector(
      {"id", "_ROW_ID", "_SEQUENCE_NUMBER"},
      {expected->childAt(0),
       makeFlatVector<int64_t>({100, 101}),
       makeFlatVector<int64_t>({7, 7})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    for (const auto& input : {expected, physical}) {
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), input, format);
      const auto split = makePaimonSplit(
          {files[0]->getPath()}, PaimonTableType::kAppendOnly, format);
      for (const auto& options : optionSets) {
        SCOPED_TRACE(
            options.empty()
                ? "no options"
                : options.begin()->first + "=" + options.begin()->second);
        const auto table =
            makePaimonTableHandle(rowType, {}, nullptr, {}, {}, options);
        const auto plan = exec::test::PlanBuilder()
                              .startTableScan()
                              .connectorId(kPaimonConnectorId)
                              .outputType(rowType)
                              .tableHandle(table)
                              .assignments(makePaimonColumnHandles(rowType))
                              .endTableScan()
                              .planNode();
        const auto copy = ISerializable::deserialize<core::PlanNode>(
            plan->serialize(), pool());
        for (const auto& scan : {plan, copy}) {
          exec::test::AssertQueryBuilder(scan)
              .connectorSessionProperty(
                  kPaimonConnectorId,
                  FileConfig::kIgnoreMissingFilesSession,
                  "true")
              .split(split)
              .assertResults(expected);
        }
      }
    }
  }
}

TEST_F(PaimonConnectorTest, taskCancellationAndLimit) {
  const auto input = makeRowVector(
      {"id"}, {makeFlatVector<int64_t>(10000, [](auto row) { return row; })});
  const auto plan = makePaimonScanPlan(asRowType(input->type()));
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(1);
    writeFile(files[0]->getPath(), input, format);
    const auto split = makePaimonSplit(
        {files[0]->getPath()}, PaimonTableType::kAppendOnly, format);
    exec::CursorParameters params;
    params.planNode = plan;
    params.queryConfigs[core::QueryConfig::kMaxOutputBufferSize] = "1";
    auto cursor = exec::TaskCursor::create(params);
    cursor->task()->addSplit(plan->id(), exec::Split(split));
    cursor->task()->noMoreSplits(plan->id());
    ASSERT_TRUE(cursor->moveNext());
    cursor->task()->requestCancel().wait();
    EXPECT_THROW(cursor->moveNext(), folly::OperationCancelled);
    cursor.reset();
    exec::test::waitForAllTasksToBeDeleted();
    const auto limitPlan =
        exec::test::PlanBuilder(
            plan, std::make_shared<core::PlanNodeIdGenerator>(1))
            .limit(0, 1, false)
            .planNode();
    exec::test::AssertQueryBuilder(limitPlan).split(split).assertTypeAndNumRows(
        input->type(), 1);
    exec::test::waitForAllTasksToBeDeleted();
  }
}

TEST_F(PaimonConnectorTest, takeoverPreparedSplit) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto columns = makePaimonColumnHandles(rowType);
  const auto table = makePaimonTableHandle(rowType);
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  auto makeSource = [&](const auto& handle) {
    return connector->createDataSource(rowType, handle, columns, context.get());
  };
  auto source = makeSource(table);
  const auto copy =
      ISerializable::deserialize<PaimonTableHandle>(table->serialize());
  auto preloaded = makeSource(copy);
  const auto paths = makeFilePaths(1);
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({5, 6})});
  writeToFile(paths[0]->getPath(), input);
  preloaded->addSplit(makePaimonSplit({paths[0]->getPath()}));
  source->setFromDataSource(std::move(preloaded));
  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(*result);
  test::assertEqualVectors(input, *result);
}

TEST_F(PaimonConnectorTest, requiresPaimonTableHandle) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto columns = makePaimonColumnHandles(rowType);
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(rowType, nullptr, columns, context.get()),
      "Missing Paimon table handle");
  const auto tableHandle = std::make_shared<HiveTableHandle>(
      kPaimonConnectorId,
      "paimon_table",
      common::SubfieldFilters{},
      nullptr,
      rowType);
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(rowType, tableHandle, columns, context.get()),
      "Paimon connector requires PaimonTableHandle");
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(tableHandle)
                        .assignments(columns)
                        .endTableScan()
                        .planNode();
  for (auto tableType :
       {PaimonTableType::kAppendOnly, PaimonTableType::kPrimaryKey}) {
    SCOPED_TRACE(fmt::format("table type {}", tableType));
    const auto split = makePaimonSplit(
        {"paimon-table-test:/not-opened"},
        tableType,
        dwio::common::FileFormat::DWRF);
    VELOX_ASSERT_USER_THROW(
        exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
        "Paimon connector requires PaimonTableHandle");
    exec::test::waitForAllTasksToBeDeleted();
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

// The current append reader requires rawConvertible splits.
TEST_F(PaimonConnectorTest, rejectsNonRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kAppendOnly,
      dwio::common::FileFormat::DWRF,
      /*partitionKeys=*/{},
      /*rawConvertible=*/false);

  auto plan = makePaimonScanPlan(rowType);

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "Paimon same-schema append reader requires rawConvertible=true");
}

// A PK split cannot bypass the append table definition, even when Raw.
TEST_F(PaimonConnectorTest, rejectsPrimaryKeySplits) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  const auto plan = makePaimonScanPlan(rowType);
  const auto fileSystem = validationFileSystem();
  fileSystem->openedFiles = 0;
  for (auto rawConvertible : {true, false}) {
    SCOPED_TRACE(rawConvertible);
    const auto split = makePaimonSplit(
        {"paimon-table-test:/not-opened"},
        PaimonTableType::kPrimaryKey,
        dwio::common::FileFormat::DWRF,
        /*partitionKeys=*/{},
        rawConvertible);
    VELOX_ASSERT_USER_THROW(
        exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
        "Paimon split table type does not match table handle");
    exec::test::waitForAllTasksToBeDeleted();
  }
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

// E2E test: multiple splits each with multiple files.
TEST_F(PaimonConnectorTest, appendOnlyMultipleSplits) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});

  // Split 1 with 2 files.
  auto vectors1a = makeVectors(rowType, 1, 50);
  auto vectors1b = makeVectors(rowType, 1, 30);
  // Split 2 with 1 file.
  auto vectors2a = makeVectors(rowType, 1, 40);

  auto filePaths = makeFilePaths(3);
  writeToFile(filePaths[0]->getPath(), vectors1a);
  writeToFile(filePaths[1]->getPath(), vectors1b);
  writeToFile(filePaths[2]->getPath(), vectors2a);

  auto split1 =
      makePaimonSplit({filePaths[0]->getPath(), filePaths[1]->getPath()});
  auto split2 = makePaimonSplit({filePaths[2]->getPath()});

  auto plan = makePaimonScanPlan(rowType);

  std::vector<RowVectorPtr> expected;
  expected.insert(expected.end(), vectors1a.begin(), vectors1a.end());
  expected.insert(expected.end(), vectors1b.begin(), vectors1b.end());
  expected.insert(expected.end(), vectors2a.begin(), vectors2a.end());

  exec::test::AssertQueryBuilder(plan)
      .splits({split1, split2})
      .assertResults(expected);
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
