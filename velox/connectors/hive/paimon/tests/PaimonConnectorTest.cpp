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

  static PaimonSchemaPtr makeTableSchema(
      const RowTypePtr& rowType,
      const std::vector<std::string>& partitionNames = {},
      const std::vector<std::string>& primaryNames = {},
      bool rowTracking = false) {
    PaimonTypeMetadata metadata{"ROW", false};
    for (column_index_t i = 0; i < rowType->size(); ++i) {
      const auto& type = rowType->childAt(i);
      VELOX_CHECK(
          type->isBoolean() || type->isBigint() || type->isInteger() ||
          type->isDouble() || type->isVarchar());
      metadata.children.push_back(
          PaimonTypeMetadata{
              type->kindName(),
              true,
              static_cast<int32_t>(i + 10),
              type->isVarchar() ? std::vector<int32_t>{INT32_MAX}
                                : std::vector<int32_t>{}});
    }
    const auto keyIds = [&](const auto& names) {
      std::vector<int32_t> ids;
      ids.reserve(names.size());
      for (const auto& name : names) {
        ids.push_back(rowType->getChildIdx(name) + 10);
      }
      return ids;
    };
    return std::make_shared<PaimonSchema>(
        17,
        3,
        rowType,
        std::move(metadata),
        keyIds(partitionNames),
        keyIds(primaryNames),
        std::unordered_map<std::string, std::string>{
            {"type", "table"},
            {"batch-scan-mode", "none"},
            {"deletion-vectors.enabled", "false"},
            {"data-evolution.enabled", "false"},
            {"data-file.thin-mode", "false"},
            {"row-tracking.enabled", rowTracking ? "true" : "false"}});
  }

  static std::shared_ptr<PaimonTableHandle> makePaimonTableHandle(
      const RowTypePtr& rowType,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr,
      const std::vector<std::string>& partitionNames = {},
      const std::vector<std::string>& primaryNames = {}) {
    return std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        makeTableSchema(rowType, partitionNames, primaryNames),
        PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch,
        std::move(subfieldFilters),
        remainingFilter);
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
      const core::TypedExprPtr& remainingFilter = nullptr,
      bool legacyHandle = false) {
    const auto rowType = dataColumns ? dataColumns : outputType;
    ConnectorTableHandlePtr tableHandle;
    if (legacyHandle) {
      tableHandle = std::make_shared<HiveTableHandle>(
          kPaimonConnectorId,
          "paimon_table",
          std::move(subfieldFilters),
          remainingFilter,
          rowType);
    } else {
      tableHandle = makePaimonTableHandle(
          rowType, std::move(subfieldFilters), remainingFilter);
    }
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
  auto tableHandle = std::make_shared<PaimonTableHandle>(
      kPaimonConnectorId,
      "paimon_table",
      makeTableSchema(rowType),
      PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch,
      std::move(filters),
      remainingFilter,
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
    source->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(4, 5, false));
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = source->next(100, future);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(*result);
    test::assertEqualVectors(
        makeRowVector({"selected_id"}, {makeFlatVector<int64_t>({4})}),
        *result);
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
      const auto table = std::make_shared<PaimonTableHandle>(
          kPaimonConnectorId,
          "paimon_table",
          makeTableSchema(rowType, {"p"}),
          PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch,
          common::SubfieldFilters{},
          parseExpr("p is null or length(p) = 0", rowType),
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
    const auto table = std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        makeTableSchema(rowType, {"p"}),
        PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch,
        std::move(filters),
        nullptr,
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
    VELOX_ASSERT_USER_THROW(
        source->addDynamicFilter(
            0,
            std::make_shared<common::BytesRange>(
                "a", false, false, "z", false, false, false)),
        "is not supported for target type BIGINT");
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
      "Paimon split table type does not match table schema");
  EXPECT_EQ(fileSystem->openedFiles, 0);
  VELOX_ASSERT_THROW(
      source->addSplit(makeSplit(
          PaimonTableType::kAppendOnly,
          std::unordered_map<std::string, std::optional<std::string>>{})),
      "Paimon split partition names do not match table schema");
  EXPECT_EQ(fileSystem->openedFiles, 0);
  VELOX_ASSERT_THROW(
      source->addSplit(makeSplit(
          PaimonTableType::kAppendOnly,
          std::unordered_map<std::string, std::optional<std::string>>{
              {"q", ""}})),
      "Missing Paimon split partition 'p' declared by table schema");
  EXPECT_EQ(fileSystem->openedFiles, 0);
  const auto foreignSplit = PaimonConnectorSplitBuilder(
                                "other-connector",
                                1,
                                PaimonTableType::kAppendOnly,
                                dwio::common::FileFormat::DWRF)
                                .addFile(path, 0)
                                .partitionKey("p", std::nullopt)
                                .build();
  VELOX_ASSERT_THROW(
      source->addSplit(foreignSplit),
      "Paimon split connector does not match table handle");
  EXPECT_EQ(fileSystem->openedFiles, 0);

  for (const auto& change : {"dv", "changelog", "level"}) {
    SCOPED_TRACE(change);
    auto wire = makeSplit(PaimonTableType::kAppendOnly, good)->serialize();
    if (std::string_view(change) == "dv") {
      wire["dataFiles"][0]["deletionFile"] =
          PaimonDeletionFile{"/dv", 0, 1, 1}.serialize();
    } else if (std::string_view(change) == "changelog") {
      wire["dataFiles"][0]["fileType"] = "CHANGELOG";
    } else {
      wire["dataFiles"][0]["level"] = 1;
    }
    VELOX_ASSERT_USER_THROW(
        source->addSplit(PaimonConnectorSplit::create(wire)),
        "Paimon same-schema append profile");
    EXPECT_EQ(fileSystem->openedFiles, 0);
  }

  VELOX_ASSERT_USER_THROW(
      source->addSplit(makePaimonSplit(
          {path},
          PaimonTableType::kAppendOnly,
          dwio::common::FileFormat::DWRF,
          good,
          false)),
      "Paimon same-schema append profile requires rawConvertible=true");
  EXPECT_EQ(fileSystem->openedFiles, 0);
  orc::unregisterOrcReaderFactory();
  VELOX_ASSERT_USER_THROW(
      source->addSplit(makePaimonSplit(
          {path},
          PaimonTableType::kAppendOnly,
          dwio::common::FileFormat::ORC,
          good)),
      fmt::format(
          "No Paimon reader factory for file format {}",
          dwio::common::FileFormatName::toName(dwio::common::FileFormat::ORC)));
  orc::registerOrcReaderFactory();
  EXPECT_EQ(fileSystem->openedFiles, 0);

  source->addSplit(makeSplit(PaimonTableType::kAppendOnly, good));
  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(*result);
  test::assertEqualVectors(input, *result);
  EXPECT_GT(fileSystem->openedFiles, 0);
  auto end = source->next(100, future);
  ASSERT_TRUE(end.has_value());
  EXPECT_EQ(*end, nullptr);
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
  const auto makeSource = [&](const TypePtr& outputType,
                              const std::string& name,
                              FileColumnHandle::ColumnType role,
                              const TypePtr& schemaType) {
    return connector->createDataSource(
        ROW({{"alias", outputType}}),
        table,
        {{"alias",
          std::make_shared<HiveColumnHandle>(
              name, role, outputType, schemaType)}},
        context.get());
  };
  VELOX_ASSERT_USER_THROW(
      makeSource(
          INTEGER(), "id", FileColumnHandle::ColumnType::kRegular, INTEGER()),
      "Paimon column schema type does not match target schema");
  VELOX_ASSERT_USER_THROW(
      makeSource(
          BIGINT(),
          "missing",
          FileColumnHandle::ColumnType::kRegular,
          BIGINT()),
      "Unknown Paimon column 'missing' in target schema");
  VELOX_ASSERT_USER_THROW(
      makeSource(
          VARCHAR(), "p", FileColumnHandle::ColumnType::kRegular, VARCHAR()),
      "Paimon column role does not match target schema");
  VELOX_ASSERT_USER_THROW(
      makeSource(
          BIGINT(),
          "id",
          FileColumnHandle::ColumnType::kPartitionKey,
          BIGINT()),
      "Paimon column role does not match target schema");
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"id", INTEGER()}}),
          table,
          makePaimonColumnHandles(ROW({{"id", BIGINT()}})),
          context.get()),
      "Paimon output channel type does not match column handle");
  const auto idColumns = makePaimonColumnHandles(ROW({{"id", BIGINT()}}));
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"missing", BIGINT()}}), table, idColumns, context.get()),
      "Missing Paimon output assignment 'missing'");
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}, {"id", BIGINT()}}),
          table,
          idColumns,
          context.get()),
      "Duplicate Paimon output channel name 'id'");
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}),
          table,
          {{"id", idColumns.at("id")}, {"alias", idColumns.at("id")}},
          context.get()),
      "Paimon regular column 'id' cannot map to multiple assignments; use a projection");
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
  for (const auto& path : {"missing", "id.value", "p"}) {
    SCOPED_TRACE(path);
    const auto message = std::string_view(path) == "missing"
        ? "Unknown Paimon field: missing"
        : (std::string_view(path) == "id.value"
               ? "ROW required in Paimon path id.value"
               : "Paimon required subfield does not match column name");
    VELOX_ASSERT_USER_THROW(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}),
            table,
            {{"id", requiredColumn(common::Subfield(path))}},
            context.get()),
        message);
  }
  EXPECT_NE(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}),
          table,
          {{"id", requiredColumn(common::Subfield("id"))}},
          context.get()),
      nullptr);
  for (const auto& option :
       {"primary-key", "data-evolution.enabled", "custom"}) {
    SCOPED_TRACE(option);
    auto schemaWire = table->schema()->serialize();
    if (std::string_view(option) == "primary-key") {
      schemaWire["primaryKeys"] = folly::dynamic::array(10);
    } else {
      schemaWire["options"][option] = "true";
    }
    const auto unsupported = std::make_shared<PaimonTableHandle>(
        kPaimonConnectorId,
        table->name(),
        PaimonSchema::create(schemaWire),
        PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch);
    VELOX_ASSERT_USER_THROW(
        connector->createDataSource(
            ROW({{"id", BIGINT()}}), unsupported, idColumns, context.get()),
        "Paimon same-schema append profile");
  }
  VELOX_ASSERT_USER_THROW(
      makeSource(
          BIGINT(),
          "_ROW_ID",
          FileColumnHandle::ColumnType::kRegular,
          BIGINT()),
      "Unknown Paimon column '_ROW_ID' in target schema");
  const auto foreignTable = std::make_shared<PaimonTableHandle>(
      "other-connector",
      table->name(),
      table->schema(),
      PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch);
  VELOX_ASSERT_USER_THROW(
      connector->createDataSource(
          ROW({{"id", BIGINT()}}), foreignTable, idColumns, context.get()),
      "Paimon table handle connector does not match connector instance");
  EXPECT_NE(
      makeSource(
          BIGINT(), "id", FileColumnHandle::ColumnType::kRegular, BIGINT()),
      nullptr);
  EXPECT_EQ(fileSystem->openedFiles, 0);
}

TEST_F(PaimonConnectorTest, rejectsFileSchemaContradictions) {
  const auto rowType = ROW({{"id", BIGINT()}, {"value", BIGINT()}});
  const auto plan = makePaimonScanPlan(ROW({{"id", BIGINT()}}), rowType);
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const std::vector<RowVectorPtr> invalid{
        makeRowVector({"id"}, {makeFlatVector<int64_t>({1})}),
        makeRowVector(
            {"renamed", "value"},
            {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({2})}),
        makeRowVector(
            {"id", "value"},
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<std::string>({"bad"})}),
        makeRowVector(
            {"id", "value", "extra"},
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<int64_t>({2}),
             makeFlatVector<int64_t>({3})})};
    for (const auto& input : invalid) {
      SCOPED_TRACE(input->type()->toString());
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), input, format);
      VELOX_ASSERT_USER_THROW(
          exec::test::AssertQueryBuilder(plan)
              .split(makePaimonSplit(
                  {files[0]->getPath()}, PaimonTableType::kAppendOnly, format))
              .copyResults(pool()),
          "Paimon file");
    }
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
    const auto zeroByteFiles = makeFilePaths(1);
    {
      LocalWriteFile emptyFile(zeroByteFiles[0]->getPath(), false, false);
      emptyFile.close();
    }
    EXPECT_THROW(
        exec::test::AssertQueryBuilder(plan)
            .split(makePaimonSplit(
                {zeroByteFiles[0]->getPath()},
                PaimonTableType::kAppendOnly,
                format))
            .copyResults(pool()),
        std::exception);
  }
  const auto missing = makePaimonSplit({"/missing-paimon-file"});
  EXPECT_THROW(
      exec::test::AssertQueryBuilder(plan).split(missing).copyResults(pool()),
      std::exception);
  VELOX_ASSERT_USER_THROW(
      exec::test::AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kPaimonConnectorId,
              FileConfig::kIgnoreMissingFilesSession,
              "true")
          .split(missing)
          .copyResults(pool()),
      "Paimon same-schema append profile does not support ignore_missing_files");
}

TEST_F(PaimonConnectorTest, rowTrackingOrdinaryColumns) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto table = std::make_shared<PaimonTableHandle>(
      kPaimonConnectorId,
      "paimon_table",
      makeTableSchema(rowType, {}, {}, /*rowTracking=*/true),
      PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch);
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(table)
                        .assignments(makePaimonColumnHandles(rowType))
                        .endTableScan()
                        .planNode();
  const auto expected =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({4, 5})});
  const auto physical = makeRowVector(
      {"id", "_ROW_ID", "_SEQUENCE_NUMBER"},
      {expected->childAt(0),
       makeFlatVector<int64_t>({100, 101}),
       makeFlatVector<int64_t>({7, 7})});
  for (auto format : fileFormats()) {
    for (const auto& input : {expected, physical}) {
      const auto files = makeFilePaths(1);
      writeFile(files[0]->getPath(), input, format);
      exec::test::AssertQueryBuilder(plan)
          .split(makePaimonSplit(
              {files[0]->getPath()}, PaimonTableType::kAppendOnly, format))
          .assertResults(expected);
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

TEST_F(PaimonConnectorTest, takeoverPreservesTableDefinition) {
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
  auto changedSchema = table->schema()->serialize();
  changedSchema["metadata"]["children"][0]["fieldId"] = 99;
  const auto differentTable = std::make_shared<PaimonTableHandle>(
      kPaimonConnectorId,
      table->name(),
      PaimonSchema::create(changedSchema),
      PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch);
  VELOX_ASSERT_THROW(
      source->setFromDataSource(makeSource(differentTable)),
      "Paimon takeover must preserve the fixed table definition");
  const auto legacyTable = std::make_shared<HiveTableHandle>(
      kPaimonConnectorId,
      "paimon_table",
      common::SubfieldFilters{},
      nullptr,
      rowType);
  VELOX_ASSERT_THROW(
      source->setFromDataSource(makeSource(legacyTable)),
      "Paimon takeover must preserve the fixed table definition");
  auto legacySource = makeSource(legacyTable);
  VELOX_ASSERT_THROW(
      legacySource->setFromDataSource(makeSource(table)),
      "Paimon takeover must preserve the fixed table definition");
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"), std::make_unique<common::IsNotNull>());
  const auto filteredTable = makePaimonTableHandle(rowType, std::move(filters));
  VELOX_ASSERT_USER_THROW(
      source->setFromDataSource(makeSource(filteredTable)),
      "Paimon takeover must preserve scan conditions and query columns");
  const auto differentCondition =
      makePaimonTableHandle(rowType, {}, parseExpr("id % 2 = 0", rowType));
  VELOX_ASSERT_USER_THROW(
      source->setFromDataSource(makeSource(differentCondition)),
      "Paimon takeover must preserve scan conditions and query columns");
  VELOX_ASSERT_USER_THROW(
      source->setFromDataSource(connector->createDataSource(
          ROW({{"alias", BIGINT()}}),
          table,
          {{"alias", columns.at("id")}},
          context.get())),
      "Paimon takeover must preserve scan conditions and query columns");
  VELOX_ASSERT_USER_THROW(
      source->setFromDataSource(makeSource(table)),
      "Paimon takeover requires a prepared split");
  const auto copy =
      ISerializable::deserialize<PaimonTableHandle>(table->serialize());
  auto preloaded = makeSource(copy);
  const auto paths = makeFilePaths(1);
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({5, 6})});
  writeToFile(paths[0]->getPath(), input);
  auto alreadyRead = makeSource(table);
  alreadyRead->addSplit(makePaimonSplit({paths[0]->getPath()}));
  ContinueFuture future = ContinueFuture::makeEmpty();
  ASSERT_TRUE(alreadyRead->next(1, future).has_value());
  VELOX_ASSERT_USER_THROW(
      source->setFromDataSource(std::move(alreadyRead)),
      "Paimon takeover does not support an already-read donor");
  preloaded->addSplit(makePaimonSplit({paths[0]->getPath()}));
  source->setFromDataSource(std::move(preloaded));
  auto result = source->next(100, future);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(*result);
  test::assertEqualVectors(input, *result);
}

TEST_F(PaimonConnectorTest, takeoverPartitionHandlesAndDynamicFilters) {
  auto queryCtx = core::QueryCtx::create();
  const auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto context = queryContextBuilder(queryCtx.get())
                     .sessionProperties(config.get())
                     .build();
  const auto rowType = ROW({{"id", BIGINT()}, {"p", VARCHAR()}});
  common::SubfieldFilters filters;
  filters.emplace(common::Subfield("p"), std::make_unique<common::IsNotNull>());
  const auto table =
      makePaimonTableHandle(rowType, std::move(filters), nullptr, {"p"});
  const ColumnHandleMap columns{
      {"id",
       std::make_shared<HiveColumnHandle>(
           "id", FileColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT())},
      {"p",
       std::make_shared<HiveColumnHandle>(
           "p",
           FileColumnHandle::ColumnType::kPartitionKey,
           VARCHAR(),
           VARCHAR())}};
  const auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  const auto makeSource = [&] {
    return connector->createDataSource(rowType, table, columns, context.get());
  };
  const auto input = makeRowVector({"id"}, {makeFlatVector<int64_t>({7, 8})});
  for (auto format : fileFormats()) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    for (const auto& file : files) {
      writeFile(file->getPath(), input, format);
    }
    auto source = makeSource();
    ContinueFuture future = ContinueFuture::makeEmpty();
    source->addSplit(makePaimonSplit(
        {files[0]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", "old"}}));
    ASSERT_TRUE(source->next(100, future).has_value());
    auto end = source->next(100, future);
    ASSERT_TRUE(end.has_value());
    ASSERT_EQ(*end, nullptr);
    source->addDynamicFilter(
        1,
        std::make_shared<common::BytesValues>(
            std::vector<std::string>{"keep"}, false));
    for (const auto& partition : {"drop", "keep"}) {
      SCOPED_TRACE(partition);
      auto donor = makeSource();
      donor->addSplit(makePaimonSplit(
          {files[0]->getPath(), files[1]->getPath()},
          PaimonTableType::kAppendOnly,
          format,
          {{"p", partition}}));
      source->setFromDataSource(std::move(donor));
      ASSERT_EQ(donor, nullptr);
      if (std::string_view(partition) == "keep") {
        const auto expected = makeRowVector(
            {"id", "p"},
            {input->childAt(0), makeFlatVector<std::string>({"keep", "keep"})});
        for (const auto& file : files) {
          auto result = source->next(100, future);
          ASSERT_TRUE(result.has_value());
          ASSERT_TRUE(*result);
          test::assertEqualVectors(expected, *result);
        }
      }
      end = source->next(100, future);
      ASSERT_TRUE(end.has_value());
      EXPECT_EQ(*end, nullptr);
    }
    EXPECT_EQ(source->getCompletedRows(), 6);
  }
}

TEST_F(PaimonConnectorTest, legacyHiveHandles) {
  const auto rowType = ROW({{"id", BIGINT()}});
  const auto input =
      makeRowVector({"id"}, {makeFlatVector<int64_t>({1, 2, 3})});
  const auto files = makeFilePaths(1);
  writeToFile(files[0]->getPath(), input);
  const auto tableHandle = std::make_shared<HiveTableHandle>(
      kPaimonConnectorId,
      "paimon_table",
      common::SubfieldFilters{},
      nullptr,
      rowType);
  const auto plan = exec::test::PlanBuilder()
                        .startTableScan()
                        .connectorId(kPaimonConnectorId)
                        .outputType(rowType)
                        .tableHandle(tableHandle)
                        .assignments(makePaimonColumnHandles(rowType))
                        .endTableScan()
                        .planNode();
  for (auto tableType :
       {PaimonTableType::kAppendOnly, PaimonTableType::kPrimaryKey}) {
    const auto split = makePaimonSplit(
        {files[0]->getPath()}, tableType, dwio::common::FileFormat::DWRF);
    exec::test::AssertQueryBuilder(plan).split(split).assertResults(input);
  }
}

// Verify that non-rawConvertible splits trigger NYI (merge-on-read).
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
      "Paimon same-schema append profile requires rawConvertible=true");
}

// E2E test: primary-key table with rawConvertible=true reads successfully.
// When fully compacted (rawConvertible), primary-key tables can be read
// the same way as append-only — each file is independent, no merge needed.
TEST_F(PaimonConnectorTest, primaryKeyRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kPrimaryKey,
      dwio::common::FileFormat::DWRF);

  auto plan = makePaimonScanPlan(
      rowType,
      /*dataColumns=*/nullptr,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      /*legacyHandle=*/true);

  exec::test::AssertQueryBuilder(plan).split(split).assertResults(vectors);
}

// Verify that primary-key table with rawConvertible=false triggers NYI.
TEST_F(PaimonConnectorTest, rejectsPrimaryKeyNonRawConvertible) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto vectors = makeVectors(rowType, 1, 10);

  auto filePaths = makeFilePaths(1);
  writeToFile(filePaths[0]->getPath(), vectors);

  auto split = makePaimonSplit(
      {filePaths[0]->getPath()},
      PaimonTableType::kPrimaryKey,
      dwio::common::FileFormat::DWRF,
      /*partitionKeys=*/{},
      /*rawConvertible=*/false);

  auto plan = makePaimonScanPlan(
      rowType,
      /*dataColumns=*/nullptr,
      /*subfieldFilters=*/{},
      /*remainingFilter=*/nullptr,
      /*legacyHandle=*/true);

  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).split(split).copyResults(pool()),
      "Paimon merge-on-read is not yet implemented");
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
