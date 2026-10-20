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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/process/ThreadDebugInfo.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"

#ifdef VELOX_ENABLE_NIMBLE
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#endif

namespace facebook::velox::connector::hive::paimon {
namespace {

static const std::string kPaimonConnectorId = "test-paimon";

class PaimonConnectorTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    auto config = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    auto connector =
        PaimonConnectorFactory().newConnector(kPaimonConnectorId, config);
    ConnectorRegistry::global().insert(connector->connectorId(), connector);
#ifdef VELOX_ENABLE_NIMBLE
    nimble::registerSelectiveNimbleReaderFactory();
#endif
  }

  void TearDown() override {
#ifdef VELOX_ENABLE_NIMBLE
    nimble::unregisterSelectiveNimbleReaderFactory();
#endif
    ConnectorRegistry::global().erase(kPaimonConnectorId);
    HiveConnectorTestBase::TearDown();
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
    std::vector<dwio::common::FileFormat> formats{
        dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
    formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
    for (auto format : formats) {
      SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
      const int firstFile = existingStats ? 1 : 0;
      auto files = makeFilePaths(firstFile + fileCount);
      std::vector<std::string> paths;
      std::vector<RowVectorPtr> inputs;
      for (int i = 0; i < files.size(); ++i) {
        paths.push_back(files[i]->getPath());
        inputs.push_back(
            makeRowVector({makeFlatVector<int64_t>({10 * i + 1, 10 * i + 2})}));
        if (format == dwio::common::FileFormat::DWRF) {
          writeToFile(paths.back(), inputs.back());
        }
#ifdef VELOX_ENABLE_NIMBLE
        else {
          nimble::Writer writer(
              type,
              std::make_unique<LocalWriteFile>(paths.back(), false, false),
              *pool(),
              nimble::WriterOptions{});
          writer.write(inputs.back());
          writer.close();
        }
#endif
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

  /// Creates a table handle for the Paimon connector.
  static std::shared_ptr<HiveTableHandle> makePaimonTableHandle(
      const RowTypePtr& dataColumns = nullptr,
      common::SubfieldFilters subfieldFilters = {},
      const core::TypedExprPtr& remainingFilter = nullptr) {
    return std::make_shared<HiveTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        std::move(subfieldFilters),
        remainingFilter,
        dataColumns);
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
    auto tableHandle = makePaimonTableHandle(
        dataColumns ? dataColumns : outputType,
        std::move(subfieldFilters),
        remainingFilter);
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
  auto tableHandle = makePaimonTableHandle(outputType);

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
      "Paimon merge-on-read is not yet implemented");
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

  auto plan = makePaimonScanPlan(rowType);

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

  auto plan = makePaimonScanPlan(rowType);

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

TEST_F(PaimonConnectorTest, scanRetainsStateAcrossFiles) {
  auto queryCtx = core::QueryCtx::create();
  auto config = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  const auto makeContext = [&] {
    return ConnectorQueryCtx::Builder()
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
  };
  auto context = makeContext();
  auto connector = ConnectorRegistry::tryGet(kPaimonConnectorId);
  const auto outputType = ROW({{"c0", BIGINT()}, {"p", BIGINT()}});
  const auto makeSource = [&](ConnectorQueryCtx* readerContext) {
    return connector->createDataSource(
        outputType,
        makePaimonTableHandle(
            outputType, {}, parseExpr("c0 % 2 = 0", outputType)),
        {{"c0", regularColumn("c0", BIGINT())},
         {"p", partitionKey("p", BIGINT())}},
        readerContext);
  };
  const std::vector<RowVectorPtr> inputs{
      makeRowVector({makeFlatVector<int64_t>({1, 2})}),
      makeRowVector({makeFlatVector<int64_t>({3, 4})})};
  std::vector<dwio::common::FileFormat> formats{dwio::common::FileFormat::DWRF};
#ifdef VELOX_ENABLE_NIMBLE
  formats.push_back(dwio::common::FileFormat::NIMBLE);
#endif
  for (const auto format : formats) {
    SCOPED_TRACE(dwio::common::FileFormatName::toName(format));
    const auto files = makeFilePaths(2);
    for (int i = 0; i < files.size(); ++i) {
      if (format == dwio::common::FileFormat::DWRF) {
        writeToFile(files[i]->getPath(), inputs[i]);
      }
#ifdef VELOX_ENABLE_NIMBLE
      else {
        nimble::Writer writer(
            inputs[i]->type(),
            std::make_unique<LocalWriteFile>(files[i]->getPath(), false, false),
            *pool(),
            nimble::WriterOptions{});
        writer.write(inputs[i]);
        writer.close();
      }
#endif
    }
    auto source = makeSource(context.get());
    source->addDynamicFilter(
        0, std::make_shared<common::BigintRange>(2, 4, false));
    source->addSplit(makePaimonSplit(
        {files[0]->getPath(), files[1]->getPath()},
        PaimonTableType::kAppendOnly,
        format,
        {{"p", "7"}}));

    std::vector<RowVectorPtr> batches;
    ContinueFuture future = ContinueFuture::makeEmpty();
    while (true) {
      auto result = source->next(2, future);
      ASSERT_TRUE(result.has_value());
      ASSERT_FALSE(future.valid());
      if (!*result) {
        break;
      }
      (*result)->loadedVector();
      batches.push_back(std::move(*result));
    }
    EXPECT_EQ(source->getCompletedRows(), 4);
    // Preserve the existing physical-file preparation count.
    EXPECT_EQ(source->getRuntimeStats().at("processedSplits").sum, 2);
    source.reset();
    ASSERT_EQ(batches.size(), 2);
    test::assertEqualVectors(
        makeRowVector(
            {makeFlatVector<int64_t>({2}), makeFlatVector<int64_t>({7})}),
        batches[0]);
    test::assertEqualVectors(
        makeRowVector(
            {makeFlatVector<int64_t>({4}), makeFlatVector<int64_t>({7})}),
        batches[1]);
  }
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
