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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

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
  }

  void TearDown() override {
    ConnectorRegistry::global().erase(kPaimonConnectorId);
    HiveConnectorTestBase::TearDown();
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

} // namespace
} // namespace facebook::velox::connector::hive::paimon
