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
#include "velox/connectors/hive/paimon/PaimonDataSink.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/paimon/PaimonConnector.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/core/TableWriteTraits.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

#include <folly/json.h>

namespace facebook::velox::connector::hive::paimon {
namespace {

static const std::string kPaimonConnectorId = "test-paimon-write";

class PaimonDataSinkTest : public exec::test::HiveConnectorTestBase {
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

  /// Creates an InsertTableHandle for writing through the Paimon connector.
  std::shared_ptr<core::InsertTableHandle> makePaimonInsertTableHandle(
      const RowTypePtr& rowType,
      const std::string& targetDirectory,
      const std::vector<std::string>& partitionedBy = {},
      dwio::common::FileFormat format = dwio::common::FileFormat::DWRF) {
    auto hiveInsertHandle = makeHiveInsertTableHandle(
        rowType->names(),
        rowType->children(),
        partitionedBy,
        makeLocationHandle(targetDirectory));
    return std::make_shared<core::InsertTableHandle>(
        kPaimonConnectorId, hiveInsertHandle);
  }

  /// Creates a table handle for reading with the Paimon connector.
  static std::shared_ptr<HiveTableHandle> makePaimonTableHandle(
      const RowTypePtr& dataColumns = nullptr) {
    return std::make_shared<HiveTableHandle>(
        kPaimonConnectorId,
        "paimon_table",
        common::SubfieldFilters{},
        nullptr,
        dataColumns);
  }

  /// Creates column assignments for reading with the Paimon connector.
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

  /// Builds a write plan: values() → tableWrite() using Paimon connector.
  core::PlanNodePtr makePaimonWritePlan(
      const std::vector<RowVectorPtr>& data,
      const std::string& targetDirectory,
      const std::vector<std::string>& partitionedBy = {},
      dwio::common::FileFormat format = dwio::common::FileFormat::DWRF) {
    auto rowType = asRowType(data[0]->type());
    auto insertHandle =
        makePaimonInsertTableHandle(rowType, targetDirectory, partitionedBy);
    return exec::test::PlanBuilder()
        .values(data)
        .tableWrite(
            targetDirectory,
            /*partitionBy=*/{},
            /*bucketCount=*/0,
            /*bucketedBy=*/{},
            /*sortBy=*/{},
            format,
            /*aggregates=*/{},
            kPaimonConnectorId,
            /*serdeParameters=*/{},
            /*options=*/nullptr,
            /*outputFileName=*/"",
            common::CompressionKind_NONE,
            /*schema=*/nullptr,
            /*ensureFiles=*/false,
            connector::CommitStrategy::kNoCommit,
            insertHandle)
        .planNode();
  }

  /// Builds a read plan using the Paimon connector.
  core::PlanNodePtr makePaimonScanPlan(const RowTypePtr& outputType) {
    auto tableHandle = makePaimonTableHandle(outputType);
    auto assignments = makePaimonColumnHandles(outputType);
    return exec::test::PlanBuilder()
        .startTableScan()
        .connectorId(kPaimonConnectorId)
        .outputType(outputType)
        .tableHandle(tableHandle)
        .assignments(assignments)
        .endTableScan()
        .planNode();
  }

  /// Creates a PaimonConnectorSplit from a file path.
  std::shared_ptr<PaimonConnectorSplit> makePaimonSplit(
      const std::string& filePath) {
    PaimonConnectorSplitBuilder builder(
        kPaimonConnectorId,
        /*snapshotId=*/1,
        PaimonTableType::kAppendOnly,
        dwio::common::FileFormat::DWRF);
    builder.addFile(filePath, /*fileSize=*/0);
    return builder.build();
  }

  /// Extracts the commit message JSON strings from the write result.
  std::vector<std::string> extractCommitMessages(
      const RowVectorPtr& result) const {
    std::vector<std::string> messages;
    auto fragments = result->childAt(core::TableWriteTraits::kFragmentChannel)
                         ->as<FlatVector<StringView>>();
    for (auto i = 0; i < result->size(); ++i) {
      if (!fragments->isNullAt(i)) {
        messages.push_back(fragments->valueAt(i).str());
      }
    }
    return messages;
  }
};

// E2E test: write data to an unpartitioned Paimon table.
TEST_F(PaimonDataSinkTest, unpartitionedWrite) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto data = makeVectors(rowType, 1, 100);

  auto outputDirectory = exec::test::TempDirectoryPath::create();

  auto plan = makePaimonWritePlan(data, outputDirectory->getPath());
  auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool());

  // Verify the write produced output rows.
  ASSERT_GT(result->size(), 0);

  // Extract and validate commit messages.
  auto messages = extractCommitMessages(result);
  ASSERT_EQ(messages.size(), 1);

  auto commitJson = folly::parseJson(messages[0]);
  EXPECT_EQ(commitJson[CommitMessage::kBucketNumber].asInt(), -1);
  EXPECT_GT(commitJson[CommitMessage::kTotalRowCount].asInt(), 0);
  EXPECT_GT(commitJson[CommitMessage::kOnDiskDataSizeInBytes].asInt(), 0);

  // Verify file write infos.
  auto fileInfos = commitJson[CommitMessage::kFileWriteInfos];
  ASSERT_EQ(fileInfos.size(), 1);
  EXPECT_GT(fileInfos[0][CommitMessage::kFileSize].asInt(), 0);

  // Verify the target path includes bucket-0 directory.
  auto targetPath = commitJson[CommitMessage::kTargetPath].asString();
  EXPECT_TRUE(targetPath.find("bucket-0") != std::string::npos)
      << "Target path should contain bucket-0: " << targetPath;
}

// E2E test: write then read back data through the Paimon connector.
TEST_F(PaimonDataSinkTest, writeAndReadBack) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto data = makeVectors(rowType, 1, 100);

  auto outputDirectory = exec::test::TempDirectoryPath::create();

  // Write data.
  auto writePlan = makePaimonWritePlan(data, outputDirectory->getPath());
  auto writeResult =
      exec::test::AssertQueryBuilder(writePlan).copyResults(pool());

  // Parse the commit message to find the written file.
  auto messages = extractCommitMessages(writeResult);
  ASSERT_EQ(messages.size(), 1);

  auto commitJson = folly::parseJson(messages[0]);
  auto writePath = commitJson[CommitMessage::kWritePath].asString();
  auto fileInfos = commitJson[CommitMessage::kFileWriteInfos];
  ASSERT_EQ(fileInfos.size(), 1);
  auto writeFileName = fileInfos[0][CommitMessage::kWriteFileName].asString();

  // Read back the written file using a Paimon scan.
  auto filePath = fmt::format("{}/{}", writePath, writeFileName);
  auto split = makePaimonSplit(filePath);
  auto readPlan = makePaimonScanPlan(rowType);

  exec::test::AssertQueryBuilder(readPlan).split(split).assertResults(data);
}

// E2E test: write multiple batches to verify accumulation.
TEST_F(PaimonDataSinkTest, multipleBatches) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto batch1 = makeVectors(rowType, 1, 50);
  auto batch2 = makeVectors(rowType, 1, 30);

  std::vector<RowVectorPtr> allData;
  allData.insert(allData.end(), batch1.begin(), batch1.end());
  allData.insert(allData.end(), batch2.begin(), batch2.end());

  auto outputDirectory = exec::test::TempDirectoryPath::create();

  auto writePlan = makePaimonWritePlan(allData, outputDirectory->getPath());
  auto writeResult =
      exec::test::AssertQueryBuilder(writePlan).copyResults(pool());

  // Parse commit message and verify total row count.
  auto messages = extractCommitMessages(writeResult);
  ASSERT_EQ(messages.size(), 1);

  auto commitJson = folly::parseJson(messages[0]);
  EXPECT_EQ(commitJson[CommitMessage::kTotalRowCount].asInt(), 80);

  // Read back and verify.
  auto writePath = commitJson[CommitMessage::kWritePath].asString();
  auto writeFileName = commitJson[CommitMessage::kFileWriteInfos][0]
                                 [CommitMessage::kWriteFileName]
                                     .asString();

  auto filePath = fmt::format("{}/{}", writePath, writeFileName);
  auto split = makePaimonSplit(filePath);
  auto readPlan = makePaimonScanPlan(rowType);

  exec::test::AssertQueryBuilder(readPlan).split(split).assertResults(allData);
}

// Verify commit message JSON structure.
TEST_F(PaimonDataSinkTest, commitMessageFormat) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeVectors(rowType, 1, 10);

  auto outputDirectory = exec::test::TempDirectoryPath::create();

  auto writePlan = makePaimonWritePlan(data, outputDirectory->getPath());
  auto writeResult =
      exec::test::AssertQueryBuilder(writePlan).copyResults(pool());

  auto messages = extractCommitMessages(writeResult);
  ASSERT_EQ(messages.size(), 1);

  auto commitJson = folly::parseJson(messages[0]);

  // Verify all expected fields are present.
  EXPECT_TRUE(commitJson.count(CommitMessage::kPartitionValues));
  EXPECT_TRUE(commitJson.count(CommitMessage::kBucketNumber));
  EXPECT_TRUE(commitJson.count(CommitMessage::kWritePath));
  EXPECT_TRUE(commitJson.count(CommitMessage::kTargetPath));
  EXPECT_TRUE(commitJson.count(CommitMessage::kFileWriteInfos));
  EXPECT_TRUE(commitJson.count(CommitMessage::kTotalRowCount));
  EXPECT_TRUE(commitJson.count(CommitMessage::kInMemoryDataSizeInBytes));
  EXPECT_TRUE(commitJson.count(CommitMessage::kOnDiskDataSizeInBytes));

  // Verify file write info fields.
  auto fileInfos = commitJson[CommitMessage::kFileWriteInfos];
  ASSERT_GE(fileInfos.size(), 1);
  EXPECT_TRUE(fileInfos[0].count(CommitMessage::kWriteFileName));
  EXPECT_TRUE(fileInfos[0].count(CommitMessage::kTargetFileName));
  EXPECT_TRUE(fileInfos[0].count(CommitMessage::kFileSize));
  EXPECT_TRUE(fileInfos[0].count(CommitMessage::kFileRowCount));

  // Verify file naming convention.
  auto targetFileName = fileInfos[0][CommitMessage::kTargetFileName].asString();
  EXPECT_TRUE(targetFileName.find("data-") == 0)
      << "Target file should start with 'data-': " << targetFileName;
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon
